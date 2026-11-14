#!/usr/bin/env python3
"""Activation patching at the post-marker position (reviewer question: where does the blind spot
enter?).

Setup. For a dependency task, the same input string is presented twice: with a marker every k
characters (the "marked" run, which shows the blind spot) and without (the "clean" run, which does
not). Both are teacher-forced: the prompt is followed by the correct answer, and at every
post-marker output position (answer index j > 0 with j mod k = 0) we read the next-token
distribution and record

  logodds  log p(correct digit) - log p(reset digit), where the reset digit is the input digit at
           that position, the answer a model gives when it treats the segment as a fresh start
  acc      whether the correct digit is the argmax

Qwen's tokenizer gives one token per digit and per bar, so answer positions align character by
character between the two runs once bars are skipped; marker tokens have no clean counterpart.

Patching. The clean run's residual stream (output of decoder layer l) is cached. In the marked run
a forward hook replaces the residual stream at layer l, at a chosen set of token positions, with
the clean run's residual stream at the corresponding positions, and the metrics are re-read at the
post-marker positions. Sites:

  pre_in    the input digit right before each marker
  post_in   the input digit right after each marker
  all_in    every input digit
  prev_out  the answer digit right before each marker (the output written just before the bar)

patched at one layer at a time (the change then propagates through later layers), and at every
layer at once ("all"). Recovery is (patched - marked) / (clean - marked) in log-odds.

Output: results/activation_patching/<model_tag>.csv (one row per task, trial, site, layer)
Usage:  python scripts/run_activation_patching.py --model Qwen/Qwen2.5-7B-Instruct [--dry]
"""

import argparse
import csv
import importlib
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common.runner import _local_seed  # noqa: E402

TASKS = {"running_xor": None, "running_sum_local": 4}
N, K, TRIALS = 32, 4, 20
SITES = ["pre_in", "post_in", "all_in", "prev_out"]


def build(engine, axis, k, t):
    """Same input at k = 0 and k = K: the rng is seeded as the k = 0 cell of the trial."""
    rng = random.Random(_local_seed(N, 0, axis, t))
    return engine.build_prompt(N, k, rng, axis_val=axis) if engine.AXIS_NAME else engine.build_prompt(N, k, rng)


def positions(tokenizer, prompt, inp, answer):
    """Token indices of every input character and every answer character (bars skipped), in the
    teacher-forced sequence chat_template(prompt) + answer, plus the sequence itself.
    Asserts one token per character for the digits and bars."""
    head = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    text = head + answer
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    tok_at = {}
    for i, (a, b) in enumerate(offs):
        for c in range(a, b):
            tok_at[c] = i
    i0 = head.rindex(inp) if inp in head else head.rindex(inp.replace(" ", ""))
    assert head[i0:i0 + len(inp)] == inp
    in_pos, out_pos = [], []
    for c in range(len(inp)):
        if inp[c] == "|":
            continue
        assert offs[tok_at[i0 + c]] == (i0 + c, i0 + c + 1), "input char is not its own token"
        in_pos.append(tok_at[i0 + c])
    a0 = len(head)
    for c in range(len(answer)):
        if answer[c] == "|":
            continue
        assert offs[tok_at[a0 + c]] == (a0 + c, a0 + c + 1), "answer char is not its own token"
        out_pos.append(tok_at[a0 + c])
    return enc["input_ids"], in_pos, out_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    engines = {t: importlib.import_module(f"tasks.{t}.engine") for t in TASKS}

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # self-check on one trial: alignment and the reset digit
    e, axis = engines["running_xor"], None
    inp_m, ans_m, p_m = build(e, axis, K, 0)
    inp_c, ans_c, p_c = build(e, axis, 0, 0)
    assert inp_m.replace("|", "") == inp_c and ans_m.replace("|", "") == ans_c
    ids_m, in_m, out_m = positions(tokenizer, p_m, inp_m, ans_m)
    ids_c, in_c, out_c = positions(tokenizer, p_c, inp_c, ans_c)
    assert len(in_m) == len(in_c) == N and len(out_m) == len(out_c) == N
    post = [j for j in range(1, N) if j % K == 0]
    assert all(tokenizer.decode([ids_m[out_m[j]]]) == ans_c[j] for j in post)
    print(f"Self-check passed: {N} input and {N} answer tokens aligned; {len(post)} post-marker positions per trial.")
    if args.dry:
        print(len(TASKS) * args.trials * (2 + len(SITES) * 29), "forward passes at 28 layers (7B); more at deeper models")
        return

    import torch
    from tasks.common.runner import load_model
    model = load_model(args.model)
    model.eval()
    layers = None
    for path in ("model.layers", "model.language_model.layers", "language_model.layers", "language_model.model.layers"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            layers = obj
            break
        except AttributeError:
            continue
    assert layers is not None, "decoder layers not found"
    L = len(layers)
    digit_ids = {}
    for d in "0123456789":
        ids = tokenizer.encode(d, add_special_tokens=False)
        assert len(ids) == 1, (d, ids)
        digit_ids[d] = ids[0]
    dev = model.device

    def run(ids, hooks=()):
        hs = [l.register_forward_hook(h) for l, h in hooks]
        try:
            with torch.no_grad():
                out = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=not hooks)
        finally:
            for h in hs:
                h.remove()
        return out

    def metrics(logits, out_pos, answer_plain, inp_plain, post):
        lo, ac = [], []
        for j in post:
            lg = torch.log_softmax(logits[0, out_pos[j] - 1].float(), -1)
            correct, reset = digit_ids[answer_plain[j]], digit_ids[inp_plain[j]]
            lo.append((lg[correct] - lg[reset]).item() if correct != reset else float("nan"))
            ac.append(float(lg.argmax().item() == correct))
        import math
        lo = [x for x in lo if not math.isnan(x)]
        return (sum(lo) / len(lo) if lo else float("nan")), sum(ac) / len(ac)

    tag = args.model.split("/")[-1]
    out_dir = REPO / "results" / "activation_patching"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{tag}.csv"
    fields = ["Task", "Trial", "Site", "Layer", "logodds", "acc", "n_post"]
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            done = {(r["Task"], r["Trial"], r["Site"], r["Layer"]) for r in csv.DictReader(f)}

    def write(row):
        new = not out_csv.exists()
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)

    for task, axis in TASKS.items():
        engine = engines[task]
        for t in range(args.trials):
            inp_m, ans_m, p_m = build(engine, axis, K, t)
            inp_c, ans_c, p_c = build(engine, axis, 0, t)
            ids_m, in_m, out_m = positions(tokenizer, p_m, inp_m, ans_m)
            ids_c, in_c, out_c = positions(tokenizer, p_c, inp_c, ans_c)
            post = [j for j in range(1, N) if j % K == 0]
            # the digit before each marker is reset-ambiguous only if correct != reset; metrics skip ties
            clean = run(ids_c)
            clean_h = [h[0].detach() for h in clean.hidden_states]      # L+1 tensors [seq, d]
            if (task, str(t), "clean", "-1") not in done:
                lo, ac = metrics(clean.logits, out_c, ans_c, inp_c, post)
                write(dict(Task=task, Trial=t, Site="clean", Layer=-1, logodds=lo, acc=ac, n_post=len(post)))
            if (task, str(t), "marked", "-1") not in done:
                m = run(ids_m)
                lo, ac = metrics(m.logits, out_m, ans_m.replace("|", ""), inp_c, post)
                write(dict(Task=task, Trial=t, Site="marked", Layer=-1, logodds=lo, acc=ac, n_post=len(post)))
            site_pos = {
                "pre_in": [(in_m[j - 1], in_c[j - 1]) for j in post],
                "post_in": [(in_m[j], in_c[j]) for j in post],
                "all_in": list(zip(in_m, in_c)),
                "prev_out": [(out_m[j - 1], out_c[j - 1]) for j in post],
            }
            for site in SITES:
                pairs = site_pos[site]
                pm = torch.tensor([a for a, _ in pairs], device=dev)
                pc = [b for _, b in pairs]
                for layer in list(range(L)) + ["all"]:
                    if (task, str(t), site, str(layer)) in done:
                        continue
                    def make_hook(li):
                        src = clean_h[li + 1][pc].to(dev)
                        def hook(_mod, _inp, output):
                            h = output[0] if isinstance(output, tuple) else output
                            h[0, pm, :] = src.to(h.dtype)
                            return output
                        return hook
                    hooks = ([(layers[layer], make_hook(layer))] if layer != "all"
                             else [(layers[li], make_hook(li)) for li in range(L)])
                    m = run(ids_m, hooks)
                    lo, ac = metrics(m.logits, out_m, ans_m.replace("|", ""), inp_c, post)
                    write(dict(Task=task, Trial=t, Site=site, Layer=layer, logodds=lo, acc=ac, n_post=len(post)))
            print(f"done {task} trial {t}", flush=True)
    print("Finished.")


if __name__ == "__main__":
    main()
