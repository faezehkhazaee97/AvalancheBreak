#!/usr/bin/env python3
"""Position-aligned activation patching (reviewer question: is the cross-run patching result an
artifact of mismatched token positions and query identity?).

scripts/run_activation_patching.py patches from the *no-marker* run into the marked run. That
comparison has two structural mismatches the reviewer is right to flag: inserting markers pushes
every later token along, so a pair of inputs one apart in the no-marker run is two apart in the
marked run, and the step predicting the answer after a marker is read at the output bar in the
marked run but at the previous answer digit in the no-marker run.

This script removes both. The donor is the *across-wording* marked run: the same task on the same
input, with markers in the same places, whose instruction states that the position right after a
bar uses the input right before it. So

  - the donor carries the same ground-truth answer as the recipient (asserted below), unlike the
    reset variant, which computes a different rule and would be the wrong direction to patch from;
  - markers sit at identical positions, so every relative distance among input and answer tokens
    is identical in the two runs -- only the instruction prefix differs in length, a constant
    offset that RoPE's relative attention is insensitive to (the offset is recorded per trial);
  - the step after a marker is read at the output bar in *both* runs, so the query token is the
    same and no query-identity difference is being measured.

Sites and metric are those of the original script: the input digit before each marker, the input
digit after it, every input digit, and the answer digit written before the marker; log-odds of the
correct digit against the reset digit at post-marker answer positions. Recovery is computed in the
analysis as (patched - recipient) / (donor - recipient), so 0 is the original marked run and 1 is
the across-wording run.

Output: results/patching_aligned/<model_tag>.csv (one row per task, trial, site, layer)
Usage:  python scripts/run_patching_aligned.py --model Qwen/Qwen2.5-32B-Instruct [--dry]
"""

import argparse
import csv
import importlib
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common.runner import _local_seed  # noqa: E402

TASKS = {"running_xor": None, "running_sum_local": 4}
N, K, TRIALS = 32, 4, 20
SITES = ["pre_in", "post_in", "all_in", "prev_out"]
DONOR = "_across"


def build(engine, axis, k, t):
    """Same input at every variant: the rng is seeded as the k = 0 cell of the trial, exactly as
    scripts/run_activation_patching.py does, so a trial here is the same trial there."""
    rng = random.Random(_local_seed(N, 0, axis, t))
    return engine.build_prompt(N, k, rng, axis_val=axis) if engine.AXIS_NAME else engine.build_prompt(N, k, rng)


def positions(tokenizer, prompt, inp, answer):
    """Token indices of every input character and every answer character (bars skipped) in the
    teacher-forced sequence, plus the sequence and the head length."""
    head = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
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
    n_head = len(tokenizer(head, add_special_tokens=False)["input_ids"])
    return enc["input_ids"], in_pos, out_pos, n_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    engines = {t: importlib.import_module(f"tasks.{t}.engine") for t in TASKS}
    donors = {t: importlib.import_module(f"tasks.{t}{DONOR}.engine") for t in TASKS}

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # self-check: the donor must carry the same input and the same ground-truth answer, with the
    # markers in the same places; only the instruction may differ.
    for task, axis in TASKS.items():
        inp_r, ans_r, p_r = build(engines[task], axis, K, 0)
        inp_d, ans_d, p_d = build(donors[task], axis, K, 0)
        assert inp_r == inp_d, f"{task}: donor input differs"
        assert ans_r == ans_d, f"{task}: donor answer differs -- wrong donor for this comparison"
        assert p_r != p_d, f"{task}: donor prompt is identical, so it is not a different wording"
        ids_r, in_r, out_r, h_r = positions(tokenizer, p_r, inp_r, ans_r)
        ids_d, in_d, out_d, h_d = positions(tokenizer, p_d, inp_d, ans_d)
        assert len(in_r) == len(in_d) == N and len(out_r) == len(out_d) == N
        # relative geometry must match exactly: same gaps between consecutive inputs and answers
        assert [b - a for a, b in zip(in_r, in_r[1:])] == [b - a for a, b in zip(in_d, in_d[1:])]
        assert [b - a for a, b in zip(out_r, out_r[1:])] == [b - a for a, b in zip(out_d, out_d[1:])]
        print(f"{task}: aligned; constant prefix offset {h_d - h_r} tokens, "
              f"{len([j for j in range(1, N) if j % K == 0])} post-marker positions per trial")
    print("Self-check passed: donor and recipient share input, answer and marker placement.")
    if args.dry:
        n = len(TASKS) * args.trials * (2 + len(SITES) * 65)
        print(f"about {n} forward passes at a 64-layer model")
        return

    import torch
    from tasks.common.runner import load_model
    model = load_model(args.model)
    model.eval()
    layers = None
    for path in ("model.layers", "model.language_model.layers", "language_model.layers",
                 "language_model.model.layers"):
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

    def run(ids, hooks=(), want_hidden=False):
        hs = [l.register_forward_hook(h) for l, h in hooks]
        try:
            with torch.no_grad():
                out = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=want_hidden)
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
        lo = [x for x in lo if not math.isnan(x)]
        return (sum(lo) / len(lo) if lo else float("nan")), sum(ac) / len(ac)

    tag = args.model.split("/")[-1]
    out_dir = REPO / "results" / "patching_aligned"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{tag}.csv"
    fields = ["Task", "Trial", "Site", "Layer", "logodds", "acc", "n_post", "prefix_offset"]
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
        for t in range(args.trials):
            inp_r, ans_r, p_r = build(engines[task], axis, K, t)
            inp_d, ans_d, p_d = build(donors[task], axis, K, t)
            assert inp_r == inp_d and ans_r == ans_d
            ids_r, in_r, out_r, h_r = positions(tokenizer, p_r, inp_r, ans_r)
            ids_d, in_d, out_d, h_d = positions(tokenizer, p_d, inp_d, ans_d)
            off = h_d - h_r
            post = [j for j in range(1, N) if j % K == 0]
            plain_in, plain_ans = inp_r.replace("|", ""), ans_r.replace("|", "")

            donor = run(ids_d, want_hidden=True)
            donor_h = [h[0].detach() for h in donor.hidden_states]
            if (task, str(t), "donor", "-1") not in done:
                lo, ac = metrics(donor.logits, out_d, plain_ans, plain_in, post)
                write(dict(Task=task, Trial=t, Site="donor", Layer=-1, logodds=lo, acc=ac,
                           n_post=len(post), prefix_offset=off))
            if (task, str(t), "recipient", "-1") not in done:
                m = run(ids_r)
                lo, ac = metrics(m.logits, out_r, plain_ans, plain_in, post)
                write(dict(Task=task, Trial=t, Site="recipient", Layer=-1, logodds=lo, acc=ac,
                           n_post=len(post), prefix_offset=off))

            site_pos = {
                "pre_in": [(in_r[j - 1], in_d[j - 1]) for j in post],
                "post_in": [(in_r[j], in_d[j]) for j in post],
                "all_in": list(zip(in_r, in_d)),
                "prev_out": [(out_r[j - 1], out_d[j - 1]) for j in post],
            }
            for site in SITES:
                pairs = site_pos[site]
                pr = torch.tensor([a for a, _ in pairs], device=dev)
                pd = [b for _, b in pairs]
                for layer in list(range(L)) + ["all"]:
                    if (task, str(t), site, str(layer)) in done:
                        continue

                    def make_hook(li):
                        src = donor_h[li + 1][pd].to(dev)

                        def hook(_mod, _inp, output):
                            h = output[0] if isinstance(output, tuple) else output
                            h[0, pr, :] = src.to(h.dtype)
                            return output
                        return hook

                    hooks = ([(layers[layer], make_hook(layer))] if layer != "all"
                             else [(layers[li], make_hook(li)) for li in range(L)])
                    m = run(ids_r, hooks)
                    lo, ac = metrics(m.logits, out_r, plain_ans, plain_in, post)
                    write(dict(Task=task, Trial=t, Site=site, Layer=layer, logodds=lo, acc=ac,
                               n_post=len(post), prefix_offset=off))
            print(f"done {task} trial {t}", flush=True)
    print("Finished.")


if __name__ == "__main__":
    main()
