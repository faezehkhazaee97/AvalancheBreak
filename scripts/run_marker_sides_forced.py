#!/usr/bin/env python3
"""The input/output marker 2x2 again, with the output format forced instead of requested.

scripts/run_marker_sides.py asked the model to omit or insert markers and it frequently ignored
the instruction (62% of random_permutation generations emitted markers when asked not to; none
produced the requested placement when asked to insert them). The single-sided cells there are
therefore effects of the instruction, not of a realized format.

Here a logits processor makes the format certain. At each step it counts the answer characters
generated so far (markers and spaces ignored) and either forces the marker token, when the
condition says a marker is due at that offset, or removes every token containing a marker:

  base     no markers in the input, none permitted in the output
  in_only  markers in the input, none permitted in the output
  out_only no markers in the input, a marker forced every k answer characters
  in_out   markers in the input, a marker forced every k answer characters

All four cells are generated under the same constraint machinery, so the comparison is not
confounded by constrained decoding itself; base and in_out also serve as a check against the
unconstrained runs. The prompts are those of run_marker_sides.py, so the instruction still
matches the condition. Greedy otherwise, same seeds, same budget.

Output: results/marker_sides_forced/<model_tag>.csv
Usage:  python scripts/run_marker_sides_forced.py --model Qwen/Qwen2.5-32B-Instruct [--dry]
"""

import argparse
import csv
import importlib
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from tasks.common import scoring  # noqa: E402
from tasks.common.runner import _local_seed, max_new_tokens  # noqa: E402
from run_marker_sides import build, TASKS, LENGTHS, KS  # noqa: E402

CONDS = ["base", "in_only", "out_only", "in_out"]


def make_processor(tokenizer, prompt_len, n_answer, k, want_markers):
    """Force (or forbid) the marker at each answer offset. k is the spacing; want_markers says
    whether the output should carry markers at all."""
    import torch
    from transformers import LogitsProcessor

    bar_ids = [i for tok, i in tokenizer.get_vocab().items()
               if "|" in tokenizer.convert_tokens_to_string([tok])]
    force_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("|"))
    force_id = force_id[0] if isinstance(force_id, list) else force_id
    bar_t = torch.tensor(sorted(set(bar_ids)))

    class P(LogitsProcessor):
        def __call__(self, input_ids, scores):
            text = tokenizer.decode(input_ids[0][prompt_len:], skip_special_tokens=True)
            chars = text.replace("|", "").replace(" ", "").replace("\n", "")
            c, m = len(chars), text.count("|")
            if c >= n_answer:          # answer complete: stop constraining, let it end
                return scores
            due = want_markers and c > 0 and c % k == 0 and m < (n_answer - 1) // k
            if due and not text.rstrip().endswith("|"):
                mask = torch.full_like(scores, float("-inf"))
                mask[:, force_id] = scores[:, force_id]
                return mask
            scores[:, bar_t.to(scores.device)] = float("-inf")
            return scores

    return P()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    engines = {t: importlib.import_module(f"tasks.{t}.engine") for t in TASKS}
    if args.dry:
        print("conditions:", CONDS)
        print(len(TASKS) * len(LENGTHS) * args.trials * (1 + 3 * len(KS)), "generations")
        return

    import torch
    from transformers import AutoTokenizer, LogitsProcessorList
    from tasks.common.runner import load_model

    tag = args.model.split("/")[-1]
    out_dir = REPO / "results" / "marker_sides_forced"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{tag}.csv"
    fields = ["Task", "Condition", "Interval", "Seq_Length", "Axis", "Trial", "TAR", "SAR",
              "Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output"]
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            done = {(r["Task"], r["Condition"], r["Interval"], r["Seq_Length"], r["Trial"]) for r in csv.DictReader(f)}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    model.eval()
    for cond in CONDS:
        for task, axis in TASKS.items():
            engine = engines[task]
            for k in ([KS[0]] if cond == "base" else KS):
                for N in LENGTHS:
                    for t in range(args.trials):
                        rec_k = 0 if cond == "base" else k
                        if (task, cond, str(rec_k), str(N), str(t)) in done:
                            continue
                        # prompts: base and in_out come from the engine, the single-sided ones
                        # from run_marker_sides.build so the instruction matches the condition
                        if cond == "base":
                            rng = random.Random(_local_seed(N, 0, axis, t))
                            shown, truth, prompt = (engine.build_prompt(N, 0, rng, axis_val=axis) if axis
                                                    else engine.build_prompt(N, 0, rng))
                        elif cond == "in_out":
                            rng = random.Random(_local_seed(N, k, axis, t))
                            shown, truth, prompt = (engine.build_prompt(N, k, rng, axis_val=axis) if axis
                                                    else engine.build_prompt(N, k, rng))
                        else:
                            shown, truth, prompt = build(engine, axis, N, k, t, cond)
                        want = cond in ("out_only", "in_out")
                        enc = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                           add_generation_prompt=True, return_tensors="pt",
                                                           return_dict=True, enable_thinking=False)
                        enc = {kk: v.to(model.device) for kk, v in enc.items()}
                        plen = enc["input_ids"].shape[1]
                        proc = LogitsProcessorList([make_processor(tokenizer, plen, N, k, want)])
                        with torch.no_grad():
                            out = model.generate(**enc, max_new_tokens=max_new_tokens(engine, N, rec_k),
                                                 do_sample=False, logits_processor=proc,
                                                 pad_token_id=tokenizer.eos_token_id)
                        raw = tokenizer.decode(out[0][plen:], skip_special_tokens=True)
                        res = scoring.score(raw, truth)
                        new = not out_csv.exists()
                        with open(out_csv, "a", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=fields)
                            if new:
                                w.writeheader()
                            w.writerow(dict(Task=task, Condition=cond, Interval=rec_k, Seq_Length=N, Axis=axis or 2,
                                            Trial=t, TAR=res["TAR"], SAR=res["SAR"], Input_Given=shown,
                                            Ground_Truth=res["clean_truth"], LLM_Output=res["clean_pred"],
                                            Raw_Model_Output=raw))
                    print(f"done {cond} {task} N={N} k={rec_k}", flush=True)
    print("Finished.")


if __name__ == "__main__":
    main()
