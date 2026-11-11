#!/usr/bin/env python3
"""No-marker baselines whose prompt never mentions markers (reviewer question).

Throughout the paper the no-marker condition keeps the sentence describing the markers, so that
the only difference between conditions is whether markers appear in the input. On two tasks at
Qwen2.5-32B that sentence costs 0.02 to 0.04 (Section 7.1), which makes the reported gains
overstated by about that much. This script measures the cost across the grid: the same no-marker
prompts with every line that mentions a marker removed.

Tasks: the two tokenization controls for the position-local tasks (spaced, digits) and the four
dependency tasks. Models: the four stronger ones. N in {16, 32, 64} (at 4 and 8 the controls sit
at ceiling and the marker effect is zero), every alphabet or modulus value, 20 trials, greedy.

The stripper removes whole lines mentioning a bar or boundary from the prompt header and then
checks that no mention survives and that nothing else changed, so it works across the engines'
differing wordings.

Output: results/clean_baseline/<model_tag>.csv
Usage:  python scripts/run_clean_baseline.py --model Qwen/Qwen2.5-32B-Instruct [--dry]
"""

import argparse
import csv
import importlib
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common import scoring  # noqa: E402
from tasks.common.runner import _local_seed, max_new_tokens  # noqa: E402

TASKS = ["cyclic_shift_spaced", "random_permutation_spaced", "conditional_shift_spaced",
         "cyclic_shift_digits", "random_permutation_digits", "conditional_shift_digits",
         "copy_previous", "running_xor", "running_sum_local", "sliding_window_max"]
LENGTHS = [16, 32, 64]
MENTIONS = ("|", "bar", "boundar")


def strip_markers(prompt):
    """Remove the whole constraint block that describes the markers.

    The block starts at the line beginning "Constraint:" and continues over its wrapped lines
    ("DO NOT transform them...", "DO NOT include them in the window...") until the next
    instruction line or a blank line. Dropping only the first line would leave an orphan
    sentence referring to markers that are no longer mentioned.
    """
    head, sep, tail = prompt.rpartition("Input: ")
    assert sep, "prompt has no input line"
    lines = head.split("\n")
    start = next((i for i, l in enumerate(lines) if l.startswith("Constraint:")), None)
    assert start is not None, "no constraint line found"
    end = start + 1
    while end < len(lines):
        l = lines[end].strip()
        if not l or l.startswith("No spaces") or l.startswith("Write the output"):
            break
        end += 1
    keep = lines[:start] + lines[end:]
    clean = "\n".join(keep) + sep + tail
    low = clean[: clean.rindex("Input: ")].lower()
    for m in ("|", "bar", "boundar", "do not"):
        assert m not in low, f"'{m}' survived the strip"
    assert "no spaces" in low or "write the output" in low, "the formatting instruction was lost"
    return clean, end - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    engines = {t: importlib.import_module(f"tasks.{t}.engine") for t in TASKS}
    total = 0
    for t, e in engines.items():
        vals = e.AXIS_VALUES or [None]
        for N in LENGTHS:
            for v in vals:
                rng = random.Random(_local_seed(N, 0, v, 0))
                _, _, p = (e.build_prompt(N, 0, rng, axis_val=v) if e.AXIS_NAME else e.build_prompt(N, 0, rng))
                clean, n = strip_markers(p)
                assert n >= 1, f"{t}: no marker line found to drop"
                total += args.trials
    print(f"Self-check passed: a marker mention was found and removed in every task and cell.")
    if args.dry:
        e = engines["running_xor"]
        rng = random.Random(_local_seed(32, 0, None, 0))
        p = e.build_prompt(32, 0, rng)[2]
        print("----- original -----\n" + p[: p.index("Input: ")])
        print("----- stripped -----\n" + strip_markers(p)[0][: strip_markers(p)[0].index("Input: ")])
        print(total, "generations")
        return

    import torch
    from transformers import AutoTokenizer
    from tasks.common.runner import load_model

    tag = args.model.split("/")[-1]
    out_dir = REPO / "results" / "clean_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{tag}.csv"
    fields = ["Task", "Method", "Interval", "Seq_Length", "Axis", "Trial", "TAR", "SAR",
              "Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output"]
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            done = {(r["Task"], r["Seq_Length"], r["Axis"], r["Trial"]) for r in csv.DictReader(f)}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    model.eval()
    for task in TASKS:
        engine = engines[task]
        for axis in (engine.AXIS_VALUES or [None]):
            for N in LENGTHS:
                for t in range(args.trials):
                    if (task, str(N), str(axis or 2), str(t)) in done:
                        continue
                    rng = random.Random(_local_seed(N, 0, axis, t))
                    inp, truth, prompt = (engine.build_prompt(N, 0, rng, axis_val=axis) if engine.AXIS_NAME
                                          else engine.build_prompt(N, 0, rng))
                    prompt, _ = strip_markers(prompt)
                    enc = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                       add_generation_prompt=True, return_tensors="pt",
                                                       return_dict=True, enable_thinking=False)
                    enc = {kk: v.to(model.device) for kk, v in enc.items()}
                    with torch.no_grad():
                        out = model.generate(**enc, max_new_tokens=max_new_tokens(engine, N, 0), do_sample=False,
                                             pad_token_id=tokenizer.eos_token_id)
                    raw = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    res = scoring.score(raw, truth)
                    new = not out_csv.exists()
                    with open(out_csv, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fields)
                        if new:
                            w.writeheader()
                        w.writerow(dict(Task=task, Method="Base_clean", Interval=0, Seq_Length=N, Axis=axis or 2,
                                        Trial=t, TAR=res["TAR"], SAR=res["SAR"], Input_Given=inp,
                                        Ground_Truth=res["clean_truth"], LLM_Output=res["clean_pred"],
                                        Raw_Model_Output=raw))
                print(f"done {task} axis={axis} N={N}", flush=True)
    print("Finished.")


if __name__ == "__main__":
    main()
