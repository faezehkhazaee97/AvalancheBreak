#!/usr/bin/env python3
"""Input markers or output markers? (Reviewer question: the intervention changes two things at
once -- it segments the input and it asks for a segmented output.)

A 2x2 at Qwen2.5-32B on the same underlying content, plus a baseline that drops the marker
instruction entirely:

  base_clean  no markers in the input, none asked for in the output, and NO sentence about
              bars in the prompt (the main grid's Base keeps that sentence even though its
              input has no bars; this isolates the instruction from the markers)
  base_instr  no markers either side, marker sentence present       = the main grid's Base
  in_only     markers in the input, output asked for without them
  out_only    no markers in the input, output asked for with a bar every k characters
  in_out      markers in the input, copied through to the output    = the main grid's Bounds

base_instr and in_out are not regenerated: the main grid used the same prompts, seeds, budget,
and greedy decoding, and scripts/analyze_marker_sides.py checks that the overlap reproduces.

Every condition at a given (task, N, k, trial) is built from one underlying symbol string, so
the four cells of the 2x2 differ only in where markers appear and what the prompt asks for.
Scoring strips markers from both sides as everywhere else, so the four are directly comparable;
marker-placement compliance is scored separately by the analysis script.

Tasks: running_xor (dependency; the blind spot) and random_permutation_digits at A = 8
(position-local; the gain). N in {32, 64}, k in {4, 16}, 20 trials, greedy decoding.

Output: results/marker_sides/<model_tag>.csv
Usage:  python scripts/run_marker_sides.py --model Qwen/Qwen2.5-32B-Instruct [--dry]
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

TASKS = {"running_xor": None, "random_permutation_digits": 8}
LENGTHS, KS = [32, 64], [4, 16]
ORIGINAL = ("Constraint: The input contains vertical bars (|). These are structural boundaries.\n"
            "DO NOT transform them. Copy them exactly to the output.")
IN_ONLY = ("Constraint: The input contains vertical bars (|). These are structural boundaries.\n"
           "DO NOT transform them, and DO NOT include them in your output: write only the "
           "transformed characters.")
OUT_ONLY = ("Constraint: Insert a vertical bar (|) into your output after every {k} characters. "
            "These are structural boundaries, not part of the transformation; the input contains none.")
NEW = ["base_clean", "in_only", "out_only"]  # the conditions this script generates


def build(engine, axis, N, k, t, cond):
    """One underlying string per (task, N, k, trial); the condition decides where markers go."""
    rng = random.Random(_local_seed(N, k, axis, t))
    inp, truth, prompt = (engine.build_prompt(N, k, rng, axis_val=axis) if axis
                          else engine.build_prompt(N, k, rng))
    assert prompt.count(ORIGINAL) == 1
    plain = inp.replace("|", "")
    head, sep, tail = prompt.rpartition("Input: ")
    assert sep and tail.endswith("\nOutput: ") and tail[: -len("\nOutput: ")] == inp
    if cond == "base_clean":
        head, shown = head.replace(ORIGINAL + "\n", ""), plain
    elif cond == "in_only":
        head, shown = head.replace(ORIGINAL, IN_ONLY), inp
    elif cond == "out_only":
        head, shown = head.replace(ORIGINAL, OUT_ONLY.format(k=k)), plain
    else:
        raise ValueError(cond)
    return shown, truth, head + "Input: " + shown + "\nOutput: "


def self_check():
    for task, axis in TASKS.items():
        engine = importlib.import_module(f"tasks.{task}.engine")
        for N in LENGTHS:
            for k in KS:
                strings = set()
                for cond in NEW:
                    shown, truth, prompt = build(engine, axis, N, k, 0, cond)
                    strings.add(shown.replace("|", ""))
                    assert scoring.score(truth, truth)["TAR"] == 1.0
                    assert prompt.count("Input: ") == 1 and prompt.endswith("\nOutput: ")
                    bars = shown.count("|")
                    assert bars == ((N - 1) // k if cond == "in_only" else 0), (cond, bars)
                    assert ("vertical bar" in prompt) == (cond != "base_clean"), cond
                assert len(strings) == 1, "conditions must share one underlying string"
    # base_clean must reuse the main grid Base's input string (same seed at k = 0)
    for task, axis in TASKS.items():
        engine = importlib.import_module(f"tasks.{task}.engine")
        rng = random.Random(_local_seed(32, 0, axis, 0))
        ref = (engine.build_prompt(32, 0, rng, axis_val=axis) if axis else engine.build_prompt(32, 0, rng))[0]
        assert build(engine, axis, 32, 0, 0, "base_clean")[0] == ref, task
    print("Self-check passed: shared strings, expected marker counts, instruction present or absent.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    self_check()
    engines = {t: (importlib.import_module(f"tasks.{t}.engine"), a) for t, a in TASKS.items()}
    if args.dry:
        e, a = engines["running_xor"]
        for cond in NEW:
            print(f"----- {cond} -----\n{build(e, a, 32, 4, 0, cond)[2][-300:]}")
        n = len(TASKS) * len(LENGTHS) * args.trials * (1 + 2 * len(KS))
        print(n, "generations")
        return

    import torch
    from transformers import AutoTokenizer
    from tasks.common.runner import load_model

    tag = args.model.split("/")[-1]
    out_dir = REPO / "results" / "marker_sides"
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
    for cond in NEW:
        for task, (engine, axis) in engines.items():
            # base_clean has no markers, so it is run once per length, not once per spacing
            for k in ([KS[0]] if cond == "base_clean" else KS):
                for N in LENGTHS:
                    for t in range(args.trials):
                        rec_k = 0 if cond == "base_clean" else k
                        if (task, cond, str(rec_k), str(N), str(t)) in done:
                            continue
                        seed_k = 0 if cond == "base_clean" else k
                        shown, truth, prompt = build(engine, axis, N, seed_k, t, cond)
                        enc = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                           add_generation_prompt=True, return_tensors="pt",
                                                           return_dict=True, enable_thinking=False)
                        enc = {kk: v.to(model.device) for kk, v in enc.items()}
                        with torch.no_grad():
                            out = model.generate(**enc, max_new_tokens=max_new_tokens(engine, N, rec_k),
                                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
                        raw = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
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
