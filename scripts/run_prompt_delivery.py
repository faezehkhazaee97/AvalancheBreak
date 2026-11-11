#!/usr/bin/env python3
"""Do the effects depend on the chat template or the system prompt? (Reviewer question.)

Qwen2.5's chat template inserts its own system prompt ("You are Qwen, created by Alibaba Cloud.
You are a helpful assistant.") whenever none is supplied, so every run in the paper carries it.
Here the same prompts are delivered three other ways at Qwen2.5-32B:

  nosys       an explicit empty system message, which suppresses that default
  persona     a task-oriented system message ("You are a precise, deterministic text
              transformer. Apply the stated rule exactly and output nothing else.")
  notemplate  no chat template at all: the raw prompt text is fed to the model as a
              completion, the closest an instruction-tuned checkpoint gets to base-model use

The default-template condition is not regenerated: the main grid used the same prompts, seeds,
budget, and greedy decoding, so scripts/analyze_prompt_delivery.py reads it from
results_clean/ and checks that the overlap reproduces.

Tasks: running_xor (dependency; the blind spot) and random_permutation_digits at A = 8
(position-local; the gain). N in {32, 64}, no markers and "|" every k in {4, 16}, 20 trials.

Output: results/prompt_delivery/<model_tag>.csv
Usage:  python scripts/run_prompt_delivery.py --model Qwen/Qwen2.5-32B-Instruct [--dry]
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
LENGTHS, KS = [32, 64], [0, 4, 16]
PERSONA = ("You are a precise, deterministic text transformer. Apply the stated rule exactly and "
           "output nothing else.")
CONDITIONS = ["nosys", "persona", "notemplate"]


def build(engine, axis, N, k, t):
    rng = random.Random(_local_seed(N, k, axis, t))
    return engine.build_prompt(N, k, rng, axis_val=axis) if axis else engine.build_prompt(N, k, rng)


def encode(tokenizer, prompt, cond):
    if cond == "notemplate":
        return tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    msgs = {"nosys": [{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
            "persona": [{"role": "system", "content": PERSONA}, {"role": "user", "content": prompt}],
            }[cond]
    return tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True, enable_thinking=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    engines = {t: (importlib.import_module(f"tasks.{t}.engine"), a) for t, a in TASKS.items()}
    for t, (e, a) in engines.items():
        _, truth, _ = build(e, a, 32, 4, 0)
        assert scoring.score(truth, truth)["TAR"] == 1.0, t
    print("Self-check passed: every truth scores 1.0.")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    e, a = engines["running_xor"]
    _, _, p0 = build(e, a, 32, 4, 0)
    texts = {c: tokenizer.decode(encode(tokenizer, p0, c)["input_ids"][0]) for c in CONDITIONS}
    assert "You are Qwen" not in texts["nosys"], "empty system message did not suppress the default"
    assert PERSONA in texts["persona"] and "You are Qwen" not in texts["persona"]
    assert "<|im_start|>" not in texts["notemplate"], "raw condition still carries template markers"
    print("Self-check passed: each condition's rendered prompt is as intended.")
    if args.dry:
        for c in CONDITIONS:
            print(f"----- {c} -----\n{texts[c][:230]}\n...")
        print(len(CONDITIONS) * len(TASKS) * len(LENGTHS) * len(KS) * args.trials, "generations")
        return

    import torch
    from tasks.common.runner import load_model

    tag = args.model.split("/")[-1]
    out_dir = REPO / "results" / "prompt_delivery"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{tag}.csv"
    fields = ["Task", "Condition", "Method", "Interval", "Seq_Length", "Axis", "Trial", "TAR", "SAR",
              "Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output"]
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            done = {(r["Task"], r["Condition"], r["Interval"], r["Seq_Length"], r["Trial"]) for r in csv.DictReader(f)}
    model = load_model(args.model)
    model.eval()
    for cond in CONDITIONS:
        for task, (engine, axis) in engines.items():
            for N in LENGTHS:
                for k in KS:
                    for t in range(args.trials):
                        if (task, cond, str(k), str(N), str(t)) in done:
                            continue
                        inp, truth, prompt = build(engine, axis, N, k, t)
                        enc = {kk: v.to(model.device) for kk, v in encode(tokenizer, prompt, cond).items()}
                        with torch.no_grad():
                            out = model.generate(**enc, max_new_tokens=max_new_tokens(engine, N, k),
                                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
                        raw = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                        res = scoring.score(raw, truth)
                        new = not out_csv.exists()
                        with open(out_csv, "a", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=fields)
                            if new:
                                w.writeheader()
                            w.writerow(dict(Task=task, Condition=cond, Method="Base" if k == 0 else "Cellular_Bounds",
                                            Interval=k, Seq_Length=N, Axis=axis or 2, Trial=t, TAR=res["TAR"],
                                            SAR=res["SAR"], Input_Given=inp, Ground_Truth=res["clean_truth"],
                                            LLM_Output=res["clean_pred"], Raw_Model_Output=raw))
                    print(f"done {cond} {task} N={N} k={k}", flush=True)
    print("Finished.")


if __name__ == "__main__":
    main()
