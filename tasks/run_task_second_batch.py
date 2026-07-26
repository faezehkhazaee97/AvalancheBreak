"""Generic second-batch runner for any of the 6 tasks: runs a given trial
range through the same engine/scoring pipeline as tasks/run_task.py, so the
result can be pooled with an existing first batch to check findings survive
independent sampling -- same standard already applied to running_xor and
running_sum_local this session.

Usage:
    python -m tasks.run_task_second_batch <task_name> <model_id> <trial_start> <trial_end>

Output: results_clean/<task_name>/grid/<model_tag>_trials<start>-<end-1>.csv
"""

import importlib
import os
import random
import sys
from collections import Counter

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tasks.common import scoring as scoring_mod
from tasks.common.boundary import valid_intervals, STANDARD_LENGTHS, STANDARD_INTERVALS
from tasks.common.runner import _local_seed, _seed_everything


def is_degenerate(s):
    s = str(s)
    if len(s) < 5:
        return False
    counts = Counter(s)
    return counts.most_common(1)[0][1] / len(s) > 0.85


def main():
    task_name, model_id = sys.argv[1], sys.argv[2]
    trial_start, trial_end = int(sys.argv[3]), int(sys.argv[4])
    model_tag = model_id.split("/")[-1]
    engine = importlib.import_module(f"tasks.{task_name}.engine")

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()
    _seed_everything()

    lengths = getattr(engine, "LENGTHS", STANDARD_LENGTHS)
    intervals = getattr(engine, "INTERVALS", STANDARD_INTERVALS)
    axis_values = engine.AXIS_VALUES or [None]

    grid = [(N, k, axis_val, t) for N in lengths for k in valid_intervals(N, intervals)
            for axis_val in axis_values for t in range(trial_start, trial_end)]
    print(f"Grid size: {len(grid)} evaluations (task={task_name}, model={model_tag}, trials {trial_start}-{trial_end - 1})")

    rows = []
    for idx, (N, k, axis_val, t) in enumerate(grid):
        method = "Base" if k == 0 else "Cellular_Bounds"
        seed = _local_seed(N, k, axis_val, t)
        rng = random.Random(seed)
        if engine.AXIS_NAME:
            input_str, truth, prompt = engine.build_prompt(N, k, rng, axis_val=axis_val)
        else:
            input_str, truth, prompt = engine.build_prompt(N, k, rng)

        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        inputs = {kk: vv.to(model.device) for kk, vv in enc.items()}
        max_new = int(N * 1.5) + 20 if engine.OUTPUT_HAS_BOUNDARIES else N + 20

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_only = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(gen_only, skip_special_tokens=True)
        result = scoring_mod.score(raw_output, truth)

        row = {
            "Method": method, "Interval": k, "Seq_Length": N, "Trial": t,
            "Input_Given": input_str, "Ground_Truth": result["clean_truth"],
            "LLM_Output": result["clean_pred"], "TAR": result["TAR"], "SAR": result["SAR"],
            "Degenerate": is_degenerate(result["clean_pred"]),
        }
        if engine.AXIS_NAME:
            row[engine.AXIS_NAME] = axis_val
        rows.append(row)
        if idx % 300 == 0:
            print(f"  [{idx}/{len(grid)}] N={N} k={k} axis={axis_val} t={t}: TAR={result['TAR']:.3f}")

    df = pd.DataFrame(rows)
    out_dir = os.path.join(REPO_ROOT, "results_clean", task_name, "grid")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{model_tag}_trials{trial_start}-{trial_end - 1}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    main()
