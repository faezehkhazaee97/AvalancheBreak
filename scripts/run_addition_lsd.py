import matplotlib
matplotlib.use("Agg")

# ============================================================
# MODEL SELECTION
# ============================================================
# Change MODEL_ID (and re-run the notebook top to bottom) to repeat the
# exact same analysis at a different model scale -- e.g. to check whether
# the boundary-sink / entropy effects studied below hold up as the model
# gets bigger. Outputs are namespaced under results/<MODEL_TAG>/ so runs
# at different scales don't overwrite each other and can be compared
# side by side.

import os

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"  # ~28GB in bf16 -- fits alongside a 7B run on an 80GB A100

MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"Results directory: {RESULTS_DIR}/")

# ==== MODEL LOADER ====

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random

print(f"Loading {MODEL_ID} into VRAM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager",
)
model.eval()


# ==== LSD-FIRST ADDITION GRID ====

import pandas as pd
import numpy as np
import random
import itertools
import torch
import os
import re
from tqdm import tqdm

# ============================================================
# ADDITION, CARRY-DIRECTION HYPOTHESIS TEST (least-significant-digit-first)
# ============================================================
# Hypothesis (from external review of these results): addition's carries
# propagate least-significant to most-significant -- i.e. RIGHT-TO-LEFT --
# opposite to left-to-right reading/generation order. If the model needs to
# look right toward not-yet-"computed" less-significant digits to resolve a
# carry, periodic "|" boundaries inserted left-to-right may sever exactly
# that backward pathway, which would explain Cellular scoring *worse* than
# Base on this task (0/6 in diagnostic testing) unlike cyclic-shift or even
# modsum (whose state flows left-to-right, matching generation direction).
#
# Cheap test: write both addends AND the expected sum least-significant-
# digit-first (reversed), keep the same periodic boundary placement. If
# Cellular stops being worse than Base once digit order matches carry
# direction, that's real evidence for the carry-direction explanation
# rather than "boundaries just don't work for stateful tasks in general".
# Reuses the `model` / `tokenizer` already loaded above.

class AdditionTaskEngineLSD:
    def _sample_pair(self, length, rng):
        for _ in range(200):
            a = [rng.randrange(10) for _ in range(length)]
            b = [rng.randrange(10) for _ in range(length)]
            if a[0] == 0: a[0] = rng.randrange(1, 10)
            if b[0] == 0: b[0] = rng.randrange(1, 10)
            carry = 0
            total = []
            for i in range(length - 1, -1, -1):
                s = a[i] + b[i] + carry
                total.append(s % 10)
                carry = s // 10
            if carry == 0:
                return a, b, list(reversed(total))
        raise RuntimeError(f"failed to sample a non-overflowing {length}-digit pair")

    def build_prompt(self, length, interval, rng=None):
        rng = rng or random
        a, b, total = self._sample_pair(length, rng)
        # Reverse to least-significant-digit-first for addends AND the
        # expected output -- the only change from the normal-order task.
        a_str = "".join(map(str, a))[::-1]
        b_str = "".join(map(str, b))[::-1]
        sum_str = "".join(map(str, total))[::-1]

        combined = list(a_str) + ["+"] + list(b_str)
        bounded = []
        for i, d in enumerate(combined):
            bounded.append(d)
            if interval > 0 and (i + 1) % interval == 0 and i != len(combined) - 1:
                bounded.append("|")
        input_str = "".join(bounded)

        prompt = f"""You are a deterministic arithmetic calculator.
Task: Add two {length}-digit numbers. Both input numbers and the expected output are written LEAST-SIGNIFICANT-DIGIT FIRST (reversed order, i.e. the ones digit comes first).
Constraint: The input may contain vertical bars (|) as structural boundaries -- they carry no numeric meaning, ignore them when computing.
Output ONLY the {length}-digit sum, also least-significant-digit-first. No spaces, no explanations.

Input: {input_str}
Output: """
        return input_str, sum_str, prompt


ADD_LSD_SEQUENCE_LENGTHS = [4, 8, 16, 32, 64]
ADD_LSD_PERIODIC_INTERVALS = [0, 4, 8, 16]
ADD_LSD_TRIALS = 20

engine_add_lsd = AdditionTaskEngineLSD()

def run_addition_lsd_trial(task_params):
    N, k, t = task_params
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 130_000 * N + 100 * k + t
    rng = random.Random(local_seed)

    inp, truth, prompt = engine_add_lsd.build_prompt(N, interval=k, rng=rng)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    max_tokens_needed = N + 20
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = full_response[len(prompt):].strip()
    pred = pred.split("Explanation")[0].split("\n")[0].strip()

    match = re.match(r'^(\d+)', pred)
    clean_pred = match.group(1)[:len(truth)] if match else ""

    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == truth[i] else 0 for i in range(len(truth))]
    tar = sum(binary_series) / len(truth) if truth else 0.0
    sar = 1.0 if (len(binary_series) == len(truth) and sum(binary_series) == len(binary_series)) else 0.0
    first_error = binary_series.index(0) if 0 in binary_series else len(binary_series)

    return {
        "Method": method, "Interval": k, "Seq_Length": N, "Trial": t,
        "TAR": tar, "SAR": sar, "First_Error_Pos": first_error,
        "Input_Given": inp, "Ground_Truth": truth, "LLM_Output": clean_pred,
    }

addition_lsd_tasks = [
    (N, k, t)
    for N in ADD_LSD_SEQUENCE_LENGTHS
    for k in ADD_LSD_PERIODIC_INTERVALS
    for t in range(ADD_LSD_TRIALS)
    if k == 0 or k < 2 * N + 1
]
print(f"\nStarting LSD-first addition grid search: {len(addition_lsd_tasks)} evaluations...")

addition_lsd_results = []
for task in tqdm(addition_lsd_tasks, desc="Addition (LSD-first)"):
    try:
        addition_lsd_results.append(run_addition_lsd_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

addition_lsd_df = pd.DataFrame(addition_lsd_results)
addition_lsd_csv = os.path.join(RESULTS_DIR, "addition_lsd_grid.csv")
addition_lsd_df.to_csv(addition_lsd_csv, index=False)
print(f"\nLSD-first addition grid search complete. Saved to '{addition_lsd_csv}'")

summary = addition_lsd_df.groupby(["Seq_Length", "Method"])[["TAR", "SAR"]].mean().reset_index()
print("\n--- AVERAGE ACCURACY BY LENGTH & METHOD (LSD-first) ---")
print(summary.to_string(index=False))

pivot = summary.pivot(index="Seq_Length", columns="Method", values="TAR")
if "Base" in pivot.columns and "Cellular_Bounds" in pivot.columns:
    pivot["Cellular_minus_Base"] = pivot["Cellular_Bounds"] - pivot["Base"]
    print("\n--- DOES CELLULAR STILL SCORE WORSE THAN BASE, ONCE DIGIT ORDER MATCHES CARRY DIRECTION? ---")
    print(pivot.round(4).to_string())
    n_worse = int((pivot["Cellular_minus_Base"] < 0).sum())
    print(f"\nCellular still worse than Base at {n_worse}/{len(pivot)} tested lengths "
          f"(compare against the normal-order addition task's own Base-vs-Cellular gap above).")
