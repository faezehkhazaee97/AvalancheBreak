import matplotlib
matplotlib.use("Agg")

# ============================================================
# INTEGER ADDITION, HEADLINE GRID RE-RUN WITH THE "+" BOUNDARY BUG FIXED
# ============================================================
# Identical to run_addition_templated_v2.py (same lengths, intervals, trial
# count, and per-(N,k,t) seeds, so the sampled (a, b) pairs are IDENTICAL
# trial-for-trial) with exactly one change: boundaries are inserted within
# each operand separately, and the "+" is joined in afterward untouched --
# the same fix already used in run_addition_structured_boundaries.py --
# instead of treating list(a_str) + ["+"] + list(b_str) as one flat stream
# and slicing through the "+" arbitrarily.
#
# Everything else (prompt wording, chat template, decoding, scoring) is left
# unchanged so this is a clean, isolated test of whether the headline
# "boundaries hurt addition at every length" result survives once the
# +-straddling artifact is removed.

import os
import sys

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"Results directory: {RESULTS_DIR}/")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random, re
import numpy as np

print(f"Loading {MODEL_ID} into VRAM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager",
)
model.eval()

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit
from tqdm import tqdm


class AdditionTaskEngineBoundaryFixed:
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

    def _insert_boundaries_in_number(self, num_str, interval):
        if interval <= 0:
            return num_str
        bounded = []
        for i, d in enumerate(num_str):
            bounded.append(d)
            if (i + 1) % interval == 0 and i != len(num_str) - 1:
                bounded.append("|")
        return "".join(bounded)

    def build_prompt(self, length, interval, rng=None):
        rng = rng or random
        a, b, total = self._sample_pair(length, rng)
        a_str = "".join(map(str, a))
        b_str = "".join(map(str, b))
        sum_str = "".join(map(str, total))

        a_bounded = self._insert_boundaries_in_number(a_str, interval)
        b_bounded = self._insert_boundaries_in_number(b_str, interval)
        input_str = f"{a_bounded}+{b_bounded}"

        prompt = f"""You are a deterministic arithmetic calculator.
Task: Add two {length}-digit numbers.
Constraint: The input may contain vertical bars (|) as structural boundaries -- they carry no numeric meaning, ignore them when computing.
Output ONLY the {length}-digit sum (zero-padded to {length} digits if needed). No spaces, no explanations.

Input: {input_str}
Output: """
        return input_str, sum_str, prompt


ADD_SEQUENCE_LENGTHS = [4, 8, 16, 32, 64]
ADD_PERIODIC_INTERVALS = [0, 1, 2, 3, 4, 8, 16]
ADD_TRIALS = 20

engine_add = AdditionTaskEngineBoundaryFixed()

def run_addition_trial(task_params):
    N, k, t = task_params
    method = "Base" if k == 0 else "Cellular_Bounds"
    # Same seed formula as run_addition_templated_v2.py -> identical (a, b) pairs per trial
    local_seed = 90_000 * N + 100 * k + t
    rng = random.Random(local_seed)

    inp, truth, prompt = engine_add.build_prompt(N, interval=k, rng=rng)

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = {kk: vv.to(model.device) for kk, vv in enc.items()}

    max_tokens_needed = N + 20
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_only = outputs[0][inputs["input_ids"].shape[1]:]
    pred = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
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

addition_tasks = [
    (N, k, t)
    for N in ADD_SEQUENCE_LENGTHS
    for k in ADD_PERIODIC_INTERVALS
    for t in range(ADD_TRIALS)
    if k == 0 or k < 2 * N + 1
]
print(f"\nStarting BOUNDARY-FIXED integer-addition grid: {len(addition_tasks)} evaluations...")

addition_results = []
for task in tqdm(addition_tasks, desc="Addition (boundary-fixed)"):
    try:
        addition_results.append(run_addition_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

addition_df = pd.DataFrame(addition_results)
addition_csv = os.path.join(RESULTS_DIR, f"addition_grid_boundary_fixed_{MODEL_TAG}.csv")
addition_df.to_csv(addition_csv, index=False)
print(f"\nAddition grid (boundary-fixed) complete. Saved to '{addition_csv}'")

summary = addition_df.groupby(["Seq_Length", "Method"])[["TAR", "SAR"]].mean().reset_index()
print("\n--- AVERAGE ACCURACY BY LENGTH & METHOD (boundary-fixed) ---")
print(summary.to_string(index=False))


# ==== REGRESSION ====

df = addition_df.copy()
df['Inv_K'] = df['Interval'].apply(lambda k: 0.0 if k == 0 else 1.0/k)
df = df[df['Seq_Length'] >= 8]

reg_summary = df.groupby(['Seq_Length', 'Inv_K'])[['TAR','SAR']].agg(['mean','std'])
reg_summary.columns = ['_'.join(c) for c in reg_summary.columns]
reg_summary = reg_summary.reset_index().sort_values(['Seq_Length','Inv_K'])
reg_summary[['TAR_std','SAR_std']] = reg_summary[['TAR_std','SAR_std']].fillna(0.0)

regression_results = []
for metric in ['TAR', 'SAR']:
    for N, sub in reg_summary.groupby('Seq_Length'):
        if len(sub) >= 2:
            slope, intercept, r, p, se = linregress(sub['Inv_K'], sub[f'{metric}_mean'])
            regression_results.append(dict(Metric=metric, Seq_Length=N, Slope=slope, R_squared=r**2))

results_df = pd.DataFrame(regression_results)
results_df.to_csv(os.path.join(RESULTS_DIR, f"addition_regression_boundary_fixed_{MODEL_TAG}.csv"), index=False)
print("\n--- SCALING LAW REGRESSION (boundary-fixed) ---")
print(results_df.round(4).to_string(index=False))

print("\nBoundary-fixed integer-addition grid rerun complete.")
