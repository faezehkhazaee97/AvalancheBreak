import matplotlib
matplotlib.use("Agg")

import os
import random
import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
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


class AdditionDigitRangeSweep:
    def _sample_pair(self, length, digit_max, rng):
        for _ in range(200):
            a = [rng.randrange(digit_max) for _ in range(length)]
            b = [rng.randrange(digit_max) for _ in range(length)]
            if a[0] == 0: a[0] = rng.randrange(1, digit_max)
            if b[0] == 0: b[0] = rng.randrange(1, digit_max)
            carry = 0
            total = []
            for i in range(length - 1, -1, -1):
                s = a[i] + b[i] + carry
                total.append(s % digit_max)
                carry = s // digit_max
            if carry == 0:
                return a, b, list(reversed(total))
        raise RuntimeError(f"failed to sample")

    def _insert_boundaries_in_number(self, num_str, interval):
        if interval <= 0:
            return num_str
        bounded = []
        for i, d in enumerate(num_str):
            bounded.append(d)
            if (i + 1) % interval == 0 and i != len(num_str) - 1:
                bounded.append("|")
        return "".join(bounded)

    def build_prompt(self, length, interval, digit_max, rng=None):
        rng = rng or random
        a, b, total = self._sample_pair(length, digit_max, rng)

        a_str = "".join(map(str, a))
        b_str = "".join(map(str, b))
        sum_str = "".join(map(str, total))

        a_bounded = self._insert_boundaries_in_number(a_str, interval)
        b_bounded = self._insert_boundaries_in_number(b_str, interval)

        input_str = f"{a_bounded}+{b_bounded}"

        prompt = f"""You are a deterministic arithmetic calculator.
Task: Add two {length}-digit numbers in base {digit_max}.
Constraint: The input may contain vertical bars (|) as structural boundaries.
Ignore them when computing. The + symbol is the operator, ignored when summing.
Output ONLY the {length}-digit sum, zero-padded. No spaces, no explanations.

Input: {input_str}
Output: """

        return input_str, sum_str, prompt


ADD_SEQUENCE_LENGTHS = [4, 8, 16, 32, 64]
ADD_PERIODIC_INTERVALS = [0, 1, 2, 3, 4, 8, 16]
ADD_DIGIT_RANGES = [2, 4, 8]
ADD_TRIALS = 20

engine_add = AdditionDigitRangeSweep()


def run_addition_trial(task_params):
    N, k, D, t = task_params
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 90_000 * N + 100 * k + 10 * D + t
    rng = random.Random(local_seed)

    inp, truth, prompt = engine_add.build_prompt(N, interval=k, digit_max=D, rng=rng)

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
        "Method": method, "Interval": k, "Digit_Max": D, "Seq_Length": N, "Trial": t,
        "TAR": tar, "SAR": sar, "First_Error_Pos": first_error,
        "Input_Given": inp, "Ground_Truth": truth, "LLM_Output": clean_pred,
    }


addition_tasks = [
    (N, k, D, t)
    for N in ADD_SEQUENCE_LENGTHS
    for k in ADD_PERIODIC_INTERVALS
    for D in ADD_DIGIT_RANGES
    for t in range(ADD_TRIALS)
    if k == 0 or k < 2 * N + 1
]

print(f"\nStarting INTEGER ADDITION (digit range sweep) grid: {len(addition_tasks)} evaluations...")

addition_results = []
for task in tqdm(addition_tasks, desc="Integer Addition (digit range sweep)"):
    try:
        addition_results.append(run_addition_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

addition_df = pd.DataFrame(addition_results)
addition_csv = os.path.join(RESULTS_DIR, f"addition_digit_range_sweep_{MODEL_TAG}.csv")
addition_df.to_csv(addition_csv, index=False)
print(f"\nInteger Addition (digit range sweep) complete. Results saved to {addition_csv}")

summary = addition_df.groupby(["Digit_Max", "Seq_Length", "Interval"])["TAR"].agg(["mean", "std"]).reset_index()
print("\n--- MEAN TAR BY (Digit_Max, Seq_Length, Interval) ---")
print(summary.to_string(index=False))
