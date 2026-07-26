import matplotlib
matplotlib.use("Agg")

# ============================================================
# INTEGER ADDITION WITH STRUCTURED BOUNDARIES
# ============================================================
# CORRECTED VERSION: Boundaries are placed intelligently to respect
# the structure of the problem:
# - Boundaries are inserted within each operand (first and second number)
# - The "+" operator is kept clear, with no boundaries immediately adjacent
# - Example: 12|34 + 56|78 (instead of broken: 12|34|+5|67|8)
#
# This allows fair testing of whether boundaries help or hurt based on
# the carry-direction hypothesis, not based on malformed prompts.

import os
import random
import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import linregress
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"Results directory: {RESULTS_DIR}/")

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


class AdditionTaskEngineStructured:
    def _sample_pair(self, length, rng):
        """Sample two N-digit numbers that don't overflow when added."""
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
        """Insert boundaries within a single number every k positions."""
        if interval <= 0:
            return num_str
        bounded = []
        for i, d in enumerate(num_str):
            bounded.append(d)
            if (i + 1) % interval == 0 and i != len(num_str) - 1:
                bounded.append("|")
        return "".join(bounded)

    def build_prompt(self, length, interval, rng=None):
        """
        Build addition prompt with structured boundaries.

        Example with N=4, k=2:
            Input: 12|34+56|78
            Output: 7012

        Key difference from naive approach:
        - Boundaries are inserted within each operand independently
        - The "+" operator is clearly separated, not buried in boundaries
        - No extra spaces/tokens added — only boundary placement differs from original
        - This respects the mathematical structure of the problem
        """
        rng = rng or random
        a, b, total = self._sample_pair(length, rng)

        a_str = "".join(map(str, a))
        b_str = "".join(map(str, b))
        sum_str = "".join(map(str, total))

        # Insert boundaries within each operand separately
        a_bounded = self._insert_boundaries_in_number(a_str, interval)
        b_bounded = self._insert_boundaries_in_number(b_str, interval)

        # Combine without extra spaces — only boundary placement differs
        input_str = f"{a_bounded}+{b_bounded}"

        prompt = f"""You are a deterministic arithmetic calculator.
Task: Add two {length}-digit numbers.
Constraint: The input may contain vertical bars (|) as structural boundaries -- they carry no numeric meaning, ignore them when computing. Spaces also carry no meaning.
Output ONLY the {length}-digit sum (zero-padded to {length} digits if needed). No spaces, no explanations.

Input: {input_str}
Output: """
        return input_str, sum_str, prompt


ADD_SEQUENCE_LENGTHS = [4, 8, 16, 32, 64]
ADD_PERIODIC_INTERVALS = [0, 1, 2, 3, 4, 8, 16]
ADD_TRIALS = 20

engine_add = AdditionTaskEngineStructured()


def run_addition_trial(task_params):
    N, k, t = task_params
    method = "Base" if k == 0 else "Cellular_Bounds"
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
print(f"\nStarting STRUCTURED-BOUNDARY integer-addition grid: {len(addition_tasks)} evaluations...")

addition_results = []
for task in tqdm(addition_tasks, desc="Addition (structured boundaries)"):
    try:
        addition_results.append(run_addition_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

addition_df = pd.DataFrame(addition_results)
addition_csv = os.path.join(RESULTS_DIR, "addition_grid_structured_boundaries.csv")
addition_df.to_csv(addition_csv, index=False)
print(f"\nAddition (structured) complete. Results saved to {addition_csv}")

# Summary stats
summary = addition_df.groupby(["Seq_Length", "Interval"])["TAR"].agg(["mean", "std"]).reset_index()
print("\n--- MEAN TAR BY (Seq_Length, Interval) ---")
print(summary.to_string(index=False))
