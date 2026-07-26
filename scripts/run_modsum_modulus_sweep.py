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


class ModsumModulusSweep:
    def sample_string(self, length, modulus, rng):
        return "".join(str(rng.randint(0, modulus-1)) for _ in range(length))

    def compute_cumulative_sum_mod(self, digit_str, modulus):
        output = []
        cumsum = 0
        for digit_char in digit_str:
            cumsum = (cumsum + int(digit_char)) % modulus
            output.append(str(cumsum))
        return "".join(output)

    def _insert_boundaries_in_string(self, string_str, interval):
        if interval <= 0:
            return string_str
        bounded = []
        for i, c in enumerate(string_str):
            bounded.append(c)
            if (i + 1) % interval == 0 and i != len(string_str) - 1:
                bounded.append("|")
        return "".join(bounded)

    def build_prompt(self, length, interval, modulus, rng=None):
        rng = rng or random
        input_str_base = self.sample_string(length, modulus, rng)
        output_str_base = self.compute_cumulative_sum_mod(input_str_base, modulus)

        input_bounded = self._insert_boundaries_in_string(input_str_base, interval)
        output_bounded = self._insert_boundaries_in_string(output_str_base, interval)

        prompt = f"""You are a deterministic sequence transformer.
Task: Running sum modulo {modulus}.
Alphabet: 0-{modulus-1}
Modulus: {modulus}
Rule: At each position, output the cumulative sum of all digits so far, modulo {modulus}.
Sum all digits from the start up to this position, divide by {modulus}, output the remainder.
Example: Input 2314 with mod 10 → Output 2560
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

        return input_bounded, output_bounded, prompt


MODSUM_SEQUENCE_LENGTHS = [4, 8, 16, 32, 64]
MODSUM_PERIODIC_INTERVALS = [0, 1, 2, 3, 4, 8, 16]
MODSUM_MODULI = [2, 4, 8]
MODSUM_TRIALS = 20

engine_modsum = ModsumModulusSweep()


def run_modsum_trial(task_params):
    N, k, M, t = task_params
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 40_000 * N + 100 * k + 10 * M + t
    rng = random.Random(local_seed)

    inp, truth, prompt = engine_modsum.build_prompt(N, interval=k, modulus=M, rng=rng)

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
        "Method": method, "Interval": k, "Modulus": M, "Seq_Length": N, "Trial": t,
        "TAR": tar, "SAR": sar, "First_Error_Pos": first_error,
        "Input_Given": inp, "Ground_Truth": truth, "LLM_Output": clean_pred,
    }


modsum_tasks = [
    (N, k, M, t)
    for N in MODSUM_SEQUENCE_LENGTHS
    for k in MODSUM_PERIODIC_INTERVALS
    for M in MODSUM_MODULI
    for t in range(MODSUM_TRIALS)
    if k == 0 or k < 2 * N + 1
]

print(f"\nStarting MODSUM (modulus sweep) grid: {len(modsum_tasks)} evaluations...")

modsum_results = []
for task in tqdm(modsum_tasks, desc="Modsum (modulus sweep)"):
    try:
        modsum_results.append(run_modsum_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

modsum_df = pd.DataFrame(modsum_results)
modsum_csv = os.path.join(RESULTS_DIR, f"modsum_modulus_sweep_{MODEL_TAG}.csv")
modsum_df.to_csv(modsum_csv, index=False)
print(f"\nModsum (modulus sweep) complete. Results saved to {modsum_csv}")

summary = modsum_df.groupby(["Modulus", "Seq_Length", "Interval"])["TAR"].agg(["mean", "std"]).reset_index()
print("\n--- MEAN TAR BY (Modulus, Seq_Length, Interval) ---")
print(summary.to_string(index=False))
