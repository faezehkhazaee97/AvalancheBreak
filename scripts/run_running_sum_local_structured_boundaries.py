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


class RunningSumLocalTaskEngine:
    def sample_string(self, length, rng):
        return "".join(str(rng.randint(0, 9)) for _ in range(length))

    def compute_running_sum_local(self, digit_str, modulus=10):
        """Compute output[i] = (input[i] + input[i-1]) mod modulus (first output = first input)"""
        output = [digit_str[0]]
        for i in range(1, len(digit_str)):
            sum_result = (int(digit_str[i]) + int(digit_str[i-1])) % modulus
            output.append(str(sum_result))
        return "".join(output)

    def _insert_boundaries_in_string(self, string_str, interval):
        """Insert boundaries within a string every k positions."""
        if interval <= 0:
            return string_str
        bounded = []
        for i, c in enumerate(string_str):
            bounded.append(c)
            if (i + 1) % interval == 0 and i != len(string_str) - 1:
                bounded.append("|")
        return "".join(bounded)

    def build_prompt(self, length, interval, rng=None):
        """
        Build running sum (local window) prompt with structured boundaries.

        Example with N=8, k=4:
            Input: 2314|5298
            Output: 2545|9717
        """
        rng = rng or random
        input_str_base = self.sample_string(length, rng)
        output_str_base = self.compute_running_sum_local(input_str_base, modulus=10)

        input_bounded = self._insert_boundaries_in_string(input_str_base, interval)
        output_bounded = self._insert_boundaries_in_string(output_str_base, interval)

        prompt = f"""You are a deterministic sequence transformer.
Task: Running sum with previous input.
Alphabet: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (decimal digits)
Modulus: 10
Rule: At each position, output the sum of the current input digit and the previous input digit, modulo 10.
The first digit is copied as-is (no previous input to add).
Each subsequent digit: output[i] = (input[i] + input[i-1]) mod 10
Example: Input 2314 → Output 2545 (calculation: 2, (3+2)%10=5, (1+3)%10=4, (4+1)%10=5)
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

        return input_bounded, output_bounded, prompt


RUNNING_SUM_LOCAL_SEQUENCE_LENGTHS = [4, 8, 16, 32, 64]
RUNNING_SUM_LOCAL_PERIODIC_INTERVALS = [0, 1, 2, 3, 4, 8, 16]
RUNNING_SUM_LOCAL_TRIALS = 20

engine_sum_local = RunningSumLocalTaskEngine()


def run_running_sum_local_trial(task_params):
    N, k, t = task_params
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 80_000 * N + 100 * k + t
    rng = random.Random(local_seed)

    inp, truth, prompt = engine_sum_local.build_prompt(N, interval=k, rng=rng)

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


running_sum_local_tasks = [
    (N, k, t)
    for N in RUNNING_SUM_LOCAL_SEQUENCE_LENGTHS
    for k in RUNNING_SUM_LOCAL_PERIODIC_INTERVALS
    for t in range(RUNNING_SUM_LOCAL_TRIALS)
    if k == 0 or k < 2 * N + 1
]
print(f"\nStarting RUNNING SUM (LOCAL) grid: {len(running_sum_local_tasks)} evaluations...")

running_sum_local_results = []
for task in tqdm(running_sum_local_tasks, desc="Running Sum Local (structured boundaries)"):
    try:
        running_sum_local_results.append(run_running_sum_local_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

running_sum_local_df = pd.DataFrame(running_sum_local_results)
running_sum_local_csv = os.path.join(RESULTS_DIR, "running_sum_local_grid_structured_boundaries.csv")
running_sum_local_df.to_csv(running_sum_local_csv, index=False)
print(f"\nRunning Sum Local complete. Results saved to {running_sum_local_csv}")

# Summary stats
summary = running_sum_local_df.groupby(["Seq_Length", "Interval"])["TAR"].agg(["mean", "std"]).reset_index()
print("\n--- MEAN TAR BY (Seq_Length, Interval), RUNNING SUM LOCAL ---")
print(summary.to_string(index=False))
