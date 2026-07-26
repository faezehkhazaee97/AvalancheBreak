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


class RunningXORTaskEngine:
    def sample_string(self, length, rng):
        return "".join(str(rng.choice([0, 1])) for _ in range(length))

    def compute_xor_chain(self, binary_str):
        """Compute output[i] = input[i] XOR input[i-1] (first output = first input)"""
        output = [binary_str[0]]
        for i in range(1, len(binary_str)):
            xor_result = int(binary_str[i]) ^ int(binary_str[i-1])
            output.append(str(xor_result))
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
        Build running XOR prompt with structured boundaries.

        Example with N=8, k=4:
            Input: 1011|0110
            Output: 1110|1101
        """
        rng = rng or random
        input_str_base = self.sample_string(length, rng)
        output_str_base = self.compute_xor_chain(input_str_base)

        input_bounded = self._insert_boundaries_in_string(input_str_base, interval)
        output_bounded = self._insert_boundaries_in_string(output_str_base, interval)

        prompt = f"""You are a deterministic sequence transformer.
Task: Running XOR with previous input.
Alphabet: 0, 1 (binary digits)
Rule: At each position, output the XOR of the current input digit with the previous input digit.
The first digit is copied as-is (no previous input to XOR with).
Each subsequent digit: output[i] = input[i] XOR input[i-1]
Example: Input 1011 → Output 1110 (calculation: 1, 0⊕1=1, 1⊕0=1, 1⊕1=0)
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

        return input_bounded, output_bounded, prompt


RUNNING_XOR_SEQUENCE_LENGTHS = [4, 8, 16, 32, 64]
RUNNING_XOR_PERIODIC_INTERVALS = [0, 1, 2, 3, 4, 8, 16]
RUNNING_XOR_TRIALS = 20

engine_xor = RunningXORTaskEngine()


def run_running_xor_trial(task_params):
    N, k, t = task_params
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 70_000 * N + 100 * k + t
    rng = random.Random(local_seed)

    inp, truth, prompt = engine_xor.build_prompt(N, interval=k, rng=rng)

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

    match = re.match(r'^([01|]+)', pred)
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


running_xor_tasks = [
    (N, k, t)
    for N in RUNNING_XOR_SEQUENCE_LENGTHS
    for k in RUNNING_XOR_PERIODIC_INTERVALS
    for t in range(RUNNING_XOR_TRIALS)
    if k == 0 or k < 2 * N + 1
]
print(f"\nStarting RUNNING XOR grid: {len(running_xor_tasks)} evaluations...")

running_xor_results = []
for task in tqdm(running_xor_tasks, desc="Running XOR (structured boundaries)"):
    try:
        running_xor_results.append(run_running_xor_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

running_xor_df = pd.DataFrame(running_xor_results)
running_xor_csv = os.path.join(RESULTS_DIR, "running_xor_grid_structured_boundaries.csv")
running_xor_df.to_csv(running_xor_csv, index=False)
print(f"\nRunning XOR complete. Results saved to {running_xor_csv}")

# Summary stats
summary = running_xor_df.groupby(["Seq_Length", "Interval"])["TAR"].agg(["mean", "std"]).reset_index()
print("\n--- MEAN TAR BY (Seq_Length, Interval), RUNNING XOR ---")
print(summary.to_string(index=False))
