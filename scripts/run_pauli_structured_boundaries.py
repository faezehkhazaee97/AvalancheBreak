import matplotlib
matplotlib.use("Agg")

# ============================================================
# PAULI STRING MULTIPLICATION WITH STRUCTURED BOUNDARIES
# ============================================================
# Corrected version: Boundaries are placed intelligently to respect
# the structure of the problem:
# - Boundaries are inserted within each operand (first and second string)
# - The "*" operator is kept clear, with no boundaries immediately adjacent
# - Example: IXZ|YZI*XIZ|YXI (instead of broken: IXZ|YZI|*XIZ|YXI|X)
#
# This allows fair testing of whether boundaries help or hurt based on
# the position-local vs global-state hypothesis, not based on malformed prompts.

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


class PauliStringTaskEngineStructured:
    PAULIS = ["I", "X", "Y", "Z"]
    MULT_TABLE = {
        ("I", "I"): ("I", 0), ("I", "X"): ("X", 0), ("I", "Y"): ("Y", 0), ("I", "Z"): ("Z", 0),
        ("X", "I"): ("X", 0), ("Y", "I"): ("Y", 0), ("Z", "I"): ("Z", 0),
        ("X", "X"): ("I", 0), ("Y", "Y"): ("I", 0), ("Z", "Z"): ("I", 0),
        ("X", "Y"): ("Z", 1), ("Y", "Z"): ("X", 1), ("Z", "X"): ("Y", 1),
        ("Y", "X"): ("Z", 3), ("Z", "Y"): ("X", 3), ("X", "Z"): ("Y", 3),
    }
    PHASE_SYMBOLS = ["+", "i", "-", "j"]

    def sample_strings(self, length, rng):
        a = rng.choices(self.PAULIS, k=length)
        b = rng.choices(self.PAULIS, k=length)
        return a, b

    def multiply(self, a, b):
        result_ops = []
        total_phase = 0
        for pa, pb in zip(a, b):
            op, phase = self.MULT_TABLE[(pa, pb)]
            result_ops.append(op)
            total_phase = (total_phase + phase) % 4
        return "".join(result_ops), self.PHASE_SYMBOLS[total_phase]

    def _insert_boundaries_in_string(self, string_str, interval):
        """Insert boundaries within a single Pauli string every k positions."""
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
        Build Pauli multiplication prompt with structured boundaries.

        Example with N=4, k=2:
            Input: IXZ|YZI*XIZ|YXI
            Output: IXYZ+ (or similar)

        Key difference from naive approach:
        - Boundaries are inserted within each operand independently
        - The "*" operator is clearly separated, not buried in boundaries
        - This respects the mathematical structure of the problem
        """
        rng = rng or random
        a, b = self.sample_strings(length, rng)
        result_string, phase_symbol = self.multiply(a, b)
        truth = result_string + phase_symbol

        a_str = "".join(a)
        b_str = "".join(b)

        # Insert boundaries within each operand separately
        a_bounded = self._insert_boundaries_in_string(a_str, interval)
        b_bounded = self._insert_boundaries_in_string(b_str, interval)

        # Combine without extra spaces — only boundary placement differs
        input_str = f"{a_bounded}*{b_bounded}"

        prompt = f"""You are a deterministic Pauli algebra calculator.
Task: multiply two {length}-operator Pauli strings, position by position.
Rules: I is the identity (I*P = P for any P). P*P = I for P in {{X, Y, Z}}. X*Y=iZ, Y*Z=iX, Z*X=iY. Y*X=-iZ, Z*Y=-iX, X*Z=-iY.
Alphabet: I, X, Y, Z. The symbol * marks where the first string ends and the second begins.
Constraint: The input may contain vertical bars (|) as structural boundaries -- they carry no meaning, ignore them when computing.
Output the resulting {length}-operator string, immediately followed by ONE phase symbol for the TOTAL accumulated phase across all positions (multiply every per-position phase together): '+' for +1, 'i' for +i, '-' for -1, 'j' for -i. No spaces, no explanations.

Input: {input_str}
Output: """
        return input_str, truth, prompt


PAULI_SEQUENCE_LENGTHS = [4, 8, 16, 32]
PAULI_PERIODIC_INTERVALS = [0, 1, 2, 3, 4, 8, 16]
PAULI_TRIALS = 20

engine_pauli = PauliStringTaskEngineStructured()


def run_pauli_trial(task_params):
    N, k, t = task_params
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 170_000 * N + 100 * k + t
    rng = random.Random(local_seed)

    inp, truth, prompt = engine_pauli.build_prompt(N, interval=k, rng=rng)

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

    match = re.match(r'^([IXYZ+ij\-]+)', pred)
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


pauli_tasks = [
    (N, k, t)
    for N in PAULI_SEQUENCE_LENGTHS
    for k in PAULI_PERIODIC_INTERVALS
    for t in range(PAULI_TRIALS)
    if k == 0 or k < 2 * N + 1
]
print(f"\nStarting STRUCTURED-BOUNDARY Pauli-string grid: {len(pauli_tasks)} evaluations...")

pauli_results = []
for task in tqdm(pauli_tasks, desc="Pauli (structured boundaries)"):
    try:
        pauli_results.append(run_pauli_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

pauli_df = pd.DataFrame(pauli_results)
pauli_csv = os.path.join(RESULTS_DIR, "pauli_string_grid_structured_boundaries.csv")
pauli_df.to_csv(pauli_csv, index=False)
print(f"\nPauli (structured) complete. Results saved to {pauli_csv}")

# Summary stats
summary = pauli_df.groupby(["Seq_Length", "Interval"])["TAR"].agg(["mean", "std"]).reset_index()
print("\n--- MEAN TAR BY (Seq_Length, Interval), PAULI STRUCTURED ---")
print(summary.to_string(index=False))
