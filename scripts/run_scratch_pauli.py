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


# ==== SCRATCHPAD ADDITION ====

import pandas as pd
import numpy as np
import random
import itertools
import torch
import os
import re
from tqdm import tqdm

# ============================================================
# ADDITION WITH A SCRATCHPAD (isolate the failure mode)
# ============================================================
# The direct-output addition task forces the model to produce digits with
# no visible intermediate reasoning -- "blind" token-by-token carry
# tracking. If THAT is what boundaries disrupt (not addition in general),
# letting the model reason step-by-step before committing to a final
# answer should behave differently under boundaries than the direct
# version did. Same digit sampling and boundary placement as the direct
# addition task; only the prompt (allow reasoning) and extraction (look
# for a "Final Answer:" line) differ.
# Reuses the `model` / `tokenizer` already loaded above.

class ScratchpadAdditionTaskEngine:
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
        a_str = "".join(map(str, a))
        b_str = "".join(map(str, b))
        sum_str = "".join(map(str, total))

        combined = list(a_str) + ["+"] + list(b_str)
        bounded = []
        for i, d in enumerate(combined):
            bounded.append(d)
            if interval > 0 and (i + 1) % interval == 0 and i != len(combined) - 1:
                bounded.append("|")
        input_str = "".join(bounded)

        prompt = f"""You are a careful arithmetic assistant.
Task: Add two {length}-digit numbers.
Constraint: The input may contain vertical bars (|) as structural boundaries -- they carry no numeric meaning, ignore them when computing.
First, work through the addition step by step from the rightmost (least-significant) digit to the left, tracking the carry at each step. Then, on its own line, write exactly:
Final Answer: <the {length}-digit sum, zero-padded if needed>

Input: {input_str}
"""
        return input_str, sum_str, prompt


SCRATCH_SEQUENCE_LENGTHS = [4, 8, 16]
SCRATCH_PERIODIC_INTERVALS = [0, 4, 8, 16]
SCRATCH_TRIALS = 15

engine_scratch = ScratchpadAdditionTaskEngine()

def run_scratchpad_trial(task_params):
    N, k, t = task_params
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 150_000 * N + 100 * k + t
    rng = random.Random(local_seed)

    inp, truth, prompt = engine_scratch.build_prompt(N, interval=k, rng=rng)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    max_tokens_needed = N * 20 + 60  # generous budget for a full carry-by-carry trace
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred_full = full_response[len(prompt):]

    # Score honestly: only credit a clean "Final Answer: <digits>" line, not
    # digits found anywhere else in the reasoning trace.
    match = re.search(r'Final Answer:\s*(\d+)', pred_full)
    clean_pred = match.group(1)[:len(truth)] if match else ""

    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == truth[i] else 0 for i in range(len(truth))]
    tar = sum(binary_series) / len(truth) if truth else 0.0
    sar = 1.0 if (len(binary_series) == len(truth) and sum(binary_series) == len(binary_series)) else 0.0
    first_error = binary_series.index(0) if 0 in binary_series else len(binary_series)

    return {
        "Method": method, "Interval": k, "Seq_Length": N, "Trial": t,
        "TAR": tar, "SAR": sar, "First_Error_Pos": first_error,
        "Input_Given": inp, "Ground_Truth": truth, "LLM_Output": clean_pred,
        "Found_Final_Answer_Line": match is not None,
    }

scratch_tasks = [
    (N, k, t)
    for N in SCRATCH_SEQUENCE_LENGTHS
    for k in SCRATCH_PERIODIC_INTERVALS
    for t in range(SCRATCH_TRIALS)
    if k == 0 or k < 2 * N + 1
]
print(f"\nStarting scratchpad addition grid search: {len(scratch_tasks)} evaluations...")

scratch_results = []
for task in tqdm(scratch_tasks, desc="Scratchpad addition"):
    try:
        scratch_results.append(run_scratchpad_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

scratch_df = pd.DataFrame(scratch_results)
scratch_csv = os.path.join(RESULTS_DIR, "addition_scratchpad_grid.csv")
scratch_df.to_csv(scratch_csv, index=False)
print(f"\nScratchpad addition grid search complete. Saved to '{scratch_csv}'")

summary = scratch_df.groupby(["Seq_Length", "Method"])[["TAR", "SAR"]].mean().reset_index()
print("\n--- AVERAGE ACCURACY BY LENGTH & METHOD (scratchpad) ---")
print(summary.to_string(index=False))
print(f"\nFound a clean 'Final Answer:' line in {scratch_df['Found_Final_Answer_Line'].mean()*100:.1f}% of responses.")

pivot = summary.pivot(index="Seq_Length", columns="Method", values="TAR")
if "Base" in pivot.columns and "Cellular_Bounds" in pivot.columns:
    pivot["Cellular_minus_Base"] = pivot["Cellular_Bounds"] - pivot["Base"]
    print("\n--- DOES CELLULAR STILL SCORE WORSE THAN BASE, WITH A SCRATCHPAD? ---")
    print(pivot.round(4).to_string())
    n_worse = int((pivot["Cellular_minus_Base"] < 0).sum())
    print(f"\nCellular still worse than Base at {n_worse}/{len(pivot)} tested lengths "
          f"(compare against the direct-output addition task's gap: dramatically worse at every length).")


# ==== PAULI STRING MULTIPLICATION ====

import pandas as pd
import numpy as np
import random
import itertools
import torch
import os
import re
from tqdm import tqdm

# ============================================================
# PAULI STRING MULTIPLICATION (paper 1's third benchmark)
# ============================================================
# Note: this is implemented from first-principles Pauli algebra (verified
# against real 2x2 matrix multiplication before writing this cell), not a
# literal reproduction of paper 1's exact prompt -- that wasn't available
# here. The physics: multiply two length-N Pauli strings position-by-
# position. Each single-qubit product gives a resulting operator AND a
# phase factor (a power of i); the per-position phases accumulate into
# ONE global phase across the whole string. This makes the task a hybrid
# of the two locality regimes already tested: the resulting STRING is
# position-local (each output operator depends only on its own input pair,
# like cyclic-shift), while the PHASE is genuinely global state (depends
# on every position, like the running sum). One task, both regimes at once.
# Reuses the `model` / `tokenizer` already loaded above.

class PauliStringTaskEngine:
    PAULIS = ["I", "X", "Y", "Z"]
    # (a, b) -> (result_operator, phase_power_of_i mod 4). Verified against
    # real Pauli matrix multiplication.
    MULT_TABLE = {
        ("I", "I"): ("I", 0), ("I", "X"): ("X", 0), ("I", "Y"): ("Y", 0), ("I", "Z"): ("Z", 0),
        ("X", "I"): ("X", 0), ("Y", "I"): ("Y", 0), ("Z", "I"): ("Z", 0),
        ("X", "X"): ("I", 0), ("Y", "Y"): ("I", 0), ("Z", "Z"): ("I", 0),
        ("X", "Y"): ("Z", 1), ("Y", "Z"): ("X", 1), ("Z", "X"): ("Y", 1),
        ("Y", "X"): ("Z", 3), ("Z", "Y"): ("X", 3), ("X", "Z"): ("Y", 3),
    }
    PHASE_SYMBOLS = ["+", "i", "-", "j"]  # i^0=+1, i^1=+i, i^2=-1, i^3=-i

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

    def build_prompt(self, length, interval, rng=None):
        rng = rng or random
        a, b = self.sample_strings(length, rng)
        result_string, phase_symbol = self.multiply(a, b)
        truth = result_string + phase_symbol  # N operator chars + 1 phase char

        a_str, b_str = "".join(a), "".join(b)
        combined = list(a_str) + ["*"] + list(b_str)
        bounded = []
        for i, d in enumerate(combined):
            bounded.append(d)
            if interval > 0 and (i + 1) % interval == 0 and i != len(combined) - 1:
                bounded.append("|")
        input_str = "".join(bounded)

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
PAULI_PERIODIC_INTERVALS = [0, 4, 8, 16]
PAULI_TRIALS = 20

engine_pauli = PauliStringTaskEngine()

def run_pauli_trial(task_params):
    N, k, t = task_params
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 170_000 * N + 100 * k + t
    rng = random.Random(local_seed)

    inp, truth, prompt = engine_pauli.build_prompt(N, interval=k, rng=rng)

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

    match = re.match(r'^([IXYZ+ij-]+)', pred)
    clean_pred = match.group(1)[:len(truth)] if match else ""

    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == truth[i] else 0 for i in range(len(truth))]
    tar = sum(binary_series) / len(truth) if truth else 0.0
    sar = 1.0 if (len(binary_series) == len(truth) and sum(binary_series) == len(binary_series)) else 0.0
    first_error = binary_series.index(0) if 0 in binary_series else len(binary_series)

    # Separately: did it get the (position-local) string right vs the
    # (global) phase right? Tests the "hybrid task" hypothesis directly.
    string_correct = clean_pred[:len(truth) - 1] == truth[:-1] if len(clean_pred) >= len(truth) - 1 else False
    phase_correct = len(clean_pred) == len(truth) and clean_pred[-1] == truth[-1]

    return {
        "Method": method, "Interval": k, "Seq_Length": N, "Trial": t,
        "TAR": tar, "SAR": sar, "First_Error_Pos": first_error,
        "String_Correct": string_correct, "Phase_Correct": phase_correct,
        "Input_Given": inp, "Ground_Truth": truth, "LLM_Output": clean_pred,
    }

pauli_tasks = [
    (N, k, t)
    for N in PAULI_SEQUENCE_LENGTHS
    for k in PAULI_PERIODIC_INTERVALS
    for t in range(PAULI_TRIALS)
    if k == 0 or k < 2 * N + 1
]
print(f"\nStarting Pauli-string grid search: {len(pauli_tasks)} evaluations...")

pauli_results = []
for task in tqdm(pauli_tasks, desc="Pauli string"):
    try:
        pauli_results.append(run_pauli_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

pauli_df = pd.DataFrame(pauli_results)
pauli_csv = os.path.join(RESULTS_DIR, "pauli_string_grid.csv")
pauli_df.to_csv(pauli_csv, index=False)
print(f"\nPauli-string grid search complete. Saved to '{pauli_csv}'")

summary = pauli_df.groupby(["Seq_Length", "Method"])[["TAR", "SAR", "String_Correct", "Phase_Correct"]].mean().reset_index()
print("\n--- AVERAGE ACCURACY BY LENGTH & METHOD ---")
print(summary.to_string(index=False))
print("\n--- DOES THE LOCAL (STRING) PART BEHAVE DIFFERENTLY FROM THE GLOBAL (PHASE) PART? ---")
print("If String_Correct tracks cyclic-shift's pattern (Cellular helps) while Phase_Correct")
print("tracks modsum/addition's pattern (Cellular hurts or no effect), that's direct evidence")
print("the boundary effect depends on locality structure WITHIN a single task, not just across tasks.")

