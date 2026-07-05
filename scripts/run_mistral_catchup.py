import matplotlib
matplotlib.use("Agg")

# ============================================================
# MISTRAL CATCH-UP: LSD-first addition, scratchpad addition, Pauli strings
# ============================================================
# Mistral's own run (run_mistral_core.py) already produced addition_grid.csv,
# boundary_ablation_grid.csv, prompt_robustness_grid.csv, thermodynamics/
# modsum significance+holdout+token-overhead, etc. -- everything through the
# notebook's Prompt-Wording-Robustness section. The three analyses added to
# the notebook after that point (LSD-first addition, addition-with-a-
# scratchpad, Pauli string multiplication) were never run for this model.
# Ablation significance and clustering-vs-density were backfilled separately
# by run_postprocess_catchup.py (pure CPU, no new generation needed).

import os

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
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

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
import numpy as np
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)


# ==== ADDITION, CARRY-DIRECTION HYPOTHESIS TEST (LSD-first) ====

import pandas as pd
import itertools
import re
from tqdm import tqdm

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
    print(f"\nCellular still worse than Base at {n_worse}/{len(pivot)} tested lengths.")

normal_df = pd.read_csv(os.path.join(RESULTS_DIR, "addition_grid.csv"))
normal_summary = normal_df.groupby(["Seq_Length", "Method"])["TAR"].mean().unstack()
normal_summary["Relative_Retention"] = normal_summary["Cellular_Bounds"] / normal_summary["Base"]

lsd_summary = summary.pivot(index="Seq_Length", columns="Method", values="TAR")
lsd_summary["Relative_Retention"] = lsd_summary["Cellular_Bounds"] / lsd_summary["Base"]

comparison = pd.DataFrame({
    "Normal_Order_Retention": normal_summary["Relative_Retention"],
    "LSD_First_Retention": lsd_summary["Relative_Retention"],
})
comparison["Improved_With_LSD"] = comparison["LSD_First_Retention"] > comparison["Normal_Order_Retention"]
print("\n--- RELATIVE RETENTION (Cellular/Base TAR ratio): normal order vs LSD-first ---")
print(comparison.round(3).to_string())
n_improved = int(comparison["Improved_With_LSD"].sum())
print(f"\nLSD-first retains a HIGHER fraction of Base accuracy at {n_improved}/{len(comparison)} tested lengths.")
comparison.to_csv(os.path.join(RESULTS_DIR, "addition_carry_direction_comparison.csv"))


# ==== ADDITION WITH A SCRATCHPAD ====

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
    max_tokens_needed = N * 20 + 60
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred_full = full_response[len(prompt):]

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
    print(f"\nCellular still worse than Base at {n_worse}/{len(pivot)} tested lengths.")


# ==== PAULI STRING MULTIPLICATION ====

class PauliStringTaskEngine:
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

    def build_prompt(self, length, interval, rng=None):
        rng = rng or random
        a, b = self.sample_strings(length, rng)
        result_string, phase_symbol = self.multiply(a, b)
        truth = result_string + phase_symbol

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

print("\nMistral catch-up complete.")
