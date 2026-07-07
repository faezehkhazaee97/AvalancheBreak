import matplotlib
matplotlib.use("Agg")

# ============================================================
# PAULI STRING MULTIPLICATION, RE-RUN WITH THE PROPER CHAT TEMPLATE
# ============================================================
# Same rationale as run_core_templated.py: rerun the Pauli-string grid
# (same lengths, intervals, trial count, seeds as pauli_string_grid.csv)
# with ONLY the prompting method changed.

import os

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
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
from tqdm import tqdm


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
print(f"\nStarting TEMPLATED Pauli-string grid: {len(pauli_tasks)} evaluations...")

pauli_results = []
for task in tqdm(pauli_tasks, desc="Pauli string (templated)"):
    try:
        pauli_results.append(run_pauli_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

pauli_df = pd.DataFrame(pauli_results)
pauli_csv = os.path.join(RESULTS_DIR, "pauli_string_grid_templated.csv")
pauli_df.to_csv(pauli_csv, index=False)
print(f"\nPauli-string grid search complete. Saved to '{pauli_csv}'")

summary = pauli_df.groupby(["Seq_Length", "Method"])[["TAR", "SAR", "String_Correct", "Phase_Correct"]].mean().reset_index()
print("\n--- AVERAGE ACCURACY BY LENGTH & METHOD (templated) ---")
print(summary.to_string(index=False))
print("\n--- DOES THE LOCAL (STRING) PART BEHAVE DIFFERENTLY FROM THE GLOBAL (PHASE) PART? ---")
print("If String_Correct tracks cyclic-shift's pattern (Cellular helps) while Phase_Correct")
print("tracks modsum/addition's pattern (Cellular hurts or no effect), that's direct evidence")
print("the boundary effect depends on locality structure WITHIN a single task, not just across tasks.")

print("\nTemplated Pauli-string grid rerun complete.")
