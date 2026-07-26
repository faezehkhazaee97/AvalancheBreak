import matplotlib
matplotlib.use("Agg")

# ============================================================
# ADDITION LSD (LEAST-SIGNIFICANT-DIGIT-FIRST) WITH STRUCTURED BOUNDARIES
# ============================================================
# Same fix as run_addition_structured_boundaries.py, but for the LSD variant.
# LSD = least-significant-digit-first ordering, where carries flow left-to-right
# (instead of right-to-left as in normal order).
#
# Boundaries are placed within each operand separately, keeping "+" clear.

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


class AdditionTaskEngineLSDStructured:
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
        Build LSD addition prompt with structured boundaries.

        Example with N=4, k=2, numbers 1234 and 5678:
            Normal order: 1234 + 5678 = 7012
            LSD order:    4321 + 8765 = 2107 (reversed)

            Input (with structured boundaries): 43|21+87|65
            Output: 21|07
        """
        rng = rng or random
        a, b, total = self._sample_pair(length, rng)

        # Reverse for LSD ordering
        a_str = "".join(map(str, a))[::-1]  # Reverse to LSD
        b_str = "".join(map(str, b))[::-1]  # Reverse to LSD
        sum_str = "".join(map(str, total))[::-1]  # Reverse to LSD

        # Insert boundaries within each operand separately
        a_bounded = self._insert_boundaries_in_number(a_str, interval)
        b_bounded = self._insert_boundaries_in_number(b_str, interval)

        # Combine without extra spaces — only boundary placement differs
        input_str = f"{a_bounded}+{b_bounded}"

        prompt = f"""You are a deterministic arithmetic calculator.
Task: Add two {length}-digit numbers. Both input numbers and the expected output are written LEAST-SIGNIFICANT-DIGIT FIRST (reversed order, i.e. the ones digit comes first).
Constraint: The input may contain vertical bars (|) as structural boundaries -- they carry no numeric meaning, ignore them when computing.
Output ONLY the {length}-digit sum, also least-significant-digit-first. No spaces, no explanations.

Input: {input_str}
Output: """
        return input_str, sum_str, prompt


ADD_LSD_SEQUENCE_LENGTHS = [4, 8, 16, 32, 64]
ADD_LSD_PERIODIC_INTERVALS = [0, 1, 2, 3, 4, 8, 16]
ADD_LSD_TRIALS = 20

engine_add_lsd = AdditionTaskEngineLSDStructured()


def run_addition_lsd_trial(task_params):
    N, k, t = task_params
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 130_000 * N + 100 * k + t
    rng = random.Random(local_seed)

    inp, truth, prompt = engine_add_lsd.build_prompt(N, interval=k, rng=rng)

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


addition_lsd_tasks = [
    (N, k, t)
    for N in ADD_LSD_SEQUENCE_LENGTHS
    for k in ADD_LSD_PERIODIC_INTERVALS
    for t in range(ADD_LSD_TRIALS)
    if k == 0 or k < 2 * N + 1
]
print(f"\nStarting LSD STRUCTURED-BOUNDARY integer-addition grid: {len(addition_lsd_tasks)} evaluations...")

addition_lsd_results = []
for task in tqdm(addition_lsd_tasks, desc="Addition LSD (structured boundaries)"):
    try:
        addition_lsd_results.append(run_addition_lsd_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

addition_lsd_df = pd.DataFrame(addition_lsd_results)
addition_lsd_csv = os.path.join(RESULTS_DIR, "addition_lsd_grid_structured_boundaries.csv")
addition_lsd_df.to_csv(addition_lsd_csv, index=False)
print(f"\nAddition LSD (structured) complete. Results saved to {addition_lsd_csv}")

# Summary stats
summary = addition_lsd_df.groupby(["Seq_Length", "Interval"])["TAR"].agg(["mean", "std"]).reset_index()
print("\n--- MEAN TAR BY (Seq_Length, Interval), LSD STRUCTURED ---")
print(summary.to_string(index=False))
