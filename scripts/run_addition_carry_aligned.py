import matplotlib
matplotlib.use("Agg")

# ============================================================
# CARRY-ALIGNED BOUNDARY PLACEMENT FOR INTEGER ADDITION
# ============================================================
# Reviewer question: the addition negative result (Section 3.6) places
# boundaries at a FIXED CHARACTER INTERVAL over the flat "A_digits +
# B_digits" string -- these boundaries slice through digit-column pairs
# arbitrarily and have no relationship to where carries actually
# propagate. This tests a boundary placement that IS aligned to the
# task's real computational structure: instead of the flat "A+B" string,
# the two operands are interleaved by column (each boundary groups one
# column's two digits together: "A0B0|A1B1|...|A(N-1)B(N-1)", written in
# the same most-significant-first digit order the model already sees),
# so each boundary-delimited cell is exactly one addition column, the
# unit carries pass through. If addition's negative result was really
# about arbitrary slicing rather than boundaries per se, this format
# should behave differently (better) than the existing fixed-interval
# ablation; if it still hurts vs Base, that's further evidence the
# problem is boundaries-interrupting-long-range-carry-state per se, not
# just clumsy placement.
#
# Only the new Carry_Aligned condition is run fresh here -- Base already
# exists in addition_grid_templated_v2.csv at the identical (a, b) pairs
# (same seed formula, k=0 term), so this is a direct paired comparison,
# not a full rerun.

import os

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")

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


class AdditionTaskEngine:
    # Identical sampling logic to run_addition_templated_v2.py, so that
    # using the SAME seed reproduces the IDENTICAL (a, b, sum) triple
    # already used for that script's Base condition.
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

    def build_carry_aligned_prompt(self, length, rng=None):
        rng = rng or random
        a, b, total = self._sample_pair(length, rng)
        sum_str = "".join(map(str, total))

        # Interleave by column: "A0B0|A1B1|...|A(N-1)B(N-1)", boundaries
        # between columns only (not inside a column pair).
        columns = [f"{a[i]}{b[i]}" for i in range(length)]
        input_str = "|".join(columns)

        prompt = f"""You are a deterministic arithmetic calculator.
Task: Add two {length}-digit numbers, given column by column.
Format: The input is written as digit PAIRS separated by vertical bars (|). Each pair "XY" means: X is a digit from the first number, Y is the digit from the second number, at that same column position (most-significant column first). The bars are structural boundaries between columns -- they carry no numeric meaning.
Example (3-digit numbers, illustrative only): "12|34|56" means first number = 135, second number = 246.
Compute the {length}-digit sum of the two numbers (add column by column, with carries propagating from the last column to the first, exactly as in ordinary addition). Output ONLY the {length}-digit sum (zero-padded to {length} digits if needed). No spaces, no explanations.

Input: {input_str}
Output: """
        return input_str, sum_str, prompt


ADD_SEQUENCE_LENGTHS = [4, 8, 16, 32, 64]
ADD_TRIALS = 20

engine_add = AdditionTaskEngine()


def run_addition_trial(task_params):
    N, t = task_params
    # Same seed formula as run_addition_templated_v2.py's k=0 (Base) condition,
    # so this reproduces the identical (a, b) pair for a fair paired comparison.
    local_seed = 90_000 * N + 100 * 0 + t
    rng = random.Random(local_seed)

    inp, truth, prompt = engine_add.build_carry_aligned_prompt(N, rng=rng)

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
        "Method": "Carry_Aligned", "Seq_Length": N, "Trial": t,
        "TAR": tar, "SAR": sar, "First_Error_Pos": first_error,
        "Input_Given": inp, "Ground_Truth": truth, "LLM_Output": clean_pred,
    }


tasks = [(N, t) for N in ADD_SEQUENCE_LENGTHS for t in range(ADD_TRIALS)]
print(f"\nStarting carry-aligned addition sweep: {len(tasks)} evaluations...")

results = []
for task in tqdm(tasks, desc="Addition (carry-aligned)"):
    try:
        results.append(run_addition_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "addition_carry_aligned_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

summary = df.groupby("Seq_Length")[["TAR", "SAR"]].mean()
print("\n--- CARRY-ALIGNED TAR/SAR by length ---")
print(summary.round(4).to_string())

# --- Compare against existing Base and fixed-interval Cellular_Bounds ---
existing = pd.read_csv(os.path.join(RESULTS_DIR, "addition_grid_templated_v2.csv"))
base_ref = existing[existing.Method == "Base"].groupby("Seq_Length")["TAR"].mean()
# Best (highest-TAR) fixed-interval Cellular_Bounds density at each length, for a fair
# "best case" comparison against the fixed-interval approach.
cellular_ref = existing[existing.Method == "Cellular_Bounds"].groupby(["Seq_Length", "Interval"])["TAR"].mean()
best_cellular = cellular_ref.groupby("Seq_Length").max()

comparison = pd.DataFrame({
    "Base_TAR": base_ref,
    "Best_Fixed_Interval_Cellular_TAR": best_cellular,
    "Carry_Aligned_TAR": summary["TAR"],
})
print("\n--- DOES CARRY-ALIGNED PLACEMENT RESCUE THE ADDITION NEGATIVE RESULT? ---")
print(comparison.round(4).to_string())
comparison.to_csv(os.path.join(RESULTS_DIR, "addition_carry_aligned_comparison.csv"))

print("\nCarry-aligned addition sweep complete.")
