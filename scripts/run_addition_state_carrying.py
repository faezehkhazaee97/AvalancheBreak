import matplotlib
matplotlib.use("Agg")

# ============================================================
# ADDITION REDESIGN: STATE-CARRYING BOUNDARIES (LSD-ordered)
# ============================================================
# Reviewer question: "can you discuss whether ... state-carrying boundary
# variants narrow the gap [on addition] without retraining?"
#
# A literal port of the modsum state-carrying checkpoint ("[v]" = the true
# running value so far) is NOT well-posed for addition in its normal
# (most-significant-digit-first) presentation: addition's dependency is the
# CARRY BIT, which propagates right-to-left, opposite to both reading order
# and generation order. A checkpoint at position i in MSD order would have
# to state something that depends on digits the model hasn't been shown yet
# -- it can't be "the true carry so far" because "so far" (left-to-right)
# isn't the direction the carry actually flows.
#
# The correctly-posed version combines two things already separately
# established in this project: (1) presenting digits least-significant-
# digit-first, so the carry's true dependency direction now matches reading
# order (this alone was already tested as the LSD-reversal condition and
# did not rescue Cellular_Bounds -- see addition_lsd_grid_templated_v2.csv);
# (2) on top of LSD ordering, insert a genuine state-carrying checkpoint
# "[c]" at each boundary stating the TRUE carry-into-the-next-position
# (0 or 1) -- which, in LSD order, *is* fully determined by everything seen
# so far, exactly like modsum's running-sum checkpoint. Digits are
# presented as interleaved (a_i, b_i) pairs, LSD first, so a checkpoint
# after position i is well-defined (both digits contributing to that
# carry have been shown).
#
# Three-way comparison against the SAME underlying (a, b) pairs:
#   Base (LSD, no boundaries) -- already exists in addition_lsd_grid_templated_v2.csv
#   Cellular_ContentFree (LSD, "|" boundaries) -- already exists, same file
#   Cellular_StateCarrying (LSD, "[c]" carry checkpoints) -- new, this script

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


class AdditionStateCarryingEngine:
    def _sample_pair(self, length, rng):
        """Same sampling rule as the original AdditionTaskEngine (no
        leading zero, no overflow) so digit distributions match exactly."""
        for _ in range(200):
            a = [rng.randrange(10) for _ in range(length)]
            b = [rng.randrange(10) for _ in range(length)]
            if a[0] == 0: a[0] = rng.randrange(1, 10)
            if b[0] == 0: b[0] = rng.randrange(1, 10)
            carry = 0
            total_lsd = []  # least-significant first
            carries_lsd = []  # carry INTO the next (more significant) position
            for i in range(length - 1, -1, -1):
                s = a[i] + b[i] + carry
                total_lsd.append(s % 10)
                carry = s // 10
                carries_lsd.append(carry)
            if carry == 0:
                a_lsd = list(reversed(a))
                b_lsd = list(reversed(b))
                return a_lsd, b_lsd, total_lsd, carries_lsd
        raise RuntimeError(f"failed to sample a non-overflowing {length}-digit pair")

    def build_state_carrying(self, length, interval, rng=None):
        rng = rng or random
        a_lsd, b_lsd, total_lsd, carries_lsd = self._sample_pair(length, rng)

        input_parts, output_parts = [], []
        for i in range(length):
            input_parts.append(f"{a_lsd[i]}{b_lsd[i]}")
            output_parts.append(str(total_lsd[i]))
            is_last = (i == length - 1)
            if interval > 0 and (i + 1) % interval == 0 and not is_last:
                checkpoint = f"[{carries_lsd[i]}]"
                input_parts.append(checkpoint)
                output_parts.append(checkpoint)

        input_str = "".join(input_parts)
        output_str = "".join(output_parts)

        prompt = f"""You are a deterministic arithmetic calculator.
Task: Add two {length}-digit numbers, digit pairs given LEAST-SIGNIFICANT-DIGIT FIRST (position 0 = ones place).
Each pair "ab" means: this position's digit of the first number is a, of the second number is b. Add them plus any carry-in, write the ones digit of the result, carry the rest forward to the next (more significant) pair.
Constraint: The input contains bracketed checkpoints like [{carries_lsd[0]}]. Each checkpoint states the TRUE carry value (0 or 1) generated by every digit pair before it. Use it directly as your carry-in for the next pair -- you do not need to re-derive it from the pairs before it. Copy each checkpoint exactly to the output, unchanged, at the same position.
Output the result digits in the same least-significant-first order, one digit per input pair (plus the checkpoints, copied through). No spaces, no explanations.

Input: {input_str}
Output: """
        return input_str, output_str, prompt


ADD_SEQUENCE_LENGTHS = [8, 16, 32, 64]
INTERVAL = 4  # matches modsum's state-carrying interval for direct comparability
TRIALS_T = 20

engine = AdditionStateCarryingEngine()


def run_trial(N, t):
    # Same local-seed convention as run_addition_lsd_templated_v2.py, so
    # this pairs with the existing Base/Cellular_ContentFree LSD rows.
    local_seed = 90_000 * N + t
    rng = random.Random(local_seed)
    inp, truth, prompt = engine.build_state_carrying(N, INTERVAL, rng=rng)

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in enc.items()}

    max_tokens_needed = int(N * 2.0) + 20
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_only = outputs[0][inputs["input_ids"].shape[1]:]
    pred = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
    pred = pred.split("Explanation")[0].split("\n")[0].strip()

    clean_pred = re.sub(r"\[\d+\]", "", pred).replace(" ", "").replace("\n", "")
    clean_truth = re.sub(r"\[\d+\]", "", truth)
    clean_pred = clean_pred[:len(clean_truth)]

    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0
                      for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0

    return {
        "Method": "Cellular_StateCarrying", "Interval": INTERVAL, "Seq_Length": N, "Trial": t,
        "TAR": tar, "SAR": sar, "Input_Given": inp, "Ground_Truth": clean_truth, "LLM_Output": clean_pred,
    }


tasks = [(N, t) for N in ADD_SEQUENCE_LENGTHS for t in range(TRIALS_T)]
print(f"\nStarting state-carrying addition sweep: {len(tasks)} evaluations...")

results = []
for N, t in tqdm(tasks, desc="Addition (state-carrying)"):
    try:
        results.append(run_trial(N, t))
    except Exception as exc:
        print(f"Task failed: N={N} t={t} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "addition_state_carrying_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

summary = df.groupby("Seq_Length")[["TAR", "SAR"]].mean()
print("\n--- STATE-CARRYING ADDITION TAR/SAR by length ---")
print(summary.round(4).to_string())

# --- Compare against existing LSD Base and content-free Cellular_Bounds ---
lsd_path = os.path.join(RESULTS_DIR, "addition_lsd_grid_templated_v2.csv")
if os.path.exists(lsd_path):
    existing = pd.read_csv(lsd_path)
    rows = []
    for N in ADD_SEQUENCE_LENGTHS:
        sub = existing[existing.Seq_Length == N]
        base_ref = sub[sub.Method == "Base"]["TAR"].mean()
        old_cell_ref = sub[(sub.Method == "Cellular_Bounds") & (sub.Interval == INTERVAL)]["TAR"].mean()
        new_cell = df[df.Seq_Length == N]["TAR"].mean()
        rows.append({
            "Seq_Length": N, "Base_LSD_TAR": base_ref,
            "Cellular_ContentFree_LSD_TAR": old_cell_ref,
            "Cellular_StateCarrying_LSD_TAR": new_cell,
        })
    comparison = pd.DataFrame(rows)
    print("\n--- DOES A CARRY-STATE CHECKPOINT NARROW THE ADDITION GAP? ---")
    print(comparison.round(4).to_string(index=False))
    comparison.to_csv(os.path.join(RESULTS_DIR, "addition_state_carrying_comparison.csv"), index=False)
else:
    print(f"\nNOTE: {lsd_path} not found, skipping three-way comparison (state-carrying grid still saved above).")

print("\nState-carrying addition sweep complete.")
