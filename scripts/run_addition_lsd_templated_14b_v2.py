import matplotlib
matplotlib.use("Agg")

# ============================================================
# ADDITION (LSD-FIRST / CARRY-DIRECTION), RE-RUN WITH THE PROPER CHAT TEMPLATE
# ============================================================
# Same rationale as run_core_templated.py: rerun the least-significant-
# digit-first addition grid (same lengths, intervals, trial count, seeds
# as addition_lsd_grid.csv) with ONLY the prompting method changed. Also
# regenerates the carry-direction relative-retention comparison against
# the templated direct-order addition grid, if that file already exists
# (run run_addition_templated*.py first for a complete comparison).

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
ADD_LSD_PERIODIC_INTERVALS = [0, 1, 2, 3, 4, 8, 16]
ADD_LSD_TRIALS = 20

engine_add_lsd = AdditionTaskEngineLSD()

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
print(f"\nStarting TEMPLATED LSD-first addition grid: {len(addition_lsd_tasks)} evaluations...")

addition_lsd_results = []
for task in tqdm(addition_lsd_tasks, desc="Addition LSD (templated)"):
    try:
        addition_lsd_results.append(run_addition_lsd_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

addition_lsd_df = pd.DataFrame(addition_lsd_results)
addition_lsd_csv = os.path.join(RESULTS_DIR, "addition_lsd_grid_templated_v2.csv")
addition_lsd_df.to_csv(addition_lsd_csv, index=False)
print(f"\nLSD-first addition grid search complete. Saved to '{addition_lsd_csv}'")

summary = addition_lsd_df.groupby(["Seq_Length", "Method"])[["TAR", "SAR"]].mean().reset_index()
print("\n--- AVERAGE ACCURACY BY LENGTH & METHOD (LSD-first, templated) ---")
print(summary.to_string(index=False))

pivot = summary.pivot(index="Seq_Length", columns="Method", values="TAR")
if "Base" in pivot.columns and "Cellular_Bounds" in pivot.columns:
    pivot["Cellular_minus_Base"] = pivot["Cellular_Bounds"] - pivot["Base"]
    print("\n--- DOES CELLULAR STILL SCORE WORSE THAN BASE, ONCE DIGIT ORDER MATCHES CARRY DIRECTION? (templated) ---")
    print(pivot.round(4).to_string())
    n_worse = int((pivot["Cellular_minus_Base"] < 0).sum())
    print(f"\nCellular still worse than Base at {n_worse}/{len(pivot)} tested lengths.")

# --- Carry-direction relative-retention comparison, against the templated
# direct-order addition grid, IF it has already been generated ---
normal_path = os.path.join(RESULTS_DIR, "addition_grid_templated_v2.csv")
if os.path.exists(normal_path):
    normal_df = pd.read_csv(normal_path)
    normal_summary = normal_df.groupby(["Seq_Length", "Method"])["TAR"].mean().unstack()
    normal_summary["Relative_Retention"] = normal_summary["Cellular_Bounds"] / normal_summary["Base"]

    lsd_summary = summary.pivot(index="Seq_Length", columns="Method", values="TAR")
    lsd_summary["Relative_Retention"] = lsd_summary["Cellular_Bounds"] / lsd_summary["Base"]

    comparison = pd.DataFrame({
        "Normal_Order_Retention": normal_summary["Relative_Retention"],
        "LSD_First_Retention": lsd_summary["Relative_Retention"],
    })
    comparison["Improved_With_LSD"] = comparison["LSD_First_Retention"] > comparison["Normal_Order_Retention"]
    print("\n--- RELATIVE RETENTION (Cellular/Base TAR ratio): normal order vs LSD-first (templated) ---")
    print(comparison.round(3).to_string())
    comparison.to_csv(os.path.join(RESULTS_DIR, "addition_carry_direction_comparison_templated_v2.csv"))
else:
    print(f"\nSkipping carry-direction comparison: '{normal_path}' not found yet "
          f"(run the templated direct-order addition script first).")

print("\nTemplated LSD-first addition grid rerun complete.")
