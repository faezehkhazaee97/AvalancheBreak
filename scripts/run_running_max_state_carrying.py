import matplotlib
matplotlib.use("Agg")

# ============================================================
# RUNNING_MAX REDESIGN: STATE-CARRYING BOUNDARIES
# ============================================================
# Same design as scripts/run_modsum_state_carrying.py, applied to
# running_max: instead of a content-free "|" boundary (which the model is
# told to disregard, even though the task needs the running max carried
# across exactly that position), each boundary is replaced by the TRUE
# running maximum at that point, written as "[v]" -- a genuine checkpoint
# the model can read and use directly, rather than a divider that
# discards the one thing the task needs remembered.
#
# Reviewer ask (Q6): quantify how much explicit state offsets dependency
# disruption on running_max specifically -- modsum and addition already
# have a state-carrying data point; running_max (the easier, high-Base-
# competence Tier-3 anchor) is the remaining piece.
#
# Only the new Cellular_StateCarrying condition is run fresh here -- Base
# and the original content-free Cellular_Bounds (interval=4) both already
# exist in results_clean/running_max/grid/Qwen2.5-7B-Instruct_POOLED_
# 40trials.csv at the identical seeds (tasks.common.runner's local-seed
# formula), so this is a direct three-way paired comparison for the first
# 20 trials, not a full rerun.

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join(REPO_ROOT, "results", MODEL_TAG, "running_max_state_carrying")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random
import numpy as np
import pandas as pd
from tqdm import tqdm

from tasks.common import scoring as scoring_mod

print(f"Loading {MODEL_ID} into VRAM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
)
model.eval()

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)


def local_seed(N, k, axis_val, t):
    return 90_000 * N + 100 * k + 10 * axis_val + t  # matches tasks.common.runner exactly


def build_state_carrying_prompt(N, k, rng, axis_val):
    alphabet_size = axis_val
    raw = [str(rng.randint(0, alphabet_size - 1)) for _ in range(N)]

    running_max = -1
    running_digits = []
    for ch in raw:
        running_max = max(running_max, int(ch))
        running_digits.append(str(running_max))

    input_w_bounds, output_w_bounds = "", ""
    for i, (d, out_d) in enumerate(zip(raw, running_digits)):
        input_w_bounds += d
        output_w_bounds += out_d
        if k > 0 and (i + 1) % k == 0 and i != N - 1:
            checkpoint = f"[{out_d}]"
            input_w_bounds += checkpoint
            output_w_bounds += checkpoint

    prompt = f"""You are a deterministic sequence transformer.
Task: Running maximum.
Alphabet: 0-{alphabet_size - 1}
Rule: At each position, output the LARGEST digit seen so far, including the current position.
Scan left to right; once a digit is output, it never decreases.
Constraint: The input contains bracketed checkpoints like [{running_digits[0]}]. Each checkpoint states the TRUE, correct running maximum at that exact point in the sequence. Use it directly as your new running maximum going forward -- you do not need to re-derive it from the digits before it. Copy each checkpoint exactly to the output, unchanged, at the same position.
No spaces, no explanations.

Input: {input_w_bounds}
Output: """
    return input_w_bounds, output_w_bounds, prompt


ALPHABET_SIZES = [2, 4, 8]
INTERVAL = 4
SEQUENCE_LENGTHS = [8, 16, 32, 64]
TRIALS_T = 20


def run_trial(N, axis_val, t):
    rng = random.Random(local_seed(N, INTERVAL, axis_val, t))
    inp, truth, prompt = build_state_carrying_prompt(N, INTERVAL, rng, axis_val)

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in enc.items()}

    max_tokens_needed = int(N * 2.5) + 20  # checkpoints add extra characters vs. content-free "|"
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_only = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(gen_only, skip_special_tokens=True)

    # checkpoints use "[" "]" as structural characters, matching modsum's
    # own convention -- strip brackets same as "|" is stripped elsewhere,
    # scoring only the digit content.
    def strip_checkpoints(s):
        import re
        return re.sub(r"[\[\]]", "", s)

    result = scoring_mod.score(strip_checkpoints(raw_output), strip_checkpoints(truth))

    return {
        "Method": "Cellular_StateCarrying", "Interval": INTERVAL, "Alphabet_Size": axis_val,
        "Seq_Length": N, "Trial": t, "TAR": result["TAR"], "SAR": result["SAR"],
        "Input_Given": inp, "Ground_Truth": result["clean_truth"], "LLM_Output": result["clean_pred"],
        "Raw_Model_Output": raw_output,
    }


tasks = [(N, a, t) for N in SEQUENCE_LENGTHS for a in ALPHABET_SIZES for t in range(TRIALS_T)]
print(f"\nStarting running_max state-carrying sweep: {len(tasks)} evaluations...")

results = []
for N, a, t in tqdm(tasks, desc="running_max state-carrying"):
    try:
        results.append(run_trial(N, a, t))
    except Exception as exc:
        print(f"Task failed: N={N} axis={a} t={t} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "running_max_state_carrying_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

summary = df.groupby(["Seq_Length", "Alphabet_Size"])[["TAR", "SAR"]].mean()
print("\n--- STATE-CARRYING running_max TAR/SAR ---")
print(summary.round(4).to_string())

# --- Three-way comparison against the existing paired Base / content-free reference ---
ref_path = os.path.join(REPO_ROOT, "results_clean", "running_max", "grid",
                         "Qwen2.5-7B-Instruct_POOLED_40trials.csv")
ref_df = pd.read_csv(ref_path)
ref_df = ref_df[ref_df.Trial < TRIALS_T]
ref_base = ref_df[ref_df.Method == "Base"].groupby(["Seq_Length", "Alphabet_Size"])["TAR"].mean()
ref_bounds = ref_df[(ref_df.Method == "Cellular_Bounds") & (ref_df.Interval == INTERVAL)].groupby(
    ["Seq_Length", "Alphabet_Size"])["TAR"].mean()
sc = df.groupby(["Seq_Length", "Alphabet_Size"])["TAR"].mean()

comparison = pd.DataFrame({
    "Base_TAR": ref_base, "ContentFree_Bounds_TAR": ref_bounds, "StateCarrying_TAR": sc,
})
comparison["StateCarrying_vs_ContentFree"] = comparison["StateCarrying_TAR"] - comparison["ContentFree_Bounds_TAR"]
comparison["StateCarrying_vs_Base"] = comparison["StateCarrying_TAR"] - comparison["Base_TAR"]
print("\n--- DOES A STATE-CARRYING CHECKPOINT OFFSET running_max's DEPENDENCY DISRUPTION? ---")
print(comparison.round(4).to_string())
comparison.to_csv(os.path.join(RESULTS_DIR, "running_max_state_carrying_comparison.csv"))

print("\nDone.")
