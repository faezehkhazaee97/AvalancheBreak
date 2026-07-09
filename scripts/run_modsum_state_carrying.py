import matplotlib
matplotlib.use("Agg")

# ============================================================
# MODSUM REDESIGN: STATE-CARRYING BOUNDARIES
# ============================================================
# Reviewer's sharpest suggestion: a content-free "|" boundary is
# actively the wrong tool for running-sum-mod-M, because the model is
# told to ignore it, but the whole task requires carrying state across
# exactly the position the marker sits at. The literature's answer
# (explicit scratchpads, segment-recurrence memory tokens, checkpoint
# reinjection) is consistently to make the boundary carry the actual
# state instead of standing in for its absence. This tests that
# directly: instead of "|", each boundary is replaced by the TRUE
# running-sum value at that exact point, written as "[v]" -- a genuine
# checkpoint the model can read and use as its new running base, rather
# than a divider it's told to disregard.
#
# Only the new Cellular_StateCarrying condition is run fresh here --
# Base and the original content-free Cellular_Bounds (interval=4) both
# already exist in running_modsum_grid_templated_v2.csv at the identical
# seeds, so this is a direct three-way paired comparison, not a full
# rerun.

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


class ModSumTaskEngine:
    def __init__(self, modulus):
        self.modulus = modulus
        self.digits = [str(i) for i in range(modulus)]

    def generate_raw_input(self, length, rng=None):
        if rng is None:
            rng = random
        return rng.choices(self.digits, k=length)

    def _running_sum_digits(self, raw_input):
        running = 0
        out = []
        for d in raw_input:
            running = (running + int(d)) % self.modulus
            out.append(str(running))
        return out

    def build_state_carrying_from_raw(self, raw_input, interval):
        running_digits = self._running_sum_digits(raw_input)
        input_w_bounds, output_w_bounds = "", ""
        for i, (d, out_d) in enumerate(zip(raw_input, running_digits)):
            input_w_bounds += d
            output_w_bounds += out_d
            if interval > 0 and (i + 1) % interval == 0 and i != len(raw_input) - 1:
                checkpoint = f"[{out_d}]"
                input_w_bounds += checkpoint
                output_w_bounds += checkpoint
        prompt = f"""You are a deterministic sequence transformer.
Task: Running sum modulo {self.modulus}.
Digits: {', '.join(self.digits)}
Rule: At each position, output the cumulative sum of ALL digits seen so far since the very start of the sequence, taken modulo {self.modulus}.
Constraint: The input contains bracketed checkpoints like [{running_digits[0]}]. Each checkpoint states the TRUE, correct running sum at that exact point in the sequence. Use it directly as your new running total going forward -- you do not need to re-derive it from the digits before it. Copy each checkpoint exactly to the output, unchanged, at the same position.
No spaces, no explanations.

Input: {input_w_bounds}
Output: """
        return input_w_bounds, output_w_bounds, prompt


MODULI = [2, 4]
INTERVAL = 4
SEQUENCE_LENGTHS = [8, 16, 32, 64]
TRIALS_T = 20

engines = {M: ModSumTaskEngine(M) for M in MODULI}

# Same seed formula as run_modsum_templated_v2.py, so this pairs directly
# with the existing Base and content-free Cellular_Bounds rows.
shared_sequences = {}
for M in MODULI:
    for N in SEQUENCE_LENGTHS:
        for t in range(TRIALS_T):
            local_seed = 20_000 * M + 100 * N + t
            rng = random.Random(local_seed)
            shared_sequences[(M, N, t)] = engines[M].generate_raw_input(N, rng=rng)


def run_trial(task_params):
    M, N, t = task_params
    engine = engines[M]
    raw_input = shared_sequences[(M, N, t)]
    inp, truth, prompt = engine.build_state_carrying_from_raw(raw_input, interval=INTERVAL)

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in enc.items()}

    max_tokens_needed = int(N * 2.0) + 20  # checkpoints add extra output tokens
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
        "Method": "Cellular_StateCarrying", "Modulus": M, "Interval": INTERVAL,
        "Seq_Length": N, "Trial": t, "TAR": tar, "SAR": sar,
        "Input_Given": inp, "Ground_Truth": clean_truth, "LLM_Output": clean_pred,
    }


tasks = [(M, N, t) for M in MODULI for N in SEQUENCE_LENGTHS for t in range(TRIALS_T)]
print(f"\nStarting state-carrying modsum sweep: {len(tasks)} evaluations...")

results = []
for task in tqdm(tasks, desc="Modsum (state-carrying)"):
    try:
        results.append(run_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "modsum_state_carrying_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

summary = df.groupby(["Modulus", "Seq_Length"])[["TAR", "SAR"]].mean()
print("\n--- STATE-CARRYING MODSUM TAR/SAR by (Modulus, length) ---")
print(summary.round(4).to_string())

# --- Compare against existing Base and content-free Cellular_Bounds ---
existing = pd.read_csv(os.path.join(RESULTS_DIR, "running_modsum_grid_templated_v2.csv"))
rows = []
for M in MODULI:
    sub = existing[(existing.Modulus == M) & (existing.Seq_Length.isin(SEQUENCE_LENGTHS))]
    base_ref = sub[sub.Method == "Base"].groupby("Seq_Length")["TAR"].mean()
    old_cell_ref = sub[(sub.Method == "Cellular_Bounds") & (sub.Interval == INTERVAL)].groupby("Seq_Length")["TAR"].mean()
    new_cell = df[df.Modulus == M].groupby("Seq_Length")["TAR"].mean()
    for N in SEQUENCE_LENGTHS:
        rows.append({
            "Modulus": M, "Seq_Length": N,
            "Base_TAR": base_ref.get(N, float("nan")),
            "Cellular_ContentFree_TAR": old_cell_ref.get(N, float("nan")),
            "Cellular_StateCarrying_TAR": new_cell.get(N, float("nan")),
        })
comparison = pd.DataFrame(rows)
print("\n--- DOES A STATE-CARRYING BOUNDARY HELP WHERE A CONTENT-FREE ONE DOESN'T? ---")
print(comparison.round(4).to_string(index=False))
comparison.to_csv(os.path.join(RESULTS_DIR, "modsum_state_carrying_comparison.csv"), index=False)

print("\nState-carrying modsum sweep complete.")
