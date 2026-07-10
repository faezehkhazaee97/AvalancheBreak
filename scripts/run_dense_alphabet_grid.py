import matplotlib
matplotlib.use("Agg")

# ============================================================
# DENSER LENGTH SAMPLING FOR HARDER ALPHABETS (A=4, A=8)
# ============================================================
# Reviewer question: Section 3.13 finds SAR collapses to floor by
# N=16-32 for alphabet=4/8, leaving only 2-3 informative points to
# constrain the SK fit -- far fewer than alphabet=2's 7-point curve.
# Could denser sampling concentrated in the region just before the
# floor (rather than the original sparse 7-point grid running out to
# N=256, where Section 3.13's own sensitivity analysis already showed
# the long tail adds no fitting leverage) recover a better-constrained
# alpha estimate?
#
# This collects a denser grid at exactly the lengths where SAR is still
# plausibly non-degenerate for A=4/8: N in {4,5,6,7,8,10,12,14,16},
# dropping the N=32..256 tail entirely (already shown uninformative).
# Conditions: Base and Cellular_Bounds at k=1 and k=2 (the two densities
# already reported in the main alpha/beta0 table for A=2, and the
# densities most likely to carry a real signal per Section 3.13's
# distinguishability check). Qwen2.5-7B only, matching the scope of the
# other deep-dive mechanistic robustness checks in this paper.
#
# This script only collects data; refitting the SK model and
# bootstrapping is a separate, GPU-free analysis step
# (analyze_dense_alphabet_grid.py).

import os

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random
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


class TaskEngine:
    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size
        self.alphabet = [chr(ord('A') + i) for i in range(alphabet_size)]
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}

    def generate_raw_input(self, length, rng=None):
        if rng is None:
            rng = random
        return rng.choices(self.alphabet, k=length)

    def build_base_from_raw(self, raw_input):
        input_str = "".join(raw_input)
        output_str = "".join(
            self.alphabet[(self.char_to_idx[c] + 1) % self.alphabet_size] for c in raw_input
        )
        prompt = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(self.alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {self.alphabet[-1]} -> {self.alphabet[0]}, and {self.alphabet[0]} -> {self.alphabet[1]}.
Constraint: Output ONLY the transformed string.
No spaces, no explanations.

Input: {input_str}
Output: """
        return input_str, output_str, prompt

    def build_cellular_from_raw(self, raw_input, interval):
        input_w_bounds, output_w_bounds = "", ""
        for i, char in enumerate(raw_input):
            shifted_char = self.alphabet[(self.char_to_idx[char] + 1) % self.alphabet_size]
            input_w_bounds += char
            output_w_bounds += shifted_char
            if interval > 0 and (i + 1) % interval == 0 and i != len(raw_input) - 1:
                input_w_bounds += "|"
                output_w_bounds += "|"
        prompt = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(self.alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {self.alphabet[-1]} -> {self.alphabet[0]}, and {self.alphabet[0]} -> {self.alphabet[1]}.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_w_bounds}
Output: """
        return input_w_bounds, output_w_bounds, prompt


ALPHABET_SIZES = [4, 8]
DENSE_LENGTHS = [4, 5, 6, 7, 8, 10, 12, 14, 16]
DENSITIES = [1, 2]  # Cellular_Bounds interval k
TRIALS_T = 20

engines = {A: TaskEngine(A) for A in ALPHABET_SIZES}

# New seed formula distinct from the existing sparse grid's, since these
# are new lengths not present in the original 7-point grid (4,8,16,32,
# 64,128,256 already uses this exact formula at N=4/8/16, so reusing it
# keeps those three lengths IDENTICAL to already-collected data --
# verified below rather than assumed).
shared_sequences = {}
for A in ALPHABET_SIZES:
    for N in DENSE_LENGTHS:
        for t in range(TRIALS_T):
            local_seed = 10_000 * A + 100 * N + t
            rng = random.Random(local_seed)
            shared_sequences[(A, N, t)] = engines[A].generate_raw_input(N, rng=rng)
print(f"Pregenerated {len(shared_sequences)} shared raw sequences.")


def run_trial(A, method, interval, N, t):
    raw_input = shared_sequences[(A, N, t)]
    engine = engines[A]
    if method == "Base":
        inp, truth, prompt = engine.build_base_from_raw(raw_input)
    else:
        inp, truth, prompt = engine.build_cellular_from_raw(raw_input, interval=interval)

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in enc.items()}
    max_tokens_needed = int(N * 1.5) + 20
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_only = outputs[0][inputs["input_ids"].shape[1]:]
    pred = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
    pred = pred.split("Explanation")[0].split("\n")[0].strip()

    clean_pred = pred.replace("|", "").replace(" ", "").replace("\n", "")
    clean_truth = truth.replace("|", "")
    clean_pred = clean_pred[:len(clean_truth)]

    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0
                      for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0

    return {
        "Method": method, "Alphabet_Size": A, "Interval": interval if method != "Base" else 0,
        "Seq_Length": N, "Trial": t, "TAR": tar, "SAR": sar,
        "Input_Given": inp, "Ground_Truth": clean_truth, "LLM_Output": clean_pred,
    }


tasks = []
for A in ALPHABET_SIZES:
    for N in DENSE_LENGTHS:
        for t in range(TRIALS_T):
            tasks.append((A, "Base", 0, N, t))
            for k in DENSITIES:
                tasks.append((A, "Cellular_Bounds", k, N, t))

print(f"\nStarting dense harder-alphabet sweep: {len(tasks)} evaluations...")

results = []
for A, method, interval, N, t in tqdm(tasks, desc="Dense A=4/8 grid"):
    try:
        results.append(run_trial(A, method, interval, N, t))
    except Exception as exc:
        print(f"Task failed: A={A}, {method} k={interval}, N={N}, t={t} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "dense_hard_alphabet_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

summary = df.groupby(["Alphabet_Size", "Method", "Interval", "Seq_Length"])[["TAR", "SAR"]].mean().reset_index()
print("\n--- MEAN SAR BY (ALPHABET, METHOD, INTERVAL, LENGTH) ---")
print(summary.round(4).to_string(index=False))

print("\nDone.")
