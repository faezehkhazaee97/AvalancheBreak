import matplotlib
matplotlib.use("Agg")

# ============================================================
# REPRODUCIBILITY CHECK
# ============================================================
# Decisive test for the draft-vs-current discrepancy: rerun EXACTLY the
# draft's original grid (Alphabet in {2,4,8}, Seq_Length in {4,8,16,32},
# Interval in {0,2,4,8}, T=20 trials, identical prompt/TaskEngine/seed
# formula -- verified byte-identical to code/LLM-Focus.ipynb's cell 5)
# fresh, today, and compare row-by-row against what's already stored in
# thermodynamics_grid.csv for the exact same (Alphabet_Size, Seq_Length,
# Interval, Trial) keys. Since the input sequences are seeded
# deterministically and decoding is greedy (do_sample=False), a fresh run
# should reproduce the stored data exactly IF today's environment is
# internally consistent. Any mismatch is direct, first-hand evidence of
# non-determinism in inference itself (library/hardware), not a
# methodology or code difference -- that possibility was already ruled
# out by diffing the two notebooks.

import os

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"Results directory: {RESULTS_DIR}/")

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
import itertools
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
            self.alphabet[(self.char_to_idx[c] + 1) % self.alphabet_size]
            for c in raw_input
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
        input_w_bounds = ""
        output_w_bounds = ""
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


# Exactly the draft's grid -- verified identical to code/LLM-Focus.ipynb cell 5
ALPHABET_SIZES = [2, 4, 8]
SEQUENCE_LENGTHS = [4, 8, 16, 32]
PERIODIC_INTERVALS = [0, 2, 4, 8]
TRIALS_T = 20

engines = {A: TaskEngine(A) for A in ALPHABET_SIZES}

shared_sequences = {}
for A in ALPHABET_SIZES:
    engine = engines[A]
    for N in SEQUENCE_LENGTHS:
        for t in range(TRIALS_T):
            local_seed = 10_000 * A + 100 * N + t
            rng = random.Random(local_seed)
            shared_sequences[(A, N, t)] = engine.generate_raw_input(N, rng=rng)
print(f"Pregenerated {len(shared_sequences)} shared raw sequences (identical seed formula to the draft).")


def run_single_trial(task_params):
    A, N, k, t = task_params
    engine = engines[A]
    method = "Base" if k == 0 else "Cellular_Bounds"
    raw_input = shared_sequences[(A, N, t)]

    if method == "Base":
        inp, truth, prompt = engine.build_base_from_raw(raw_input)
    else:
        inp, truth, prompt = engine.build_cellular_from_raw(raw_input, interval=k)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    max_tokens_needed = int(N * 1.5) + 20

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = full_response[len(prompt):].strip()
    pred = pred.split("Explanation")[0].split("\n")[0].strip()

    if method == "Cellular_Bounds":
        clean_pred = pred.replace("|", "").replace(" ", "").replace("\n", "")
        clean_truth = truth.replace("|", "")
    else:
        clean_pred = pred.replace(" ", "").replace("\n", "")
        clean_truth = truth

    clean_pred = clean_pred[:len(clean_truth)]
    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0
                      for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if len(clean_truth) > 0 else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0

    return {
        "Alphabet_Size": A, "Method": method, "Interval": k, "Seq_Length": N, "Trial": t,
        "TAR": tar, "SAR": sar,
    }


tasks = list(itertools.product(ALPHABET_SIZES, SEQUENCE_LENGTHS, PERIODIC_INTERVALS, range(TRIALS_T)))
filtered_tasks = [task for task in tasks if task[2] == 0 or task[2] < task[1]]
print(f"\nStarting reproducibility check: {len(filtered_tasks)} evaluations (fresh run of the draft's exact grid)...")

fresh_results = []
for task in tqdm(filtered_tasks, desc="Repro check"):
    try:
        fresh_results.append(run_single_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

fresh_df = pd.DataFrame(fresh_results)
fresh_csv = os.path.join(RESULTS_DIR, "repro_check_fresh_grid.csv")
fresh_df.to_csv(fresh_csv, index=False)
print(f"\nFresh run complete. Saved to '{fresh_csv}'")

# --- Compare against the existing stored data for the same keys ---
stored_df = pd.read_csv(os.path.join(RESULTS_DIR, "thermodynamics_grid.csv"))
stored_df = stored_df[
    stored_df["Alphabet_Size"].isin(ALPHABET_SIZES)
    & stored_df["Seq_Length"].isin(SEQUENCE_LENGTHS)
    & stored_df["Interval"].isin(PERIODIC_INTERVALS)
][["Alphabet_Size", "Method", "Interval", "Seq_Length", "Trial", "TAR", "SAR"]]

merged = fresh_df.merge(
    stored_df, on=["Alphabet_Size", "Method", "Interval", "Seq_Length", "Trial"],
    suffixes=("_fresh", "_stored"), how="inner"
)
print(f"\nMatched {len(merged)} rows between fresh run and stored data on "
      f"(Alphabet_Size, Method, Interval, Seq_Length, Trial).")

merged["TAR_match"] = np.isclose(merged["TAR_fresh"], merged["TAR_stored"])
merged["SAR_match"] = np.isclose(merged["SAR_fresh"], merged["SAR_stored"])
merged.to_csv(os.path.join(RESULTS_DIR, "repro_check_comparison.csv"), index=False)

n_tar_match = merged["TAR_match"].sum()
n_sar_match = merged["SAR_match"].sum()
print(f"\n--- ROW-LEVEL AGREEMENT ---")
print(f"TAR matches exactly: {n_tar_match}/{len(merged)} ({100*n_tar_match/len(merged):.1f}%)")
print(f"SAR matches exactly: {n_sar_match}/{len(merged)} ({100*n_sar_match/len(merged):.1f}%)")

print("\n--- MISMATCHED ROWS (fresh vs stored) ---")
mismatches = merged[~merged["TAR_match"]]
if len(mismatches):
    print(mismatches[["Alphabet_Size", "Method", "Interval", "Seq_Length", "Trial",
                       "TAR_fresh", "TAR_stored"]].round(3).to_string(index=False))
else:
    print("None -- every row matched exactly.")

print("\n--- AGGREGATE COMPARISON: mean TAR by (Alphabet_Size, Seq_Length, Method), fresh vs stored ---")
agg = merged.groupby(["Alphabet_Size", "Seq_Length", "Method"])[["TAR_fresh", "TAR_stored"]].mean()
agg["Abs_Diff"] = (agg["TAR_fresh"] - agg["TAR_stored"]).abs()
print(agg.round(3).to_string())

print("\nReproducibility check complete.")
