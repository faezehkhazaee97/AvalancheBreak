import matplotlib
matplotlib.use("Agg")

# ============================================================
# DIVIDE-AND-CONQUER BENCHMARK + WALL-CLOCK / TOKEN COST ACCOUNTING
# ============================================================
# Open question from review: is a "|" boundary just a soft, in-context
# imitation of literally splitting the sequence into independent chunks?
# Cellular_Bounds keeps everything in ONE prompt/one generate() call, with
# the model choosing how much to attend across the "|" marker. True
# divide-and-conquer (D&C) removes that choice entirely: each chunk is a
# separate prompt, a separate generate() call, and the model has zero
# access to any other chunk's tokens. If D&C matches or beats
# Cellular_Bounds on accuracy, that's evidence boundaries mainly work by
# shrinking effective context, not by doing anything more subtle with
# attention. If D&C is worse, that's evidence the model IS using some
# cross-chunk information even through the "|" markers, which pure
# chunking throws away.
#
# This also settles the "real cost" question raised separately: D&C
# requires ceil(N/chunk) separate generate() calls per trial (each with
# its own prompt-processing overhead), vs exactly one call for Base and one
# for Cellular_Bounds. Prior grids never captured wall-clock time
# per-call -- token counts alone don't show the fixed per-call latency
# (prompt encoding, KV-cache setup) that repeated small calls pay
# redundantly. This script times all three conditions freshly, on the
# SAME shared raw sequences (same seeding formula as the boundary-ablation
# grid: random.Random(70_000*N + t), Alphabet_Size=2, chunk/interval=4,
# Seq_Lengths=[8,16,32,64], 20 trials) so accuracy AND cost are directly
# comparable against the existing ablation grid's | baseline condition.

import os

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"Results directory: {RESULTS_DIR}/")

# ==== MODEL LOADER ====

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
import time
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


DC_ALPHABET_SIZE = 2          # matches the boundary-ablation grid
DC_SEQUENCE_LENGTHS = [8, 16, 32, 64]   # matches the boundary-ablation grid
DC_CHUNK_SIZE = 4              # matches the boundary-ablation grid's interval=4
DC_TRIALS = 20                  # matches the boundary-ablation grid

engine_dc = TaskEngine(alphabet_size=DC_ALPHABET_SIZE)

# Same seed formula as the boundary-ablation grid's shared_sequences, so
# these raw sequences are IDENTICAL to the ones used for the "| periodic"
# ablation condition at the same (Seq_Length, Trial) -- a real paired
# comparison against boundary_ablation_grid.csv, not just matched means.
shared_sequences = {}
for N in DC_SEQUENCE_LENGTHS:
    for t in range(DC_TRIALS):
        rng = random.Random(70_000 * N + t)
        shared_sequences[(N, t)] = engine_dc.generate_raw_input(N, rng=rng)
print(f"Pregenerated {len(shared_sequences)} shared raw sequences (matched to boundary-ablation grid).")


def call_model(prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.perf_counter() - t0
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_decoded_len = len(prompt)
    pred = full_response[prompt_decoded_len:].strip()
    n_new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    return pred, elapsed, int(n_new_tokens)


def score(pred_str, truth_str):
    pred_str = pred_str[:len(truth_str)]
    binary_series = [1 if i < len(pred_str) and pred_str[i] == truth_str[i] else 0 for i in range(len(truth_str))]
    tar = sum(binary_series) / len(truth_str) if truth_str else 0.0
    sar = 1.0 if (len(binary_series) == len(truth_str) and sum(binary_series) == len(binary_series)) else 0.0
    return tar, sar


def run_base_trial(raw_input, N, t):
    inp, truth, prompt = engine_dc.build_base_from_raw(raw_input)
    max_tokens_needed = int(N * 1.5) + 20
    pred, elapsed, n_tokens = call_model(prompt, max_tokens_needed)
    pred = pred.split("Explanation")[0].split("\n")[0].strip()
    clean_pred = pred.replace(" ", "").replace("\n", "")
    tar, sar = score(clean_pred, truth)
    return {
        "Condition": "Base", "Seq_Length": N, "Trial": t, "N_Generate_Calls": 1,
        "TAR": tar, "SAR": sar, "Wall_Time_Sec": elapsed, "New_Tokens": n_tokens,
    }


def run_cellular_trial(raw_input, N, t):
    inp, truth, prompt = engine_dc.build_cellular_from_raw(raw_input, interval=DC_CHUNK_SIZE)
    n_boundaries = (N - 1) // DC_CHUNK_SIZE
    max_tokens_needed = int(N * 1.5) + 20 + n_boundaries
    pred, elapsed, n_tokens = call_model(prompt, max_tokens_needed)
    pred = pred.split("Explanation")[0].split("\n")[0].strip()
    clean_pred = pred.replace("|", "").replace(" ", "").replace("\n", "")
    clean_truth = truth.replace("|", "")
    tar, sar = score(clean_pred, clean_truth)
    return {
        "Condition": "Cellular_Bounds", "Seq_Length": N, "Trial": t, "N_Generate_Calls": 1,
        "TAR": tar, "SAR": sar, "Wall_Time_Sec": elapsed, "New_Tokens": n_tokens,
    }


def run_dc_trial(raw_input, N, t):
    chunks = [raw_input[i:i + DC_CHUNK_SIZE] for i in range(0, N, DC_CHUNK_SIZE)]
    full_truth = "".join(
        engine_dc.alphabet[(engine_dc.char_to_idx[c] + 1) % DC_ALPHABET_SIZE] for c in raw_input
    )
    outputs = []
    total_time = 0.0
    total_tokens = 0
    for chunk in chunks:
        inp, chunk_truth, prompt = engine_dc.build_base_from_raw(chunk)
        max_tokens_needed = int(len(chunk) * 1.5) + 20
        pred, elapsed, n_tokens = call_model(prompt, max_tokens_needed)
        pred = pred.split("Explanation")[0].split("\n")[0].strip()
        clean_pred = pred.replace(" ", "").replace("\n", "")[:len(chunk)]
        outputs.append(clean_pred)
        total_time += elapsed
        total_tokens += n_tokens

    full_pred = "".join(outputs)
    tar, sar = score(full_pred, full_truth)
    return {
        "Condition": "Divide_Conquer", "Seq_Length": N, "Trial": t, "N_Generate_Calls": len(chunks),
        "TAR": tar, "SAR": sar, "Wall_Time_Sec": total_time, "New_Tokens": total_tokens,
    }


all_results = []
tasks = [(N, t) for N in DC_SEQUENCE_LENGTHS for t in range(DC_TRIALS)]
print(f"\nStarting divide-and-conquer + cost benchmark: {len(tasks)} trials x 3 conditions...")

for N, t in tqdm(tasks, desc="D&C benchmark"):
    raw_input = shared_sequences[(N, t)]
    try:
        all_results.append(run_base_trial(raw_input, N, t))
    except Exception as exc:
        print(f"Base trial failed: N={N}, t={t} | Error: {exc}")
    try:
        all_results.append(run_cellular_trial(raw_input, N, t))
    except Exception as exc:
        print(f"Cellular trial failed: N={N}, t={t} | Error: {exc}")
    try:
        all_results.append(run_dc_trial(raw_input, N, t))
    except Exception as exc:
        print(f"D&C trial failed: N={N}, t={t} | Error: {exc}")

df = pd.DataFrame(all_results)
csv_path = os.path.join(RESULTS_DIR, "divide_conquer_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nBenchmark complete. Saved to '{csv_path}'")

print("\n--- ACCURACY BY CONDITION & LENGTH ---")
acc_summary = df.groupby(["Condition", "Seq_Length"])[["TAR", "SAR"]].mean().round(4)
print(acc_summary.to_string())

print("\n--- COST BY CONDITION & LENGTH (mean per trial) ---")
cost_summary = df.groupby(["Condition", "Seq_Length"])[["N_Generate_Calls", "Wall_Time_Sec", "New_Tokens"]].mean().round(3)
print(cost_summary.to_string())

print("\n--- TOTAL COST ACROSS ALL TRIALS, BY CONDITION ---")
total_cost = df.groupby("Condition")[["N_Generate_Calls", "Wall_Time_Sec", "New_Tokens"]].sum().round(1)
print(total_cost.to_string())

# --- Headline comparison: does D&C match or beat Cellular_Bounds accuracy? ---
pivot_tar = df.groupby(["Condition", "Seq_Length"])["TAR"].mean().unstack("Condition")
if "Divide_Conquer" in pivot_tar.columns and "Cellular_Bounds" in pivot_tar.columns:
    print("\n--- DOES TRUE DIVIDE-AND-CONQUER MATCH/BEAT THE SOFT '|' BOUNDARY? ---")
    diff = pivot_tar["Divide_Conquer"] - pivot_tar["Cellular_Bounds"]
    print(diff.round(4).to_string())
    n_dc_better = int((diff > 0).sum())
    print(f"\nD&C scores higher TAR than Cellular_Bounds at {n_dc_better}/{len(diff)} tested lengths.")

pivot_time = df.groupby(["Condition", "Seq_Length"])["Wall_Time_Sec"].mean().unstack("Condition")
if "Divide_Conquer" in pivot_time.columns and "Cellular_Bounds" in pivot_time.columns:
    print("\n--- WALL-CLOCK COST: D&C vs single-pass Cellular_Bounds (seconds per trial) ---")
    ratio = pivot_time["Divide_Conquer"] / pivot_time["Cellular_Bounds"]
    print(pd.DataFrame({"Cellular_Bounds": pivot_time["Cellular_Bounds"],
                         "Divide_Conquer": pivot_time["Divide_Conquer"],
                         "DC_over_Cellular_Ratio": ratio}).round(3).to_string())

print("\nDivide-and-conquer + cost benchmark complete.")
