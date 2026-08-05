import matplotlib
matplotlib.use("Agg")

# ============================================================
# Q4: jitter placement (small perturbation), vs. the existing full
# random-scatter test
# ============================================================
# Appendix B already tests the EXTREME case: markers scattered uniformly
# at random across the sequence, ignoring periodicity entirely. The
# reviewer's actual question is about small JITTER -- does the effect
# survive if boundary positions are approximately, but not exactly,
# periodic? This tests that directly: canonical periodic positions
# {k, 2k, 3k, ...} each perturbed by +/-1 or +/-2 positions (drawn fresh
# per boundary, per trial), clipped to stay in-range, duplicates removed
# (so boundary count can be very slightly below the exact-periodic count
# -- logged per trial).
#
# Same task/model/density as the other marker-property ablations
# (cyclic_shift, alphabet=2, Qwen2.5-7B), compared against the existing
# exact-periodic Cellular_Bounds condition at the same k, same seeds
# (so this is a direct paired addition, not a full rerun).

import os

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random
import numpy as np
import pandas as pd
from tqdm import tqdm

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

ALPHABET_SIZE = 2
ALPHABET = [chr(ord('A') + i) for i in range(ALPHABET_SIZE)]
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
INTERVAL = 4
SEQUENCE_LENGTHS = [8, 16, 32, 64]
JITTER_AMOUNTS = [1, 2]
TRIALS_T = 20


def generate_raw_input(length, rng):
    return rng.choices(ALPHABET, k=length)


def shift_char(c):
    return ALPHABET[(CHAR_TO_IDX[c] + 1) % ALPHABET_SIZE]


def jittered_positions(N, k, jitter, rng):
    canonical = list(range(k, N, k))  # same convention as insert_boundaries: k,2k,... excluding trailing
    positions = set()
    for pos in canonical:
        offset = rng.randint(-jitter, jitter)
        jp = pos + offset
        jp = max(1, min(N - 1, jp))
        positions.add(jp)
    return sorted(positions)


def build_jittered(raw_input, k, jitter, rng):
    N = len(raw_input)
    positions = set(jittered_positions(N, k, jitter, rng))
    input_bounded, output_bounded = [], []
    for i, c in enumerate(raw_input):
        input_bounded.append(c)
        output_bounded.append(shift_char(c))
        if i in positions and i != N - 1:
            input_bounded.append("|")
            output_bounded.append("|")
    return "".join(input_bounded), "".join(output_bounded), len(positions)


def build_prompt(input_bounded):
    return f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(ALPHABET)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {ALPHABET[-1]} -> {ALPHABET[0]}, and {ALPHABET[0]} -> {ALPHABET[1]}.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """


def run_trial(N, jitter, t):
    local_seed = 10_000 * ALPHABET_SIZE + 100 * N + t
    rng = random.Random(local_seed)
    raw_input = generate_raw_input(N, rng)
    input_bounded, truth, n_boundaries = build_jittered(raw_input, INTERVAL, jitter, rng)
    prompt = build_prompt(input_bounded)

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
        "Method": f"Jitter_{jitter}", "Interval": INTERVAL, "Jitter": jitter,
        "Seq_Length": N, "Trial": t, "TAR": tar, "SAR": sar, "N_Boundaries": n_boundaries,
        "Input_Given": input_bounded, "Ground_Truth": clean_truth, "LLM_Output": clean_pred,
    }


tasks = [(N, j, t) for N in SEQUENCE_LENGTHS for j in JITTER_AMOUNTS for t in range(TRIALS_T)]
print(f"\nStarting jitter-placement sweep: {len(tasks)} evaluations...")

results = []
for N, j, t in tqdm(tasks, desc="Jitter placement"):
    try:
        results.append(run_trial(N, j, t))
    except Exception as exc:
        print(f"Task failed: N={N} jitter={j} t={t} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "jitter_placement_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

print("\n--- Mean TAR by jitter amount and length ---")
print(df.groupby(["Jitter", "Seq_Length"])["TAR"].mean().unstack().round(4).to_string())

# --- Compare against existing exact-periodic Cellular_Bounds and Base ---
existing_path = os.path.join(RESULTS_DIR, "thermodynamics_grid_templated_v2.csv")
if os.path.exists(existing_path):
    existing = pd.read_csv(existing_path)
    existing = existing[(existing.Alphabet_Size == ALPHABET_SIZE) & (existing.Seq_Length.isin(SEQUENCE_LENGTHS))]
    rows = []
    for N in SEQUENCE_LENGTHS:
        sub = existing[existing.Seq_Length == N]
        base_ref = sub[sub.Method == "Base"]["TAR"].mean()
        exact_ref = sub[(sub.Method == "Cellular_Bounds") & (sub.Interval == INTERVAL)]["TAR"].mean()
        row = {"Seq_Length": N, "Base_TAR": base_ref, "Exact_Periodic_TAR": exact_ref}
        for j in JITTER_AMOUNTS:
            row[f"Jitter{j}_TAR"] = df[(df.Seq_Length == N) & (df.Jitter == j)]["TAR"].mean()
        rows.append(row)
    comparison = pd.DataFrame(rows)
    print("\n--- DOES SMALL JITTER PRESERVE THE EXACT-PERIODIC BENEFIT? ---")
    print(comparison.round(4).to_string(index=False))
    comparison.to_csv(os.path.join(RESULTS_DIR, "jitter_placement_comparison.csv"), index=False)
else:
    print(f"\nNOTE: {existing_path} not found, skipping comparison (jitter grid still saved above).")

print("\nJitter placement sweep complete.")
