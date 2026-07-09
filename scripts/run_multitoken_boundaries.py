import matplotlib
matplotlib.use("Agg")

# ============================================================
# MULTI-TOKEN / NATURAL-TOKEN BOUNDARY MARKERS
# ============================================================
# Reviewer ask (Q6): does a multi-character marker ("||", "<|>") or a
# naturally-occurring structuring token (newline, comma) work as well as
# the single "|" character already tested in Fig 5's marker-identity
# panel? Same task, same alphabet, same periodic placement/interval, same
# shared raw sequences (identical seed formula: random.Random(70_000*N+t))
# as the existing boundary_ablation_grid.csv, so these four new
# conditions are directly paired against the existing |/;/# results, not
# just qualitatively similar.
#
# One real gotcha caught before running: the established scoring
# convention truncates a response at its first newline (to strip trailing
# "Explanation: ..." chatter the model sometimes adds). If "\n" is itself
# the boundary marker, that truncation would cut the answer off at the
# very first boundary the model echoes back -- silently breaking that one
# condition. Handled below by skipping the newline-truncation step
# specifically when marker == "\n".

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


ABLATION_ALPHABET_SIZE = 2
ABLATION_SEQUENCE_LENGTHS = [8, 16, 32, 64]
ABLATION_TRIALS = 20
INTERVAL = 4  # matches the existing marker-identity panel exactly

engine_ablation = TaskEngine(alphabet_size=ABLATION_ALPHABET_SIZE)


def boundary_positions_periodic(length, interval):
    return set(i for i in range(interval - 1, length - 1, interval))


def build_variant(raw_input, marker, interval=4):
    length = len(raw_input)
    boundary_positions = boundary_positions_periodic(length, interval)

    input_w_bounds, output_w_bounds = "", ""
    for i, char in enumerate(raw_input):
        shifted_char = engine_ablation.alphabet[(engine_ablation.char_to_idx[char] + 1) % ABLATION_ALPHABET_SIZE]
        input_w_bounds += char
        output_w_bounds += shifted_char
        if i in boundary_positions:
            input_w_bounds += marker
            output_w_bounds += marker

    marker_desc = repr(marker)  # so "\n" shows up readably in the prompt text itself
    prompt = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(engine_ablation.alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {engine_ablation.alphabet[-1]} -> {engine_ablation.alphabet[0]}, and {engine_ablation.alphabet[0]} -> {engine_ablation.alphabet[1]}.
Constraint: The input contains structural boundary markers ({marker_desc}). DO NOT shift them. Copy them exactly to the output.
No spaces, no other explanations.

Input: {input_w_bounds}
Output: """
    return input_w_bounds, output_w_bounds, prompt, len(boundary_positions)


NEW_CONDITIONS = [
    {"Marker": "||", "Ablation_Axis": "multi-token marker"},
    {"Marker": "<|>", "Ablation_Axis": "multi-token marker"},
    {"Marker": "\n", "Ablation_Axis": "natural token marker"},
    {"Marker": ",", "Ablation_Axis": "natural token marker"},
]


def run_trial(raw_input, condition, N, trial):
    marker = condition["Marker"]
    inp, truth, prompt, n_boundaries = build_variant(raw_input, marker=marker, interval=INTERVAL)

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)
    max_tokens_needed = int(N * 1.5) + 20 + n_boundaries * max(1, len(marker))
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_only = outputs[0][inputs["input_ids"].shape[1]:]
    pred_full = tokenizer.decode(gen_only, skip_special_tokens=True).strip()

    if marker == "\n":
        # Do NOT truncate at first newline here -- that IS the boundary marker.
        pred = pred_full.split("Explanation")[0].strip()
    else:
        pred = pred_full.split("Explanation")[0].split("\n")[0].strip()

    clean_pred = pred.replace(marker, "").replace(" ", "")
    if marker != "\n":
        clean_pred = clean_pred.replace("\n", "")
    clean_truth = truth.replace(marker, "")
    clean_pred = clean_pred[:len(clean_truth)]

    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0
                      for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0

    return {
        "Marker": marker, "Ablation_Axis": condition["Ablation_Axis"],
        "Seq_Length": N, "Trial": trial, "N_Boundaries": n_boundaries,
        "Shared_Sequence": "".join(raw_input), "TAR": tar, "SAR": sar,
        "Raw_Response": pred_full,
    }


# Identical seed formula to boundary_ablation_grid.csv's shared_sequences,
# so these raw sequences -- and therefore this comparison -- are paired
# with the existing |/;/# marker-identity results, not just similar.
shared_sequences = {}
for N in ABLATION_SEQUENCE_LENGTHS:
    for t in range(ABLATION_TRIALS):
        rng = random.Random(70_000 * N + t)
        shared_sequences[(N, t)] = engine_ablation.generate_raw_input(N, rng=rng)
print(f"Pregenerated {len(shared_sequences)} shared raw sequences (paired with boundary_ablation_grid.csv).")

tasks = [
    (cond, N, t)
    for cond in NEW_CONDITIONS
    for N in ABLATION_SEQUENCE_LENGTHS
    for t in range(ABLATION_TRIALS)
]
print(f"\nStarting multi-token/natural-token boundary sweep: {len(tasks)} evaluations...")

results = []
for (cond, N, t) in tqdm(tasks, desc="Multi-token boundaries"):
    raw_input = shared_sequences[(N, t)]
    try:
        results.append(run_trial(raw_input, cond, N, t))
    except Exception as exc:
        print(f"Task failed: {cond}, N={N}, trial={t} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "boundary_ablation_multitoken_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSweep complete. Saved to '{csv_path}'")

summary = df.groupby(["Marker", "Seq_Length"])[["TAR", "SAR"]].mean().reset_index()
print("\n--- MEAN ACCURACY BY MARKER ---")
print(summary.round(4).to_string(index=False))

print("\nDone.")
