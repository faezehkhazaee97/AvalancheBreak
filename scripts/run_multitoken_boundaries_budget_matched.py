import matplotlib
matplotlib.use("Agg")

# ============================================================
# TOKEN-BUDGET-MATCHED MULTI-TOKEN BOUNDARY CONTROL
# ============================================================
# Section 3.5's own "tokenization caveat" flags a specific, named gap:
# `<|>` tokenizes as 3 separate tokens (`<`, `|`, `>`) versus 1 token for
# every single-character marker, so at the SAME character-level interval
# (4), `<|>` injects 3x the token-level structural signal per boundary
# that `|` does. That sweep has no condition that holds total token
# budget constant while still using a genuinely multi-token marker --
# this script is exactly that condition.
#
# Design: `|` at interval=4 injects (roughly) N/4 extra tokens across a
# length-N sequence (1 token per boundary, ~N/4 boundaries). `<|>` at
# interval=12 injects the same total token budget: 3 tokens per
# boundary, but only ~N/12 boundaries, so 3 * (N/12) = N/4 -- identical
# to `|`@interval=4 in total extra tokens, while never being other than
# a genuine 3-token marker. If `<|>`@interval=12 performs like
# `|`@interval=4 (not like `<|>`@interval=4, which is NOT budget-matched
# and already in boundary_ablation_multitoken_grid.csv), that would
# indicate total token budget is what matters, not marker identity or
# per-boundary token count. If it instead still outperforms
# `|`@interval=4, that argues the effect is about checkpoint frequency
# in token-space some other way, not simply "more tokens somewhere in
# the sequence."
#
# Same shared-sequence seed formula (70_000*N+t) as
# boundary_ablation_grid.csv and boundary_ablation_multitoken_grid.csv,
# so all three are directly paired, not just qualitatively similar.

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
BASELINE_INTERVAL = 4       # matches |'s existing interval (1 token/boundary)
BUDGET_MATCHED_INTERVAL = 12  # 3x interval -> 3 tokens/boundary at 1/3 the boundary count = same total budget

engine_ablation = TaskEngine(alphabet_size=ABLATION_ALPHABET_SIZE)


def boundary_positions_periodic(length, interval):
    return set(i for i in range(interval - 1, length - 1, interval))


def build_variant(raw_input, marker, interval):
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

    marker_desc = repr(marker)
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


# ============================================================
# Empirical token-budget check (same methodology as Section 3.5's
# tokenization caveat: tokenize with the model's own tokenizer, not
# character counts, since BPE merging can change the picture).
# ============================================================
print("\n--- Empirical token-budget check (N=16) ---")
sample_rng = random.Random(70_000 * 16 + 0)
sample_raw = engine_ablation.generate_raw_input(16, rng=sample_rng)
_, _, prompt_pipe_4, nb_pipe_4 = build_variant(sample_raw, marker="|", interval=BASELINE_INTERVAL)
_, _, prompt_multi_4, nb_multi_4 = build_variant(sample_raw, marker="<|>", interval=BASELINE_INTERVAL)
_, _, prompt_multi_12, nb_multi_12 = build_variant(sample_raw, marker="<|>", interval=BUDGET_MATCHED_INTERVAL)

n_tok_pipe_4 = len(tokenizer(prompt_pipe_4)["input_ids"])
n_tok_multi_4 = len(tokenizer(prompt_multi_4)["input_ids"])
n_tok_multi_12 = len(tokenizer(prompt_multi_12)["input_ids"])
print(f"|  @ interval={BASELINE_INTERVAL}:  {nb_pipe_4} boundaries, full-prompt token count = {n_tok_pipe_4}")
print(f"<|> @ interval={BASELINE_INTERVAL}:  {nb_multi_4} boundaries, full-prompt token count = {n_tok_multi_4}  (NOT budget-matched, for reference)")
print(f"<|> @ interval={BUDGET_MATCHED_INTERVAL}: {nb_multi_12} boundaries, full-prompt token count = {n_tok_multi_12}  (budget-matched target)")

token_budget_check = pd.DataFrame([
    {"Condition": f"|@interval={BASELINE_INTERVAL}", "N_Boundaries": nb_pipe_4, "Full_Prompt_Tokens": n_tok_pipe_4},
    {"Condition": f"<|>@interval={BASELINE_INTERVAL}", "N_Boundaries": nb_multi_4, "Full_Prompt_Tokens": n_tok_multi_4},
    {"Condition": f"<|>@interval={BUDGET_MATCHED_INTERVAL}", "N_Boundaries": nb_multi_12, "Full_Prompt_Tokens": n_tok_multi_12},
])
token_budget_check.to_csv(os.path.join(RESULTS_DIR, "boundary_token_budget_check.csv"), index=False)


def run_trial(raw_input, marker, interval, N, trial):
    inp, truth, prompt, n_boundaries = build_variant(raw_input, marker=marker, interval=interval)

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
    pred = pred_full.split("Explanation")[0].split("\n")[0].strip()

    clean_pred = pred.replace(marker, "").replace(" ", "").replace("\n", "")
    clean_truth = truth.replace(marker, "")
    clean_pred = clean_pred[:len(clean_truth)]

    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0
                      for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0

    return {
        "Marker": marker, "Interval": interval, "Ablation_Axis": "token-budget-matched control",
        "Seq_Length": N, "Trial": trial, "N_Boundaries": n_boundaries,
        "Shared_Sequence": "".join(raw_input), "TAR": tar, "SAR": sar,
        "Raw_Response": pred_full,
    }


# Identical seed formula to boundary_ablation_grid.csv / boundary_ablation_multitoken_grid.csv
shared_sequences = {}
for N in ABLATION_SEQUENCE_LENGTHS:
    for t in range(ABLATION_TRIALS):
        rng = random.Random(70_000 * N + t)
        shared_sequences[(N, t)] = engine_ablation.generate_raw_input(N, rng=rng)
print(f"\nPregenerated {len(shared_sequences)} shared raw sequences (paired with boundary_ablation_multitoken_grid.csv).")

tasks = [(N, t) for N in ABLATION_SEQUENCE_LENGTHS for t in range(ABLATION_TRIALS)]
print(f"\nStarting token-budget-matched <|>@interval={BUDGET_MATCHED_INTERVAL} sweep: {len(tasks)} evaluations...")

results = []
for (N, t) in tqdm(tasks, desc="Budget-matched <|>"):
    raw_input = shared_sequences[(N, t)]
    try:
        results.append(run_trial(raw_input, "<|>", BUDGET_MATCHED_INTERVAL, N, t))
    except Exception as exc:
        print(f"Task failed: N={N}, trial={t} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "boundary_ablation_multitoken_budget_matched_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSweep complete. Saved to '{csv_path}'")

summary = df.groupby("Seq_Length")[["TAR", "SAR"]].mean()
print("\n--- MEAN ACCURACY, <|>@interval=12 (budget-matched) ---")
print(summary.round(4).to_string())

# --- Compare against the existing |@4 baseline and the non-budget-matched <|>@4 ---
baseline_df = pd.read_csv(os.path.join(RESULTS_DIR, "boundary_ablation_grid.csv"))
baseline_pipe = baseline_df[
    (baseline_df.Marker == "|") & (baseline_df.Ablation_Axis == "marker+placement baseline")
].groupby("Seq_Length")["TAR"].mean()

multitoken_df = pd.read_csv(os.path.join(RESULTS_DIR, "boundary_ablation_multitoken_grid.csv"))
multitoken_pipe3_at4 = multitoken_df[multitoken_df.Marker == "<|>"].groupby("Seq_Length")["TAR"].mean()

comparison = pd.DataFrame({
    "Pipe_at_interval4_TAR": baseline_pipe,
    "TriToken_at_interval4_TAR_NOT_budget_matched": multitoken_pipe3_at4,
    "TriToken_at_interval12_TAR_budget_matched": summary["TAR"],
})
print("\n--- IS <|> STILL AHEAD OF | ONCE TOTAL TOKEN BUDGET IS EQUALIZED? ---")
print(comparison.round(4).to_string())
comparison.to_csv(os.path.join(RESULTS_DIR, "boundary_ablation_budget_matched_comparison.csv"))

print("\nDone.")
