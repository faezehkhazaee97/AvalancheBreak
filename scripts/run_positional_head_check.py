import matplotlib
matplotlib.use("Agg")

# ============================================================
# POSITIONAL-HEAD OVERLAP CHECK
# ============================================================
# Section 3.17 already checks whether the 10 boundary-attending heads
# overlap with induction heads (content-based copying). This is the
# companion check for a functionally distinct, equally well-documented
# head type: "positional" or "previous-token" heads -- heads whose
# attention is dominated by a fixed RELATIVE offset (canonically,
# attending to the token immediately before the current query) rather
# than by content matching. This is measured the same way induction
# heads are: on random, non-repeating synthetic token sequences (no
# induction structure, no boundaries, no task semantics at all), so any
# attention concentration at a fixed relative offset reflects a
# position-driven default, not something learned from this task's
# specific structure.
#
# Same 784-head population, same model, same operational-test style as
# run_induction_head_check.py: previous-token attention mass vs.
# baseline attention to all other earlier positions, for all heads, then
# where the 10 real target heads (causal_ablation_target_heads.csv)
# rank in that distribution.

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

print(f"Loading {MODEL_ID} into VRAM (eager attention, need output_attentions)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager",
)
model.eval()

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads

N_TRIALS = 20
SEQ_LEN = 60  # random, non-repeating -- no induction structure present
SAFE_ID_LO, SAFE_ID_HI = 2000, 100_000

prevtoken_mass = np.zeros((n_layers, n_heads))
baseline_mass = np.zeros((n_layers, n_heads))

print(f"\nRunning {N_TRIALS} synthetic positional-test trials (seq_len={SEQ_LEN})...")
for trial in range(N_TRIALS):
    rng = random.Random(2000 + trial)
    seq = [rng.randrange(SAFE_ID_LO, SAFE_ID_HI) for _ in range(SEQ_LEN)]  # fully random, no repeats
    input_ids = torch.tensor([seq], device=model.device)
    attn_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)

    # Query positions start at 10, not 1: excluding both the previous-token
    # target and self from a causal window that's only 1-2 tokens wide (as
    # it is for the very first few query positions) leaves zero columns to
    # average over, producing NaN that poisons the whole per-layer mean via
    # np.mean. Starting at 10 guarantees at least 8 other causal-visible
    # columns remain at every query position used here.
    query_positions = list(range(10, SEQ_LEN))
    prevtoken_targets = [q - 1 for q in query_positions]

    for layer in range(n_layers):
        layer_attn = out.attentions[layer][0]  # (heads, q_len, k_len)
        prev_vals = layer_attn[:, query_positions, prevtoken_targets]  # (heads, n_queries)
        prevtoken_mass[layer] += prev_vals.mean(dim=-1).float().cpu().numpy()

        base_vals = []
        for qi, q in enumerate(query_positions):
            row = layer_attn[:, q, :q + 1].clone()  # (heads, q+1) causal-visible positions
            row[:, prevtoken_targets[qi]] = float("nan")
            row[:, q] = float("nan")
            base_vals.append(np.nanmean(row.float().cpu().numpy(), axis=-1))
        baseline_mass[layer] += np.mean(base_vals, axis=0)

prevtoken_mass /= N_TRIALS
baseline_mass /= N_TRIALS
positional_score = prevtoken_mass - baseline_mass

flat_df = pd.DataFrame([
    {"Layer": l, "Head": h, "PrevToken_Mass": prevtoken_mass[l, h],
     "Baseline_Mass": baseline_mass[l, h], "Positional_Score": positional_score[l, h]}
    for l in range(n_layers) for h in range(n_heads)
])
flat_df = flat_df.sort_values("Positional_Score", ascending=False).reset_index(drop=True)
flat_df["Positional_Rank"] = flat_df.index + 1  # 1 = strongest positional (previous-token) head
flat_df["Positional_Percentile"] = 100 * (1 - flat_df.index / (len(flat_df) - 1))
flat_df.to_csv(os.path.join(RESULTS_DIR, "positional_head_scores.csv"), index=False)

print("\n--- TOP 10 POSITIONAL (PREVIOUS-TOKEN) HEADS (by positional score) ---")
print(flat_df.head(10).round(4).to_string(index=False))

# ============================================================
# Compare against the boundary-attending target heads
# ============================================================
target_df = pd.read_csv(os.path.join(RESULTS_DIR, "causal_ablation_target_heads.csv"))
merged = target_df.merge(flat_df, on=["Layer", "Head"], how="left")
merged.to_csv(os.path.join(RESULTS_DIR, "positional_vs_boundary_heads.csv"), index=False)

print("\n--- BOUNDARY-ATTENDING TARGET HEADS: where do they rank on positional score? ---")
print(merged[["Layer", "Head", "Mass", "Positional_Score", "Positional_Rank", "Positional_Percentile"]].round(4).to_string(index=False))

n_total_heads = n_layers * n_heads
mean_percentile = merged["Positional_Percentile"].mean()
median_all = flat_df["Positional_Score"].median()
n_above_median = (merged["Positional_Score"] > median_all).sum()
print(f"\nMean positional-score percentile of the 10 boundary-attending heads: {mean_percentile:.1f} "
      f"(50.0 = indistinguishable from a random head; 100.0 = the single strongest positional head)")
print(f"{n_above_median}/10 boundary-attending heads score above the population median positional score.")
print(f"(Population: {n_total_heads} total heads, {N_TRIALS}-trial synthetic positional test)")

print("\nPositional-head overlap check complete.")
