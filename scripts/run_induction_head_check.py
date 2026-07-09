import matplotlib
matplotlib.use("Agg")

# ============================================================
# INDUCTION-HEAD OVERLAP CHECK
# ============================================================
# Reviewer question: do the 10 boundary-attending heads identified in
# run_causal_ablation.py (causal_ablation_target_heads.csv) overlap with
# "induction heads" (Olsson et al., arXiv:2209.11895) -- head pairs whose
# signature behavior is: given a repeated sequence ... X Y ... X, attend
# from the second X back to the position right after the FIRST X (i.e.
# copy Y). This is the standard operational test for induction-head
# behavior: build synthetic sequences of random token IDs, repeat a
# chunk immediately afterward, and measure how much attention each head
# places on the "induction target" position (the token right after the
# matching earlier occurrence) versus attention to other, non-matching
# earlier positions in the same sequence.
#
# We compute this induction score for all 784 heads (28 layers x 28
# heads, same model as the causal ablation) and report where the 10
# target heads rank in that full distribution -- if they cluster near
# the top, the boundary-attending heads are plausibly repurposed
# induction heads; if they're unremarkable, the two head populations are
# functionally distinct.

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
vocab_size = model.config.vocab_size

N_TRIALS = 20
CHUNK_LEN = 30  # length of the random chunk that gets repeated
# Avoid the first ~1000 ids (often special/control tokens) and the very
# top of the vocab (byte-fallback / added tokens); sample from a broad,
# safe middle range of ordinary subword token ids.
SAFE_ID_LO, SAFE_ID_HI = 2000, 100_000

induction_mass = np.zeros((n_layers, n_heads))
baseline_mass = np.zeros((n_layers, n_heads))

print(f"\nRunning {N_TRIALS} synthetic induction-test trials (chunk_len={CHUNK_LEN})...")
for trial in range(N_TRIALS):
    rng = random.Random(1000 + trial)
    chunk = [rng.randrange(SAFE_ID_LO, SAFE_ID_HI) for _ in range(CHUNK_LEN)]
    # sequence = chunk repeated twice back-to-back -> classic induction test
    seq = chunk + chunk
    input_ids = torch.tensor([seq], device=model.device)
    attn_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)

    # Query positions in the second copy: global index CHUNK_LEN .. 2*CHUNK_LEN-1
    # Induction target for query q (0-indexed global) is q - CHUNK_LEN + 1.
    query_positions = list(range(CHUNK_LEN, 2 * CHUNK_LEN))
    induction_targets = [q - CHUNK_LEN + 1 for q in query_positions]  # in [1, CHUNK_LEN]

    for layer in range(n_layers):
        layer_attn = out.attentions[layer][0]  # (heads, q_len, k_len)
        # Induction mass: attention from each query position to its specific induction target.
        ind_vals = layer_attn[:, query_positions, induction_targets]  # (heads, n_queries)
        induction_mass[layer] += ind_vals.mean(dim=-1).float().cpu().numpy()

        # Baseline: average attention from the same query positions to all OTHER
        # earlier positions (excluding the induction target itself and self-attention),
        # i.e. generic "attends somewhere back in the context" rate for this head.
        base_vals = []
        for qi, q in enumerate(query_positions):
            row = layer_attn[:, q, :q + 1].clone()  # (heads, q+1) causal-visible positions
            row[:, induction_targets[qi]] = float("nan")
            row[:, q] = float("nan")
            base_vals.append(np.nanmean(row.float().cpu().numpy(), axis=-1))
        baseline_mass[layer] += np.mean(base_vals, axis=0)

induction_mass /= N_TRIALS
baseline_mass /= N_TRIALS
induction_score = induction_mass - baseline_mass

flat_df = pd.DataFrame([
    {"Layer": l, "Head": h, "Induction_Mass": induction_mass[l, h],
     "Baseline_Mass": baseline_mass[l, h], "Induction_Score": induction_score[l, h]}
    for l in range(n_layers) for h in range(n_heads)
])
flat_df = flat_df.sort_values("Induction_Score", ascending=False).reset_index(drop=True)
flat_df["Induction_Rank"] = flat_df.index + 1  # 1 = strongest induction head
flat_df["Induction_Percentile"] = 100 * (1 - flat_df.index / (len(flat_df) - 1))
flat_df.to_csv(os.path.join(RESULTS_DIR, "induction_head_scores.csv"), index=False)

print("\n--- TOP 10 INDUCTION HEADS (by induction score) ---")
print(flat_df.head(10).round(4).to_string(index=False))

# ============================================================
# Compare against the boundary-attending target heads
# ============================================================
target_df = pd.read_csv(os.path.join(RESULTS_DIR, "causal_ablation_target_heads.csv"))
merged = target_df.merge(flat_df, on=["Layer", "Head"], how="left")
merged.to_csv(os.path.join(RESULTS_DIR, "induction_vs_boundary_heads.csv"), index=False)

print("\n--- BOUNDARY-ATTENDING TARGET HEADS: where do they rank on induction score? ---")
print(merged[["Layer", "Head", "Mass", "Induction_Score", "Induction_Rank", "Induction_Percentile"]].round(4).to_string(index=False))

n_total_heads = n_layers * n_heads
mean_percentile = merged["Induction_Percentile"].mean()
median_all = flat_df["Induction_Score"].median()
n_above_median = (merged["Induction_Score"] > median_all).sum()
print(f"\nMean induction-score percentile of the 10 boundary-attending heads: {mean_percentile:.1f} "
      f"(50.0 = indistinguishable from a random head; 100.0 = the single strongest induction head)")
print(f"{n_above_median}/10 boundary-attending heads score above the population median induction score.")
print(f"(Population: {n_total_heads} total heads, {N_TRIALS}-trial synthetic induction test)")

print("\nInduction-head overlap check complete.")
