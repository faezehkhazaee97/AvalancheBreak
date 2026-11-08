import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns

# ============================================================
# CLUSTERING-COEFFICIENT CHECK, Qwen2.5-32B-Instruct (CPU-only)
# ============================================================
# Same computation as run_postprocess_catchup.py's run_clustering(), which
# backfilled Qwen2.5-14B and Mistral-7B from their existing attention_metrics.csv.
# 32B has no bare attention_metrics.csv (only the chat-templated variant),
# so this reads attention_metrics_templated.csv instead. Verified first that
# this substitution doesn't change the finding: recomputing 7B's clustering
# regression from ITS OWN attention_metrics_templated.csv (rather than the
# bare file the existing results/Qwen2.5-7B-Instruct/clustering_vs_density_regression.csv
# was built from) reproduces the same pattern -- negative slope, R^2 0.64-0.70,
# N=8 not significant (p=0.24 templated vs. p=0.56 bare), every other length
# p<1e-13 -- so using the templated file for 32B is consistent with, not a
# departure from, the other three models' results.

MODEL_TAG = "Qwen2.5-32B-Instruct"
RESULTS_DIR = os.path.join("results", MODEL_TAG)
metrics_path = os.path.join(RESULTS_DIR, "attention_metrics_templated.csv")

df = pd.read_csv(metrics_path)
print(f"Loaded {metrics_path}: {len(df)} rows")
print(f"\n=== [{MODEL_TAG}] DELTA CLUSTERING (Cellular - Base) BY SEQ_LENGTH ===")
summary = df.groupby("Seq_Length")[["Base_Clustering", "Cell_Clustering", "Delta_Clustering"]].mean()
print(summary.round(4).to_string())

sns.set_theme(style="whitegrid", font_scale=1.1)
seq_lengths = sorted(df["Seq_Length"].unique())
fig, axes = plt.subplots(1, len(seq_lengths), figsize=(3.2 * len(seq_lengths), 5), sharey=True)

regression_results = []
for i, N in enumerate(seq_lengths):
    ax = axes[i] if len(seq_lengths) > 1 else axes
    sub = df[df["Seq_Length"] == N]
    ax.scatter(sub["Inv_K"], sub["Delta_Clustering"], color="#7b3fa0", alpha=0.35, s=25)

    if sub["Inv_K"].nunique() >= 2:
        slope, intercept, r_value, p_value, std_err = linregress(sub["Inv_K"], sub["Delta_Clustering"])
        regression_results.append({"Seq_Length": N, "Slope": slope, "P_value": p_value, "R_squared": r_value**2})
        x_fit = np.linspace(0, sub["Inv_K"].max() * 1.1, 100)
        ax.plot(x_fit, slope * x_fit + intercept, linestyle="--", color="#7b3fa0", linewidth=1.5)

    ax.axhline(0.0, color="black", linestyle="-", alpha=0.3, linewidth=1)
    ax.set_title(f"N={N}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.set_ylabel("Delta Clustering (Cellular - Base)", fontsize=12)

fig.supxlabel("$1/k$ (boundary frequency)", fontsize=12)
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "clustering_vs_density.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved plot to {plot_path}")

reg_df = pd.DataFrame(regression_results)
print(f"\n--- [{MODEL_TAG}] DELTA CLUSTERING ~ BOUNDARY DENSITY REGRESSION ---")
print(reg_df.round(4).to_string(index=False))
reg_df.to_csv(os.path.join(RESULTS_DIR, "clustering_vs_density_regression.csv"), index=False)

n_negative = (reg_df["Slope"] < 0).sum()
print(f"[{MODEL_TAG}] Slope is NEGATIVE at {n_negative}/{len(reg_df)} tested lengths.")
print("\nClustering check (32B) complete.")
