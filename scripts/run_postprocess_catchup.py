import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress, wilcoxon, binomtest
import seaborn as sns

# ============================================================
# CPU-ONLY CATCH-UP: analyses that only need CSVs already on disk
# ============================================================
# 14B and Mistral ran on frozen snapshots of the notebook and are missing
# some of the later reviewer-driven analyses. Several of those don't need
# a new GPU generation pass at all -- they just weren't backfilled onto the
# older result directories. This script fills in exactly those, for
# whichever model directories already have the prerequisite CSVs:
#   - clustering_vs_density_regression.csv   (needs attention_metrics.csv)
#   - boundary_ablation_significance.csv     (needs boundary_ablation_grid.csv)
# Both models' attention_metrics.csv already has Base_Clustering /
# Cell_Clustering / Delta_Clustering / Inv_K columns, so clustering runs for
# both. Mistral already has boundary_ablation_grid.csv (from its full run),
# so its significance test can run now too; 14B's ablation grid doesn't
# exist yet (needs the GPU catch-up job) so that one is skipped here and
# handled inside run_14b_catchup.py instead.

RESULTS_ROOT = "results"


def run_clustering(model_tag):
    results_dir = os.path.join(RESULTS_ROOT, model_tag)
    metrics_path = os.path.join(results_dir, "attention_metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"[{model_tag}] SKIP clustering: no attention_metrics.csv")
        return

    df = pd.read_csv(metrics_path)
    print(f"\n=== [{model_tag}] DELTA CLUSTERING (Cellular - Base) BY SEQ_LENGTH ===")
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
    plot_path = os.path.join(results_dir, "clustering_vs_density.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {plot_path}")

    reg_df = pd.DataFrame(regression_results)
    print(f"\n--- [{model_tag}] DELTA CLUSTERING ~ BOUNDARY DENSITY REGRESSION ---")
    print(reg_df.round(4).to_string(index=False))
    reg_df.to_csv(os.path.join(results_dir, "clustering_vs_density_regression.csv"), index=False)

    n_negative = (reg_df["Slope"] < 0).sum()
    print(f"[{model_tag}] Slope is NEGATIVE at {n_negative}/{len(reg_df)} tested lengths.")


def mcnemar_exact_pvalue(b, c):
    if b + c == 0:
        return 1.0
    return binomtest(min(b, c), b + c, 0.5).pvalue


def paired_ablation_test(df, cond_a, cond_b, label_a, label_b):
    a_df = df[(df['Marker'] == cond_a[0]) & (df['Placement'] == cond_a[1])]
    b_df = df[(df['Marker'] == cond_b[0]) & (df['Placement'] == cond_b[1])]
    results = []
    for N in sorted(df['Seq_Length'].unique()):
        a_N = a_df[a_df['Seq_Length'] == N]
        b_N = b_df[b_df['Seq_Length'] == N]
        merged = a_N[['Trial', 'TAR', 'SAR']].merge(b_N[['Trial', 'TAR', 'SAR']], on='Trial', suffixes=('_a', '_b'))
        if len(merged) < 2:
            continue

        diffs = merged['TAR_b'] - merged['TAR_a']
        p_tar = wilcoxon(merged['TAR_b'], merged['TAR_a'])[1] if np.any(diffs != 0) else 1.0

        a_pass = merged['SAR_a'] >= 0.5
        b_pass = merged['SAR_b'] >= 0.5
        b_disc = int((a_pass & ~b_pass).sum())
        c_disc = int((~a_pass & b_pass).sum())
        p_sar = mcnemar_exact_pvalue(b_disc, c_disc)

        results.append({
            'Comparison': f'{label_a} vs {label_b}', 'Seq_Length': N, 'N_pairs': len(merged),
            'Mean_TAR_A': merged['TAR_a'].mean(), 'Mean_TAR_B': merged['TAR_b'].mean(),
            'P_value_TAR': p_tar, 'P_value_SAR': p_sar,
            'Significant_TAR_p05': p_tar < 0.05, 'Significant_SAR_p05': p_sar < 0.05,
        })
    return pd.DataFrame(results)


def run_ablation_significance(model_tag):
    results_dir = os.path.join(RESULTS_ROOT, model_tag)
    ablation_path = os.path.join(results_dir, "boundary_ablation_grid.csv")
    if not os.path.exists(ablation_path):
        print(f"[{model_tag}] SKIP ablation significance: no boundary_ablation_grid.csv yet")
        return

    ablation_df = pd.read_csv(ablation_path)
    comparisons = [
        (("|", "periodic"), (";", "periodic"), "| baseline", "; marker"),
        (("|", "periodic"), ("#", "periodic"), "| baseline", "# marker"),
        (("|", "periodic"), ("|", "random"), "| periodic", "| random"),
        (("|", "periodic"), ("|", "periodic_fixed_count"), "| density", "| fixed-count"),
    ]

    all_results = []
    for cond_a, cond_b, label_a, label_b in comparisons:
        res = paired_ablation_test(ablation_df, cond_a, cond_b, label_a, label_b)
        all_results.append(res)
        print(f"\n--- [{model_tag}] {label_a} vs {label_b} ---")
        print(res.round(4).to_string(index=False))

    sig_df = pd.concat(all_results, ignore_index=True)
    sig_df.to_csv(os.path.join(results_dir, "boundary_ablation_significance.csv"), index=False)

    n_sig_tar = sig_df['Significant_TAR_p05'].sum()
    n_sig_sar = sig_df['Significant_SAR_p05'].sum()
    print(f"\n[{model_tag}] Significant at p<0.05 across all ablation comparisons: "
          f"TAR {n_sig_tar}/{len(sig_df)}, SAR {n_sig_sar}/{len(sig_df)}")


for tag in ["Qwen2.5-14B-Instruct", "Mistral-7B-Instruct-v0.3"]:
    run_clustering(tag)
    run_ablation_significance(tag)

print("\nPost-process catch-up complete.")
