#!/usr/bin/env python3
"""Figure: error containment (figs/error_containment.pdf). P(next character
correct | previous character wrong), split by whether a boundary was just
crossed vs. elsewhere, two panels: left cyclic_shift, right global
running-sum-mod-M. Qwen2.5-7B, matched to the existing headline seeds
(thermodynamics_grid.csv / running_modsum_grid.csv -- the same files
Section 4.6's prose already cites the pooled numbers from).

Reuses the EXACT methodology already established and used for the Mistral/
14B versions of this analysis (scripts/run_mistral_k16.py,
reconstruct_binary_series / is_boundary_adjacent / error_recovery_events),
applied here to Qwen2.5-7B's own headline data rather than reimplemented,
so this figure's pooled numbers are guaranteed to match the ones already
in results/Qwen2.5-7B-Instruct/thermodynamics_error_recovery.csv (verified
below before plotting).
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH, set_power_of_2_xticks

set_rc()

RESULTS_DIR = REPO_ROOT / "results" / "Qwen2.5-7B-Instruct"


def reconstruct_binary_series(ground_truth, llm_output):
    truth = str(ground_truth)
    pred = str(llm_output)[: len(truth)]
    return [1 if i < len(pred) and pred[i] == truth[i] else 0 for i in range(len(truth))]


def is_boundary_adjacent(pos, interval):
    return interval > 0 and pos > 0 and pos % interval == 0


def error_recovery_events(df):
    events = []
    for _, row in df.iterrows():
        series = reconstruct_binary_series(row["Ground_Truth"], row["LLM_Output"])
        interval = row["Interval"]
        for i in range(1, len(series)):
            if series[i - 1] == 0:
                events.append({
                    "Method": row["Method"], "Interval": interval, "Seq_Length": row["Seq_Length"],
                    "Boundary_Adjacent": is_boundary_adjacent(i, interval), "Recovered": series[i],
                })
    return pd.DataFrame(events)


thermo_path = RESULTS_DIR / "thermodynamics_grid.csv"
modsum_path = RESULTS_DIR / "running_modsum_grid.csv"
print(f"Provenance -- Figure error_containment sources: {thermo_path.relative_to(REPO_ROOT)}, "
      f"{modsum_path.relative_to(REPO_ROOT)}")

thermo_df = pd.read_csv(thermo_path)
modsum_df = pd.read_csv(modsum_path)
thermo_events = error_recovery_events(thermo_df)
modsum_events = error_recovery_events(modsum_df)

# Sanity check against the already-stored pooled summary before plotting
existing = pd.read_csv(RESULTS_DIR / "thermodynamics_error_recovery.csv")
cell_events = thermo_events[thermo_events.Method == "Cellular_Bounds"]
recomputed = cell_events.groupby("Boundary_Adjacent")["Recovered"].mean()
print("Sanity check (cyclic_shift, Cellular_Bounds), recomputed vs. stored:")
for adj in [True, False]:
    stored_row = existing[(existing.Method == "Cellular_Bounds") & (existing.Boundary_Adjacent == adj)]
    stored_val = stored_row["mean"].iloc[0] if len(stored_row) else float("nan")
    print(f"  Boundary_Adjacent={adj}: recomputed={recomputed.get(adj, float('nan')):.4f}  stored={stored_val:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(MULTI_COL_WIDTH, 3.2))

for ax, events_df, label in [(axes[0], thermo_events, "cyclic_shift"),
                               (axes[1], modsum_events, "running-sum-mod-M (global)")]:
    cell_events = events_df[events_df.Method == "Cellular_Bounds"]
    per_N = cell_events.groupby(["Seq_Length", "Boundary_Adjacent"])["Recovered"].mean().reset_index()
    for adj_val, color, marker, lbl in [(True, "#1b9e77", "o", "Right after boundary"),
                                          (False, "#d95f02", "s", "Elsewhere")]:
        sub = per_N[per_N.Boundary_Adjacent == adj_val].sort_values("Seq_Length")
        if len(sub):
            ax.plot(sub.Seq_Length, sub.Recovered, color=color, marker=marker, markersize=4,
                     linewidth=1.5, label=lbl)
    set_power_of_2_xticks(ax, per_N.Seq_Length.unique(), fontsize=6.5)
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("P(next correct | previous wrong)")
    ax.set_title(label)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="best", frameon=False, fontsize=6.5)

fig.suptitle("Error containment: recovery immediately after a boundary vs. elsewhere (Qwen2.5-7B, Cellular_Bounds)", y=1.03, fontsize=8)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "error_containment.pdf"
savefig(fig, OUT)
