#!/usr/bin/env python3
"""Figure rebuild: error containment, Qwen2.5-32B-Instruct (was Qwen-7B).
P(next character correct | previous character wrong), split by whether a
boundary was just crossed vs. elsewhere. Left: cyclic_shift. Right:
modsum (labeled "running-sum-mod-M (global)" -- a genuinely different task
from running_sum_local, note the name).

Also checks the pooling question directly: the original script pools
every alphabet/modulus together with no split. This version reports BOTH
the pooled version (kept as the main panel, for continuity) AND prints a
per-alphabet/per-modulus breakdown so we know whether pooling was hiding
real differences before trusting it.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tasks.common.figure_style import set_rc, savefig, MULTI_COL_WIDTH

set_rc()

RESULTS_DIR = REPO_ROOT / "results" / "Qwen2.5-32B-Instruct"


def reconstruct_binary_series(ground_truth, llm_output):
    truth = str(ground_truth)
    pred = str(llm_output)[: len(truth)]
    return [1 if i < len(pred) and pred[i] == truth[i] else 0 for i in range(len(truth))]


def is_boundary_adjacent(pos, interval):
    return interval > 0 and pos > 0 and pos % interval == 0


def error_recovery_events(df, axis_col):
    events = []
    for _, row in df.iterrows():
        series = reconstruct_binary_series(row["Ground_Truth"], row["LLM_Output"])
        interval = row["Interval"]
        axis_val = row[axis_col] if axis_col else None
        for i in range(1, len(series)):
            if series[i - 1] == 0:
                events.append({
                    "Method": row["Method"], "Interval": interval, "Seq_Length": row["Seq_Length"],
                    "Axis_Val": axis_val,
                    "Boundary_Adjacent": is_boundary_adjacent(i, interval), "Recovered": series[i],
                })
    return pd.DataFrame(events)


thermo_path = RESULTS_DIR / "thermodynamics_grid_templated_v2.csv"
modsum_path = RESULTS_DIR / "running_modsum_grid_templated_v2.csv"
print(f"Provenance: {thermo_path.relative_to(REPO_ROOT)}, {modsum_path.relative_to(REPO_ROOT)}")

thermo_df = pd.read_csv(thermo_path)
modsum_df = pd.read_csv(modsum_path)
thermo_events = error_recovery_events(thermo_df, "Alphabet_Size")
modsum_events = error_recovery_events(modsum_df, "Modulus")

# --- Check the pooling question: does the boundary-adjacent effect hold
#     up the same way split by alphabet/modulus, or does pooling hide
#     real differences? ---
print("\n=== POOLING CHECK: cyclic_shift, per-alphabet ===")
for a in sorted(thermo_events.Axis_Val.dropna().unique()):
    sub = thermo_events[(thermo_events.Method == "Cellular_Bounds") & (thermo_events.Axis_Val == a)]
    g = sub.groupby("Boundary_Adjacent")["Recovered"].mean()
    print(f"  Alphabet={a}: adjacent={g.get(True, float('nan')):.3f}  elsewhere={g.get(False, float('nan')):.3f}  "
          f"diff={g.get(True, float('nan')) - g.get(False, float('nan')):+.3f}")

print("\n=== POOLING CHECK: modsum, per-modulus ===")
for m in sorted(modsum_events.Axis_Val.dropna().unique()):
    sub = modsum_events[(modsum_events.Method == "Cellular_Bounds") & (modsum_events.Axis_Val == m)]
    g = sub.groupby("Boundary_Adjacent")["Recovered"].mean()
    print(f"  Modulus={m}: adjacent={g.get(True, float('nan')):.3f}  elsewhere={g.get(False, float('nan')):.3f}  "
          f"diff={g.get(True, float('nan')) - g.get(False, float('nan')):+.3f}")

fig, axes = plt.subplots(1, 2, figsize=(MULTI_COL_WIDTH, 3.2))

for ax, events_df, label in [(axes[0], thermo_events, "cyclic_shift"),
                               (axes[1], modsum_events, "modsum (global full-prefix dependency)")]:
    cell_events = events_df[events_df.Method == "Cellular_Bounds"]
    per_N = cell_events.groupby(["Seq_Length", "Boundary_Adjacent"])["Recovered"].mean().reset_index()
    for adj_val, color, marker, lbl in [(True, "#1b9e77", "o", "Right after boundary"),
                                          (False, "#eb6834", "s", "Elsewhere")]:
        sub = per_N[per_N.Boundary_Adjacent == adj_val].sort_values("Seq_Length")
        if len(sub):
            ax.plot(sub.Seq_Length, sub.Recovered, color=color, marker=marker, markersize=4,
                     linewidth=1.5, label=lbl)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("P(next correct | previous wrong)")
    ax.set_title(label)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="best", frameon=False, fontsize=6.5)

fig.suptitle("Error containment: recovery immediately after a boundary vs. elsewhere\n"
             "(Qwen2.5-32B-Instruct, Cellular_Bounds, pooled across alphabet/modulus -- see console for per-axis breakdown)",
             y=1.08, fontsize=8)
fig.tight_layout()

OUT = REPO_ROOT / "figs" / "error_containment_32b.pdf"
savefig(fig, OUT)
fig.savefig(str(OUT).replace("figs/", "figs/png/").replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
