import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

RESULTS_DIR = "results/Qwen2.5-7B-Instruct"

# Load all 3 task results
char_sub = pd.read_csv(os.path.join(RESULTS_DIR, "character_substitution_grid_structured_boundaries.csv"))
running_xor = pd.read_csv(os.path.join(RESULTS_DIR, "running_xor_grid_structured_boundaries.csv"))
running_sum_local = pd.read_csv(os.path.join(RESULTS_DIR, "running_sum_local_grid_structured_boundaries.csv"))

print("=" * 80)
print("ANALYSIS: CHARACTER SUBSTITUTION, RUNNING XOR, RUNNING SUM (LOCAL)")
print("=" * 80)

for task_name, df in [("Character Substitution", char_sub),
                       ("Running XOR", running_xor),
                       ("Running Sum (Local)", running_sum_local)]:
    print(f"\n{'='*80}")
    print(f"TASK: {task_name}")
    print(f"{'='*80}")

    # Basic stats
    print(f"\nShape: {df.shape[0]} trials across grid")
    print(f"Sequence lengths: {sorted(df['Seq_Length'].unique())}")
    print(f"Boundary intervals: {sorted(df['Interval'].unique())}")

    # TAR by sequence length (Base vs Cellular)
    print(f"\n--- TAR by Sequence Length ---")
    tar_by_length = df.groupby(['Seq_Length', 'Method'])['TAR'].agg(['mean', 'std', 'count'])
    print(tar_by_length)

    # Effect of boundaries: k=0 (Base) vs best boundary interval
    print(f"\n--- Boundary Effect ---")
    base_tar = df[df['Interval'] == 0].groupby('Seq_Length')['TAR'].mean()
    cellular_tar = df[df['Interval'] > 0].groupby('Seq_Length')['TAR'].mean()

    for n in sorted(df['Seq_Length'].unique()):
        base = df[(df['Interval'] == 0) & (df['Seq_Length'] == n)]['TAR'].mean()
        cellular_best = df[(df['Interval'] > 0) & (df['Seq_Length'] == n)]['TAR'].max()
        interval_best = df[(df['Seq_Length'] == n) & (df['TAR'] == cellular_best)]['Interval'].iloc[0]
        effect = cellular_best - base
        print(f"N={n:2d}: Base={base:.3f}, Best Cellular (k={interval_best})={cellular_best:.3f}, Δ={effect:+.3f}")

    # TAR by boundary interval (all lengths)
    print(f"\n--- TAR by Boundary Interval (averaged over all N) ---")
    tar_by_interval = df[df['Interval'] > 0].groupby('Interval')['TAR'].agg(['mean', 'std'])
    print(tar_by_interval)

    # SAR (Sequence Accuracy)
    print(f"\n--- SAR (Sequence Accuracy) by Sequence Length ---")
    sar_by_length = df.groupby(['Seq_Length', 'Method'])['SAR'].agg(['mean', 'std'])
    print(sar_by_length)

    # Error recovery: P(correct at position i | error at position i-1)
    print(f"\n--- Error Recovery Analysis ---")
    errors_by_pos = []
    for n in sorted(df['Seq_Length'].unique())[:2]:  # Just first 2 lengths for clarity
        subset = df[df['Seq_Length'] == n]
        for _, row in subset.iterrows():
            output = str(row['LLM_Output'])
            truth = str(row['Ground_Truth'])

            # Remove boundaries for comparison
            output_clean = output.replace('|', '')
            truth_clean = truth.replace('|', '')

            if len(output_clean) == len(truth_clean):
                for i in range(1, len(truth_clean)):
                    if output_clean[i-1] != truth_clean[i-1]:  # Previous was wrong
                        correct_here = 1 if output_clean[i] == truth_clean[i] else 0
                        errors_by_pos.append({
                            'N': n,
                            'Interval': row['Interval'],
                            'at_boundary': (i % row['Interval'] == 0) if row['Interval'] > 0 else False,
                            'recovered': correct_here
                        })

    if errors_by_pos:
        errors_df = pd.DataFrame(errors_by_pos)
        recovery_after_boundary = errors_df[errors_df['at_boundary']]['recovered'].mean()
        recovery_elsewhere = errors_df[~errors_df['at_boundary']]['recovered'].mean()
        print(f"Recovery right after boundary: {recovery_after_boundary:.3f}")
        print(f"Recovery elsewhere: {recovery_elsewhere:.3f}")
        print(f"Difference: {recovery_after_boundary - recovery_elsewhere:+.3f}")

    # First error position analysis
    print(f"\n--- First Error Position (where accuracy breaks) ---")
    first_error_by_length = df.groupby(['Seq_Length', 'Interval'])['First_Error_Pos'].mean().reset_index()
    print(first_error_by_length.pivot(index='Seq_Length', columns='Interval', values='First_Error_Pos'))

print(f"\n{'='*80}")
print("CROSS-TASK COMPARISON")
print(f"{'='*80}")

# Summary table
summary_data = []
for task_name, df in [("Char Sub", char_sub), ("Running XOR", running_xor), ("Running Sum Local", running_sum_local)]:
    base_tar = df[df['Interval'] == 0]['TAR'].mean()
    cellular_tar = df[df['Interval'] > 0]['TAR'].mean()
    best_tar = df['TAR'].max()
    worst_tar = df['TAR'].min()

    summary_data.append({
        'Task': task_name,
        'Base TAR': f"{base_tar:.3f}",
        'Cellular TAR': f"{cellular_tar:.3f}",
        'Best TAR': f"{best_tar:.3f}",
        'Worst TAR': f"{worst_tar:.3f}",
        'Boundary Effect': f"{cellular_tar - base_tar:+.3f}"
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

# Determine pattern for each task
for task_name, df in [("Character Substitution", char_sub),
                       ("Running XOR", running_xor),
                       ("Running Sum (Local)", running_sum_local)]:
    base_avg = df[df['Interval'] == 0]['TAR'].mean()
    cellular_avg = df[df['Interval'] > 0]['TAR'].mean()

    if cellular_avg > base_avg:
        pattern = "BOUNDARIES HELP ✓"
    elif cellular_avg < base_avg:
        pattern = "BOUNDARIES HURT ✗"
    else:
        pattern = "NO SIGNIFICANT EFFECT"

    print(f"\n{task_name}: {pattern}")
    print(f"  Base={base_avg:.3f}, Cellular={cellular_avg:.3f}, Δ={cellular_avg - base_avg:+.3f}")
