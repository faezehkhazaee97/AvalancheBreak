#!/usr/bin/env python3
"""
Generate TAR/SAR vs Boundary Density plots with LINEAR REGRESSION FITS.

Key features:
- Plots: TAR and SAR vs boundary density (1/k)
- Linear regression fit lines (scipy.stats.linregress)
- **FIX**: Fitted lines CLIPPED to [0, 1] to prevent extrapolation artifacts
- Random-chance baseline shown as dotted horizontal line
- Error bars: ±1 SEM (Standard Error of Mean)
- Separate plots for each task and model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("/group/gquongrp/workspaces/baran/AvalancheBreak/results")
OUTPUT_DIR = Path("/group/gquongrp/workspaces/baran/AvalancheBreak/analysis_output/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Task configurations
TASKS = {
    'character_substitution': {
        'label': 'Character Substitution',
        'colors_by_n': {4: '#1f77b4', 8: '#ff7f0e', 16: '#2ca02c', 32: '#d62728', 64: '#9467bd'},
        'baseline_type': 'alphabet',  # 1/A
        'alphabet_sizes': [2, 4, 8],
    },
    'running_sum_local': {
        'label': 'Running-Sum-Mod-M',
        'colors_by_n': {4: '#1f77b4', 8: '#ff7f0e', 16: '#2ca02c', 32: '#d62728', 64: '#9467bd'},
        'baseline_type': 'modulus',  # 1/M
        'moduli': [2, 4, 8],
    },
    'modsum': {
        'label': 'Modsum',
        'colors_by_n': {4: '#1f77b4', 8: '#ff7f0e', 16: '#2ca02c', 32: '#d62728', 64: '#9467bd'},
        'baseline_type': 'modulus',
        'moduli': [2, 4, 8],
    },
    'addition': {
        'label': 'Integer Addition',
        'colors_by_n': {4: '#1f77b4', 8: '#ff7f0e', 16: '#2ca02c', 32: '#d62728', 64: '#9467bd'},
        'baseline_type': 'digit_range',  # 1/10^D
        'digit_ranges': [1, 2, 3],
    },
    'running_xor': {
        'label': 'Running-XOR',
        'colors_by_n': {4: '#1f77b4', 8: '#ff7f0e', 16: '#2ca02c', 32: '#d62728', 64: '#9467bd'},
        'baseline_type': 'binary',  # 0.5
    },
}

def load_task_results(model_name, task_key):
    """Load results for a task, aggregate by length/interval."""
    model_dir = RESULTS_DIR / model_name

    # Find matching CSV files
    results = []
    for csv_file in model_dir.glob(f"*{task_key}*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if df.empty or 'Interval' not in df.columns or 'Seq_Length' not in df.columns:
                continue
            if 'TAR' not in df.columns:
                continue

            results.append(df)
        except:
            pass

    if not results:
        return None

    df = pd.concat(results, ignore_index=True)

    # Compute 1/k (boundary density), treating interval=0 as density=0
    df['Inv_K'] = df['Interval'].apply(lambda k: 1/k if k > 0 else 0)

    # Aggregate by Seq_Length and Inv_K
    agg_data = []
    for length in sorted(df['Seq_Length'].unique()):
        for inv_k in sorted(df[df['Seq_Length'] == length]['Inv_K'].unique()):
            subset = df[(df['Seq_Length'] == length) & (df['Inv_K'] == inv_k)]

            tar_mean = subset['TAR'].mean()
            tar_std = subset['TAR'].std() / np.sqrt(len(subset))  # SEM

            sar_mean = subset['SAR'].mean() if 'SAR' in subset.columns else np.nan
            sar_std = subset['SAR'].std() / np.sqrt(len(subset)) if 'SAR' in subset.columns else np.nan

            agg_data.append({
                'length': int(length),
                'inv_k': inv_k,
                'tar_mean': tar_mean,
                'tar_sem': tar_std,
                'sar_mean': sar_mean,
                'sar_sem': sar_std,
                'n_trials': len(subset),
            })

    return pd.DataFrame(agg_data)

def get_baseline(task_key, param_level=None):
    """Get random-chance baseline for a task."""
    task = TASKS[task_key]
    baseline_type = task['baseline_type']

    if baseline_type == 'alphabet':
        # Position-independent: 1/A (A=2,4,8)
        return 1.0 / (param_level if param_level else 2)
    elif baseline_type == 'modulus':
        # Modular: 1/M
        return 1.0 / (param_level if param_level else 2)
    elif baseline_type == 'digit_range':
        # Addition: 1/10^D
        return 1.0 / (10 ** (param_level if param_level else 1))
    elif baseline_type == 'binary':
        # XOR: 0.5
        return 0.5
    return 0

def plot_task_boundary_density(model_name, task_key, task_info, agg_df):
    """Create TAR/SAR vs boundary density plot for a task."""

    if agg_df is None or agg_df.empty:
        return None

    lengths = sorted(agg_df['length'].unique())
    colors = task_info['colors_by_n']

    # Create figure with TAR and SAR subplots
    fig, axes = plt.subplots(2, len(lengths), figsize=(4*len(lengths), 8))
    if len(lengths) == 1:
        axes = axes.reshape(2, 1)

    for col_idx, length in enumerate(lengths):
        subset = agg_df[agg_df['length'] == length].sort_values('inv_k')

        if subset.empty:
            continue

        color = colors.get(length, '#1f77b4')

        # --- TAR subplot ---
        ax_tar = axes[0, col_idx]

        # Data points with error bars
        ax_tar.errorbar(subset['inv_k'], subset['tar_mean'],
                       yerr=subset['tar_sem'],
                       marker='o', linewidth=2, markersize=8,
                       color=color, alpha=0.8, capsize=4, label='Data')

        # Linear regression fit (CLIPPED to [0, 1])
        if len(subset) >= 2:
            slope, intercept, r_value, p_value, std_err = linregress(subset['inv_k'], subset['tar_mean'])
            x_fit = np.linspace(0, subset['inv_k'].max() * 1.1, 100)
            y_fit = np.clip(slope * x_fit + intercept, 0, 1)  # **FIX: CLIP HERE**
            ax_tar.plot(x_fit, y_fit, linestyle='--', color=color, alpha=0.6, linewidth=2, label='Fit')

        # Random-chance baseline
        baseline = get_baseline(task_key, param_level=2)
        ax_tar.axhline(baseline, color='gray', linestyle=':', alpha=0.5, linewidth=2, label=f'Chance (1/{2})')

        ax_tar.set_xlabel('Boundary Density (1/k)', fontsize=11)
        ax_tar.set_ylabel('TAR', fontsize=11)
        ax_tar.set_title(f'TAR, N={length}', fontsize=12, fontweight='bold')
        ax_tar.set_ylim([0, 1.05])
        ax_tar.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_tar.legend(fontsize=9, loc='best')

        # --- SAR subplot ---
        ax_sar = axes[1, col_idx]

        # Data points with error bars
        if not subset['sar_mean'].isna().all():
            ax_sar.errorbar(subset['inv_k'], subset['sar_mean'],
                           yerr=subset['sar_sem'],
                           marker='s', linewidth=2, markersize=8,
                           color=color, alpha=0.8, capsize=4, label='Data')

            # Linear regression fit (CLIPPED to [0, 1])
            valid = ~subset['sar_mean'].isna()
            if valid.sum() >= 2:
                slope, intercept, r_value, p_value, std_err = linregress(
                    subset[valid]['inv_k'], subset[valid]['sar_mean'])
                x_fit = np.linspace(0, subset['inv_k'].max() * 1.1, 100)
                y_fit = np.clip(slope * x_fit + intercept, 0, 1)  # **FIX: CLIP HERE**
                ax_sar.plot(x_fit, y_fit, linestyle='--', color=color, alpha=0.6, linewidth=2, label='Fit')

            ax_sar.set_ylim([0, 1.05])

        ax_sar.set_xlabel('Boundary Density (1/k)', fontsize=11)
        ax_sar.set_ylabel('SAR', fontsize=11)
        ax_sar.set_title(f'SAR, N={length}', fontsize=12, fontweight='bold')
        ax_sar.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_sar.legend(fontsize=9, loc='best')

    plt.suptitle(f'{task_info["label"]} - {model_name}', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    output_file = OUTPUT_DIR / f"{task_key}_{model_name.split('-')[1]}_linear.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return str(output_file)

def main():
    """Generate all plots."""
    print("Generating boundary-density linear regression plots...")
    print("(with TAR/SAR fits CLIPPED to [0,1] to prevent extrapolation artifacts)\n")

    models = ['Qwen2.5-7B-Instruct', 'Qwen2.5-32B-Instruct']

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        for task_key, task_info in TASKS.items():
            print(f"  {task_key}...", end=' ', flush=True)

            agg_df = load_task_results(model_name, task_key)

            if agg_df is None:
                print("✗ (no data)")
                continue

            output_file = plot_task_boundary_density(model_name, task_key, task_info, agg_df)
            print(f"✓ ({output_file})")

    print(f"\n✓ All plots saved to {OUTPUT_DIR}/\n")

if __name__ == '__main__':
    main()
