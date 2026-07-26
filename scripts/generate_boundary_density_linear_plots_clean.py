#!/usr/bin/env python3
"""
Clean version of generate_boundary_density_linear_plots.py.

Differences from the original:
- Reads exactly one designated CSV per task (no wildcard glob that
  silently concatenates unrelated experiment variants: templated
  re-wordings, scratchpad conditions, carry-direction comparisons, etc.)
- Fixes an output-filename collision: the original used
  model_name.split('-')[0], which is "Qwen2.5" for both the 7B and 32B
  runs, so the second model's plot silently overwrote the first's.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("/group/gquongrp/workspaces/baran/AvalancheBreak/results")
OUTPUT_DIR = Path("/tmp/claude-2342120/-group-gquongrp-workspaces-baran-AvalancheBreak/dc5ef812-e9e0-469f-b195-1961d9d3c5af/scratchpad/clean_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = {
    'character_substitution': {
        'label': 'Character Substitution',
        'colors_by_n': {4: '#1f77b4', 8: '#ff7f0e', 16: '#2ca02c', 32: '#d62728', 64: '#9467bd'},
        'baseline_type': 'alphabet',
        'file_template': 'character_substitution_alphabet_sweep_{model}.csv',
    },
    'running_sum_local': {
        'label': 'Running-Sum-Mod-M',
        'colors_by_n': {4: '#1f77b4', 8: '#ff7f0e', 16: '#2ca02c', 32: '#d62728', 64: '#9467bd'},
        'baseline_type': 'modulus',
        'file_template': 'running_sum_local_modulus_sweep_{model}.csv',
    },
    'modsum': {
        'label': 'Modsum',
        'colors_by_n': {4: '#1f77b4', 8: '#ff7f0e', 16: '#2ca02c', 32: '#d62728', 64: '#9467bd'},
        'baseline_type': 'modulus',
        'file_template': 'modsum_modulus_sweep_{model}.csv',
    },
    'addition': {
        'label': 'Integer Addition',
        'colors_by_n': {4: '#1f77b4', 8: '#ff7f0e', 16: '#2ca02c', 32: '#d62728', 64: '#9467bd'},
        'baseline_type': 'digit_range',
        'file_template': 'addition_digit_range_sweep_{model}.csv',
    },
    'running_xor': {
        'label': 'Running-XOR',
        'colors_by_n': {4: '#1f77b4', 8: '#ff7f0e', 16: '#2ca02c', 32: '#d62728', 64: '#9467bd'},
        'baseline_type': 'binary',
        'file_template': 'running_xor_grid_structured_boundaries.csv',  # not model-specific
    },
}

def load_task_results(model_name, task_key):
    model_dir = RESULTS_DIR / model_name
    task = TASKS[task_key]
    csv_path = model_dir / task['file_template'].format(model=model_name)

    if not csv_path.exists():
        return None, None

    df = pd.read_csv(csv_path)
    if df.empty or 'Interval' not in df.columns or 'Seq_Length' not in df.columns or 'TAR' not in df.columns:
        return None, None

    df['Inv_K'] = df['Interval'].apply(lambda k: 1/k if k > 0 else 0)

    agg_data = []
    for length in sorted(df['Seq_Length'].unique()):
        for inv_k in sorted(df[df['Seq_Length'] == length]['Inv_K'].unique()):
            subset = df[(df['Seq_Length'] == length) & (df['Inv_K'] == inv_k)]

            tar_mean = subset['TAR'].mean()
            tar_std = subset['TAR'].std() / np.sqrt(len(subset))

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

    return pd.DataFrame(agg_data), csv_path.name

def get_baseline(task_key, param_level=None):
    task = TASKS[task_key]
    baseline_type = task['baseline_type']
    if baseline_type == 'alphabet':
        return 1.0 / (param_level if param_level else 2)
    elif baseline_type == 'modulus':
        return 1.0 / (param_level if param_level else 2)
    elif baseline_type == 'digit_range':
        return 1.0 / (10 ** (param_level if param_level else 1))
    elif baseline_type == 'binary':
        return 0.5
    return 0

def plot_task_boundary_density(model_label, task_key, task_info, agg_df):
    if agg_df is None or agg_df.empty:
        return None

    lengths = sorted(agg_df['length'].unique())
    colors = task_info['colors_by_n']

    fig, axes = plt.subplots(2, len(lengths), figsize=(4*len(lengths), 8))
    if len(lengths) == 1:
        axes = axes.reshape(2, 1)

    fit_report = []

    for col_idx, length in enumerate(lengths):
        subset = agg_df[agg_df['length'] == length].sort_values('inv_k')
        if subset.empty:
            continue

        color = colors.get(length, '#1f77b4')

        ax_tar = axes[0, col_idx]
        ax_tar.errorbar(subset['inv_k'], subset['tar_mean'], yerr=subset['tar_sem'],
                         marker='o', linewidth=2, markersize=8, color=color, alpha=0.8,
                         capsize=4, label='Data')

        if len(subset) >= 2:
            slope, intercept, r_value, p_value, std_err = linregress(subset['inv_k'], subset['tar_mean'])
            x_fit = np.linspace(0, subset['inv_k'].max() * 1.1, 100)
            y_fit = np.clip(slope * x_fit + intercept, 0, 1)
            ax_tar.plot(x_fit, y_fit, linestyle='--', color=color, alpha=0.6, linewidth=2, label='Fit')
            fit_report.append((length, 'TAR', slope, intercept, r_value**2))

        baseline = get_baseline(task_key, param_level=2)
        ax_tar.axhline(baseline, color='gray', linestyle=':', alpha=0.5, linewidth=2, label=f'Chance (1/{2})')

        ax_tar.set_xlabel('Boundary Density (1/k)', fontsize=11)
        ax_tar.set_ylabel('TAR', fontsize=11)
        ax_tar.set_title(f'TAR, N={length}', fontsize=12, fontweight='bold')
        ax_tar.set_ylim([0, 1.05])
        ax_tar.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_tar.legend(fontsize=9, loc='best')

        ax_sar = axes[1, col_idx]
        if not subset['sar_mean'].isna().all():
            ax_sar.errorbar(subset['inv_k'], subset['sar_mean'], yerr=subset['sar_sem'],
                             marker='s', linewidth=2, markersize=8, color=color, alpha=0.8,
                             capsize=4, label='Data')
            valid = ~subset['sar_mean'].isna()
            if valid.sum() >= 2:
                slope, intercept, r_value, p_value, std_err = linregress(
                    subset[valid]['inv_k'], subset[valid]['sar_mean'])
                x_fit = np.linspace(0, subset['inv_k'].max() * 1.1, 100)
                y_fit = np.clip(slope * x_fit + intercept, 0, 1)
                ax_sar.plot(x_fit, y_fit, linestyle='--', color=color, alpha=0.6, linewidth=2, label='Fit')
                fit_report.append((length, 'SAR', slope, intercept, r_value**2))
            ax_sar.set_ylim([0, 1.05])

        ax_sar.set_xlabel('Boundary Density (1/k)', fontsize=11)
        ax_sar.set_ylabel('SAR', fontsize=11)
        ax_sar.set_title(f'SAR, N={length}', fontsize=12, fontweight='bold')
        ax_sar.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_sar.legend(fontsize=9, loc='best')

    plt.suptitle(f'{task_info["label"]} - {model_label}', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_file = OUTPUT_DIR / f"{task_key}_{model_label}_linear.jpg"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return str(output_file), fit_report

def main():
    models = [('Qwen2.5-7B-Instruct', '7B'), ('Qwen2.5-32B-Instruct', '32B')]

    for model_name, model_label in models:
        print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
        for task_key, task_info in TASKS.items():
            agg_df, source_file = load_task_results(model_name, task_key)
            if agg_df is None:
                print(f"  {task_key}: NO DATA")
                continue
            output_file, fit_report = plot_task_boundary_density(model_label, task_key, task_info, agg_df)
            print(f"  {task_key}: source={source_file} -> {output_file}")

if __name__ == '__main__':
    main()
