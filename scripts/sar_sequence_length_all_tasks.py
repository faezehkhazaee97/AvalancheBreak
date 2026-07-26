#!/usr/bin/env python3
"""
SAR vs Sequence Length for all 6 tasks
Replicates the cyclic_vs_modsum comparison style across all tasks
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

RESULTS_DIR = Path("/group/gquongrp/workspaces/baran/AvalancheBreak/results")
OUTPUT_DIR = Path("/group/gquongrp/workspaces/baran/AvalancheBreak/analysis_output/plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Task info
TASKS = {
    'character_substitution': {
        'label': 'Character Substitution',
        'tier': 'Tier 1 (Position-Local)',
        'color': '#e74c3c',
    },
    'running_xor': {
        'label': 'Running XOR',
        'tier': 'Tier 2 (Single-Previous, Self-Correct)',
        'color': '#9b59b6',
    },
    'running_sum_local': {
        'label': 'Running Sum Local',
        'tier': 'Tier 2 (Single-Previous, Cascading)',
        'color': '#3498db',
    },
    'modsum': {
        'label': 'Modsum',
        'tier': 'Tier 3 (Global-State)',
        'color': '#f39c12',
    },
    'addition': {
        'label': 'Integer Addition',
        'tier': 'Tier 3 (Global-State)',
        'color': '#1abc9c',
    },
}

def load_task_data(model_name, task_key):
    """Load all result files for a task and extract SAR by sequence length."""
    model_dir = RESULTS_DIR / model_name
    results = []

    # Look for any file matching the task
    for csv_file in model_dir.glob(f"*{task_key}*.csv"):
        try:
            df = pd.read_csv(csv_file)

            # Skip empty files
            if df.empty or 'Seq_Length' not in df.columns or 'SAR' not in df.columns:
                continue

            # Group by sequence length
            for length in sorted(df['Seq_Length'].unique()):
                subset = df[df['Seq_Length'] == length]
                results.append({
                    'length': int(length),
                    'sar_mean': subset['SAR'].mean(),
                    'sar_std': subset['SAR'].std(),
                    'n': len(subset),
                })
        except Exception as e:
            print(f"  Error reading {csv_file.name}: {e}")

    return pd.DataFrame(results)

def plot_sar_sequence_length_individual():
    """Create individual SAR vs length plots for each task."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    task_list = list(TASKS.keys())

    for idx, task_key in enumerate(task_list):
        ax = axes[idx]
        task_info = TASKS[task_key]

        # Load 7B data
        df_7b = load_task_data('Qwen2.5-7B-Instruct', task_key)

        if df_7b.empty:
            ax.text(0.5, 0.5, f'No data for {task_key}', ha='center', va='center')
            ax.set_title(task_info['label'])
            continue

        df_7b = df_7b.sort_values('length')

        # Plot with error bars
        ax.errorbar(df_7b['length'], df_7b['sar_mean'] * 100,
                   yerr=df_7b['sar_std'] * 100,
                   marker='o', linewidth=2, markersize=8, capsize=5,
                   color=task_info['color'], label='7B')

        # Load 32B if available
        df_32b = load_task_data('Qwen2.5-32B-Instruct', task_key)
        if not df_32b.empty:
            df_32b = df_32b.sort_values('length')
            ax.errorbar(df_32b['length'], df_32b['sar_mean'] * 100,
                       yerr=df_32b['sar_std'] * 100,
                       marker='s', linewidth=2, markersize=8, capsize=5,
                       color=task_info['color'], alpha=0.5, linestyle='--', label='32B')

        ax.set_xlabel('Sequence Length', fontsize=11, fontweight='bold')
        ax.set_ylabel('SAR (%)', fontsize=11, fontweight='bold')
        ax.set_title(f"{task_info['label']}\n{task_info['tier']}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_ylim([0, 100])

    # Hide extra subplot
    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sar_sequence_length_all_tasks.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: sar_sequence_length_all_tasks.png")
    plt.close()

def plot_sar_sequence_length_comparison():
    """Create a single comprehensive plot comparing all tasks (like cyclic vs modsum)."""

    fig, ax = plt.subplots(figsize=(14, 8))

    task_list = list(TASKS.keys())
    colors_cycle = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c', '#2ecc71']

    for idx, task_key in enumerate(task_list):
        task_info = TASKS[task_key]

        # Load 7B data (primary)
        df_7b = load_task_data('Qwen2.5-7B-Instruct', task_key)

        if df_7b.empty:
            continue

        df_7b = df_7b.sort_values('length')

        # Plot with error bars
        ax.errorbar(df_7b['length'], df_7b['sar_mean'] * 100,
                   yerr=df_7b['sar_std'] * 100,
                   marker='o', linewidth=2.5, markersize=9, capsize=6,
                   color=colors_cycle[idx], label=task_info['label'], alpha=0.8)

    ax.set_xlabel('Sequence Length (N)', fontsize=13, fontweight='bold')
    ax.set_ylabel('SAR - 1/k slope (mean across groups)', fontsize=13, fontweight='bold')
    ax.set_title('Does the boundary effect generalize across task types?\n(7B Models)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.set_ylim([-5, 105])

    # Use log scale for x-axis like the original
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sar_all_tasks_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: sar_all_tasks_comparison.png")
    plt.close()

def generate_sar_table():
    """Generate table showing SAR degradation across sequence lengths."""

    print("\n" + "="*100)
    print("SAR (Sequence Accuracy Rate) vs Sequence Length - All Tasks")
    print("="*100 + "\n")

    for task_key, task_info in TASKS.items():
        df_7b = load_task_data('Qwen2.5-7B-Instruct', task_key)

        if df_7b.empty:
            print(f"❌ {task_info['label']}: No data\n")
            continue

        df_7b = df_7b.sort_values('length')

        print(f"📊 {task_info['label']} ({task_info['tier']})")
        print("-" * 100)

        # Print data
        for _, row in df_7b.iterrows():
            sar_pct = row['sar_mean'] * 100
            err_pct = row['sar_std'] * 100
            print(f"  Length {int(row['length']):3d}: SAR = {sar_pct:5.1f}% ± {err_pct:4.1f}%")

        # Calculate slope
        if len(df_7b) > 1:
            lengths = df_7b['length'].values
            sars = df_7b['sar_mean'].values

            # Simple linear regression on log scale
            log_lengths = np.log2(lengths)
            slope = (sars[-1] - sars[0]) / (log_lengths[-1] - log_lengths[0])

            degradation = (sars[0] - sars[-1]) * 100
            print(f"\n  Degradation: {degradation:.1f}% (from {sars[0]*100:.1f}% → {sars[-1]*100:.1f}%)")
            print(f"  Slope (log scale): {slope:.3f}\n")

if __name__ == "__main__":
    print("="*100)
    print("SAR vs SEQUENCE LENGTH ANALYSIS - ALL 6 TASKS")
    print("="*100 + "\n")

    # Generate plots
    print("Generating individual task plots...")
    plot_sar_sequence_length_individual()

    print("Generating comparison plot...")
    plot_sar_sequence_length_comparison()

    # Generate table
    generate_sar_table()

    print("\n" + "="*100)
    print("FILES GENERATED:")
    print("  - sar_sequence_length_all_tasks.png (2x3 individual plots)")
    print("  - sar_all_tasks_comparison.png (single comprehensive comparison)")
    print("="*100)
