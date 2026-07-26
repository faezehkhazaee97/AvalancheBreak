#!/usr/bin/env python3
"""
TAR vs Sequence Length for all 6 tasks
Same style as SAR plots
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

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
    """Load all result files for a task and extract TAR by sequence length."""
    model_dir = RESULTS_DIR / model_name
    results = []

    for csv_file in model_dir.glob(f"*{task_key}*.csv"):
        try:
            df = pd.read_csv(csv_file)

            if df.empty or 'Seq_Length' not in df.columns or 'TAR' not in df.columns:
                continue

            # Group by sequence length
            for length in sorted(df['Seq_Length'].unique()):
                subset = df[df['Seq_Length'] == length]
                results.append({
                    'length': int(length),
                    'tar_mean': subset['TAR'].mean(),
                    'tar_std': subset['TAR'].std(),
                    'n': len(subset),
                })
        except Exception as e:
            pass

    return pd.DataFrame(results)

def plot_tar_sequence_length_individual():
    """Create individual TAR vs length plots for each task."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    task_list = list(TASKS.keys())

    for idx, task_key in enumerate(task_list):
        ax = axes[idx]
        task_info = TASKS[task_key]

        # Load 7B data
        df_7b = load_task_data('Qwen2.5-7B-Instruct', task_key)

        if df_7b.empty:
            ax.text(0.5, 0.5, f'No data', ha='center', va='center')
            ax.set_title(task_info['label'])
            continue

        df_7b = df_7b.sort_values('length')

        # Plot with error bars
        ax.errorbar(df_7b['length'], df_7b['tar_mean'] * 100,
                   yerr=df_7b['tar_std'] * 100,
                   marker='o', linewidth=2, markersize=8, capsize=5,
                   color=task_info['color'], label='7B')

        # Load 32B if available
        df_32b = load_task_data('Qwen2.5-32B-Instruct', task_key)
        if not df_32b.empty:
            df_32b = df_32b.sort_values('length')
            ax.errorbar(df_32b['length'], df_32b['tar_mean'] * 100,
                       yerr=df_32b['tar_std'] * 100,
                       marker='s', linewidth=2, markersize=8, capsize=5,
                       color=task_info['color'], alpha=0.5, linestyle='--', label='32B')

        ax.set_xlabel('Sequence Length', fontsize=11, fontweight='bold')
        ax.set_ylabel('TAR (%)', fontsize=11, fontweight='bold')
        ax.set_title(f"{task_info['label']}\n{task_info['tier']}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_ylim([0, 100])

    # Hide extra subplot
    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tar_sequence_length_all_tasks.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: tar_sequence_length_all_tasks.png")
    plt.close()

def plot_tar_sequence_length_comparison():
    """Create a single comprehensive plot comparing all tasks."""

    fig, ax = plt.subplots(figsize=(14, 8))

    task_list = list(TASKS.keys())
    colors_cycle = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c', '#2ecc71']

    for idx, task_key in enumerate(task_list):
        task_info = TASKS[task_key]

        # Load 7B data
        df_7b = load_task_data('Qwen2.5-7B-Instruct', task_key)

        if df_7b.empty:
            continue

        df_7b = df_7b.sort_values('length')

        # Plot with error bars
        ax.errorbar(df_7b['length'], df_7b['tar_mean'] * 100,
                   yerr=df_7b['tar_std'] * 100,
                   marker='o', linewidth=2.5, markersize=9, capsize=6,
                   color=colors_cycle[idx], label=task_info['label'], alpha=0.8)

    ax.set_xlabel('Sequence Length (N)', fontsize=13, fontweight='bold')
    ax.set_ylabel('TAR - 1/k slope (mean across groups)', fontsize=13, fontweight='bold')
    ax.set_title('TAR Performance Across All Tasks by Sequence Length\n(7B Models)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.set_ylim([-5, 105])

    # Use log scale for x-axis
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tar_all_tasks_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: tar_all_tasks_comparison.png")
    plt.close()

def generate_tar_table():
    """Generate table showing TAR degradation across sequence lengths."""

    print("\n" + "="*100)
    print("TAR (Total Token Accuracy) vs Sequence Length - All Tasks")
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
            tar_pct = row['tar_mean'] * 100
            err_pct = row['tar_std'] * 100
            print(f"  Length {int(row['length']):3d}: TAR = {tar_pct:5.1f}% ± {err_pct:4.1f}%")

        # Calculate degradation
        if len(df_7b) > 1:
            tars = df_7b['tar_mean'].values
            degradation = (tars[0] - tars[-1]) * 100
            print(f"\n  Total Degradation: {degradation:.1f}% (from {tars[0]*100:.1f}% → {tars[-1]*100:.1f}%)")
            print()

if __name__ == "__main__":
    print("="*100)
    print("TAR vs SEQUENCE LENGTH ANALYSIS - ALL 6 TASKS")
    print("="*100 + "\n")

    print("Generating individual task plots...")
    plot_tar_sequence_length_individual()

    print("Generating comparison plot...")
    plot_tar_sequence_length_comparison()

    # Generate table
    generate_tar_table()

    print("\n" + "="*100)
    print("FILES GENERATED:")
    print("  - tar_sequence_length_all_tasks.png (2x3 individual plots)")
    print("  - tar_all_tasks_comparison.png (single comprehensive comparison)")
    print("="*100)
