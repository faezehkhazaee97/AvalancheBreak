#!/usr/bin/env python3
"""
Running XOR: Interval (boundary spacing) sweep analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_DIR = Path("/group/gquongrp/workspaces/baran/AvalancheBreak/results")
OUTPUT_DIR = Path("/group/gquongrp/workspaces/baran/AvalancheBreak/analysis_output/plots")
OUTPUT_DIR.mkdir(exist_ok=True)

def analyze_xor_interval_sweep():
    """Analyze XOR performance vs boundary interval."""

    results = []

    for model_name in ['Qwen2.5-7B-Instruct', 'Qwen2.5-32B-Instruct']:
        model_dir = RESULTS_DIR / model_name

        xor_file = model_dir / 'running_xor_grid_structured_boundaries.csv'

        if not xor_file.exists():
            print(f"⚠️  No XOR file for {model_name}")
            continue

        df = pd.read_csv(xor_file)

        # Group by Interval (boundary spacing) and Seq_Length
        for interval in sorted(df['Interval'].unique()):
            for length in sorted(df['Seq_Length'].unique()):
                subset = df[(df['Interval'] == interval) & (df['Seq_Length'] == length)]

                if len(subset) == 0:
                    continue

                results.append({
                    'model': model_name.replace('Qwen2.5-', '').replace('-Instruct', ''),
                    'interval': int(interval),
                    'seq_length': int(length),
                    'tar_mean': subset['TAR'].mean(),
                    'tar_std': subset['TAR'].std(),
                    'sar_mean': subset['SAR'].mean(),
                    'sar_std': subset['SAR'].std(),
                    'n_samples': len(subset),
                })

    return pd.DataFrame(results)

def plot_xor_interval_sweep():
    """Plot XOR TAR vs boundary interval."""

    df = analyze_xor_interval_sweep()

    if df.empty:
        print("No data to plot")
        return

    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Create 2x2 subplot for different sequence lengths
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    seq_lengths = sorted(df['seq_length'].unique())

    for idx, length in enumerate(seq_lengths[:4]):
        ax = axes[idx]

        df_7b = df[(df['model'] == '7B') & (df['seq_length'] == length)].sort_values('interval')
        df_32b = df[(df['model'] == '32B') & (df['seq_length'] == length)].sort_values('interval')

        if df_7b.empty or df_32b.empty:
            ax.text(0.5, 0.5, f'No data for length {length}', ha='center', va='center')
            ax.set_title(f'Seq Length: {length}')
            continue

        ax.errorbar(df_7b['interval'], df_7b['tar_mean']*100,
                   yerr=df_7b['tar_std']*100, label='7B', marker='o',
                   capsize=5, linewidth=2, markersize=8, color='#e74c3c')
        ax.errorbar(df_32b['interval'], df_32b['tar_mean']*100,
                   yerr=df_32b['tar_std']*100, label='32B', marker='s',
                   capsize=5, linewidth=2, markersize=8, color='#3498db')

        ax.set_xlabel('Boundary Interval (0=dense, 16=sparse)', fontsize=11, fontweight='bold')
        ax.set_ylabel('TAR (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Running XOR - Seq Length: {length}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'running_xor_interval_sweep.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: running_xor_interval_sweep.png")
    plt.close()

def generate_xor_table():
    """Generate table of XOR performance by interval."""

    df = analyze_xor_interval_sweep()

    if df.empty:
        print("No data")
        return

    # Summary table
    print("\n" + "="*80)
    print("RUNNING XOR: PERFORMANCE BY BOUNDARY INTERVAL")
    print("="*80)
    print("\nInterpretation:")
    print("  Interval = 0-1: Dense boundaries (easy - max structure)")
    print("  Interval = 4-8: Sparse boundaries (medium)")
    print("  Interval = 16: Very sparse (hard - minimal structure)")
    print()

    for length in sorted(df['seq_length'].unique()):
        print(f"\n📊 Sequence Length: {length}")
        print("-" * 80)

        length_data = df[df['seq_length'] == length].sort_values('interval')

        for _, row in length_data.iterrows():
            tar_7b = row['tar_mean'] * 100 if row['model'] == '7B' else None
            tar_32b = row['tar_mean'] * 100 if row['model'] == '32B' else None

            if tar_7b is not None:
                print(f"  Interval {row['interval']:2d}: 7B={tar_7b:5.1f}%", end="")

            if tar_32b is not None:
                gain = tar_32b - tar_7b if (tar_7b is not None) else None
                if gain is not None:
                    print(f" → 32B={tar_32b:5.1f}% [+{gain:4.1f}%]")
                else:
                    print(f" | 32B={tar_32b:5.1f}%")

    # Correlation: does larger interval (sparser boundaries) hurt performance?
    print("\n" + "="*80)
    print("KEY INSIGHT: Boundary Density Effect")
    print("="*80)

    for model in ['7B', '32B']:
        model_data = df[df['model'] == model]

        # Compare dense (interval 0-1) vs sparse (interval 8-16)
        dense = model_data[model_data['interval'] <= 1]['tar_mean'].mean()
        sparse = model_data[model_data['interval'] >= 8]['tar_mean'].mean()

        if not np.isnan(dense) and not np.isnan(sparse):
            drop = (dense - sparse) * 100
            print(f"\n{model}:")
            print(f"  Dense boundaries (Interval 0-1): {dense*100:.1f}% TAR")
            print(f"  Sparse boundaries (Interval 8-16): {sparse*100:.1f}% TAR")
            print(f"  Performance drop: {drop:.1f}% (sparser = harder)")

if __name__ == "__main__":
    print("="*80)
    print("RUNNING XOR INTERVAL SWEEP ANALYSIS")
    print("="*80 + "\n")

    # Analyze
    df = analyze_xor_interval_sweep()
    print(f"Found {len(df)} data points\n")

    # Generate plots
    plot_xor_interval_sweep()

    # Generate table
    generate_xor_table()

    print("\n" + "="*80)
    print("FILES GENERATED:")
    print("  - running_xor_interval_sweep.png (4-panel by sequence length)")
    print("="*80)
