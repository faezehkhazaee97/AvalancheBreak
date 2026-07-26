#!/usr/bin/env python3
"""
Detailed scaling analysis: Show TAR/SAR by alphabet/modulus/digit levels
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = Path("/group/gquongrp/workspaces/baran/AvalancheBreak/results")
OUTPUT_DIR = Path("/group/gquongrp/workspaces/baran/AvalancheBreak/analysis_output/plots")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def extract_difficulty_level(filename, df):
    """Extract difficulty parameter from filename and dataframe."""
    # Map actual column names in the CSVs
    if 'alphabet_sweep' in filename:
        param_name = 'Alphabet_Size'
    elif 'modulus_sweep' in filename:
        param_name = 'Modulus'
    elif 'digit_range_sweep' in filename:
        param_name = 'Digit_Max'
    else:
        return None

    # Check for both exact case and lowercase
    for col in df.columns:
        if col.lower() == param_name.lower():
            return param_name, df[col].values

    return None

def load_and_parse_sweep(model_name, task_pattern):
    """Load sweep results and extract performance by difficulty level."""
    model_dir = RESULTS_DIR / model_name
    results = []

    for csv_file in model_dir.glob(f"*{task_pattern}*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                continue

            # Extract parameter
            param_info = extract_difficulty_level(csv_file.stem, df)
            if not param_info:
                continue

            param_name, param_vals = param_info

            # Extract TAR/SAR (look for various column names)
            tar_col = 'TAR'
            sar_col = 'SAR' if 'SAR' in df.columns else None
            length_col = 'Seq_Length' if 'Seq_Length' in df.columns else None

            if tar_col is None:
                continue

            # Group by parameter level
            for param_level in np.unique(param_vals):
                mask = param_vals == param_level
                subset = df[mask]

                if len(subset) == 0:
                    continue

                result = {
                    'model': model_name.replace('Qwen2.5-', '').replace('-Instruct', ''),
                    'task': task_pattern,
                    'parameter': param_name,
                    'level': int(param_level),
                    'tar_mean': subset[tar_col].mean(),
                    'tar_std': subset[tar_col].std(),
                    'n_samples': len(subset),
                }

                if sar_col and sar_col in subset.columns:
                    result['sar_mean'] = subset[sar_col].mean()
                    result['sar_std'] = subset[sar_col].std()

                if length_col and length_col in subset.columns:
                    result['length_mean'] = subset[length_col].mean()

                results.append(result)

        except Exception as e:
            print(f"  Error processing {csv_file.stem}: {e}")

    return pd.DataFrame(results)

def plot_scaling_comparison():
    """Plot TAR/SAR vs difficulty level for each task and model."""

    tasks = [
        ('character_substitution_alphabet_sweep', 'Character Substitution (Alphabet Size)'),
        ('running_sum_local_modulus_sweep', 'Running Sum Local (Modulus)'),
        ('modsum_modulus_sweep', 'Modsum (Modulus)'),
        ('addition_digit_range_sweep', 'Integer Addition (Digit Range)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (pattern, title) in enumerate(tasks):
        ax = axes[idx]

        # Load both models
        df_7b = load_and_parse_sweep('Qwen2.5-7B-Instruct', pattern)
        df_32b = load_and_parse_sweep('Qwen2.5-32B-Instruct', pattern)

        if df_7b.empty or df_32b.empty:
            ax.text(0.5, 0.5, f'No data for {pattern}', ha='center', va='center')
            ax.set_title(title)
            continue

        # Plot with error bars
        df_7b_sorted = df_7b.sort_values('level')
        df_32b_sorted = df_32b.sort_values('level')

        ax.errorbar(df_7b_sorted['level'], df_7b_sorted['tar_mean']*100,
                   yerr=df_7b_sorted['tar_std']*100, label='7B', marker='o',
                   capsize=5, linewidth=2, markersize=8, color='#e74c3c')
        ax.errorbar(df_32b_sorted['level'], df_32b_sorted['tar_mean']*100,
                   yerr=df_32b_sorted['tar_std']*100, label='32B', marker='s',
                   capsize=5, linewidth=2, markersize=8, color='#3498db')

        ax.set_xlabel('Difficulty Level', fontsize=11, fontweight='bold')
        ax.set_ylabel('TAR (%)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tar_scaling_by_difficulty.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: tar_scaling_by_difficulty.png")
    plt.close()

def generate_scaling_table():
    """Generate detailed table of performance by difficulty level."""

    tasks = [
        ('character_substitution_alphabet_sweep', 'Char Sub'),
        ('running_sum_local_modulus_sweep', 'Run Sum'),
        ('modsum_modulus_sweep', 'Modsum'),
        ('addition_digit_range_sweep', 'Addition'),
    ]

    all_data = []

    for pattern, label in tasks:
        df_7b = load_and_parse_sweep('Qwen2.5-7B-Instruct', pattern)
        df_32b = load_and_parse_sweep('Qwen2.5-32B-Instruct', pattern)

        if df_7b.empty or df_32b.empty:
            continue

        # Merge on level
        df_7b_pivot = df_7b.groupby('level')['tar_mean'].mean()
        df_32b_pivot = df_32b.groupby('level')['tar_mean'].mean()

        for level in sorted(set(list(df_7b_pivot.index) + list(df_32b_pivot.index))):
            tar_7b = df_7b_pivot.get(level, np.nan)
            tar_32b = df_32b_pivot.get(level, np.nan)

            all_data.append({
                'Task': label,
                'Level': level,
                '7B TAR': f"{tar_7b*100:.1f}%" if not np.isnan(tar_7b) else "N/A",
                '32B TAR': f"{tar_32b*100:.1f}%" if not np.isnan(tar_32b) else "N/A",
                'Gain': f"+{(tar_32b-tar_7b)*100:.1f}%" if not (np.isnan(tar_7b) or np.isnan(tar_32b)) else "N/A",
            })

    table_df = pd.DataFrame(all_data)
    table_df.to_csv(OUTPUT_DIR / 'scaling_by_difficulty_table.csv', index=False)

    print("\n" + "="*80)
    print("PERFORMANCE BY DIFFICULTY LEVEL (Alphabet/Modulus/Digit Range)")
    print("="*80)
    print(table_df.to_string(index=False))
    print("\n")

def analyze_by_length():
    """Analyze performance by sequence length if available."""
    print("\nAnalyzing performance by sequence length...")

    results = []

    for model_name in ['Qwen2.5-7B-Instruct', 'Qwen2.5-32B-Instruct']:
        model_dir = RESULTS_DIR / model_name

        for csv_file in model_dir.glob('*sweep*.csv'):
            try:
                df = pd.read_csv(csv_file)

                if 'Seq_Length' not in df.columns:
                    continue

                tar_col = 'TAR' if 'TAR' in df.columns else None
                if tar_col is None:
                    continue

                # Group by length
                for length in sorted(df['Seq_Length'].unique()):
                    subset = df[df['Seq_Length'] == length]
                    results.append({
                        'model': model_name.replace('Qwen2.5-', '').replace('-Instruct', ''),
                        'task': csv_file.stem[:30],
                        'length': int(length),
                        'tar_mean': subset[tar_col].mean(),
                        'n': len(subset),
                    })
            except:
                pass

    if results:
        length_df = pd.DataFrame(results)
        length_df.to_csv(OUTPUT_DIR / 'performance_by_length.csv', index=False)

        print("\nSample of performance by sequence length:")
        print(length_df.head(15).to_string(index=False))

if __name__ == "__main__":
    print("="*80)
    print("DETAILED SCALING ANALYSIS: Performance by Difficulty Level")
    print("="*80 + "\n")

    print("Loading sweep results by difficulty level...\n")

    # Generate plots
    plot_scaling_comparison()

    # Generate table
    generate_scaling_table()

    # Analyze by length
    analyze_by_length()

    print(f"\n✓ All plots saved to: {OUTPUT_DIR}")
    print("  - tar_scaling_by_difficulty.png (4-panel comparison)")
    print("  - scaling_by_difficulty_table.csv (detailed table)")
    print("  - performance_by_length.csv (if available)")
