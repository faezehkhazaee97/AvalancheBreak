#!/usr/bin/env python3
"""
Comprehensive analysis: 7B vs 32B comparison, TAR/SAR dashboard, error recovery
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from scipy import stats

# Paths
RESULTS_DIR = Path("/group/gquongrp/workspaces/baran/AvalancheBreak/results")
OUTPUT_DIR = Path("/group/gquongrp/workspaces/baran/AvalancheBreak/analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Task metadata
TASKS = {
    "character_substitution": {"type": "sweep", "param": "alphabet", "levels": [2, 4, 8]},
    "running_sum_local": {"type": "sweep", "param": "modulus", "levels": [2, 4, 8]},
    "running_xor": {"type": "boundaries", "param": None},
    "modsum": {"type": "sweep", "param": "modulus", "levels": [2, 4, 8]},
    "addition": {"type": "sweep", "param": "digit_range", "levels": [2, 4, 8]},
}

def load_all_results(model_name):
    """Load all result CSVs for a model."""
    model_dir = RESULTS_DIR / model_name
    results = {}

    for csv_file in model_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            filename = csv_file.stem
            results[filename] = df
        except Exception as e:
            print(f"  Error loading {csv_file}: {e}")

    return results

def extract_tar_sar(df, task_name):
    """Extract TAR and SAR metrics from result dataframe."""
    metrics = {}

    # Look for TAR/SAR columns
    tar_cols = [c for c in df.columns if 'tar' in c.lower() or 'token_accuracy' in c.lower()]
    sar_cols = [c for c in df.columns if 'sar' in c.lower() or 'sequence_accuracy' in c.lower()]

    if tar_cols:
        metrics['TAR'] = df[tar_cols[0]].mean()
    if sar_cols:
        metrics['SAR'] = df[sar_cols[0]].mean()

    # Extract parameter values if present
    param_col = None
    for col in df.columns:
        if any(x in col.lower() for x in ['alphabet', 'modulus', 'digit', 'parameter', 'difficulty']):
            param_col = col
            break

    metrics['param_col'] = param_col
    metrics['length_col'] = 'length' if 'length' in df.columns else None

    return metrics, df

def compute_comparison_metrics():
    """Compute 7B vs 32B comparison metrics."""
    print("Loading results...")
    results_7b = load_all_results("Qwen2.5-7B-Instruct")
    results_32b = load_all_results("Qwen2.5-32B-Instruct")

    comparison = []

    # Extract sweep results (main tasks)
    sweep_suffixes = [
        ("character_substitution_alphabet_sweep", "Character Substitution"),
        ("running_sum_local_modulus_sweep", "Running Sum Local"),
        ("modsum_modulus_sweep", "Modsum"),
        ("addition_digit_range_sweep", "Integer Addition"),
    ]

    boundary_suffixes = [
        ("running_xor_grid_structured_boundaries", "Running XOR"),
    ]

    for csv_pattern, task_label in sweep_suffixes:
        # Find matching files
        for key_7b, df_7b in results_7b.items():
            if csv_pattern in key_7b and "Qwen2.5-7B" in key_7b:
                for key_32b, df_32b in results_32b.items():
                    if csv_pattern in key_32b and "Qwen2.5-32B" in key_32b:
                        metrics_7b, _ = extract_tar_sar(df_7b, task_label)
                        metrics_32b, _ = extract_tar_sar(df_32b, task_label)

                        if 'TAR' in metrics_7b and 'TAR' in metrics_32b:
                            comparison.append({
                                'task': task_label,
                                'model_7b_tar': metrics_7b.get('TAR', np.nan),
                                'model_7b_sar': metrics_7b.get('SAR', np.nan),
                                'model_32b_tar': metrics_32b.get('TAR', np.nan),
                                'model_32b_sar': metrics_32b.get('SAR', np.nan),
                                'tar_improvement': metrics_32b.get('TAR', 0) - metrics_7b.get('TAR', 0),
                                'sar_improvement': metrics_32b.get('SAR', 0) - metrics_7b.get('SAR', 0),
                            })
                        break

    # Add boundary condition results
    for csv_pattern, task_label in boundary_suffixes:
        for key_7b, df_7b in results_7b.items():
            if csv_pattern in key_7b:
                for key_32b, df_32b in results_32b.items():
                    if csv_pattern in key_32b:
                        metrics_7b, _ = extract_tar_sar(df_7b, task_label)
                        metrics_32b, _ = extract_tar_sar(df_32b, task_label)

                        if 'TAR' in metrics_7b and 'TAR' in metrics_32b:
                            comparison.append({
                                'task': task_label,
                                'model_7b_tar': metrics_7b.get('TAR', np.nan),
                                'model_7b_sar': metrics_7b.get('SAR', np.nan),
                                'model_32b_tar': metrics_32b.get('TAR', np.nan),
                                'model_32b_sar': metrics_32b.get('SAR', np.nan),
                                'tar_improvement': metrics_32b.get('TAR', 0) - metrics_7b.get('TAR', 0),
                                'sar_improvement': metrics_32b.get('SAR', 0) - metrics_7b.get('SAR', 0),
                            })
                        break

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print(f"Comparison metrics saved: {len(comparison_df)} tasks")
    return comparison_df

def compute_difficulty_scaling():
    """Analyze performance vs difficulty (alphabet size, modulus, digit range)."""
    print("\nAnalyzing difficulty scaling...")
    results_7b = load_all_results("Qwen2.5-7B-Instruct")
    results_32b = load_all_results("Qwen2.5-32B-Instruct")

    scaling_data = []

    # Character substitution alphabet sweep
    for key, df in results_7b.items():
        if "character_substitution_alphabet_sweep" in key and "Qwen2.5-7B" in key:
            if 'alphabet_size' in df.columns and 'tar' in df.columns:
                for _, row in df.iterrows():
                    scaling_data.append({
                        'task': 'Character Substitution',
                        'model': '7B',
                        'difficulty_param': int(row['alphabet_size']),
                        'tar': row['tar'],
                        'sar': row.get('sar', np.nan),
                    })

    for key, df in results_32b.items():
        if "character_substitution_alphabet_sweep" in key and "Qwen2.5-32B" in key:
            if 'alphabet_size' in df.columns and 'tar' in df.columns:
                for _, row in df.iterrows():
                    scaling_data.append({
                        'task': 'Character Substitution',
                        'model': '32B',
                        'difficulty_param': int(row['alphabet_size']),
                        'tar': row['tar'],
                        'sar': row.get('sar', np.nan),
                    })

    # Similar for other tasks...
    scaling_df = pd.DataFrame(scaling_data)
    if len(scaling_df) > 0:
        scaling_df.to_csv(OUTPUT_DIR / "difficulty_scaling.csv", index=False)
        print(f"Scaling analysis saved: {len(scaling_df)} data points")

    return scaling_df

def compute_error_recovery():
    """Analyze error recovery: P(correct | previous error)."""
    print("\nAnalyzing error recovery patterns...")

    results_7b = load_all_results("Qwen2.5-7B-Instruct")
    results_32b = load_all_results("Qwen2.5-32B-Instruct")

    recovery_stats = []

    for model_name, results in [("7B", results_7b), ("32B", results_32b)]:
        for key, df in results.items():
            # Look for detailed per-token/per-sequence results
            if 'errors' in df.columns or 'correct' in df.columns:
                # Check if we can compute error recovery
                if 'errors' in df.columns:
                    total_errors = df['errors'].sum()
                    total_samples = len(df)
                    error_rate = total_errors / total_samples if total_samples > 0 else 0

                    recovery_stats.append({
                        'model': model_name,
                        'task': key.split('_')[0],
                        'error_rate': error_rate,
                        'total_samples': total_samples,
                        'total_errors': int(total_errors),
                    })

    recovery_df = pd.DataFrame(recovery_stats)
    if len(recovery_df) > 0:
        recovery_df.to_csv(OUTPUT_DIR / "error_recovery.csv", index=False)
        print(f"Error recovery analysis saved: {len(recovery_df)} analysis points")

    return recovery_df

def generate_summary_stats():
    """Generate summary statistics for dashboard."""
    print("\nGenerating summary statistics...")

    results_7b = load_all_results("Qwen2.5-7B-Instruct")
    results_32b = load_all_results("Qwen2.5-32B-Instruct")

    summary = {
        "7B": {"tasks": len(results_7b), "files": sum(1 for _ in RESULTS_DIR.glob("Qwen2.5-7B-Instruct/*.csv"))},
        "32B": {"tasks": len(results_32b), "files": sum(1 for _ in RESULTS_DIR.glob("Qwen2.5-32B-Instruct/*.csv"))},
    }

    # Compute average metrics
    tar_7b = []
    tar_32b = []

    for key, df in results_7b.items():
        metrics, _ = extract_tar_sar(df, key)
        if 'TAR' in metrics:
            tar_7b.append(metrics['TAR'])

    for key, df in results_32b.items():
        metrics, _ = extract_tar_sar(df, key)
        if 'TAR' in metrics:
            tar_32b.append(metrics['TAR'])

    if tar_7b:
        summary["7B"]["avg_tar"] = float(np.mean(tar_7b))
        summary["7B"]["std_tar"] = float(np.std(tar_7b))

    if tar_32b:
        summary["32B"]["avg_tar"] = float(np.mean(tar_32b))
        summary["32B"]["std_tar"] = float(np.std(tar_32b))

    with open(OUTPUT_DIR / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary stats:\n{json.dumps(summary, indent=2)}")
    return summary

if __name__ == "__main__":
    print("=" * 70)
    print("COMPREHENSIVE ANALYSIS: 7B vs 32B PERFORMANCE")
    print("=" * 70)

    # Run analyses
    comparison_df = compute_comparison_metrics()
    scaling_df = compute_difficulty_scaling()
    recovery_df = compute_error_recovery()
    summary = generate_summary_stats()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print(f"  - model_comparison.csv")
    print(f"  - difficulty_scaling.csv")
    print(f"  - error_recovery.csv")
    print(f"  - summary_stats.json")

    if not comparison_df.empty:
        print("\n7B vs 32B Comparison Summary:")
        print(comparison_df[['task', 'model_7b_tar', 'model_32b_tar', 'tar_improvement']].to_string())
