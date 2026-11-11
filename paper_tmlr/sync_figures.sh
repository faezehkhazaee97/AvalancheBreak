#!/bin/bash
# Re-copy the paper's figures from ../figs after regenerating them.
# Regenerate Figure 1 first when new results land:
#   .conda/envs/qwen-focus/bin/python scripts/make_fig_split_heatmap.py
set -euo pipefail
cd "$(dirname "$0")"
FIGS=(fig0_overview fig_three_formats fig_profile fig_profile_by_task fig1_per_position fig_reset_per_position fig1_split_heatmap fig2_mechanism_merged fig2_vertex_migration
      divide_and_conquer causal_ablation attention_mechanism_32b head_heatmaps_32b
      error_containment_combined attention_digits_comparison fig_by_model_local fig_by_model_dependency
      external_vowel_tagging external_procbench
      external_string_copy external_benchmark_gemini)
mkdir -p figures
for f in "${FIGS[@]}"; do cp "../figs/$f.pdf" figures/; done
echo "Copied ${#FIGS[@]} figures."
