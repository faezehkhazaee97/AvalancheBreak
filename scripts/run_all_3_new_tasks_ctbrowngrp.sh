#!/bin/bash
#SBATCH --job-name=all-3-new-tasks
#SBATCH --account=ctbrowngrp
#SBATCH --partition=gpu-a100-h
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/all-3-new-tasks-%j.log

set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
cd /group/gquongrp/workspaces/baran/AvalancheBreak

echo "=================================================="
echo "Node:  $(hostname)"
echo "Account: ctbrowngrp"
echo "Start Time: $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "========== TASK 1/3: CHARACTER SUBSTITUTION =========="
echo "Start: $(date)"
"$ENV/bin/python" scripts/run_character_substitution_structured_boundaries.py
echo "Done: $(date)"

echo ""
echo "========== TASK 2/3: RUNNING XOR =========="
echo "Start: $(date)"
"$ENV/bin/python" scripts/run_running_xor_structured_boundaries.py
echo "Done: $(date)"

echo ""
echo "========== TASK 3/3: RUNNING SUM (LOCAL) =========="
echo "Start: $(date)"
"$ENV/bin/python" scripts/run_running_sum_local_structured_boundaries.py
echo "Done: $(date)"

echo ""
echo "=================================================="
echo "ALL 3 TASKS COMPLETE!"
echo "End Time: $(date)"
echo "=================================================="
echo "Results saved to: results/Qwen2.5-7B-Instruct/"
echo "  - character_substitution_grid_structured_boundaries.csv"
echo "  - running_xor_grid_structured_boundaries.csv"
echo "  - running_sum_local_grid_structured_boundaries.csv"
