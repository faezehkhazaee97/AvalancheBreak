#!/bin/bash
#SBATCH --job-name=temperature-robustness
#SBATCH --account=ctbrowngrp
#SBATCH --partition=gpu-a100-h
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/temperature-robustness-%j.log

set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
cd /group/gquongrp/workspaces/baran/AvalancheBreak

echo "=================================================="
echo "Node:  $(hostname)"
echo "Time:  $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$ENV/bin/python" scripts/run_temperature_robustness_check.py

echo "=================================================="
echo "Done: $(date)"
echo "=================================================="
