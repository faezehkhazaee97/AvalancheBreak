#!/bin/bash
#SBATCH --job-name=core-templated-14b
#SBATCH --account=quonbiogrp
#SBATCH --partition=gpu-a100-h
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/core-templated-14b-%j.log

set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
cd /group/gquongrp/workspaces/baran/AvalancheBreak

echo "=================================================="
echo "Node:  $(hostname)"
echo "Time:  $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$ENV/bin/python" scripts/run_core_templated_14b.py

echo "=================================================="
echo "Done: $(date)"
echo "=================================================="
