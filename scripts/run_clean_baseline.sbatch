#!/bin/bash
#SBATCH --account=quonbiogrp          # group allocation
#SBATCH --partition=gpu-a100-h        # 27B-32B models need an 80 GB A100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=10:00:00               # ~1-3 h per model (1,500 generations); resumable
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%A_%a.log

# No-marker baselines whose prompt never mentions markers (scripts/run_clean_baseline.py),
# at the four stronger models. Submit as an array over the model list:
#   sbatch --array=0-3%2 --job-name=cleanbase scripts/run_clean_baseline.sbatch
set -euo pipefail
MODELS=(Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct Qwen/Qwen3.8-27B google/gemma-4-31B-it)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd /group/gquongrp/workspaces/baran/AvalancheBreak
echo "Node: $(hostname) | Model: $M | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" scripts/run_clean_baseline.py --model "$M"
echo "Done: $(date)"
