#!/bin/bash
#SBATCH --account=quonbiogrp          # group allocation
#SBATCH --partition=gpu-a100-h        # Qwen2.5-32B (~65 GB in bf16) needs an 80 GB A100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=06:00:00               # ~1 h expected; resumable
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%j.log

# Input vs output markers (scripts/run_marker_sides.py): 2x2 at Qwen2.5-32B on running_xor
# and random_permutation_digits, plus a baseline with no marker sentence. 400 generations.
#   sbatch --job-name=sides-32B scripts/run_marker_sides.sbatch Qwen/Qwen2.5-32B-Instruct
set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd /group/gquongrp/workspaces/baran/AvalancheBreak
echo "Node: $(hostname) | Model: $1 | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" scripts/run_marker_sides.py --model "$1" "${@:2}"
echo "Done: $(date)"
