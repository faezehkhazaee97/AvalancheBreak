#!/bin/bash
#SBATCH --account=quonbiogrp          # group allocation
#SBATCH --partition=gpu-a100-h        # Qwen2.5-32B (~65 GB in bf16) needs an 80 GB A100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=06:00:00               # ~1 h expected (560 generations); resumable
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%j.log

# Forced-format 2x2 (scripts/run_marker_sides_forced.py): a logits processor makes the
# output format certain instead of requested. 560 generations.
#   sbatch --job-name=forced-32B scripts/run_marker_sides_forced.sbatch Qwen/Qwen2.5-32B-Instruct
set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd /group/gquongrp/workspaces/baran/AvalancheBreak
echo "Node: $(hostname) | Model: $1 | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" scripts/run_marker_sides_forced.py --model "$1" "${@:2}"
echo "Done: $(date)"
