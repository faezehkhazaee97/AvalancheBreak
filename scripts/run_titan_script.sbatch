#!/bin/bash
#SBATCH --account=quonbiogrp          # group allocation
#SBATCH --partition=gpuh              # the group's Titan RTX GPUs (24 GB): 7B-class models only
#SBATCH --gres=gpu:titan:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%j.log

# Run one of the resumable scripts/run_*.py experiment scripts on a Titan.
#   sbatch --job-name=<name> scripts/run_titan_script.sbatch scripts/run_paired_inputs.py --model Qwen/Qwen2.5-7B-Instruct
set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd /group/gquongrp/workspaces/baran/AvalancheBreak
echo "Node: $(hostname) | $* | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" "$@"
echo "Done: $(date)"
