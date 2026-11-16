#!/bin/bash
#SBATCH --account=quonbiogrp
#SBATCH --partition=gpu-a100-h        # Qwen2.5-32B (~65 GB in bf16) needs an 80 GB A100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=6:00:00               # ~10,500 forward passes at 64 layers; resumable
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%j.log

# Position-aligned activation patching (scripts/run_delimiter_variants.py): donor is the
# across-wording marked run, so marker placement, ground-truth answer and query token all match.
#   sbatch --job-name=patchalign-32B scripts/run_patching_aligned.sbatch Qwen/Qwen2.5-32B-Instruct
set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd /group/gquongrp/workspaces/baran/AvalancheBreak
echo "Node: $(hostname) | Model: $1 | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" scripts/run_delimiter_variants.py --model "$1" "${@:2}"
echo "Done: $(date)"
