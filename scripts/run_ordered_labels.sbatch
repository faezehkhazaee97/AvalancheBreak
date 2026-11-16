#!/bin/bash
#SBATCH --account=quonbiogrp          # group allocation
#SBATCH --partition=gpu-a100-h        # Qwen2.5-32B / Gemma-4-31B (~65 GB in bf16) need an 80 GB A100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=8:00:00                # ~2 h expected (480 generations, N <= 64); resumable
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%j.log

# Distinct ordered boundary labels vs a repeating marker (scripts/run_ordered_labels.py):
# running_xor, running_sum_local and random_permutation_digits, N in {32, 64}, k = 4,
# markers bar / letter / ordered plus the no-marker baseline.
#   sbatch --job-name=ordlab-32B scripts/run_ordered_labels.sbatch Qwen/Qwen2.5-32B-Instruct
set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd /group/gquongrp/workspaces/baran/AvalancheBreak
echo "Node: $(hostname) | Model: $1 | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" scripts/run_ordered_labels.py --model "$1" "${@:2}"
echo "Done: $(date)"
