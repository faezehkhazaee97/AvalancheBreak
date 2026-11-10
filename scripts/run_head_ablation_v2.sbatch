#!/bin/bash
#SBATCH --account=quonbiogrp          # group allocation
#SBATCH --partition=gpu-a100-h        # 32B needs an A100; 7B is submitted to gpuh (Titan) by the caller
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=08:00:00               # ~2 h expected at 32B (about 1,100 generations); resumable
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%j.log

# Head ablation v2 (scripts/run_head_ablation_v2.py): merged letters vs spaced letters vs
# digits, random_permutation A=8, N in {32, 64}, k in {2, 8}, with random-head and
# sham-position controls.
#   sbatch --job-name=ablation-<tag> scripts/run_head_ablation_v2.sbatch <hf_model_id>
set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd /group/gquongrp/workspaces/baran/AvalancheBreak
echo "Node: $(hostname) | Model: $1 | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" scripts/run_head_ablation_v2.py --model "$1" "${@:2}"
echo "Done: $(date)"
