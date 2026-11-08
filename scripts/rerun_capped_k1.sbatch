#!/bin/bash
#SBATCH --account=quonbiogrp          # group allocation (4 A100s max at once)
#SBATCH --partition=gpu-a100-h        # A100 80GB nodes
#SBATCH --gres=gpu:a100:1             # one GPU; every model fits on one A100 in bf16
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G                    # host RAM for loading weights; launcher lowers it for small models
#SBATCH --time=12:00:00               # launcher sets per-model limits (about 2x the estimate)
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%j.log

# Re-run the k=1, N>=64 cells whose answers were cut off by the old token budget
# (see tasks/rerun_capped_k1.py). Resumable: rows already in the staging CSVs
# under results_clean/_k1_rerun/ are skipped, so a requeue or resubmit continues.
#   sbatch --job-name=k1fix-<tag> scripts/rerun_capped_k1.sbatch <hf_model_id>

set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
export HF_HUB_OFFLINE=1               # use the cached model files the original runs used
export PYTHONUNBUFFERED=1             # progress lines reach the log as they happen
cd /group/gquongrp/workspaces/baran/AvalancheBreak

MODEL_ID="$1"
echo "Node: $(hostname) | Model: $MODEL_ID | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" -m tasks.rerun_capped_k1 "$MODEL_ID"
echo "Done: $(date)"
