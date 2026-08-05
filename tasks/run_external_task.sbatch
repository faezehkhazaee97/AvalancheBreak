#!/bin/bash
#SBATCH --account=ctbrowngrp
#SBATCH --partition=gpu-a100-h
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%j.log

set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
cd /group/gquongrp/workspaces/baran/AvalancheBreak

TASK_NAME="$1"
MODEL_ID="$2"
MODE="$3"
TRIALS="${4:-}"

echo "=================================================="
echo "Node:  $(hostname)"
echo "Time:  $(date)"
echo "Task:  $TASK_NAME | Model: $MODEL_ID | Mode: $MODE | Trials: ${TRIALS:-default}"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$ENV/bin/python" -m tasks.run_external_task "$TASK_NAME" "$MODEL_ID" "$MODE" $TRIALS

echo "=================================================="
echo "Done: $(date)"
echo "=================================================="
