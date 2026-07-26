#!/bin/bash
#SBATCH --account=quonbiogrp
#SBATCH --partition=gpu-a100-h
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%j.log

# Generic second-batch launcher for any (task, model, trial_start, trial_end).
# Submit with explicit overrides, e.g.:
#   sbatch --job-name=2nd-cyclic_shift-7b --mem=64G --time=02:00:00 \
#     tasks/run_task_second_batch.sbatch cyclic_shift Qwen/Qwen2.5-7B-Instruct 0 20

set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
cd /group/gquongrp/workspaces/baran/AvalancheBreak

TASK_NAME="$1"
MODEL_ID="$2"
TRIAL_START="$3"
TRIAL_END="$4"

echo "=================================================="
echo "Node:  $(hostname)"
echo "Time:  $(date)"
echo "Task:  $TASK_NAME | Model: $MODEL_ID | Trials: $TRIAL_START-$TRIAL_END"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$ENV/bin/python" -m tasks.run_task_second_batch "$TASK_NAME" "$MODEL_ID" "$TRIAL_START" "$TRIAL_END"

echo "=================================================="
echo "Done: $(date)"
echo "=================================================="
