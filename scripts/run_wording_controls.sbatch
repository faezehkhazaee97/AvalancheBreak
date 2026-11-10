#!/bin/bash
#SBATCH --account=quonbiogrp          # group allocation (launcher may use ctbrowngrp)
#SBATCH --partition=gpu-a100-h        # A100 80GB nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G                     # earlier runs peaked under 2 GB host RAM
#SBATCH --time=06:00:00               # ~0.5-1.5 h per task at 27-32B
#SBATCH --array=0-3%1                 # four tasks, one at a time
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%A_%a.log

# Instruction-wording test for the post-marker blind spot (tasks/common/wording.py):
# running_xor and running_sum_local with "neutral" and "across" marker instructions.
#   sbatch --job-name=wording-<tag> scripts/run_wording_controls.sbatch <hf_model_id>
set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd /group/gquongrp/workspaces/baran/AvalancheBreak
MODEL_ID="$1"
TASKS=(running_xor_neutral running_xor_across running_sum_local_neutral running_sum_local_across)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "Node: $(hostname) | Model: $MODEL_ID | Task: $TASK | $(date)"
"$ENV/bin/python" -m tasks.run_digit_controls "$TASK" "$MODEL_ID" 20
echo "Done: $(date)"
