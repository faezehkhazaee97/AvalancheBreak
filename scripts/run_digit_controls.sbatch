#!/bin/bash
#SBATCH --account=quonbiogrp          # group allocation
#SBATCH --partition=gpu-a100-h        # launcher adds gpuh (Titan) for the small models
#SBATCH --gres=gpu:a100:1             # launcher uses an untyped gpu:1 for the small models
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=08:00:00               # ~1-2 h per task at 27-32B; launcher shortens for small models
#SBATCH --array=0-2%1                 # the three digit tasks, one at a time per model
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%A_%a.log

# Digit-alphabet versions of the three position-local tasks (tokenization control).
#   sbatch --job-name=digits-<tag> scripts/run_digit_controls.sbatch <hf_model_id>

set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd /group/gquongrp/workspaces/baran/AvalancheBreak

MODEL_ID="$1"
TASKS=(cyclic_shift_digits random_permutation_digits conditional_shift_digits)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "Node: $(hostname) | Model: $MODEL_ID | Task: $TASK | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" -m tasks.run_digit_controls "$TASK" "$MODEL_ID" 20
echo "Done: $(date)"
