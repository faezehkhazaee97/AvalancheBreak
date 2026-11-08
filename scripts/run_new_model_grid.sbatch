#!/bin/bash
#SBATCH --account=quonbiogrp          # group allocation (4 A100s max at once)
#SBATCH --partition=gpu-a100-h        # A100 80GB nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=08:00:00               # 32B took 1-2.5 h per task; headroom for slower architectures
#SBATCH --array=0-10%1                # 11 tasks, one at a time per model (leaves GPUs for other work)
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%A_%a.log

# Core grid for a newly added model: the 3 Tier-1 tasks, the 4 dependency
# tasks, and the 4 reset variants, at the same trials/cell Qwen2.5-32B used.
# Output: results_clean/<task>/grid/<model_tag>.csv
#   sbatch --job-name=grid-<tag> --dependency=afterok:<smoke_jobid> \
#     scripts/run_new_model_grid.sbatch <hf_model_id>

set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
cd /group/gquongrp/workspaces/baran/AvalancheBreak

MODEL_ID="$1"
TASKS=(cyclic_shift random_permutation conditional_shift
       copy_previous running_xor running_sum_local sliding_window_max
       copy_previous_reset running_xor_reset running_sum_local_reset sliding_window_max_reset)
TRIALS=(20 20 20
        40 40 40 20
        20 20 20 20)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
N_TRIALS="${TRIALS[$SLURM_ARRAY_TASK_ID]}"

echo "Node: $(hostname) | Model: $MODEL_ID | Task: $TASK | Trials/cell: $N_TRIALS | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" -m tasks.run_task "$TASK" "$MODEL_ID" grid "$N_TRIALS"
echo "Done: $(date)"
