#!/bin/bash
#SBATCH --account=quonbiogrp          # group allocation
#SBATCH --partition=gpu-a100-h        # A100 80GB nodes; the launcher overrides for Titans
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00               # 0.5-2 h per task at 27-32B (wording runs: 25 min to 2 h)
#SBATCH --array=0-3%1                 # four dependency tasks, one at a time
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%A_%a.log

# Worked-example wording (tasks/common/wording.py, make_worked): the four dependency tasks with the
# "across" instruction plus two worked examples with bars, the first annotated at the position right
# after its first bar. Reviewer question: do explicit cross-boundary demonstrations remove the
# post-marker blind spot? Same grid as the wording test (N <= 64, all spacings, 20 trials).
#   sbatch --job-name=worked-<tag> scripts/run_worked_examples.sbatch <hf_model_id>
#   Titans: sbatch --partition=gpuh --gres=gpu:titan:1 --cpus-per-task=6 --job-name=worked-<tag> scripts/run_worked_examples.sbatch <hf_model_id>
set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd /group/gquongrp/workspaces/baran/AvalancheBreak
MODEL_ID="$1"
TASKS=(running_xor_worked copy_previous_worked running_sum_local_worked sliding_window_max_worked)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "Node: $(hostname) | Model: $MODEL_ID | Task: $TASK | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" -m tasks.run_digit_controls "$TASK" "$MODEL_ID" 20
echo "Done: $(date)"
