#!/bin/bash
#SBATCH --account=ctbrowngrp
#SBATCH --partition=low
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-12-92
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%A_%a.log
# Step 2 of the byte-level few-shot check: the standard grid (N in {4,8,16,32,64}, k in
# {0,1,2,3,4,8,16} with k < N, every alphabet value, 20 trials, greedy) for EvaByte-SFT with the
# three solved examples the base checkpoints get, one array task per task:
#   sbatch --array=0-3%2 --job-name=evabyte-fs scripts/evabyte_fewshot_grid.sbatch
# AB_DELIVERY (raw|chat, default raw) picks how the few-shot prompt is delivered; the pilot
# (scripts/evabyte_fewshot_pilot.sbatch) decides which one.
# Rows are appended as they are produced and finished cells are skipped on restart, so a preempted
# array task resumes instead of starting over. partition low: see the pilot script's note.
set -uo pipefail
TASKS=(running_xor copy_previous random_permutation_digits cyclic_shift_digits)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 EVABYTE_NO_TRITON=1
cd /group/gquongrp/workspaces/baran/AvalancheBreak
echo "Node: $(hostname) | Task: $TASK | delivery: ${AB_DELIVERY:-raw} | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" scripts/run_evabyte_fewshot.py --task "$TASK" --trials 20 --delivery "${AB_DELIVERY:-raw}"
echo "Done: $(date)"
