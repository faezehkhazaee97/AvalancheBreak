#!/bin/bash
#SBATCH --account=quonbiogrp
#SBATCH --partition=gpuh
#SBATCH --gres=gpu:titan:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%j.log
# Byte-level check with EvaByte-SFT: smoke test on the PyTorch attention path, then, only if the model
# produces answers, the standard grids on two dependency tasks and two position-local digit tasks.
#   sbatch --job-name=evabyte-grid scripts/evabyte_grid.sbatch
set -uo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 EVABYTE_NO_TRITON=1
cd /group/gquongrp/workspaces/baran/AvalancheBreak
MODEL=evabyte/EvaByte-SFT
echo "===== smoke test $(date)"
"$ENV/bin/python" scripts/smoke_evabyte.py "$MODEL" --no-triton 2>&1 | grep -vE "Warning|warn" | tee /tmp/evabyte_smoke_$SLURM_JOB_ID.txt | grep -E "^k=|loaded|FAILED|EVABYTE"
if ! grep -E "^k=[04] \[manual loop\]: truth" /tmp/evabyte_smoke_$SLURM_JOB_ID.txt | grep -qv FAILED; then
  echo "Manual loop did not produce answers; stopping before the grids."; exit 1
fi
for TASK in running_xor copy_previous; do
  echo "===== $TASK $(date)"; "$ENV/bin/python" -m tasks.run_task "$TASK" "$MODEL" grid 20
done
for TASK in random_permutation_digits cyclic_shift_digits; do
  echo "===== $TASK $(date)"; "$ENV/bin/python" -m tasks.run_digit_controls "$TASK" "$MODEL" 20
done
echo "Done: $(date)"
