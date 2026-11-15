#!/bin/bash
#SBATCH --account=quonbiogrp
#SBATCH --partition=gpuh
#SBATCH --gres=gpu:titan:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%j.log
# Byte-level check with EvaByte-SFT: smoke test on the PyTorch attention path, then, only if the model
# answers the model card's example prompt coherently, the standard grids on two dependency tasks and
# two position-local digit tasks.
#   sbatch --job-name=evabyte-grid scripts/evabyte_grid.sbatch
set -uo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 EVABYTE_NO_TRITON=1
cd /group/gquongrp/workspaces/baran/AvalancheBreak
MODEL=evabyte/EvaByte-SFT
echo "===== smoke test $(date)"
"$ENV/bin/python" scripts/smoke_evabyte.py "$MODEL" --no-triton 2>&1 | grep -vE "Warning|warn" | tee /tmp/evabyte_smoke_$SLURM_JOB_ID.txt | grep -E "^(k=|card prompt|loaded|EVABYTE|EvaByte|SANITY)|FAILED"
if ! grep -q "^SANITY PASS" /tmp/evabyte_smoke_$SLURM_JOB_ID.txt; then
  echo "The model did not answer the model card's prompt coherently; stopping before the grids."; exit 1
fi
# results of an earlier run with uninitialized rotary tables are noise; keep them out of results_clean
for TASK in running_xor copy_previous random_permutation_digits cyclic_shift_digits; do
  F=results_clean/$TASK/grid/EvaByte-SFT.csv
  if [ -e "$F" ]; then
    mkdir -p slurm_logs/evabyte_superseded_$SLURM_JOB_ID && mv "$F" slurm_logs/evabyte_superseded_$SLURM_JOB_ID/$TASK.csv
    echo "moved an earlier $F to slurm_logs/evabyte_superseded_$SLURM_JOB_ID/"
  fi
done
for TASK in running_xor copy_previous; do
  echo "===== $TASK $(date)"; "$ENV/bin/python" -m tasks.run_task "$TASK" "$MODEL" grid 20
done
for TASK in random_permutation_digits cyclic_shift_digits; do
  echo "===== $TASK $(date)"; "$ENV/bin/python" -m tasks.run_digit_controls "$TASK" "$MODEL" 20
done
echo "Done: $(date)"
