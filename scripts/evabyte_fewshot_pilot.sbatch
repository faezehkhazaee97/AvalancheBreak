#!/bin/bash
#SBATCH --account=ctbrowngrp
#SBATCH --partition=low
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-12-92
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%j.log
# Step 1 of the byte-level few-shot check: the existing smoke test (the checkpoint must still load
# and answer the model card's pangram prompt), then a small three-way comparison on one task at one
# length -- zero-shot chat (what Appendix L ran), 3-shot chat, 3-shot raw completion -- to see
# whether demonstrations stop EvaByte copying its input before any full grid is run.
#   sbatch --job-name=evabyte-fs-pilot scripts/evabyte_fewshot_pilot.sbatch
# partition low (preemptible, ctbrowngrp QOS): the gpuh titans and the ctbrowngrp A100 slot are
# taken by this user's other jobs, which must not be disturbed; a low job cannot preempt them.
# --exclude=gpu-12-92: its single V100 has 16 GB, too little for 6.5B in bf16 with a long prompt.
set -uo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1 EVABYTE_NO_TRITON=1
cd /group/gquongrp/workspaces/baran/AvalancheBreak
MODEL=evabyte/EvaByte-SFT
echo "Node: $(hostname) | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

SMOKE=slurm_logs/evabyte_fs_smoke_$SLURM_JOB_ID.txt
echo "===== smoke test $(date)"
"$ENV/bin/python" scripts/smoke_evabyte.py "$MODEL" --no-triton 2>&1 | grep -vE "Warning|warn" \
  | tee "$SMOKE" | grep -E "^(k=|card prompt|loaded|EVABYTE|EvaByte|SANITY)|FAILED"
if ! grep -q "^SANITY PASS" "$SMOKE"; then
  echo "The model did not answer the model card's prompt coherently; stopping before the pilot."
  exit 1
fi

echo "===== pilot $(date)"
# Arguments after the script name choose the tasks, lengths, alphabet and trial count:
#   sbatch ... scripts/evabyte_fewshot_pilot.sbatch --task running_xor,cyclic_shift_digits --N 32 --trials 20
# (not --export: sbatch splits an --export value on commas, which silently truncates a task list).
if [ $# -gt 0 ]; then ARGS=("$@"); else ARGS=(--task running_xor --N 16 --trials 5); fi
"$ENV/bin/python" scripts/run_evabyte_fewshot.py --pilot "${ARGS[@]}"
echo "Done: $(date)"
