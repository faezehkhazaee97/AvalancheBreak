#!/bin/bash
# Second seed batch for the Qwen2.5-32B results that are still single-batch
# (copy_previous, conditional_shift, running_sum_local, sliding_window_max).
# 32B is the paper's primary model, and copy_previous 32B sits behind
# Figures 1(b) and 3, so these need the same two-batch standard as 7B.
#
# Each job runs NEW trial indices (seeds derive from the trial index) and
# writes results_clean/<task>/grid/Qwen2.5-32B-Instruct_trials<start>-<end-1>.csv,
# so batch 1 is never overwritten. Figure scripts that glob
# Qwen2.5-32B-Instruct*.csv pool the new file automatically.
#
# Batch-1 sizes: copy_previous 40 trials, conditional_shift 20,
# running_sum_local 40, sliding_window_max 20 -> batch 2 matches each.
# Past 32B grid jobs used 110G and ran 50 min (copy_previous) to 2h20
# (running_sum_local); time limits below leave ~2x headroom.
#
# These use the quonbiogrp A100 allocation (gpu-a100-h). Check that other
# work isn't waiting on it first:  squeue -A quonbiogrp -p gpu-a100-h
#
# Run:  bash scripts/launch_32b_second_batch.sh
set -euo pipefail
cd /group/gquongrp/workspaces/baran/AvalancheBreak
source /etc/profile.d/modules.sh 2>/dev/null || true
export PATH=/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/bin:$PATH

M=Qwen/Qwen2.5-32B-Instruct
S=tasks/run_task_second_batch.sbatch

sbatch --job-name=2nd-copy_previous-32b      --mem=110G --time=02:00:00 "$S" copy_previous      "$M" 40 80
sbatch --job-name=2nd-conditional_shift-32b  --mem=110G --time=02:00:00 "$S" conditional_shift  "$M" 20 40
sbatch --job-name=2nd-running_sum_local-32b  --mem=110G --time=04:30:00 "$S" running_sum_local  "$M" 40 80
sbatch --job-name=2nd-sliding_window_max-32b --mem=110G --time=02:30:00 "$S" sliding_window_max "$M" 20 40
