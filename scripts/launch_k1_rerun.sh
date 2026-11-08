#!/bin/bash
# Queue the k=1 budget rerun for every local model in the paper (one job per model).
# Estimates (rows at N=64/128/256, measured speeds): Phi ~1 h, 7B/14B/Mistral ~2 h,
# Qwen3.8/Gemma ~3.5 h, Qwen2.5-32B ~5.5 h; limits are about twice that. The group's
# 4-GPU cap means some jobs wait for others.
#   bash scripts/launch_k1_rerun.sh
set -euo pipefail
cd /group/gquongrp/workspaces/baran/AvalancheBreak
export PATH=/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/bin:$PATH

#        model id                              time      mem
JOBS=("Qwen/Qwen2.5-32B-Instruct               12:00:00  110G"
      "Qwen/Qwen3.8-27B                        08:00:00  110G"
      "google/gemma-4-31B-it                   08:00:00  110G"
      "Qwen/Qwen2.5-7B-Instruct                05:00:00  64G"
      "Qwen/Qwen2.5-14B-Instruct               05:00:00  64G"
      "mistralai/Mistral-7B-Instruct-v0.3      05:00:00  64G"
      "microsoft/Phi-3.5-mini-instruct         03:00:00  64G")

for j in "${JOBS[@]}"; do
  read -r MODEL_ID TIME MEM <<< "$j"
  TAG="${MODEL_ID##*/}"
  ID=$(sbatch --parsable --job-name="k1fix-$TAG" --time="$TIME" --mem="$MEM" \
       scripts/rerun_capped_k1.sbatch "$MODEL_ID")
  echo "$MODEL_ID: job $ID"
done
