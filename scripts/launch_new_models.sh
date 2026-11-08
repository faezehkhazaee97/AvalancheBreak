#!/bin/bash
# Adds two current-generation open models to the study:
#   Qwen/Qwen3.8-27B       newest dense Qwen (Aug 2026): continues the Qwen2.5 scale story
#   google/gemma-4-31B-it  current Gemma (Mar 2026): a second, independent family
# Both are ungated (no Hugging Face token needed) and fit on one A100 80GB in bf16.
#
# For each model: a smoke job (weights download + 1 trial/cell on two tasks +
# output validation), then the 11-task core grid as an array that only starts
# if the smoke job passes (afterok; cancelled automatically if it fails).
#
# Run:  bash scripts/launch_new_models.sh
set -euo pipefail
cd /group/gquongrp/workspaces/baran/AvalancheBreak
export PATH=/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/bin:$PATH

for MODEL_ID in Qwen/Qwen3.8-27B google/gemma-4-31B-it; do
  TAG="${MODEL_ID##*/}"
  SMOKE=$(sbatch --parsable --job-name="smoke-$TAG" scripts/run_new_model_smoke.sbatch "$MODEL_ID")
  GRID=$(sbatch --parsable --job-name="grid-$TAG" --dependency=afterok:"$SMOKE" --kill-on-invalid-dep=yes \
         scripts/run_new_model_grid.sbatch "$MODEL_ID")
  echo "$MODEL_ID: smoke job $SMOKE -> grid array $GRID"
done
