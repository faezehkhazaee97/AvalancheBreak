#!/bin/bash
# Queue the spaced-letter tokenization control (3 position-local tasks x 7 models).
# Large models on A100s; 7B-class models on the group's Titan GPUs (partition gpuh).
#   bash scripts/launch_spaced_controls.sh
set -euo pipefail
cd /group/gquongrp/workspaces/baran/AvalancheBreak
export PATH=/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/bin:$PATH
SMALL=(--partition=gpuh --gres=gpu:titan:1 --cpus-per-task=6 --mem=48G --time=05:00:00)
for M in Qwen/Qwen2.5-32B-Instruct Qwen/Qwen3.8-27B google/gemma-4-31B-it; do
  echo "$M spaced: $(sbatch --parsable --job-name=spaced-${M##*/} scripts/run_digit_controls.sbatch "$M" spaced)"
done
echo "Qwen2.5-14B spaced: $(sbatch --parsable --job-name=spaced-Qwen2.5-14B-Instruct --mem=64G --time=05:00:00 scripts/run_digit_controls.sbatch Qwen/Qwen2.5-14B-Instruct spaced)"
for M in Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.3 microsoft/Phi-3.5-mini-instruct; do
  echo "$M spaced: $(sbatch --parsable --job-name=spaced-${M##*/} "${SMALL[@]}" scripts/run_digit_controls.sbatch "$M" spaced)"
done
