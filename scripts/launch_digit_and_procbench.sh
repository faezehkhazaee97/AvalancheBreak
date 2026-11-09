#!/bin/bash
# Queue the tokenization control (digit-alphabet position-local tasks, 7 models) and the
# ProcBench regrid (7B, 32B). Large models need A100s; 7B-class models run on the
# group's Titan GPUs (partition gpuh, 24 GB). (Submitting one job to both partitions is
# rejected: the account is tied to one partition per association.)
#   bash scripts/launch_digit_and_procbench.sh
set -euo pipefail
cd /group/gquongrp/workspaces/baran/AvalancheBreak
export PATH=/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/bin:$PATH
SMALL=(--partition=gpuh --gres=gpu:titan:1 --cpus-per-task=6 --mem=48G)

# Most important first: Qwen2.5-32B decides the framing.
for M in Qwen/Qwen2.5-32B-Instruct Qwen/Qwen3.8-27B google/gemma-4-31B-it; do
  echo "$M digits: $(sbatch --parsable --job-name=digits-${M##*/} scripts/run_digit_controls.sbatch "$M")"
done
echo "Qwen2.5-14B digits: $(sbatch --parsable --job-name=digits-Qwen2.5-14B-Instruct --mem=64G --time=05:00:00 scripts/run_digit_controls.sbatch Qwen/Qwen2.5-14B-Instruct)"
for M in Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.3 microsoft/Phi-3.5-mini-instruct; do
  echo "$M digits: $(sbatch --parsable --job-name=digits-${M##*/} "${SMALL[@]}" --time=05:00:00 scripts/run_digit_controls.sbatch "$M")"
done
echo "ProcBench 32B: $(sbatch --parsable --job-name=procbench-Qwen2.5-32B-Instruct scripts/regrid_procbench.sbatch Qwen/Qwen2.5-32B-Instruct)"
echo "ProcBench 7B: $(sbatch --parsable --job-name=procbench-Qwen2.5-7B-Instruct "${SMALL[@]}" --time=03:00:00 scripts/regrid_procbench.sbatch Qwen/Qwen2.5-7B-Instruct)"
