#!/bin/bash
#SBATCH --account=quonbiogrp
#SBATCH --partition=gpuh
#SBATCH --gres=gpu:titan:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%j.log
# Smoke-test the byte-level EvaByte-SFT twice: with its Triton kernels, then with its PyTorch fallback.
#   sbatch --job-name=smoke-evabyte scripts/smoke_evabyte.sbatch
set -uo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd /group/gquongrp/workspaces/baran/AvalancheBreak
echo "===== attempt A: triton kernels"
"$ENV/bin/python" scripts/smoke_evabyte.py evabyte/EvaByte-SFT 2>&1 | grep -vE "Warning|warn" | tail -8
echo "===== attempt B: pytorch fallback"
"$ENV/bin/python" scripts/smoke_evabyte.py evabyte/EvaByte-SFT --no-triton 2>&1 | grep -vE "Warning|warn" | tail -8
echo "Done: $(date)"
