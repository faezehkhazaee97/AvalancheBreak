#!/bin/bash
#SBATCH --account=publicgrp
#SBATCH --partition=low
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%j.log
# Run a CPU-only python script with network access (downloads, parsing).
#   sbatch --job-name=<name> scripts/run_cpu_script.sbatch scripts/<script>.py [args]
set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
unset HF_HUB_OFFLINE
cd /group/gquongrp/workspaces/baran/AvalancheBreak
"$ENV/bin/python" "$@"
echo "Done: $(date)"
