#!/bin/bash
#SBATCH --job-name=qwen-jupyter
#SBATCH --account=quonbiogrp
#SBATCH --partition=gpu-a100-h
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/jupyter-%j.log

set -euo pipefail

ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
PORT=$(shuf -i 20000-29999 -n 1)

echo "=================================================="
echo "Node:  $(hostname)"
echo "Port:  ${PORT}"
echo "Time:  $(date)"
echo "=================================================="

source /home/fkhazaee/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

exec jupyter lab \
    --no-browser \
    --ip=0.0.0.0 \
    --port="${PORT}" \
    --notebook-dir=/group/gquongrp/workspaces/baran/AvalancheBreak
