#!/bin/bash
#SBATCH --job-name=job-titan-7b
#SBATCH --account=ctbrowngrp
#SBATCH --partition=gpuh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/job-titan-7b-%j.log

set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
cd /group/gquongrp/workspaces/baran/AvalancheBreak

echo "=================================================="
echo "JOB B: Titan GPU - ALL 7B MODELS"
echo "Start Time: $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "========== CHAR SUB 7B =========="
"$ENV/bin/python" scripts/run_character_substitution_alphabet_sweep.py "Qwen/Qwen2.5-7B-Instruct"

echo ""
echo "========== RUNNING XOR 7B =========="
"$ENV/bin/python" scripts/run_running_xor_structured_boundaries.py

echo ""
echo "========== RUNNING SUM LOCAL 7B =========="
"$ENV/bin/python" scripts/run_running_sum_local_modulus_sweep.py "Qwen/Qwen2.5-7B-Instruct"

echo ""
echo "========== MODSUM 7B =========="
"$ENV/bin/python" scripts/run_modsum_modulus_sweep.py "Qwen/Qwen2.5-7B-Instruct"

echo ""
echo "========== INTEGER ADDITION 7B =========="
"$ENV/bin/python" scripts/run_addition_digit_range_sweep.py "Qwen/Qwen2.5-7B-Instruct"

echo ""
echo "=================================================="
echo "JOB B COMPLETE: $(date)"
echo "=================================================="
