#!/bin/bash
#SBATCH --job-name=parallel-job2-titan
#SBATCH --account=ctbrowngrp
#SBATCH --partition=gpuh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/parallel-job2-%j.log

set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
cd /group/gquongrp/workspaces/baran/AvalancheBreak

echo "=================================================="
echo "PARALLEL JOB 2: Running XOR + Modsum + Integer Addition"
echo "GPU: Titan (gpuh)"
echo "Start Time: $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# TASK 1: RUNNING XOR
echo ""
echo "========== TASK 1: RUNNING XOR (No Sweep) =========="
echo "[1/6] Qwen2.5-7B: Running XOR"
"$ENV/bin/python" scripts/run_running_xor_structured_boundaries.py
echo "[2/6] Qwen2.5-32B: Running XOR"
sed 's/MODEL_ID = "Qwen\/Qwen2.5-7B-Instruct"/MODEL_ID = "Qwen\/Qwen2.5-32B-Instruct"/' scripts/run_running_xor_structured_boundaries.py > /tmp/xor_32b.py
"$ENV/bin/python" /tmp/xor_32b.py

# TASK 2: MODSUM
echo ""
echo "========== TASK 2: MODSUM (Modulus Sweep) =========="
echo "[3/6] Qwen2.5-7B: Modsum"
"$ENV/bin/python" scripts/run_modsum_modulus_sweep.py "Qwen/Qwen2.5-7B-Instruct"
echo "[4/6] Qwen2.5-32B: Modsum"
"$ENV/bin/python" scripts/run_modsum_modulus_sweep.py "Qwen/Qwen2.5-32B-Instruct"

# TASK 3: INTEGER ADDITION
echo ""
echo "========== TASK 3: INTEGER ADDITION (Digit Range Sweep) =========="
echo "[5/6] Qwen2.5-7B: Integer Addition"
"$ENV/bin/python" scripts/run_addition_digit_range_sweep.py "Qwen/Qwen2.5-7B-Instruct"
echo "[6/6] Qwen2.5-32B: Integer Addition"
"$ENV/bin/python" scripts/run_addition_digit_range_sweep.py "Qwen/Qwen2.5-32B-Instruct"

echo ""
echo "=================================================="
echo "JOB 2 COMPLETE: $(date)"
echo "=================================================="
