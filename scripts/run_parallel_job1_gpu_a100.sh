#!/bin/bash
#SBATCH --job-name=parallel-job1-a100
#SBATCH --account=ctbrowngrp
#SBATCH --partition=gpu-a100-h
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/parallel-job1-%j.log

set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
cd /group/gquongrp/workspaces/baran/AvalancheBreak

echo "=================================================="
echo "PARALLEL JOB 1: Character Substitution + Running Sum Local"
echo "GPU: A100"
echo "Start Time: $(date)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# TASK 1: CHARACTER SUBSTITUTION
echo ""
echo "========== TASK 1: CHARACTER SUBSTITUTION (Alphabet Sweep) =========="
echo "[1/4] Qwen2.5-7B: Character Substitution"
"$ENV/bin/python" scripts/run_character_substitution_alphabet_sweep.py "Qwen/Qwen2.5-7B-Instruct"
echo "[2/4] Qwen2.5-32B: Character Substitution"
"$ENV/bin/python" scripts/run_character_substitution_alphabet_sweep.py "Qwen/Qwen2.5-32B-Instruct"

# TASK 2: RUNNING SUM LOCAL
echo ""
echo "========== TASK 2: RUNNING SUM (Local, Modulus Sweep) =========="
echo "[3/4] Qwen2.5-7B: Running Sum Local"
"$ENV/bin/python" scripts/run_running_sum_local_modulus_sweep.py "Qwen/Qwen2.5-7B-Instruct"
echo "[4/4] Qwen2.5-32B: Running Sum Local"
"$ENV/bin/python" scripts/run_running_sum_local_modulus_sweep.py "Qwen/Qwen2.5-32B-Instruct"

echo ""
echo "=================================================="
echo "JOB 1 COMPLETE: $(date)"
echo "=================================================="
