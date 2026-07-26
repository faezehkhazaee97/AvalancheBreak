#!/bin/bash
#SBATCH --job-name=full-grid-sweep
#SBATCH --account=ctbrowngrp
#SBATCH --partition=gpu-a100-h
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/full-grid-sweep-%j.log

set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
cd /group/gquongrp/workspaces/baran/AvalancheBreak

echo "=================================================="
echo "FULL GRID SWEEP: ALL TASKS, BOTH MODELS"
echo "Start Time: $(date)"
echo "Account: ctbrowngrp"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ============================================================
# TASK 1: CHARACTER SUBSTITUTION (Alphabet Sweep)
# ============================================================
echo ""
echo "========== TASK 1: CHARACTER SUBSTITUTION (Alphabet Sweep A∈{2,4,8}) =========="

echo "[1/5] Qwen2.5-7B: Character Substitution"
echo "Start: $(date)"
"$ENV/bin/python" scripts/run_character_substitution_alphabet_sweep.py "Qwen/Qwen2.5-7B-Instruct"
echo "Done: $(date)"

echo "[2/5] Qwen2.5-32B: Character Substitution"
echo "Start: $(date)"
"$ENV/bin/python" scripts/run_character_substitution_alphabet_sweep.py "Qwen/Qwen2.5-32B-Instruct"
echo "Done: $(date)"

# ============================================================
# TASK 2: RUNNING XOR (No Sweep)
# ============================================================
echo ""
echo "========== TASK 2: RUNNING XOR (A=2, Binary, No Sweep) =========="

echo "[3/5] Qwen2.5-7B: Running XOR"
echo "Start: $(date)"
"$ENV/bin/python" scripts/run_running_xor_structured_boundaries.py
echo "Done: $(date)"

echo "[4/5] Qwen2.5-32B: Running XOR"
echo "Start: $(date)"
sed 's/MODEL_ID = "Qwen\/Qwen2.5-7B-Instruct"/MODEL_ID = "Qwen\/Qwen2.5-32B-Instruct"/' scripts/run_running_xor_structured_boundaries.py > /tmp/run_xor_32b.py
"$ENV/bin/python" /tmp/run_xor_32b.py
echo "Done: $(date)"

# ============================================================
# TASK 3: RUNNING SUM (Local Window, Modulus Sweep)
# ============================================================
echo ""
echo "========== TASK 3: RUNNING SUM (Local Window, Modulus Sweep M∈{2,4,8}) =========="

echo "[5/5] Qwen2.5-7B: Running Sum (Local, Modulus Sweep)"
echo "Start: $(date)"
"$ENV/bin/python" scripts/run_running_sum_local_modulus_sweep.py "Qwen/Qwen2.5-7B-Instruct"
echo "Done: $(date)"

echo "[6/10] Qwen2.5-32B: Running Sum (Local, Modulus Sweep)"
echo "Start: $(date)"
"$ENV/bin/python" scripts/run_running_sum_local_modulus_sweep.py "Qwen/Qwen2.5-32B-Instruct"
echo "Done: $(date)"

# ============================================================
# TASK 4: MODSUM (Modulus Sweep)
# ============================================================
echo ""
echo "========== TASK 4: MODSUM (Running Sum Mod M, Modulus Sweep M∈{2,4,8}) =========="

echo "[7/10] Qwen2.5-7B: Modsum (Modulus Sweep)"
echo "Start: $(date)"
"$ENV/bin/python" scripts/run_modsum_modulus_sweep.py "Qwen/Qwen2.5-7B-Instruct"
echo "Done: $(date)"

echo "[8/10] Qwen2.5-32B: Modsum (Modulus Sweep)"
echo "Start: $(date)"
"$ENV/bin/python" scripts/run_modsum_modulus_sweep.py "Qwen/Qwen2.5-32B-Instruct"
echo "Done: $(date)"

# ============================================================
# TASK 5: INTEGER ADDITION (Digit Range Sweep)
# ============================================================
echo ""
echo "========== TASK 5: INTEGER ADDITION (Digit Range Sweep M∈{2,4,8}) =========="

echo "[9/10] Qwen2.5-7B: Integer Addition (Digit Range Sweep)"
echo "Start: $(date)"
"$ENV/bin/python" scripts/run_addition_digit_range_sweep.py "Qwen/Qwen2.5-7B-Instruct"
echo "Done: $(date)"

echo "[10/10] Qwen2.5-32B: Integer Addition (Digit Range Sweep)"
echo "Start: $(date)"
"$ENV/bin/python" scripts/run_addition_digit_range_sweep.py "Qwen/Qwen2.5-32B-Instruct"
echo "Done: $(date)"

echo ""
echo "=================================================="
echo "ALL TASKS COMPLETE!"
echo "End Time: $(date)"
echo "=================================================="
echo "Results saved to:"
echo "  results/Qwen2.5-7B-Instruct/"
echo "  results/Qwen2.5-32B-Instruct/"
