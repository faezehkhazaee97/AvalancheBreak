#!/bin/bash
# Batch script to run all 6 tasks in the tier-structured suite
# Each task runs N ∈ {4, 8, 16, 32, 64}, k ∈ {0, 1, 2, 3, 4, 8, 16}, 10 trials per config

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "TIER 1: Position-Local Tasks"
echo "=========================================="

echo "[1/6] Running Cyclic Shift..."
python run_cyclic_shift_structured_boundaries.py

echo "[2/6] Running Character Substitution..."
python run_character_substitution_structured_boundaries.py

echo "=========================================="
echo "TIER 2: Single-Previous Dependency Tasks"
echo "=========================================="

echo "[3/6] Running Running XOR..."
python run_running_xor_structured_boundaries.py

echo "[4/6] Running Running Sum (Local Window)..."
python run_running_sum_local_structured_boundaries.py

echo "=========================================="
echo "TIER 3: Global-State Tasks"
echo "=========================================="

echo "[5/6] Running Modsum (Running Sum Mod 10)..."
python run_modsum_structured_boundaries.py

echo "[6/6] Running Integer Addition..."
python run_addition_structured_boundaries.py

echo "=========================================="
echo "ALL TASKS COMPLETE"
echo "=========================================="
echo "Results saved to: results/Qwen2.5-7B-Instruct/"
echo "Files:"
echo "  - cyclic_shift_grid_structured_boundaries.csv"
echo "  - character_substitution_grid_structured_boundaries.csv"
echo "  - running_xor_grid_structured_boundaries.csv"
echo "  - running_sum_local_grid_structured_boundaries.csv"
echo "  - modsum_grid_structured_boundaries.csv"
echo "  - addition_grid_structured_boundaries.csv"
