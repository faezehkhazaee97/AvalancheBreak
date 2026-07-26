#!/bin/bash
# Submit all 4 new tasks with 20 trials each to GPU queue

set -e

echo "=================================================="
echo "Submitting 4 new tasks (20 trials each)"
echo "Model: Qwen2.5-7B-Instruct"
echo "=================================================="

cd "$(dirname "$0")"

echo ""
echo "[1/4] Submitting Cyclic Shift..."
JOB1=$(sbatch run_cyclic_shift_structured_boundaries_batch.sh | awk '{print $NF}')
echo "Job ID: $JOB1"

echo ""
echo "[2/4] Submitting Character Substitution..."
JOB2=$(sbatch run_character_substitution_structured_boundaries_batch.sh | awk '{print $NF}')
echo "Job ID: $JOB2"

echo ""
echo "[3/4] Submitting Running XOR..."
JOB3=$(sbatch run_running_xor_structured_boundaries_batch.sh | awk '{print $NF}')
echo "Job ID: $JOB3"

echo ""
echo "[4/4] Submitting Running Sum (Local)..."
JOB4=$(sbatch run_running_sum_local_structured_boundaries_batch.sh | awk '{print $NF}')
echo "Job ID: $JOB4"

echo ""
echo "=================================================="
echo "All jobs submitted!"
echo "=================================================="
echo "Job IDs: $JOB1, $JOB2, $JOB3, $JOB4"
echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo ""
echo "Check logs:"
echo "  tail -f ../slurm_logs/cyclic-shift-structured-$JOB1.log"
echo "  tail -f ../slurm_logs/char-sub-structured-$JOB2.log"
echo "  tail -f ../slurm_logs/running-xor-structured-$JOB3.log"
echo "  tail -f ../slurm_logs/running-sum-local-structured-$JOB4.log"
echo ""
echo "Results will be saved to:"
echo "  results/Qwen2.5-7B-Instruct/"
echo "  - cyclic_shift_grid_structured_boundaries.csv"
echo "  - character_substitution_grid_structured_boundaries.csv"
echo "  - running_xor_grid_structured_boundaries.csv"
echo "  - running_sum_local_grid_structured_boundaries.csv"
