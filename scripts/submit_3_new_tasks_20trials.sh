#!/bin/bash
# Submit 3 NEW tasks with 20 trials each to GPU queue
# (Cyclic shift already has results, so skipped)

set -e

echo "=================================================="
echo "Submitting 3 NEW tasks (20 trials each)"
echo "Model: Qwen2.5-7B-Instruct"
echo "=================================================="

cd "$(dirname "$0")"

echo ""
echo "[1/3] Submitting Character Substitution..."
JOB1=$(sbatch run_character_substitution_structured_boundaries_batch.sh | awk '{print $NF}')
echo "Job ID: $JOB1"

echo ""
echo "[2/3] Submitting Running XOR..."
JOB2=$(sbatch run_running_xor_structured_boundaries_batch.sh | awk '{print $NF}')
echo "Job ID: $JOB2"

echo ""
echo "[3/3] Submitting Running Sum (Local Window)..."
JOB3=$(sbatch run_running_sum_local_structured_boundaries_batch.sh | awk '{print $NF}')
echo "Job ID: $JOB3"

echo ""
echo "=================================================="
echo "All 3 jobs submitted!"
echo "=================================================="
echo "Job IDs: $JOB1, $JOB2, $JOB3"
echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo ""
echo "Check logs:"
echo "  tail -f ../slurm_logs/char-sub-structured-$JOB1.log"
echo "  tail -f ../slurm_logs/running-xor-structured-$JOB2.log"
echo "  tail -f ../slurm_logs/running-sum-local-structured-$JOB3.log"
echo ""
echo "Results will be saved to:"
echo "  results/Qwen2.5-7B-Instruct/"
echo "  - character_substitution_grid_structured_boundaries.csv"
echo "  - running_xor_grid_structured_boundaries.csv"
echo "  - running_sum_local_grid_structured_boundaries.csv"
