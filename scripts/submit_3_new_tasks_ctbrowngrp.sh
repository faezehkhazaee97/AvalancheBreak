#!/bin/bash
# Submit 3 NEW tasks (20 trials each) to ctbrowngrp on gpu-a100-h

set -e

echo "=================================================="
echo "Submitting 3 NEW tasks to ctbrowngrp"
echo "Partition: gpu-a100-h (A100 GPU)"
echo "Model: Qwen2.5-7B-Instruct"
echo "Trials: 20 per config"
echo "=================================================="

cd "$(dirname "$0")"

echo ""
echo "[1/3] Submitting Character Substitution..."
JOB1=$(sbatch run_character_substitution_ctbrowngrp_batch.sh | awk '{print $NF}')
echo "Job ID: $JOB1"

echo ""
echo "[2/3] Submitting Running XOR..."
JOB2=$(sbatch run_running_xor_ctbrowngrp_batch.sh | awk '{print $NF}')
echo "Job ID: $JOB2"

echo ""
echo "[3/3] Submitting Running Sum (Local Window)..."
JOB3=$(sbatch run_running_sum_local_ctbrowngrp_batch.sh | awk '{print $NF}')
echo "Job ID: $JOB3"

echo ""
echo "=================================================="
echo "All 3 jobs submitted to ctbrowngrp!"
echo "=================================================="
echo "Job IDs: $JOB1, $JOB2, $JOB3"
echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  squeue -A ctbrowngrp"
echo ""
echo "Check logs:"
echo "  tail -f ../slurm_logs/char-sub-ctbrown-$JOB1.log"
echo "  tail -f ../slurm_logs/running-xor-ctbrown-$JOB2.log"
echo "  tail -f ../slurm_logs/running-sum-local-ctbrown-$JOB3.log"
