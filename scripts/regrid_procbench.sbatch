#!/bin/bash
#SBATCH --account=quonbiogrp          # group allocation (4 A100s max at once)
#SBATCH --partition=gpu-a100-h        # A100 80GB nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=04:00:00               # 670 rows: ~15 min at 7B, ~1.5 h at 32B
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%j.log

# Regenerate the whole ProcBench-Substitute grid (N = 4..128, 20 trials/cell) for one model.
# Why: the vendored ProcBench generator drew its substitution table from an unordered set,
# so the table (and the ground truth) depended on the per-process hash seed and could not
# be rebuilt from the seed. generators.py now sorts the set; this run uses the fixed
# generator and the fixed k=1 token budget. The previous files are kept in
# results_clean/procbench_substitute/grid_nondeterministic_table/.
#   sbatch --job-name=procbench-<tag> scripts/regrid_procbench.sbatch <hf_model_id>

set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd /group/gquongrp/workspaces/baran/AvalancheBreak

MODEL_ID="$1"
echo "Node: $(hostname) | Model: $MODEL_ID | Task: procbench_substitute | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ENV/bin/python" -m tasks.run_task procbench_substitute "$MODEL_ID" grid 20
echo "Done: $(date)"
