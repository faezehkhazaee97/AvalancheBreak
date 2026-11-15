#!/bin/bash
#SBATCH --account=publicgrp
#SBATCH --partition=low
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%j.log
# Add peft (LoRA) to the project's own conda environment from a compute node.
set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
"$ENV/bin/pip" install --no-cache-dir "peft>=0.14" && "$ENV/bin/python" -c "import peft; print('peft', peft.__version__)"
