#!/bin/bash
#SBATCH --account=publicgrp
#SBATCH --partition=low
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%j.log

# Fetch a Hugging Face checkpoint into the shared cache from a compute node (the login node
# must not carry a 15 GB download). Weights only, no training artifacts.
#   sbatch --job-name=dl-Qwen2.5-7B scripts/download_model.sbatch Qwen/Qwen2.5-7B
set -euo pipefail
ENV=/group/gquongrp/workspaces/baran/AvalancheBreak/.conda/envs/qwen-focus
export HF_HOME=/group/gquongrp/workspaces/baran/AvalancheBreak/hf_cache
unset HF_HUB_OFFLINE
cd /group/gquongrp/workspaces/baran/AvalancheBreak
"$ENV/bin/python" - "$1" <<'EOF'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.py", "*.model", "*.tiktoken"])
print("downloaded to", p)
EOF
echo "Done: $(date)"
