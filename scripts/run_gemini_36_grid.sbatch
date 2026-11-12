#!/bin/bash
#SBATCH --account=publicgrp
#SBATCH --partition=low                # CPU only: this job makes API calls
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/group/gquongrp/workspaces/baran/AvalancheBreak/slurm_logs/%x-%j.log

# Full Gemini grid on the model that gemini-3.5-flash now resolves to (gemini-3.6-flash):
# letters, dependency tasks, and both tokenization controls, N <= 64, one output file per trial.
# The key is read from ~/.config/gemini/api_key at run time and is never written to this file.
#   sbatch --job-name=gemini36 scripts/run_gemini_36_grid.sbatch <first_trial> <last_trial>
set -euo pipefail
cd /group/gquongrp/workspaces/baran/AvalancheBreak
export GEMINI_API_KEY="$(cat "$HOME/.config/gemini/api_key")"
PY=.conda/envs/qwen-focus/bin/python
TASKS="cyclic_shift,random_permutation,conditional_shift,copy_previous,running_xor,running_sum_local,sliding_window_max,cyclic_shift_digits,random_permutation_digits,conditional_shift_digits,cyclic_shift_spaced,random_permutation_spaced,conditional_shift_spaced"
for T in $(seq "${1:-0}" "${2:-2}"); do
  OUT="results_clean/gemini_pilot/gemini-3.6-flash_full_grid_trial${T}.csv"
  if [ -s "$OUT" ]; then echo "trial $T exists, skipping"; continue; fi
  echo "=== trial $T start $(date)"
  "$PY" -m tasks.common.gemini_full_grid --model gemini-3.5-flash --max-n 64 --tasks "$TASKS" --trial "$T" --output "$OUT" 2>&1 | grep -viE "api.?key" | tail -3
  "$PY" -c "
import pandas as pd; d=pd.read_csv('$OUT',dtype=str)
print('trial $T rows',len(d),'| resolved',d.Model_Resolved.unique().tolist(),'| errors',(d.Error.fillna('')!='').sum(),'| tokens',d.Total_Tokens.astype(float).sum())"
done
echo "Done: $(date)"
