# Boundary-Token Task Suite: Complete Scripts

All 6 task scripts are ready to run on GPU. Each script follows the same structure and generates a CSV results file.

---

## Quick Start

### Run All Tasks at Once
```bash
cd scripts/
bash run_all_tasks_structured_boundaries.sh
```

This runs all 6 tasks sequentially and saves results to `results/Qwen2.5-7B-Instruct/`.

### Run Individual Task
```bash
cd scripts/
python run_<task_name>_structured_boundaries.py
```

---

## Tasks Overview

### TIER 1: Position-Local (output depends only on current input)

**Task 1.1: Cyclic Character Shift**
- Script: `run_cyclic_shift_structured_boundaries.py`
- Input: Random A-Z characters
- Output: Each character shifted by 1 (Z→A wraps)
- Example: `YZ|AB` → `ZA|BC`

**Task 1.2: Character Substitution**
- Script: `run_character_substitution_structured_boundaries.py`
- Input: Random A-Z characters
- Output: Each character mapped via fixed cipher (A↔X, B↔Y, C↔Z, D↔W, E↔P, F↔Q, G↔R, H↔S, I↔T, J↔U, K↔V, L↔M, N↔O)
- Example: `ABCD|EFGH` → `XYZW|PQRS`

---

### TIER 2: Single-Previous (output depends on current input + previous input)

**Task 2.1: Running XOR**
- Script: `run_running_xor_structured_boundaries.py`
- Input: Random binary digits (0, 1)
- Output: `output[i] = input[i] XOR input[i-1]` (first = first input)
- Example: `1011|0110` → `1110|1101`

**Task 2.2: Running Sum (Local Window)**
- Script: `run_running_sum_local_structured_boundaries.py`
- Input: Random decimal digits (0-9)
- Output: `output[i] = (input[i] + input[i-1]) mod 10` (first = first input)
- Example: `2314|5298` → `2545|9717`

---

### TIER 3: Global-State (output depends on all previous inputs)

**Task 3.1: Running Sum Mod 10 (Modsum) — Direction-Aligned**
- Script: `run_modsum_structured_boundaries.py`
- Input: Random decimal digits (0-9)
- Output: `output[i] = (sum of all inputs 0 to i) mod 10`
- Example: `2314|5298` → `2560|5764`
- Note: Cumulative operation; direction-aligned but shows NO error-recovery benefit

**Task 3.2: Integer Addition — Direction-Misaligned**
- Script: `run_addition_structured_boundaries.py`
- Input: Two N-digit numbers separated by `+`
- Output: Their sum (N digits, zero-padded)
- Example: `12|34+56|78` → `6912`
- Note: Carry-based; direction-misaligned with boundaries, actively hurts performance

---

## Experimental Grid Parameters

All tasks use:
- **Sequence lengths:** N ∈ {4, 8, 16, 32, 64}
- **Boundary intervals:** k ∈ {0, 1, 2, 3, 4, 8, 16}
  - k=0: No boundaries (baseline)
  - k≥1: Insert boundary every k positions within sequence
- **Trials per config:** 10
- **Model:** Qwen2.5-7B-Instruct (chat-templated prompting)
- **Total evaluations per task:** 5 × 7 × 10 = 350 samples

---

## Output Files

Each script saves a CSV to `results/Qwen2.5-7B-Instruct/`:

| Task | Output File |
|------|------------|
| Cyclic Shift | `cyclic_shift_grid_structured_boundaries.csv` |
| Character Substitution | `character_substitution_grid_structured_boundaries.csv` |
| Running XOR | `running_xor_grid_structured_boundaries.csv` |
| Running Sum Local | `running_sum_local_grid_structured_boundaries.csv` |
| Modsum | `modsum_grid_structured_boundaries.csv` |
| Integer Addition | `addition_grid_structured_boundaries.csv` |

### CSV Columns

All output files have identical structure:
```
Method              "Base" (k=0) or "Cellular_Bounds" (k>0)
Interval            k value (0, 1, 2, 3, 4, 8, 16)
Seq_Length          N value (4, 8, 16, 32, 64)
Trial               Trial number (0-9)
TAR                 Total Token Accuracy (% correct digits)
SAR                 Sequence Accuracy Rate (1.0 if all correct, 0.0 otherwise)
First_Error_Pos     Position of first incorrect digit (or len(truth) if perfect)
Input_Given         The actual input string with boundaries
Ground_Truth        The correct output string (with boundaries)
LLM_Output          What the model predicted (truncated to len(truth))
```

---

## Usage Example: Single Task with Logging

```bash
cd scripts/
python run_cyclic_shift_structured_boundaries.py 2>&1 | tee cyclic_shift.log
```

This saves both stdout and stderr to `cyclic_shift.log` while displaying real-time progress.

---

## Hardware Requirements

- **GPU:** At least 1× GPU with ≥16GB VRAM (Qwen2.5-7B in bfloat16)
- **Time per task:** ~5-15 minutes (depending on GPU speed)
- **Total time for all 6 tasks:** ~30-90 minutes

---

## Debugging / Single Trial

To test a single trial for debugging:

```python
# In any script, replace the loop with:
task = (N=8, k=2, t=0)  # Example: 8-char sequence, boundary interval 2, trial 0
result = run_<task>_trial(task)
print(result)
```

---

## Notes

1. **Seeding:** All scripts use fixed global seed (42) and deterministic local seeds per task config to ensure reproducibility.

2. **Chat template:** All prompts use `tokenizer.apply_chat_template()` to match production Qwen2.5-7B-Instruct inference behavior.

3. **Boundary placement:** Boundaries are inserted **within each operand separately** (for multi-operand tasks like addition), not uniformly across the concatenated input. This respects task structure.

4. **Error handling:** Failed tasks are logged but don't crash the run; results are saved regardless.

5. **Model saving:** Results are saved to CSV immediately after each task completes, so you can check progress mid-run.

---

## Next Steps After Running

Once you have all 6 CSVs, you can:

1. **Load and analyze:**
   ```python
   import pandas as pd
   df = pd.read_csv("results/Qwen2.5-7B-Instruct/cyclic_shift_grid_structured_boundaries.csv")
   print(df.groupby("Seq_Length")["TAR"].mean())
   ```

2. **Compare tiers:** Plot TAR vs k for each tier to see boundary effects

3. **Check alignment hypothesis:** Compare Modsum (3a, direction-aligned) vs Addition (3b, direction-misaligned) to see if direction explains severity of performance drop

4. **Error recovery:** Check if Tier 1 tasks show recovery benefits that Tier 2/3 tasks don't

