# Re-Tokenization, Segmentation, and a Blind Spot: What Content-Free Boundary Markers Do in Character-Level Transformations

Inserting a content-free marker (`|`) into the input every *k* characters of a character-level
transformation, and asking the model to copy it through, is a cheap single-call intervention.
This repository holds the tasks, runners, raw generations, analysis scripts, and paper sources
for a study of what that intervention does across seven synthetic tasks, seven open
instruction-tuned models (Qwen2.5-7B/14B/32B, Mistral-7B, Phi-3.5-mini, Qwen3.8-27B,
Gemma-4-31B) and two versions of Gemini Flash (3.5 and 3.6).

Three effects are separated with two tokenization controls that fix one token per character
(spaced letters, and digit relabelings of the same tasks):

1. **On letter strings, markers mostly re-tokenize.** Spacing the letters with no markers
   improves accuracy more than markers do at five of the seven open models.
2. **The gain survives one token per character.** With spaced letters or digits, markers remove
   33% to 83% of the remaining errors at the four strongest open models, persist to N = 256, and
   are the same with commas or line breaks as markers.
3. **A post-marker blind spot.** With a marker every two or more characters, the position right
   after a marker fails whenever its output depends on the input before the marker: on four
   dependency tasks it falls to near chance at the stronger models while the next position
   recovers. Rewriting the tasks so that position needs nothing before the marker repairs it,
   and worked examples of the crossing step repair most of it at Gemma 4 and about a third at
   Qwen2.5-32B.

## Layout

| Path | Contents |
|---|---|
| `tasks/<task>/engine.py` | One module per task: input generation, ground truth, prompt. `tasks/common/` holds the shared runner, marker insertion, scoring, and the wording and spaced-letter variants. |
| `tasks/run_task.py`, `tasks/run_digit_controls.py` | Command-line entry points that run one task on one model over the full grid (lengths 4 to 64, every spacing, 20 trials per cell). |
| `results_clean/<task>/grid/<model>.csv` | Every stored generation of the open models (one row per prompt, with the raw output). `results_clean/gemini_pilot/` holds the Gemini runs. |
| `results_clean/_summary/` | Pooled effects, per-position values, and every derived table as CSV. |
| `results/` | Follow-up experiments (head ablations, constrained 2x2, paired inputs, base checkpoints, clean baselines, activation patching, LoRA). |
| `scripts/` | `summarize_pooled.py` and `analyze_*.py` produce the summary CSVs and the LaTeX tables; `make_fig_*.py` produce the figures; `run_*.py` are the follow-up experiment runners with matching `.sbatch` files for Slurm. |
| `paper_tmlr/` | The paper source (`main.tex`, `refs.bib`, figures, tables) and the compiled `paper.pdf`. |
| `paper_ms/` | The master's project version of the same text in the UC Davis thesis format; `scripts/make_ms_version.py` regenerates it from the paper source. |
| `legacy/` | Earlier drafts and reviewer-response notes from the project's first phase, kept for the record and not used by anything above. |

## Reproducing

Every input is generated from a deterministic seed, so every prompt can be rebuilt from its
(task, length, spacing, alphabet or modulus, trial) coordinates without the stored CSVs.

```bash
# environment: Python 3.10, torch, transformers, pandas, matplotlib (see .conda/envs/qwen-focus)
python -m tasks.run_task running_xor Qwen/Qwen2.5-7B-Instruct grid 20      # one task, one model
python -m tasks.run_digit_controls cyclic_shift_digits Qwen/Qwen2.5-7B-Instruct 20

python scripts/summarize_pooled.py          # pooled, length-matched effects for every cell
python scripts/analyze_per_position.py      # post-marker vs within-segment accuracy
python scripts/make_tables_pooled.py        # LaTeX tables under paper_tmlr/tables/
python scripts/make_fig_overview.py         # figures under paper_tmlr/figures/
```

The models run on one 80 GB GPU for the 27B to 32B checkpoints and on a 24 GB GPU for the 7B
class; a full grid for one task takes 25 minutes to 2 hours. The Gemini runs need an API key in
the `GEMINI_API_KEY` environment variable (`tasks/common/gemini_full_grid.py`).

## Paper

`paper_tmlr/main.tex` compiles with Tectonic or any TeX Live installation:

```bash
cd paper_tmlr && tectonic --keep-logs --outdir build main.tex
```
