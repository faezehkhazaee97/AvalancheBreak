#!/usr/bin/env python3
"""Build the anonymized code-and-data release for double-blind review.

Copies the git-tracked code (tasks/, scripts/) and result data (results_clean/, results/)
into release/anon/, replaces cluster paths, account names and other identifying strings
with neutral placeholders, writes README.md and requirements.txt, and then scans every
file in the release: it fails (exit 1) if any identifying string or credential-like
token is left. This script itself is not copied (it lists the strings it removes).

Rebuild after results change (commit them first: only tracked files are copied):
    .conda/envs/qwen-focus/bin/python scripts/make_anonymous_release.py
Then push release/anon/ to a fresh GitHub repository with no personal history
and mirror it with https://anonymous.4open.science.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "release" / "anon"
INCLUDE = ("tasks/", "scripts/", "results_clean/", "results/")
EXCLUDE = (
    "results_clean/_excluded_undocumented_models/",  # models not in the paper
    "results_clean/_k1_rerun/",                      # staging area, merged into the sources
    "results_clean/APPENDIX_HANDOFF_FOR_FABLE.md",   # internal note
    "scripts/make_anonymous_release.py",             # this file
)
TEXT_EXT = {".py", ".sh", ".sbatch", ".md", ".txt", ".csv", ".json", ".typed", ""}

ROOT_PATH = "/group/gquongrp/workspaces/baran/AvalancheBreak"
SUBS = [  # (regex, replacement), applied in order
    # Python: absolute repo paths -> paths relative to the script (all such files live in scripts/)
    (re.compile(r'Path\("' + re.escape(ROOT_PATH) + r'/?([^"]*)"\)'),
     r'(Path(__file__).resolve().parents[1] / "\1")'),
    (re.compile(r'Path\("/tmp/claude-[^"]*/scratchpad/([^"]*)"\)'), r'Path("/tmp/\1")'),
    # Slurm directives: log paths relative to the submission directory (the repo root)
    (re.compile(r"^(#SBATCH[^\n]*?)" + re.escape(ROOT_PATH) + "/", re.M), r"\1"),
    # Shell: repo root
    (re.compile(re.escape(ROOT_PATH)), "${REPO_DIR:-$PWD}"),
    (re.compile(r"/home/[A-Za-z0-9_]+/miniconda3"), "$HOME/miniconda3"),
    (re.compile(r"/cvmfs/hpc\.ucdavis\.edu/sw/spack/environments/core/view/generic/slurm/bin:"), ""),
    # Accounts and partitions
    (re.compile(r"g?quonbiogrp"), "YOUR_ACCOUNT"),
    (re.compile(r"ctbrowngrp"), "YOUR_OTHER_ACCOUNT"),
    (re.compile(r"ctbrown"), "alt"),
    (re.compile(r"\bgpu-a100-h\b|\bgpuh\b"), "YOUR_GPU_PARTITION"),
]
FILENAME_SUBS = [("ctbrowngrp", "altaccount"), ("ctbrown", "alt")]
FORBIDDEN = re.compile(
    r"gquon|quonbio|\bbaran\b|khazaee|faezeh|ucdavis|davis|cvmfs|ctbrown|/group/|claude-\d|"
    r"\bfarm\b|hpccf|AIza[0-9A-Za-z_-]{30,}|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}",
    re.I)

REQUIREMENTS = """\
# Versions used for the final runs (Python 3.11; CUDA 12.4 build of torch).
torch==2.6.0
transformers==5.12.1
accelerate==1.14.0
numpy==2.4.6
pandas==3.0.3
scipy==1.17.1
matplotlib==3.11.0
seaborn==0.13.2
tqdm==4.68.3
google-genai==2.10.0   # Gemini runs only
"""

README = """\
# Re-Tokenization, Segmentation, and a Blind Spot: code and data

Code, prompts, raw generations, and analysis scripts for the anonymous submission "Re-Tokenization,
Segmentation, and a Blind Spot: What Content-Free Boundary Markers Do in Sequence Transformation".
Every prompt is rebuilt deterministically from (task, format, length N, marker spacing k, alphabet
or modulus, trial), so the result files store inputs and outputs but not prompts.

## Layout

| Path | Contents |
|---|---|
| `tasks/<task>/engine.py` | One task per folder: input generation, ground truth, marker insertion, prompt. `<task>_spaced` and `<task>_digits` are the tokenization controls (spaced letters; digits), exact reformattings of the letter tasks; `<task>_reset` are the reset variants of the dependency tasks. |
| `tasks/common/` | Shared runner (`runner.py`, including the generation budget `max_new_tokens`), scoring, the spaced-format wrapper (`spaced.py`), analysis modules, Gemini client, vendored ProcBench and CLRS code (with their licenses). |
| `tasks/run_task.py`, `tasks/run_digit_controls.py` | Entry points for a task grid on one open model (the second runs the spaced and digit controls at N <= 64). |
| `scripts/` | Slurm job scripts, analysis and figure scripts (`analyze_per_position.py`, `make_fig_*.py`, `make_table*.py`), head-ablation experiments (`run_head_ablation_v2.py` is the one in the paper's main ablation table). Many other scripts are earlier exploratory runs kept for completeness. |
| `results_clean/<task>/grid/<model>.csv` | Raw generations, one row per call (columns below). |
| `results_clean/_summary/` | Derived values behind the figures and tables. |
| `results_clean/gemini_pilot/` | Gemini runs (`gemini-flash-latest_POOLED_3trials.csv` is the pooled grid). |
| `results/ablation_v2/` | Head-ablation results and the heads blocked. |
| `results/<model>/` | Earlier mechanism experiments (attention captures, the first head ablation) and the grids they reuse. |

Result columns: `Method` (`Base` = no markers, `Cellular_Bounds` = markers), `Interval` (k; 0 for Base),
`Seq_Length` (N), `Trial`, `Alphabet_Size` or `Modulus` where the task has one, `Input_Given`,
`Ground_Truth`, `LLM_Output` (markers and spaces stripped, clipped to the truth length), `TAR`
(per-position accuracy), `SAR` (exact match), `First_Error_Pos`, and `Raw_Model_Output` (full generation;
absent in some early files). `_trialsA-B` files are additional, non-overlapping seed batches; `_N128`
files extend a grid to N = 128. Read answer columns as text (e.g. `pd.read_csv(..., dtype=str)` for
those columns): all-digit answers lose leading zeros if parsed as numbers.

## Setup

Python 3.11 and `pip install -r requirements.txt`. Models load from the Hugging Face hub in
bfloat16 with greedy decoding: `Qwen/Qwen2.5-{7B,14B,32B}-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`,
`microsoft/Phi-3.5-mini-instruct`, `Qwen/Qwen3.8-27B`, `google/gemma-4-31B-it`. The 27B to 32B models
need one 80 GB GPU. Gemini runs need `GEMINI_API_KEY`.

Job scripts expect a Python environment at `$REPO_DIR/.conda/envs/qwen-focus` (edit the `ENV=` line
otherwise), are submitted from the repository root, and contain placeholders `YOUR_ACCOUNT` and
`YOUR_GPU_PARTITION` for your cluster.

## Running

    python -m tasks.run_task <task> <hf_model_id> grid <trials_per_cell> [trial_start]
    python -m tasks.run_digit_controls <task>_spaced|<task>_digits <hf_model_id> 20

write `results_clean/<task>/grid/<model_tag>.csv`. Position-local tasks: `cyclic_shift`,
`random_permutation`, `conditional_shift`; dependency tasks: `copy_previous`, `running_xor`,
`running_sum_local`, `sliding_window_max`, each with a `_reset` variant. External tasks run through
`python -m tasks.run_external_task`. Head ablation: `python scripts/run_head_ablation_v2.py --model <id>`.

Generation budget: int(1.5N)+20 new tokens, and 2N+20 at k = 1, where a correct marked answer is
2N-1 tokens. Cells at k = 1 and N >= 64 were first generated under the smaller budget and then
regenerated with the larger one by `tasks/rerun_capped_k1.py` (greedy decoding: the new outputs extend
the old ones).

## Figures and tables

Run `scripts/analyze_per_position.py`, `scripts/summarize_pooled.py`, and `scripts/make_fig_split_heatmap.py`
first; they write the summary CSVs that the figures and tables read.

| Paper item | Script (`scripts/`) |
|---|---|
| Overview figure (Figure 1) | `make_fig_overview.py` |
| Three-format comparison; length and spacing profile; per-task profile | `make_fig_three_formats.py` |
| Per-position accuracy; reset variants per position | `make_fig_per_position.py` |
| Pooled effects, every task and model (heatmap) | `make_fig_split_heatmap.py` |
| Head-ablation tables (32B in the main text, 7B in the appendix) | `make_table_ablation_v2.py` |
| Appendix per-position and distance tables | `make_tables_per_position.py` |
| Appendix reset, Gemini/CLRS and external tables | `make_tables_pooled.py` |
| Appendix: mechanism figure | `make_fig2_mechanism_merged.py` |
| Appendix: best density by length | `make_fig2_vertex_migration.py` |
| Appendix: per-model accuracy by length (all seven open models) | `make_fig_by_model.py` |
| Appendix: other figures | `make_fig_divide_and_conquer.py`, `make_fig_head_heatmaps_32b.py`, `make_fig_causal_ablation.py`, `make_fig_external_tier1_checks.py`, `make_fig_external_benchmark_gemini.py` |

## Follow-up experiments and robustness checks

Each pair is a run script (Slurm job, `.sbatch` alongside) and the analysis that writes the
corresponding appendix table into `paper_tmlr/tables/`.

| Paper item | Run | Analysis |
|---|---|---|
| Alignment-robust (edit-distance) re-scoring | (reuses the main grid) | `analyze_aligned_scoring.py` |
| Instruction wording for the blind spot; head-count dose response | `run_wording_controls.sbatch`; `run_head_ablation_v2.py --top 50/200` | `analyze_wording_and_dose.py` |
| Delimiter type (bar, comma, line break) and N up to 256 | `run_delimiter_length.py` | `analyze_delimiter_length.py` |
| Sampling instead of greedy decoding | `run_sampling_decoding.py` | `analyze_sampling_decoding.py` |
| Larger state space (modulus 8, 16, 32) | `run_large_state.py` | `analyze_large_state.py` |
| Prompt delivery (system prompt, no chat template) | `run_prompt_delivery.py` | `analyze_prompt_delivery.py` |
| Input vs output markers, instruction-based 2x2 | `run_marker_sides.py` | `analyze_marker_sides.py`, `analyze_interaction.py` |
| Input vs output markers, format constrained by a logits processor | `run_marker_sides_forced.py` | `analyze_marker_sides_forced.py` |
| No-marker baselines whose prompt never mentions markers (300 cells) | `run_clean_baseline.py` | `analyze_clean_baseline.py` |
| Generation-budget cut-off check | (reuses the main grid) | `analyze_budget.py` |
| Exact transformation match and marker-placement compliance | (reuses the main grid) | `analyze_compliance.py` |
| Bootstrap sensitivity and the A = 2 / M = 2 redundancy check | (reuses the main grid) | `analyze_statistics_robustness.py` |

Their raw outputs are under `results/<experiment>/` and `results_clean/<task>_{neutral,across}/`.

## Statistics

A pooled effect is Delta = mean TAR(Bounds) - sum_N p_N mean TAR(Base at N), where p_N is the share of
marker runs at length N (only k < N is run, so marker runs sit at longer lengths); it pools N in
{4, 8, 16, 32, 64}, k in {1, 2, 4, 8, 16} (k = 3 is run but excluded), and alphabet or modulus values
(`tasks/common/pooled.py`). The share of remaining errors removed is the mean Delta over a model's tasks
divided by one minus the mean length-matched no-marker accuracy.
Per-position accuracy classes positions as right after a marker (i > 0, i mod k = 0) or within a segment
and compares each with the no-marker run at the same indices (`scripts/analyze_per_position.py`).

## License

Our code is released under the MIT License (`LICENSE`). Vendored code keeps its own license:
`tasks/common/procbench_vendor/LICENSE` (MIT) and `tasks/common/clrs_vendor/LICENSE` (Apache 2.0).
"""


MIT_LICENSE = """\
MIT License

Copyright (c) 2026 Anonymous Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines()
            if f.startswith(INCLUDE) and not f.startswith(EXCLUDE) and "__pycache__" not in f]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    files = tracked_files()
    n_changed = 0
    for rel in files:
        dst_rel = rel
        for a, b in FILENAME_SUBS:
            dst_rel = dst_rel.replace(a, b)
        src, dst = REPO / rel, OUT / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if Path(rel).suffix in TEXT_EXT:
            text = src.read_text(errors="surrogateescape")
            new = text
            for pat, rep in SUBS:
                new = pat.sub(rep, new)
            n_changed += new != text
            dst.write_text(new, errors="surrogateescape")
        else:
            shutil.copy2(src, dst)
    (OUT / "README.md").write_text(README)
    (OUT / "LICENSE").write_text(MIT_LICENSE)
    (OUT / "requirements.txt").write_text(REQUIREMENTS)
    for d in ("slurm_logs", "figs", "figs/png"):  # output folders the scripts write into
        (OUT / d).mkdir(parents=True, exist_ok=True)
        (OUT / d / ".gitkeep").write_text("")

    hits = []
    for p in OUT.rglob("*"):
        if p.is_file():
            data = p.read_bytes().decode("utf-8", errors="ignore")
            for m in FORBIDDEN.finditer(data):
                hits.append(f"{p.relative_to(OUT)}: ...{data[max(0, m.start() - 40):m.end() + 40]!r}...")
            if FORBIDDEN.search(str(p.relative_to(OUT))):
                hits.append(f"file name: {p.relative_to(OUT)}")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{len(files)} files copied ({n_changed} rewritten), {size / 1e6:.0f} MB -> {OUT}")
    if hits:
        print(f"FAILED: {len(hits)} identifying or secret-like strings remain:")
        print("\n".join(hits[:40]))
        sys.exit(1)
    print("Scan clean: no identifying strings or credential-like tokens.")


if __name__ == "__main__":
    main()
