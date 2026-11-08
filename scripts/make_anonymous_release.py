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
# Boundary Markers Help Until They Cut a Dependency: code and data

Code, prompts, raw generations, and analysis scripts for the anonymous submission. Every
prompt is rebuilt deterministically from (task, length N, marker interval k, alphabet or
modulus, trial), so the result files store inputs and outputs but not prompts.

## Layout

| Path | Contents |
|---|---|
| `tasks/<task>/engine.py` | One task per folder: input generation, ground truth, marker insertion, prompt. |
| `tasks/common/` | Shared runner (`runner.py`), scoring, statistics and analysis modules, Gemini client, vendored ProcBench and CLRS code (with their licenses). |
| `tasks/run_task.py` | Main entry point for a task grid on one open model. |
| `scripts/` | Slurm job scripts, mechanism experiments (attention capture, head ablation, controls), and figure scripts (`make_fig_*.py`). Many are earlier exploratory runs kept for completeness. |
| `results_clean/<task>/grid/<model>.csv` | Raw generations, one row per call (columns below). |
| `results_clean/gemini_pilot/` | Gemini runs (`gemini-flash-latest_POOLED_3trials.csv` is the pooled grid). |
| `results/<model>/` | Outputs of the mechanism experiments (attention captures, ablations) and earlier grids they reuse. |

Result columns: `Method` (`Base` = no markers, `Cellular_Bounds` = markers), `Interval` (k; 0 for Base),
`Seq_Length` (N), `Trial`, `Alphabet_Size` or `Modulus` where the task has one, `Input_Given`,
`Ground_Truth`, `LLM_Output` (markers stripped, clipped to the truth length), `TAR` (per-position
accuracy), `SAR` (exact match), `First_Error_Pos`, and `Raw_Model_Output` (full generation; absent in
some early files). `_trialsA-B` files are additional, non-overlapping seed batches; `_N128` files
extend a grid to N = 128. Read answer columns as text (e.g. `pd.read_csv(..., dtype=str)` for
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

writes `results_clean/<task>/grid/<model_tag>.csv` (`verify` instead of `grid` runs one trial per
cell as a quick check). Core tasks: `cyclic_shift`, `random_permutation`, `conditional_shift`
(position-local); `copy_previous`, `running_xor`, `running_sum_local`, `sliding_window_max`
(dependency); and `<task>_reset` variants of the four dependency tasks, whose recurrence restarts at
every marker. External tasks run through `python -m tasks.run_external_task`.

Generation budget (`tasks/common/runner.py`, `max_new_tokens`): int(1.5N)+20 new tokens, and 2N+20
at k = 1, where a correct marked answer is 2N-1 tokens. Cells at k = 1 and N >= 64 were first
generated under the smaller budget and then regenerated with the larger one by
`tasks/rerun_capped_k1.py` (greedy decoding: the new outputs extend the old ones).

## Figures

Run `scripts/make_fig_split_heatmap.py` first: it also writes
`results_clean/_summary/split_heatmap_values.csv`, which `make_fig3_causal_only.py` reads.

| Figure file (`figs/`) | Script (`scripts/`) |
|---|---|
| `fig1_split_heatmap` (effect of markers, every task and model) | `make_fig_split_heatmap.py` |
| `fig3_causal_only` (reset variants) | `make_fig3_causal_only.py` |
| `fig2_mechanism_merged` (attention sinks, head ablation) | `make_fig2_mechanism_merged.py` |
| `fig2_vertex_migration` (best density by length) | `make_fig2_vertex_migration.py` |
| `divide_and_conquer` | `make_fig_divide_and_conquer.py` |
| `attention_mechanism_32b` | `make_fig_attention_mechanism_32b.py` |
| `head_heatmaps_32b` | `make_fig_head_heatmaps_32b.py` |
| `error_containment_combined` | `make_fig_error_containment_combined.py` |
| `attention_digits_comparison` | `make_fig_attention_digits_comparison.py` |
| `tier1_cyclic_shift` | `make_fig_tier1_cyclic_shift.py` |
| `anchors_grid` | `make_fig_anchors_grid.py` |
| `hard_dependency_grid` | `make_fig_hard_dependency_grid.py` |
| `external_vowel_tagging`, `external_procbench`, `external_string_copy` | `make_fig_external_tier1_checks.py` |
| `external_benchmark_gemini` | `make_fig_external_benchmark_gemini.py` |

## Statistics

An effect is the pooled difference in TAR between marker and no-marker conditions,
Delta = mean TAR(Bounds) - mean TAR(Base), pooled over N in {4, 8, 16, 32, 64}, k in {1, 2, 4, 8, 16}
(k = 3 is run but excluded), and alphabet or modulus values, with z = Delta / sqrt(SE_Base^2 + SE_Bounds^2)
(see `pooled_delta` in `scripts/make_fig_split_heatmap.py`).
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
