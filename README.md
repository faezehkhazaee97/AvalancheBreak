# Keeping LLMs Focused: Delaying Error Avalanches via Structural Boundaries

Do periodic structural boundary tokens (`|`) delay the "accuracy cliff" LLMs hit on long
deterministic sequence tasks? This repo replicates and extends the paper draft in
[`Keep_LLMs_Focused.pdf`](Keep_LLMs_Focused.pdf), which itself builds on the SK spin-glass
framework from Hou et al., ["How Focused Are LLMs?"](https://arxiv.org/abs/2511.00763)
(arXiv:2511.00763).

Read [`project_explainer.html`](project_explainer.html) first (open it in a browser) for a
full visual walkthrough of the project, the method, and the headline results.

## Repository layout

| Path | What it is |
|---|---|
| `LLM-Focus.ipynb` | **The main notebook.** Canonical, up-to-date, bug-fixed source for every experiment. Run top to bottom to reproduce everything for one model (set `MODEL_ID` in the "MODEL SELECTION" cell). |
| `code/LLM-Focus.ipynb` | The original first-draft notebook that produced the paper draft's numbers, kept for reference/history. |
| `notebooks/` | Frozen executed/input snapshots from the 14B and Mistral runs, plus fully-executed copies of the main notebook (with embedded output). Historical record, not meant to be re-run. |
| `results/<model_tag>/` | All experiment output: CSVs (grids, regressions, SK fits, significance tests) and figures (PNG), one directory per model (`Qwen2.5-7B-Instruct`, `Qwen2.5-14B-Instruct`, `Mistral-7B-Instruct-v0.3`). |
| `results/model_scale_comparison.{csv,png}` | Cross-model comparison of the headline effect. |
| `scripts/` | Standalone Python scripts + matching SLURM batch scripts, each one a self-contained patch that backfilled a specific analysis (ablation, addition, LSD carry-direction test, scratchpad, Pauli strings, divide-and-conquer benchmark, k=16 grid extension, etc.) for a specific model, without needing to rerun the whole notebook. Useful as a reference for exactly how each experiment was run; the notebook is the source of truth for current logic. |
| `Keep_LLMs_Focused.pdf` | The paper draft this project is based on/extending. |

## What's studied

- **Cyclic-shift task**: replace each character with the next in a fixed alphabet; deterministic, position-local.
- **Running-sum-mod-M task**: cumulative sum tracking; deterministic, requires long-range state.
- **Integer addition** (direct, LSD-first/carry-direction variant, with a scratchpad/CoT variant).
- **Pauli string multiplication**: hybrid task with a position-local part (the operator string) and a global part (the accumulated phase).
- All tasks are run with and without periodic `|` boundary markers inserted into the prompt ("Cellular_Bounds" vs "Base"), and fit to the SK spin-glass model `SAR(N) = exp(-β0 · N · α^(N-1))` from Hou et al. to quantify whether boundaries reduce the error-accumulation factor α and push out the cliff length N*.
- Mechanistic follow-ups: boundary ablation (marker identity/placement/density), attention-sink analysis, entropy, clustering, a true divide-and-conquer benchmark (actually splitting the sequence into separate model calls) as a cost/accuracy baseline against the single-pass boundary trick, and cross-model replication on Qwen2.5-7B/14B-Instruct and Mistral-7B-Instruct-v0.3.

## Reproducing

1. Conda env: `.conda/envs/qwen-focus` (not tracked in git — rebuild from the imports at the top of `LLM-Focus.ipynb`: `transformers`, `torch`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `networkx`, `tqdm`).
2. GPU cluster: designed for SLURM (A100). See `scripts/run_notebook_batch.sh` for the canonical submission pattern:
   ```bash
   sbatch scripts/run_notebook_batch.sh       # 7B
   sbatch scripts/run_notebook_batch_14b.sh   # 14B
   sbatch scripts/run_notebook_batch_mistral.sh  # Mistral
   ```
   These run `jupyter nbconvert --execute` on the main notebook end-to-end.
3. Set `HF_HOME` to a local cache directory with enough space for model weights (see the "HIVE STORAGE REDIRECTION" cell at the top of the notebook) and authenticate via `hf auth login` if you hit HF Hub rate limits.
4. Outputs land in `results/<model_tag>/`, namespaced so multiple model scales can coexist and be compared.

## Key gotchas (already handled in the current notebook, documented here so they don't get reintroduced)

- Attention-weight capture requires `attn_implementation="eager"` at model load time — `sdpa`/flash-attention can't return attention weights.
- Use `bf16`, not 8-bit quantization — quantization noise confounds the attention/entropy analysis.
- NetworkX directed weighted clustering coefficient produces `NaN` on self-loops (self-attention) — strip self-loops before computing, and use a length-adaptive threshold, not a fixed one.
- SK-model curve fits on degenerate data (SAR constant/all-zero across every tested N) still "succeed" numerically but are meaningless — always check the `Fit_Valid` column before trusting `Beta0`/`Alpha`/`N_star`.
- Addition-task prompts need an explicit separator between the two addends, and scoring must only credit a *leading* run of digits (not any digit found anywhere in the response) — otherwise stray digits from explanation text silently pollute the score.
