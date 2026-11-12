#!/usr/bin/env python3
"""Could the generation budget explain any marker effect? (Reviewer question.)

The budget is int(1.5 N) + 20 new tokens for the no-marker run and for every spacing k >= 2, and
2N + 20 at k = 1 (tasks/common/runner.max_new_tokens). Predictions are clipped to the truth's
length before scoring, so a budget can only matter by cutting an answer off before it has N
characters. For every stored generation of the seven open models (N <= 64) this script
re-tokenizes the raw output and reports, per condition class:
  at_budget : share of generations that used the whole budget (within 2 tokens)
  cut_off   : share that used the whole budget AND have fewer than N answer characters
Output: results_clean/_summary/budget_check.csv
"""

import importlib
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from analyze_per_position import load, MODELS, BLOCKS  # noqa: E402
from tasks.common.runner import max_new_tokens  # noqa: E402

HF = {"Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct": "Qwen/Qwen2.5-14B-Instruct",
      "Qwen2.5-32B-Instruct": "Qwen/Qwen2.5-32B-Instruct", "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
      "Phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct", "Qwen3.8-27B": "Qwen/Qwen3.8-27B",
      "gemma-4-31B-it": "google/gemma-4-31B-it"}


def main():
    rows = []
    missing_raw = [0, 0]  # generations without a stored raw output, of all generations
    for label, tag in MODELS:
        if tag is None or str(tag).startswith("gemini"):
            continue
        tok = AutoTokenizer.from_pretrained(HF[tag])
        for block, tasks in BLOCKS:
            if "reset" in block:
                continue
            for task in tasks:
                d = load(task, tag)
                if d is None or "Raw_Model_Output" not in d.columns:
                    continue
                engine = importlib.import_module(f"tasks.{task}.engine")
                d = d.assign(N=d.Seq_Length.astype(int), k=d.Interval.astype(int))
                total = int(((d.N <= 64) & d.k.isin([0, 1, 2, 4, 8, 16])).sum())
                d = d[(d.N <= 64) & d.k.isin([0, 1, 2, 4, 8, 16])
                      & d.Raw_Model_Output.map(lambda x: isinstance(x, str) and x != "")]
                missing_raw[0] += total - len(d); missing_raw[1] += total
                if d.empty:
                    continue
                ntok = [len(x) for x in tok(d.Raw_Model_Output.tolist(), add_special_tokens=False)["input_ids"]]
                budget = [max_new_tokens(engine, n, k) for n, k in zip(d.N, d.k)]
                d = d.assign(at_budget=[a >= b - 2 for a, b in zip(ntok, budget)],
                             short=d.LLM_Output.str.len() < d.Ground_Truth.str.len())
                d["cls"] = d.k.map(lambda k: "no markers" if k == 0 else ("k = 1" if k == 1 else "k >= 2"))
                for cls, g in d.groupby("cls"):
                    rows.append(dict(model=label, block=block, task=task, cls=cls, n=len(g),
                                     at_budget=g.at_budget.sum(), cut_off=(g.at_budget & g.short).sum()))
        print("done", label, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(REPO / "results_clean" / "_summary" / "budget_check.csv", index=False)
    print(f"generations without a stored raw output (skipped): {missing_raw[0]} of {missing_raw[1]}")
    s = out.groupby("cls")[["n", "at_budget", "cut_off"]].sum()
    s["at_budget_share"] = s.at_budget / s.n
    s["cut_off_share"] = s.cut_off / s.n
    print(s.to_string())
    t = out.groupby(["block", "cls"])[["n", "at_budget", "cut_off"]].sum()
    t["cut_off_share"] = t.cut_off / t.n
    print(t.to_string())
    w = out.assign(share=out.cut_off / out.n).sort_values("share", ascending=False).head(8)
    print(w.to_string(index=False))


if __name__ == "__main__":
    main()
