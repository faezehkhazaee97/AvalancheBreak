#!/usr/bin/env python3
"""Cross-boundary attention at the failing position (reviewer question: at the post-marker step, how
much attention reaches the input before the marker?).

Same teacher-forced setup as scripts/run_activation_patching.py: a dependency task's input is
presented with markers every k characters (marked) and without (clean), each followed by its
correct answer, and the forward pass is run with attention weights kept. For the query position
that predicts answer character j (the token just before it), we record the attention mass placed on

  prev_in   the input character j-1 (before the marker, when j is a post-marker position)
  cur_in    the input character j
  marker    the marker token right before input character j (marked run only; 0 in the clean run)
  prev_out  the answer character j-1 (just written)

per layer, averaged over heads, and also the largest single-head mass on prev_in in that layer.
Positions are classified as post-marker (j > 0, j mod k = 0) or within-segment (the rest, j > 0),
so the marked run's post-marker rows can be compared with its own within-segment rows and with the
clean run at the same indices.

Output: results/attention_crossing/<model_tag>.csv (Task, Trial, Run, PosType, Layer, prev_in, cur_in,
        marker, prev_out, max_head_prev_in, n_pos)
Usage:  python scripts/run_attention_crossing.py --model Qwen/Qwen2.5-32B-Instruct [--trials 20] [--dry]
"""

import argparse
import csv
import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from run_activation_patching import build, positions, TASKS, N, K, TRIALS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    engines = {t: importlib.import_module(f"tasks.{t}.engine") for t in TASKS}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    e = engines["running_xor"]
    inp_m, ans_m, p_m = build(e, None, K, 0); inp_c, ans_c, p_c = build(e, None, 0, 0)
    ids_m, in_m, out_m = positions(tokenizer, p_m, inp_m, ans_m)
    ids_c, in_c, out_c = positions(tokenizer, p_c, inp_c, ans_c)
    assert len(in_m) == len(in_c) == N and len(out_m) == len(out_c) == N
    post = [j for j in range(1, N) if j % K == 0]
    assert all(tokenizer.decode([ids_m[in_m[j] - 1]]) == "|" for j in post), "token before a post-marker input is not the bar"
    print(f"Self-check passed: aligned; {len(post)} post-marker and {N - 1 - len(post)} within-segment positions per input.")
    if args.dry:
        return

    import torch
    from tasks.common.runner import load_model
    model = load_model(args.model); model.eval()
    dev = model.device
    tag = args.model.split("/")[-1]
    out_dir = REPO / "results" / "attention_crossing"; out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{tag}.csv"
    fields = ["Task", "Trial", "Run", "PosType", "Layer", "prev_in", "cur_in", "marker", "prev_out", "max_head_prev_in", "n_pos"]
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            done = {(r["Task"], r["Trial"], r["Run"]) for r in csv.DictReader(f)}
    f = open(out_csv, "a", newline=""); w = csv.DictWriter(f, fieldnames=fields)
    if not done and f.tell() == 0:
        w.writeheader()

    def record(task, t, run, ids, in_pos, out_pos):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], device=dev), output_attentions=True)
        groups = {"post": [j for j in range(1, N) if j % K == 0], "within": [j for j in range(1, N) if j % K != 0]}
        for li, att in enumerate(out.attentions):           # [1, heads, L, L]
            a = att[0].float()
            for ptype, js in groups.items():
                acc = dict(prev_in=0.0, cur_in=0.0, marker=0.0, prev_out=0.0, max_head_prev_in=0.0)
                for j in js:
                    q = out_pos[j] - 1
                    row = a[:, q, :]                          # [heads, L]
                    acc["prev_in"] += row[:, in_pos[j - 1]].mean().item()
                    acc["cur_in"] += row[:, in_pos[j]].mean().item()
                    acc["prev_out"] += row[:, out_pos[j - 1]].mean().item()
                    if run == "marked" and j % K == 0:
                        acc["marker"] += row[:, in_pos[j] - 1].mean().item()
                    acc["max_head_prev_in"] += row[:, in_pos[j - 1]].max().item()
                w.writerow(dict(Task=task, Trial=t, Run=run, PosType=ptype, Layer=li, n_pos=len(js),
                                **{k: v / len(js) for k, v in acc.items()}))
        f.flush()
        del out

    for task, axis in TASKS.items():
        engine = engines[task]
        for t in range(args.trials):
            inp_m, ans_m, p_m = build(engine, axis, K, t); inp_c, ans_c, p_c = build(engine, axis, 0, t)
            ids_m, in_m, out_m = positions(tokenizer, p_m, inp_m, ans_m)
            ids_c, in_c, out_c = positions(tokenizer, p_c, inp_c, ans_c)
            if (task, str(t), "clean") not in done:
                record(task, t, "clean", ids_c, in_c, out_c)
            if (task, str(t), "marked") not in done:
                record(task, t, "marked", ids_m, in_m, out_m)
            print(f"done {task} trial {t}", flush=True)
    f.close(); print("Finished.")


if __name__ == "__main__":
    main()
