#!/usr/bin/env python3
"""EvaByte-SFT (byte-level, one token per character and per marker) prompted with three solved
examples, the way the base-checkpoint runs are prompted (scripts/run_base_model.py).

Appendix L ran EvaByte strictly zero-shot and it copied its input instead of applying the rule, so
the byte-level question (does the post-marker blind spot survive one-token-per-character
tokenization?) stayed unanswered. The base checkpoints had the same adherence problem and were
given three solved examples in the Input/Output format of the query; this script gives EvaByte the
same demonstrations, from the same builder (run_base_model.build), so the demonstration format is
identical and the only difference from the base-checkpoint runs is the model.

Two deliveries of the same few-shot prompt are supported, because EvaByte-SFT is a chat model
while the base-checkpoint convention is a plain completion:
  --delivery chat  demonstrations inside one user turn of the chat template (as the zero-shot
                   EvaByte grid was delivered);
  --delivery raw   no chat template at all, the model completes the final "Output:" line (exactly
                   the base-checkpoint convention).
--pilot compares both against the zero-shot prompt on one task at one length before any full grid
is run.

Grid: the same cells as the stored zero-shot grid (N in {4,8,16,32,64}, k in {0,1,2,3,4,8,16} with
k < N, every alphabet value, 20 trials, greedy), the same scoring, and the same generation budget.

Output: results/evabyte_fewshot/<task>.csv -- deliberately outside results_clean/<task>/grid/,
where several loaders glob <tag>*.csv (scripts/analyze_per_position.py,
scripts/make_fig_three_formats.py, ...) and tasks/common/compute_diagnostics.py globs *.csv: a
file named EvaByte-SFT-fewshot.csv there would be silently pooled into the zero-shot EvaByte
numbers. Rows are appended as they are produced and completed cells are skipped on restart, so a
preempted job resumes where it stopped.

Usage:
  python scripts/run_evabyte_fewshot.py --pilot --task running_xor --N 16 --trials 5
  python scripts/run_evabyte_fewshot.py --task running_xor --delivery raw --trials 20
"""

import argparse
import csv
import importlib
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from tasks.common import scoring  # noqa: E402
from tasks.common.boundary import STANDARD_LENGTHS, STANDARD_INTERVALS, valid_intervals  # noqa: E402
from tasks.common.runner import _local_seed, max_new_tokens  # noqa: E402
from run_base_model import SHOTS, build, gen  # noqa: E402  (same demonstrations as the base runs)

MODEL = "evabyte/EvaByte-SFT"
TASKS = ["running_xor", "copy_previous", "random_permutation_digits", "cyclic_shift_digits"]
FIELDS = ["Method", "Interval", "Seq_Length", "Trial", "Input_Given", "Ground_Truth", "LLM_Output",
          "TAR", "SAR", "First_Error_Pos", "Raw_Model_Output"]


def load(model_id):
    from transformers import AutoTokenizer
    from tasks.common.runner import load_model
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = load_model(model_id)          # rebuilds EvaByte's rotary tables under transformers 5
    model.eval()
    return tok, model


def answer(tok, model, prompt, max_new, delivery):
    """Greedy decode one prompt through EvaByte's own cache (manual_greedy)."""
    from tasks.common.evabyte_generate import manual_greedy
    if delivery == "chat":
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                      add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, enable_thinking=False)
        ids = enc["input_ids"]
    else:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
    ids = ids.to(model.device)
    return tok.decode(manual_greedy(model, tok, ids, max_new), skip_special_tokens=True)


def copy_rate(pred, inp):
    """1.0 when the model reproduced its input; also the per-character share equal to the input."""
    clean_in = inp.replace("|", "")
    exact = float(pred == clean_in)
    share = (sum(p == c for p, c in zip(pred, clean_in)) / len(clean_in)) if clean_in else 0.0
    return exact, share


def pilot(args):
    tok, model = load(MODEL)
    for task in args.task.split(","):
        engine = importlib.import_module(f"tasks.{task}.engine")
        axis = args.axis if (engine.AXIS_VALUES and args.axis) else (engine.AXIS_VALUES or [None])[0]
        for N in [int(x) for x in str(args.N).split(",")]:
            ks = [k for k in (int(x) for x in args.ks.split(",")) if k == 0 or k < N]
            print(f"\npilot: {task} N={N} axis={axis} k={ks} trials={args.trials} shots={args.shots}", flush=True)
            for label in ("zero-shot chat", f"{args.shots}-shot chat", f"{args.shots}-shot raw"):
                rows, t0 = [], time.time()
                for k in ks:
                    for t in range(args.trials):
                        if label == "zero-shot chat":
                            inp, truth, prompt = gen(engine, axis, N, k, random.Random(_local_seed(N, k, axis, t)))
                        else:
                            inp, truth, prompt = build(engine, axis, N, k, t)
                        raw = answer(tok, model, prompt, max_new_tokens(engine, N, k),
                                     "raw" if label.endswith("raw") else "chat")
                        r = scoring.score(raw, truth)
                        ex, sh = copy_rate(r["clean_pred"], inp)
                        rows.append((k, t, r["TAR"], r["SAR"], ex, sh, r["clean_pred"], r["clean_truth"], raw))
                n = len(rows)
                print(f"  {label:16s} TAR={sum(r[2] for r in rows)/n:.3f}  exact={sum(r[3] for r in rows)/n:.3f}  "
                      f"copied-input={sum(r[4] for r in rows)/n:.3f}  char-overlap-with-input={sum(r[5] for r in rows)/n:.3f}"
                      f"  ({time.time()-t0:.0f} s for {n} generations)", flush=True)
                for r in rows[:2] + rows[-2:]:
                    print(f"      k={r[0]} t={r[1]} truth={r[7]!r} pred={r[6]!r} raw={r[8][:90]!r}", flush=True)


def grid(args):
    engine = importlib.import_module(f"tasks.{args.task}.engine")
    lengths = STANDARD_LENGTHS
    out = REPO / "results" / "evabyte_fewshot" / f"{args.task}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = FIELDS + ([engine.AXIS_NAME] if engine.AXIS_NAME else [])
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(r["Interval"], r["Seq_Length"], r.get(engine.AXIS_NAME, ""), r["Trial"])
                    for r in csv.DictReader(f)}
    cells = [(N, k, a, t) for N in lengths for k in valid_intervals(N, STANDARD_INTERVALS)
             for a in (engine.AXIS_VALUES or [None]) for t in range(args.trials)]
    todo = [c for c in cells if (str(c[1]), str(c[0]), "" if c[2] is None else str(c[2]), str(c[3])) not in done]
    print(f"{args.task}: {len(cells)} cells, {len(done)} already done, {len(todo)} to run -> {out}", flush=True)
    if not todo:
        return
    tok, model = load(MODEL)
    t0, last = time.time(), None
    for i, (N, k, a, t) in enumerate(todo):
        inp, truth, prompt = build(engine, a, N, k, t)
        raw = answer(tok, model, prompt, max_new_tokens(engine, N, k), args.delivery)
        r = scoring.score(raw, truth)
        row = {"Method": "Base" if k == 0 else "Cellular_Bounds", "Interval": k, "Seq_Length": N,
               "Trial": t, "Input_Given": inp, "Ground_Truth": r["clean_truth"],
               "LLM_Output": r["clean_pred"], "TAR": r["TAR"], "SAR": r["SAR"],
               "First_Error_Pos": r["First_Error_Pos"], "Raw_Model_Output": raw}
        if engine.AXIS_NAME:
            row[engine.AXIS_NAME] = a
        new = not out.exists()
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)
        if (N, k, a) != last:
            last = (N, k, a)
            print(f"  N={N} k={k} axis={a} | {i}/{len(todo)} done | {time.time()-t0:.0f} s", flush=True)
    print(f"{args.task}: finished {len(todo)} generations in {time.time()-t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="running_xor", help=f"one of {TASKS} (--pilot: a comma-separated list)")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--delivery", default="raw", choices=["raw", "chat"])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--N", default="16", help="length, or a comma-separated list under --pilot")
    ap.add_argument("--ks", default="0,4", help="--pilot: the spacings to compare")
    ap.add_argument("--axis", type=int, default=0, help="--pilot: alphabet size (0 = the task's first)")
    ap.add_argument("--shots", type=int, default=SHOTS,
                    help="solved examples before the query (default 3, the base-checkpoint convention)")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    if args.shots != SHOTS:        # build() reads the module-level constant
        import run_base_model
        run_base_model.SHOTS = args.shots
        print(f"shots: {args.shots} (the base-checkpoint convention is {SHOTS})")
    if not args.pilot:
        assert args.task in TASKS, f"--task must be one of {TASKS}"
        args.N = int(args.N)
    if args.dry:
        engine = importlib.import_module(f"tasks.{args.task}.engine")
        axis = (engine.AXIS_VALUES or [None])[0]
        inp, truth, p = build(engine, axis, args.N, 4, 0)
        assert p.count("Input: ") == SHOTS + 1 and p.endswith(f"Input: {inp}\nOutput:")
        print(p)
        print(f"\n[truth {truth}]")
        return
    (pilot if args.pilot else grid)(args)


if __name__ == "__main__":
    main()
