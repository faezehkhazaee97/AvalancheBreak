#!/usr/bin/env python3
"""Head ablation, version 2: does the marker benefit run through marker-attending heads,
with and without the tokenization channel?

For each task variant (merged letters, spaced letters, digits; same seeds and rule), length
N and marker spacing k, on a Qwen2 model:
  1. Identify the TOP marker-attending heads on held-out trials (100-104): mean attention
     mass from all positions to marker tokens, per (layer, head).
  2. On test trials 0..T-1, generate under five conditions:
       base          no markers (k = 0 prompt)
       bounds        markers, no intervention
       ablate_top    TOP heads cannot attend to marker tokens (weights zeroed, rows renormalized)
       ctrl_random   the same intervention on TOP-many random heads (control)
       ctrl_sham     TOP heads blocked from as many random non-marker input tokens (control)
Marker and input-span token positions come from the tokenizer's character offsets, so they
are exact for merged ("AB|CD"), spaced ("A B | C D") and digit inputs.

The script checks at start that the intervention zeroes the targeted attention weights and
stops otherwise. Rows are appended one at a time (resumable).

Output: results/ablation_v2/<model_tag>.csv and <model_tag>_heads.csv
Usage:
  python scripts/run_head_ablation_v2.py --model Qwen/Qwen2.5-32B-Instruct
"""

import argparse
import csv
import importlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common import scoring  # noqa: E402
from tasks.common.runner import _local_seed, load_model, max_new_tokens  # noqa: E402
import transformers.models.qwen2.modeling_qwen2 as qwen2_mod  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

# ----------------------------------------------------------------------------- intervention
STATE = {"active": False, "heads_by_layer": {}, "positions": None}
_ORIG = qwen2_mod.eager_attention_forward


def patched_eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = qwen2_mod.repeat_kv(key, module.num_key_value_groups)
    value_states = qwen2_mod.repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    heads = STATE["heads_by_layer"].get(getattr(module, "layer_idx", None))
    if STATE["active"] and heads and STATE["positions"] is not None:
        pos = STATE["positions"]
        pos = pos[pos < attn_weights.shape[-1]]
        if pos.numel():
            h = torch.tensor(heads, device=attn_weights.device)
            sub = attn_weights[:, h]                      # (b, nh, q, k)
            sub[..., pos] = 0.0
            sub = sub / sub.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            attn_weights[:, h] = sub
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, attn_weights


qwen2_mod.eager_attention_forward = patched_eager_attention_forward


def set_intervention(heads, positions, device):
    by_layer = {}
    for l, h in heads:
        by_layer.setdefault(int(l), []).append(int(h))
    STATE.update(active=True, heads_by_layer=by_layer,
                 positions=torch.tensor(sorted(positions), device=device, dtype=torch.long))


def clear_intervention():
    STATE.update(active=False, heads_by_layer={}, positions=None)


# ----------------------------------------------------------------------------- prompts
def encode(tokenizer, prompt):
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
    ref = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                        return_dict=True, return_tensors="pt", enable_thinking=False)
    assert torch.equal(enc["input_ids"], ref["input_ids"]), "offset tokenization differs from chat template"
    start = text.rfind("Input: ") + len("Input: ")
    end = text.rfind("\nOutput:")
    markers, others = [], []
    for i, (a, b) in enumerate(enc["offset_mapping"][0].tolist()):
        if a >= start and b <= end and b > a:
            (markers if text[a:b].strip() == "|" else others).append(i)
    return ref, markers, others


def generate(model, tokenizer, enc, max_new):
    inputs = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def build(engine, N, k, A, t):
    return engine.build_prompt(N, k, random.Random(_local_seed(N, k, A, t)), axis_val=A)


def top_heads(model, tokenizer, engine, N, k, A, id_trials, n_top):
    mass = None
    for t in id_trials:
        _, _, prompt = build(engine, N, k, A, t)
        enc, markers, _ = encode(tokenizer, prompt)
        with torch.no_grad():
            out = model(**{kk: v.to(model.device) for kk, v in enc.items()}, output_attentions=True)
        m = torch.stack([a[0][:, :, markers].sum(-1).mean(-1).float() for a in out.attentions]).cpu().numpy()
        mass = m if mass is None else mass + m
    mass /= len(id_trials)
    flat = np.argsort(mass.ravel())[::-1][:n_top]
    nh = mass.shape[1]
    return [(int(i // nh), int(i % nh)) for i in flat], mass


def self_test(model, tokenizer, engine, heads):
    """The intervention must zero the targeted weights; otherwise stop."""
    _, _, prompt = build(engine, 16, 4, 8, 999)
    enc, markers, _ = encode(tokenizer, prompt)
    set_intervention(heads, markers, model.device)
    with torch.no_grad():
        out = model(**{k: v.to(model.device) for k, v in enc.items()}, output_attentions=True)
    clear_intervention()
    l, h = heads[0]
    leak = out.attentions[l][0, h][:, markers].abs().max().item()
    if leak > 1e-6:
        sys.exit(f"Self-test failed: blocked head L{l}/H{h} still attends to markers ({leak:.3g}).")
    print(f"Self-test passed: L{l}/H{h} attention to markers after blocking = {leak:.2g}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--tasks", default="random_permutation,random_permutation_spaced,random_permutation_digits")
    ap.add_argument("--axis", type=int, default=8)
    ap.add_argument("--ks", default="2,8")
    ap.add_argument("--lengths", default="32,64")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--suffix", default="", help="appended to the output file names, e.g. _top50")
    args = ap.parse_args()

    tag = args.model.split("/")[-1]
    out_dir = REPO / "results" / "ablation_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv, heads_csv = out_dir / f"{tag}{args.suffix}.csv", out_dir / f"{tag}{args.suffix}_heads.csv"
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            done = {(r["Task"], r["Seq_Length"], r["Interval"], r["Trial"], r["Condition"]) for r in csv.DictReader(f)}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)
    model.eval()
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    A, ks, lengths = args.axis, [int(x) for x in args.ks.split(",")], [int(x) for x in args.lengths.split(",")]
    id_trials = range(100, 105)
    fields = ["Task", "Condition", "Interval", "Seq_Length", "Alphabet_Size", "Trial", "TAR", "SAR",
              "N_Markers", "Input_Given", "Ground_Truth", "LLM_Output", "Raw_Model_Output"]

    def write(row):
        new = not out_csv.exists()
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)

    tested = False
    for task in args.tasks.split(","):
        engine = importlib.import_module(f"tasks.{task}.engine")
        for N in lengths:
            for k in ks:
                heads, mass = top_heads(model, tokenizer, engine, N, k, A, id_trials, args.top)
                rng = random.Random(f"{task}-{N}-{k}")
                pool = [(l, h) for l in range(n_layers) for h in range(n_heads) if (l, h) not in heads]
                rand_heads = rng.sample(pool, args.top)
                new_heads = not heads_csv.exists()
                with open(heads_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    if new_heads:
                        w.writerow(["Task", "Seq_Length", "Interval", "Set", "Layer", "Head", "Marker_Mass"])
                    for s, hs in [("top", heads), ("random", rand_heads)]:
                        for l, h in hs:
                            w.writerow([task, N, k, s, l, h, round(float(mass[l, h]), 5)])
                print(f"{task} N={N} k={k}: top heads {heads}", flush=True)
                if not tested:
                    self_test(model, tokenizer, engine, heads)
                    tested = True
                for t in range(args.trials):
                    plans = [("base", 0, None, None), ("bounds", k, None, None),
                             ("ablate_top", k, heads, "markers"), ("ctrl_random", k, rand_heads, "markers"),
                             ("ctrl_sham", k, heads, "sham")]
                    for cond, kk, hs, where in plans:
                        if cond == "base" and k != ks[0]:
                            continue  # Base does not depend on k; run it once per (task, N)
                        key = (task, str(N), str(kk), str(t), cond)
                        if key in done:
                            continue
                        inp, truth, prompt = build(engine, N, kk, A, t)
                        enc, markers, others = encode(tokenizer, prompt)
                        if hs is not None:
                            pos = markers if where == "markers" else \
                                random.Random(f"sham-{task}-{N}-{k}-{t}").sample(others, min(len(markers), len(others)))
                            set_intervention(hs, pos, model.device)
                        raw = generate(model, tokenizer, enc, max_new_tokens(engine, N, kk))
                        clear_intervention()
                        res = scoring.score(raw, truth)
                        write(dict(Task=task, Condition=cond, Interval=kk, Seq_Length=N, Alphabet_Size=A, Trial=t,
                                   TAR=res["TAR"], SAR=res["SAR"], N_Markers=len(markers), Input_Given=inp,
                                   Ground_Truth=res["clean_truth"], LLM_Output=res["clean_pred"], Raw_Model_Output=raw))
                print(f"  done {task} N={N} k={k}", flush=True)
    qwen2_mod.eager_attention_forward = _ORIG
    print("Finished.")


if __name__ == "__main__":
    main()
