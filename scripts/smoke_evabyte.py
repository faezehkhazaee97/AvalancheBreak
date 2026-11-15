#!/usr/bin/env python3
"""Smoke test for the byte-level chat model EvaByte-SFT on the exact path the runner uses.

Loads the model with tasks.common.runner.load_model, which rebuilds the rotary tables that
transformers 5 leaves uninitialized, and decodes with tasks.common.evabyte_generate.manual_greedy.
Then:
  1. checks that every input character is one token;
  2. answers the model card's example prompt. The card's greedy answer is an English pangram
     ("The quick brown fox jumps over the lazy dog."), so a correct decoding path must mention a
     pangram or the fox; noise from broken attention does not;
  3. answers one no-marker and one marked running_xor prompt, for the log.
Prints 'SANITY PASS' and exits 0 only when check 2 succeeds; scripts/evabyte_grid.sbatch runs the
grids only then.
"""
import os, random, sys, time, traceback
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
MODEL = next((a for a in sys.argv[1:] if not a.startswith("--")), "evabyte/EvaByte-SFT")
if "--no-triton" in sys.argv:
    os.environ["EVABYTE_NO_TRITON"] = "1"   # honored by the locally patched modeling_evabyte.py
    print("EVABYTE_NO_TRITON=1: using the model's PyTorch attention fallback")
from transformers import AutoTokenizer
from tasks.running_xor import engine as e
from tasks.common.runner import _local_seed, load_model
from tasks.common.evabyte_generate import manual_greedy

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
for k in (0, 4):
    inp, ans, prompt = e.build_prompt(16, k, random.Random(_local_seed(16, k, None, 0)))
    ids = tok(inp, add_special_tokens=False)["input_ids"]
    print(f"k={k}: input {inp!r} -> {len(ids)} tokens for {len(inp)} chars")

t0 = time.time()
model = load_model(MODEL)
model.eval()
print(f"loaded in {time.time() - t0:.0f} s; type {type(model).__name__}", flush=True)


def ask(content, max_new):
    enc = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True, enable_thinking=False)
    ids = enc["input_ids"].to(model.device)
    t0 = time.time()
    return tok.decode(manual_greedy(model, tok, ids, max_new), skip_special_tokens=True), time.time() - t0


def failed(label, ex):
    tb = traceback.extract_tb(ex.__traceback__)[-1]
    print(f"{label}: FAILED {type(ex).__name__}: {str(ex)[:160]} at {tb.filename.split('/')[-1]}:{tb.lineno}", flush=True)


ok = False
try:
    text, dt = ask("Write me an English pangram.", 160)
    ok = "pangram" in text.lower() or "fox" in text.lower()
    print(f"card prompt: output {text!r} | {dt:.1f} s | {'coherent' if ok else 'NOT coherent'}", flush=True)
except Exception as ex:
    failed("card prompt", ex)
for k in (0, 4):
    inp, ans, prompt = e.build_prompt(16, k, random.Random(_local_seed(16, k, None, 0)))
    try:
        text, dt = ask(prompt, 60)
        print(f"k={k} [manual loop]: truth {ans} | output {text!r} | {dt:.1f} s", flush=True)
    except Exception as ex:
        failed(f"k={k} [manual loop]", ex)
print("SANITY PASS" if ok else "SANITY FAIL")
sys.exit(0 if ok else 1)
