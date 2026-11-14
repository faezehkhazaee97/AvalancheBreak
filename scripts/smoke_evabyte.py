#!/usr/bin/env python3
"""Smoke test for a byte-level chat model (EvaByte-SFT): load with its remote code, check that every
input character is one token, and generate one no-marker and one marked running_xor prompt."""
import random, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tasks.running_xor import engine as e
from tasks.common.runner import _local_seed

MODEL = next((a for a in sys.argv[1:] if not a.startswith("--")), "evabyte/EvaByte-SFT")
if "--no-triton" in sys.argv:
    # make `import triton` fail so the model's own code picks its pure-PyTorch attention (eva_pt_ref)
    sys.modules["triton"] = None
    print("triton disabled: using the model's PyTorch attention fallback")
# transformers 5 passes additional_special_tokens=None to the remote tokenizer class, whose
# constructor iterates over it; an empty list takes the branch that keeps the config's own specials.
try:
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
except TypeError:
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, additional_special_tokens=[])
print("specials:", tok.special_tokens_map)
for k in (0, 4):
    inp, ans, prompt = e.build_prompt(16, k, random.Random(_local_seed(16, k, None, 0)))
    ids = tok(inp, add_special_tokens=False)["input_ids"]
    print(f"k={k}: input {inp!r} -> {len(ids)} tokens for {len(inp)} chars")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
model.eval(); print("loaded in", round(time.time() - t0), "s; type", type(model).__name__)
for k in (0, 4):
    inp, ans, prompt = e.build_prompt(16, k, random.Random(_local_seed(16, k, None, 0)))
    msgs = [{"role": "user", "content": prompt}]
    try:
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    except Exception as ex:
        print("chat template failed:", ex); enc = tok(prompt, return_tensors="pt")
    enc = {a: b.to(model.device) for a, b in enc.items()}
    import traceback
    attempts = [("use_cache=False", {"use_cache": False}), ("default cache", {})]
    for mode, kw in attempts:
        t0 = time.time()
        try:
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=60, do_sample=False, **kw)
            text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"k={k} [{mode}]: truth {ans} | output {text!r} | {round(time.time() - t0, 1)} s", flush=True)
        except Exception as ex:
            tb = traceback.extract_tb(ex.__traceback__)[-1]
            print(f"k={k} [{mode}]: FAILED {type(ex).__name__}: {str(ex)[:120]} at {tb.filename.split('/')[-1]}:{tb.lineno}", flush=True)
