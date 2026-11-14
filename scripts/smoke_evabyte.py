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
    import os
    os.environ["EVABYTE_NO_TRITON"] = "1"   # honored by the locally patched modeling_evabyte.py
    print("EVABYTE_NO_TRITON=1: using the model's PyTorch attention fallback")
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

# ---- attempt with a manual greedy loop on the model's own cache (no library generate)
def manual_greedy(model, tok, enc, max_new=60, chunked_prefill=True):
    """Greedy decoding with EvaCache: prefill the prompt (in one call, or one token at a time), then
    feed one token per step with explicit position ids."""
    import importlib
    mod = sys.modules[type(model).__module__]
    pkg = mod.__name__.rsplit(".", 1)[0]
    EvaCache = importlib.import_module(pkg + ".eva_cache").EvaCache
    ids = enc["input_ids"]
    cache = EvaCache()
    dev = ids.device
    with torch.no_grad():
        if chunked_prefill:
            pos = torch.arange(ids.shape[1], device=dev)[None]
            out = model(input_ids=ids, position_ids=pos, past_key_values=cache, use_cache=True)
            logits = out.logits[:, -1]
        else:
            for i in range(ids.shape[1]):
                out = model(input_ids=ids[:, i:i + 1], position_ids=torch.tensor([[i]], device=dev), past_key_values=cache, use_cache=True)
            logits = out.logits[:, -1]
        gen = []
        n = ids.shape[1]
        eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")}
        for step in range(max_new):
            nxt = int(logits.argmax(-1))
            if nxt in eos:
                break
            gen.append(nxt)
            out = model(input_ids=torch.tensor([[nxt]], device=dev), position_ids=torch.tensor([[n + step]], device=dev),
                        past_key_values=cache, use_cache=True)
            logits = out.logits[:, -1]
    return tok.decode(gen, skip_special_tokens=True)


for k in (0, 4):
    inp, ans, prompt = e.build_prompt(16, k, random.Random(_local_seed(16, k, None, 0)))
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
    enc = {a: b.to(model.device) for a, b in enc.items()}
    for chunked in (True, False):
        t0 = time.time()
        try:
            text = manual_greedy(model, tok, enc, chunked_prefill=chunked)
            print(f"k={k} [manual loop, {'one-call' if chunked else 'token-by-token'} prefill]: truth {ans} | output {text!r} | {round(time.time() - t0, 1)} s", flush=True)
            break
        except Exception as ex:
            tb = traceback.extract_tb(ex.__traceback__)[-1]
            print(f"k={k} [manual loop, {'one-call' if chunked else 'token-by-token'} prefill]: FAILED {type(ex).__name__}: {str(ex)[:120]} at {tb.filename.split('/')[-1]}:{tb.lineno}", flush=True)

