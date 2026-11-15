"""Loading fix and greedy decoding for the byte-level EvaByte-SFT model.

Two problems with EvaByte's remote code under current transformers releases:

1. Rotary tables. transformers 5 builds every model on the meta device and refills non-persistent
   buffers with torch.empty_like, relying on each model's _init_weights to recompute them. EvaByte's
   _init_weights only handles Linear and Embedding layers, so its rotary tables (inv_freq,
   cos_cached, sin_cached, the model's only non-persistent buffers) are left as uninitialized memory
   and every attention score is noise: the model then emits frequent byte fragments ("t tint the")
   whatever the prompt. rebuild_rotary() recomputes them with the model's own constructor, which
   restores the values the original release computed at load time.

2. Decoding. The model keeps its own cache class (EvaCache) and its config asks for tuple outputs;
   current releases hand generate() a DynamicCache that the model cannot use. manual_greedy()
   prefills the prompt in one call with an EvaCache, then feeds one byte per step with explicit
   position ids. Run with EVABYTE_NO_TRITON=1 so the model selects its pure-PyTorch attention.

Used only for EvaByte (tasks/common/runner.py); other models are unchanged.
"""

import importlib
import sys

import torch


def rebuild_rotary(model):
    """Recompute the rotary tables after from_pretrained. Returns the largest absolute difference
    between the loaded and the rebuilt inverse frequencies (0 means the loaded tables were right)."""
    base = model.model
    dev = base.embed_tokens.weight.device
    old = base.rotary_emb.inv_freq.detach().float().cpu().clone()
    base._init_rope()
    base.rotary_emb.to(dev)
    new = base.rotary_emb.inv_freq.detach().float().cpu()
    diff = float(torch.nan_to_num((old - new).abs(), nan=float("inf")).max()) if old.shape == new.shape else float("inf")
    print(f"EvaByte rotary tables rebuilt; the loaded inverse frequencies were off by {diff:.3g}", flush=True)
    return diff


def _logits(out):
    return out.logits if hasattr(out, "logits") else out[0]


def manual_greedy(model, tokenizer, input_ids, max_new_tokens):
    pkg = sys.modules[type(model).__module__].__name__.rsplit(".", 1)[0]
    EvaCache = importlib.import_module(pkg + ".eva_cache").EvaCache
    dev = input_ids.device
    cache = EvaCache()
    n = input_ids.shape[1]
    stop = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")}
    gen = []
    with torch.no_grad():
        out = model(input_ids=input_ids, position_ids=torch.arange(n, device=dev)[None],
                    past_key_values=cache, use_cache=True)
        logits = _logits(out)[:, -1]
        for step in range(max_new_tokens):
            nxt = int(logits.argmax(-1))
            if nxt in stop:
                break
            gen.append(nxt)
            out = model(input_ids=torch.tensor([[nxt]], device=dev), position_ids=torch.tensor([[n + step]], device=dev),
                        past_key_values=cache, use_cache=True)
            logits = _logits(out)[:, -1]
    return gen
