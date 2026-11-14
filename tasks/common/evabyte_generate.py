"""Greedy decoding for the byte-level EvaByte-SFT model without the library's generate().

EvaByte's remote code keeps its own cache class (EvaCache) and its config asks for tuple outputs;
current transformers releases hand generate() a DynamicCache that the model cannot use. This loop
prefills the prompt in one call with an EvaCache, then feeds one byte per step with explicit
position ids. Used only for EvaByte (tasks/common/runner.py); other models are unchanged.
Run with EVABYTE_NO_TRITON=1 so the model selects its pure-PyTorch attention.
"""

import importlib
import sys

import torch


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
