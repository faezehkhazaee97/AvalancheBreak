"""Quick check before spending a full sweep on this: does the attention-ablation
monkey-patch actually change the model's output at all, and does it run without
shape errors? Runs 3 trials with ablation off, 3 with it on, same inputs, and
diffs the outputs. If ablation-off matches the existing non-ablated grid's
accuracy pattern and ablation-on produces different (ideally worse) outputs,
the patch is doing something real, not silently no-op-ing."""

import os
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, torch.nn as nn, random

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager",
)
model.eval()


class TaskEngine:
    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size
        self.alphabet = [chr(ord('A') + i) for i in range(alphabet_size)]
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}

    def generate_raw_input(self, length, rng=None):
        rng = rng or random
        return rng.choices(self.alphabet, k=length)

    def build_cellular_from_raw(self, raw_input, interval):
        input_w_bounds, output_w_bounds = "", ""
        for i, char in enumerate(raw_input):
            shifted_char = self.alphabet[(self.char_to_idx[char] + 1) % self.alphabet_size]
            input_w_bounds += char
            output_w_bounds += shifted_char
            if interval > 0 and (i + 1) % interval == 0 and i != len(raw_input) - 1:
                input_w_bounds += "|"
                output_w_bounds += "|"
        prompt = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(self.alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {self.alphabet[-1]} -> {self.alphabet[0]}, and {self.alphabet[0]} -> {self.alphabet[1]}.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_w_bounds}
Output: """
        return input_w_bounds, output_w_bounds, prompt


engine = TaskEngine(2)


def find_boundary_token_indices(input_ids_row, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(input_ids_row.tolist())
    input_idx = next(i for i, t in enumerate(tokens) if 'Input' in t)
    output_idx = next(i for i in range(input_idx, len(tokens)) if 'Output' in tokens[i])
    return [i for i in range(input_idx, output_idx) if tokens[i] == '|']


import transformers.models.qwen2.modeling_qwen2 as qwen2_mod

CALL_COUNT = {"total": 0, "ablated_rows_touched": 0}
CURRENT_BOUNDARY_IDX = []
ABLATION_ACTIVE = False
TARGET_BY_LAYER = {l: [0, 1, 2, 3, 4] for l in range(5)}  # crude test target: heads 0-4 in layers 0-4


def ablated_eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    CALL_COUNT["total"] += 1
    key_states = qwen2_mod.repeat_kv(key, module.num_key_value_groups)
    value_states = qwen2_mod.repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    if ABLATION_ACTIVE and CURRENT_BOUNDARY_IDX:
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx in TARGET_BY_LAYER:
            heads_here = TARGET_BY_LAYER[layer_idx]
            idx_tensor = torch.tensor(CURRENT_BOUNDARY_IDX, device=attn_weights.device, dtype=torch.long)
            k_len = attn_weights.shape[-1]
            valid_idx = idx_tensor[idx_tensor < k_len]
            if valid_idx.numel() > 0:
                before = attn_weights[:, heads_here, :, :][:, :, :, valid_idx].sum().item()
                for h in heads_here:
                    attn_weights[:, h, :, valid_idx] = 0.0
                row_sums = attn_weights[:, heads_here, :, :].sum(dim=-1, keepdim=True)
                attn_weights[:, heads_here, :, :] = attn_weights[:, heads_here, :, :] / row_sums.clamp_min(1e-12)
                after = attn_weights[:, heads_here, :, :][:, :, :, valid_idx].sum().item()
                CALL_COUNT["ablated_rows_touched"] += 1
                if CALL_COUNT["ablated_rows_touched"] <= 3:
                    print(f"    [ablation active] layer={layer_idx} boundary-mass before={before:.4f} after={after:.4f}")

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


qwen2_mod.eager_attention_forward = ablated_eager_attention_forward
print("Patched.")

rng = random.Random(999)
raw = engine.generate_raw_input(16, rng=rng)
_, truth, prompt = engine.build_cellular_from_raw(raw, interval=2)
enc = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
)
inputs = {k: v.to(model.device) for k, v in enc.items()}
boundary_idx = find_boundary_token_indices(enc["input_ids"][0], tokenizer)
print(f"Truth: {truth}")
print(f"Boundary token indices found: {boundary_idx} (expect 7 for N=16, interval=2)")

for label, ablate in [("ABLATION OFF", False), ("ABLATION ON", True)]:
    CURRENT_BOUNDARY_IDX = boundary_idx
    ABLATION_ACTIVE = ablate
    CALL_COUNT["total"] = 0
    CALL_COUNT["ablated_rows_touched"] = 0
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    ABLATION_ACTIVE = False
    gen_only = out[0][inputs["input_ids"].shape[1]:]
    pred = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
    print(f"\n=== {label} ===")
    print(f"Attention fn calls: {CALL_COUNT['total']}, rows touched by ablation: {CALL_COUNT['ablated_rows_touched']}")
    print(f"Prediction: {pred!r}")
    print(f"Matches truth: {pred.split(chr(10))[0].strip() == truth}")

print("\nSmoke test complete.")
