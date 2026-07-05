import matplotlib
matplotlib.use("Agg")

# ============================================================
# DOES THE CHAT TEMPLATE CHANGE THE ATTENTION PATTERN ITSELF?
# ============================================================
# We already showed raw-vs-templated prompting changes the model's OUTPUT
# behavior (rambling vs clean stops, case-collapse). This checks whether it
# also changes what's happening INTERNALLY -- the actual attention weights
# -- using the exact same representative example the original Figure 2/3
# attention analysis was built from (Alphabet=2, N=16, k=2, space-separated
# characters, ATTENTION_SEED=123). Crops both raw and templated attention
# tensors to just the "Input: ... Output:" token span (excluding the
# chat-template wrapper tokens) so the two are directly comparable.

import os

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random
import numpy as np

print(f"Loading {MODEL_ID} into VRAM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager",
)
model.eval()

num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads

ATTENTION_SEED = 123
random.seed(ATTENTION_SEED)


class TaskEngine:
    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size
        self.alphabet = [chr(ord('A') + i) for i in range(alphabet_size)]

    def generate_paired_prompts(self, length, interval, rng=None):
        rng = rng or random
        raw_chars = rng.choices(self.alphabet, k=length)

        base_str = " ".join(raw_chars)
        prompt_base = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(self.alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {self.alphabet[-1]} -> {self.alphabet[0]}, and {self.alphabet[0]} -> {self.alphabet[1]}.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {base_str}
Output: """

        cellular_w_bounds = []
        for i, char in enumerate(raw_chars):
            cellular_w_bounds.append(char)
            if interval > 0 and (i + 1) % interval == 0 and i != length - 1:
                cellular_w_bounds.append("|")
        cell_str = " ".join(cellular_w_bounds)

        prompt_cell = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(self.alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {self.alphabet[-1]} -> {self.alphabet[0]}, and {self.alphabet[0]} -> {self.alphabet[1]}.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {cell_str}
Output: """
        return prompt_base, prompt_cell


engine = TaskEngine(alphabet_size=2)
N_length = 16
k_interval = 2
prompt_base, prompt_cell = engine.generate_paired_prompts(length=N_length, interval=k_interval)


def extract_all_layers_and_heads(prompt, use_template):
    if use_template:
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in enc.items()}
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    tokens = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]

    try:
        input_idx = next(i for i, t in enumerate(tokens) if 'Input' in t)
        output_idx = next(i for i in range(input_idx, len(tokens)) if 'Output' in tokens[i])
        start_idx = input_idx + 1
        while start_idx < output_idx and tokens[start_idx].strip() in [':', '']:
            start_idx += 1
        end_idx = output_idx
        while end_idx > start_idx and tokens[end_idx - 1].strip() == '':
            end_idx -= 1
    except StopIteration:
        start_idx, end_idx = 0, len(tokens)

    clean_tokens = [t.strip() if t.strip() else '_' for t in tokens[start_idx:end_idx]]

    attentions = outputs.attentions
    seq_len = end_idx - start_idx
    full_tensor = np.zeros((num_layers, num_heads, seq_len, seq_len), dtype=np.float32)
    for l in range(num_layers):
        layer_attn = attentions[l][0]
        for h in range(num_heads):
            attn_matrix = layer_attn[h].cpu().to(torch.float32).numpy()
            full_tensor[l, h] = attn_matrix[start_idx:end_idx, start_idx:end_idx]

    return full_tensor, clean_tokens


print("\n=== RAW prompting ===")
attn_base_raw, tokens_base_raw = extract_all_layers_and_heads(prompt_base, use_template=False)
attn_cell_raw, tokens_cell_raw = extract_all_layers_and_heads(prompt_cell, use_template=False)
print(f"Base tokens (raw): {tokens_base_raw}")
print(f"Cell tokens (raw): {tokens_cell_raw}")

print("\n=== TEMPLATED prompting ===")
attn_base_tmpl, tokens_base_tmpl = extract_all_layers_and_heads(prompt_base, use_template=True)
attn_cell_tmpl, tokens_cell_tmpl = extract_all_layers_and_heads(prompt_cell, use_template=True)
print(f"Base tokens (templated): {tokens_base_tmpl}")
print(f"Cell tokens (templated): {tokens_cell_tmpl}")

# Sanity check: cropped token spans should match content (same underlying prompt content)
print(f"\nToken spans match between raw and templated (base)? {tokens_base_raw == tokens_base_tmpl}")
print(f"Token spans match between raw and templated (cell)?  {tokens_cell_raw == tokens_cell_tmpl}")

LAYER = 13
for head_idx in [7, 10]:
    print(f"\n{'='*70}")
    print(f"LAYER {LAYER}, HEAD {head_idx} -- BASE PROMPT")
    print(f"{'='*70}")
    m_raw = attn_base_raw[LAYER, head_idx]
    m_tmpl = attn_base_tmpl[LAYER, head_idx]
    if m_raw.shape == m_tmpl.shape:
        diag_raw = np.mean(np.diag(m_raw))
        diag_tmpl = np.mean(np.diag(m_tmpl))
        corr = np.corrcoef(m_raw.flatten(), m_tmpl.flatten())[0, 1]
        max_abs_diff = np.max(np.abs(m_raw - m_tmpl))
        print(f"Mean diagonal attention -- raw: {diag_raw:.4f}, templated: {diag_tmpl:.4f}")
        print(f"Pearson correlation between raw and templated attention matrices: {corr:.4f}")
        print(f"Max absolute per-cell difference: {max_abs_diff:.4f}")
    else:
        print(f"Shape mismatch: raw={m_raw.shape}, templated={m_tmpl.shape} (token cropping differed)")

    print(f"\n--- raw matrix (rounded) ---")
    print(np.round(m_raw, 3))
    print(f"\n--- templated matrix (rounded) ---")
    print(np.round(m_tmpl, 3))

print("\nAttention comparison complete.")
