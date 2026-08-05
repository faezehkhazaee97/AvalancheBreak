import matplotlib
matplotlib.use("Agg")

# ============================================================
# MECHANISTIC ATTENTION CAPTURE, REAL LETTER FORMAT (concatenated, no
# space-separation workaround -- the ACTUAL prompt format the accuracy
# grid uses, merged tokens and all)
# ============================================================
# Every attention capture so far either used space-separated letters
# (scripts/run_attention_templated.py, a workaround to force clean
# tokenization) or digits (scripts/run_attention_digits.py, clean without
# needing a workaround). Neither shows what attention actually looks like
# on the REAL letter-alphabet prompt format used throughout the accuracy
# grid: concatenated letters, no spaces, with real BPE merging intact
# (e.g. "AAAA" as a single token). This captures exactly that, so the
# merged-token structure is visible directly in the attention heatmap
# rather than only inferred from the tokenization audit's summary
# statistics.

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join(REPO_ROOT, "results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random
import numpy as np

print(f"Loading {MODEL_ID} into VRAM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
)
model.eval()

num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads

ATTENTION_SEED = 123
random.seed(ATTENTION_SEED)


class TaskEngine:
    """Letters, CONCATENATED -- matches tasks/cyclic_shift/engine.py's
    real prompt format exactly (no space-join workaround)."""

    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size
        self.alphabet = [chr(ord('A') + i) for i in range(alphabet_size)]
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}

    def generate_paired_prompts(self, length, interval, rng=None):
        rng = rng or random
        raw_chars = rng.choices(self.alphabet, k=length)

        base_str = "".join(raw_chars)  # concatenated -- the REAL format
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
        cell_str = "".join(cellular_w_bounds)  # concatenated -- the REAL format

        prompt_cell = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(self.alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {self.alphabet[-1]} -> {self.alphabet[0]}, and {self.alphabet[0]} -> {self.alphabet[1]}.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {cell_str}
Output: """
        return prompt_base, prompt_cell, raw_chars


engine = TaskEngine(alphabet_size=2)
N_length = 16
k_interval = 2
prompt_base, prompt_cell, raw_chars = engine.generate_paired_prompts(length=N_length, interval=k_interval)
print(f"Raw character sequence (N={N_length}): {''.join(raw_chars)}")


def extract_all_layers_and_heads(prompt):
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    tokens = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]

    input_idx = max(i for i, t in enumerate(tokens) if 'Input' in t)
    output_idx = next(i for i in range(input_idx, len(tokens)) if 'Output' in tokens[i])
    start_idx = input_idx + 1
    while start_idx < output_idx and tokens[start_idx].strip() in [':', '']:
        start_idx += 1
    end_idx = output_idx
    while end_idx > start_idx and tokens[end_idx - 1].strip() == '':
        end_idx -= 1

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


print("\nRunning forward pass for Base Prompt (REAL concatenated letters)...")
attn_base_tensor, tokens_base = extract_all_layers_and_heads(prompt_base)
print(f"Base: {N_length} characters -> {len(tokens_base)} tokens "
      f"({'CLEAN 1:1' if len(tokens_base) == N_length else 'MERGED -- ' + str(N_length - len(tokens_base)) + ' fewer tokens than characters'})")
print(f"Base tokens: {tokens_base}")

print("Running forward pass for Cellular Prompt (REAL concatenated letters)...")
attn_cell_tensor, tokens_cell = extract_all_layers_and_heads(prompt_cell)
print(f"Cellular tokens: {tokens_cell} (len={len(tokens_cell)})")

print("Saving full attention tensors (real concatenated letters)...")
np.savez_compressed(
    os.path.join(RESULTS_DIR, "attention_run_0_real_letters.npz"),
    attn_base=attn_base_tensor,
    attn_cell=attn_cell_tensor,
    tokens_base=np.array(tokens_base),
    tokens_cell=np.array(tokens_cell),
    N_length=N_length,
    k_interval=k_interval,
    raw_chars="".join(raw_chars),
)
print("Done.")
