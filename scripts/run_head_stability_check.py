import matplotlib
matplotlib.use("Agg")

# ============================================================
# HEAD STABILITY CHECK
# ============================================================
# Reviewer question: "How stable are the identified sink heads across
# seeds, lengths, and densities? Is there a persistent subset of heads,
# or does the boundary-attending set drift?"
#
# Reuses the EXACT head-identification procedure from run_causal_ablation.py
# (same TaskEngine, same find_boundary_token_indices crop, same top-10-by-
# mean-boundary-attention-mass selection) but repeats it under several
# different conditions -- a different trial/seed subset, a different
# sequence length, and different boundary densities -- and measures Jaccard
# overlap of each condition's top-10 head set against the original
# reference set (N=16, k=2, alphabet=2, trials 0-4, already saved in
# causal_ablation_target_heads.csv).

import os

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random
import numpy as np
import pandas as pd

print(f"Loading {MODEL_ID} into VRAM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager",
)
model.eval()

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

ALPHABET_SIZE = 2
TOP_K_HEADS = 10


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


engine = TaskEngine(ALPHABET_SIZE)


def find_boundary_token_indices(input_ids_row, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(input_ids_row.tolist())
    input_idx = next(i for i, t in enumerate(tokens) if 'Input' in t)
    output_idx = next(i for i in range(input_idx, len(tokens)) if 'Output' in tokens[i])
    return [i for i in range(input_idx, output_idx) if tokens[i] == '|']


n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads


def identify_top_heads(N, interval, trial_offset, n_trials=5):
    mass_accum = np.zeros((n_layers, n_heads))
    for t in range(trial_offset, trial_offset + n_trials):
        local_seed = 10_000 * ALPHABET_SIZE + 100 * N + t
        rng = random.Random(local_seed)
        raw_input = engine.generate_raw_input(N, rng=rng)
        _, _, prompt = engine.build_cellular_from_raw(raw_input, interval=interval)
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        input_ids = enc["input_ids"].to(model.device)
        attn_mask = enc["attention_mask"].to(model.device)
        boundary_idx = find_boundary_token_indices(enc["input_ids"][0], tokenizer)
        if not boundary_idx:
            continue
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
        for layer in range(n_layers):
            layer_attn = out.attentions[layer][0]
            mass_accum[layer] += layer_attn[:, :, boundary_idx].sum(dim=-1).mean(dim=-1).float().cpu().numpy()
    mass_accum /= n_trials
    flat_idx = np.argsort(mass_accum.ravel())[::-1][:TOP_K_HEADS]
    return set((int(i // n_heads), int(i % n_heads)) for i in flat_idx)


# Reference condition: matches run_causal_ablation.py exactly (N=16, k=2, trials 0-4)
CONDITIONS = {
    "reference_N16_k2_t0-4": dict(N=16, interval=2, trial_offset=0),
    "diff_seeds_N16_k2_t5-9": dict(N=16, interval=2, trial_offset=5),
    "diff_seeds_N16_k2_t10-14": dict(N=16, interval=2, trial_offset=10),
    "diff_length_N32_k2_t0-4": dict(N=32, interval=2, trial_offset=0),
    "diff_length_N64_k2_t0-4": dict(N=64, interval=2, trial_offset=0),
    "diff_density_N16_k1_t0-4": dict(N=16, interval=1, trial_offset=0),
    "diff_density_N16_k4_t0-4": dict(N=16, interval=4, trial_offset=0),
}

print("\n--- Identifying top-10 boundary-attending heads under each condition ---")
head_sets = {}
for name, cfg in CONDITIONS.items():
    heads = identify_top_heads(**cfg)
    head_sets[name] = heads
    print(f"{name}: {sorted(heads)}")

reference = head_sets["reference_N16_k2_t0-4"]
rows = []
for name, heads in head_sets.items():
    overlap = reference & heads
    jaccard = len(overlap) / len(reference | heads) if (reference | heads) else float("nan")
    rows.append({
        "Condition": name, "N_Overlap_With_Reference": len(overlap),
        "Jaccard_vs_Reference": jaccard, "Heads": sorted(heads),
    })

summary = pd.DataFrame(rows)
out_path = os.path.join(RESULTS_DIR, "head_stability_summary.csv")
summary.to_csv(out_path, index=False)
print(f"\nSaved {out_path}")
print(summary.to_string())

# Heads present in EVERY condition tested (the persistent core, if any)
persistent = set.intersection(*head_sets.values())
print(f"\nHeads present in ALL {len(head_sets)} conditions (persistent core): {sorted(persistent)}")

# Union size vs top-10 size: how many distinct heads ever appear across all conditions
all_heads_seen = set.union(*head_sets.values())
print(f"Total distinct heads appearing in any condition's top-10: {len(all_heads_seen)} (vs {TOP_K_HEADS} per condition)")
