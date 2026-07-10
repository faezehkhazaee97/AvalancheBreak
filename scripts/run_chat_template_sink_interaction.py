import matplotlib
matplotlib.use("Agg")

# ============================================================
# DO CHAT-TEMPLATE TOKENS THEMSELVES ACT AS SINKS, AND DO THEY
# INTERACT WITH BOUNDARY DENSITY?
# ============================================================
# Reviewer ask: Section 2.3 shows chat-template formatting materially
# changes results, and this paper's own mechanism story is that
# boundary tokens become attention sinks. Do the chat-template's own
# special tokens (<|im_start|>, <|im_end|> for Qwen2.5) ALSO act as
# sinks, and if so, does attention mass on them trade off against
# attention mass on boundary tokens as density increases (i.e. are
# boundary tokens and template tokens competing for the same limited
# "sink budget", or largely independent)?
#
# Same diagnostic methodology as the existing boundary-attention-mass
# sweep (Section 3.3): forward pass with output_attentions=True,
# measure mean attention mass landing on template-token positions
# (<|im_start|>, <|im_end|>, and the "system"/"user"/"assistant" role
# tokens immediately following <|im_start|>) vs. on boundary-token
# positions, across the same density range, same lengths, same model.

import os

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random
import numpy as np

print(f"Loading {MODEL_ID} into VRAM (eager attention, need output_attentions)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager",
)
model.eval()

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

import pandas as pd
from tqdm import tqdm


class TaskEngine:
    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size
        self.alphabet = [chr(ord('A') + i) for i in range(alphabet_size)]
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}

    def generate_raw_input(self, length, rng=None):
        if rng is None:
            rng = random
        return rng.choices(self.alphabet, k=length)

    def build_cellular_from_raw(self, raw_input, interval):
        input_w_bounds = ""
        for i, char in enumerate(raw_input):
            input_w_bounds += char
            if interval > 0 and (i + 1) % interval == 0 and i != len(raw_input) - 1:
                input_w_bounds += "|"
        prompt = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(self.alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {self.alphabet[-1]} -> {self.alphabet[0]}, and {self.alphabet[0]} -> {self.alphabet[1]}.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_w_bounds}
Output: """
        return prompt


ALPHABET_SIZE = 2
SEQUENCE_LENGTHS = [16, 64, 256]
DENSITIES = [1, 2, 3, 4, 8]
N_TRIALS = 20

engine = TaskEngine(ALPHABET_SIZE)

IM_START_ID = tokenizer.convert_tokens_to_ids("<|im_start|>")
IM_END_ID = tokenizer.convert_tokens_to_ids("<|im_end|>")
print(f"<|im_start|> id={IM_START_ID}, <|im_end|> id={IM_END_ID}")


def analyze_positions(input_ids_row, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(input_ids_row.tolist())
    template_idx = [i for i, tid in enumerate(input_ids_row.tolist()) if tid in (IM_START_ID, IM_END_ID)]
    input_idx = next(i for i, t in enumerate(tokens) if 'Input' in t)
    output_idx = next(i for i in range(input_idx, len(tokens)) if 'Output' in tokens[i])
    boundary_idx = [i for i in range(input_idx, output_idx) if tokens[i] == '|']
    return template_idx, boundary_idx


results = []
for A_interval in DENSITIES:
    for N in SEQUENCE_LENGTHS:
        template_masses, boundary_masses = [], []
        for trial in range(N_TRIALS):
            rng = random.Random(10_000 * ALPHABET_SIZE + 100 * N + trial)
            raw_input = engine.generate_raw_input(N, rng=rng)
            prompt = engine.build_cellular_from_raw(raw_input, interval=A_interval)
            enc = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            )
            input_ids = enc["input_ids"].to(model.device)
            attn_mask = enc["attention_mask"].to(model.device)
            template_idx, boundary_idx = analyze_positions(enc["input_ids"][0], tokenizer)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)

            # Mean, over all layers and heads, of total attention mass
            # (averaged over query positions) landing on template vs.
            # boundary token positions.
            n_layers = len(out.attentions)
            template_mass_trial, boundary_mass_trial = 0.0, 0.0
            for layer in range(n_layers):
                layer_attn = out.attentions[layer][0]  # (heads, q_len, k_len)
                if template_idx:
                    template_mass_trial += layer_attn[:, :, template_idx].sum(dim=-1).mean().item()
                if boundary_idx:
                    boundary_mass_trial += layer_attn[:, :, boundary_idx].sum(dim=-1).mean().item()
            template_masses.append(template_mass_trial / n_layers)
            boundary_masses.append(boundary_mass_trial / n_layers if boundary_idx else float("nan"))

        results.append({
            "Interval": A_interval, "Seq_Length": N,
            "Mean_Template_Attn_Mass": np.mean(template_masses),
            "SEM_Template_Attn_Mass": np.std(template_masses) / np.sqrt(N_TRIALS),
            "Mean_Boundary_Attn_Mass": np.nanmean(boundary_masses),
            "SEM_Boundary_Attn_Mass": np.nanstd(boundary_masses) / np.sqrt(N_TRIALS),
        })
        print(f"interval={A_interval} N={N}: template_mass={np.mean(template_masses):.5f}, boundary_mass={np.nanmean(boundary_masses):.5f}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "chat_template_sink_interaction.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

# Correlation between boundary density and template-token attention mass,
# per length -- does template attention shrink as boundary density rises
# (competing for a shared sink budget) or stay flat (independent)?
from scipy.stats import pearsonr
print("\n--- Correlation: boundary density (1/k) vs. template-token attention mass ---")
for N in SEQUENCE_LENGTHS:
    sub = df[df.Seq_Length == N].sort_values("Interval")
    density = 1.0 / sub["Interval"]
    r, p = pearsonr(density, sub["Mean_Template_Attn_Mass"])
    print(f"N={N}: r={r:.4f}, p={p:.4f}")

print("\nDone.")
