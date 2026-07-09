import matplotlib
matplotlib.use("Agg")

# ============================================================
# DOSE-RESPONSE CURVE FOR THE CAUSAL ABLATION
# ============================================================
# run_causal_ablation.py ablates a single fixed count (10, chosen
# somewhat arbitrarily) of the top boundary-attending heads and shows a
# large TAR drag. This asks the natural follow-up: how does the drag
# scale with how many of the top boundary-attending heads are ablated?
# Same mechanism (zero post-softmax, renormalize), same headline
# condition (alphabet=2, k=2) and seeds as every other ablation script,
# but repeated for K in {1, 2, 5, 10, 20, 40, 80} heads (out of 784
# total), always taking the top-K by boundary-attention mass from the
# same diagnostic ranking used throughout (5 trials, N=16).

import os

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, torch.nn as nn, random
import numpy as np

print(f"Loading {MODEL_ID} into VRAM (eager attention, needed for the monkey-patch below)...")
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


ALPHABET_SIZE = 2
INTERVAL = 2
SEQUENCE_LENGTHS = [8, 16, 32, 64]
TRIALS_T = 20
K_VALUES = [1, 2, 5, 10, 20, 40, 80]

engine = TaskEngine(ALPHABET_SIZE)

shared_sequences = {}
for N in SEQUENCE_LENGTHS:
    for t in range(TRIALS_T):
        local_seed = 10_000 * ALPHABET_SIZE + 100 * N + t
        rng = random.Random(local_seed)
        shared_sequences[(N, t)] = engine.generate_raw_input(N, rng=rng)
print(f"Pregenerated {len(shared_sequences)} shared raw sequences (paired with thermodynamics_grid_templated_v2.csv).")


def find_boundary_token_indices(input_ids_row, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(input_ids_row.tolist())
    input_idx = next(i for i, t in enumerate(tokens) if 'Input' in t)
    output_idx = next(i for i in range(input_idx, len(tokens)) if 'Output' in tokens[i])
    return [i for i in range(input_idx, output_idx) if tokens[i] == '|']


# ============================================================
# STEP 1: diagnostic pass -> full boundary-attention-mass ranking
# (identical procedure to run_causal_ablation.py's Step 1, and to
# run_causal_ablation_lowattend.py, so all three scripts agree on the
# same ranking).
# ============================================================
print("\n--- Diagnostic pass: full boundary-attention mass matrix ---")
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
mass_accum = np.zeros((n_layers, n_heads))
n_diag_trials = 5

for t in range(n_diag_trials):
    raw_input = shared_sequences[(16, t)]
    _, _, prompt = engine.build_cellular_from_raw(raw_input, interval=INTERVAL)
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    input_ids = enc["input_ids"].to(model.device)
    attn_mask = enc["attention_mask"].to(model.device)
    boundary_idx = find_boundary_token_indices(enc["input_ids"][0], tokenizer)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
    for layer in range(n_layers):
        layer_attn = out.attentions[layer][0]
        mass_accum[layer] += layer_attn[:, :, boundary_idx].sum(dim=-1).mean(dim=-1).float().cpu().numpy()

mass_accum /= n_diag_trials
all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
ranked_desc = sorted(all_heads, key=lambda lh: -mass_accum[lh[0], lh[1]])  # descending: highest mass first
print(f"Top-1 head: {ranked_desc[0]}, mass={mass_accum[ranked_desc[0]]:.4f}")
print(f"Top-80 mass range: {mass_accum[ranked_desc[0]]:.4f} down to {mass_accum[ranked_desc[79]]:.4f}")

# ============================================================
# STEP 2: monkey-patch eager attention. TARGET_BY_LAYER is rebuilt fresh
# before each K-value sweep below (module-level dict the patched
# function reads from each call).
# ============================================================
import transformers.models.qwen2.modeling_qwen2 as qwen2_mod

_ORIGINAL_EAGER_ATTENTION_FORWARD = qwen2_mod.eager_attention_forward
CURRENT_BOUNDARY_IDX = []
ABLATION_ACTIVE = False
TARGET_BY_LAYER = {}


def ablated_eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
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
                for h in heads_here:
                    attn_weights[:, h, :, valid_idx] = 0.0
                row_sums = attn_weights[:, heads_here, :, :].sum(dim=-1, keepdim=True)
                attn_weights[:, heads_here, :, :] = attn_weights[:, heads_here, :, :] / row_sums.clamp_min(1e-12)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


qwen2_mod.eager_attention_forward = ablated_eager_attention_forward
print("\nMonkey-patched eager_attention_forward for dose-response ablation.")


def run_ablated_trial(N, t):
    global CURRENT_BOUNDARY_IDX, ABLATION_ACTIVE
    raw_input = shared_sequences[(N, t)]
    inp, truth, prompt = engine.build_cellular_from_raw(raw_input, interval=INTERVAL)

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in enc.items()}
    CURRENT_BOUNDARY_IDX = find_boundary_token_indices(enc["input_ids"][0], tokenizer)
    ABLATION_ACTIVE = True

    max_tokens_needed = int(N * 1.5) + 20
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    ABLATION_ACTIVE = False

    gen_only = outputs[0][inputs["input_ids"].shape[1]:]
    pred = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
    pred = pred.split("Explanation")[0].split("\n")[0].strip()

    clean_pred = pred.replace("|", "").replace(" ", "").replace("\n", "")
    clean_truth = truth.replace("|", "")
    clean_pred = clean_pred[:len(clean_truth)]

    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0
                      for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0
    return tar, sar, inp, clean_truth, clean_pred


all_results = []
for K in K_VALUES:
    TARGET_HEADS = ranked_desc[:K]
    TARGET_BY_LAYER.clear()
    for l, h in TARGET_HEADS:
        TARGET_BY_LAYER.setdefault(l, []).append(h)

    tasks = [(N, t) for N in SEQUENCE_LENGTHS for t in range(TRIALS_T)]
    print(f"\n--- K={K} heads ablated: {len(tasks)} evaluations ---")
    for N, t in tqdm(tasks, desc=f"K={K}"):
        try:
            tar, sar, inp, truth, pred = run_ablated_trial(N, t)
            all_results.append({
                "Method": "Cellular_Bounds_Ablated_DoseResponse", "Alphabet_Size": ALPHABET_SIZE,
                "Interval": INTERVAL, "K_Heads_Ablated": K, "Seq_Length": N, "Trial": t,
                "TAR": tar, "SAR": sar, "Input_Given": inp, "Ground_Truth": truth, "LLM_Output": pred,
            })
        except Exception as exc:
            print(f"Task failed: K={K}, N={N}, t={t} | Error: {exc}")

df = pd.DataFrame(all_results)
csv_path = os.path.join(RESULTS_DIR, "causal_ablation_doseresponse_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

summary = df.groupby(["K_Heads_Ablated", "Seq_Length"])[["TAR", "SAR"]].mean().reset_index()
print("\n--- DOSE-RESPONSE: TAR by K (heads ablated) and length ---")
print(summary.round(4).to_string(index=False))

base_df = pd.read_csv(os.path.join(RESULTS_DIR, "thermodynamics_grid_templated_v2.csv"))
ref = base_df[(base_df.Alphabet_Size == ALPHABET_SIZE) & (base_df.Seq_Length.isin(SEQUENCE_LENGTHS))]
ref_cellular = ref[(ref.Method == "Cellular_Bounds") & (ref.Interval == INTERVAL)].groupby("Seq_Length")["TAR"].mean()
ref_base = ref[ref.Method == "Base"].groupby("Seq_Length")["TAR"].mean()

pivot = summary.pivot(index="Seq_Length", columns="K_Heads_Ablated", values="TAR")
pivot.insert(0, "Base_TAR", ref_base)
pivot.insert(1, "Cellular_Normal_TAR", ref_cellular)
print("\n--- FULL DOSE-RESPONSE TABLE (TAR) ---")
print(pivot.round(4).to_string())
pivot.to_csv(os.path.join(RESULTS_DIR, "causal_ablation_doseresponse_comparison.csv"))

qwen2_mod.eager_attention_forward = _ORIGINAL_EAGER_ATTENTION_FORWARD
print("\nRestored original attention function. Done.")
