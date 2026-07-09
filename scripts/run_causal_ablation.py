import matplotlib
matplotlib.use("Agg")

# ============================================================
# CAUSAL ABLATION: is boundary attention NECESSARY for the accuracy gain?
# ============================================================
# Reviewer ask (Q8): the attention-sink story (Figs 3/11/12) is
# correlational -- boundary attention mass correlates with density, and
# density correlates with lower alpha, but nothing so far shows the
# attention-sink mechanism is actually LOAD-BEARING. This does the
# causal version: identify the heads that attend to boundary tokens the
# most (fresh diagnostic pass, not just eyeballing Fig 12's heatmap),
# then during generation, zero out attention TO boundary token positions
# specifically for those heads (renormalizing each row so it still sums
# to 1 -- a clean "this head can no longer see the boundary" ablation,
# not a magnitude reduction) and re-run the headline condition
# (alphabet=2, k=2, same seeds as thermodynamics_grid_templated_v2.csv).
# If ablating those heads drags TAR back toward Base, the mechanism is
# necessary, not just correlated with the benefit.
#
# Only the intervention condition is run fresh here -- Base and normal
# (non-ablated) Cellular_Bounds already exist in
# thermodynamics_grid_templated_v2.csv at the identical seeds, so this
# is a direct paired addition to that existing grid, not a full rerun.

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

# ============================================================
# TASK ENGINE (identical to run_core_templated_v2.py)
# ============================================================
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
TOP_K_HEADS = 10  # how many of the strongest boundary-attending heads to ablate

engine = TaskEngine(ALPHABET_SIZE)

# Same seed formula as thermodynamics_grid_templated_v2.csv, so this is a
# paired comparison against the existing non-ablated Cellular_Bounds rows.
shared_sequences = {}
for N in SEQUENCE_LENGTHS:
    for t in range(TRIALS_T):
        local_seed = 10_000 * ALPHABET_SIZE + 100 * N + t
        rng = random.Random(local_seed)
        shared_sequences[(N, t)] = engine.generate_raw_input(N, rng=rng)
print(f"Pregenerated {len(shared_sequences)} shared raw sequences (paired with thermodynamics_grid_templated_v2.csv).")


def find_boundary_token_indices(input_ids_row, tokenizer):
    """Locate boundary '|' token positions strictly within the Input:...Output:
    span (excludes the '(|)' mention of the marker in the instructions text,
    and excludes <|im_start|>/<|im_end|> chat-template tokens, both of which
    a naive substring search over the whole sequence would incorrectly catch).
    Same crop convention already used in multitrial_attention_figures.py."""
    tokens = tokenizer.convert_ids_to_tokens(input_ids_row.tolist())
    input_idx = next(i for i, t in enumerate(tokens) if 'Input' in t)
    output_idx = next(i for i in range(input_idx, len(tokens)) if 'Output' in tokens[i])
    return [i for i in range(input_idx, output_idx) if tokens[i] == '|']


# ============================================================
# STEP 1: fresh diagnostic pass to identify top boundary-attending heads
# ============================================================
print("\n--- Diagnostic pass: identifying top boundary-attending heads ---")
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
        # (batch=1, heads, q_len, k_len) -- mean over query positions of mass landing on boundary_idx
        layer_attn = out.attentions[layer][0]  # (heads, q_len, k_len)
        mass_accum[layer] += layer_attn[:, :, boundary_idx].sum(dim=-1).mean(dim=-1).float().cpu().numpy()

mass_accum /= n_diag_trials
flat_idx = np.argsort(mass_accum.ravel())[::-1][:TOP_K_HEADS]
TARGET_HEADS = [(int(i // n_heads), int(i % n_heads)) for i in flat_idx]
print(f"Top {TOP_K_HEADS} boundary-attending (layer, head) pairs: {TARGET_HEADS}")
print(f"Their boundary-attention mass: {[round(mass_accum[l, h], 4) for l, h in TARGET_HEADS]}")

pd.DataFrame(TARGET_HEADS, columns=["Layer", "Head"]).assign(
    Mass=[mass_accum[l, h] for l, h in TARGET_HEADS]
).to_csv(os.path.join(RESULTS_DIR, "causal_ablation_target_heads.csv"), index=False)

TARGET_SET = set(TARGET_HEADS)
TARGET_BY_LAYER = {}
for l, h in TARGET_HEADS:
    TARGET_BY_LAYER.setdefault(l, []).append(h)

# ============================================================
# STEP 2: monkey-patch eager attention to ablate target heads' access
# to boundary token positions (zero + renormalize)
# ============================================================
import transformers.models.qwen2.modeling_qwen2 as qwen2_mod

_ORIGINAL_EAGER_ATTENTION_FORWARD = qwen2_mod.eager_attention_forward
CURRENT_BOUNDARY_IDX = []  # set fresh before each generate() call
ABLATION_ACTIVE = False


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
            # attn_weights: (batch, heads, q_len, k_len); only zero for target heads
            # and only for key positions that are actually within range (k_len grows
            # during incremental decoding but boundary positions are in the prompt,
            # always < current k_len once the prompt has been processed).
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
print("\nMonkey-patched eager_attention_forward for causal ablation.")


# ============================================================
# STEP 3: run the ablated Cellular_Bounds condition
# ============================================================
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

    return {
        "Method": "Cellular_Bounds_Ablated", "Alphabet_Size": ALPHABET_SIZE, "Interval": INTERVAL,
        "Seq_Length": N, "Trial": t, "TAR": tar, "SAR": sar,
        "Input_Given": inp, "Ground_Truth": clean_truth, "LLM_Output": clean_pred,
        "N_Target_Heads": len(TARGET_HEADS),
    }


tasks = [(N, t) for N in SEQUENCE_LENGTHS for t in range(TRIALS_T)]
print(f"\nStarting causal ablation sweep: {len(tasks)} evaluations...")

results = []
for N, t in tqdm(tasks, desc="Ablated Cellular_Bounds"):
    try:
        results.append(run_ablated_trial(N, t))
    except Exception as exc:
        print(f"Task failed: N={N}, t={t} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "causal_ablation_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

summary = df.groupby("Seq_Length")[["TAR", "SAR"]].mean()
print("\n--- ABLATED Cellular_Bounds TAR/SAR by length ---")
print(summary.round(4).to_string())

# --- Compare against the existing non-ablated grid ---
# Base rows always have Interval==0 (boundaries are a Cellular_Bounds-only
# concept), so the Interval filter must only apply to the Cellular_Bounds
# rows -- applying it uniformly to both methods silently drops all Base
# rows and produces NaN for Base_TAR.
base_df = pd.read_csv(os.path.join(RESULTS_DIR, "thermodynamics_grid_templated_v2.csv"))
ref = base_df[(base_df.Alphabet_Size == ALPHABET_SIZE) & (base_df.Seq_Length.isin(SEQUENCE_LENGTHS))]
ref_cellular = ref[(ref.Method == "Cellular_Bounds") & (ref.Interval == INTERVAL)].groupby("Seq_Length")["TAR"].mean()
ref_base = ref[ref.Method == "Base"].groupby("Seq_Length")["TAR"].mean()

comparison = pd.DataFrame({
    "Base_TAR": ref_base, "Cellular_Normal_TAR": ref_cellular, "Cellular_Ablated_TAR": summary["TAR"],
})
comparison["Ablation_Effect"] = comparison["Cellular_Ablated_TAR"] - comparison["Cellular_Normal_TAR"]
print("\n--- DOES ABLATING TOP BOUNDARY-ATTENDING HEADS DRAG PERFORMANCE TOWARD BASE? ---")
print(comparison.round(4).to_string())
comparison.to_csv(os.path.join(RESULTS_DIR, "causal_ablation_comparison.csv"))

qwen2_mod.eager_attention_forward = _ORIGINAL_EAGER_ATTENTION_FORWARD
print("\nRestored original attention function. Done.")
