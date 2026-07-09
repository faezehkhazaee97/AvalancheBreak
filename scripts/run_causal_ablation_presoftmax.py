import matplotlib
matplotlib.use("Agg")

# ============================================================
# ALTERNATIVE CAUSAL PROBE: pre-softmax masking vs post-softmax
# zero-and-renormalize (run_causal_ablation.py)
# ============================================================
# Reviewer concern: the original causal ablation (run_causal_ablation.py)
# zeros attention weight AFTER softmax and renormalizes the row so it
# still sums to 1. The reviewer asks whether this could have side effects
# (on entropy, or on "local competition" among the surviving weights)
# distinct from a cleaner pre-softmax masking approach (setting the raw
# attention score to -inf before softmax, exactly how a causal mask
# works), and suggests trying pre-softmax masking as an alternative.
#
# These two operations are actually mathematically identical: zeroing a
# softmax output at a subset of positions and renormalizing the remainder
# to sum to 1 is algebraically the same as computing softmax over only
# the remaining (non-masked) positions from the start (both reduce to
# exp(x_i) / sum_{j in kept}(exp(x_j))). This script verifies that
# equivalence empirically rather than asserting it: same model, same 10
# target heads (reused directly from causal_ablation_target_heads.csv,
# not re-diagnosed), same seeds, same headline condition (alphabet=2,
# k=2) -- only the ablation mechanism itself changes, from
# zero-then-renormalize to mask-then-softmax.

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
# Reuse the EXACT same 10 target heads already identified in
# run_causal_ablation.py -- this script tests a different ablation
# mechanism on the same heads, not a fresh head-selection.
# ============================================================
target_heads_df = pd.read_csv(os.path.join(RESULTS_DIR, "causal_ablation_target_heads.csv"))
TARGET_HEADS = list(zip(target_heads_df["Layer"], target_heads_df["Head"]))
print(f"Reusing {len(TARGET_HEADS)} target heads from causal_ablation_target_heads.csv: {TARGET_HEADS}")

TARGET_BY_LAYER = {}
for l, h in TARGET_HEADS:
    TARGET_BY_LAYER.setdefault(int(l), []).append(int(h))

# ============================================================
# Monkey-patch: mask BEFORE softmax (set raw score to -inf at boundary
# positions for target heads), then softmax as normal -- the softmax
# itself does the renormalization, with no separate zero+divide step.
# ============================================================
import transformers.models.qwen2.modeling_qwen2 as qwen2_mod

_ORIGINAL_EAGER_ATTENTION_FORWARD = qwen2_mod.eager_attention_forward
CURRENT_BOUNDARY_IDX = []
ABLATION_ACTIVE = False
NEG_INF = torch.finfo(torch.bfloat16).min


def presoftmax_masked_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = qwen2_mod.repeat_kv(key, module.num_key_value_groups)
    value_states = qwen2_mod.repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    if ABLATION_ACTIVE and CURRENT_BOUNDARY_IDX:
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx in TARGET_BY_LAYER:
            heads_here = TARGET_BY_LAYER[layer_idx]
            idx_tensor = torch.tensor(CURRENT_BOUNDARY_IDX, device=attn_weights.device, dtype=torch.long)
            k_len = attn_weights.shape[-1]
            valid_idx = idx_tensor[idx_tensor < k_len]
            if valid_idx.numel() > 0:
                for h in heads_here:
                    attn_weights[:, h, :, valid_idx] = NEG_INF

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


qwen2_mod.eager_attention_forward = presoftmax_masked_attention_forward
print("\nMonkey-patched eager_attention_forward for PRE-SOFTMAX masked ablation.")


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
        "Method": "Cellular_Bounds_Ablated_PreSoftmax", "Alphabet_Size": ALPHABET_SIZE, "Interval": INTERVAL,
        "Seq_Length": N, "Trial": t, "TAR": tar, "SAR": sar,
        "Input_Given": inp, "Ground_Truth": clean_truth, "LLM_Output": clean_pred,
        "N_Target_Heads": len(TARGET_HEADS),
    }


tasks = [(N, t) for N in SEQUENCE_LENGTHS for t in range(TRIALS_T)]
print(f"\nStarting pre-softmax-masked ablation sweep: {len(tasks)} evaluations...")

results = []
for N, t in tqdm(tasks, desc="Ablated Cellular_Bounds (pre-softmax)"):
    try:
        results.append(run_ablated_trial(N, t))
    except Exception as exc:
        print(f"Task failed: N={N}, t={t} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "causal_ablation_presoftmax_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

summary = df.groupby("Seq_Length")[["TAR", "SAR"]].mean()
print("\n--- PRE-SOFTMAX-MASKED Cellular_Bounds TAR/SAR by length ---")
print(summary.round(4).to_string())

# --- Compare directly against the original post-softmax zero+renormalize ablation ---
orig_ablated = pd.read_csv(os.path.join(RESULTS_DIR, "causal_ablation_grid.csv"))
orig_summary = orig_ablated.groupby("Seq_Length")["TAR"].mean()

base_df = pd.read_csv(os.path.join(RESULTS_DIR, "thermodynamics_grid_templated_v2.csv"))
ref = base_df[(base_df.Alphabet_Size == ALPHABET_SIZE) & (base_df.Seq_Length.isin(SEQUENCE_LENGTHS))]
ref_cellular = ref[(ref.Method == "Cellular_Bounds") & (ref.Interval == INTERVAL)].groupby("Seq_Length")["TAR"].mean()
ref_base = ref[ref.Method == "Base"].groupby("Seq_Length")["TAR"].mean()

comparison = pd.DataFrame({
    "Base_TAR": ref_base,
    "Cellular_Normal_TAR": ref_cellular,
    "Ablated_PostSoftmax_TAR": orig_summary,
    "Ablated_PreSoftmax_TAR": summary["TAR"],
})
comparison["Method_Diff"] = comparison["Ablated_PreSoftmax_TAR"] - comparison["Ablated_PostSoftmax_TAR"]
print("\n--- DOES THE ABLATION MECHANISM (pre- vs post-softmax) MATTER? ---")
print(comparison.round(4).to_string())
comparison.to_csv(os.path.join(RESULTS_DIR, "causal_ablation_mechanism_comparison.csv"))

qwen2_mod.eager_attention_forward = _ORIGINAL_EAGER_ATTENTION_FORWARD
print("\nRestored original attention function. Done.")
