import matplotlib
matplotlib.use("Agg")

# ============================================================
# CAUSAL ABLATION ON copy_previous: is the boundary-attention mechanism
# that CORRELATES identically on help/harm tasks (run_anchor_attention_
# geometry.py) also causally responsible for the HARM on a task boundaries
# hurt? Same pipeline as scripts/run_causal_ablation.py (cyclic_shift,
# where boundaries help), applied to copy_previous (where boundaries hurt),
# with one fix: that script's find_boundary_token_indices() takes the
# FIRST "Input"/"Output" token occurrence, which is safe for cyclic_shift's
# prompt (no earlier occurrence) but WRONG for copy_previous's prompt,
# which contains "Example: Input 1011 -> Output 1101" before the real
# "Input: ...\nOutput:" line -- the same crop bug found and fixed in
# run_anchor_attention_geometry.py. Fixed here by taking the LAST
# occurrence instead.
#
# Uses the real tasks/copy_previous/engine.py prompt/task definition and
# the study's canonical local-seed formula (90_000*N + 100*k + 10*axis_val
# + t, axis_val=None here) so the diagnostic pass, ablated run, and the
# Base/normal-Bounds reference (read from the existing pooled 40-trial
# grid, first 20 trials) all use IDENTICAL sequences -- a genuinely paired
# comparison, not a fresh independent sample.
#
# Question: does ablating the top boundary-attending heads on a task where
# boundaries HURT shrink that harm (same circuit carries help and harm),
# or leave it unchanged (a different pathway routes the disruption)?
# Either outcome is informative; report both.

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join(REPO_ROOT, "results", MODEL_TAG, "causal_ablation_copy_previous")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, torch.nn as nn, random
import numpy as np
import pandas as pd
from tqdm import tqdm

from tasks.copy_previous.engine import build_prompt
from tasks.common import scoring as scoring_mod

print(f"Loading {MODEL_ID} into VRAM (eager attention, needed for the monkey-patch below)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
)
model.eval()

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

INTERVAL = 2
SEQUENCE_LENGTHS = [8, 16, 32, 64]
TRIALS_T = 20
TOP_K_HEADS = 10


def local_seed(N, k, t):
    return 90_000 * N + 100 * k + 10 * 0 + t  # axis_val=None -> 0, matches tasks.common.runner


def find_boundary_token_indices(input_ids_row, tokenizer):
    """Locate boundary '|' token positions within the Input:...Output: span.
    Uses the LAST 'Input'/'Output' occurrence (crop fix from
    run_anchor_attention_geometry.py): copy_previous's prompt has an
    "Example: Input 1011 -> Output 1101" line before the real one."""
    tokens = tokenizer.convert_ids_to_tokens(input_ids_row.tolist())
    input_idx = max(i for i, t in enumerate(tokens) if 'Input' in t)
    output_idx = next(i for i in range(input_idx, len(tokens)) if 'Output' in tokens[i])
    return [i for i in range(input_idx, output_idx) if tokens[i] == '|']


# ============================================================
# STEP 1: fresh diagnostic pass to identify top boundary-attending heads
# ============================================================
print("\n--- Diagnostic pass: identifying top boundary-attending heads (copy_previous) ---")
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
mass_accum = np.zeros((n_layers, n_heads))
n_diag_trials = 5

for t in range(n_diag_trials):
    rng = random.Random(local_seed(16, INTERVAL, t))
    _, _, prompt = build_prompt(16, INTERVAL, rng)
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
flat_idx = np.argsort(mass_accum.ravel())[::-1][:TOP_K_HEADS]
TARGET_HEADS = [(int(i // n_heads), int(i % n_heads)) for i in flat_idx]
print(f"Top {TOP_K_HEADS} boundary-attending (layer, head) pairs: {TARGET_HEADS}")
print(f"Their boundary-attention mass: {[round(mass_accum[l, h], 4) for l, h in TARGET_HEADS]}")

pd.DataFrame(TARGET_HEADS, columns=["Layer", "Head"]).assign(
    Mass=[mass_accum[l, h] for l, h in TARGET_HEADS]
).to_csv(os.path.join(RESULTS_DIR, "causal_ablation_copy_previous_target_heads.csv"), index=False)

CYCLIC_SHIFT_CORE = {(1, 24), (3, 24), (15, 5)}
overlap = set(TARGET_HEADS) & CYCLIC_SHIFT_CORE
print(f"\nOverlap with cyclic_shift's persistent 3-head core {sorted(CYCLIC_SHIFT_CORE)}: {sorted(overlap)} "
      f"({len(overlap)}/3)")

TARGET_SET = set(TARGET_HEADS)
TARGET_BY_LAYER = {}
for l, h in TARGET_HEADS:
    TARGET_BY_LAYER.setdefault(l, []).append(h)

# ============================================================
# STEP 2: monkey-patch eager attention (identical mechanism to
# scripts/run_causal_ablation.py -- zero + renormalize target heads'
# attention to boundary positions)
# ============================================================
import transformers.models.qwen2.modeling_qwen2 as qwen2_mod

_ORIGINAL_EAGER_ATTENTION_FORWARD = qwen2_mod.eager_attention_forward
CURRENT_BOUNDARY_IDX = []
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
# STEP 3: run the ablated Cellular_Bounds condition (Base + normal
# Cellular_Bounds are read from the existing pooled 40-trial grid,
# same seed formula, so this is a paired comparison)
# ============================================================
def run_ablated_trial(N, t):
    global CURRENT_BOUNDARY_IDX, ABLATION_ACTIVE
    rng = random.Random(local_seed(N, INTERVAL, t))
    inp, truth, prompt = build_prompt(N, INTERVAL, rng)

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
            pad_token_id=tokenizer.eos_token_id,
        )
    ABLATION_ACTIVE = False

    gen_only = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(gen_only, skip_special_tokens=True)
    result = scoring_mod.score(raw_output, truth)

    return {
        "Method": "Cellular_Bounds_Ablated", "Interval": INTERVAL,
        "Seq_Length": N, "Trial": t, "TAR": result["TAR"], "SAR": result["SAR"],
        "Input_Given": inp, "Ground_Truth": result["clean_truth"], "LLM_Output": result["clean_pred"],
        "N_Target_Heads": len(TARGET_HEADS),
    }


tasks = [(N, t) for N in SEQUENCE_LENGTHS for t in range(TRIALS_T)]
print(f"\nStarting causal ablation sweep (copy_previous): {len(tasks)} evaluations...")

results = []
for N, t in tqdm(tasks, desc="Ablated Cellular_Bounds (copy_previous)"):
    try:
        results.append(run_ablated_trial(N, t))
    except Exception as exc:
        print(f"Task failed: N={N}, t={t} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "causal_ablation_copy_previous_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

qwen2_mod.eager_attention_forward = _ORIGINAL_EAGER_ATTENTION_FORWARD
print("Restored original attention function.")

summary = df.groupby("Seq_Length")[["TAR", "SAR"]].mean()
print("\n--- ABLATED Cellular_Bounds TAR/SAR by length (copy_previous) ---")
print(summary.round(4).to_string())

# --- Compare against the existing paired Base / normal-Bounds reference ---
ref_path = os.path.join(REPO_ROOT, "results_clean", "copy_previous", "grid",
                         "Qwen2.5-7B-Instruct_POOLED_40trials.csv")
ref_df = pd.read_csv(ref_path)
ref_df = ref_df[ref_df.Trial < TRIALS_T]  # first 20 trials only, matched to the ablated run
ref_base = ref_df[ref_df.Method == "Base"].groupby("Seq_Length")["TAR"].mean()
ref_cellular = ref_df[(ref_df.Method == "Cellular_Bounds") & (ref_df.Interval == INTERVAL)].groupby("Seq_Length")["TAR"].mean()

comparison = pd.DataFrame({
    "Base_TAR": ref_base, "Cellular_Normal_TAR": ref_cellular, "Cellular_Ablated_TAR": summary["TAR"],
})
comparison["Harm_Normal"] = comparison["Cellular_Normal_TAR"] - comparison["Base_TAR"]
comparison["Harm_Ablated"] = comparison["Cellular_Ablated_TAR"] - comparison["Base_TAR"]
comparison["Pct_Harm_Mitigated"] = 100 * (1 - comparison["Harm_Ablated"] / comparison["Harm_Normal"])
print("\n--- DOES ABLATING TOP BOUNDARY-ATTENDING HEADS MITIGATE THE HARM ON copy_previous? ---")
print(comparison.round(4).to_string())
comparison.to_csv(os.path.join(RESULTS_DIR, "causal_ablation_copy_previous_comparison.csv"))

print("\nDone.")
