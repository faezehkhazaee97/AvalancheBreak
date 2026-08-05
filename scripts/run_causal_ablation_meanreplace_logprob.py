import matplotlib
matplotlib.use("Agg")

# ============================================================
# Q3 remainder: mean-replacement ablation + log-prob deltas
# ============================================================
# Reviewer: "Can you report log-prob changes (not only TAR) under head
# ablations, and compare post-softmax masking to mean-replacement
# ablation to address potential OOD concerns?"
#
# Two things the original causal ablation (run_causal_ablation.py) never
# measured: (1) a gentler control where target heads' attention to
# boundary positions is replaced by that row's OWN mean attention weight
# (still ~1 total, no renormalization hack) instead of zeroed -- tests
# whether it's specifically the ABSENCE of boundary attention that
# matters, or any perturbation of the normal pattern; (2) log-probability
# of the correct next token (teacher-forced, not just generate()+TAR) --
# a continuous, more sensitive measure than binary correctness.
#
# Reuses the exact same target heads already identified and saved in
# causal_ablation_target_heads.csv (not re-identified), same task/model/
# density (cyclic_shift, alphabet=2, k=2) as the original ablation, for
# direct comparability. Scored on a smaller trial count (10, not 20) since
# this is a targeted robustness check, not a full production run.

import os

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)

print(f"Model: {MODEL_ID}")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, torch.nn as nn, random
import numpy as np
import pandas as pd
from tqdm import tqdm

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


ALPHABET_SIZE = 2
INTERVAL = 2
SEQUENCE_LENGTHS = [8, 16, 32, 64]
TRIALS_T = 10

engine = TaskEngine(ALPHABET_SIZE)

shared_sequences = {}
for N in SEQUENCE_LENGTHS:
    for t in range(TRIALS_T):
        local_seed = 10_000 * ALPHABET_SIZE + 100 * N + t
        rng = random.Random(local_seed)
        shared_sequences[(N, t)] = engine.generate_raw_input(N, rng=rng)


def find_boundary_token_indices(input_ids_row, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(input_ids_row.tolist())
    input_idx = next(i for i, t in enumerate(tokens) if 'Input' in t)
    output_idx = next(i for i in range(input_idx, len(tokens)) if 'Output' in tokens[i])
    return [i for i in range(input_idx, output_idx) if tokens[i] == '|']


# Load the ALREADY-IDENTIFIED target heads (not re-identified) for direct comparability
target_heads_path = os.path.join(RESULTS_DIR, "causal_ablation_target_heads.csv")
target_df = pd.read_csv(target_heads_path)
TARGET_HEADS = list(zip(target_df.Layer.tolist(), target_df.Head.tolist()))
TARGET_BY_LAYER = {}
for l, h in TARGET_HEADS:
    TARGET_BY_LAYER.setdefault(l, []).append(h)
print(f"Loaded {len(TARGET_HEADS)} target heads from {target_heads_path}: {TARGET_HEADS}")

import transformers.models.qwen2.modeling_qwen2 as qwen2_mod

CURRENT_BOUNDARY_IDX = []
ABLATION_MODE = None  # None | "zero" | "mean"


def patched_eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = qwen2_mod.repeat_kv(key, module.num_key_value_groups)
    value_states = qwen2_mod.repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    if ABLATION_MODE is not None and CURRENT_BOUNDARY_IDX:
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx in TARGET_BY_LAYER:
            heads_here = TARGET_BY_LAYER[layer_idx]
            idx_tensor = torch.tensor(CURRENT_BOUNDARY_IDX, device=attn_weights.device, dtype=torch.long)
            k_len = attn_weights.shape[-1]
            valid_idx = idx_tensor[idx_tensor < k_len]
            if valid_idx.numel() > 0:
                if ABLATION_MODE == "zero":
                    for h in heads_here:
                        attn_weights[:, h, :, valid_idx] = 0.0
                    row_sums = attn_weights[:, heads_here, :, :].sum(dim=-1, keepdim=True)
                    attn_weights[:, heads_here, :, :] = attn_weights[:, heads_here, :, :] / row_sums.clamp_min(1e-12)
                elif ABLATION_MODE == "mean":
                    for h in heads_here:
                        # row-wise mean attention weight (over ALL key positions, this head, this
                        # query), replacing boundary-position weight with that mean instead of 0 --
                        # total row mass shifts only by (mean - original) per replaced position,
                        # no renormalization hack needed, still a well-formed near-1-summing row
                        row_mean = attn_weights[:, h, :, :].mean(dim=-1, keepdim=True)  # (batch, q_len, 1)
                        attn_weights[:, h, :, valid_idx] = row_mean.squeeze(-1).unsqueeze(-1).expand(-1, -1, valid_idx.numel())

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


qwen2_mod.eager_attention_forward = patched_eager_attention_forward
print("Monkey-patched eager_attention_forward (zero | mean | normal modes).")


def teacher_forced_logprob(prompt, truth_bars_stripped_or_kept, mode):
    """Teacher-forced mean log-prob of the TRUE output tokens (as they'd be
    generated greedily), under the given ablation mode. Returns mean
    per-token log-prob of the ground-truth continuation."""
    global CURRENT_BOUNDARY_IDX, ABLATION_MODE

    enc_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    prompt_ids = enc_prompt["input_ids"].to(model.device)
    target_ids = tokenizer(truth_bars_stripped_or_kept, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)

    full_ids = torch.cat([prompt_ids, target_ids], dim=1)
    full_mask = torch.ones_like(full_ids)

    CURRENT_BOUNDARY_IDX = find_boundary_token_indices(prompt_ids[0], tokenizer)
    ABLATION_MODE = mode

    with torch.no_grad():
        out = model(input_ids=full_ids, attention_mask=full_mask)
    ABLATION_MODE = None

    logits = out.logits[0]  # (seq_len, vocab)
    prompt_len = prompt_ids.shape[1]
    # logits at position i predict token i+1; we want logits at positions
    # [prompt_len-1 .. full_len-2] predicting target_ids[0..len-1]
    pred_logits = logits[prompt_len - 1: full_ids.shape[1] - 1]
    log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
    target_flat = target_ids[0]
    token_logprobs = log_probs[torch.arange(len(target_flat)), target_flat]
    return token_logprobs.mean().item(), token_logprobs.cpu().numpy().tolist()


results = []
for N in SEQUENCE_LENGTHS:
    for t in tqdm(range(TRIALS_T), desc=f"N={N}"):
        raw_input = shared_sequences[(N, t)]
        inp, truth, prompt = engine.build_cellular_from_raw(raw_input, interval=INTERVAL)
        # truth includes '|' at boundary positions, tokenizer will handle them as literal chars
        for mode, mode_label in [(None, "Normal"), ("zero", "Zero_Ablated"), ("mean", "Mean_Replaced")]:
            try:
                mean_lp, per_token_lp = teacher_forced_logprob(prompt, truth, mode)
            except Exception as exc:
                print(f"Failed: N={N} t={t} mode={mode_label} | {exc}")
                continue
            results.append({
                "Seq_Length": N, "Trial": t, "Mode": mode_label,
                "Mean_LogProb": mean_lp, "N_Target_Tokens": len(per_token_lp),
            })

df = pd.DataFrame(results)
out_path = os.path.join(RESULTS_DIR, "causal_ablation_meanreplace_logprob.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved {out_path}")

print("\n--- Mean log-prob of true output tokens, by mode and length ---")
summary = df.groupby(["Seq_Length", "Mode"])["Mean_LogProb"].mean().unstack()
print(summary.round(4).to_string())

print("\n--- Delta vs Normal (log-prob DROP under ablation -- more negative = more disruption) ---")
delta = summary.subtract(summary["Normal"], axis=0)
print(delta.round(4).to_string())
