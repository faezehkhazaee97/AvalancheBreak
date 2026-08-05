import matplotlib
matplotlib.use("Agg")

# ============================================================
# MECHANISTIC EXPLANATION FOR THE k=2 "DIP" ON 1-BACK DEPENDENCY TASKS
# ============================================================
# Qwen2.5-32B-Instruct, copy_previous (output[i] = input[i-1], binary
# alphabet), REAL prompt format -- concatenated digits, no space-separation
# workaround (binary digits already tokenize 1:1 on Qwen, verified in the
# alphabet tokenization audit, so this needs no workaround at all, unlike
# the letter-alphabet Tier-1 captures).
#
# N=32, where the empirical dip is large and clean on Qwen-32B:
#   Base=0.83  k=1=0.88  k=2=0.65 (worst)  k=8=0.84 (recovered)
#
# Method: teacher-forced forward pass over the FULL sequence (prompt +
# ground-truth output appended), output_attentions=True, so we can look at
# attention FROM each output-region query position back to the INPUT
# region -- i.e., while "generating" output position i, does the model
# actually attend to the correct target input[i-1], or does it get
# diverted (e.g. onto the boundary token, or onto the wrong offset)?
# This is a direct, position-resolved test of the "regularity" hypothesis:
# at k=1 the correct target is always exactly 2 tokens back (skip one
# bar) -- a fixed, learnable offset. At k=2, the correct target is
# SOMETIMES 1 token back (no bar in between) and SOMETIMES 3 tokens back
# (one bar in between), depending on parity within the chunk -- an
# irregular offset. If that's the mechanism, attention-to-correct-target
# should be high and consistent at k=1, degraded/inconsistent at k=2, and
# recovered at k=8 (where most positions have no boundary in their
# lookback at all).

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join(REPO_ROOT, "results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random
import numpy as np
import pandas as pd

print(f"Loading {MODEL_ID} into VRAM (eager attention)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
)
model.eval()

num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads
print(f"num_layers={num_layers} num_heads={num_heads}")

ATTENTION_SEED = 123


def compute_shift(binary_str):
    out = [binary_str[0]]
    for i in range(1, len(binary_str)):
        out.append(binary_str[i - 1])
    return "".join(out)


def insert_boundaries(seq, interval):
    if interval <= 0:
        return seq
    out = []
    for i, ch in enumerate(seq):
        out.append(ch)
        if (i + 1) % interval == 0 and i != len(seq) - 1:
            out.append("|")
    return "".join(out)


def build_prompt(input_bounded):
    return f"""You are a deterministic sequence transformer.
Task: Copy the previous input digit.
Alphabet: 0, 1 (binary digits)
Rule: At each position, output the PREVIOUS input digit (not the current one).
The first digit is copied as-is (no previous input to copy).
Each subsequent digit: output[i] = input[i-1]
Example: Input 1011 -> Output 1101 (calculation: 1 (copied), then 1, 0, 1 -- each shifted from the position before)
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """


N_LENGTH = 32
K_VALUES = [1, 2, 8]  # best, dip, recovered -- per the empirical Qwen-32B N=32 table
N_TRIALS = 8


def capture_one(input_base, k, tokenizer, model):
    output_base = compute_shift(input_base)
    input_bounded = insert_boundaries(input_base, k)
    output_bounded = insert_boundaries(output_base, k)
    prompt = build_prompt(input_bounded)

    # Teacher-forced: append the correct answer so the output region has
    # real token positions to inspect attention FROM.
    full_text_for_capture = prompt + output_bounded

    enc_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    prompt_len = enc_prompt["input_ids"].shape[1]

    full_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )["input_ids"]
    answer_ids = tokenizer(output_bounded, add_special_tokens=False, return_tensors="pt")["input_ids"]
    full_ids = torch.cat([full_ids, answer_ids], dim=1)
    attn_mask = torch.ones_like(full_ids)

    inputs = {"input_ids": full_ids.to(model.device), "attention_mask": attn_mask.to(model.device)}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    tokens = [tokenizer.decode([t]) for t in full_ids[0]]

    # locate "Input:" digit span and the appended answer span -- use the
    # LAST "Input"/"Output" occurrence, not the first: the prompt's own
    # worked example ("Example: Input 1011 -> Output 1101...") contains
    # both words earlier in the text, before the real "Input: {data}" line.
    input_marker = max(i for i, t in enumerate(tokens[:prompt_len]) if 'Input' in t)
    output_marker = max(i for i in range(input_marker, prompt_len) if 'Output' in tokens[i])
    input_start = input_marker + 1
    while input_start < output_marker and tokens[input_start].strip() in [':', '']:
        input_start += 1
    input_end = output_marker
    while input_end > input_start and tokens[input_end - 1].strip() == '':
        input_end -= 1
    input_span_tokens = [t.strip() for t in tokens[input_start:input_end]]

    answer_start = prompt_len
    answer_span_tokens = [t.strip() for t in tokens[answer_start:]]

    return outputs.attentions, tokens, input_start, input_end, input_span_tokens, answer_start, answer_span_tokens, input_bounded, output_bounded


def analyze_retrieval(attentions, input_start, input_end, input_span_tokens, answer_start, answer_span_tokens, input_bounded):
    """For each answer-region query position i (corresponding to output
    char i, i>=1), find the fraction of attention mass (summed/averaged
    over ALL heads at each layer) directed at the CORRECT target
    (input position i-1, mapped through the boundary-inserted string) vs.
    at boundary tokens vs. elsewhere. Returns per-layer arrays."""
    # map raw (unbounded) output index -> token index within input_span_tokens
    # input_bounded has the same boundary layout as answer tokens (both built
    # with insert_boundaries at the same k), so char-for-char alignment holds
    # between input_span_tokens and answer_span_tokens by construction.
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    correct_mass = np.zeros((n_layers, n_heads))
    boundary_mass = np.zeros((n_layers, n_heads))
    n_scored = 0

    for i, ch in enumerate(input_span_tokens):
        if ch == '|':
            continue
        # this is raw input digit at "raw position" j (0-indexed among digits only)
        pass

    # Build raw-digit-index -> token-index maps for both spans (skip '|').
    input_raw_idx_to_tok = [idx for idx, t in enumerate(input_span_tokens) if t != '|']
    answer_raw_idx_to_tok = [idx for idx, t in enumerate(answer_span_tokens) if t != '|']

    for raw_i in range(1, len(answer_raw_idx_to_tok)):  # skip position 0 (no dependency)
        ans_tok_idx = answer_raw_idx_to_tok[raw_i]
        query_pos = answer_start + ans_tok_idx
        target_raw_j = raw_i - 1
        target_tok_idx = input_raw_idx_to_tok[target_raw_j]
        target_pos = input_start + target_tok_idx

        boundary_positions = [input_start + idx for idx, t in enumerate(input_span_tokens) if t == '|']

        for l in range(n_layers):
            layer_attn = attentions[l][0]  # (heads, seq, seq)
            for h in range(n_heads):
                row = layer_attn[h, query_pos].to(torch.float32).cpu().numpy()
                correct_mass[l, h] += row[target_pos]
                if boundary_positions:
                    boundary_mass[l, h] += row[boundary_positions].sum()
        n_scored += 1

    if n_scored > 0:
        correct_mass /= n_scored
        boundary_mass /= n_scored
    return correct_mass, boundary_mass, n_scored


all_rows = []
per_k_tensors = {}

for k in K_VALUES:
    print(f"\n=== k={k} ===")
    correct_accum = None
    boundary_accum = None
    trials_done = 0
    for t in range(N_TRIALS):
        rng = random.Random(1000 * k + t + ATTENTION_SEED)
        input_base = "".join(str(rng.choice([0, 1])) for _ in range(N_LENGTH))
        (attentions, tokens, input_start, input_end, input_span_tokens,
         answer_start, answer_span_tokens, input_bounded, output_bounded) = capture_one(input_base, k, tokenizer, model)

        correct_mass, boundary_mass, n_scored = analyze_retrieval(
            attentions, input_start, input_end, input_span_tokens, answer_start, answer_span_tokens, input_bounded
        )
        if correct_accum is None:
            correct_accum = correct_mass
            boundary_accum = boundary_mass
        else:
            correct_accum += correct_mass
            boundary_accum += boundary_mass
        trials_done += 1
        print(f"  trial {t}: n_scored={n_scored}, max correct-target attn (best head)={correct_mass.max():.4f}")

    correct_accum /= trials_done
    boundary_accum /= trials_done
    per_k_tensors[k] = {"correct": correct_accum, "boundary": boundary_accum}

    best_layer, best_head = np.unravel_index(np.argmax(correct_accum), correct_accum.shape)
    print(f"  k={k}: best retrieval head = L{best_layer}/H{best_head}, "
          f"mean attn-to-correct-target = {correct_accum[best_layer, best_head]:.4f}")

    for l in range(correct_accum.shape[0]):
        for h in range(correct_accum.shape[1]):
            all_rows.append({
                "k": k, "layer": l, "head": h,
                "correct_target_attn": correct_accum[l, h],
                "boundary_attn": boundary_accum[l, h],
            })

df = pd.DataFrame(all_rows)
out_csv = os.path.join(RESULTS_DIR, "dip_explanation_retrieval_attn.csv")
df.to_csv(out_csv, index=False)
print(f"\nSaved {out_csv}")

np.savez_compressed(
    os.path.join(RESULTS_DIR, "dip_explanation_tensors.npz"),
    **{f"correct_k{k}": v["correct"] for k, v in per_k_tensors.items()},
    **{f"boundary_k{k}": v["boundary"] for k, v in per_k_tensors.items()},
)

print("\n=== SUMMARY: mean attention-to-correct-target, top-10 heads averaged, by k ===")
for k in K_VALUES:
    correct = per_k_tensors[k]["correct"]
    top10 = np.sort(correct.flatten())[-10:]
    print(f"  k={k}: top-10-head mean = {top10.mean():.4f}")
