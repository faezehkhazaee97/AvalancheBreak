import matplotlib
matplotlib.use("Agg")

# ============================================================
# DRAFT-PAPER REPLICATION DIAGNOSTIC
# ============================================================
# The first draft (Keep_LLMs_Focused.pdf, Appendix B / Table 2) reported much
# cleaner R^2 values for TAR-vs-boundary-frequency on Qwen2.5-7B-Instruct
# than the current notebook's thermodynamics_regression.csv (e.g. A=4,
# N=16: 0.930 in the draft vs 0.421 now; A=4, N=32: 0.799 vs 0.409 now).
#
# The draft's Appendix A prompt template differs from the current
# notebook's in several ways, and the draft's own worked example is
# SPACE-SEPARATED per character ("A B C A | D A B C"), while the current
# notebook concatenates characters with no spaces ("ABCA|DABC"). Spacing
# changes tokenization: single space-separated letters are guaranteed to
# be their own token, while concatenated runs can get merged into
# different BPE tokens, plausibly making the task easier and less noisy.
#
# This script isolates that variable directly: same simplified draft
# prompt wording, same T=10 trial count as stated in the draft, at BOTH
# spacing variants, scored exactly as the draft describes (transformed
# alphabetic content only, boundaries as scaffolding, not scored). If the
# spaced variant reproduces something close to the draft's R^2 values and
# the concatenated variant reproduces something close to the current
# notebook's, that confirms tokenization/spacing (not any of the bug
# fixes made along the way) as the main driver of the discrepancy.

import os

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"Results directory: {RESULTS_DIR}/")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random
import numpy as np

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

import pandas as pd
import itertools
from scipy.stats import linregress
from tqdm import tqdm


ALPHABET_SIZES = [2, 4, 8]
SEQUENCE_LENGTHS = [4, 8, 16, 32]
PERIODIC_INTERVALS = [0, 2, 4, 8]
TRIALS_T = 10   # matches the draft's stated trial count, not the current notebook's 20


def make_alphabet(a):
    return [chr(ord('A') + i) for i in range(a)]


def build_prompt(alphabet, input_tokens, spaced):
    # Draft's Appendix A template, verbatim wording (no "Mapping:" line, no
    # "no spaces, no explanations" instruction -- just what the draft shows).
    joiner = " " if spaced else ""
    input_str = joiner.join(input_tokens)
    return f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(alphabet)}.
Rule: Replace every character with the next one in the alphabet.
Constraint: Vertical bars (|) are structural boundaries. Do not shift them. Copy them exactly.
Output only the transformed sequence.
Input: {input_str}
Output: """


def run_trial(a, n, k, t, spaced):
    alphabet = make_alphabet(a)
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    local_seed = 500_000 * a + 10_000 * n + 100 * k + t
    rng = random.Random(local_seed)
    raw = rng.choices(alphabet, k=n)

    tokens = []
    truth_tokens = []
    for i, ch in enumerate(raw):
        tokens.append(ch)
        truth_tokens.append(alphabet[(char_to_idx[ch] + 1) % a])
        if k > 0 and (i + 1) % k == 0 and i != n - 1:
            tokens.append("|")
            truth_tokens.append("|")

    prompt = build_prompt(alphabet, tokens, spaced)
    truth_full = ("".join(truth_tokens)) if not spaced else " ".join(truth_tokens)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    max_tokens_needed = int(n * (2.0 if spaced else 1.5)) + 20
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = full_response[len(prompt):].strip()
    pred = pred.split("Explanation")[0].split("\n")[0].strip()

    # Score transformed alphabetic content only -- strip delimiters AND
    # (if spaced) strip spaces from both sides before comparing, matching
    # the draft's stated "boundaries as scaffolding, not scored" metric.
    clean_pred = pred.replace("|", "").replace(" ", "").replace("\n", "")
    clean_truth = "".join(truth_tokens)  # delimiters+letters, no spaces, for stripping
    clean_truth = clean_truth.replace("|", "")
    clean_pred = clean_pred[:len(clean_truth)]

    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0
                      for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    return {
        "Spacing": "spaced" if spaced else "concatenated",
        "Alphabet_Size": a, "Seq_Length": n, "Interval": k, "Trial": t,
        "TAR": tar,
    }


tasks = [
    (a, n, k, t, spaced)
    for a in ALPHABET_SIZES
    for n in SEQUENCE_LENGTHS
    for k in PERIODIC_INTERVALS
    for t in range(TRIALS_T)
    for spaced in [True, False]
    if k == 0 or k < n
]
print(f"\nStarting draft-replication diagnostic: {len(tasks)} evaluations "
      f"(2 spacing variants x {len(ALPHABET_SIZES)} alphabets x {len(SEQUENCE_LENGTHS)} lengths x "
      f"~{len(PERIODIC_INTERVALS)} intervals x {TRIALS_T} trials)...")

results = []
for (a, n, k, t, spaced) in tqdm(tasks, desc="Draft replication"):
    try:
        results.append(run_trial(a, n, k, t, spaced))
    except Exception as exc:
        print(f"Task failed: a={a}, n={n}, k={k}, t={t}, spaced={spaced} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "draft_replication_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nDiagnostic complete. Saved to '{csv_path}'")

df['Inv_K'] = df['Interval'].apply(lambda k: 0.0 if k == 0 else 1.0 / k)

print("\n--- R^2 (TAR vs 1/k) BY SPACING VARIANT, ALPHABET, LENGTH ---")
print("(compare against the draft's Table 2: A=2/4/8 x N=8/16/32)")
rows = []
for spacing, sub_S in df.groupby('Spacing'):
    for a, sub_A in sub_S.groupby('Alphabet_Size'):
        for n, sub_N in sub_A.groupby('Seq_Length'):
            per_k = sub_N.groupby('Inv_K')['TAR'].mean().reset_index()
            if len(per_k) >= 2:
                slope, intercept, r, p, se = linregress(per_k['Inv_K'], per_k['TAR'])
                rows.append(dict(Spacing=spacing, Alphabet_Size=a, Seq_Length=n,
                                  R_squared=r**2, N_points=len(per_k)))
result_df = pd.DataFrame(rows)
print(result_df.round(4).to_string(index=False))
result_df.to_csv(os.path.join(RESULTS_DIR, "draft_replication_r2.csv"), index=False)

print("\n--- DRAFT'S TABLE 2 (for direct comparison) ---")
print("A=2: N=8:0.998 N=16:0.979 N=32:0.929")
print("A=4: N=8:0.717 N=16:0.930 N=32:0.799")
print("A=8: N=8:0.855 N=16:0.775 N=32:0.676")

print("\nDraft-replication diagnostic complete.")
