import matplotlib
matplotlib.use("Agg")

# ============================================================
# IS THE A=8, N=8 DIP AT 14B REAL, OR JUST THIS PARTICULAR SAMPLE?
# ============================================================
# The templated core rerun showed Alphabet=8, N=8, 14B dipping at k=4
# (Base=0.700, k=4=0.675, k=2=0.8125) -- non-monotonic, unlike 7B raw and
# 7B templated at the same condition (both monotonic, no dip). With only
# 3 samples total (7B raw, 7B templated, 14B templated), 2-clean-1-dip is
# not strong evidence either way. This reruns the EXACT same condition on
# 14B, chat-templated, with a COMPLETELY DIFFERENT random seed (fresh
# sequences, not the same 20 as before) and DOUBLE the trial count (40
# instead of 20) for a more decisive read. If the dip disappears with a
# fresh draw, that's real evidence it was noise. If it persists, that
# points to something more systematic in 14B at this specific condition.

import os

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random
import numpy as np

print(f"Loading {MODEL_ID} into VRAM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager",
)
model.eval()

# Deliberately a DIFFERENT global seed from the main pipeline's 42, so this
# draws genuinely different raw sequences, not a re-run of the same ones.
GLOBAL_SEED = 9001
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

    def build_base_from_raw(self, raw_input):
        input_str = "".join(raw_input)
        output_str = "".join(
            self.alphabet[(self.char_to_idx[c] + 1) % self.alphabet_size]
            for c in raw_input
        )
        prompt = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(self.alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {self.alphabet[-1]} -> {self.alphabet[0]}, and {self.alphabet[0]} -> {self.alphabet[1]}.
Constraint: Output ONLY the transformed string.
No spaces, no explanations.

Input: {input_str}
Output: """
        return input_str, output_str, prompt

    def build_cellular_from_raw(self, raw_input, interval):
        input_w_bounds = ""
        output_w_bounds = ""
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


A = 8
N = 8
INTERVALS = [0, 2, 4]
TRIALS_T = 40   # double the original, for a more decisive read

engine = TaskEngine(A)

shared_sequences = {}
for t in range(TRIALS_T):
    local_seed = 777_000 + 100 * N + t   # fresh seed space, unrelated to the main pipeline's
    rng = random.Random(local_seed)
    shared_sequences[t] = engine.generate_raw_input(N, rng=rng)
print(f"Pregenerated {len(shared_sequences)} fresh shared raw sequences.")


def run_single_trial(k, t):
    method = "Base" if k == 0 else "Cellular_Bounds"
    raw_input = shared_sequences[t]

    if method == "Base":
        inp, truth, prompt = engine.build_base_from_raw(raw_input)
    else:
        inp, truth, prompt = engine.build_cellular_from_raw(raw_input, interval=k)

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)

    max_tokens_needed = int(N * 1.5) + 20
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_only = outputs[0][inputs["input_ids"].shape[1]:]
    pred = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
    pred = pred.split("Explanation")[0].split("\n")[0].strip()

    if method == "Cellular_Bounds":
        clean_pred = pred.replace("|", "").replace(" ", "").replace("\n", "")
        clean_truth = truth.replace("|", "")
    else:
        clean_pred = pred.replace(" ", "").replace("\n", "")
        clean_truth = truth
    clean_pred = clean_pred[:len(clean_truth)]

    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0
                      for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if len(clean_truth) > 0 else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0

    return {"Method": method, "Interval": k, "Trial": t, "TAR": tar, "SAR": sar}


tasks = [(k, t) for k in INTERVALS for t in range(TRIALS_T)]
print(f"\nStarting fresh-seed check: {len(tasks)} evaluations...")

results = []
for k, t in tqdm(tasks, desc="Fresh-seed A=8,N=8"):
    try:
        results.append(run_single_trial(k, t))
    except Exception as exc:
        print(f"Task failed: k={k}, t={t} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = "/tmp/a8n8_freshseed_check.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved to {csv_path}")

summary = df.groupby(['Method', 'Interval'])[['TAR', 'SAR']].agg(['mean', 'std', 'count'])
print("\n--- FRESH-SEED RESULT: Alphabet=8, N=8, 14B templated, T=40 ---")
print(summary.to_string())

pivot = df.groupby('Interval')['TAR'].mean()
print(f"\nk=0 (Base): {pivot.get(0, float('nan')):.4f}")
print(f"k=4:        {pivot.get(4, float('nan')):.4f}")
print(f"k=2:        {pivot.get(2, float('nan')):.4f}")
if 0 in pivot.index and 4 in pivot.index and 2 in pivot.index:
    if pivot[0] <= pivot[4] <= pivot[2]:
        print("\n=> MONOTONIC (k=4 sits between k=0 and k=2) -- the original dip did NOT reproduce with fresh trials.")
    else:
        print("\n=> NON-MONOTONIC (dip reproduces) -- this may be a real, systematic effect, not just noise.")

print("\nFresh-seed check complete.")
