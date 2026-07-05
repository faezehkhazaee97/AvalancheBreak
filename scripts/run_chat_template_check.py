import matplotlib
matplotlib.use("Agg")

# ============================================================
# CHAT-TEMPLATE COMPARISON
# ============================================================
# The notebook (both the original draft and the current version) has
# always fed Qwen2.5-7B-Instruct raw, unwrapped prompt text via
# tokenizer(prompt, return_tensors="pt") -- never through the model's own
# chat template (tokenizer.apply_chat_template), even though this is an
# "Instruct" model specifically fine-tuned to expect
# <|im_start|>user...<|im_end|><|im_start|>assistant\n formatting.
#
# This script runs the exact same grid two ways -- "raw" (today's actual
# method) and "templated" (wrapped in the proper chat template) -- on
# identical prompts/seeds, to test directly: does templating change
# accuracy, and does it produce cleaner (higher R^2, less noisy) scaling
# trends? Same grid as the earlier reproducibility check: Alphabet
# {2,4,8}, Seq_Length {4,8,16,32}, Interval {0,2,4,8}, T=20.

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


ALPHABET_SIZES = [2, 4, 8]
SEQUENCE_LENGTHS = [4, 8, 16, 32]
PERIODIC_INTERVALS = [0, 2, 4, 8]
TRIALS_T = 20

engines = {A: TaskEngine(A) for A in ALPHABET_SIZES}

shared_sequences = {}
for A in ALPHABET_SIZES:
    engine = engines[A]
    for N in SEQUENCE_LENGTHS:
        for t in range(TRIALS_T):
            local_seed = 10_000 * A + 100 * N + t
            rng = random.Random(local_seed)
            shared_sequences[(A, N, t)] = engine.generate_raw_input(N, rng=rng)
print(f"Pregenerated {len(shared_sequences)} shared raw sequences (same seed formula as before).")


def run_single_trial(task_params, use_template):
    A, N, k, t = task_params
    engine = engines[A]
    method = "Base" if k == 0 else "Cellular_Bounds"
    raw_input = shared_sequences[(A, N, t)]

    if method == "Base":
        inp, truth, prompt = engine.build_base_from_raw(raw_input)
    else:
        inp, truth, prompt = engine.build_cellular_from_raw(raw_input, interval=k)

    if use_template:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(model.device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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

    return {
        "Prompt_Style": "templated" if use_template else "raw",
        "Alphabet_Size": A, "Method": method, "Interval": k, "Seq_Length": N, "Trial": t,
        "TAR": tar, "SAR": sar,
    }


tasks = list(itertools.product(ALPHABET_SIZES, SEQUENCE_LENGTHS, PERIODIC_INTERVALS, range(TRIALS_T)))
filtered_tasks = [task for task in tasks if task[2] == 0 or task[2] < task[1]]
print(f"\nStarting chat-template comparison: {len(filtered_tasks)} evaluations x 2 prompt styles "
      f"= {2*len(filtered_tasks)} total...")

results = []
for use_template in [False, True]:
    label = "templated" if use_template else "raw"
    for task in tqdm(filtered_tasks, desc=f"Prompt style: {label}"):
        try:
            results.append(run_single_trial(task, use_template))
        except Exception as exc:
            print(f"Task failed: {task}, template={use_template} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "chat_template_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nComparison complete. Saved to '{csv_path}'")

print("\n--- OVERALL ACCURACY: raw vs templated ---")
print(df.groupby(["Prompt_Style", "Method"])[["TAR", "SAR"]].mean().round(4).to_string())

print("\n--- ACCURACY BY (Alphabet_Size, Seq_Length), raw vs templated (TAR, Method=Base) ---")
base_df = df[df["Method"] == "Base"]
pivot = base_df.groupby(["Alphabet_Size", "Seq_Length", "Prompt_Style"])["TAR"].mean().unstack("Prompt_Style")
print(pivot.round(4).to_string())

# --- R^2 (TAR vs 1/k) comparison: raw vs templated ---
df['Inv_K'] = df['Interval'].apply(lambda k: 0.0 if k == 0 else 1.0 / k)
print("\n--- R^2 (TAR vs 1/k) BY PROMPT STYLE, ALPHABET, LENGTH ---")
rows = []
for style, sub_S in df.groupby('Prompt_Style'):
    for a, sub_A in sub_S.groupby('Alphabet_Size'):
        for n, sub_N in sub_A.groupby('Seq_Length'):
            per_k = sub_N.groupby('Inv_K')['TAR'].mean().reset_index()
            if len(per_k) >= 2:
                slope, intercept, r, p, se = linregress(per_k['Inv_K'], per_k['TAR'])
                rows.append(dict(Prompt_Style=style, Alphabet_Size=a, Seq_Length=n,
                                  R_squared=r**2, N_points=len(per_k)))
result_df = pd.DataFrame(rows)
print(result_df.round(4).to_string(index=False))
result_df.to_csv(os.path.join(RESULTS_DIR, "chat_template_r2.csv"), index=False)

pivot_r2 = result_df.pivot_table(index=["Alphabet_Size", "Seq_Length"], columns="Prompt_Style", values="R_squared")
if "raw" in pivot_r2.columns and "templated" in pivot_r2.columns:
    n_templated_better = int((pivot_r2["templated"] > pivot_r2["raw"]).sum())
    print(f"\nTemplated R^2 higher than raw R^2 at {n_templated_better}/{len(pivot_r2)} (Alphabet, Length) combos.")
    print(f"Mean R^2 -- raw: {pivot_r2['raw'].mean():.4f}, templated: {pivot_r2['templated'].mean():.4f}")

print("\nChat-template comparison complete.")
