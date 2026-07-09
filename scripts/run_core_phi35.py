import matplotlib
matplotlib.use("Agg")

# ============================================================
# 4TH MODEL FAMILY: Phi-3.5-mini-instruct
# ============================================================
# Reviewer ask: does the headline effect (Fig 1: boundaries reduce alpha,
# robustly, for alphabet=2, across model scale/architecture) hold on a
# fourth, genuinely different architecture family? Llama-3.1-8B-Instruct
# and Gemma-2-9b-it were both tried first (the two most standard choices)
# and both are gated on HF with no access granted to this token yet
# (GatedRepoError on the actual weights, even though model_info() itself
# succeeds). Phi-3.5-mini-instruct is ungated and accessible immediately,
# so it's used here to keep moving; swap in Llama once access clears if
# that specific family matters more than not waiting.
#
# Identical grid, seeds, and task logic to run_core_templated_v2.py (the
# same script already run on Qwen2.5-7B/14B and Mistral-7B) -- only
# MODEL_ID changes -- so results slot directly into the existing Fig 1 /
# Fig 1B cross-model comparisons without any reconciliation.

import os

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"Results directory: {RESULTS_DIR}/")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random
import numpy as np

print(f"Loading {MODEL_ID} into VRAM...")
# trust_remote_code=True is deliberately omitted: it would load the model
# repo's own bundled modeling_phi3.py, whose Cache-object usage
# ('DynamicCache.seen_tokens') predates this transformers version's Cache
# API and crashes on every generate() call. Since model_type=='phi3' is
# natively supported in transformers, omitting the flag falls back to the
# library's own maintained Phi3ForCausalLM implementation instead.
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
from scipy.optimize import curve_fit
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
SEQUENCE_LENGTHS = [4, 8, 16, 32, 64, 128, 256]
PERIODIC_INTERVALS = [0, 1, 2, 3, 4, 8, 16]
API_TEMPERATURES = [0.0]
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
print(f"Pregenerated {len(shared_sequences)} shared raw sequences (same seed formula as the other 3 models).")


def run_single_trial(task_params):
    T_api, A, N, k, t = task_params
    engine = engines[A]
    method = "Base" if k == 0 else "Cellular_Bounds"
    raw_input = shared_sequences[(A, N, t)]

    if method == "Base":
        inp, truth, prompt = engine.build_base_from_raw(raw_input)
    else:
        inp, truth, prompt = engine.build_cellular_from_raw(raw_input, interval=k)

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = {kk: vv.to(model.device) for kk, vv in enc.items()}

    max_tokens_needed = int(N * 1.5) + 20
    do_sample = T_api > 0.0
    temp_val = T_api if do_sample else None

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, temperature=temp_val,
            do_sample=do_sample, pad_token_id=tokenizer.eos_token_id
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
    first_error = binary_series.index(0) if 0 in binary_series else len(binary_series)

    return {
        "API_Temp": T_api, "Alphabet_Size": A, "Method": method, "Interval": k,
        "Seq_Length": N, "Trial": t, "Shared_Sequence": "".join(raw_input),
        "TAR": tar, "SAR": sar, "First_Error_Pos": first_error,
        "Input_Given": inp, "Ground_Truth": clean_truth, "LLM_Output": clean_pred,
    }


tasks = list(itertools.product(API_TEMPERATURES, ALPHABET_SIZES, SEQUENCE_LENGTHS, PERIODIC_INTERVALS, range(TRIALS_T)))
filtered_tasks = [task for task in tasks if task[3] == 0 or task[3] < task[2]]
print(f"\nStarting Phi-3.5-mini core grid: {len(filtered_tasks)} evaluations.")

all_results = []
for task in tqdm(filtered_tasks, total=len(filtered_tasks), desc="Generating (Phi-3.5-mini)"):
    try:
        all_results.append(run_single_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

df = pd.DataFrame(all_results)
csv_path = os.path.join(RESULTS_DIR, "thermodynamics_grid_templated_v2.csv")
df.to_csv(csv_path, index=False)
print(f"\nExperiment complete. Saved to '{csv_path}'")

summary = df.groupby(["Alphabet_Size", "Seq_Length", "Method"])[["TAR", "SAR"]].mean().reset_index()
print("\n--- AVERAGE ACCURACY BY ALPHABET & LENGTH ---")
print(summary.to_string(index=False))


# ==== SK FIT (same as run_core_templated_v2.py) ====

def sk_model(N, beta0, alpha):
    N = np.asarray(N, dtype=np.float64)
    return np.exp(-beta0 * N * np.power(alpha, N - 1))

def fit_sk(N_vals, sar_vals):
    N_vals = np.asarray(N_vals, dtype=np.float64)
    sar_vals = np.asarray(sar_vals, dtype=np.float64)
    n_distinct = len(set(np.round(sar_vals, 6)))
    if n_distinct < 2 or np.all(sar_vals == 0):
        return np.nan, np.nan, np.nan, False
    try:
        popt, _ = curve_fit(sk_model, N_vals, sar_vals, p0=[0.05, 1.2],
                             bounds=([1e-6, 1.0 + 1e-6], [1.0, 5.0]), maxfev=20000)
        beta0, alpha = popt
        n_star = 1 + np.log(1.0 / beta0) / np.log(alpha) if alpha > 1 else np.inf
        return beta0, alpha, n_star, True
    except RuntimeError:
        return np.nan, np.nan, np.nan, False

sk_results = []
for grp in sorted(df['Alphabet_Size'].unique()):
    sub_G = df[df['Alphabet_Size'] == grp]
    for method, sub_M in sub_G.groupby('Method'):
        for interval, sub_I in sub_M.groupby('Interval'):
            per_N = sub_I.groupby('Seq_Length')['SAR'].mean().reset_index()
            if len(per_N) < 3:
                continue
            beta0, alpha, n_star, fit_valid = fit_sk(per_N['Seq_Length'], per_N['SAR'])
            sk_results.append({
                'Alphabet_Size': grp, 'Method': method, 'Interval': interval,
                'Beta0': beta0, 'Alpha': alpha, 'N_star': n_star,
                'N_points': len(per_N), 'Fit_Valid': fit_valid,
            })

sk_df = pd.DataFrame(sk_results)
sk_df.to_csv(os.path.join(RESULTS_DIR, "thermodynamics_sk_fit_templated_v2.csv"), index=False)
print("\n--- SK FIT RESULTS ---")
print(sk_df.round(4).to_string(index=False))

print("\nDone.")
