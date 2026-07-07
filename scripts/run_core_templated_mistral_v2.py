import matplotlib
matplotlib.use("Agg")

# ============================================================
# CORE HEADLINE GRID, RE-RUN WITH THE PROPER CHAT TEMPLATE
# ============================================================
# The main notebook (both the original draft and every rerun since) has
# always fed Qwen2.5-7B-Instruct raw, unwrapped prompt text instead of
# wrapping it in the model's own chat template (verified via
# tokenizer.apply_chat_template) -- confirmed via a side experiment to
# measurably improve accuracy and stability, and to close most of the gap
# against the original draft paper's Table 2 numbers. This script reruns
# the SAME core cyclic-shift grid (same alphabets, lengths, intervals,
# trial count, seeds as the main notebook's thermodynamics_grid.csv) with
# ONLY the prompting method changed, and saves under distinct
# "_templated" filenames so the raw-prompt results are preserved
# unchanged for direct side-by-side comparison.

import os

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
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
print(f"Pregenerated {len(shared_sequences)} shared raw sequences (same seed formula as the main notebook).")


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
print(f"\nStarting TEMPLATED core grid search: {len(filtered_tasks)} evaluations.")

all_results = []
for task in tqdm(filtered_tasks, total=len(filtered_tasks), desc="Generating (templated)"):
    try:
        all_results.append(run_single_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

df = pd.DataFrame(all_results)
csv_path = os.path.join(RESULTS_DIR, "thermodynamics_grid_templated_v2.csv")
df.to_csv(csv_path, index=False)
print(f"\nExperiment complete. Saved to '{csv_path}'")

summary = df.groupby(["Alphabet_Size", "Seq_Length", "Method"])[["TAR", "SAR"]].mean().reset_index()
print("\n--- AVERAGE ACCURACY BY ALPHABET & LENGTH (templated) ---")
print(summary.to_string(index=False))


# ==== REGRESSION ====

df['Inv_K'] = df['Interval'].apply(lambda k: 0.0 if k == 0 else 1.0 / k)
reg_summary = (
    df.groupby(['Seq_Length', 'Alphabet_Size', 'Inv_K'])[['TAR', 'SAR']]
      .agg(['mean', 'std'])
)
reg_summary.columns = ['_'.join(c) for c in reg_summary.columns]
reg_summary = reg_summary.reset_index().sort_values(['Seq_Length', 'Alphabet_Size', 'Inv_K'])
reg_summary[['TAR_std', 'SAR_std']] = reg_summary[['TAR_std', 'SAR_std']].fillna(0.0)

regression_results = []
for metric in ['TAR', 'SAR']:
    for (N, A), sub in reg_summary.groupby(['Seq_Length', 'Alphabet_Size']):
        if len(sub) >= 2:
            slope, intercept, r, p, se = linregress(sub['Inv_K'], sub[f'{metric}_mean'])
            regression_results.append(dict(Metric=metric, Seq_Length=N, Alphabet=A,
                                             Slope=slope, **{'Intercept (Base Fit)': intercept},
                                             R_squared=r**2))
results_df = pd.DataFrame(regression_results)
results_df.to_csv(os.path.join(RESULTS_DIR, "thermodynamics_regression_templated_v2.csv"), index=False)
print("\n--- SCALING LAW REGRESSION (templated), R^2 by metric ---")
for metric in ['TAR', 'SAR']:
    print(f"\n{metric}:")
    pivot = results_df[results_df['Metric'] == metric].pivot_table(index='Alphabet', columns='Seq_Length', values='R_squared')
    print(pivot.round(4).to_string())


# ==== SK FIT ====

def sk_model(N, beta0, alpha):
    N = np.asarray(N, dtype=np.float64)
    return np.exp(-beta0 * N * np.power(alpha, N - 1))

def fit_sk(N_vals, sar_vals):
    N_vals = np.asarray(N_vals, dtype=np.float64)
    sar_vals = np.asarray(sar_vals, dtype=np.float64)
    n_distinct = len(set(np.round(sar_vals, 6)))
    is_valid = n_distinct >= 2 and not np.all(sar_vals == 0)
    if not is_valid:
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
print("\n--- SK MODEL FIT (templated) ---")
print(sk_df.round(4).to_string(index=False))

valid_sk_df = sk_df[sk_df['Fit_Valid']]
base_alpha = valid_sk_df[valid_sk_df['Method'] == 'Base'].set_index('Alphabet_Size')['Alpha']
cell_alpha_mean = valid_sk_df[valid_sk_df['Method'] == 'Cellular_Bounds'].groupby('Alphabet_Size')['Alpha'].mean()
print("\n--- ALPHA: Base vs mean(Cellular) per Alphabet (templated) ---")
for grp in sorted(set(base_alpha.index) & set(cell_alpha_mean.index)):
    b, c = base_alpha[grp], cell_alpha_mean[grp]
    pct = 100 * (b - c) / b if b else float('nan')
    print(f"Alphabet={grp}: alpha_base={b:.4f}  alpha_cellular_mean={c:.4f}  "
          f"({'-' if pct >= 0 else '+'}{abs(pct):.1f}% {'reduction' if pct >= 0 else 'increase'})")

print("\nTemplated core grid rerun complete.")
