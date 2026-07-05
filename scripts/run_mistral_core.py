import matplotlib
matplotlib.use("Agg")

# ============================================================
# MODEL SELECTION
# ============================================================
# Change MODEL_ID (and re-run the notebook top to bottom) to repeat the
# exact same analysis at a different model scale -- e.g. to check whether
# the boundary-sink / entropy effects studied below hold up as the model
# gets bigger. Outputs are namespaced under results/<MODEL_TAG>/ so runs
# at different scales don't overwrite each other and can be compared
# side by side.

import os

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"  # ~28GB in bf16 -- fits alongside a 7B run on an 80GB A100

MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"Results directory: {RESULTS_DIR}/")

# ==== CLEANUP CELL (no-op, CLEAR_PREVIOUS_RESULTS=False) ====

import os
import shutil

# Set to True only if you explicitly want to wipe THIS model's prior
# results (including previously captured attention tensors) before a
# fresh run. Only clears results/<MODEL_TAG>/ -- results for other
# model scales (see the MODEL SELECTION cell above) are untouched.
CLEAR_PREVIOUS_RESULTS = False

if CLEAR_PREVIOUS_RESULTS:
    if os.path.isdir(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"Cleared '{RESULTS_DIR}/'")
else:
    print(f"Skipping cleanup (CLEAR_PREVIOUS_RESULTS=False) -- existing {RESULTS_DIR}/ is preserved.")

# ==== SCALING: MODEL LOAD + CYCLIC-SHIFT GRID SEARCH (needed for SK fit) ====

import pandas as pd
import numpy as np
import random
import itertools
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. LOCAL MODEL SETUP ---
# MODEL_ID / MODEL_TAG / RESULTS_DIR come from the MODEL SELECTION cell above.
import os

print(f"Loading {MODEL_ID} into VRAM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Full A100 headroom available: run in native bf16 rather than 8-bit.
# (int8 quantization noise is an unwanted confound for the attention/entropy
# analysis later in this notebook, and bnb int8 is often *slower* than bf16
# on A100 anyway.)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",  # required: sdpa/flash-attn can't return attention weights,
                                   # and this notebook's Attention Analysis section needs them.
)
model.eval()

# --- 2. THE PARAMETER SPACE ---
ALPHABET_SIZES = [2, 4, 8]
SEQUENCE_LENGTHS = [4, 8, 16, 32, 64, 128, 256]
# Runtime note (measured on an A100): each generate() call costs roughly
# max_new_tokens / 34 seconds. At N=256 (~400 new tokens) that's ~12s/call;
# the full grid at this length list adds ~1.5h beyond the N<=32 baseline.
# Trim this list or TRIALS_T below for a faster pass.
PERIODIC_INTERVALS = [0, 2, 4, 8]   # 0 = Base
API_TEMPERATURES = [0.0]
TRIALS_T = 20

# Optional global seed for reproducibility of the whole experiment
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

# --- 3. TASK ENGINE ---
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

# Cache TaskEngines
engines = {A: TaskEngine(A) for A in ALPHABET_SIZES}

# --- 4. PREGENERATE SHARED RAW SEQUENCES ---
# One shared raw sequence per (Alphabet_Size, Seq_Length, Trial)
shared_sequences = {}

for A in ALPHABET_SIZES:
    engine = engines[A]
    for N in SEQUENCE_LENGTHS:
        for t in range(TRIALS_T):
            # Independent deterministic seed per condition key
            local_seed = 10_000 * A + 100 * N + t
            rng = random.Random(local_seed)
            shared_sequences[(A, N, t)] = engine.generate_raw_input(N, rng=rng)

print(f"Pregenerated {len(shared_sequences)} shared raw sequences.")

# --- 5. WORKER FUNCTION FOR A SINGLE TRIAL ---
def run_single_trial(task_params):
    T_api, A, N, k, t = task_params
    engine = engines[A]
    method = "Base" if k == 0 else "Cellular_Bounds"

    # Use the SAME raw sequence for base and all cellular variants with same (A, N, t)
    raw_input = shared_sequences[(A, N, t)]

    if method == "Base":
        inp, truth, prompt = engine.build_base_from_raw(raw_input)
    else:
        inp, truth, prompt = engine.build_cellular_from_raw(raw_input, interval=k)

    # --- LOCAL INFERENCE ---
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    max_tokens_needed = int(N * 1.5) + 20

    do_sample = T_api > 0.0
    temp_val = T_api if do_sample else None

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens_needed,
            temperature=temp_val,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = full_response[len(prompt):].strip()

    pred = pred.split("Explanation")[0].split("\n")[0].strip()

    # --- CLEANING AND METRICS ---
    if method == "Cellular_Bounds":
        clean_pred = pred.replace("|", "").replace(" ", "").replace("\n", "")
        clean_truth = truth.replace("|", "")
    else:
        clean_pred = pred.replace(" ", "").replace("\n", "")
        clean_truth = truth

    clean_pred = clean_pred[:len(clean_truth)]

    binary_series = []
    for i in range(len(clean_truth)):
        if i < len(clean_pred) and clean_pred[i] == clean_truth[i]:
            binary_series.append(1)
        else:
            binary_series.append(0)

    tar = sum(binary_series) / len(clean_truth) if len(clean_truth) > 0 else 0.0
    sar = 1.0 if (
        len(binary_series) == len(clean_truth)
        and sum(binary_series) == len(binary_series)
    ) else 0.0
    first_error = binary_series.index(0) if 0 in binary_series else len(binary_series)

    return {
        "API_Temp": T_api,
        "Alphabet_Size": A,
        "Method": method,
        "Interval": k,
        "Seq_Length": N,
        "Trial": t,
        "Shared_Sequence": "".join(raw_input),   # useful for verification
        "TAR": tar,
        "SAR": sar,
        "First_Error_Pos": first_error,
        "Input_Given": inp,
        "Ground_Truth": clean_truth,
        "LLM_Output": clean_pred
    }

# --- 6. MAIN EXPERIMENT SETUP ---
tasks = list(itertools.product(
    API_TEMPERATURES,
    ALPHABET_SIZES,
    SEQUENCE_LENGTHS,
    PERIODIC_INTERVALS,
    range(TRIALS_T)
))

filtered_tasks = [task for task in tasks if task[3] == 0 or task[3] < task[2]]

total_calls = len(filtered_tasks)
print(f"\nStarting Grid Search with 7B Model: {total_calls} total evaluations.")

all_results = []

# --- 7. SEQUENTIAL EXECUTION ---
for task in tqdm(filtered_tasks, total=total_calls, desc="Generating"):
    try:
        result = run_single_trial(task)
        all_results.append(result)
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

# --- 8. SAVE & SUMMARIZE ---
df = pd.DataFrame(all_results)

csv_filename = os.path.join(RESULTS_DIR, "thermodynamics_grid.csv")
df.to_csv(csv_filename, index=False)
print(f"\nExperiment complete. Saved to '{csv_filename}'")

summary = (
    df.groupby(["Alphabet_Size", "Seq_Length", "Method"])[["TAR", "SAR"]]
      .mean()
      .reset_index()
)

print("\n--- AVERAGE ACCURACY BY ALPHABET & LENGTH ---")
print(summary.to_string(index=False))

# --- 9. OPTIONAL VERIFICATION CHECK ---
# Confirms that for each (A, N, Trial), all intervals share the same raw sequence
verification = (
    df.groupby(["Alphabet_Size", "Seq_Length", "Trial"])["Shared_Sequence"]
      .nunique()
      .reset_index(name="Unique_Shared_Sequences")
)

print("\n--- SHARED SEQUENCE VERIFICATION ---")
print(verification["Unique_Shared_Sequences"].value_counts().sort_index())

if verification["Unique_Shared_Sequences"].max() == 1:
    print("Success: Base and Cellular conditions use the same raw sequence within each (A, N, Trial).")
else:
    print("Warning: Some trials do not share the same raw sequence across conditions.")

# ==== THERMO SAR/TAR PLOT (defines regression, harmless to include) ====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
import os

# --- 1. LOAD DATA ---
file_path = os.path.join(RESULTS_DIR, "thermodynamics_grid.csv")
df = pd.read_csv(file_path)

# Robustly handle column names
if 'Accuracy' in df.columns and 'TAR' not in df.columns:
    df['TAR'] = df['Accuracy']
elif 'TAR' not in df.columns:
    raise ValueError("Dataset missing 'TAR' or 'Accuracy' column.")

# --- 2. CALCULATE 1/k (DEFECT DENSITY) WITH BASE CASE AT 0 ---

# Logic: If k=0 (Base), density is 0. If k>0, density is 1/k.
df['Inv_K'] = df['Interval'].apply(lambda k: 0.0 if k == 0 else 1.0/k)

# # --- FILTER: ONLY PLOT N >= 8 ---
df = df[df['Seq_Length'] >= 8]
#target_lengths = [128, 256, 512]
#df = df[df['Seq_Length'].isin(target_lengths)]

# Calculate mean AND std (across the 20 trials per condition) for both TAR and
# SAR -- SAR (whole-sequence exact match) is the paper's headline "accuracy
# cliff" metric, TAR (per-token accuracy) is the softer companion metric.
summary = (
    df.groupby(['Seq_Length', 'Alphabet_Size', 'Inv_K'])[['TAR', 'SAR']]
      .agg(['mean', 'std'])
)
summary.columns = ['_'.join(c) for c in summary.columns]
summary = summary.reset_index().sort_values(['Seq_Length', 'Alphabet_Size', 'Inv_K'])
summary[['TAR_std', 'SAR_std']] = summary[['TAR_std', 'SAR_std']].fillna(0.0)  # single-sample groups

print(df[['Seq_Length', 'Alphabet_Size', 'Interval', 'TAR', 'SAR']])

# --- 3. PLOTTING (TAR on top row, SAR on bottom row) ---
seq_lengths = sorted(summary['Seq_Length'].unique())
alphabet_sizes = sorted(summary['Alphabet_Size'].unique())
num_lengths = len(seq_lengths)

# Dynamic Styling
palette = sns.color_palette("bright", n_colors=len(alphabet_sizes))
colors = {A: col for A, col in zip(alphabet_sizes, palette)}
marker_list = ['o', 's', 'D', '^', 'v', '<', '>']
markers = {A: marker_list[i % len(marker_list)] for i, A in enumerate(alphabet_sizes)}

sns.set_theme(style="whitegrid", font_scale=1.1)

fig, axes = plt.subplots(2, max(num_lengths, 1), figsize=(4 * max(num_lengths, 1), 10),
                          sharex=True, sharey=True, squeeze=False)

regression_results = []

for row, (metric, ylabel) in enumerate([('TAR', 'Total Token Accuracy (TAR)'),
                                         ('SAR', 'Sequence Accuracy Rate (SAR)')]):
    for i, N in enumerate(seq_lengths):
        ax = axes[row][i]
        sub_N = summary[summary['Seq_Length'] == N]

        for A in alphabet_sizes:
            sub_A = sub_N[sub_N['Alphabet_Size'] == A]

            # Need at least 2 points for a line
            if len(sub_A) < 2: continue

            inv_k_vals = sub_A['Inv_K'].values
            metric_vals = sub_A[f'{metric}_mean'].values
            metric_err = sub_A[f'{metric}_std'].values

            # Linear Regression (Including the Base point at x=0)
            slope, intercept, r_value, p_value, std_err = linregress(inv_k_vals, metric_vals)

            regression_results.append({
                'Metric': metric,
                'Seq_Length': N,
                'Alphabet': A,
                'Slope': slope,
                'Intercept (Base Fit)': intercept,
                'R_squared': r_value**2
            })

            # --- PLOT DYNAMIC 1/A REFERENCE LINE (chance baseline) ---
            ax.axhline(1.0/A, color=colors[A], linestyle=':', alpha=0.4, linewidth=3)

            # Plot Data Points with error bars (std across the 20 trials)
            ax.errorbar(inv_k_vals, metric_vals, yerr=metric_err,
                        fmt=markers[A], color=colors[A], markersize=8, alpha=0.85,
                        capsize=4, linestyle='none')

            # Plot Fit Line
            x_fit = np.linspace(0, max(inv_k_vals) * 1.1, 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, linestyle='--', color=colors[A], alpha=0.5, linewidth=1.5)

        ax.set_title(f'Length $N = {N}$', fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_ylabel(ylabel, fontsize=13)
        if row == 1 and i == num_lengths // 2:
            ax.set_xlabel('$1/k$', fontsize=14)

# Legend
handles = [plt.Line2D([0], [0], marker=markers[A], color='w', markerfacecolor=colors[A],
                      markersize=10, label=f'Alphabet $A={A}$') for A in alphabet_sizes]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=len(alphabet_sizes), fontsize=12)

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'scaling_law_N8_plus.png')
plt.savefig(plot_path, bbox_inches='tight', dpi=300)
print(f"Saved plot to {plot_path}")
plt.show()

# --- 4. PRINT RESULTS ---
print("\n--- SCALING LAW REGRESSION (N >= 8), TAR and SAR ---")
results_df = pd.DataFrame(regression_results).round(3)
for metric in ['TAR', 'SAR']:
    print(f"\n{metric}:")
    pivot = results_df[results_df['Metric'] == metric].pivot_table(
        index='Alphabet', columns='Seq_Length', values='R_squared')
    print(pivot)

results_df.to_csv(os.path.join(RESULTS_DIR, "thermodynamics_regression.csv"), index=False)

# ==== SK FIT ====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import os

# ============================================================
# SK SPIN-GLASS FIT
# ============================================================
# Hou, Zhou, Hu, Chen, You, Qi -- "How Focused Are LLMs? A Quantitative
# Study via Repetitive Deterministic Prediction Tasks" (arXiv:2511.00763).
#
# Eq. 14:  SAR(N) = exp(-beta0 * N * alpha**(N-1))
#   beta0 = intrinsic per-token error rate
#   alpha = error-accumulation factor (alpha > 1 produces the cliff;
#           alpha == 1 would just be ordinary per-token exponential decay)
# Eq. 15:  cliff length  N* ~= 1 + log(1/beta0) / log(alpha)
#
# Fit separately per (Alphabet, Method, Interval) so alpha_base and
# alpha_cellular are directly comparable -- this is the notebook's actual
# test of whether "|" boundaries slow the accuracy cliff, stated in the
# same language as the paper instead of just an empirical linear slope.

def sk_model(N, beta0, alpha):
    N = np.asarray(N, dtype=np.float64)
    return np.exp(-beta0 * N * np.power(alpha, N - 1))

def fit_sk(N_vals, sar_vals):
    N_vals = np.asarray(N_vals, dtype=np.float64)
    sar_vals = np.asarray(sar_vals, dtype=np.float64)
    try:
        popt, _ = curve_fit(
            sk_model, N_vals, sar_vals,
            p0=[0.05, 1.2], bounds=([1e-6, 1.0 + 1e-6], [1.0, 5.0]), maxfev=20000,
        )
        beta0, alpha = popt
        n_star = 1 + np.log(1.0 / beta0) / np.log(alpha) if alpha > 1 else np.inf
        return beta0, alpha, n_star
    except RuntimeError:
        return np.nan, np.nan, np.nan

df_sk = pd.read_csv(os.path.join(RESULTS_DIR, "thermodynamics_grid.csv"))

sk_results = []
for grp in sorted(df_sk['Alphabet_Size'].unique()):
    sub_G = df_sk[df_sk['Alphabet_Size'] == grp]
    for method, sub_M in sub_G.groupby('Method'):
        for interval, sub_I in sub_M.groupby('Interval'):
            per_N = sub_I.groupby('Seq_Length')['SAR'].mean().reset_index()
            if len(per_N) < 3:  # need enough distinct N to fit 2 free params
                continue
            beta0, alpha, n_star = fit_sk(per_N['Seq_Length'], per_N['SAR'])
            sk_results.append({
                'Alphabet_Size': grp, 'Method': method, 'Interval': interval,
                'Beta0': beta0, 'Alpha': alpha, 'N_star': n_star,
                'N_points': len(per_N),
            })

sk_df = pd.DataFrame(sk_results)
print("--- SK MODEL FIT: beta0 (intrinsic error rate), alpha (error accumulation), N* (cliff length) ---")
print(sk_df.round(4).to_string(index=False))
sk_df.to_csv(os.path.join(RESULTS_DIR, "thermodynamics_sk_fit.csv"), index=False)

# --- Headline comparison: does alpha drop from Base to Cellular? ---
base_alpha = sk_df[sk_df['Method'] == 'Base'].set_index('Alphabet_Size')['Alpha']
cell_alpha_mean = (
    sk_df[sk_df['Method'] == 'Cellular_Bounds']
    .groupby('Alphabet_Size')['Alpha']
    .mean()
)
print("\n--- ALPHA: Base vs mean(Cellular) per Alphabet ---")
for grp in sorted(set(base_alpha.index) & set(cell_alpha_mean.index)):
    b, c = base_alpha[grp], cell_alpha_mean[grp]
    pct = 100 * (b - c) / b if b else float('nan')
    print(f"Alphabet={grp}: alpha_base={b:.4f}  alpha_cellular_mean={c:.4f}  "
          f"({'-' if pct >= 0 else '+'}{abs(pct):.1f}% {'reduction' if pct >= 0 else 'increase'})")

# --- Plot: alpha vs boundary density (Inv_K), Base shown as a reference line ---
sk_df['Inv_K'] = sk_df['Interval'].apply(lambda k: 0.0 if k == 0 else 1.0 / k)
groups = sorted(sk_df['Alphabet_Size'].unique())
palette = sns.color_palette("bright", n_colors=len(groups))
colors = {g: c for g, c in zip(groups, palette)}

sns.set_theme(style="whitegrid", font_scale=1.1)
fig, ax = plt.subplots(figsize=(7, 5.5))
for grp in groups:
    sub = sk_df[(sk_df['Alphabet_Size'] == grp) & (sk_df['Method'] == 'Cellular_Bounds')].sort_values('Inv_K')
    if len(sub):
        ax.plot(sub['Inv_K'], sub['Alpha'], marker='o', markersize=8, linewidth=2,
                color=colors[grp], label=f'Alphabet={grp} (Cellular)')
    base_row = sk_df[(sk_df['Alphabet_Size'] == grp) & (sk_df['Method'] == 'Base')]
    if len(base_row):
        ax.axhline(base_row['Alpha'].iloc[0], color=colors[grp], linestyle=':', alpha=0.6, linewidth=2)

ax.axhline(1.0, color='black', linestyle='-', alpha=0.3, linewidth=1)
ax.set_xlabel('$1/k$ (boundary frequency)', fontsize=13)
ax.set_ylabel(r'Error-accumulation factor $\alpha$', fontsize=13)
ax.set_title('Does boundary density reduce $\\alpha$?\n(dotted lines = Base condition, no boundaries)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'thermodynamics_sk_alpha_vs_density.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nSaved plot to {plot_path}")
plt.show()

# ==== ATTENTION SWEEP (boundary sink, entropy, clustering) ====

import os
import csv
import random
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================
# 1. SETUP
# ============================================================

run_id = 0  # representative example id (see SECTION 3 below)

output_dir = RESULTS_DIR  # from the MODEL SELECTION cell -- results/<MODEL_TAG>/

# Reuses the `model` / `tokenizer` already loaded above -- no reload needed.
# (output_attentions is requested per forward-call below instead of at load
# time; the shared model was loaded with attn_implementation="eager" so
# attentions are actually computable.)
model.eval()

num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads

# Seed the representative-example prompt draw so it's reproducible.
ATTENTION_SEED = 123
random.seed(ATTENTION_SEED)

# ============================================================
# 2. TASK ENGINE
# ============================================================

class TaskEngine:
    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size
        self.alphabet = [chr(ord('A') + i) for i in range(alphabet_size)]

    def generate_paired_prompts(self, length, interval, rng=None):
        rng = rng or random
        raw_chars = rng.choices(self.alphabet, k=length)

        base_str = " ".join(raw_chars)
        prompt_base = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(self.alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {self.alphabet[-1]} -> {self.alphabet[0]}, and {self.alphabet[0]} -> {self.alphabet[1]}.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {base_str}
Output: """

        cellular_w_bounds = []
        for i, char in enumerate(raw_chars):
            cellular_w_bounds.append(char)
            if interval > 0 and (i + 1) % interval == 0 and i != length - 1:
                cellular_w_bounds.append("|")

        cell_str = " ".join(cellular_w_bounds)

        prompt_cell = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(self.alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {self.alphabet[-1]} -> {self.alphabet[0]}, and {self.alphabet[0]} -> {self.alphabet[1]}.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {cell_str}
Output: """

        return prompt_base, prompt_cell


engine = TaskEngine(alphabet_size=2)
N_length = 16
k_interval = 2
prompt_base, prompt_cell = engine.generate_paired_prompts(
    length=N_length,
    interval=k_interval
)

# ============================================================
# 3. ATTENTION EXTRACTION (FULL TENSOR)
# ============================================================

def extract_all_layers_and_heads(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    tokens = [tokenizer.decode([t]) for t in inputs.input_ids[0]]

    # Cropping
    try:
        input_idx = next(i for i, t in enumerate(tokens) if 'Input' in t)
        output_idx = next(i for i in range(input_idx, len(tokens)) if 'Output' in tokens[i])

        start_idx = input_idx + 1
        while start_idx < output_idx and tokens[start_idx].strip() in [':', '']:
            start_idx += 1

        end_idx = output_idx
        while end_idx > start_idx and tokens[end_idx - 1].strip() == '':
            end_idx -= 1
    except StopIteration:
        start_idx, end_idx = 0, len(tokens)

    clean_tokens = [t.strip() if t.strip() else '_' for t in tokens[start_idx:end_idx]]

    attentions = outputs.attentions
    seq_len = end_idx - start_idx

    full_tensor = np.zeros(
        (num_layers, num_heads, seq_len, seq_len),
        dtype=np.float32
    )

    for l in range(num_layers):
        layer_attn = attentions[l][0]
        for h in range(num_heads):
            attn_matrix = layer_attn[h].cpu().to(torch.float32).numpy()
            cropped = attn_matrix[start_idx:end_idx, start_idx:end_idx]
            full_tensor[l, h] = cropped

    return full_tensor, clean_tokens


print("\nRunning forward pass for Base Prompt (representative example)...")
attn_base_tensor, tokens_base = extract_all_layers_and_heads(prompt_base)

print("Running forward pass for Cellular Prompt (representative example)...")
attn_cell_tensor, tokens_cell = extract_all_layers_and_heads(prompt_cell)

# ============================================================
# 4. SAVE FULL MATRICES (representative example only -- used by the
#    detailed single-example cells further down: Head Entropy, Network
#    Analysis, Sample Presentation, attention-weight distributions,
#    Correlation Decay)
# ============================================================

print("Saving full attention tensors...")

np.savez_compressed(
    os.path.join(output_dir, f"attention_run_{run_id}.npz"),
    attn_base=attn_base_tensor,
    attn_cell=attn_cell_tensor,
    tokens_base=np.array(tokens_base),
    tokens_cell=np.array(tokens_cell),
    N_length=N_length,
    k_interval=k_interval
)

print("Full tensors saved successfully.\n")

# ============================================================
# 5. MULTI-CONDITION SWEEP (statistics, not just one anecdote)
# ============================================================
# The representative example above is a single (N, k) case. To see whether
# the boundary-sink / entropy effects are systematic -- not an artifact of
# one random draw -- sweep across sequence lengths, boundary intervals, and
# many trials, the same way the "Qwen Scaling" grid search does for
# accuracy. Only scalar metrics (averaged over all layers/heads) are kept
# per condition -- full attention tensors are only saved for the single
# representative example above, to keep storage bounded.

def compute_condition_metrics(attn_base_tensor, attn_cell_tensor, tokens_cell):
    boundary_indices = [i for i, t in enumerate(tokens_cell) if '|' in t]
    eps = 1e-12
    rows = []
    for l in range(num_layers):
        for h in range(num_heads):
            m_base = attn_base_tensor[l, h]
            m_cell = attn_cell_tensor[l, h]

            N_base = m_base.shape[0]
            N_cell = m_cell.shape[0]

            if N_base > 1:
                base_diag = np.mean(np.diag(m_base))
                base_entropy = np.mean(-np.sum(m_base * np.log2(m_base + eps), axis=-1))
                dist_base = np.abs(np.arange(N_base)[:, None] - np.arange(N_base)[None, :]) / N_base
                base_dist = np.mean(np.sum(m_base * dist_base, axis=-1))
            else:
                base_diag = base_entropy = base_dist = 0.0

            if N_cell > 1:
                cell_diag = np.mean(np.diag(m_cell))
                cell_entropy = np.mean(-np.sum(m_cell * np.log2(m_cell + eps), axis=-1))
                dist_cell = np.abs(np.arange(N_cell)[:, None] - np.arange(N_cell)[None, :]) / N_cell
                cell_dist = np.mean(np.sum(m_cell * dist_cell, axis=-1))
                if len(boundary_indices) > 0:
                    cell_boundary = np.mean(np.sum(m_cell[:, boundary_indices], axis=-1))
                else:
                    cell_boundary = 0.0
            else:
                cell_diag = cell_entropy = cell_dist = cell_boundary = 0.0

            rows.append({
                "Base_Diag": base_diag, "Cell_Diag": cell_diag,
                "Base_Entropy": base_entropy, "Cell_Entropy": cell_entropy,
                "Base_Dist": base_dist, "Cell_Dist": cell_dist,
                "Cell_Boundary_Attn": cell_boundary,
            })
    # Average over all (layer, head) pairs -> one summary row per condition
    return pd.DataFrame(rows).mean().to_dict()


# A *fixed* threshold (like the 0.1 the single-example Network Analysis cell
# uses) does not work here: averaging over 28 layers x 28 heads dilutes the
# attention mass, and that dilution gets worse as N grows (a uniform
# attention baseline is already only 1/seq_len). Measured on real examples:
# threshold=0.1 zeroed out 99% of the averaged matrix even at N=16, and any
# *fixed* absolute threshold degenerates back to a near-empty graph again
# once N reaches the hundreds. Instead threshold relative to the uniform-
# attention baseline (1/seq_len) for each matrix -- verified non-degenerate
# (positive edge count, non-zero clustering) from N=8 through N=256.
CLUSTERING_THRESHOLD_MULTIPLIER = 0.5  # threshold = multiplier / seq_len

def compute_condition_clustering(attn_base_tensor, attn_cell_tensor,
                                  multiplier=CLUSTERING_THRESHOLD_MULTIPLIER):
    # Per-(layer,head) clustering for every sweep condition would mean ~28*28
    # graph builds PER condition (~375k total across the full sweep) -- too
    # slow to be worth it. Instead average the attention matrix across all
    # layers/heads into one aggregate network per condition/side, and compute
    # a single clustering coefficient on that. Coarser than the single-example
    # per-layer view in the Network Analysis cell below, but captures the same
    # "does chunking induce more local modularity" question with real N/k
    # statistics instead of one anecdote.
    def clustering_of(m_avg):
        threshold = multiplier / m_avg.shape[0]
        m = m_avg.copy()
        m[m < threshold] = 0.0
        G = nx.from_numpy_array(m, create_using=nx.DiGraph)
        G.remove_edges_from(nx.selfloop_edges(G))  # see the Network Analysis
                                                     # cell for why self-loops
                                                     # must be dropped (NaN fix)
        vals = list(nx.clustering(G, weight='weight').values())
        return np.nanmean(vals) if vals else 0.0

    base_clustering = clustering_of(attn_base_tensor.mean(axis=(0, 1)))
    cell_clustering = clustering_of(attn_cell_tensor.mean(axis=(0, 1)))
    return base_clustering, cell_clustering


SWEEP_SEQUENCE_LENGTHS = [4, 8, 16, 32, 64, 128, 256]
# Runtime note: forward passes stay cheap even at N=256 (measured ~0.05-0.1s
# each on an A100), so the sweep itself is fast (a few minutes). Trim this
# list or SWEEP_TRIALS below if you want a quicker pass.
SWEEP_INTERVALS = [2, 4, 8]   # no 0: a k=0 "cellular" prompt is identical to the base prompt
SWEEP_TRIALS = 20

sweep_conditions = [
    (N, k, t)
    for N in SWEEP_SEQUENCE_LENGTHS
    for k in SWEEP_INTERVALS
    for t in range(SWEEP_TRIALS)
    if k < N
]
print(f"\nStarting attention sweep: {len(sweep_conditions)} conditions "
      f"({len(sweep_conditions) * 2} forward passes)...")

sweep_rows = []
for (N, k, t) in tqdm(sweep_conditions, desc="Attention sweep"):
    # Independent, deterministic seed per condition -- does not disturb the
    # global `random` state used for the representative example above.
    local_seed = 50_000 * N + 100 * k + t
    rng = random.Random(local_seed)

    p_base, p_cell = engine.generate_paired_prompts(length=N, interval=k, rng=rng)
    ab, _ = extract_all_layers_and_heads(p_base)
    ac, tc = extract_all_layers_and_heads(p_cell)

    m = compute_condition_metrics(ab, ac, tc)
    m["Delta_Diag"] = m["Cell_Diag"] - m["Base_Diag"]
    m["Delta_Entropy"] = m["Cell_Entropy"] - m["Base_Entropy"]
    m["Delta_Dist"] = m["Cell_Dist"] - m["Base_Dist"]
    m["Base_Clustering"], m["Cell_Clustering"] = compute_condition_clustering(ab, ac)
    m["Delta_Clustering"] = m["Cell_Clustering"] - m["Base_Clustering"]
    m.update(Seq_Length=N, Interval=k, Trial=t)
    sweep_rows.append(m)

sweep_df = pd.DataFrame(sweep_rows)
sweep_df["Inv_K"] = 1.0 / sweep_df["Interval"]

csv_filename = os.path.join(output_dir, "attention_metrics.csv")
sweep_df.to_csv(csv_filename, index=False)
print(f"\nAttention sweep complete. Saved {len(sweep_df)} condition-level rows to '{csv_filename}'")

summary = (
    sweep_df.groupby(["Seq_Length", "Interval"])[["Cell_Boundary_Attn", "Delta_Entropy", "Delta_Diag"]]
            .mean()
            .reset_index()
)
print("\n--- MEAN ATTENTION METRICS BY (Seq_Length, Interval) ---")
print(summary.to_string(index=False))

# ============================================================
# 6. HEATMAP GENERATION (representative example only, commented out)
# ============================================================

# print("Generating heatmaps...")

# for l in tqdm(range(num_layers), desc="Layers"):
#     for h in range(num_heads):

#         m_base = attn_base_tensor[l, h]
#         m_cell = attn_cell_tensor[l, h]

#         fig, (ax1, ax2, cbar_ax) = plt.subplots(
#             1, 3, figsize=(18, 7),
#             gridspec_kw={'width_ratios': [1, 1, 0.05]}
#         )

#         sns.heatmap(
#             m_base,
#             xticklabels=tokens_base,
#             yticklabels=tokens_base,
#             cmap="rocket",
#             ax=ax1,
#             cbar=False,
#             vmin=0,
#             vmax=1
#         )

#         ax1.set_title(f"Base | Layer {l}, Head {h}")
#         ax1.set_aspect('equal')

#         sns.heatmap(
#             m_cell,
#             xticklabels=tokens_cell,
#             yticklabels=tokens_cell,
#             cmap="rocket",
#             ax=ax2,
#             cbar_ax=cbar_ax,
#             vmin=0,
#             vmax=1
#         )

#         ax2.set_title(f"Cellular | Layer {l}, Head {h}")
#         ax2.set_aspect('equal')

#         if len(boundary_indices) > 0:
#             for b in boundary_indices:
#                 ax2.axvline(x=b, color='cyan', linestyle='--', linewidth=1)
#                 ax2.axhline(y=b, color='cyan', linestyle='--', linewidth=1)

#         plt.tight_layout()

#         filename = os.path.join(
#             output_dir,
#             f"layer_{l:02d}_head_{h:02d}.png"
#         )
#         plt.savefig(filename, dpi=100)
#         plt.close(fig)

# print("Done.")

# ==== BOUNDARY SINK ANALYSIS (statistical) ====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
import os

# ============================================================
# BOUNDARY SINK ANALYSIS (statistical, across the attention sweep)
# ============================================================
# attention_metrics.csv now holds one row per (Seq_Length, Interval, Trial)
# condition from the sweep in the Attention Analysis setup cell -- not a
# single snapshot. Cell_Boundary_Attn is the average attention mass (across
# all layers/heads) that lands on the "|" boundary tokens; if boundaries
# act as attention sinks, this should be systematically elevated, and grow
# as boundaries get more frequent (larger Inv_K = 1/Interval).

csv_filename = os.path.join(RESULTS_DIR, "attention_metrics.csv")
df = pd.read_csv(csv_filename)

print(f"Loaded {len(df)} sweep conditions from '{csv_filename}'")
print(df[["Seq_Length", "Interval", "Trial", "Cell_Boundary_Attn", "Delta_Entropy", "Delta_Diag"]].head(10))

# --- Overall summary ---
print("\n--- OVERALL (across all conditions) ---")
print(df[["Cell_Boundary_Attn", "Delta_Entropy", "Delta_Diag", "Delta_Dist"]].agg(["mean", "std"]))

# --- Plot: Boundary attention vs 1/k, faceted by sequence length ---
seq_lengths = sorted(df["Seq_Length"].unique())
num_lengths = len(seq_lengths)

sns.set_theme(style="whitegrid", font_scale=1.1)

if num_lengths == 1:
    fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    axes = [axes]
else:
    fig, axes = plt.subplots(1, num_lengths, figsize=(4 * num_lengths, 5), sharex=True, sharey=True)

regression_results = []

for i, N in enumerate(seq_lengths):
    ax = axes[i]
    sub_N = df[df["Seq_Length"] == N]

    # Per-(Interval) mean +/- std across trials, then regress the trial-level points
    summary_N = sub_N.groupby("Inv_K")["Cell_Boundary_Attn"].agg(["mean", "std"]).reset_index()

    ax.scatter(sub_N["Inv_K"], sub_N["Cell_Boundary_Attn"], color="tab:red", alpha=0.35, s=30, label="Trials")
    ax.errorbar(summary_N["Inv_K"], summary_N["mean"], yerr=summary_N["std"],
                fmt="o", color="tab:red", markersize=8, capsize=4, label="Mean +/- std")

    if sub_N["Inv_K"].nunique() >= 2:
        slope, intercept, r_value, p_value, std_err = linregress(sub_N["Inv_K"], sub_N["Cell_Boundary_Attn"])
        regression_results.append({
            "Seq_Length": N, "Slope": slope, "Intercept": intercept,
            "R_squared": r_value ** 2, "P_value": p_value,
        })
        x_fit = np.linspace(0, sub_N["Inv_K"].max() * 1.1, 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, linestyle="--", color="tab:red", alpha=0.6, linewidth=1.5)

    ax.set_title(f"Length $N = {N}$", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.set_ylabel("Mean Attention on Boundary Tokens", fontsize=13)
    if i == num_lengths // 2:
        ax.set_xlabel("$1/k$ (boundary frequency)", fontsize=13)

axes[0].legend(fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "boundary_sink_analysis.png"), bbox_inches="tight", dpi=300)
plt.show()

print("\n--- BOUNDARY-SINK REGRESSION (Cell_Boundary_Attn ~ Inv_K, by Seq_Length) ---")
print(pd.DataFrame(regression_results).round(4).to_string(index=False))

# --- Does chunking also change entropy/diagonal focus, beyond the boundary tokens themselves? ---
delta_summary = (
    df.groupby(["Seq_Length", "Interval"])[["Delta_Entropy", "Delta_Diag", "Delta_Dist"]]
      .mean()
      .reset_index()
      .sort_values(["Seq_Length", "Interval"])
)
print("\n--- MEAN DELTA (Cellular - Base) BY (Seq_Length, Interval) ---")
print(delta_summary.to_string(index=False))
