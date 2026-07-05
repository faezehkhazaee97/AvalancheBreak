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

# ==== MODEL LOADER ====

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random

print(f"Loading {MODEL_ID} into VRAM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager",
)
model.eval()

class TaskEngine:
    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size
        self.alphabet = [chr(ord("A") + i) for i in range(alphabet_size)]
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}

    def generate_raw_input(self, length, rng=None):
        if rng is None:
            rng = random
        return rng.choices(self.alphabet, k=length)

ABLATION_ALPHABET_SIZE = 2  # matches the ablation grid cell -- not re-running that cell, just need this constant


# ==== CYCLIC-SHIFT GRID SEARCH (extended: k up to 16) ====

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
PERIODIC_INTERVALS = [0, 2, 4, 8, 16]   # 0 = Base; k=16 added to match the paper's Appendix B grid exactly
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

# ==== TAR/SAR REGRESSION ====

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
    # A curve_fit on degenerate data (e.g. SAR pinned at 0 for every tested
    # N, which happens a lot at large N/alphabet since that IS the cliff)
    # still "succeeds" -- it just converges to an arbitrary point in the
    # (beta0, alpha) space, since there's no real decay curve to constrain
    # it. That produced identical-looking beta0/alpha across totally
    # unrelated conditions before this fix. Require at least 2 distinct
    # SAR values, not all zero, before trusting the fit.
    N_vals = np.asarray(N_vals, dtype=np.float64)
    sar_vals = np.asarray(sar_vals, dtype=np.float64)
    n_distinct = len(set(np.round(sar_vals, 6)))
    is_valid = n_distinct >= 2 and not np.all(sar_vals == 0)
    if not is_valid:
        return np.nan, np.nan, np.nan, False
    try:
        popt, _ = curve_fit(
            sk_model, N_vals, sar_vals,
            p0=[0.05, 1.2], bounds=([1e-6, 1.0 + 1e-6], [1.0, 5.0]), maxfev=20000,
        )
        beta0, alpha = popt
        n_star = 1 + np.log(1.0 / beta0) / np.log(alpha) if alpha > 1 else np.inf
        return beta0, alpha, n_star, True
    except RuntimeError:
        return np.nan, np.nan, np.nan, False

df_sk = pd.read_csv(os.path.join(RESULTS_DIR, "thermodynamics_grid.csv"))

sk_results = []
for grp in sorted(df_sk['Alphabet_Size'].unique()):
    sub_G = df_sk[df_sk['Alphabet_Size'] == grp]
    for method, sub_M in sub_G.groupby('Method'):
        for interval, sub_I in sub_M.groupby('Interval'):
            per_N = sub_I.groupby('Seq_Length')['SAR'].mean().reset_index()
            if len(per_N) < 3:  # need enough distinct N to fit 2 free params
                continue
            beta0, alpha, n_star, fit_valid = fit_sk(per_N['Seq_Length'], per_N['SAR'])
            sk_results.append({
                'Alphabet_Size': grp, 'Method': method, 'Interval': interval,
                'Beta0': beta0, 'Alpha': alpha, 'N_star': n_star,
                'N_points': len(per_N), 'Fit_Valid': fit_valid,
            })

sk_df = pd.DataFrame(sk_results)
n_invalid = (~sk_df['Fit_Valid']).sum()
print("--- SK MODEL FIT: beta0 (intrinsic error rate), alpha (error accumulation), N* (cliff length) ---")
print(sk_df.round(4).to_string(index=False))
if n_invalid:
    print(f"\nWARNING: {n_invalid}/{len(sk_df)} fits are INVALID (degenerate input -- SAR was"
          f" constant/all-zero across every tested N, so beta0/alpha are arbitrary, not real"
          f" measurements). These rows are excluded from the headline comparison and plot below.")
sk_df.to_csv(os.path.join(RESULTS_DIR, "thermodynamics_sk_fit.csv"), index=False)

# --- Headline comparison: does alpha drop from Base to Cellular? ---
valid_sk_df = sk_df[sk_df['Fit_Valid']]
base_alpha = valid_sk_df[valid_sk_df['Method'] == 'Base'].set_index('Alphabet_Size')['Alpha']
cell_alpha_mean = (
    valid_sk_df[valid_sk_df['Method'] == 'Cellular_Bounds']
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
plot_df = sk_df[sk_df['Fit_Valid']]  # skip degenerate (all-zero-SAR) fits in the plot
groups = sorted(plot_df['Alphabet_Size'].unique())
palette = sns.color_palette("bright", n_colors=len(groups))
colors = {g: c for g, c in zip(groups, palette)}

sns.set_theme(style="whitegrid", font_scale=1.1)
fig, ax = plt.subplots(figsize=(7, 5.5))
for grp in groups:
    sub = plot_df[(plot_df['Alphabet_Size'] == grp) & (plot_df['Method'] == 'Cellular_Bounds')].sort_values('Inv_K')
    if len(sub):
        ax.plot(sub['Inv_K'], sub['Alpha'], marker='o', markersize=8, linewidth=2,
                color=colors[grp], label=f'Alphabet={grp} (Cellular)')
    base_row = plot_df[(plot_df['Alphabet_Size'] == grp) & (plot_df['Method'] == 'Base')]
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

# ==== ERROR CONTAINMENT ====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# ERROR-CONTAINMENT ANALYSIS (does crossing a boundary help recovery?)
# ============================================================
# First_Error_Pos already exists in the saved CSVs but nothing analyzes it.
# This is the direct, mechanistic test of "boundaries contain errors"
# rather than the indirect evidence from the boundary-density regression:
# for every position immediately following a WRONG token, does the model
# recover (produce the next token correctly) more often when a "|" was just
# crossed than when it wasn't? Reconstructed entirely from the already-saved
# Ground_Truth / LLM_Output strings -- no new data collection needed.

def reconstruct_binary_series(ground_truth, llm_output):
    truth = str(ground_truth)
    pred = str(llm_output)[:len(truth)]
    return [1 if i < len(pred) and pred[i] == truth[i] else 0 for i in range(len(truth))]

def is_boundary_adjacent(pos, interval):
    # True if `pos` (0-indexed in the clean, boundary-stripped sequence) is
    # the first character immediately after a "|" was crossed.
    return interval > 0 and pos > 0 and pos % interval == 0

def error_recovery_events(df):
    events = []
    for _, row in df.iterrows():
        series = reconstruct_binary_series(row['Ground_Truth'], row['LLM_Output'])
        interval = row['Interval']
        for i in range(1, len(series)):
            if series[i - 1] == 0:  # previous token was wrong
                events.append({
                    'Method': row['Method'],
                    'Interval': interval,
                    'Seq_Length': row['Seq_Length'],
                    'Boundary_Adjacent': is_boundary_adjacent(i, interval),
                    'Recovered': series[i],
                })
    return pd.DataFrame(events)


def summarize_recovery(events_df, task_label):
    if len(events_df) == 0:
        print(f"[{task_label}] No error-recovery events found (no errors occurred in this data).")
        return None

    summary = (
        events_df.groupby(['Method', 'Boundary_Adjacent'])['Recovered']
                  .agg(['mean', 'count'])
                  .reset_index()
    )
    print(f"\n--- [{task_label}] RECOVERY PROBABILITY: P(correct | previous token wrong) ---")
    print(summary.round(4).to_string(index=False))

    cell_rows = summary[summary['Method'] == 'Cellular_Bounds']
    adj = cell_rows[cell_rows['Boundary_Adjacent'] == True]
    non_adj = cell_rows[cell_rows['Boundary_Adjacent'] == False]
    if len(adj) and len(non_adj):
        p_adj, p_non = adj['mean'].iloc[0], non_adj['mean'].iloc[0]
        print(f"\n[{task_label}] Recovery right after a boundary: {p_adj:.3f}  "
              f"vs elsewhere in Cellular trials: {p_non:.3f}  "
              f"({'+' if p_adj >= p_non else ''}{100*(p_adj-p_non):.1f} pp)")
    return summary


thermo_df = pd.read_csv(os.path.join(RESULTS_DIR, "thermodynamics_grid.csv"))
thermo_events = error_recovery_events(thermo_df)
thermo_recovery_summary = summarize_recovery(thermo_events, "Cyclic Shift")

modsum_df = pd.read_csv(os.path.join(RESULTS_DIR, "running_modsum_grid.csv"))
modsum_events = error_recovery_events(modsum_df)
modsum_recovery_summary = summarize_recovery(modsum_events, "Running Sum Mod M")

# --- Plot: recovery probability by Seq_Length, boundary-adjacent vs not (Cellular only) ---
sns.set_theme(style="whitegrid", font_scale=1.1)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, events_df, label in [(axes[0], thermo_events, "Cyclic Shift"),
                              (axes[1], modsum_events, "Running Sum Mod M")]:
    if len(events_df) == 0:
        ax.set_title(f"{label}\n(no error-recovery events)")
        continue
    cell_events = events_df[events_df['Method'] == 'Cellular_Bounds']
    per_N = (
        cell_events.groupby(['Seq_Length', 'Boundary_Adjacent'])['Recovered']
                   .mean()
                   .reset_index()
    )
    for adj_val, color, marker, lbl in [(True, '#2b83ba', 'o', 'Right after boundary'),
                                          (False, '#d7191c', 's', 'Elsewhere')]:
        sub = per_N[per_N['Boundary_Adjacent'] == adj_val].sort_values('Seq_Length')
        if len(sub):
            ax.plot(sub['Seq_Length'], sub['Recovered'], marker=marker, markersize=8,
                    linewidth=2, color=color, label=lbl)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Sequence Length $N$', fontsize=12)
    ax.set_ylabel('P(recover | previous token wrong)', fontsize=12)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'error_containment_recovery.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nSaved plot to {plot_path}")
plt.show()

if thermo_recovery_summary is not None:
    thermo_recovery_summary.to_csv(os.path.join(RESULTS_DIR, "thermodynamics_error_recovery.csv"), index=False)
if modsum_recovery_summary is not None:
    modsum_recovery_summary.to_csv(os.path.join(RESULTS_DIR, "modsum_error_recovery.csv"), index=False)

# ==== SIGNIFICANCE ====

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, binomtest
import os

# ============================================================
# STATISTICAL SIGNIFICANCE: Cellular vs Base, paired by seed
# ============================================================
# The regressions show boundary density correlates with accuracy, but don't
# establish that Cellular significantly beats Base at a given N. Base and
# Cellular trials share the same underlying raw sequence per Trial (see the
# "Shared_Sequence" verification in the grid-search cells), so a paired
# test is valid and more powerful than an unpaired one.
#   TAR (continuous, per-token accuracy): paired Wilcoxon signed-rank test.
#   SAR (binary, whole-sequence pass/fail): exact McNemar's test on the
#   discordant-pair counts (implemented directly with scipy.stats.binomtest
#   to avoid adding a statsmodels dependency).

def mcnemar_exact_pvalue(b, c):
    if b + c == 0:
        return 1.0
    return binomtest(min(b, c), b + c, 0.5).pvalue

def paired_significance(df, group_col, task_label):
    results = []
    base = df[df['Method'] == 'Base']
    cell = df[df['Method'] == 'Cellular_Bounds']

    for grp in sorted(df[group_col].unique()):
        for N in sorted(df['Seq_Length'].unique()):
            for interval in sorted(cell['Interval'].unique()):
                b_df = base[(base[group_col] == grp) & (base['Seq_Length'] == N)]
                c_df = cell[(cell[group_col] == grp) & (cell['Seq_Length'] == N) & (cell['Interval'] == interval)]
                if len(b_df) == 0 or len(c_df) == 0:
                    continue

                merged = b_df[['Trial', 'TAR', 'SAR']].merge(
                    c_df[['Trial', 'TAR', 'SAR']], on='Trial', suffixes=('_base', '_cell')
                )
                if len(merged) < 2:
                    continue

                diffs = merged['TAR_cell'] - merged['TAR_base']
                if np.any(diffs != 0):
                    _, p_tar = wilcoxon(merged['TAR_cell'], merged['TAR_base'])
                else:
                    p_tar = 1.0

                base_pass = merged['SAR_base'] >= 0.5
                cell_pass = merged['SAR_cell'] >= 0.5
                b_disc = int((base_pass & ~cell_pass).sum())
                c_disc = int((~base_pass & cell_pass).sum())
                p_sar = mcnemar_exact_pvalue(b_disc, c_disc)

                results.append({
                    group_col: grp, 'Seq_Length': N, 'Interval': interval,
                    'N_pairs': len(merged),
                    'Mean_TAR_Base': merged['TAR_base'].mean(),
                    'Mean_TAR_Cell': merged['TAR_cell'].mean(),
                    'P_value_TAR_Wilcoxon': p_tar,
                    'SAR_Base_pass': int(base_pass.sum()),
                    'SAR_Cell_pass': int(cell_pass.sum()),
                    'P_value_SAR_McNemar': p_sar,
                    'Significant_TAR_p05': p_tar < 0.05,
                    'Significant_SAR_p05': p_sar < 0.05,
                })

    results_df = pd.DataFrame(results)
    print(f"\n--- [{task_label}] PAIRED SIGNIFICANCE: Cellular vs Base ---")
    print(results_df.round(4).to_string(index=False))
    if len(results_df):
        n_sig_tar = results_df['Significant_TAR_p05'].sum()
        n_sig_sar = results_df['Significant_SAR_p05'].sum()
        print(f"\n[{task_label}] Significant at p<0.05: TAR {n_sig_tar}/{len(results_df)} conditions, "
              f"SAR {n_sig_sar}/{len(results_df)} conditions")
    return results_df

thermo_df = pd.read_csv(os.path.join(RESULTS_DIR, "thermodynamics_grid.csv"))
thermo_sig = paired_significance(thermo_df, 'Alphabet_Size', "Cyclic Shift")
thermo_sig.to_csv(os.path.join(RESULTS_DIR, "thermodynamics_significance.csv"), index=False)

modsum_df = pd.read_csv(os.path.join(RESULTS_DIR, "running_modsum_grid.csv"))
modsum_sig = paired_significance(modsum_df, 'Modulus', "Running Sum Mod M")
modsum_sig.to_csv(os.path.join(RESULTS_DIR, "modsum_significance.csv"), index=False)

# ==== TOKEN OVERHEAD ====

import pandas as pd
from transformers import AutoTokenizer
import os

# ============================================================
# TOKEN OVERHEAD: Cellular vs Base
# ============================================================
# The framing here (and in the source paper) is "cheaper than multi-call
# divide-and-conquer" -- this quantifies the cost side of that tradeoff.
# Boundaries add extra "|" characters to both input and output; this counts
# that overhead directly from the already-saved Input_Given / LLM_Output
# strings, no new generation needed. Wall-clock latency isn't recoverable
# retroactively since it wasn't timed during the grid search -- add a timer
# around model.generate() in run_single_trial in a future run for that.

tok_for_counting = AutoTokenizer.from_pretrained(MODEL_ID)

def count_tokens(text):
    return len(tok_for_counting(str(text), add_special_tokens=False)['input_ids'])

def token_overhead_summary(df, task_label):
    df = df.copy()
    df['Input_Tokens'] = df['Input_Given'].apply(count_tokens)
    df['Output_Tokens'] = df['LLM_Output'].apply(count_tokens)
    df['Total_Tokens'] = df['Input_Tokens'] + df['Output_Tokens']

    summary = (
        df.groupby(['Method', 'Seq_Length'])[['Input_Tokens', 'Output_Tokens', 'Total_Tokens']]
          .mean()
          .reset_index()
    )
    print(f"\n--- [{task_label}] MEAN TOKEN COUNT BY METHOD, SEQ_LENGTH ---")
    print(summary.round(2).to_string(index=False))

    pivot = summary.pivot(index='Seq_Length', columns='Method', values='Total_Tokens')
    if 'Base' in pivot.columns and 'Cellular_Bounds' in pivot.columns:
        pivot['Overhead_%'] = 100 * (pivot['Cellular_Bounds'] - pivot['Base']) / pivot['Base']
        print(f"\n[{task_label}] Cellular token overhead vs Base:")
        print(pivot.round(2).to_string())
    return summary

thermo_df = pd.read_csv(os.path.join(RESULTS_DIR, "thermodynamics_grid.csv"))
thermo_token_summary = token_overhead_summary(thermo_df, "Cyclic Shift")
thermo_token_summary.to_csv(os.path.join(RESULTS_DIR, "thermodynamics_token_overhead.csv"), index=False)

modsum_df = pd.read_csv(os.path.join(RESULTS_DIR, "running_modsum_grid.csv"))
modsum_token_summary = token_overhead_summary(modsum_df, "Running Sum Mod M")
modsum_token_summary.to_csv(os.path.join(RESULTS_DIR, "modsum_token_overhead.csv"), index=False)

# ==== HELD-OUT VALIDATION ====

import pandas as pd
import numpy as np
import os

# ============================================================
# HELD-OUT VALIDATION OF THE SK MODEL FIT
# ============================================================
# Fitting Eq. 14 and reporting R^2 only shows the model can interpolate the
# grid it was fit on. The original design here held out the LARGEST tested
# lengths (128, 256) -- but by then SAR has already collapsed to floor (0)
# for virtually every condition, including in the training data, so
# "predicting floor" was trivial and told us nothing (confirmed empirically:
# every held-out test came back exactly on-the-nose with near-zero error).
#
# Real test instead: leave-one-out at the LAST sequence length where the
# model still had any success at all. Train on everything else (including
# points before and after that collapse to zero), then check whether the
# fitted curve correctly predicts that this one specific point still had
# non-trivial residual accuracy. This actually exercises the fitted shape
# instead of just matching an already-flat tail.
# Reuses sk_model / fit_sk from the SK-fit cells above.

def fit_sk_holdout(N_vals, sar_vals):
    N_vals = np.asarray(N_vals, dtype=np.float64)
    sar_vals = np.asarray(sar_vals, dtype=np.float64)
    order = np.argsort(N_vals)
    N_vals, sar_vals = N_vals[order], sar_vals[order]

    nonzero_idx = np.where(sar_vals > 0)[0]
    if len(nonzero_idx) == 0:
        return None  # nothing ever succeeded -- no transition to test against

    test_idx = nonzero_idx[-1]  # the last length where the model still succeeded sometimes
    train_mask = np.ones(len(N_vals), dtype=bool)
    train_mask[test_idx] = False
    N_train, sar_train = N_vals[train_mask], sar_vals[train_mask]
    N_test, sar_test = N_vals[test_idx:test_idx + 1], sar_vals[test_idx:test_idx + 1]

    if len(N_train) < 3:  # need enough remaining points to fit 2 free params
        return None
    beta0, alpha, n_star, fit_valid = fit_sk(N_train, sar_train)
    if not fit_valid:
        return None
    sar_pred = sk_model(N_test, beta0, alpha)
    return {
        'N_held_out': float(N_test[0]), 'Beta0_train': beta0, 'Alpha_train': alpha,
        'N_star_train': n_star,
        'SAR_test_actual': float(sar_test[0]), 'SAR_test_predicted': float(sar_pred[0]),
        'Abs_Error': abs(float(sar_pred[0]) - float(sar_test[0])),
    }

def run_holdout_validation(csv_name, group_col, task_label):
    df_sk = pd.read_csv(os.path.join(RESULTS_DIR, csv_name))
    rows = []
    for grp in sorted(df_sk[group_col].unique()):
        sub_G = df_sk[df_sk[group_col] == grp]
        for method, sub_M in sub_G.groupby('Method'):
            for interval, sub_I in sub_M.groupby('Interval'):
                per_N = sub_I.groupby('Seq_Length')['SAR'].mean().reset_index()
                if len(per_N) < 4:  # need >=3 for training plus 1 held out
                    continue
                result = fit_sk_holdout(per_N['Seq_Length'], per_N['SAR'])
                if result is None:
                    continue
                result.update({group_col: grp, 'Method': method, 'Interval': interval})
                rows.append(result)

    result_df = pd.DataFrame(rows)
    print(f"\n--- [{task_label}] SK MODEL HELD-OUT VALIDATION (leave-one-out at last successful length) ---")
    if len(result_df):
        print(result_df[[group_col, 'Method', 'Interval', 'N_held_out',
                          'SAR_test_actual', 'SAR_test_predicted', 'Abs_Error']].round(4).to_string(index=False))
        print(f"\n[{task_label}] Mean absolute error across {len(result_df)} non-trivial held-out points: "
              f"{result_df['Abs_Error'].mean():.4f}")
    else:
        print(f"[{task_label}] No condition had enough pre-collapse data points for a non-trivial held-out test.")
    return result_df

thermo_holdout = run_holdout_validation("thermodynamics_grid.csv", "Alphabet_Size", "Cyclic Shift")
thermo_holdout.to_csv(os.path.join(RESULTS_DIR, "thermodynamics_sk_holdout.csv"), index=False)

modsum_holdout = run_holdout_validation("running_modsum_grid.csv", "Modulus", "Running Sum Mod M")
modsum_holdout.to_csv(os.path.join(RESULTS_DIR, "modsum_sk_holdout.csv"), index=False)

# ==== ABLATION ANALYSIS (refresh Base reference) ====

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

ablation_df = pd.read_csv(os.path.join(RESULTS_DIR, "boundary_ablation_grid.csv"))

base_df = pd.read_csv(os.path.join(RESULTS_DIR, "thermodynamics_grid.csv"))
base_ref = (
    base_df[(base_df["Method"] == "Base") & (base_df["Alphabet_Size"] == ABLATION_ALPHABET_SIZE)]
    .groupby("Seq_Length")[["TAR", "SAR"]]
    .mean()
    .reset_index()
)
print("--- BASE (no boundaries) REFERENCE ---")
print(base_ref.round(4).to_string(index=False))

summary = ablation_df.groupby(["Ablation_Axis", "Marker", "Placement", "Seq_Length"])[["TAR", "SAR"]].mean().reset_index()
print("\n--- MEAN ACCURACY BY ABLATION CONDITION ---")
print(summary.round(4).to_string(index=False))

sns.set_theme(style="whitegrid", font_scale=1.1)
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

ax = axes[0]
marker_sub = summary[summary["Placement"] == "periodic"]
for marker, sub in marker_sub.groupby("Marker"):
    sub = sub.sort_values("Seq_Length")
    ax.plot(sub["Seq_Length"], sub["TAR"], marker='o', markersize=7, linewidth=2, label=f"marker='{marker}'")
ax.plot(base_ref["Seq_Length"], base_ref["TAR"], marker='x', markersize=8, linewidth=2,
        linestyle='--', color='black', label="Base (no boundaries)")
ax.set_xscale('log', base=2)
ax.set_xlabel("Sequence Length $N$")
ax.set_ylabel("TAR")
ax.set_title("Marker identity\n(fixed periodic interval=4)", fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
placement_sub = summary[(summary["Marker"] == "|") & (summary["Placement"].isin(["periodic", "random"]))]
for placement, sub in placement_sub.groupby("Placement"):
    sub = sub.sort_values("Seq_Length")
    ax.plot(sub["Seq_Length"], sub["TAR"], marker='s', markersize=7, linewidth=2, label=placement)
ax.plot(base_ref["Seq_Length"], base_ref["TAR"], marker='x', markersize=8, linewidth=2,
        linestyle='--', color='black', label="Base (no boundaries)")
ax.set_xscale('log', base=2)
ax.set_xlabel("Sequence Length $N$")
ax.set_title("Placement regularity\n(marker='|', matched boundary count)", fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[2]
count_sub = summary[(summary["Marker"] == "|") & (summary["Placement"].isin(["periodic", "periodic_fixed_count"]))]
labels = {"periodic": "density (interval=4, count grows with N)", "periodic_fixed_count": "fixed count=4 (density falls with N)"}
for placement, sub in count_sub.groupby("Placement"):
    sub = sub.sort_values("Seq_Length")
    ax.plot(sub["Seq_Length"], sub["TAR"], marker='^', markersize=7, linewidth=2, label=labels[placement])
ax.plot(base_ref["Seq_Length"], base_ref["TAR"], marker='x', markersize=8, linewidth=2,
        linestyle='--', color='black', label="Base (no boundaries)")
ax.set_xscale('log', base=2)
ax.set_xlabel("Sequence Length $N$")
ax.set_title("Density vs fixed count", fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "boundary_ablation_analysis.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"\nSaved plot to {plot_path}")
plt.show()

