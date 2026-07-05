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

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"  # ~28GB in bf16 -- fits alongside a 7B run on an 80GB A100

MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"Results directory: {RESULTS_DIR}/")

# ==== MODEL LOADER (no grid search rerun) ====

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


# ==== SK MODEL (from thermodynamics SK-fit cell) ====

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

# ==== ABLATION GRID ====

import pandas as pd
import numpy as np
import random
import itertools
import torch
import os
from tqdm import tqdm

ABLATION_ALPHABET_SIZE = 2
ABLATION_SEQUENCE_LENGTHS = [8, 16, 32, 64]
ABLATION_TRIALS = 15

engine_ablation = TaskEngine(alphabet_size=ABLATION_ALPHABET_SIZE)

def boundary_positions_periodic(length, interval):
    return set(i for i in range(interval - 1, length - 1, interval))

def boundary_positions_random(length, count, rng):
    candidates = list(range(length - 1))
    return set(rng.sample(candidates, min(count, len(candidates))))

def boundary_positions_fixed_count(length, count):
    if count <= 0 or length <= 1:
        return set()
    step = length / (count + 1)
    positions = {int(round(step * (j + 1))) - 1 for j in range(count)}
    return {p for p in positions if 0 <= p < length - 1}

def build_ablation_variant(raw_input, marker, placement, interval=4, fixed_count=4, rng=None):
    rng = rng or random
    length = len(raw_input)
    if placement == "periodic":
        boundary_positions = boundary_positions_periodic(length, interval)
    elif placement == "random":
        n_boundaries = len(boundary_positions_periodic(length, interval))
        boundary_positions = boundary_positions_random(length, n_boundaries, rng)
    elif placement == "periodic_fixed_count":
        boundary_positions = boundary_positions_fixed_count(length, fixed_count)
    else:
        raise ValueError(placement)

    input_w_bounds, output_w_bounds = "", ""
    for i, char in enumerate(raw_input):
        shifted_char = engine_ablation.alphabet[(engine_ablation.char_to_idx[char] + 1) % ABLATION_ALPHABET_SIZE]
        input_w_bounds += char
        output_w_bounds += shifted_char
        if i in boundary_positions:
            input_w_bounds += marker
            output_w_bounds += marker

    prompt = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(engine_ablation.alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {engine_ablation.alphabet[-1]} -> {engine_ablation.alphabet[0]}, and {engine_ablation.alphabet[0]} -> {engine_ablation.alphabet[1]}.
Constraint: The input contains structural boundary markers ({marker}). DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_w_bounds}
Output: """
    return input_w_bounds, output_w_bounds, prompt, len(boundary_positions)


ABLATION_CONDITIONS = [
    {"Marker": "|", "Placement": "periodic",            "Ablation_Axis": "marker+placement baseline"},
    {"Marker": ";", "Placement": "periodic",            "Ablation_Axis": "marker identity"},
    {"Marker": "#", "Placement": "periodic",            "Ablation_Axis": "marker identity"},
    {"Marker": "|", "Placement": "random",               "Ablation_Axis": "placement regularity"},
    {"Marker": "|", "Placement": "periodic_fixed_count", "Ablation_Axis": "density vs fixed count"},
]

def run_ablation_trial(raw_input, condition, N, trial, rng):
    marker, placement = condition["Marker"], condition["Placement"]
    inp, truth, prompt, n_boundaries = build_ablation_variant(
        raw_input, marker=marker, placement=placement, interval=4, fixed_count=4, rng=rng
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    max_tokens_needed = int(N * 1.5) + 20 + n_boundaries
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = full_response[len(prompt):].strip()
    pred = pred.split("Explanation")[0].split("\n")[0].strip()

    clean_pred = pred.replace(marker, "").replace(" ", "").replace("\n", "")
    clean_truth = truth.replace(marker, "")
    clean_pred = clean_pred[:len(clean_truth)]

    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0
                      for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0

    return {
        "Marker": marker, "Placement": placement, "Ablation_Axis": condition["Ablation_Axis"],
        "Seq_Length": N, "Trial": trial, "N_Boundaries": n_boundaries,
        "TAR": tar, "SAR": sar,
    }


ablation_tasks = [
    (cond, N, t)
    for cond in ABLATION_CONDITIONS
    for N in ABLATION_SEQUENCE_LENGTHS
    for t in range(ABLATION_TRIALS)
]
print(f"\nStarting boundary ablation sweep: {len(ablation_tasks)} evaluations...")

ablation_results = []
for (cond, N, t) in tqdm(ablation_tasks, desc="Boundary ablation"):
    local_seed = 70_000 * N + 1000 * ABLATION_CONDITIONS.index(cond) + t
    rng = random.Random(local_seed)
    raw_input = engine_ablation.generate_raw_input(N, rng=rng)
    try:
        result = run_ablation_trial(raw_input, cond, N, t, rng)
        ablation_results.append(result)
    except Exception as exc:
        print(f"Task failed: {cond}, N={N}, trial={t} | Error: {exc}")

ablation_df = pd.DataFrame(ablation_results)
ablation_csv = os.path.join(RESULTS_DIR, "boundary_ablation_grid.csv")
ablation_df.to_csv(ablation_csv, index=False)
print(f"\nAblation sweep complete. Saved to '{ablation_csv}'")

summary = ablation_df.groupby(["Ablation_Axis", "Marker", "Placement", "Seq_Length"])[["TAR", "SAR"]].mean().reset_index()
print("\n--- MEAN ACCURACY BY ABLATION CONDITION ---")
print(summary.round(4).to_string(index=False))


# ==== ABLATION ANALYSIS ====

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


# ==== ADDITION GRID ====

import pandas as pd
import numpy as np
import random
import itertools
import torch
import os
from tqdm import tqdm

class AdditionTaskEngine:
    def _sample_pair(self, length, rng):
        for _ in range(200):
            a = [rng.randrange(10) for _ in range(length)]
            b = [rng.randrange(10) for _ in range(length)]
            if a[0] == 0: a[0] = rng.randrange(1, 10)
            if b[0] == 0: b[0] = rng.randrange(1, 10)
            carry = 0
            total = []
            for i in range(length - 1, -1, -1):
                s = a[i] + b[i] + carry
                total.append(s % 10)
                carry = s // 10
            if carry == 0:
                return a, b, list(reversed(total))
        raise RuntimeError(f"failed to sample a non-overflowing {length}-digit pair")

    def build_prompt(self, length, interval, rng=None):
        rng = rng or random
        a, b, total = self._sample_pair(length, rng)
        a_str = "".join(map(str, a))
        b_str = "".join(map(str, b))
        sum_str = "".join(map(str, total))

        combined = list(a_str + b_str)
        bounded = []
        for i, d in enumerate(combined):
            bounded.append(d)
            if interval > 0 and (i + 1) % interval == 0 and i != len(combined) - 1:
                bounded.append("|")
        input_str = "".join(bounded)

        prompt = f"""You are a deterministic arithmetic calculator.
Task: Add two {length}-digit numbers, given as a single digit stream: the first {length} digits are the first number, the next {length} digits are the second number.
Constraint: The input may contain vertical bars (|) as structural boundaries -- they carry no numeric meaning, ignore them when computing.
Output ONLY the {length}-digit sum (zero-padded to {length} digits if needed). No spaces, no explanations.

Input: {input_str}
Output: """
        return input_str, sum_str, prompt


ADD_SEQUENCE_LENGTHS = [4, 8, 16, 32, 64]
ADD_PERIODIC_INTERVALS = [0, 4, 8, 16]
ADD_TRIALS = 20

engine_add = AdditionTaskEngine()

def run_addition_trial(task_params):
    N, k, t = task_params
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 90_000 * N + 100 * k + t
    rng = random.Random(local_seed)

    inp, truth, prompt = engine_add.build_prompt(N, interval=k, rng=rng)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    max_tokens_needed = N + 20
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = full_response[len(prompt):].strip()
    pred = pred.split("Explanation")[0].split("\n")[0].strip()

    clean_pred = "".join(ch for ch in pred if ch.isdigit())[:len(truth)]
    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == truth[i] else 0 for i in range(len(truth))]
    tar = sum(binary_series) / len(truth) if truth else 0.0
    sar = 1.0 if (len(binary_series) == len(truth) and sum(binary_series) == len(binary_series)) else 0.0
    first_error = binary_series.index(0) if 0 in binary_series else len(binary_series)

    return {
        "Method": method, "Interval": k, "Seq_Length": N, "Trial": t,
        "TAR": tar, "SAR": sar, "First_Error_Pos": first_error,
        "Input_Given": inp, "Ground_Truth": truth, "LLM_Output": clean_pred,
    }

addition_tasks = [
    (N, k, t)
    for N in ADD_SEQUENCE_LENGTHS
    for k in ADD_PERIODIC_INTERVALS
    for t in range(ADD_TRIALS)
    if k == 0 or k < 2 * N
]
print(f"\nStarting integer-addition grid search: {len(addition_tasks)} evaluations...")

addition_results = []
for task in tqdm(addition_tasks, desc="Addition"):
    try:
        addition_results.append(run_addition_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

addition_df = pd.DataFrame(addition_results)
addition_csv = os.path.join(RESULTS_DIR, "addition_grid.csv")
addition_df.to_csv(addition_csv, index=False)
print(f"\nAddition grid search complete. Saved to '{addition_csv}'")

summary = addition_df.groupby(["Seq_Length", "Method"])[["TAR", "SAR"]].mean().reset_index()
print("\n--- AVERAGE ACCURACY BY LENGTH & METHOD ---")
print(summary.to_string(index=False))


# ==== ADDITION ANALYSIS ====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
import os

df = pd.read_csv(os.path.join(RESULTS_DIR, "addition_grid.csv"))
df['Inv_K'] = df['Interval'].apply(lambda k: 0.0 if k == 0 else 1.0/k)
df = df[df['Seq_Length'] >= 8]

summary = df.groupby(['Seq_Length', 'Inv_K'])[['TAR','SAR']].agg(['mean','std'])
summary.columns = ['_'.join(c) for c in summary.columns]
summary = summary.reset_index().sort_values(['Seq_Length','Inv_K'])
summary[['TAR_std','SAR_std']] = summary[['TAR_std','SAR_std']].fillna(0.0)

seq_lengths = sorted(summary['Seq_Length'].unique())
sns.set_theme(style="whitegrid", font_scale=1.1)
fig, axes = plt.subplots(2, max(len(seq_lengths),1), figsize=(4*max(len(seq_lengths),1), 10), sharex=True, sharey=True, squeeze=False)

regression_results = []
for row, (metric, ylabel) in enumerate([('TAR','Total Token Accuracy (TAR)'), ('SAR','Sequence Accuracy Rate (SAR)')]):
    for i, N in enumerate(seq_lengths):
        ax = axes[row][i]
        sub = summary[summary['Seq_Length']==N]
        if len(sub) >= 2:
            slope, intercept, r, p, se = linregress(sub['Inv_K'], sub[f'{metric}_mean'])
            regression_results.append(dict(Metric=metric, Seq_Length=N, Slope=slope, R_squared=r**2))
            ax.errorbar(sub['Inv_K'], sub[f'{metric}_mean'], yerr=sub[f'{metric}_std'], fmt='o',
                        color='#2b83ba', markersize=8, capsize=4, linestyle='none')
            x_fit = np.linspace(0, sub['Inv_K'].max()*1.1, 100)
            ax.plot(x_fit, slope*x_fit+intercept, linestyle='--', color='#2b83ba', alpha=0.6)
        ax.set_title(f'N={N}', fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if i == 0: ax.set_ylabel(ylabel)
        if row==1 and i==len(seq_lengths)//2: ax.set_xlabel('$1/k$')

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'addition_scaling_law.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {plot_path}")
plt.show()

results_df = pd.DataFrame(regression_results)
print(results_df.round(4).to_string(index=False))
results_df.to_csv(os.path.join(RESULTS_DIR, "addition_regression.csv"), index=False)

sk_results = []
for method, sub_M in df.groupby('Method'):
    for interval, sub_I in sub_M.groupby('Interval'):
        per_N = sub_I.groupby('Seq_Length')['SAR'].mean().reset_index()
        if len(per_N) < 3:
            continue
        beta0, alpha, n_star = fit_sk(per_N['Seq_Length'], per_N['SAR'])
        sk_results.append(dict(Method=method, Interval=interval, Beta0=beta0, Alpha=alpha, N_star=n_star))

sk_df = pd.DataFrame(sk_results)
print("\n--- ADDITION TASK: SK MODEL FIT ---")
print(sk_df.round(4).to_string(index=False))
sk_df.to_csv(os.path.join(RESULTS_DIR, "addition_sk_fit.csv"), index=False)


# ==== ROBUSTNESS GRID ====

import pandas as pd
import numpy as np
import random
import torch
import os
from tqdm import tqdm

def prompt_template_original(alphabet, input_str):
    return f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {alphabet[-1]} -> {alphabet[0]}, and {alphabet[0]} -> {alphabet[1]}.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_str}
Output: """

def prompt_template_paraphrase_1(alphabet, input_str):
    return f"""You perform one job only: shifting letters forward in a fixed alphabet.
Letters in use: {', '.join(alphabet)}
For every letter you see, write the next letter in this list instead (wrapping from {alphabet[-1]} back to {alphabet[0]}).
Some characters in the input are pipes (|); these are dividers, not letters -- reproduce them unchanged in your answer.
Respond with the transformed string only. Do not add spaces or commentary.

Sequence: {input_str}
Result: """

def prompt_template_paraphrase_2(alphabet, input_str):
    return f"""Apply a fixed substitution cipher: {' -> '.join(alphabet[1:] + alphabet[:1])} -> {alphabet[0]}.
In plain terms, replace each symbol with the one that follows it in {', '.join(alphabet)}, cycling back to the start after the last symbol.
Any "|" characters present must be left exactly where they are -- they are not part of the alphabet.
Give only the resulting string, with no spaces and no explanation of your reasoning.

>>> {input_str}
<<< """

PROMPT_TEMPLATES = {
    "original": prompt_template_original,
    "paraphrase_1": prompt_template_paraphrase_1,
    "paraphrase_2": prompt_template_paraphrase_2,
}

ROBUST_ALPHABET_SIZE = 2
ROBUST_SEQUENCE_LENGTHS = [8, 16, 32]
ROBUST_INTERVAL = 4
ROBUST_TRIALS = 15

engine_robust = TaskEngine(alphabet_size=ROBUST_ALPHABET_SIZE)

def build_robust_variant(raw_input, interval):
    input_w_bounds, output_w_bounds = "", ""
    for i, char in enumerate(raw_input):
        shifted = engine_robust.alphabet[(engine_robust.char_to_idx[char] + 1) % ROBUST_ALPHABET_SIZE]
        input_w_bounds += char
        output_w_bounds += shifted
        if interval > 0 and (i + 1) % interval == 0 and i != len(raw_input) - 1:
            input_w_bounds += "|"
            output_w_bounds += "|"
    return input_w_bounds, output_w_bounds

def run_robust_trial(template_name, N, k, t):
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 110_000 * N + 100 * k + t
    rng = random.Random(local_seed)
    raw_input = engine_robust.generate_raw_input(N, rng=rng)
    input_str, truth = build_robust_variant(raw_input, interval=k)
    prompt = PROMPT_TEMPLATES[template_name](engine_robust.alphabet, input_str)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    max_tokens_needed = int(N * 1.5) + 20
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = full_response[len(prompt):].strip()
    pred = pred.split("Explanation")[0].split("\n")[0].strip()

    clean_pred = pred.replace("|", "").replace(" ", "").replace("\n", "")
    clean_truth = truth.replace("|", "")
    clean_pred = clean_pred[:len(clean_truth)]
    binary_series = [1 if i < len(clean_pred) and clean_pred[i]==clean_truth[i] else 0 for i in range(len(clean_truth))]
    tar = sum(binary_series)/len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series)==len(clean_truth) and sum(binary_series)==len(binary_series)) else 0.0

    return dict(Template=template_name, Method=method, Interval=k, Seq_Length=N, Trial=t, TAR=tar, SAR=sar)

robust_tasks = [
    (tmpl, N, k, t)
    for tmpl in PROMPT_TEMPLATES
    for N in ROBUST_SEQUENCE_LENGTHS
    for k in [0, ROBUST_INTERVAL]
    for t in range(ROBUST_TRIALS)
]
print(f"\nStarting prompt-robustness sweep: {len(robust_tasks)} evaluations...")

robust_results = []
for (tmpl, N, k, t) in tqdm(robust_tasks, desc="Prompt robustness"):
    try:
        robust_results.append(run_robust_trial(tmpl, N, k, t))
    except Exception as exc:
        print(f"Task failed: {(tmpl,N,k,t)} | Error: {exc}")

robust_df = pd.DataFrame(robust_results)
robust_csv = os.path.join(RESULTS_DIR, "prompt_robustness_grid.csv")
robust_df.to_csv(robust_csv, index=False)
print(f"\nPrompt-robustness sweep complete. Saved to '{robust_csv}'")

summary = robust_df.groupby(['Template','Method','Seq_Length'])[['TAR','SAR']].mean().reset_index()
print("\n--- MEAN ACCURACY BY TEMPLATE, METHOD, LENGTH ---")
print(summary.round(4).to_string(index=False))


# ==== ROBUSTNESS ANALYSIS ====

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

robust_df = pd.read_csv(os.path.join(RESULTS_DIR, "prompt_robustness_grid.csv"))
summary = robust_df.groupby(['Template','Method','Seq_Length'])[['TAR','SAR']].mean().reset_index()

sns.set_theme(style="whitegrid", font_scale=1.1)
fig, axes = plt.subplots(1, 2, figsize=(13,5))
templates = summary['Template'].unique()
palette = sns.color_palette("bright", n_colors=len(templates))
colors = {t:c for t,c in zip(templates, palette)}

for ax, metric in zip(axes, ['TAR','SAR']):
    for template in templates:
        for method, marker, ls in [('Base','o','--'), ('Cellular_Bounds','s','-')]:
            sub = summary[(summary['Template']==template) & (summary['Method']==method)].sort_values('Seq_Length')
            if len(sub):
                ax.plot(sub['Seq_Length'], sub[metric], marker=marker, linestyle=ls,
                        color=colors[template], label=f'{template} ({method})', alpha=0.85)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Sequence Length $N$')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} across prompt phrasings', fontweight='bold')
    ax.set_ylim(-0.05,1.05)
    ax.grid(True, alpha=0.3)

axes[0].legend(fontsize=7, loc='upper right')
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'prompt_robustness.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {plot_path}")
plt.show()

gap = summary.pivot_table(index=['Template','Seq_Length'], columns='Method', values='TAR').reset_index()
if 'Base' in gap.columns and 'Cellular_Bounds' in gap.columns:
    gap['Gap'] = gap['Cellular_Bounds'] - gap['Base']
    print("\n--- TAR GAP (Cellular - Base) BY TEMPLATE ---")
    print(gap.round(4).to_string(index=False))
    print(f"\nPositive gap (boundaries help) in {int((gap['Gap']>0).sum())}/{len(gap)} (template, N) combinations.")

