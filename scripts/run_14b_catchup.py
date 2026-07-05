import matplotlib
matplotlib.use("Agg")

# ============================================================
# 14B CATCH-UP
# ============================================================
# The 14B run executed a frozen snapshot of the notebook (LLM-Focus-14B-
# input.ipynb) that predates almost every analysis added after the core
# cyclic-shift / running-sum-mod-M thermodynamics grids. This script
# reproduces, in order, everything that's missing: boundary ablation (+
# significance), integer addition (+ regression/SK-fit), the LSD-first
# carry-direction test, addition-with-a-scratchpad, Pauli string
# multiplication, and prompt-wording robustness. Error-containment,
# significance, token-overhead, and held-out validation for the
# thermodynamics/modsum grids were already produced by the 14B run itself;
# clustering-vs-density was backfilled separately (pure CPU, no GPU needed)
# by run_postprocess_catchup.py.

import os

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"Results directory: {RESULTS_DIR}/")

# ==== MODEL LOADER ====

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

# ==== SHARED TASK ENGINE (cyclic-shift; reused by ablation & robustness) ====

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


# ==== SK SPIN-GLASS FIT (needed by addition_sk_fit below) ====

def sk_model(N, beta0, alpha):
    N = np.asarray(N, dtype=np.float64)
    return np.exp(-beta0 * N * np.power(alpha, N - 1))

def fit_sk(N_vals, sar_vals):
    from scipy.optimize import curve_fit
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


import pandas as pd
import itertools
import re
from tqdm import tqdm
from scipy.stats import wilcoxon, binomtest, linregress
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# BOUNDARY ABLATION (marker identity, placement, density vs count)
# ============================================================

ABLATION_ALPHABET_SIZE = 2
ABLATION_SEQUENCE_LENGTHS = [8, 16, 32, 64]
ABLATION_TRIALS = 20

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
        "Shared_Sequence": "".join(raw_input), "TAR": tar, "SAR": sar,
    }


shared_sequences = {}
for N in ABLATION_SEQUENCE_LENGTHS:
    for t in range(ABLATION_TRIALS):
        rng = random.Random(70_000 * N + t)
        shared_sequences[(N, t)] = engine_ablation.generate_raw_input(N, rng=rng)
print(f"Pregenerated {len(shared_sequences)} shared raw sequences.")

ablation_tasks = [
    (cond, N, t)
    for cond in ABLATION_CONDITIONS
    for N in ABLATION_SEQUENCE_LENGTHS
    for t in range(ABLATION_TRIALS)
]
print(f"\nStarting boundary ablation sweep: {len(ablation_tasks)} evaluations...")

ablation_results = []
for (cond, N, t) in tqdm(ablation_tasks, desc="Boundary ablation"):
    cond_seed = 70_000 * N + 1000 * ABLATION_CONDITIONS.index(cond) + t
    rng = random.Random(cond_seed)
    raw_input = shared_sequences[(N, t)]
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

# --- Ablation plot vs Base reference ---
base_df = pd.read_csv(os.path.join(RESULTS_DIR, "thermodynamics_grid.csv"))
base_ref = (
    base_df[(base_df["Method"] == "Base") & (base_df["Alphabet_Size"] == ABLATION_ALPHABET_SIZE)]
    .groupby("Seq_Length")[["TAR", "SAR"]]
    .mean()
    .reset_index()
)
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
plt.close(fig)
print(f"\nSaved plot to {plot_path}")

# --- Ablation significance testing ---
def mcnemar_exact_pvalue(b, c):
    if b + c == 0:
        return 1.0
    return binomtest(min(b, c), b + c, 0.5).pvalue

def paired_ablation_test(df, cond_a, cond_b, label_a, label_b):
    a_df = df[(df['Marker'] == cond_a[0]) & (df['Placement'] == cond_a[1])]
    b_df = df[(df['Marker'] == cond_b[0]) & (df['Placement'] == cond_b[1])]
    results = []
    for N in sorted(df['Seq_Length'].unique()):
        a_N = a_df[a_df['Seq_Length'] == N]
        b_N = b_df[b_df['Seq_Length'] == N]
        merged = a_N[['Trial', 'TAR', 'SAR']].merge(b_N[['Trial', 'TAR', 'SAR']], on='Trial', suffixes=('_a', '_b'))
        if len(merged) < 2:
            continue
        diffs = merged['TAR_b'] - merged['TAR_a']
        p_tar = wilcoxon(merged['TAR_b'], merged['TAR_a'])[1] if np.any(diffs != 0) else 1.0
        a_pass = merged['SAR_a'] >= 0.5
        b_pass = merged['SAR_b'] >= 0.5
        b_disc = int((a_pass & ~b_pass).sum())
        c_disc = int((~a_pass & b_pass).sum())
        p_sar = mcnemar_exact_pvalue(b_disc, c_disc)
        results.append({
            'Comparison': f'{label_a} vs {label_b}', 'Seq_Length': N, 'N_pairs': len(merged),
            'Mean_TAR_A': merged['TAR_a'].mean(), 'Mean_TAR_B': merged['TAR_b'].mean(),
            'P_value_TAR': p_tar, 'P_value_SAR': p_sar,
            'Significant_TAR_p05': p_tar < 0.05, 'Significant_SAR_p05': p_sar < 0.05,
        })
    return pd.DataFrame(results)

comparisons = [
    (("|", "periodic"), (";", "periodic"), "| baseline", "; marker"),
    (("|", "periodic"), ("#", "periodic"), "| baseline", "# marker"),
    (("|", "periodic"), ("|", "random"), "| periodic", "| random"),
    (("|", "periodic"), ("|", "periodic_fixed_count"), "| density", "| fixed-count"),
]
all_results = []
for cond_a, cond_b, label_a, label_b in comparisons:
    res = paired_ablation_test(ablation_df, cond_a, cond_b, label_a, label_b)
    all_results.append(res)
    print(f"\n--- {label_a} vs {label_b} ---")
    print(res.round(4).to_string(index=False))

sig_df = pd.concat(all_results, ignore_index=True)
sig_df.to_csv(os.path.join(RESULTS_DIR, "boundary_ablation_significance.csv"), index=False)
n_sig_tar = sig_df['Significant_TAR_p05'].sum()
n_sig_sar = sig_df['Significant_SAR_p05'].sum()
print(f"\nSignificant at p<0.05 across all ablation comparisons: TAR {n_sig_tar}/{len(sig_df)}, SAR {n_sig_sar}/{len(sig_df)}")


# ============================================================
# INTEGER ADDITION TASK
# ============================================================

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

        combined = list(a_str) + ["+"] + list(b_str)
        bounded = []
        for i, d in enumerate(combined):
            bounded.append(d)
            if interval > 0 and (i + 1) % interval == 0 and i != len(combined) - 1:
                bounded.append("|")
        input_str = "".join(bounded)

        prompt = f"""You are a deterministic arithmetic calculator.
Task: Add two {length}-digit numbers.
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

    match = re.match(r'^(\d+)', pred)
    clean_pred = match.group(1)[:len(truth)] if match else ""
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
    if k == 0 or k < 2 * N + 1
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

# --- Addition regression + SK fit ---
df = addition_df.copy()
df['Inv_K'] = df['Interval'].apply(lambda k: 0.0 if k == 0 else 1.0/k)
df = df[df['Seq_Length'] >= 8]

reg_summary = df.groupby(['Seq_Length', 'Inv_K'])[['TAR','SAR']].agg(['mean','std'])
reg_summary.columns = ['_'.join(c) for c in reg_summary.columns]
reg_summary = reg_summary.reset_index().sort_values(['Seq_Length','Inv_K'])
reg_summary[['TAR_std','SAR_std']] = reg_summary[['TAR_std','SAR_std']].fillna(0.0)

seq_lengths_add = sorted(reg_summary['Seq_Length'].unique())
fig, axes = plt.subplots(2, max(len(seq_lengths_add),1), figsize=(4*max(len(seq_lengths_add),1), 10), sharex=True, sharey=True, squeeze=False)

regression_results = []
for row, (metric, ylabel) in enumerate([('TAR','Total Token Accuracy (TAR)'), ('SAR','Sequence Accuracy Rate (SAR)')]):
    for i, N in enumerate(seq_lengths_add):
        ax = axes[row][i]
        sub = reg_summary[reg_summary['Seq_Length']==N]
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
        if row==1 and i==len(seq_lengths_add)//2: ax.set_xlabel('$1/k$')

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'addition_scaling_law.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved plot to {plot_path}")

results_df = pd.DataFrame(regression_results)
print(results_df.round(4).to_string(index=False))
results_df.to_csv(os.path.join(RESULTS_DIR, "addition_regression.csv"), index=False)

sk_results = []
for method, sub_M in df.groupby('Method'):
    for interval, sub_I in sub_M.groupby('Interval'):
        per_N = sub_I.groupby('Seq_Length')['SAR'].mean().reset_index()
        if len(per_N) < 3:
            continue
        beta0, alpha, n_star, fit_valid = fit_sk(per_N['Seq_Length'], per_N['SAR'])
        sk_results.append(dict(Method=method, Interval=interval, Beta0=beta0, Alpha=alpha,
                                N_star=n_star, Fit_Valid=fit_valid))

sk_df = pd.DataFrame(sk_results)
n_invalid = (~sk_df['Fit_Valid']).sum() if len(sk_df) else 0
print("\n--- ADDITION TASK: SK MODEL FIT ---")
print(sk_df.round(4).to_string(index=False))
if n_invalid:
    print(f"\nWARNING: {n_invalid}/{len(sk_df)} fits are INVALID (degenerate SAR input).")
sk_df.to_csv(os.path.join(RESULTS_DIR, "addition_sk_fit.csv"), index=False)


# ============================================================
# ADDITION, CARRY-DIRECTION HYPOTHESIS TEST (LSD-first)
# ============================================================

class AdditionTaskEngineLSD:
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
        a_str = "".join(map(str, a))[::-1]
        b_str = "".join(map(str, b))[::-1]
        sum_str = "".join(map(str, total))[::-1]

        combined = list(a_str) + ["+"] + list(b_str)
        bounded = []
        for i, d in enumerate(combined):
            bounded.append(d)
            if interval > 0 and (i + 1) % interval == 0 and i != len(combined) - 1:
                bounded.append("|")
        input_str = "".join(bounded)

        prompt = f"""You are a deterministic arithmetic calculator.
Task: Add two {length}-digit numbers. Both input numbers and the expected output are written LEAST-SIGNIFICANT-DIGIT FIRST (reversed order, i.e. the ones digit comes first).
Constraint: The input may contain vertical bars (|) as structural boundaries -- they carry no numeric meaning, ignore them when computing.
Output ONLY the {length}-digit sum, also least-significant-digit-first. No spaces, no explanations.

Input: {input_str}
Output: """
        return input_str, sum_str, prompt


ADD_LSD_SEQUENCE_LENGTHS = [4, 8, 16, 32, 64]
ADD_LSD_PERIODIC_INTERVALS = [0, 4, 8, 16]
ADD_LSD_TRIALS = 20

engine_add_lsd = AdditionTaskEngineLSD()

def run_addition_lsd_trial(task_params):
    N, k, t = task_params
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 130_000 * N + 100 * k + t
    rng = random.Random(local_seed)

    inp, truth, prompt = engine_add_lsd.build_prompt(N, interval=k, rng=rng)

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

    match = re.match(r'^(\d+)', pred)
    clean_pred = match.group(1)[:len(truth)] if match else ""

    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == truth[i] else 0 for i in range(len(truth))]
    tar = sum(binary_series) / len(truth) if truth else 0.0
    sar = 1.0 if (len(binary_series) == len(truth) and sum(binary_series) == len(binary_series)) else 0.0
    first_error = binary_series.index(0) if 0 in binary_series else len(binary_series)

    return {
        "Method": method, "Interval": k, "Seq_Length": N, "Trial": t,
        "TAR": tar, "SAR": sar, "First_Error_Pos": first_error,
        "Input_Given": inp, "Ground_Truth": truth, "LLM_Output": clean_pred,
    }

addition_lsd_tasks = [
    (N, k, t)
    for N in ADD_LSD_SEQUENCE_LENGTHS
    for k in ADD_LSD_PERIODIC_INTERVALS
    for t in range(ADD_LSD_TRIALS)
    if k == 0 or k < 2 * N + 1
]
print(f"\nStarting LSD-first addition grid search: {len(addition_lsd_tasks)} evaluations...")

addition_lsd_results = []
for task in tqdm(addition_lsd_tasks, desc="Addition (LSD-first)"):
    try:
        addition_lsd_results.append(run_addition_lsd_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

addition_lsd_df = pd.DataFrame(addition_lsd_results)
addition_lsd_csv = os.path.join(RESULTS_DIR, "addition_lsd_grid.csv")
addition_lsd_df.to_csv(addition_lsd_csv, index=False)
print(f"\nLSD-first addition grid search complete. Saved to '{addition_lsd_csv}'")

summary = addition_lsd_df.groupby(["Seq_Length", "Method"])[["TAR", "SAR"]].mean().reset_index()
print("\n--- AVERAGE ACCURACY BY LENGTH & METHOD (LSD-first) ---")
print(summary.to_string(index=False))

pivot = summary.pivot(index="Seq_Length", columns="Method", values="TAR")
if "Base" in pivot.columns and "Cellular_Bounds" in pivot.columns:
    pivot["Cellular_minus_Base"] = pivot["Cellular_Bounds"] - pivot["Base"]
    print("\n--- DOES CELLULAR STILL SCORE WORSE THAN BASE, ONCE DIGIT ORDER MATCHES CARRY DIRECTION? ---")
    print(pivot.round(4).to_string())
    n_worse = int((pivot["Cellular_minus_Base"] < 0).sum())
    print(f"\nCellular still worse than Base at {n_worse}/{len(pivot)} tested lengths.")

normal_summary = addition_df.groupby(["Seq_Length", "Method"])["TAR"].mean().unstack()
normal_summary["Relative_Retention"] = normal_summary["Cellular_Bounds"] / normal_summary["Base"]

lsd_summary = summary.pivot(index="Seq_Length", columns="Method", values="TAR")
lsd_summary["Relative_Retention"] = lsd_summary["Cellular_Bounds"] / lsd_summary["Base"]

comparison = pd.DataFrame({
    "Normal_Order_Retention": normal_summary["Relative_Retention"],
    "LSD_First_Retention": lsd_summary["Relative_Retention"],
})
comparison["Improved_With_LSD"] = comparison["LSD_First_Retention"] > comparison["Normal_Order_Retention"]
print("\n--- RELATIVE RETENTION (Cellular/Base TAR ratio): normal order vs LSD-first ---")
print(comparison.round(3).to_string())
n_improved = int(comparison["Improved_With_LSD"].sum())
print(f"\nLSD-first retains a HIGHER fraction of Base accuracy at {n_improved}/{len(comparison)} tested lengths.")
comparison.to_csv(os.path.join(RESULTS_DIR, "addition_carry_direction_comparison.csv"))


# ============================================================
# ADDITION WITH A SCRATCHPAD
# ============================================================

class ScratchpadAdditionTaskEngine:
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

        combined = list(a_str) + ["+"] + list(b_str)
        bounded = []
        for i, d in enumerate(combined):
            bounded.append(d)
            if interval > 0 and (i + 1) % interval == 0 and i != len(combined) - 1:
                bounded.append("|")
        input_str = "".join(bounded)

        prompt = f"""You are a careful arithmetic assistant.
Task: Add two {length}-digit numbers.
Constraint: The input may contain vertical bars (|) as structural boundaries -- they carry no numeric meaning, ignore them when computing.
First, work through the addition step by step from the rightmost (least-significant) digit to the left, tracking the carry at each step. Then, on its own line, write exactly:
Final Answer: <the {length}-digit sum, zero-padded if needed>

Input: {input_str}
"""
        return input_str, sum_str, prompt


SCRATCH_SEQUENCE_LENGTHS = [4, 8, 16]
SCRATCH_PERIODIC_INTERVALS = [0, 4, 8, 16]
SCRATCH_TRIALS = 15

engine_scratch = ScratchpadAdditionTaskEngine()

def run_scratchpad_trial(task_params):
    N, k, t = task_params
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 150_000 * N + 100 * k + t
    rng = random.Random(local_seed)

    inp, truth, prompt = engine_scratch.build_prompt(N, interval=k, rng=rng)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    max_tokens_needed = N * 20 + 60
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred_full = full_response[len(prompt):]

    match = re.search(r'Final Answer:\s*(\d+)', pred_full)
    clean_pred = match.group(1)[:len(truth)] if match else ""

    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == truth[i] else 0 for i in range(len(truth))]
    tar = sum(binary_series) / len(truth) if truth else 0.0
    sar = 1.0 if (len(binary_series) == len(truth) and sum(binary_series) == len(binary_series)) else 0.0
    first_error = binary_series.index(0) if 0 in binary_series else len(binary_series)

    return {
        "Method": method, "Interval": k, "Seq_Length": N, "Trial": t,
        "TAR": tar, "SAR": sar, "First_Error_Pos": first_error,
        "Input_Given": inp, "Ground_Truth": truth, "LLM_Output": clean_pred,
        "Found_Final_Answer_Line": match is not None,
    }

scratch_tasks = [
    (N, k, t)
    for N in SCRATCH_SEQUENCE_LENGTHS
    for k in SCRATCH_PERIODIC_INTERVALS
    for t in range(SCRATCH_TRIALS)
    if k == 0 or k < 2 * N + 1
]
print(f"\nStarting scratchpad addition grid search: {len(scratch_tasks)} evaluations...")

scratch_results = []
for task in tqdm(scratch_tasks, desc="Scratchpad addition"):
    try:
        scratch_results.append(run_scratchpad_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

scratch_df = pd.DataFrame(scratch_results)
scratch_csv = os.path.join(RESULTS_DIR, "addition_scratchpad_grid.csv")
scratch_df.to_csv(scratch_csv, index=False)
print(f"\nScratchpad addition grid search complete. Saved to '{scratch_csv}'")

summary = scratch_df.groupby(["Seq_Length", "Method"])[["TAR", "SAR"]].mean().reset_index()
print("\n--- AVERAGE ACCURACY BY LENGTH & METHOD (scratchpad) ---")
print(summary.to_string(index=False))
print(f"\nFound a clean 'Final Answer:' line in {scratch_df['Found_Final_Answer_Line'].mean()*100:.1f}% of responses.")

pivot = summary.pivot(index="Seq_Length", columns="Method", values="TAR")
if "Base" in pivot.columns and "Cellular_Bounds" in pivot.columns:
    pivot["Cellular_minus_Base"] = pivot["Cellular_Bounds"] - pivot["Base"]
    print("\n--- DOES CELLULAR STILL SCORE WORSE THAN BASE, WITH A SCRATCHPAD? ---")
    print(pivot.round(4).to_string())
    n_worse = int((pivot["Cellular_minus_Base"] < 0).sum())
    print(f"\nCellular still worse than Base at {n_worse}/{len(pivot)} tested lengths.")


# ============================================================
# PAULI STRING MULTIPLICATION
# ============================================================

class PauliStringTaskEngine:
    PAULIS = ["I", "X", "Y", "Z"]
    MULT_TABLE = {
        ("I", "I"): ("I", 0), ("I", "X"): ("X", 0), ("I", "Y"): ("Y", 0), ("I", "Z"): ("Z", 0),
        ("X", "I"): ("X", 0), ("Y", "I"): ("Y", 0), ("Z", "I"): ("Z", 0),
        ("X", "X"): ("I", 0), ("Y", "Y"): ("I", 0), ("Z", "Z"): ("I", 0),
        ("X", "Y"): ("Z", 1), ("Y", "Z"): ("X", 1), ("Z", "X"): ("Y", 1),
        ("Y", "X"): ("Z", 3), ("Z", "Y"): ("X", 3), ("X", "Z"): ("Y", 3),
    }
    PHASE_SYMBOLS = ["+", "i", "-", "j"]

    def sample_strings(self, length, rng):
        a = rng.choices(self.PAULIS, k=length)
        b = rng.choices(self.PAULIS, k=length)
        return a, b

    def multiply(self, a, b):
        result_ops = []
        total_phase = 0
        for pa, pb in zip(a, b):
            op, phase = self.MULT_TABLE[(pa, pb)]
            result_ops.append(op)
            total_phase = (total_phase + phase) % 4
        return "".join(result_ops), self.PHASE_SYMBOLS[total_phase]

    def build_prompt(self, length, interval, rng=None):
        rng = rng or random
        a, b = self.sample_strings(length, rng)
        result_string, phase_symbol = self.multiply(a, b)
        truth = result_string + phase_symbol

        a_str, b_str = "".join(a), "".join(b)
        combined = list(a_str) + ["*"] + list(b_str)
        bounded = []
        for i, d in enumerate(combined):
            bounded.append(d)
            if interval > 0 and (i + 1) % interval == 0 and i != len(combined) - 1:
                bounded.append("|")
        input_str = "".join(bounded)

        prompt = f"""You are a deterministic Pauli algebra calculator.
Task: multiply two {length}-operator Pauli strings, position by position.
Rules: I is the identity (I*P = P for any P). P*P = I for P in {{X, Y, Z}}. X*Y=iZ, Y*Z=iX, Z*X=iY. Y*X=-iZ, Z*Y=-iX, X*Z=-iY.
Alphabet: I, X, Y, Z. The symbol * marks where the first string ends and the second begins.
Constraint: The input may contain vertical bars (|) as structural boundaries -- they carry no meaning, ignore them when computing.
Output the resulting {length}-operator string, immediately followed by ONE phase symbol for the TOTAL accumulated phase across all positions (multiply every per-position phase together): '+' for +1, 'i' for +i, '-' for -1, 'j' for -i. No spaces, no explanations.

Input: {input_str}
Output: """
        return input_str, truth, prompt


PAULI_SEQUENCE_LENGTHS = [4, 8, 16, 32]
PAULI_PERIODIC_INTERVALS = [0, 4, 8, 16]
PAULI_TRIALS = 20

engine_pauli = PauliStringTaskEngine()

def run_pauli_trial(task_params):
    N, k, t = task_params
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 170_000 * N + 100 * k + t
    rng = random.Random(local_seed)

    inp, truth, prompt = engine_pauli.build_prompt(N, interval=k, rng=rng)

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

    match = re.match(r'^([IXYZ+ij-]+)', pred)
    clean_pred = match.group(1)[:len(truth)] if match else ""

    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == truth[i] else 0 for i in range(len(truth))]
    tar = sum(binary_series) / len(truth) if truth else 0.0
    sar = 1.0 if (len(binary_series) == len(truth) and sum(binary_series) == len(binary_series)) else 0.0
    first_error = binary_series.index(0) if 0 in binary_series else len(binary_series)

    string_correct = clean_pred[:len(truth) - 1] == truth[:-1] if len(clean_pred) >= len(truth) - 1 else False
    phase_correct = len(clean_pred) == len(truth) and clean_pred[-1] == truth[-1]

    return {
        "Method": method, "Interval": k, "Seq_Length": N, "Trial": t,
        "TAR": tar, "SAR": sar, "First_Error_Pos": first_error,
        "String_Correct": string_correct, "Phase_Correct": phase_correct,
        "Input_Given": inp, "Ground_Truth": truth, "LLM_Output": clean_pred,
    }

pauli_tasks = [
    (N, k, t)
    for N in PAULI_SEQUENCE_LENGTHS
    for k in PAULI_PERIODIC_INTERVALS
    for t in range(PAULI_TRIALS)
    if k == 0 or k < 2 * N + 1
]
print(f"\nStarting Pauli-string grid search: {len(pauli_tasks)} evaluations...")

pauli_results = []
for task in tqdm(pauli_tasks, desc="Pauli string"):
    try:
        pauli_results.append(run_pauli_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

pauli_df = pd.DataFrame(pauli_results)
pauli_csv = os.path.join(RESULTS_DIR, "pauli_string_grid.csv")
pauli_df.to_csv(pauli_csv, index=False)
print(f"\nPauli-string grid search complete. Saved to '{pauli_csv}'")

summary = pauli_df.groupby(["Seq_Length", "Method"])[["TAR", "SAR", "String_Correct", "Phase_Correct"]].mean().reset_index()
print("\n--- AVERAGE ACCURACY BY LENGTH & METHOD ---")
print(summary.to_string(index=False))


# ============================================================
# PROMPT-WORDING ROBUSTNESS
# ============================================================

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

print("\n14B catch-up complete.")
