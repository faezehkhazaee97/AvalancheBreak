import matplotlib
matplotlib.use("Agg")

# ============================================================
# INTEGER ADDITION GRID, RE-RUN WITH THE PROPER CHAT TEMPLATE
# ============================================================
# Same rationale as run_core_templated.py: rerun the direct-output integer
# addition grid (same lengths, intervals, trial count, seeds as
# addition_grid.csv) with ONLY the prompting method changed
# (tokenizer.apply_chat_template instead of raw tokenizer(prompt)), saved
# under a distinct "_templated" filename for direct comparison against the
# raw-prompt result.

import os

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"Results directory: {RESULTS_DIR}/")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random, re
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
from scipy.stats import linregress
from scipy.optimize import curve_fit
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

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = {kk: vv.to(model.device) for kk, vv in enc.items()}

    max_tokens_needed = N + 20
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_only = outputs[0][inputs["input_ids"].shape[1]:]
    pred = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
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
print(f"\nStarting TEMPLATED integer-addition grid: {len(addition_tasks)} evaluations...")

addition_results = []
for task in tqdm(addition_tasks, desc="Addition (templated)"):
    try:
        addition_results.append(run_addition_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

addition_df = pd.DataFrame(addition_results)
addition_csv = os.path.join(RESULTS_DIR, "addition_grid_templated.csv")
addition_df.to_csv(addition_csv, index=False)
print(f"\nAddition grid search complete. Saved to '{addition_csv}'")

summary = addition_df.groupby(["Seq_Length", "Method"])[["TAR", "SAR"]].mean().reset_index()
print("\n--- AVERAGE ACCURACY BY LENGTH & METHOD (templated) ---")
print(summary.to_string(index=False))


# ==== REGRESSION ====

df = addition_df.copy()
df['Inv_K'] = df['Interval'].apply(lambda k: 0.0 if k == 0 else 1.0/k)
df = df[df['Seq_Length'] >= 8]

reg_summary = df.groupby(['Seq_Length', 'Inv_K'])[['TAR','SAR']].agg(['mean','std'])
reg_summary.columns = ['_'.join(c) for c in reg_summary.columns]
reg_summary = reg_summary.reset_index().sort_values(['Seq_Length','Inv_K'])
reg_summary[['TAR_std','SAR_std']] = reg_summary[['TAR_std','SAR_std']].fillna(0.0)

regression_results = []
for metric in ['TAR', 'SAR']:
    for N, sub in reg_summary.groupby('Seq_Length'):
        if len(sub) >= 2:
            slope, intercept, r, p, se = linregress(sub['Inv_K'], sub[f'{metric}_mean'])
            regression_results.append(dict(Metric=metric, Seq_Length=N, Slope=slope, R_squared=r**2))

results_df = pd.DataFrame(regression_results)
results_df.to_csv(os.path.join(RESULTS_DIR, "addition_regression_templated.csv"), index=False)
print("\n--- SCALING LAW REGRESSION (templated) ---")
print(results_df.round(4).to_string(index=False))


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
for method, sub_M in df.groupby('Method'):
    for interval, sub_I in sub_M.groupby('Interval'):
        per_N = sub_I.groupby('Seq_Length')['SAR'].mean().reset_index()
        if len(per_N) < 3:
            continue
        beta0, alpha, n_star, fit_valid = fit_sk(per_N['Seq_Length'], per_N['SAR'])
        sk_results.append(dict(Method=method, Interval=interval, Beta0=beta0, Alpha=alpha,
                                N_star=n_star, Fit_Valid=fit_valid))

sk_df = pd.DataFrame(sk_results)
print("\n--- ADDITION TASK: SK MODEL FIT (templated) ---")
print(sk_df.round(4).to_string(index=False))
sk_df.to_csv(os.path.join(RESULTS_DIR, "addition_sk_fit_templated.csv"), index=False)

print("\nTemplated integer-addition grid rerun complete.")
