import matplotlib
matplotlib.use("Agg")

# ============================================================
# WFR-UNDER-SAMPLING ROBUSTNESS CHECK (Job C)
# ============================================================
# Extends scripts/run_temperature_robustness_check.py (which only reported
# TAR/SAR under sampling) to the full well-formedness diagnostic suite --
# Well_Formed_Rate (WFR), Vocab_Valid_Rate (VVR), Complete_Rate (CR) --
# reusing tasks.common.scoring / tasks.common.metrics so the definitions
# match the rest of the study exactly, on the raw (untruncated)
# Raw_Model_Output the existing temperature check never saved.
#
# Question: the study elsewhere finds a NEGATIVE WFR delta for
# cyclic_shift under boundaries at greedy decoding (Cellular_Bounds is
# well-formed less often than Base) -- does that persist at T=0.2/0.7
# (near-greedy sampling), or is it a greedy-decoding artifact?
#
# Same grid as the original temperature check: cyclic_shift, alphabet=2,
# N in {16, 32, 64}, Base vs the single k=2 boundary condition (same
# density used for the causal-ablation/SK-fit headline numbers), 10
# trials/cell. T=0.0 (greedy) included as the reference point alongside
# the near-greedy T=0.2/0.7 conditions the reviewer asked about.

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join(REPO_ROOT, "results", MODEL_TAG, "wfr_decoding_robustness")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random
import numpy as np
import pandas as pd
from tqdm import tqdm

from tasks.common import scoring as scoring_mod
from tasks.common import metrics as metrics_mod

print(f"Loading {MODEL_ID} into VRAM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
)
model.eval()

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

TASK_NAME = "cyclic_shift"


class TaskEngine:
    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size
        self.alphabet = [chr(ord('A') + i) for i in range(alphabet_size)]
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}

    def generate_raw_input(self, length, rng=None):
        rng = rng or random
        return rng.choices(self.alphabet, k=length)

    def build_cellular_from_raw(self, raw_input, interval):
        input_w_bounds, output_w_bounds = "", ""
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


ALPHABET_SIZE = 2
SEQUENCE_LENGTHS = [16, 32, 64]
INTERVAL = 2
TEMPERATURES = [0.0, 0.2, 0.7]
TRIALS_T = 10

engine = TaskEngine(ALPHABET_SIZE)


def run_trial(N, method, temperature, t):
    local_seed = 10_000 * ALPHABET_SIZE + 100 * N + t  # same convention as thermodynamics_grid_templated_v2.csv
    rng = random.Random(local_seed)
    raw_input = engine.generate_raw_input(N, rng=rng)
    interval = INTERVAL if method == "Cellular_Bounds" else 0
    inp, truth, prompt = engine.build_cellular_from_raw(raw_input, interval=interval)

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in enc.items()}

    do_sample = temperature > 0.0
    gen_kwargs = dict(max_new_tokens=int(N * 1.5) + 20, pad_token_id=tokenizer.eos_token_id, do_sample=do_sample)
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    gen_only = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(gen_only, skip_special_tokens=True)

    result = scoring_mod.score(raw_output, truth)

    row = {
        "Temperature": temperature, "Method": method, "Interval": interval,
        "Seq_Length": N, "Alphabet_Size": ALPHABET_SIZE, "Trial": t,
        "Ground_Truth": result["clean_truth"], "LLM_Output": result["clean_pred"],
        "TAR": result["TAR"], "SAR": result["SAR"],
        "First_Error_Pos": result["First_Error_Pos"], "Raw_Model_Output": raw_output,
    }
    diag = metrics_mod.row_diagnostics(TASK_NAME, pd.Series(row))
    row.update(diag)
    return row


tasks = [(N, method, temp, t)
         for temp in TEMPERATURES
         for N in SEQUENCE_LENGTHS
         for method in ["Base", "Cellular_Bounds"]
         for t in range(TRIALS_T)]
print(f"\nStarting WFR decoding-robustness sweep: {len(tasks)} evaluations...")

results = []
for N, method, temp, t in tqdm(tasks, desc="WFR decoding robustness"):
    try:
        results.append(run_trial(N, method, temp, t))
    except Exception as exc:
        print(f"Task failed: N={N} method={method} temp={temp} t={t} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "wfr_decoding_robustness_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

print("\n--- TAR by temperature, length, method ---")
print(df.groupby(["Temperature", "Seq_Length", "Method"])["TAR"].mean().unstack().round(4).to_string())

print("\n--- WFR (Well_Formed_Rate) by temperature, method (pooled over length) ---")
wfr = df.groupby(["Temperature", "Method"])["Well_Formed"].mean().unstack()
print(wfr.round(4).to_string())
print("\nWFR delta (Cellular_Bounds - Base) by temperature:")
print((wfr["Cellular_Bounds"] - wfr["Base"]).round(4).to_string())

print("\n--- VVR (Vocab_Valid_Rate) by temperature, method ---")
vvr = df.groupby(["Temperature", "Method"])["Vocab_Valid"].mean().unstack()
print(vvr.round(4).to_string())

print("\n--- CR (Complete_Rate) by temperature, method ---")
cr = df.groupby(["Temperature", "Method"])["Complete"].mean().unstack()
print(cr.round(4).to_string())

summary_path = os.path.join(RESULTS_DIR, "wfr_decoding_robustness_summary.csv")
summary_rows = []
for temp, sub in df.groupby("Temperature"):
    for method, msub in sub.groupby("Method"):
        summary_rows.append({
            "Temperature": temp, "Method": method,
            "TAR": msub["TAR"].mean(), "WFR": msub["Well_Formed"].mean(),
            "VVR": msub["Vocab_Valid"].mean(), "CR": msub["Complete"].mean(),
            "N": len(msub),
        })
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
print(f"\nSaved summary to {summary_path}")
