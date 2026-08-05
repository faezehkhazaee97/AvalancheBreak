import matplotlib
matplotlib.use("Agg")

# ============================================================
# TEMPERATURE / SAMPLING ROBUSTNESS CHECK
# ============================================================
# Reviewer question: "Did you try sampling (e.g., temperature 0.2-0.7) or
# nucleus decoding? Are the trends robust, or does stochastic decoding
# change TAR/SAR or the alpha fits?" This is a real, previously
# uncharacterized gap -- every other result in this project uses greedy
# decoding (do_sample=False) exclusively.
#
# Reuses the exact headline TaskEngine (cyclic_shift, alphabet=2) and the
# same seed convention as thermodynamics_grid_templated_v2.csv, but is
# deliberately scoped down from the full 7x3x7 grid to a targeted
# robustness spot-check: 3 lengths (16/32/64), Base vs the single densest
# boundary condition (k=2, the same interval already used for the causal
# ablation and SK-fit headline numbers), 3 temperatures (0.0 reference,
# 0.2, 0.7), 10 trials/cell -- enough to see whether the Base-vs-Cellular
# gap and its direction survive sampling, not a full re-run under
# sampling.

import os

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random
import numpy as np
import pandas as pd
from tqdm import tqdm

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
INTERVAL = 2  # same density used for the causal ablation / SK-fit headline numbers
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
    pred = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
    pred = pred.split("Explanation")[0].split("\n")[0].strip()

    clean_pred = pred.replace("|", "").replace(" ", "").replace("\n", "")
    clean_truth = truth.replace("|", "")
    clean_pred = clean_pred[:len(clean_truth)]
    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0
                      for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0

    return {"Temperature": temperature, "Method": method, "Interval": interval,
            "Seq_Length": N, "Trial": t, "TAR": tar, "SAR": sar}


tasks = [(N, method, temp, t)
         for temp in TEMPERATURES
         for N in SEQUENCE_LENGTHS
         for method in ["Base", "Cellular_Bounds"]
         for t in range(TRIALS_T)]
print(f"\nStarting temperature robustness sweep: {len(tasks)} evaluations...")

results = []
for N, method, temp, t in tqdm(tasks, desc="Temperature robustness"):
    try:
        results.append(run_trial(N, method, temp, t))
    except Exception as exc:
        print(f"Task failed: N={N} method={method} temp={temp} t={t} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "temperature_robustness_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

print("\n--- Base vs Cellular_Bounds TAR by temperature and length ---")
summary = df.groupby(["Temperature", "Seq_Length", "Method"])["TAR"].mean().unstack()
print(summary.round(4).to_string())

print("\n--- Delta (Cellular_Bounds - Base) by temperature and length ---")
delta = summary["Cellular_Bounds"] - summary["Base"]
print(delta.round(4).to_string())

print("\n--- SAR variance by temperature (does sampling noise swamp the effect?) ---")
print(df.groupby(["Temperature"])["SAR"].agg(["mean", "std"]).round(4).to_string())
