import matplotlib
matplotlib.use("Agg")

# ============================================================
# LOCALITY-SPECTRUM TASK 1 (POINTWISE-LOCAL): DIGIT COMPLEMENT
# ============================================================
# Reviewer suggestion: fill in the locality spectrum between "each output
# depends on exactly one input character" (cyclic shift) and "genuine
# unbounded history" (running-sum, addition). This is a second pointwise
# task in a different domain (digits, not letters) and a different rule
# (9's complement, not cyclic increment), to check the headline effect
# isn't an artifact of the specific cyclic-shift construction. Each
# output digit is the 9's complement of the corresponding input digit
# (d -> 9-d), with no carry and no dependency on any other position --
# the same locality profile as cyclic shift, just a different rule and
# a different (digit, not letter) alphabet.

import os

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")

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
from tqdm import tqdm


class DigitComplementEngine:
    def generate_raw_input(self, length, rng=None):
        rng = rng or random
        return [rng.randrange(10) for _ in range(length)]

    def build_base_from_raw(self, raw_input):
        input_str = "".join(map(str, raw_input))
        output_str = "".join(str(9 - d) for d in raw_input)
        prompt = f"""You are a deterministic digit transformer.
Task: 9's complement.
Rule: Replace every digit d with (9 - d). For example, 0->9, 3->6, 9->0.
Constraint: Output ONLY the transformed string. No spaces, no explanations.

Input: {input_str}
Output: """
        return input_str, output_str, prompt

    def build_cellular_from_raw(self, raw_input, interval):
        input_w_bounds, output_w_bounds = "", ""
        for i, d in enumerate(raw_input):
            input_w_bounds += str(d)
            output_w_bounds += str(9 - d)
            if interval > 0 and (i + 1) % interval == 0 and i != len(raw_input) - 1:
                input_w_bounds += "|"
                output_w_bounds += "|"
        prompt = f"""You are a deterministic digit transformer.
Task: 9's complement.
Rule: Replace every digit d with (9 - d). For example, 0->9, 3->6, 9->0.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_w_bounds}
Output: """
        return input_w_bounds, output_w_bounds, prompt


SEQUENCE_LENGTHS = [8, 16, 32, 64]
PERIODIC_INTERVALS = [0, 1, 2, 4]
TRIALS_T = 20

engine = DigitComplementEngine()

shared_sequences = {}
for N in SEQUENCE_LENGTHS:
    for t in range(TRIALS_T):
        local_seed = 60_000 * N + t
        rng = random.Random(local_seed)
        shared_sequences[(N, t)] = engine.generate_raw_input(N, rng=rng)


def run_trial(task_params):
    N, k, t = task_params
    method = "Base" if k == 0 else "Cellular_Bounds"
    raw_input = shared_sequences[(N, t)]

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
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0

    return {
        "Method": method, "Interval": k, "Seq_Length": N, "Trial": t,
        "TAR": tar, "SAR": sar,
        "Input_Given": inp, "Ground_Truth": clean_truth, "LLM_Output": clean_pred,
    }


tasks = [(N, k, t) for N in SEQUENCE_LENGTHS for k in PERIODIC_INTERVALS for t in range(TRIALS_T) if k == 0 or k < N]
print(f"\nStarting digit-complement grid: {len(tasks)} evaluations...")

results = []
for task in tqdm(tasks, desc="Digit complement"):
    try:
        results.append(run_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "digit_complement_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

summary = df.groupby(["Interval", "Seq_Length"])[["TAR", "SAR"]].mean().reset_index()
print("\n--- DIGIT COMPLEMENT: ACCURACY BY INTERVAL & LENGTH ---")
print(summary.round(4).to_string(index=False))

print("\nDigit-complement sweep complete.")
