import matplotlib
matplotlib.use("Agg")

# ============================================================
# WALL-CLOCK / THROUGHPUT ACROSS BASE VS. DENSITIES
# ============================================================
# Reviewer ask: token counts (Section 2.4) quantify the compute
# picture, but wall-clock time and throughput are only measured for
# the divide-and-conquer comparison (Section 3.12, cyclic shift,
# interval=4 only), not systematically across Base and the full
# density range. This measures per-call wall-clock time and resulting
# tokens/sec for Base and every tested Cellular_Bounds density on the
# headline condition (alphabet=2), same lengths and seeds as
# thermodynamics_grid_templated_v2.csv, on the same single A100 GPU
# used for every other timing measurement in this paper.

import os

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random, time
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
            self.alphabet[(self.char_to_idx[c] + 1) % self.alphabet_size] for c in raw_input
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
SEQUENCE_LENGTHS = [8, 16, 32, 64]
DENSITIES = [0, 1, 2, 4, 8]  # 0 = Base
TRIALS_T = 20

engine = TaskEngine(ALPHABET_SIZE)

shared_sequences = {}
for N in SEQUENCE_LENGTHS:
    for t in range(TRIALS_T):
        local_seed = 10_000 * ALPHABET_SIZE + 100 * N + t
        rng = random.Random(local_seed)
        shared_sequences[(N, t)] = engine.generate_raw_input(N, rng=rng)
print(f"Pregenerated {len(shared_sequences)} shared raw sequences (paired with thermodynamics_grid_templated_v2.csv).")


def run_trial(interval, N, t):
    raw_input = shared_sequences[(N, t)]
    if interval == 0:
        inp, truth, prompt = engine.build_base_from_raw(raw_input)
    else:
        inp, truth, prompt = engine.build_cellular_from_raw(raw_input, interval=interval)

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in enc.items()}
    prompt_tokens = inputs["input_ids"].shape[1]
    max_tokens_needed = int(N * 1.5) + 20

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    gen_only = outputs[0][prompt_tokens:]
    n_new_tokens = gen_only.shape[0]
    pred = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
    pred = pred.split("Explanation")[0].split("\n")[0].strip()

    clean_pred = pred.replace("|", "").replace(" ", "").replace("\n", "")
    clean_truth = truth.replace("|", "")
    clean_pred = clean_pred[:len(clean_truth)]
    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0
                      for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0

    return {
        "Method": "Base" if interval == 0 else "Cellular_Bounds", "Interval": interval,
        "Seq_Length": N, "Trial": t, "TAR": tar,
        "Prompt_Tokens": prompt_tokens, "New_Tokens": n_new_tokens,
        "Wall_Time_Sec": elapsed, "Tokens_Per_Sec": n_new_tokens / elapsed if elapsed > 0 else float("nan"),
    }


tasks = [(k, N, t) for k in DENSITIES for N in SEQUENCE_LENGTHS for t in range(TRIALS_T)]
print(f"\nStarting wall-clock/throughput sweep: {len(tasks)} evaluations...")

results = []
for k, N, t in tqdm(tasks, desc="Wall-clock sweep"):
    try:
        results.append(run_trial(k, N, t))
    except Exception as exc:
        print(f"Task failed: k={k}, N={N}, t={t} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "wallclock_throughput_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

summary = df.groupby(["Method", "Interval", "Seq_Length"])[["Wall_Time_Sec", "Tokens_Per_Sec", "Prompt_Tokens", "New_Tokens"]].mean().reset_index()
print("\n--- WALL-CLOCK / THROUGHPUT BY (METHOD, DENSITY, LENGTH) ---")
print(summary.round(4).to_string(index=False))
summary.to_csv(os.path.join(RESULTS_DIR, "wallclock_throughput_summary.csv"), index=False)

print("\nDone.")
