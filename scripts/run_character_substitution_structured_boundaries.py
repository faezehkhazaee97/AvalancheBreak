import matplotlib
matplotlib.use("Agg")

import os
import random
import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"Results directory: {RESULTS_DIR}/")

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


class CharacterSubstitutionTaskEngine:
    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # Fixed mapping: A↔X, B↔Y, C↔Z, D↔W, E↔P, F↔Q, G↔R, H↔S, I↔T, J↔U, K↔V, L↔M, N↔O
    MAPPING = {
        'A': 'X', 'B': 'Y', 'C': 'Z', 'D': 'W', 'E': 'P', 'F': 'Q', 'G': 'R', 'H': 'S',
        'I': 'T', 'J': 'U', 'K': 'V', 'L': 'M', 'M': 'L', 'N': 'O', 'O': 'N',
        'P': 'E', 'Q': 'F', 'R': 'G', 'S': 'H', 'T': 'I', 'U': 'J', 'V': 'K',
        'W': 'D', 'X': 'A', 'Y': 'B', 'Z': 'C'
    }

    def sample_string(self, length, rng):
        return "".join(rng.choices(self.ALPHABET, k=length))

    def substitute(self, char):
        return self.MAPPING[char]

    def _insert_boundaries_in_string(self, string_str, interval):
        """Insert boundaries within a string every k positions."""
        if interval <= 0:
            return string_str
        bounded = []
        for i, c in enumerate(string_str):
            bounded.append(c)
            if (i + 1) % interval == 0 and i != len(string_str) - 1:
                bounded.append("|")
        return "".join(bounded)

    def build_prompt(self, length, interval, rng=None):
        """
        Build character substitution prompt with structured boundaries.

        Example with N=8, k=4:
            Input: ABCD|EFGH
            Output: XYZW|PQRS
        """
        rng = rng or random
        input_str_base = self.sample_string(length, rng)
        output_str_base = "".join(self.substitute(c) for c in input_str_base)

        input_bounded = self._insert_boundaries_in_string(input_str_base, interval)
        output_bounded = self._insert_boundaries_in_string(output_str_base, interval)

        prompt = f"""You are a deterministic sequence transformer.
Task: Character substitution mapping.
Alphabet: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
Mapping: A↔X, B↔Y, C↔Z, D↔W, E↔P, F↔Q, G↔R, H↔S, I↔T, J↔U, K↔V, L↔M, N↔O
Rule: Replace each character with its mapped character. This is a fixed one-to-one substitution cipher.
Constraint: The input contains vertical bars (|). These are structural boundaries.
DO NOT transform them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """

        return input_bounded, output_bounded, prompt


CHAR_SUB_SEQUENCE_LENGTHS = [4, 8, 16, 32, 64]
CHAR_SUB_PERIODIC_INTERVALS = [0, 1, 2, 3, 4, 8, 16]
CHAR_SUB_TRIALS = 20

engine_char_sub = CharacterSubstitutionTaskEngine()


def run_character_substitution_trial(task_params):
    N, k, t = task_params
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 60_000 * N + 100 * k + t
    rng = random.Random(local_seed)

    inp, truth, prompt = engine_char_sub.build_prompt(N, interval=k, rng=rng)

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

    match = re.match(r'^([A-Z|]+)', pred)
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


char_sub_tasks = [
    (N, k, t)
    for N in CHAR_SUB_SEQUENCE_LENGTHS
    for k in CHAR_SUB_PERIODIC_INTERVALS
    for t in range(CHAR_SUB_TRIALS)
    if k == 0 or k < 2 * N + 1
]
print(f"\nStarting CHARACTER SUBSTITUTION grid: {len(char_sub_tasks)} evaluations...")

char_sub_results = []
for task in tqdm(char_sub_tasks, desc="Character Substitution (structured boundaries)"):
    try:
        char_sub_results.append(run_character_substitution_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

char_sub_df = pd.DataFrame(char_sub_results)
char_sub_csv = os.path.join(RESULTS_DIR, "character_substitution_grid_structured_boundaries.csv")
char_sub_df.to_csv(char_sub_csv, index=False)
print(f"\nCharacter Substitution complete. Results saved to {char_sub_csv}")

# Summary stats
summary = char_sub_df.groupby(["Seq_Length", "Interval"])["TAR"].agg(["mean", "std"]).reset_index()
print("\n--- MEAN TAR BY (Seq_Length, Interval), CHARACTER SUBSTITUTION ---")
print(summary.to_string(index=False))
