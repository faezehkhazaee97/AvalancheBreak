"""Vowel/consonant tagging, extended to N=128. Same task/prompt/scoring as
run_real_task_vowel_tagging(_32b).py, parameterized by MODEL_ID via argv
so one script covers both 7B and 32B reruns instead of two near-duplicates.
Usage: python run_vowel_tagging_n128.py <model_id>
"""

import matplotlib
matplotlib.use("Agg")

import os
import sys
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
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


class VowelTaggerEngine:
    VOWELS = set('AEIOU')
    CONSONANTS = set('BCDFGHJKLMNPQRSTVWXYZ')

    def generate_random_string(self, length, rng=None):
        if rng is None:
            rng = random
        all_letters = list(self.VOWELS | self.CONSONANTS)
        return ''.join(rng.choices(all_letters, k=length))

    def build_prompt_with_bounds(self, char_string, interval=0):
        input_str = ""
        expected_output = ""
        for i, char in enumerate(char_string):
            tag = 'V' if char in self.VOWELS else 'C'
            input_str += char
            expected_output += tag
            if interval > 0 and (i + 1) % interval == 0 and i != len(char_string) - 1:
                input_str += "|"
                expected_output += "|"
        prompt = f"""You are a character classifier.
Task: Vowel/Consonant tagging.
Rule: For each character, output 'V' if it's a vowel (A, E, I, O, U) or 'C' if it's a consonant.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT tag them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_str}
Output: """
        return input_str, expected_output, prompt

    def score_output(self, pred, truth):
        clean_pred = pred.replace("|", "").replace(" ", "").replace("\n", "").upper()
        clean_truth = truth.replace("|", "")
        clean_pred = clean_pred[:len(clean_truth)]
        if not clean_truth:
            return 0.0, 0.0
        correct = sum(1 for i in range(len(clean_truth)) if i < len(clean_pred) and clean_pred[i] == clean_truth[i])
        tar = correct / len(clean_truth)
        sar = 1.0 if correct == len(clean_truth) else 0.0
        return tar, sar


engine = VowelTaggerEngine()

SEQUENCE_LENGTHS = [8, 16, 32, 64, 128]
BOUNDARY_INTERVALS = [0, 2, 4, 8, 16]
TRIALS_PER_CONFIG = 10

shared_sequences = {}
for N in SEQUENCE_LENGTHS:
    for t in range(TRIALS_PER_CONFIG):
        local_seed = 50_000 + 100 * N + t
        rng = random.Random(local_seed)
        shared_sequences[(N, t)] = engine.generate_random_string(N, rng=rng)

print(f"Generated {len(shared_sequences)} random strings for vowel tagging task.")


def run_trial(N, t, interval):
    char_string = shared_sequences[(N, t)]
    input_str, expected_output, prompt = engine.build_prompt_with_bounds(char_string, interval=interval)
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in enc.items()}
    max_tokens_needed = int(N * 1.5) + 20
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_only = outputs[0][inputs["input_ids"].shape[1]:]
    pred = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
    pred = pred.split("Explanation")[0].split("\n")[0].strip()
    tar, sar = engine.score_output(pred, expected_output)
    return {
        "Task": "Vowel_Tagging", "Seq_Length": N, "Interval": interval, "Trial": t,
        "TAR": tar, "SAR": sar, "Input": input_str,
        "Expected": expected_output.replace("|", ""), "Predicted": pred.replace("|", ""),
    }


all_results = []
tasks = [(N, k, t) for N in SEQUENCE_LENGTHS for k in BOUNDARY_INTERVALS for t in range(TRIALS_PER_CONFIG)]
for N, k, t in tqdm(tasks, desc="Vowel tagging"):
    all_results.append(run_trial(N, t, k))

df = pd.DataFrame(all_results)
output_file = os.path.join(RESULTS_DIR, "vowel_tagging_boundary_sweep.csv")
df.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")
print(df.groupby(["Interval", "Seq_Length"])["TAR"].mean().unstack().round(3).to_string())
