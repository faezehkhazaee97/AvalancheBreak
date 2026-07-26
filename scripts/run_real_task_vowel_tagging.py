"""
Real-task validation of monotonic-locality principle.

Task: Character-level vowel/consonant tagging (position-independent)
- Input: random English word characters
- Output: tag each character as VOWEL (A,E,I,O,U) or CONSONANT
- This is genuinely position-independent: each character's label depends only on itself
- Hypothesis: boundaries should help (as with cyclic shift)

This tests whether the monotonic-locality principle generalizes beyond synthetic cyclic shift.
"""

import matplotlib
matplotlib.use("Agg")

import os
import random
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
    """Position-independent task: tag each character as VOWEL or CONSONANT."""

    VOWELS = set('AEIOU')
    CONSONANTS = set('BCDFGHJKLMNPQRSTVWXYZ')

    def generate_random_string(self, length, rng=None):
        """Generate random uppercase English letters."""
        if rng is None:
            rng = random
        all_letters = list(self.VOWELS | self.CONSONANTS)
        return ''.join(rng.choices(all_letters, k=length))

    def get_ground_truth_tags(self, char_string):
        """Tag each character as V or C (position-independent)."""
        tags = []
        for char in char_string:
            if char in self.VOWELS:
                tags.append('V')
            elif char in self.CONSONANTS:
                tags.append('C')
            else:
                tags.append('?')
        return ''.join(tags)

    def build_prompt_with_bounds(self, char_string, interval=0):
        """Build prompt with optional boundaries."""
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
        """Compute TAR and SAR."""
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

# Test parameters
SEQUENCE_LENGTHS = [8, 16, 32, 64]
BOUNDARY_INTERVALS = [0, 2, 4, 8]  # 0 = no boundaries, 2 = boundary every 2 chars, etc.
TRIALS_PER_CONFIG = 10  # Smaller than 20 for faster iteration

# Generate sequences with fixed seeds for reproducibility
shared_sequences = {}
for N in SEQUENCE_LENGTHS:
    for t in range(TRIALS_PER_CONFIG):
        local_seed = 50_000 + 100 * N + t  # Different seed space than cyclic shift
        rng = random.Random(local_seed)
        shared_sequences[(N, t)] = engine.generate_random_string(N, rng=rng)

print(f"Generated {len(shared_sequences)} random strings for vowel tagging task.")


def run_trial(N, t, interval):
    """Run a single vowel-tagging trial."""
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
        "Task": "Vowel_Tagging",
        "Seq_Length": N,
        "Interval": interval,
        "Trial": t,
        "TAR": tar,
        "SAR": sar,
        "Input": input_str,
        "Expected": expected_output.replace("|", ""),
        "Predicted": pred.replace("|", ""),
    }


# Run all trials
all_results = []
total_trials = len(SEQUENCE_LENGTHS) * len(BOUNDARY_INTERVALS) * TRIALS_PER_CONFIG
pbar = tqdm(total=total_trials, desc="Vowel tagging trials")

for N in SEQUENCE_LENGTHS:
    for interval in BOUNDARY_INTERVALS:
        for t in range(TRIALS_PER_CONFIG):
            result = run_trial(N, t, interval)
            all_results.append(result)
            pbar.update(1)

pbar.close()

# Save results
df = pd.DataFrame(all_results)
output_file = os.path.join(RESULTS_DIR, "vowel_tagging_boundary_sweep.csv")
df.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")

# Analysis
print("\n" + "="*80)
print("VOWEL TAGGING RESULTS (Position-Independent Real Task)")
print("="*80)

for N in SEQUENCE_LENGTHS:
    print(f"\nSequence Length N={N}:")
    for interval in BOUNDARY_INTERVALS:
        subset = df[(df['Seq_Length'] == N) & (df['Interval'] == interval)]
        tar_mean = subset['TAR'].mean()
        tar_std = subset['TAR'].std()
        sar_mean = subset['SAR'].mean()
        print(f"  k={interval:2d} (density={1/interval if interval > 0 else 0:.2f}): "
              f"TAR={tar_mean:.3f}±{tar_std:.3f}, SAR={sar_mean:.3f}")

# Summary: check if monotonic-locality principle holds (upward TAR trend with boundary density)
print("\n" + "="*80)
print("MONOTONIC-LOCALITY CHECK: Do boundaries help position-independent vowel tagging?")
print("="*80)

for N in SEQUENCE_LENGTHS:
    base_tar = df[(df['Seq_Length'] == N) & (df['Interval'] == 0)]['TAR'].mean()
    best_tar = df[df['Seq_Length'] == N]['TAR'].max()
    best_interval = df[df['Seq_Length'] == N].loc[df[df['Seq_Length'] == N]['TAR'].idxmax(), 'Interval']

    trend = "↑ HELPS" if best_tar > base_tar else "↓ HURTS"
    gap = best_tar - base_tar
    print(f"N={N:2d}: Base={base_tar:.3f}, Best(k={best_interval})={best_tar:.3f}, Gap={gap:+.3f} {trend}")

print("\n✓ If boundaries help on vowel tagging (like they do on cyclic shift),")
print("  this validates monotonic-locality for real position-independent tasks!")
