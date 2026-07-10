import matplotlib
matplotlib.use("Agg")

# ============================================================
# ECHOING VS. SEGMENTATION CONTROL
# ============================================================
# Reviewer question: in the standard Cellular_Bounds condition, the
# model is instructed to copy boundary markers through to the output
# unchanged. This conflates two things that could each independently
# explain the benefit: (a) the marker segments attention in the INPUT
# (a purely structural/positional signal), and (b) the model must
# correctly ECHO the marker in its OUTPUT (a copying/formatting
# sub-task that could itself act as a periodic "checkpoint" reminding
# the model where it is in the sequence, independent of any attention
# effect). This script isolates (a) from (b): boundary markers are
# still present in the INPUT at the same periodic positions as the
# standard condition, but the model is explicitly instructed NOT to
# include them in its output -- only the transformed letters, no bars.
# If the benefit survives without the echoing requirement, that
# supports the paper's attention-segmentation account. If the benefit
# disappears or shrinks substantially, the echoing requirement itself
# was doing more of the work than the attention story alone predicts.
#
# Same seeds, same headline condition (alphabet=2, k=2), same lengths
# as thermodynamics_grid_templated_v2.csv, so this is a direct paired
# addition to the existing Base / Cellular_Bounds comparison.

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


class TaskEngine:
    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size
        self.alphabet = [chr(ord('A') + i) for i in range(alphabet_size)]
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}

    def generate_raw_input(self, length, rng=None):
        if rng is None:
            rng = random
        return rng.choices(self.alphabet, k=length)

    def build_no_echo_from_raw(self, raw_input, interval):
        # Input DOES contain boundary markers at the standard periodic
        # positions; the expected OUTPUT omits them entirely.
        input_w_bounds = ""
        output_no_bounds = ""
        for i, char in enumerate(raw_input):
            shifted_char = self.alphabet[(self.char_to_idx[char] + 1) % self.alphabet_size]
            input_w_bounds += char
            output_no_bounds += shifted_char
            if interval > 0 and (i + 1) % interval == 0 and i != len(raw_input) - 1:
                input_w_bounds += "|"
        prompt = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(self.alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {self.alphabet[-1]} -> {self.alphabet[0]}, and {self.alphabet[0]} -> {self.alphabet[1]}.
Constraint: The input contains vertical bars (|) as structural boundaries. Do NOT include them in your output -- output only the transformed letters, with the boundaries removed.
No spaces, no explanations.

Input: {input_w_bounds}
Output: """
        return input_w_bounds, output_no_bounds, prompt


ALPHABET_SIZE = 2
INTERVAL = 2
SEQUENCE_LENGTHS = [8, 16, 32, 64]
TRIALS_T = 20

engine = TaskEngine(ALPHABET_SIZE)

# Identical seed formula to thermodynamics_grid_templated_v2.csv's shared
# sequences, so this is directly paired against the existing Base and
# Cellular_Bounds (echo-required) rows at the same (N, t).
shared_sequences = {}
for N in SEQUENCE_LENGTHS:
    for t in range(TRIALS_T):
        local_seed = 10_000 * ALPHABET_SIZE + 100 * N + t
        rng = random.Random(local_seed)
        shared_sequences[(N, t)] = engine.generate_raw_input(N, rng=rng)
print(f"Pregenerated {len(shared_sequences)} shared raw sequences (paired with thermodynamics_grid_templated_v2.csv).")


def run_trial(N, t):
    raw_input = shared_sequences[(N, t)]
    inp, truth, prompt = engine.build_no_echo_from_raw(raw_input, interval=INTERVAL)

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

    # Model may still disobey and emit a bar despite the instruction;
    # strip any stray bars before scoring, same convention as elsewhere.
    clean_pred = pred.replace("|", "").replace(" ", "").replace("\n", "")
    clean_truth = truth  # already bar-free by construction
    clean_pred = clean_pred[:len(clean_truth)]

    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0
                      for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0
    disobeyed = "|" in pred  # did the model echo a bar despite being told not to?

    return {
        "Method": "Cellular_Bounds_NoEcho", "Alphabet_Size": ALPHABET_SIZE, "Interval": INTERVAL,
        "Seq_Length": N, "Trial": t, "TAR": tar, "SAR": sar, "Disobeyed_Echoed_Bar": disobeyed,
        "Input_Given": inp, "Ground_Truth": clean_truth, "LLM_Output": clean_pred, "Raw_Output": pred,
    }


tasks = [(N, t) for N in SEQUENCE_LENGTHS for t in range(TRIALS_T)]
print(f"\nStarting no-echo boundary sweep: {len(tasks)} evaluations...")

results = []
for N, t in tqdm(tasks, desc="Cellular_Bounds (no echo)"):
    try:
        results.append(run_trial(N, t))
    except Exception as exc:
        print(f"Task failed: N={N}, t={t} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "boundary_no_echo_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

summary = df.groupby("Seq_Length")[["TAR", "SAR"]].mean()
disobey_rate = df.groupby("Seq_Length")["Disobeyed_Echoed_Bar"].mean()
print("\n--- NO-ECHO Cellular_Bounds TAR/SAR by length ---")
print(summary.round(4).to_string())
print("\n--- Rate of disobeying the no-echo instruction (still emitted a bar) ---")
print(disobey_rate.round(4).to_string())

base_df = pd.read_csv(os.path.join(RESULTS_DIR, "thermodynamics_grid_templated_v2.csv"))
ref = base_df[(base_df.Alphabet_Size == ALPHABET_SIZE) & (base_df.Seq_Length.isin(SEQUENCE_LENGTHS))]
ref_cellular = ref[(ref.Method == "Cellular_Bounds") & (ref.Interval == INTERVAL)].groupby("Seq_Length")["TAR"].mean()
ref_base = ref[ref.Method == "Base"].groupby("Seq_Length")["TAR"].mean()

comparison = pd.DataFrame({
    "Base_TAR": ref_base, "Cellular_EchoRequired_TAR": ref_cellular, "Cellular_NoEcho_TAR": summary["TAR"],
})
comparison["NoEcho_vs_EchoRequired"] = comparison["Cellular_NoEcho_TAR"] - comparison["Cellular_EchoRequired_TAR"]
comparison["NoEcho_vs_Base"] = comparison["Cellular_NoEcho_TAR"] - comparison["Base_TAR"]
print("\n--- DOES THE BENEFIT SURVIVE WITHOUT THE ECHOING REQUIREMENT? ---")
print(comparison.round(4).to_string())
comparison.to_csv(os.path.join(RESULTS_DIR, "boundary_no_echo_comparison.csv"))

print("\nDone.")
