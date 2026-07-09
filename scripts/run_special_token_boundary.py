import matplotlib
matplotlib.use("Agg")

# ============================================================
# SPECIAL/RESERVED TOKEN AS BOUNDARY MARKER
# ============================================================
# Reviewer question: does using a genuine out-of-band special token
# (rather than an ordinary printable character) as the boundary marker
# change the effect? All markers tested so far (Section 3.5's |, ;, #,
# and Section 3.5's ||, <|>, \n, ,) are printable characters that exist
# in ordinary text. This tests <|object_ref_start|> -- a real special
# token already in Qwen2.5-7B-Instruct's vocabulary (reserved for
# Qwen2.5-VL's vision-grounding use case, never otherwise meaningful in
# a pure-text prompt, and structurally inert for the chat template,
# which only parses <|im_start|>/<|im_end|> for turn boundaries) -- used
# inline as the boundary marker at the same fixed interval as the
# existing marker-identity ablation (alphabet=2, interval=4, same seeds
# as boundary_ablation_grid.csv / boundary_ablation_multitoken_grid.csv).
#
# Important decoding subtlety: because the model is asked to copy the
# marker through unchanged, and the marker here IS a special token,
# scoring must decode with skip_special_tokens=False (otherwise the
# marker would be silently stripped from the model's response before we
# even see it, making every trial look like a "successful" removal
# regardless of what the model actually did). Chat-template artifacts
# (<|im_end|>, <|endoftext|>) are stripped explicitly afterward instead.

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

MARKER = "<|object_ref_start|>"
assert len(tokenizer.encode(MARKER, add_special_tokens=False)) == 1, "marker must be a single real token"


class TaskEngine:
    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size
        self.alphabet = [chr(ord('A') + i) for i in range(alphabet_size)]
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}

    def generate_raw_input(self, length, rng=None):
        rng = rng or random
        return rng.choices(self.alphabet, k=length)

    def build_cellular_from_raw(self, raw_input, interval, marker):
        input_w_bounds, output_w_bounds = "", ""
        for i, char in enumerate(raw_input):
            shifted_char = self.alphabet[(self.char_to_idx[char] + 1) % self.alphabet_size]
            input_w_bounds += char
            output_w_bounds += shifted_char
            if interval > 0 and (i + 1) % interval == 0 and i != len(raw_input) - 1:
                input_w_bounds += marker
                output_w_bounds += marker
        prompt = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(self.alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {self.alphabet[-1]} -> {self.alphabet[0]}, and {self.alphabet[0]} -> {self.alphabet[1]}.
Constraint: The input contains occurrences of the special marker {marker!r}. These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_w_bounds}
Output: """
        return input_w_bounds, output_w_bounds, prompt


ALPHABET_SIZE = 2
INTERVAL = 4
SEQUENCE_LENGTHS = [8, 16, 32, 64]
TRIALS_T = 20

engine = TaskEngine(ALPHABET_SIZE)


def run_trial(N, t):
    # Same seed formula as boundary_ablation_multitoken_grid.csv, so this
    # is a direct paired comparison against the existing marker sweep.
    local_seed = 70_000 * N + t
    rng = random.Random(local_seed)
    raw_input = engine.generate_raw_input(N, rng=rng)
    inp, truth, prompt = engine.build_cellular_from_raw(raw_input, interval=INTERVAL, marker=MARKER)

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
    # skip_special_tokens=False: the marker IS a special token and must
    # survive decoding to be scored; chat-template artifacts stripped below.
    pred_raw = tokenizer.decode(gen_only, skip_special_tokens=False)
    pred_raw = pred_raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    pred = pred_raw.split("Explanation")[0].split("\n")[0].strip()

    clean_pred = pred.replace(MARKER, "").replace(" ", "").replace("\n", "")
    clean_truth = truth.replace(MARKER, "")
    clean_pred = clean_pred[:len(clean_truth)]

    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0
                      for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0

    return {
        "Method": "Cellular_Bounds", "Marker": "special_token", "Placement": "periodic",
        "Alphabet_Size": ALPHABET_SIZE, "Interval": INTERVAL, "Seq_Length": N, "Trial": t,
        "TAR": tar, "SAR": sar, "Raw_Response": pred_raw,
        "Input_Given": inp, "Ground_Truth": clean_truth, "LLM_Output": clean_pred,
    }


tasks = [(N, t) for N in SEQUENCE_LENGTHS for t in range(TRIALS_T)]
print(f"\nStarting special-token boundary sweep: {len(tasks)} evaluations (marker={MARKER!r})...")

results = []
for N, t in tqdm(tasks, desc="Special-token marker"):
    try:
        results.append(run_trial(N, t))
    except Exception as exc:
        print(f"Task failed: N={N}, t={t} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "boundary_ablation_special_token_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

summary = df.groupby("Seq_Length")[["TAR", "SAR"]].mean()
print("\n--- SPECIAL-TOKEN MARKER TAR/SAR by length ---")
print(summary.round(4).to_string())

# --- Compare against Base and the existing printable-character markers ---
base_df = pd.read_csv(os.path.join(RESULTS_DIR, "thermodynamics_grid.csv"))
base_ref = base_df[(base_df.Method == "Base") & (base_df.Alphabet_Size == ALPHABET_SIZE)].groupby("Seq_Length")["TAR"].mean()

old_df = pd.read_csv(os.path.join(RESULTS_DIR, "boundary_ablation_grid.csv"))
old_df = old_df[old_df["Placement"] == "periodic"]
printable_mean = old_df.groupby(["Marker", "Seq_Length"])["TAR"].mean().groupby("Seq_Length").mean()

comparison = pd.DataFrame({
    "Base_TAR": base_ref,
    "Mean_Printable_Marker_TAR": printable_mean,
    "Special_Token_TAR": summary["TAR"],
})
print("\n--- DOES A REAL SPECIAL TOKEN BEHAVE DIFFERENTLY FROM PRINTABLE MARKERS? ---")
print(comparison.round(4).to_string())
comparison.to_csv(os.path.join(RESULTS_DIR, "special_token_comparison.csv"))

print("\nSpecial-token boundary sweep complete.")
