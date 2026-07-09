import matplotlib
matplotlib.use("Agg")

# ============================================================
# LOCALITY-SPECTRUM TASK 3 (FIXED-WINDOW-LOCAL): LOCAL BLOCK REVERSAL
# ============================================================
# Reviewer's own pick as "the most interesting addition scientifically":
# genuinely new middle ground between pointwise-local (cyclic shift) and
# unbounded-history (addition, running-sum). Reverse every non-overlapping
# k-character block (k=REVERSAL_WINDOW): each output position now
# depends on OTHER characters, but only the ones within its own fixed
# window, never outside it. This is the first task in the whole project
# where a boundary marker's placement relative to a real dependency
# window can be tested directly:
#   - Cellular_Aligned: boundary interval == REVERSAL_WINDOW, so every
#     boundary sits exactly BETWEEN windows, never splitting one.
#   - Cellular_Misaligned: boundary interval == REVERSAL_WINDOW // 2, so
#     every window has a boundary planted right in its middle, forcing
#     the model to reverse across a boundary it's told "carries no
#     meaning" it must nonetheless reverse straight through.
# If boundaries help only when aligned (or hurt specifically when
# misaligned), that is direct, clean evidence that boundaries interfere
# with a real dependency window when placed inside it -- something no
# other task in this paper can test, since cyclic shift has no windows
# to split and addition/running-sum have only one all-encompassing window
# (the whole sequence).

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

ALPHABET_SIZE = 4
ALPHABET = [chr(ord('A') + i) for i in range(ALPHABET_SIZE)]
REVERSAL_WINDOW = 4  # each non-overlapping 4-char block gets reversed


class BlockReversalEngine:
    def generate_raw_input(self, length, rng=None):
        rng = rng or random
        return rng.choices(ALPHABET, k=length)

    def _reversed_output(self, raw_input):
        out = []
        for start in range(0, len(raw_input), REVERSAL_WINDOW):
            block = raw_input[start:start + REVERSAL_WINDOW]
            out.extend(reversed(block))
        return out

    def build_base_from_raw(self, raw_input):
        input_str = "".join(raw_input)
        output_str = "".join(self._reversed_output(raw_input))
        prompt = f"""You are a deterministic sequence transformer.
Task: Local block reversal.
Alphabet: {', '.join(ALPHABET)}
Rule: Split the input into consecutive, non-overlapping blocks of {REVERSAL_WINDOW} characters each (the last block may be shorter). Reverse the characters WITHIN each block. Do not reverse the order of the blocks themselves, and do not let characters cross from one block into another.
Constraint: Output ONLY the transformed string. No spaces, no explanations.

Input: {input_str}
Output: """
        return input_str, output_str, prompt

    def build_cellular_from_raw(self, raw_input, interval):
        # Boundaries inserted in the INPUT at fixed `interval` (either
        # REVERSAL_WINDOW = aligned, or REVERSAL_WINDOW // 2 = misaligned).
        # Output boundaries are placed at the SAME raw character positions
        # they marked in the input (i.e. also every `interval` characters,
        # counted in original input order) -- the reversal itself still
        # happens per REVERSAL_WINDOW block regardless of where boundaries
        # fall, so a misaligned boundary genuinely sits inside a block that
        # must be reversed across it.
        reversed_chars = self._reversed_output(raw_input)

        input_w_bounds = ""
        for i, c in enumerate(raw_input):
            input_w_bounds += c
            if interval > 0 and (i + 1) % interval == 0 and i != len(raw_input) - 1:
                input_w_bounds += "|"

        # Output boundary positions: a boundary appears after output position
        # j if the corresponding INPUT position (within its reversed block)
        # crosses an `interval` boundary in the original input ordering.
        # Simplest faithful construction: place output boundaries at the same
        # fixed `interval` over the OUTPUT string's own positions, since the
        # model is instructed boundaries are structural and must be copied
        # through in the same relative density (matching every other task's
        # convention in this paper).
        output_w_bounds = ""
        for i, c in enumerate(reversed_chars):
            output_w_bounds += c
            if interval > 0 and (i + 1) % interval == 0 and i != len(reversed_chars) - 1:
                output_w_bounds += "|"

        prompt = f"""You are a deterministic sequence transformer.
Task: Local block reversal.
Alphabet: {', '.join(ALPHABET)}
Rule: Split the input into consecutive, non-overlapping blocks of {REVERSAL_WINDOW} characters each (the last block may be shorter), ignoring any vertical bars when counting characters into blocks. Reverse the characters WITHIN each block. Do not reverse the order of the blocks themselves, and do not let characters cross from one block into another.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT treat them as characters to reverse or move -- copy each one through to the output at the same relative position (i.e. after the same number of transformed characters have been produced).
No spaces, no explanations.

Input: {input_w_bounds}
Output: """
        return input_w_bounds, output_w_bounds, prompt


SEQUENCE_LENGTHS = [8, 16, 32, 64]
TRIALS_T = 20
CONDITIONS = {
    "Base": 0,
    "Cellular_Aligned": REVERSAL_WINDOW,       # k=4: boundaries between windows
    "Cellular_Misaligned": REVERSAL_WINDOW // 2,  # k=2: boundaries split windows
}

engine = BlockReversalEngine()

shared_sequences = {}
for N in SEQUENCE_LENGTHS:
    for t in range(TRIALS_T):
        local_seed = 68_000 * N + t
        rng = random.Random(local_seed)
        shared_sequences[(N, t)] = engine.generate_raw_input(N, rng=rng)


def run_trial(task_params):
    N, method, t = task_params
    interval = CONDITIONS[method]
    raw_input = shared_sequences[(N, t)]

    if method == "Base":
        inp, truth, prompt = engine.build_base_from_raw(raw_input)
    else:
        inp, truth, prompt = engine.build_cellular_from_raw(raw_input, interval=interval)

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

    if method != "Base":
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
        "Method": method, "Interval": interval, "Seq_Length": N, "Trial": t,
        "TAR": tar, "SAR": sar,
        "Input_Given": inp, "Ground_Truth": clean_truth, "LLM_Output": clean_pred,
    }


tasks = [(N, method, t) for N in SEQUENCE_LENGTHS for method in CONDITIONS for t in range(TRIALS_T)]
print(f"\nStarting block-reversal grid: {len(tasks)} evaluations...")

results = []
for task in tqdm(tasks, desc="Block reversal"):
    try:
        results.append(run_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "block_reversal_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

summary = df.groupby(["Method", "Seq_Length"])[["TAR", "SAR"]].mean().reset_index()
print("\n--- BLOCK REVERSAL: ACCURACY BY CONDITION & LENGTH ---")
print(summary.round(4).to_string(index=False))

print("\nBlock-reversal sweep complete.")
