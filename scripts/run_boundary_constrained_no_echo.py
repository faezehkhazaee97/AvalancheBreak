import matplotlib
matplotlib.use("Agg")

# ============================================================
# CONSTRAINED-DECODING NO-ECHO TEST
# ============================================================
# The earlier no-echo test (run_boundary_no_echo.py) asked the model
# not to include boundary markers in its output, but the model
# disobeyed 90-100% of the time -- the instruction alone could not
# isolate segmentation (input structure) from echoing (output
# copying). This is the clean version: the boundary token ('|', a
# single token, id 91 for Qwen2.5-7B's tokenizer) is banned from the
# output vocabulary entirely via a LogitsProcessor that forces its
# logit to -inf at every generation step. The model is now LITERALLY
# INCAPABLE of emitting the marker, regardless of what it "wants" to
# do -- a genuine test of whether the input-side segmentation benefit
# survives when the model cannot echo the boundary at all.
#
# Same seeds, same headline condition (alphabet=2, k=2), same lengths
# as thermodynamics_grid_templated_v2.csv and the instruction-only
# no-echo test, so this is a direct three-way paired comparison
# (echo-required / instruction-only no-echo / constrained no-echo).

import os

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")

from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
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

# Ban every token whose surface form CONTAINS '|', not just the standalone
# token -- BPE merges mean '|' also appears inside ~113 other multi-character
# tokens (e.g. '||', '|\n', 'A|'-style merges). Banning only the standalone
# token leaves the model free to route around the constraint via one of
# these merged tokens, which a first attempt at this script confirmed
# happens 15-55% of the time -- not the "literally impossible" guarantee
# the experiment is supposed to provide.
#
# CAUGHT SECOND BUG: a blanket "contains '|'" filter also bans 14 of the
# model's own special/control tokens (all_special_ids), including
# <|im_end|> -- this model's actual EOS token -- and <|endoftext|>. With
# EOS banned, the model could never terminate generation naturally,
# forcing every trial to run to the max_new_tokens ceiling regardless of
# the echoing question this experiment is actually about. Excluding
# all_special_ids keeps the ban to the 99 genuine PRINTABLE '|'-merges
# (e.g. '||', '|\n') while leaving EOS and every other control token free.
vocab = tokenizer.get_vocab()
BAR_TOKEN_IDS = [i for t, i in vocab.items() if "|" in t and i not in set(tokenizer.all_special_ids)]
print(f"Banned {len(BAR_TOKEN_IDS)} printable token ids whose surface form contains '|' (excluding {len(tokenizer.all_special_ids)} special/control tokens, notably preserving EOS)")


class BanTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, banned_ids):
        self.banned_ids = banned_ids

    def __call__(self, input_ids, scores):
        scores[:, self.banned_ids] = -float("inf")
        return scores


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

shared_sequences = {}
for N in SEQUENCE_LENGTHS:
    for t in range(TRIALS_T):
        local_seed = 10_000 * ALPHABET_SIZE + 100 * N + t
        rng = random.Random(local_seed)
        shared_sequences[(N, t)] = engine.generate_raw_input(N, rng=rng)
print(f"Pregenerated {len(shared_sequences)} shared raw sequences (paired with thermodynamics_grid_templated_v2.csv and boundary_no_echo_grid.csv).")

logits_processor = LogitsProcessorList([BanTokensLogitsProcessor(BAR_TOKEN_IDS)])


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
            pad_token_id=tokenizer.eos_token_id, logits_processor=logits_processor,
        )
    gen_only = outputs[0][inputs["input_ids"].shape[1]:]
    pred = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
    pred = pred.split("Explanation")[0].split("\n")[0].strip()

    # The banned-token constraint guarantees no '|' can appear in gen_only,
    # but strip defensively anyway (e.g. if a different surface form slipped
    # through via a multi-token path) and to keep scoring identical to the
    # instruction-only no-echo script.
    clean_pred = pred.replace("|", "").replace(" ", "").replace("\n", "")
    clean_truth = truth
    clean_pred = clean_pred[:len(clean_truth)]

    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0
                      for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0
    still_contains_bar = "|" in pred  # sanity check: should always be False

    return {
        "Method": "Cellular_Bounds_ConstrainedNoEcho", "Alphabet_Size": ALPHABET_SIZE, "Interval": INTERVAL,
        "Seq_Length": N, "Trial": t, "TAR": tar, "SAR": sar, "Bar_Leaked_Through": still_contains_bar,
        "Input_Given": inp, "Ground_Truth": clean_truth, "LLM_Output": clean_pred, "Raw_Output": pred,
    }


tasks = [(N, t) for N in SEQUENCE_LENGTHS for t in range(TRIALS_T)]
print(f"\nStarting constrained no-echo sweep: {len(tasks)} evaluations...")

results = []
for N, t in tqdm(tasks, desc="Cellular_Bounds (constrained no-echo)"):
    try:
        results.append(run_trial(N, t))
    except Exception as exc:
        print(f"Task failed: N={N}, t={t} | Error: {exc}")

df = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "boundary_constrained_no_echo_grid.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved {csv_path}")

summary = df.groupby("Seq_Length")[["TAR", "SAR"]].mean()
leak_rate = df.groupby("Seq_Length")["Bar_Leaked_Through"].mean()
print("\n--- CONSTRAINED NO-ECHO Cellular_Bounds TAR/SAR by length ---")
print(summary.round(4).to_string())
print("\n--- Bar leak-through rate (should be exactly 0.0 -- sanity check on the constraint) ---")
print(leak_rate.to_string())

base_df = pd.read_csv(os.path.join(RESULTS_DIR, "thermodynamics_grid_templated_v2.csv"))
ref = base_df[(base_df.Alphabet_Size == ALPHABET_SIZE) & (base_df.Seq_Length.isin(SEQUENCE_LENGTHS))]
ref_cellular = ref[(ref.Method == "Cellular_Bounds") & (ref.Interval == INTERVAL)].groupby("Seq_Length")["TAR"].mean()
ref_base = ref[ref.Method == "Base"].groupby("Seq_Length")["TAR"].mean()

no_echo_instr_path = os.path.join(RESULTS_DIR, "boundary_no_echo_grid.csv")
instr_no_echo = pd.read_csv(no_echo_instr_path).groupby("Seq_Length")["TAR"].mean() if os.path.exists(no_echo_instr_path) else None

comparison = pd.DataFrame({
    "Base_TAR": ref_base, "Cellular_EchoRequired_TAR": ref_cellular,
    "Cellular_InstructionOnly_NoEcho_TAR": instr_no_echo, "Cellular_ConstrainedDecoding_NoEcho_TAR": summary["TAR"],
})
print("\n--- DOES THE BENEFIT SURVIVE WHEN ECHOING IS LITERALLY IMPOSSIBLE? ---")
print(comparison.round(4).to_string())
comparison.to_csv(os.path.join(RESULTS_DIR, "boundary_constrained_no_echo_comparison.csv"))

print("\nDone.")
