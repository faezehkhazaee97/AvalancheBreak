import matplotlib
matplotlib.use("Agg")

# ============================================================
# INSPECT RAW INPUT/OUTPUT: why does A=4, N=16 diverge so much?
# ============================================================
# Prints the FULL prompt and FULL raw model output (before any cleaning
# or extraction) for a handful of representative trials at Alphabet=4,
# Seq_Length=16 -- the condition with the single biggest gap against the
# draft's Table 2 (draft R^2=0.930, raw-prompt R^2=0.421, templated
# R^2=0.885). Runs both prompt styles (raw vs chat-templated) on the
# IDENTICAL sequences so the qualitative difference in model behavior is
# directly visible, not just the aggregate score.

import os

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)

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
            self.alphabet[(self.char_to_idx[c] + 1) % self.alphabet_size]
            for c in raw_input
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
        input_w_bounds = ""
        output_w_bounds = ""
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


A = 4
N = 16
INTERVALS = [0, 2, 4, 8]
TRIALS_TO_SHOW = 2

engine = TaskEngine(A)

def run_and_print(k, t, use_template):
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 10_000 * A + 100 * N + t
    rng = random.Random(local_seed)
    raw_input = engine.generate_raw_input(N, rng=rng)

    if method == "Base":
        inp, truth, prompt = engine.build_base_from_raw(raw_input)
    else:
        inp, truth, prompt = engine.build_cellular_from_raw(raw_input, interval=k)

    if use_template:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(model.device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    max_tokens_needed = int(N * 1.5) + 20
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_only = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(gen_only, skip_special_tokens=True)
    pred = raw_output.strip().split("Explanation")[0].split("\n")[0].strip()

    if method == "Cellular_Bounds":
        clean_pred = pred.replace("|", "").replace(" ", "").replace("\n", "")
        clean_truth = truth.replace("|", "")
    else:
        clean_pred = pred.replace(" ", "").replace("\n", "")
        clean_truth = truth
    clean_pred_scored = clean_pred[:len(clean_truth)]
    correct = clean_pred_scored == clean_truth

    style = "TEMPLATED" if use_template else "RAW"
    print(f"\n{'='*70}")
    print(f"[{style}] Method={method}  k={k}  trial={t}")
    print(f"{'='*70}")
    print(f"--- PROMPT SENT TO MODEL ---")
    print(repr(prompt))
    print(f"--- FULL RAW MODEL OUTPUT (before any cleaning) ---")
    print(repr(raw_output))
    print(f"--- EXTRACTED PREDICTION (after cleaning) ---")
    print(f"  {clean_pred_scored!r}")
    print(f"--- GROUND TRUTH ---")
    print(f"  {clean_truth!r}")
    print(f"--- CORRECT? {correct} ---")


for use_template in [False, True]:
    for k in INTERVALS:
        for t in range(TRIALS_TO_SHOW):
            try:
                run_and_print(k, t, use_template)
            except Exception as exc:
                print(f"FAILED: k={k}, t={t}, template={use_template} | {exc}")

print("\n\nInspection complete.")
