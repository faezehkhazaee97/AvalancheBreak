import matplotlib
matplotlib.use("Agg")

# ============================================================
# MARKER-PROPERTY ABLATION SUITE, Qwen2.5-32B-Instruct
# ============================================================
# Combines five previously 7B-only ablations into one script that loads
# the 32B model once: (A) marker identity + random placement, (B) jitter
# placement, (C) a genuine reserved special token as marker, (D) a
# token-budget-matched multi-token marker control, (E) three-way prompt-
# template robustness. All five originally ran on cyclic_shift,
# alphabet=2, Qwen2.5-7B-Instruct.
#
# Fix applied here: (A) and (E) come from scripts/run_new_experiments.py,
# which calls the tokenizer directly on a raw prompt string and skips
# tokenizer.apply_chat_template(...) -- inconsistent with (B)/(C)/(D)'s
# scripts and with this study's own documented finding that skipping the
# chat template materially and non-uniformly changes results. All five
# experiments here use apply_chat_template consistently.

import os

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"Results directory: {RESULTS_DIR}/")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random
import numpy as np
import pandas as pd
from tqdm import tqdm

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


def chat_generate(prompt, max_new_tokens, skip_special_tokens=True):
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_only = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_only, skip_special_tokens=skip_special_tokens)


class TaskEngine:
    def __init__(self, alphabet_size):
        self.alphabet_size = alphabet_size
        self.alphabet = [chr(ord("A") + i) for i in range(alphabet_size)]
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}

    def generate_raw_input(self, length, rng=None):
        rng = rng or random
        return rng.choices(self.alphabet, k=length)


ALPHABET_SIZE = 2
engine = TaskEngine(ALPHABET_SIZE)

# ============================================================
# (A) MARKER IDENTITY + RANDOM PLACEMENT
# ============================================================
print("\n" + "=" * 60)
print("(A) Marker identity + random placement")
print("=" * 60)

ABLATION_SEQUENCE_LENGTHS = [8, 16, 32, 64]
ABLATION_TRIALS = 15


def boundary_positions_periodic(length, interval):
    return set(i for i in range(interval - 1, length - 1, interval))


def boundary_positions_random(length, count, rng):
    candidates = list(range(length - 1))
    return set(rng.sample(candidates, min(count, len(candidates))))


def build_ablation_variant(raw_input, marker, placement, interval=4, rng=None):
    rng = rng or random
    length = len(raw_input)
    if placement == "periodic":
        boundary_positions = boundary_positions_periodic(length, interval)
    elif placement == "random":
        n_boundaries = len(boundary_positions_periodic(length, interval))
        boundary_positions = boundary_positions_random(length, n_boundaries, rng)
    else:
        raise ValueError(placement)

    input_w_bounds, output_w_bounds = "", ""
    for i, char in enumerate(raw_input):
        shifted_char = engine.alphabet[(engine.char_to_idx[char] + 1) % ALPHABET_SIZE]
        input_w_bounds += char
        output_w_bounds += shifted_char
        if i in boundary_positions:
            input_w_bounds += marker
            output_w_bounds += marker

    prompt = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(engine.alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {engine.alphabet[-1]} -> {engine.alphabet[0]}, and {engine.alphabet[0]} -> {engine.alphabet[1]}.
Constraint: The input contains structural boundary markers ({marker}). DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_w_bounds}
Output: """
    return input_w_bounds, output_w_bounds, prompt, len(boundary_positions)


ABLATION_CONDITIONS = [
    {"Marker": "|", "Placement": "periodic", "Ablation_Axis": "marker+placement baseline"},
    {"Marker": ";", "Placement": "periodic", "Ablation_Axis": "marker identity"},
    {"Marker": "#", "Placement": "periodic", "Ablation_Axis": "marker identity"},
    {"Marker": "|", "Placement": "random", "Ablation_Axis": "placement regularity"},
]


def run_ablation_trial(raw_input, condition, N, trial, rng):
    marker, placement = condition["Marker"], condition["Placement"]
    inp, truth, prompt, n_boundaries = build_ablation_variant(raw_input, marker=marker, placement=placement, interval=4, rng=rng)
    max_tokens_needed = int(N * 1.5) + 20 + n_boundaries
    pred_full = chat_generate(prompt, max_tokens_needed)
    pred = pred_full.strip().split("Explanation")[0].split("\n")[0].strip()

    clean_pred = pred.replace(marker, "").replace(" ", "").replace("\n", "")
    clean_truth = truth.replace(marker, "")
    clean_pred = clean_pred[:len(clean_truth)]
    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0 for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0

    return {
        "Marker": marker, "Placement": placement, "Ablation_Axis": condition["Ablation_Axis"],
        "Seq_Length": N, "Trial": trial, "N_Boundaries": n_boundaries, "TAR": tar, "SAR": sar,
    }


ablation_tasks = [(cond, N, t) for cond in ABLATION_CONDITIONS for N in ABLATION_SEQUENCE_LENGTHS for t in range(ABLATION_TRIALS)]
print(f"Starting: {len(ablation_tasks)} evaluations...")
ablation_results = []
for (cond, N, t) in tqdm(ablation_tasks, desc="(A) identity+placement"):
    local_seed = 70_000 * N + 1000 * ABLATION_CONDITIONS.index(cond) + t
    rng = random.Random(local_seed)
    raw_input = engine.generate_raw_input(N, rng=rng)
    try:
        ablation_results.append(run_ablation_trial(raw_input, cond, N, t, rng))
    except Exception as exc:
        print(f"Task failed: {cond}, N={N}, trial={t} | Error: {exc}")

ablation_df = pd.DataFrame(ablation_results)
ablation_csv = os.path.join(RESULTS_DIR, "boundary_ablation_grid.csv")
ablation_df.to_csv(ablation_csv, index=False)
print(f"Saved {ablation_csv}")
print(ablation_df.groupby(["Ablation_Axis", "Marker", "Placement", "Seq_Length"])["TAR"].mean().round(4).to_string())

# ============================================================
# (B) JITTER PLACEMENT
# ============================================================
print("\n" + "=" * 60)
print("(B) Jitter placement")
print("=" * 60)

INTERVAL = 4
JITTER_AMOUNTS = [1, 2]
JITTER_TRIALS = 20


def shift_char(c):
    return engine.alphabet[(engine.char_to_idx[c] + 1) % ALPHABET_SIZE]


def jittered_positions(N, k, jitter, rng):
    canonical = list(range(k, N, k))
    positions = set()
    for pos in canonical:
        offset = rng.randint(-jitter, jitter)
        jp = pos + offset
        jp = max(1, min(N - 1, jp))
        positions.add(jp)
    return sorted(positions)


def build_jittered(raw_input, k, jitter, rng):
    N = len(raw_input)
    positions = set(jittered_positions(N, k, jitter, rng))
    input_bounded, output_bounded = [], []
    for i, c in enumerate(raw_input):
        input_bounded.append(c)
        output_bounded.append(shift_char(c))
        if i in positions and i != N - 1:
            input_bounded.append("|")
            output_bounded.append("|")
    return "".join(input_bounded), "".join(output_bounded), len(positions)


def build_prompt_pipe(input_bounded):
    return f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(engine.alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {engine.alphabet[-1]} -> {engine.alphabet[0]}, and {engine.alphabet[0]} -> {engine.alphabet[1]}.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_bounded}
Output: """


def run_jitter_trial(N, jitter, t):
    local_seed = 10_000 * ALPHABET_SIZE + 100 * N + t
    rng = random.Random(local_seed)
    raw_input = engine.generate_raw_input(N, rng)
    input_bounded, truth, n_boundaries = build_jittered(raw_input, INTERVAL, jitter, rng)
    prompt = build_prompt_pipe(input_bounded)
    max_tokens_needed = int(N * 1.5) + 20
    pred_full = chat_generate(prompt, max_tokens_needed)
    pred = pred_full.strip().split("Explanation")[0].split("\n")[0].strip()

    clean_pred = pred.replace("|", "").replace(" ", "").replace("\n", "")
    clean_truth = truth.replace("|", "")
    clean_pred = clean_pred[:len(clean_truth)]
    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0 for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0
    return {"Method": f"Jitter_{jitter}", "Interval": INTERVAL, "Jitter": jitter, "Seq_Length": N, "Trial": t, "TAR": tar, "SAR": sar, "N_Boundaries": n_boundaries}


jitter_tasks = [(N, j, t) for N in ABLATION_SEQUENCE_LENGTHS for j in JITTER_AMOUNTS for t in range(JITTER_TRIALS)]
print(f"Starting: {len(jitter_tasks)} evaluations...")
jitter_results = []
for N, j, t in tqdm(jitter_tasks, desc="(B) jitter"):
    try:
        jitter_results.append(run_jitter_trial(N, j, t))
    except Exception as exc:
        print(f"Task failed: N={N} jitter={j} t={t} | Error: {exc}")

jitter_df = pd.DataFrame(jitter_results)
jitter_csv = os.path.join(RESULTS_DIR, "jitter_placement_grid.csv")
jitter_df.to_csv(jitter_csv, index=False)
print(f"Saved {jitter_csv}")
print(jitter_df.groupby(["Jitter", "Seq_Length"])["TAR"].mean().unstack().round(4).to_string())

# ============================================================
# (C) SPECIAL RESERVED TOKEN AS MARKER
# ============================================================
print("\n" + "=" * 60)
print("(C) Special reserved token as marker")
print("=" * 60)

SPECIAL_MARKER = "<|object_ref_start|>"
assert len(tokenizer.encode(SPECIAL_MARKER, add_special_tokens=False)) == 1, "marker must be a single real token"


def build_special(raw_input, interval, marker):
    input_w_bounds, output_w_bounds = "", ""
    for i, char in enumerate(raw_input):
        shifted_char = engine.alphabet[(engine.char_to_idx[char] + 1) % ALPHABET_SIZE]
        input_w_bounds += char
        output_w_bounds += shifted_char
        if interval > 0 and (i + 1) % interval == 0 and i != len(raw_input) - 1:
            input_w_bounds += marker
            output_w_bounds += marker
    prompt = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(engine.alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {engine.alphabet[-1]} -> {engine.alphabet[0]}, and {engine.alphabet[0]} -> {engine.alphabet[1]}.
Constraint: The input contains occurrences of the special marker {marker!r}. These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_w_bounds}
Output: """
    return input_w_bounds, output_w_bounds, prompt


def run_special_trial(N, t):
    local_seed = 70_000 * N + t
    rng = random.Random(local_seed)
    raw_input = engine.generate_raw_input(N, rng=rng)
    inp, truth, prompt = build_special(raw_input, interval=INTERVAL, marker=SPECIAL_MARKER)
    max_tokens_needed = int(N * 1.5) + 20
    pred_raw = chat_generate(prompt, max_tokens_needed, skip_special_tokens=False)
    pred_raw = pred_raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    pred = pred_raw.split("Explanation")[0].split("\n")[0].strip()

    clean_pred = pred.replace(SPECIAL_MARKER, "").replace(" ", "").replace("\n", "")
    clean_truth = truth.replace(SPECIAL_MARKER, "")
    clean_pred = clean_pred[:len(clean_truth)]
    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0 for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0
    return {"Method": "Cellular_Bounds", "Marker": "special_token", "Seq_Length": N, "Trial": t, "TAR": tar, "SAR": sar}


special_tasks = [(N, t) for N in ABLATION_SEQUENCE_LENGTHS for t in range(20)]
print(f"Starting: {len(special_tasks)} evaluations (marker={SPECIAL_MARKER!r})...")
special_results = []
for N, t in tqdm(special_tasks, desc="(C) special token"):
    try:
        special_results.append(run_special_trial(N, t))
    except Exception as exc:
        print(f"Task failed: N={N}, t={t} | Error: {exc}")

special_df = pd.DataFrame(special_results)
special_csv = os.path.join(RESULTS_DIR, "boundary_ablation_special_token_grid.csv")
special_df.to_csv(special_csv, index=False)
print(f"Saved {special_csv}")
print(special_df.groupby("Seq_Length")[["TAR", "SAR"]].mean().round(4).to_string())

# --- comparison ---
try:
    base_df = pd.read_csv(os.path.join(RESULTS_DIR, "thermodynamics_grid_templated_v2.csv"))
    base_ref = base_df[(base_df.Method == "Base") & (base_df.Alphabet_Size == ALPHABET_SIZE)].groupby("Seq_Length")["TAR"].mean()
    printable_mean = ablation_df[ablation_df.Placement == "periodic"].groupby(["Marker", "Seq_Length"])["TAR"].mean().groupby("Seq_Length").mean()
    special_mean = special_df.groupby("Seq_Length")["TAR"].mean()
    comparison = pd.DataFrame({"Base_TAR": base_ref, "Mean_Printable_Marker_TAR": printable_mean, "Special_Token_TAR": special_mean})
    print(comparison.round(4).to_string())
    comparison.to_csv(os.path.join(RESULTS_DIR, "special_token_comparison.csv"))
except Exception as exc:
    print(f"Comparison step failed (non-fatal): {exc}")

# ============================================================
# (D) TOKEN-BUDGET-MATCHED MULTI-TOKEN MARKER
# ============================================================
print("\n" + "=" * 60)
print("(D) Token-budget-matched multi-token marker")
print("=" * 60)

BUDGET_MATCHED_INTERVAL = 12


def build_multitoken(raw_input, marker, interval):
    length = len(raw_input)
    boundary_positions = boundary_positions_periodic(length, interval)
    input_w_bounds, output_w_bounds = "", ""
    for i, char in enumerate(raw_input):
        shifted_char = engine.alphabet[(engine.char_to_idx[char] + 1) % ALPHABET_SIZE]
        input_w_bounds += char
        output_w_bounds += shifted_char
        if i in boundary_positions:
            input_w_bounds += marker
            output_w_bounds += marker
    prompt = f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(engine.alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {engine.alphabet[-1]} -> {engine.alphabet[0]}, and {engine.alphabet[0]} -> {engine.alphabet[1]}.
Constraint: The input contains structural boundary markers ({marker!r}). DO NOT shift them. Copy them exactly to the output.
No spaces, no other explanations.

Input: {input_w_bounds}
Output: """
    return input_w_bounds, output_w_bounds, prompt, len(boundary_positions)


def run_budget_trial(raw_input, N, t):
    inp, truth, prompt, n_boundaries = build_multitoken(raw_input, "<|>", BUDGET_MATCHED_INTERVAL)
    max_tokens_needed = int(N * 1.5) + 20 + n_boundaries * 3
    pred_full = chat_generate(prompt, max_tokens_needed)
    pred = pred_full.strip().split("Explanation")[0].split("\n")[0].strip()

    clean_pred = pred.replace("<|>", "").replace(" ", "").replace("\n", "")
    clean_truth = truth.replace("<|>", "")
    clean_pred = clean_pred[:len(clean_truth)]
    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0 for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0
    return {"Marker": "<|>", "Interval": BUDGET_MATCHED_INTERVAL, "Seq_Length": N, "Trial": t, "N_Boundaries": n_boundaries, "TAR": tar, "SAR": sar}


shared_sequences = {}
for N in ABLATION_SEQUENCE_LENGTHS:
    for t in range(20):
        rng = random.Random(70_000 * N + t)
        shared_sequences[(N, t)] = engine.generate_raw_input(N, rng=rng)

budget_tasks = [(N, t) for N in ABLATION_SEQUENCE_LENGTHS for t in range(20)]
print(f"Starting: {len(budget_tasks)} evaluations...")
budget_results = []
for N, t in tqdm(budget_tasks, desc="(D) budget-matched"):
    raw_input = shared_sequences[(N, t)]
    try:
        budget_results.append(run_budget_trial(raw_input, N, t))
    except Exception as exc:
        print(f"Task failed: N={N}, trial={t} | Error: {exc}")

budget_df = pd.DataFrame(budget_results)
budget_csv = os.path.join(RESULTS_DIR, "boundary_ablation_multitoken_budget_matched_grid.csv")
budget_df.to_csv(budget_csv, index=False)
print(f"Saved {budget_csv}")

baseline_pipe = ablation_df[(ablation_df.Marker == "|") & (ablation_df.Ablation_Axis == "marker+placement baseline")].groupby("Seq_Length")["TAR"].mean()
budget_summary = budget_df.groupby("Seq_Length")["TAR"].mean()
budget_comparison = pd.DataFrame({"Pipe_at_interval4_TAR": baseline_pipe, "TriToken_at_interval12_TAR_budget_matched": budget_summary})
print(budget_comparison.round(4).to_string())
budget_comparison.to_csv(os.path.join(RESULTS_DIR, "boundary_ablation_budget_matched_comparison.csv"))

# ============================================================
# (E) PROMPT-TEMPLATE ROBUSTNESS
# ============================================================
print("\n" + "=" * 60)
print("(E) Prompt-template robustness")
print("=" * 60)


def prompt_template_original(alphabet, input_str):
    return f"""You are a deterministic sequence transformer.
Task: Cyclic character shift.
Alphabet: {', '.join(alphabet)}
Rule: Replace every character with the next one in the alphabet.
Mapping: {alphabet[-1]} -> {alphabet[0]}, and {alphabet[0]} -> {alphabet[1]}.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT shift them. Copy them exactly to the output.
No spaces, no explanations.

Input: {input_str}
Output: """


def prompt_template_paraphrase_1(alphabet, input_str):
    return f"""You perform one job only: shifting letters forward in a fixed alphabet.
Letters in use: {', '.join(alphabet)}
For every letter you see, write the next letter in this list instead (wrapping from {alphabet[-1]} back to {alphabet[0]}).
Some characters in the input are pipes (|); these are dividers, not letters -- reproduce them unchanged in your answer.
Respond with the transformed string only. Do not add spaces or commentary.

Sequence: {input_str}
Result: """


def prompt_template_paraphrase_2(alphabet, input_str):
    return f"""Apply a fixed substitution cipher: {' -> '.join(alphabet[1:] + alphabet[:1])} -> {alphabet[0]}.
In plain terms, replace each symbol with the one that follows it in {', '.join(alphabet)}, cycling back to the start after the last symbol.
Any "|" characters present must be left exactly where they are -- they are not part of the alphabet.
Give only the resulting string, with no spaces and no explanation of your reasoning.

>>> {input_str}
<<< """


PROMPT_TEMPLATES = {"original": prompt_template_original, "paraphrase_1": prompt_template_paraphrase_1, "paraphrase_2": prompt_template_paraphrase_2}
ROBUST_SEQUENCE_LENGTHS = [8, 16, 32]
ROBUST_INTERVAL = 4
ROBUST_TRIALS = 15


def build_robust_variant(raw_input, interval):
    input_w_bounds, output_w_bounds = "", ""
    for i, char in enumerate(raw_input):
        shifted = engine.alphabet[(engine.char_to_idx[char] + 1) % ALPHABET_SIZE]
        input_w_bounds += char
        output_w_bounds += shifted
        if interval > 0 and (i + 1) % interval == 0 and i != len(raw_input) - 1:
            input_w_bounds += "|"
            output_w_bounds += "|"
    return input_w_bounds, output_w_bounds


def run_robust_trial(template_name, N, k, t):
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 110_000 * N + 100 * k + t
    rng = random.Random(local_seed)
    raw_input = engine.generate_raw_input(N, rng=rng)
    input_str, truth = build_robust_variant(raw_input, interval=k)
    prompt = PROMPT_TEMPLATES[template_name](engine.alphabet, input_str)
    max_tokens_needed = int(N * 1.5) + 20
    pred_full = chat_generate(prompt, max_tokens_needed)
    pred = pred_full.strip().split("Explanation")[0].split("\n")[0].strip()

    clean_pred = pred.replace("|", "").replace(" ", "").replace("\n", "")
    clean_truth = truth.replace("|", "")
    clean_pred = clean_pred[:len(clean_truth)]
    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0 for i in range(len(clean_truth))]
    tar = sum(binary_series) / len(clean_truth) if clean_truth else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0
    return dict(Template=template_name, Method=method, Interval=k, Seq_Length=N, Trial=t, TAR=tar, SAR=sar)


robust_tasks = [(tmpl, N, k, t) for tmpl in PROMPT_TEMPLATES for N in ROBUST_SEQUENCE_LENGTHS for k in [0, ROBUST_INTERVAL] for t in range(ROBUST_TRIALS)]
print(f"Starting: {len(robust_tasks)} evaluations...")
robust_results = []
for (tmpl, N, k, t) in tqdm(robust_tasks, desc="(E) prompt robustness"):
    try:
        robust_results.append(run_robust_trial(tmpl, N, k, t))
    except Exception as exc:
        print(f"Task failed: {(tmpl, N, k, t)} | Error: {exc}")

robust_df = pd.DataFrame(robust_results)
robust_csv = os.path.join(RESULTS_DIR, "prompt_robustness_grid.csv")
robust_df.to_csv(robust_csv, index=False)
print(f"Saved {robust_csv}")
print(robust_df.groupby(["Template", "Method", "Seq_Length"])[["TAR", "SAR"]].mean().round(4).to_string())

print("\nAll five marker-ablation experiments complete.")
