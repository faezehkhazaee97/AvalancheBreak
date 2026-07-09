import matplotlib
matplotlib.use("Agg")

# ============================================================
# ADDITION WITH A SCRATCHPAD, WITH A MORE PERMISSIVE PARSER
# ============================================================
# Reviewer suggestion: the original scratchpad run only found a clean
# "Final Answer: <N digits>" line in 52.7% of responses, scoring
# everything else as a total miss even when the model's scratchpad
# reasoning clearly arrived at (or near) the right answer without using
# the exact literal phrasing asked for. This reruns the identical task
# (same prompt, same seeds, same lengths/intervals/trials) but with a
# two-stage parser: try the strict "Final Answer:" regex first, and if
# that fails, fall back to the LAST N-digit number appearing anywhere in
# the response. The raw response text is now always saved (previously
# only saved for successfully-parsed trials), so parser choices can be
# audited or changed after the fact without regenerating.

import os

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG = MODEL_ID.split("/")[-1]
RESULTS_DIR = os.path.join("results", MODEL_TAG)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_ID}")
print(f"Results directory: {RESULTS_DIR}/")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random, re
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


class ScratchpadAdditionTaskEngine:
    def _sample_pair(self, length, rng):
        for _ in range(200):
            a = [rng.randrange(10) for _ in range(length)]
            b = [rng.randrange(10) for _ in range(length)]
            if a[0] == 0: a[0] = rng.randrange(1, 10)
            if b[0] == 0: b[0] = rng.randrange(1, 10)
            carry = 0
            total = []
            for i in range(length - 1, -1, -1):
                s = a[i] + b[i] + carry
                total.append(s % 10)
                carry = s // 10
            if carry == 0:
                return a, b, list(reversed(total))
        raise RuntimeError(f"failed to sample a non-overflowing {length}-digit pair")

    def build_prompt(self, length, interval, rng=None):
        rng = rng or random
        a, b, total = self._sample_pair(length, rng)
        a_str = "".join(map(str, a))
        b_str = "".join(map(str, b))
        sum_str = "".join(map(str, total))

        combined = list(a_str) + ["+"] + list(b_str)
        bounded = []
        for i, d in enumerate(combined):
            bounded.append(d)
            if interval > 0 and (i + 1) % interval == 0 and i != len(combined) - 1:
                bounded.append("|")
        input_str = "".join(bounded)

        prompt = f"""You are a careful arithmetic assistant.
Task: Add two {length}-digit numbers.
Constraint: The input may contain vertical bars (|) as structural boundaries -- they carry no numeric meaning, ignore them when computing.
First, work through the addition step by step from the rightmost (least-significant) digit to the left, tracking the carry at each step. Then, on its own line, write exactly:
Final Answer: <the {length}-digit sum, zero-padded if needed>

Input: {input_str}
"""
        return input_str, sum_str, prompt


SCRATCH_SEQUENCE_LENGTHS = [4, 8, 16]
SCRATCH_PERIODIC_INTERVALS = [0, 4, 8, 16]
SCRATCH_TRIALS = 15

engine_scratch = ScratchpadAdditionTaskEngine()

def run_scratchpad_trial(task_params):
    N, k, t = task_params
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 150_000 * N + 100 * k + t
    rng = random.Random(local_seed)

    inp, truth, prompt = engine_scratch.build_prompt(N, interval=k, rng=rng)

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = {kk: vv.to(model.device) for kk, vv in enc.items()}

    max_tokens_needed = N * 20 + 60
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_only = outputs[0][inputs["input_ids"].shape[1]:]
    pred_full = tokenizer.decode(gen_only, skip_special_tokens=True)

    strict_match = re.search(r'Final Answer:\s*(\d+)', pred_full)
    if strict_match:
        clean_pred = strict_match.group(1)[:len(truth)]
        parse_method = "strict"
    else:
        # Fallback: the last standalone run of digits at least as long as
        # the expected answer, anywhere in the response (the model often
        # states the correct sum in its scratchpad reasoning without ever
        # producing the literal "Final Answer:" line).
        all_digit_runs = re.findall(r'\d+', pred_full)
        candidates = [d for d in all_digit_runs if len(d) >= len(truth)]
        if candidates:
            clean_pred = candidates[-1][-len(truth):]
            parse_method = "fallback"
        elif all_digit_runs:
            clean_pred = all_digit_runs[-1][:len(truth)]
            parse_method = "fallback_short"
        else:
            clean_pred = ""
            parse_method = "none"

    binary_series = [1 if i < len(clean_pred) and clean_pred[i] == truth[i] else 0 for i in range(len(truth))]
    tar = sum(binary_series) / len(truth) if truth else 0.0
    sar = 1.0 if (len(binary_series) == len(truth) and sum(binary_series) == len(binary_series)) else 0.0
    first_error = binary_series.index(0) if 0 in binary_series else len(binary_series)

    return {
        "Raw_Response": pred_full,
        "Parse_Method": parse_method,
        "Method": method, "Interval": k, "Seq_Length": N, "Trial": t,
        "TAR": tar, "SAR": sar, "First_Error_Pos": first_error,
        "Input_Given": inp, "Ground_Truth": truth, "LLM_Output": clean_pred,
        "Found_Final_Answer_Line": strict_match is not None,
    }

scratch_tasks = [
    (N, k, t)
    for N in SCRATCH_SEQUENCE_LENGTHS
    for k in SCRATCH_PERIODIC_INTERVALS
    for t in range(SCRATCH_TRIALS)
    if k == 0 or k < 2 * N + 1
]
print(f"\nStarting scratchpad addition grid (permissive parser): {len(scratch_tasks)} evaluations...")

scratch_results = []
for task in tqdm(scratch_tasks, desc="Scratchpad addition (permissive)"):
    try:
        scratch_results.append(run_scratchpad_trial(task))
    except Exception as exc:
        print(f"Task failed: {task} | Error: {exc}")

scratch_df = pd.DataFrame(scratch_results)
scratch_csv = os.path.join(RESULTS_DIR, "addition_scratchpad_grid_permissive.csv")
scratch_df.to_csv(scratch_csv, index=False)
print(f"\nScratchpad addition grid search complete. Saved to '{scratch_csv}'")
print(f"\nParse method breakdown:\n{scratch_df['Parse_Method'].value_counts().to_string()}")

summary = scratch_df.groupby(["Seq_Length", "Method"])[["TAR", "SAR"]].mean().reset_index()
print("\n--- AVERAGE ACCURACY BY LENGTH & METHOD (scratchpad, templated) ---")
print(summary.to_string(index=False))
print(f"\nFound a clean 'Final Answer:' line in {scratch_df['Found_Final_Answer_Line'].mean()*100:.1f}% of responses.")

pivot = summary.pivot(index="Seq_Length", columns="Method", values="TAR")
if "Base" in pivot.columns and "Cellular_Bounds" in pivot.columns:
    pivot["Cellular_minus_Base"] = pivot["Cellular_Bounds"] - pivot["Base"]
    print("\n--- DOES CELLULAR STILL SCORE WORSE THAN BASE, WITH A SCRATCHPAD? (templated) ---")
    print(pivot.round(4).to_string())
    n_worse = int((pivot["Cellular_minus_Base"] < 0).sum())
    print(f"\nCellular still worse than Base at {n_worse}/{len(pivot)} tested lengths.")

print("\nTemplated scratchpad addition grid rerun complete.")
