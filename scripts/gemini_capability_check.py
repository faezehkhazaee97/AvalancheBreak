"""
Does a frontier proprietary model (Gemini) still show boundaries hurting
addition / Pauli string, the way Qwen 7B/14B and Mistral-7B did locally?

Motivation: Qwen2.5-14B and Mistral-7B are both dramatically better than
Qwen2.5-7B at the raw addition/modsum task (e.g. addition Base TAR ~0.97-0.98
vs 7B's ~0.81), yet the boundary penalty doesn't shrink -- if anything it
gets worse in absolute terms. That rules out "model too small" as the
explanation for THOSE three models. This script extends the same check to
a genuinely frontier-class model via API, without needing attention access
(TAR/SAR only need text in, text out) -- if boundaries still hurt addition
on Gemini specifically, that's strong evidence this is structural (carry
direction), not a capability ceiling. Pauli string is included too, since
locally even Qwen 14B got the position-local sub-task exactly right in
~0% of trials (String_Correct ~ 0.00-0.02) -- a stronger model might
actually attempt that task meaningfully, unlike Qwen/Mistral.

TaskEngine classes below are copied verbatim from the project notebook
(LLM-Focus.ipynb, cells 25 and 32) so the task itself is byte-for-byte
identical to what was already run locally -- only the model-calling layer
changes, from `model.generate(...)` to the Gemini API.

Setup:
    pip install google-genai
    export GEMINI_API_KEY=...   (https://aistudio.google.com/apikey)

Usage:
    python scripts/gemini_capability_check.py            # prints cost estimate, asks to confirm
    python scripts/gemini_capability_check.py --yes       # skips confirmation
    python scripts/gemini_capability_check.py --model gemini-2.5-flash   # cheaper/faster, lower ceiling
"""

import argparse
import os
import random
import re
import time

import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIG -- deliberately small for a first pass. Bump TRIALS / lengths /
# intervals once you've seen this run cleanly and decided it's worth the
# extra cost.
# ============================================================
MODEL_NAME = "gemini-2.5-pro"       # override with --model gemini-2.5-flash for a cheaper first look
RESULTS_DIR = "results/gemini-2.5-pro"

ADD_SEQUENCE_LENGTHS = [8, 16, 32]
ADD_PERIODIC_INTERVALS = [0, 4]      # Base, and the interval most used in the local addition analysis
ADD_TRIALS = 10

PAULI_SEQUENCE_LENGTHS = [8, 16, 32]
PAULI_PERIODIC_INTERVALS = [0, 4]
PAULI_TRIALS = 10

MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 4


# ============================================================
# TASK ENGINES -- copied verbatim from LLM-Focus.ipynb (cells 25, 32)
# ============================================================
class AdditionTaskEngine:
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

        prompt = f"""You are a deterministic arithmetic calculator.
Task: Add two {length}-digit numbers.
Constraint: The input may contain vertical bars (|) as structural boundaries -- they carry no numeric meaning, ignore them when computing.
Output ONLY the {length}-digit sum (zero-padded to {length} digits if needed). No spaces, no explanations.

Input: {input_str}
Output: """
        return input_str, sum_str, prompt


class PauliStringTaskEngine:
    PAULIS = ["I", "X", "Y", "Z"]
    MULT_TABLE = {
        ("I", "I"): ("I", 0), ("I", "X"): ("X", 0), ("I", "Y"): ("Y", 0), ("I", "Z"): ("Z", 0),
        ("X", "I"): ("X", 0), ("Y", "I"): ("Y", 0), ("Z", "I"): ("Z", 0),
        ("X", "X"): ("I", 0), ("Y", "Y"): ("I", 0), ("Z", "Z"): ("I", 0),
        ("X", "Y"): ("Z", 1), ("Y", "Z"): ("X", 1), ("Z", "X"): ("Y", 1),
        ("Y", "X"): ("Z", 3), ("Z", "Y"): ("X", 3), ("X", "Z"): ("Y", 3),
    }
    PHASE_SYMBOLS = ["+", "i", "-", "j"]

    def sample_strings(self, length, rng):
        a = rng.choices(self.PAULIS, k=length)
        b = rng.choices(self.PAULIS, k=length)
        return a, b

    def multiply(self, a, b):
        result_ops = []
        total_phase = 0
        for pa, pb in zip(a, b):
            op, phase = self.MULT_TABLE[(pa, pb)]
            result_ops.append(op)
            total_phase = (total_phase + phase) % 4
        return "".join(result_ops), self.PHASE_SYMBOLS[total_phase]

    def build_prompt(self, length, interval, rng=None):
        rng = rng or random
        a, b = self.sample_strings(length, rng)
        result_string, phase_symbol = self.multiply(a, b)
        truth = result_string + phase_symbol

        a_str, b_str = "".join(a), "".join(b)
        combined = list(a_str) + ["*"] + list(b_str)
        bounded = []
        for i, d in enumerate(combined):
            bounded.append(d)
            if interval > 0 and (i + 1) % interval == 0 and i != len(combined) - 1:
                bounded.append("|")
        input_str = "".join(bounded)

        prompt = f"""You are a deterministic Pauli algebra calculator.
Task: multiply two {length}-operator Pauli strings, position by position.
Rules: I is the identity (I*P = P for any P). P*P = I for P in {{X, Y, Z}}. X*Y=iZ, Y*Z=iX, Z*X=iY. Y*X=-iZ, Z*Y=-iX, X*Z=-iY.
Alphabet: I, X, Y, Z. The symbol * marks where the first string ends and the second begins.
Constraint: The input may contain vertical bars (|) as structural boundaries -- they carry no meaning, ignore them when computing.
Output the resulting {length}-operator string, immediately followed by ONE phase symbol for the TOTAL accumulated phase across all positions (multiply every per-position phase together): '+' for +1, 'i' for +i, '-' for -1, 'j' for -i. No spaces, no explanations.

Input: {input_str}
Output: """
        return input_str, truth, prompt


# ============================================================
# GEMINI BACKEND
# ============================================================
def call_gemini(client, model_name, prompt, max_output_tokens):
    from google.genai import types
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,           # match local greedy decoding as closely as the API allows
                    max_output_tokens=max_output_tokens,
                ),
            )
            return (resp.text or "").strip()
        except Exception as exc:
            wait = RETRY_BACKOFF_SEC * (2 ** attempt)
            print(f"  [retry {attempt+1}/{MAX_RETRIES}] {exc} -- waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} retries")


# ============================================================
# TRIAL RUNNERS
# ============================================================
def run_addition_trial(client, model_name, engine, N, k, t):
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 90_000 * N + 100 * k + t
    rng = random.Random(local_seed)
    inp, truth, prompt = engine.build_prompt(N, interval=k, rng=rng)

    raw = call_gemini(client, model_name, prompt, max_output_tokens=N + 40)
    pred = raw.split("Explanation")[0].split("\n")[0].strip()
    match = re.match(r"^(\d+)", pred)
    clean_pred = match.group(1)[:len(truth)] if match else ""

    binary = [1 if i < len(clean_pred) and clean_pred[i] == truth[i] else 0 for i in range(len(truth))]
    tar = sum(binary) / len(truth) if truth else 0.0
    sar = 1.0 if (len(binary) == len(truth) and sum(binary) == len(binary)) else 0.0
    return {"Method": method, "Interval": k, "Seq_Length": N, "Trial": t, "TAR": tar, "SAR": sar,
            "Input_Given": inp, "Ground_Truth": truth, "LLM_Output": clean_pred}


def run_pauli_trial(client, model_name, engine, N, k, t):
    method = "Base" if k == 0 else "Cellular_Bounds"
    local_seed = 170_000 * N + 100 * k + t
    rng = random.Random(local_seed)
    inp, truth, prompt = engine.build_prompt(N, interval=k, rng=rng)

    raw = call_gemini(client, model_name, prompt, max_output_tokens=N + 20)
    pred = raw.split("Explanation")[0].split("\n")[0].strip()
    match = re.match(r"^([IXYZ+ij-]+)", pred)
    clean_pred = match.group(1)[:len(truth)] if match else ""

    binary = [1 if i < len(clean_pred) and clean_pred[i] == truth[i] else 0 for i in range(len(truth))]
    tar = sum(binary) / len(truth) if truth else 0.0
    sar = 1.0 if (len(binary) == len(truth) and sum(binary) == len(binary)) else 0.0
    string_correct = clean_pred[:len(truth) - 1] == truth[:-1] if len(clean_pred) >= len(truth) - 1 else False
    phase_correct = len(clean_pred) == len(truth) and clean_pred[-1] == truth[-1]
    return {"Method": method, "Interval": k, "Seq_Length": N, "Trial": t, "TAR": tar, "SAR": sar,
            "String_Correct": string_correct, "Phase_Correct": phase_correct,
            "Input_Given": inp, "Ground_Truth": truth, "LLM_Output": clean_pred}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--yes", action="store_true", help="skip the cost-estimate confirmation prompt")
    parser.add_argument("--skip-pauli", action="store_true")
    parser.add_argument("--skip-addition", action="store_true")
    args = parser.parse_args()

    add_tasks = [] if args.skip_addition else [
        (N, k, t) for N in ADD_SEQUENCE_LENGTHS for k in ADD_PERIODIC_INTERVALS for t in range(ADD_TRIALS)
    ]
    pauli_tasks = [] if args.skip_pauli else [
        (N, k, t) for N in PAULI_SEQUENCE_LENGTHS for k in PAULI_PERIODIC_INTERVALS for t in range(PAULI_TRIALS)
    ]
    total_calls = len(add_tasks) + len(pauli_tasks)

    print(f"Model: {args.model}")
    print(f"Addition trials queued: {len(add_tasks)}  (lengths={ADD_SEQUENCE_LENGTHS}, intervals={ADD_PERIODIC_INTERVALS}, trials={ADD_TRIALS})")
    print(f"Pauli string trials queued: {len(pauli_tasks)}  (lengths={PAULI_SEQUENCE_LENGTHS}, intervals={PAULI_PERIODIC_INTERVALS}, trials={PAULI_TRIALS})")
    print(f"Total API calls: {total_calls}")
    print("Check current per-token pricing at https://ai.google.dev/gemini-api/docs/pricing before running --yes;"
          " prompts here are short (well under 1K input tokens each) but cost still scales with total_calls above.")

    if not args.yes:
        resp = input("\nProceed? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("Set GEMINI_API_KEY in your environment first (https://aistudio.google.com/apikey)")

    from google import genai
    client = genai.Client()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if add_tasks:
        add_engine = AdditionTaskEngine()
        add_results = []
        for (N, k, t) in tqdm(add_tasks, desc="Addition (Gemini)"):
            try:
                add_results.append(run_addition_trial(client, args.model, add_engine, N, k, t))
            except Exception as exc:
                print(f"Task failed: N={N}, k={k}, t={t} | {exc}")
        add_df = pd.DataFrame(add_results)
        add_csv = os.path.join(RESULTS_DIR, "addition_grid_api.csv")
        add_df.to_csv(add_csv, index=False)
        print(f"\nSaved {add_csv}")
        summary = add_df.groupby(["Seq_Length", "Method"])["TAR"].mean().unstack()
        if "Base" in summary.columns and "Cellular_Bounds" in summary.columns:
            summary["Gap"] = summary["Cellular_Bounds"] - summary["Base"]
        print("\n--- ADDITION: TAR by length/method ---")
        print(summary.round(4).to_string())

    if pauli_tasks:
        pauli_engine = PauliStringTaskEngine()
        pauli_results = []
        for (N, k, t) in tqdm(pauli_tasks, desc="Pauli string (Gemini)"):
            try:
                pauli_results.append(run_pauli_trial(client, args.model, pauli_engine, N, k, t))
            except Exception as exc:
                print(f"Task failed: N={N}, k={k}, t={t} | {exc}")
        pauli_df = pd.DataFrame(pauli_results)
        pauli_csv = os.path.join(RESULTS_DIR, "pauli_string_grid_api.csv")
        pauli_df.to_csv(pauli_csv, index=False)
        print(f"\nSaved {pauli_csv}")
        summary = pauli_df.groupby(["Seq_Length", "Method"])[["TAR", "String_Correct"]].mean()
        print("\n--- PAULI STRING: TAR / String_Correct by length/method ---")
        print(summary.round(4).to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
