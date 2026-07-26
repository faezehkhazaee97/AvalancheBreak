"""Exploratory side-test (NOT part of the main clean pipeline): does
restricting generation to only the task's valid output alphabet
(0, 1, |, plus required special/EOS tokens) fix the stray-out-of-alphabet /
repetition-collapse behavior seen in running_xor's unconstrained verify run?

Reuses the exact same grid and per-cell seeds as
results_clean/running_xor/verify/Qwen2.5-7B-Instruct.csv (via
tasks.common.runner's _build_grid/_local_seed), so this is a clean paired
comparison against already-reviewed baseline data -- only the decoding
constraint differs.

Background: a prior, different experiment in this repo (banning tokens
containing '|' to test an echo-vs-segmentation question) tried a similar
vocabulary-masking technique and found it caused decoding instability
(repetition collapse, stray out-of-alphabet characters, accuracy WORSE than
even the no-boundary baseline at every tested length). That experiment's
second attempt also broke by accidentally banning the model's own special
tokens (including EOS), so the model could never terminate -- this script
explicitly excludes tokenizer.all_special_ids from the ban to avoid
repeating that specific bug.
"""

import os
import random
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from tasks.common import scoring as scoring_mod
from tasks.common.runner import _build_grid, _local_seed, _seed_everything
import tasks.running_xor.engine as running_xor

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ALLOWED_CHARS = set("01|")


class AlphabetConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, allowed_chars):
        vocab_size = len(tokenizer)
        allowed_ids = set(tokenizer.all_special_ids)
        for token_id in range(vocab_size):
            text = tokenizer.decode([token_id])
            if text and all(c in allowed_chars for c in text):
                allowed_ids.add(token_id)
        self.allowed_ids = torch.tensor(sorted(allowed_ids), dtype=torch.long)
        print(f"  Constrained vocab: {len(self.allowed_ids)} / {vocab_size} tokens allowed "
              f"({len(tokenizer.all_special_ids)} special + rest are pure '{''.join(sorted(allowed_chars))}' merges)")

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_ids.to(scores.device)] = 0
        return scores + mask


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()

    processor = AlphabetConstrainedLogitsProcessor(tokenizer, ALLOWED_CHARS)
    logits_processor = LogitsProcessorList([processor])

    _seed_everything()
    grid = _build_grid(running_xor, trials_per_cell=1)
    print(f"Grid size: {len(grid)} evaluations (matches the existing verify pass)")

    rows = []
    for N, k, axis_val, t in grid:
        method = "Base" if k == 0 else "Cellular_Bounds"
        seed = _local_seed(N, k, axis_val, t)
        rng = random.Random(seed)
        input_str, truth, prompt = running_xor.build_prompt(N, k, rng)

        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        inputs = {kk: vv.to(model.device) for kk, vv in enc.items()}
        max_new = int(N * 1.5) + 20

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.eos_token_id, logits_processor=logits_processor,
            )
        gen_only = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(gen_only, skip_special_tokens=True)

        leaked = any(c not in ALLOWED_CHARS for c in raw_output)
        result = scoring_mod.score(raw_output, truth)

        rows.append({
            "Method": method, "Interval": k, "Seq_Length": N, "Trial": t,
            "Input_Given": input_str, "Ground_Truth": result["clean_truth"],
            "LLM_Output_Constrained": result["clean_pred"], "TAR_Constrained": result["TAR"],
            "SAR_Constrained": result["SAR"], "Leaked_OOV_Char": leaked,
            "Raw_Model_Output_Constrained": raw_output,
        })
        print(f"  N={N} k={k} t={t}: raw={raw_output!r} TAR={result['TAR']:.3f} leaked={leaked}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(REPO_ROOT, "results_clean", "running_xor", "verify", "Qwen2.5-7B-Instruct_constrained_test.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")
    print(f"\nLeak rate (should be 0): {df['Leaked_OOV_Char'].mean():.4f}")
    print("\n--- TAR by length/method (constrained) ---")
    print(df.groupby(["Seq_Length", "Method"])["TAR_Constrained"].mean())


if __name__ == "__main__":
    main()
