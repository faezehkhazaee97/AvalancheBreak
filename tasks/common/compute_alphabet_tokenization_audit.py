#!/usr/bin/env python3
"""Alphabet single-token audit (CPU-only, tokenizer-only -- no model weights
loaded). Extends the existing boundary-marker tokenization check
(results/marker_tokenization_cross_model.csv, which only covers the '|'
marker and 3-4 models) to every alphabet actually used across all 9 core
tasks, on all 5 documented models' tokenizers.

Why this matters: every mechanistic/attention analysis in the paper that
reasons about "the token at position i" implicitly assumes one alphabet
character == one token when the alphabet is used as it actually appears in
a task prompt. If characters merge into multi-character tokens, that
per-position assumption is silently wrong for however many characters got
folded together.

Method: token count is measured as (tokens of PREFIX+sequence) minus
(tokens of PREFIX alone), where PREFIX="Input: " mimics real prompt
structure. This avoids two failure modes of a naive isolated-string
encode(): (1) SentencePiece tokenizers (Mistral, Phi-3.5) inject a single
leading "_" (space) token when a string is encoded with no preceding
context at all -- an artifact of testing in isolation that doesn't occur
in an actual templated prompt, confirmed by checking the raw token list
(e.g. Phi-3.5 on a 60-digit string: ['_', '0', '1', '2', ...], with a
prefix that leading '_' disappears and every digit is its own token); (2)
diffing by naive string length after removing one character can misattribute
a token that actually spans several characters. Diffing (prefix+seq) vs
(prefix) sidesteps both, since the tokenizer boundary right after the
prefix's trailing character reliably starts fresh for both families.
"""

import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tasks.common.boundary import insert_boundaries  # noqa: E402

OUT_CSV = REPO_ROOT / "results_clean" / "_diagnostics" / "alphabet_tokenization_audit.csv"

MODELS = {
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct": "Qwen/Qwen2.5-14B-Instruct",
    "Qwen2.5-32B-Instruct": "Qwen/Qwen2.5-32B-Instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "Phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
}

PREFIX = "Input: "

# Representative sequences for each alphabet actually used in the 9-task
# grid (from each task's engine.py). Repeated x3 so k=16 boundary insertion
# still produces multiple boundaries, not just zero/one.
ALPHABETS = {
    "letters_A-H (size 8)": "ABCDEFGHDCBAHGFE" * 3,
    "letters_A-D (size 4)": "ABCDDCBAABCD" * 3,
    "letters_A-B (size 2)": "ABABBABA" * 3,
    "digits_0-7 (size 8)": "01234567765432107654" * 3,
    "digits_0-3 (size 4)": "01230321012301" * 3,
    "digits_0-1 (binary)": "0110100101101" * 3,
}


def tokens_for(tok, prefix_ids, text):
    return len(tok.encode(PREFIX + text, add_special_tokens=False)) - len(prefix_ids)


def main():
    rows = []
    for tag, model_id in MODELS.items():
        print(f"Loading tokenizer: {tag} ({model_id})")
        tok = AutoTokenizer.from_pretrained(model_id)
        prefix_ids = tok.encode(PREFIX, add_special_tokens=False)
        for label, seq in ALPHABETS.items():
            bare_tok = tokens_for(tok, prefix_ids, seq)
            bare_ok = bare_tok == len(seq)
            for k in (1, 2, 3, 4, 8, 16):
                bounded = insert_boundaries(seq, k)
                bounded_tok = tokens_for(tok, prefix_ids, bounded)
                bounded_ok = bounded_tok == len(bounded)
                rows.append({
                    "Model": tag, "Alphabet": label, "K": k,
                    "Bare_Seq_Len": len(seq), "Bare_Tokens": bare_tok,
                    "Bare_Bijective": bare_ok,
                    "Bounded_Seq_Len": len(bounded), "Bounded_Tokens": bounded_tok,
                    "Bounded_Bijective": bounded_ok,
                })
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_CSV}")

    bare_summary = df.drop_duplicates(["Model", "Alphabet"])
    n_fail_bare = (~bare_summary["Bare_Bijective"]).sum()
    n_fail_bounded = (~df["Bounded_Bijective"]).sum()
    print(f"\n{n_fail_bare}/{len(bare_summary)} (model, alphabet) bare sequences are NOT a "
          f"1:1 char<->token bijection")
    print(f"{n_fail_bounded}/{len(df)} (model, alphabet, k) boundary-inserted rows are NOT a "
          f"1:1 char<->token bijection")

    print("\nBare-sequence merge ratio (tokens/chars) by model x alphabet:")
    bare_summary = bare_summary.copy()
    bare_summary["Ratio"] = bare_summary["Bare_Tokens"] / bare_summary["Bare_Seq_Len"]
    print(bare_summary[["Model", "Alphabet", "Bare_Seq_Len", "Bare_Tokens", "Ratio"]]
          .sort_values(["Alphabet", "Model"]).to_string(index=False))

    print("\nAt what k does the boundary-inserted sequence become bijective again "
          "(letters only, worst case)?")
    letters = df[df.Alphabet.str.startswith("letters")]
    for (model, alpha), sub in letters.groupby(["Model", "Alphabet"]):
        ok_ks = sorted(sub[sub.Bounded_Bijective].K.tolist())
        bad_ks = sorted(sub[~sub.Bounded_Bijective].K.tolist())
        print(f"  {model:<26} {alpha:<22} bijective at k={ok_ks or 'none'}  "
              f"NOT bijective at k={bad_ks or 'none'}")


if __name__ == "__main__":
    main()
