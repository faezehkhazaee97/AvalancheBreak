"""Single, shared scoring implementation for all 6 tasks.

Fixes the truncation bug (Finding A): the broken modsum/running_sum_local
scripts captured the model's output with re.match(r'^(\\d+)', pred), which
stops at the first '|' while the ground truth still contains bars, so every
position after the first boundary auto-scored wrong regardless of what the
model produced.

This module instead strips '|' from BOTH the raw output and the ground
truth before comparing -- the convention already used (and cited in the
paper's own "Scoring conventions" text) by the real cyclic-shift and modsum
headline scripts (run_core_templated_v2.py, run_modsum_templated_v2.py).
Boundaries are structural, not part of the answer, on both sides of the
comparison.
"""

import re


def score(raw_output, truth):
    """raw_output: full decoded generation (untruncated). truth: ground truth
    (may contain '|' at boundary positions for tasks where boundaries are
    echoed in the expected output).

    Returns dict with clean_pred, clean_truth, TAR, SAR, First_Error_Pos.
    """
    pred = raw_output.strip()
    pred = pred.split("Explanation")[0].split("\n")[0].strip()

    clean_pred = pred.replace("|", "").replace(" ", "").replace("\n", "")
    clean_truth = truth.replace("|", "")
    clean_pred = clean_pred[: len(clean_truth)]

    binary_series = [
        1 if i < len(clean_pred) and clean_pred[i] == clean_truth[i] else 0
        for i in range(len(clean_truth))
    ]
    tar = sum(binary_series) / len(clean_truth) if len(clean_truth) > 0 else 0.0
    sar = 1.0 if (len(binary_series) == len(clean_truth) and sum(binary_series) == len(binary_series)) else 0.0
    first_error = binary_series.index(0) if 0 in binary_series else len(binary_series)

    return {
        "clean_pred": clean_pred,
        "clean_truth": clean_truth,
        "TAR": tar,
        "SAR": sar,
        "First_Error_Pos": first_error,
    }


def score_tokens(raw_output, truth, sep=", ", token_regex=None):
    """Token-level counterpart to score(), for tasks whose natural unit is a
    multi-character token (CLRS-Text trace steps) rather than a single
    character -- same conventions otherwise: boundaries ("|") are structural
    noise stripped from both sides before comparing (never validated for
    position, exactly like score()'s character-level "|" stripping),
    prediction clipped to truth's token count, TAR/SAR/First_Error_Pos now
    measured in tokens instead of chars.

    Model output formatting is not fully reliable -- models sometimes drop
    the space after a comma, or use "|" on its own as an implicit separator
    between tokens instead of ", ". Two parsing strategies, picked by
    whether the token shape has internal punctuation of its own:
      - token_regex=None (plain tokens, e.g. bare integers): split pred on
        any run of "," and/or "|" with optional whitespace, tolerating
        either convention.
      - token_regex=<pattern> (structured tokens, e.g. "(2,3)" tuples, which
        contain their own internal comma): re.findall the pattern directly
        instead of splitting on any delimiter, sidestepping the delimiter
        ambiguity entirely since the token's own syntax marks its boundary.

    truth: the token sequence joined with `sep`, "|" tokens included at
    boundary positions (mirrors score()'s truth argument, which may contain
    "|" for tasks that echo boundaries in the expected output).
    """
    pred = raw_output.strip()
    pred = pred.split("Explanation")[0].split("\n")[0].strip()

    if token_regex:
        pred_tokens = re.findall(token_regex, pred)
    else:
        pred_tokens = [t.strip() for t in re.split(r"[,|]+\s*", pred) if t.strip() != ""]
    truth_tokens = [t.strip() for t in truth.split(sep) if t.strip() != "|"]
    pred_tokens = pred_tokens[: len(truth_tokens)]

    binary_series = [
        1 if i < len(pred_tokens) and pred_tokens[i] == truth_tokens[i] else 0
        for i in range(len(truth_tokens))
    ]
    tar = sum(binary_series) / len(truth_tokens) if len(truth_tokens) > 0 else 0.0
    sar = 1.0 if (len(binary_series) == len(truth_tokens) and sum(binary_series) == len(binary_series)) else 0.0
    first_error = binary_series.index(0) if 0 in binary_series else len(binary_series)

    return {
        "clean_pred": sep.join(pred_tokens),
        "clean_truth": sep.join(truth_tokens),
        "TAR": tar,
        "SAR": sar,
        "First_Error_Pos": first_error,
    }
