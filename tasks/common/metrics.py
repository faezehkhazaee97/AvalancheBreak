"""Failure-mode diagnostic metrics, computed retroactively from grid-mode CSVs.

Most of these are computed on top of the already-stored, already-scored
columns (LLM_Output, Ground_Truth, First_Error_Pos, Seq_Length) written by
common.scoring / common.runner. Two metrics -- Overlong and
Boundary_Fidelity -- need the full, untruncated Raw_Model_Output instead,
because common.scoring.score() strips information they depend on before a
row is ever written to a grid CSV:
  - over-length "rambling" output (silently clipped to len(clean_truth)
    before storage -- only under-length output is detectable without the
    raw text)
  - boundary-copy fidelity (the "|" characters are stripped from both
    prediction and truth before comparison, by the paper's own scoring
    convention -- whether the model actually echoed boundaries correctly
    at the right positions needs the raw, bar-inclusive text)
common.runner now always stores Raw_Model_Output (see runner.py), so these
two are only available on rows generated after that change -- older rows
without a Raw_Model_Output column simply get None for both.

Alphabets mirror each task's engine.py exactly:
  - letter tasks (cyclic_shift, character_substitution, random_permutation,
    conditional_shift, composed_permutation): first Alphabet_Size letters
    starting at 'A'
  - running_xor, copy_previous: binary {0,1}
  - running_sum_local: digits 0..Modulus-1
  - running_max: digits 0..Alphabet_Size-1
"""

import pandas as pd

from .boundary import insert_boundaries

LETTER_TASKS = {
    "cyclic_shift", "character_substitution", "random_permutation",
    "conditional_shift", "composed_permutation",
}
BINARY_TASKS = {"running_xor", "copy_previous"}
MODULUS_TASKS = {"running_sum_local"}
ALPHABET_SIZE_DIGIT_TASKS = {"running_max"}


def valid_alphabet(task_name, row):
    """Returns the set of valid single characters for this row's condition."""
    if task_name in LETTER_TASKS:
        size = int(row["Alphabet_Size"])
        return set(chr(ord("A") + i) for i in range(size))
    if task_name in BINARY_TASKS:
        return {"0", "1"}
    if task_name in MODULUS_TASKS:
        modulus = int(row["Modulus"])
        return set(str(d) for d in range(modulus))
    if task_name in ALPHABET_SIZE_DIGIT_TASKS:
        size = int(row["Alphabet_Size"])
        return set(str(d) for d in range(size))
    raise ValueError(f"Unknown task: {task_name}")


def is_vocab_valid(output_str, alphabet):
    """True iff every character in output_str is in the task's alphabet.
    (Boundaries are already stripped from LLM_Output before storage, so the
    alphabet does not need to include '|'.)"""
    s = str(output_str)
    if s == "" or s.lower() == "nan":
        return False
    return all(ch in alphabet for ch in s)


def is_complete(output_str, truth_str):
    """True iff the (already-clipped) output is at least as long as truth --
    i.e. the model did not stop short. Cannot detect over-length rambling
    (already clipped upstream before storage)."""
    return len(str(output_str)) >= len(str(truth_str))


def normalized_error_onset(first_error_pos, seq_length):
    """First_Error_Pos / Seq_Length, clipped to [0, 1]. Note First_Error_Pos
    is measured against len(clean_truth), which for boundary conditions is
    the content length (boundaries stripped) -- same basis as Seq_Length."""
    if seq_length <= 0:
        return None
    return min(first_error_pos / seq_length, 1.0)


def _clean_first_line(raw_output, keep_bars):
    """Mirrors common.scoring.score()'s first-line/whitespace cleanup, minus
    the length-clip -- and optionally keeps '|' (score() always strips it;
    boundary_fidelity() needs it kept)."""
    s = str(raw_output).strip()
    s = s.split("Explanation")[0].split("\n")[0].strip()
    s = s.replace(" ", "").replace("\n", "")
    if not keep_bars:
        s = s.replace("|", "")
    return s


def is_overlong(raw_output, truth_content):
    """True iff the model's full (untruncated) output content is longer than
    the expected content length, once cleaned the same way scoring.score()
    cleans it -- but without score()'s length-clip, since that clip is
    exactly what makes over-length rambling invisible in stored LLM_Output.
    None if raw_output isn't available."""
    if raw_output is None or (isinstance(raw_output, float) and pd.isna(raw_output)):
        return None
    content = _clean_first_line(raw_output, keep_bars=False)
    return len(content) > len(str(truth_content))


def boundary_fidelity(raw_output, truth_content, interval):
    """Fraction of the boundary ('|') positions the model's raw output
    actually reproduces as '|' at the same index, compared against the
    bar-inclusive expected output (reconstructed from the already-stored
    Ground_Truth + Interval via insert_boundaries -- no need to separately
    store the bar-inclusive truth). None if raw_output isn't available or
    this is a Base row (interval==0, no boundaries expected)."""
    if raw_output is None or (isinstance(raw_output, float) and pd.isna(raw_output)):
        return None
    interval = int(interval)
    if interval <= 0:
        return None
    truth_bounded = insert_boundaries(str(truth_content), interval)
    bar_positions = [i for i, ch in enumerate(truth_bounded) if ch == "|"]
    if not bar_positions:
        return None
    raw_bounded = _clean_first_line(raw_output, keep_bars=True)
    hits = sum(1 for i in bar_positions if i < len(raw_bounded) and raw_bounded[i] == "|")
    return hits / len(bar_positions)


def _first_error_pos(output_str, truth_str):
    """Recomputes First_Error_Pos the same way common.scoring.score() does,
    for rows/files that predate that column being written."""
    pred, truth = str(output_str), str(truth_str)
    for i in range(len(truth)):
        if i >= len(pred) or pred[i] != truth[i]:
            return i
    return len(truth)


def row_diagnostics(task_name, row):
    """Compute all per-row diagnostic flags for one grid-CSV row (a pandas
    Series or dict-like with LLM_Output, Ground_Truth, Seq_Length, and the
    task's axis column if any; First_Error_Pos is recomputed if absent)."""
    alphabet = valid_alphabet(task_name, row)
    output_str = row["LLM_Output"]
    truth_str = row["Ground_Truth"]

    vocab_valid = is_vocab_valid(output_str, alphabet)
    complete = is_complete(output_str, truth_str)
    well_formed = vocab_valid and complete
    first_error_pos = row["First_Error_Pos"] if "First_Error_Pos" in row.index and not pd.isna(row["First_Error_Pos"]) else _first_error_pos(output_str, truth_str)
    neo = normalized_error_onset(first_error_pos, row["Seq_Length"])

    raw_output = row["Raw_Model_Output"] if "Raw_Model_Output" in row.index else None
    interval = row["Interval"] if "Interval" in row.index else 0
    overlong = is_overlong(raw_output, truth_str)
    bfr = boundary_fidelity(raw_output, truth_str, interval)

    return {
        "Vocab_Valid": vocab_valid,
        "Complete": complete,
        "Well_Formed": well_formed,
        "Normalized_Error_Onset": neo,
        "Overlong": overlong,
        "Boundary_Fidelity": bfr,
    }
