"""Single, shared boundary-insertion implementation for all 6 tasks.

Fixes the degenerate-grid-cell bug (Finding C): boundary insertion into a
string of length N produces zero "|" characters whenever k >= N, because the
only candidate insertion position is excluded as "trailing". A row generated
with such a k was previously labeled Method="Cellular_Bounds" but was
structurally identical to Base. valid_intervals() enforces k < N so those
rows are never generated in the first place, matching the filter already
used (and proven correct) by the paper's real cyclic-shift headline script.
"""

STANDARD_LENGTHS = [4, 8, 16, 32, 64]
STANDARD_INTERVALS = [0, 1, 2, 3, 4, 8, 16]


def valid_intervals(N, intervals=STANDARD_INTERVALS):
    """Intervals that produce a real (non-degenerate) boundary for length N."""
    return [k for k in intervals if k == 0 or k < N]


def insert_boundaries(seq, interval):
    """Insert '|' into seq every `interval` positions. No-op if interval<=0."""
    if interval <= 0:
        return seq
    bounded = []
    for i, ch in enumerate(seq):
        bounded.append(ch)
        if (i + 1) % interval == 0 and i != len(seq) - 1:
            bounded.append("|")
    return "".join(bounded)


def insert_boundaries_tokens(tokens, interval):
    """Token-level counterpart to insert_boundaries, for tasks whose natural
    unit is a multi-character token (e.g. a CLRS-Text trace step like "3" or
    "(2,2)") rather than a single character. Same placement rule as
    insert_boundaries -- "|" inserted as its own token every `interval`
    tokens, never trailing. No-op if interval<=0."""
    if interval <= 0:
        return list(tokens)
    bounded = []
    for i, tok in enumerate(tokens):
        bounded.append(tok)
        if (i + 1) % interval == 0 and i != len(tokens) - 1:
            bounded.append("|")
    return bounded
