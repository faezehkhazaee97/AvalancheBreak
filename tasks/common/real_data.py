"""Real sequences for the non-synthetic tasks (tasks/dna_* and tasks/text_*).

Two public sources, fetched once by scripts/fetch_real_data.py into data/real/:
  promoters.txt        human promoter sequences (300 bp, test split of the promoter_all task of the
                       Nucleotide Transformer downstream benchmark, InstaDeepAI), one per line, ACGT only
  wikitext_letters.txt WikiText-2 (raw, test split) paragraphs reduced to lowercase a-z letters, one per line

window(pool, N, rng) draws one paragraph or sequence and one offset with the trial's rng, so every
window is a function of (task, length, spacing, trial) exactly like the synthetic inputs.
"""

from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "real"


@lru_cache(maxsize=None)
def pool(name, min_len=64):
    f = DATA / f"{name}.txt"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run scripts/fetch_real_data.py first")
    lines = [l.strip() for l in f.read_text().splitlines()]
    return [l for l in lines if len(l) >= min_len]


def window(name, N, rng):
    p = pool(name)
    s = p[rng.randrange(len(p))]
    i = rng.randrange(len(s) - N + 1)
    return s[i:i + N]
