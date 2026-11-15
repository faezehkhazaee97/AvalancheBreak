#!/usr/bin/env python3
"""data/real/wikitext_natural.txt: WikiText-2 (raw, test split) paragraphs in lowercase keeping letters,
spaces, commas, and periods (other characters removed, spacing around punctuation normalized), one per
line, from the copy cached by scripts/fetch_real_data.py."""
import re
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parents[1]
pq = next((REPO / "hf_cache/hub/datasets--Salesforce--wikitext/snapshots").glob("*/wikitext-2-raw-v1/test-*.parquet"))
paras = []
for t in pd.read_parquet(pq)["text"].tolist():
    t = t.strip()
    if not t or t.startswith("="):
        continue
    t = t.lower().replace(" @-@ ", " ").replace(" @,@ ", "").replace(" @.@ ", "")
    t = re.sub(r"[^a-z ,.]", " ", t)
    t = re.sub(r"\s+([,.])", r"\1", t)          # "word ," -> "word,"
    t = re.sub(r"([,.])(?=[a-z])", r"\1 ", t)    # ",word" -> ", word"
    t = re.sub(r"[ ]+", " ", t).strip()
    t = re.sub(r"^[ ,.]+", "", t)
    if sum(c.isalpha() for c in t) >= 64:
        paras.append(t)
(REPO / "data/real/wikitext_natural.txt").write_text("\n".join(paras) + "\n")
print(f"wikitext_natural: {len(paras)} paragraphs; sample: {paras[0][:160]!r}")
