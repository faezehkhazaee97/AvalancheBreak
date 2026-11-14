#!/usr/bin/env python3
"""Fetch the two public sources behind tasks/dna_* and tasks/text_* into data/real/ (needs network;
run inside a job: sbatch --job-name=fetch-real scripts/run_cpu_script.sbatch scripts/fetch_real_data.py).

  promoters.txt        InstaDeepAI/nucleotide_transformer_downstream_tasks_revised, promoter_all/test.fna
  wikitext_letters.txt Salesforce/wikitext, wikitext-2-raw-v1 test split, paragraphs -> lowercase a-z
"""
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parents[1] / "data" / "real"
OUT.mkdir(parents=True, exist_ok=True)

fna = hf_hub_download("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised", "promoter_all/test.fna", repo_type="dataset")
seqs, cur = [], []
for line in Path(fna).read_text().splitlines():
    if line.startswith(">"):
        if cur:
            seqs.append("".join(cur).upper())
        cur = []
    else:
        cur.append(line.strip())
if cur:
    seqs.append("".join(cur).upper())
seqs = [s for s in seqs if s and set(s) <= set("ACGT")]
(OUT / "promoters.txt").write_text("\n".join(seqs) + "\n")
print(f"promoters: {len(seqs)} sequences, lengths {min(map(len, seqs))}-{max(map(len, seqs))}")

pq = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
import pandas as pd
text = pd.read_parquet(pq)["text"].tolist()
paras = []
for t in text:
    t = t.strip()
    if not t or t.startswith("="):
        continue
    letters = re.sub(r"[^a-z]", "", t.lower())
    if len(letters) >= 64:
        paras.append(letters)
(OUT / "wikitext_letters.txt").write_text("\n".join(paras) + "\n")
print(f"wikitext: {len(paras)} paragraphs, {sum(map(len, paras))} letters")
