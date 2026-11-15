#!/usr/bin/env python3
"""Verbatim prompts for Appendix A, regenerated from the task engines with the runner's seeds
(N = 8, k = 4, trial 0; alphabet or modulus 4), so they are exactly what the models received.
Output: paper_tmlr/tables/prompts_verbatim.tex"""
import importlib
import random
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common.runner import _local_seed  # noqa: E402

ITEMS = [   # (task, title, axis, base task whose prompt it is shown against, or None for in full)
    ("conditional_shift", "\\task{conditional\\_shift}, letters", 4, None),
    ("conditional_shift_digits", "\\task{conditional\\_shift}, digits: lines that differ from the letter prompt", 4, "conditional_shift"),
    ("random_permutation", "\\task{random\\_permutation}, letters", 4, None),
    ("running_xor", "\\task{running\\_xor}, original wording", None, None),
    ("running_xor_neutral", "\\task{running\\_xor}, neutral wording: lines that differ", None, "running_xor"),
    ("running_xor_across", "\\task{running\\_xor}, across wording: lines that differ", None, "running_xor"),
    ("running_xor_worked", "\\task{running\\_xor}, across wording with worked examples: lines that differ", None, "running_xor"),
]
N, K, T = 8, 4, 0


def main():
    out = []
    def render(task, axis):
        e = importlib.import_module(f"tasks.{task}.engine")
        a = axis if e.AXIS_NAME else None
        rng = random.Random(_local_seed(N, K, a, T))
        inp, truth, prompt = e.build_prompt(N, K, rng, axis_val=a) if e.AXIS_NAME else e.build_prompt(N, K, rng)
        return prompt.rstrip().split("\n"), truth, a

    for task, title, axis, base in ITEMS:
        plines, truth, a = render(task, axis)
        if base:
            blines = set(render(base, axis)[0])
            plines = [l for l in plines if l not in blines]
        lines = []
        for line in plines:
            lines += textwrap.wrap(line, 92, break_long_words=False, break_on_hyphens=False) or [""]
        if not base:
            lines[-1] = lines[-1] + f"            (expected: {truth})"
        body = "\n".join(lines).replace("→", "->").replace("⊕", "^")
        out.append(f"\\par\\medskip\\noindent\\textbf{{{title} ($N = {N}$, $k = {K}${', $A = 4$' if a else ''}).}}\\par\n\\begin{{small}}\n\\begin{{verbatim}}\n{body}\n\\end{{verbatim}}\n\\end{{small}}\n")
    (REPO / "paper_tmlr" / "tables" / "prompts_verbatim.tex").write_text("\n".join(out))
    print("\n".join(out)[:3000])


if __name__ == "__main__":
    main()
