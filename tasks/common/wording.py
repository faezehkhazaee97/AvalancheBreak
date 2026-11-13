"""Instruction-wording variants for the marker constraint (tests whether the blind spot
depends on calling the markers "structural boundaries").

  neutral: only asks to copy each bar through, with no "structural boundaries" framing.
  across:  states that bars are only visual separators and that the rule applies across them
           (the digit after a bar uses the digit before it as its previous input).
  worked:  the "across" statement plus two worked examples with bars (N = 8, k = 2 and k = 4),
           the first annotated at the position right after its first bar, where the output
           visibly uses the input before the bar (reviewer question: do explicit cross-boundary
           demonstrations remove the blind spot?). The examples are produced by the task's own
           rule, so they are correct by construction, and they are fixed per task (and per
           alphabet or modulus), so the same text appears in the no-marker and marker prompts.
Everything else in the prompt, and every seed, is unchanged.
"""

import random

ORIGINAL = ("Constraint: The input contains vertical bars (|). These are structural boundaries.\n"
            "DO NOT transform them. Copy them exactly to the output.")
WORKED_HEAD = ("Constraint: The input contains vertical bars (|). They are only visual separators: apply the rule "
               "to the digits as if the bars were not there, and copy each bar to the output at the same position. "
               "The digit right after a bar still depends on the input digit(s) right before the bar.")
EXAMPLE_SEED = 777_001   # outside every grid seed (tasks.common.runner._local_seed at N = 8 stays below 722,000)


def make_worked(engine):
    span = getattr(engine, "WINDOW_SIZE", 2) - 1          # how many earlier inputs the rule reads
    uses_current = engine.TASK_NAME != "copy_previous"      # copy_previous ignores the current digit

    def gen(N, k, rng, axis_val):
        return engine.build_prompt(N, k, rng, axis_val=axis_val) if engine.AXIS_NAME else engine.build_prompt(N, k, rng)

    def examples(axis_val):
        """Two fixed examples at N = 8 (k = 2, then k = 4). The first keeps drawing until the output right
        after its first bar differs from the input there, so the dependence on the pre-bar input is visible."""
        out, seed = [], EXAMPLE_SEED
        for k_ex in (2, 4):
            while True:
                inp, truth, _ = gen(8, k_ex, random.Random(seed), axis_val)
                seed += 1
                plain_in, plain_out = inp.replace("|", ""), truth.replace("|", "")
                if plain_out[k_ex] != plain_in[k_ex] or (out and k_ex == 4):
                    out.append((inp, truth, plain_in[max(0, k_ex - span):k_ex], plain_in[k_ex], plain_out[k_ex]))
                    break
        return out

    def build_prompt(N, k, rng, axis_val=None):
        inp, out, prompt = gen(N, k, rng, axis_val)
        (i1, o1, before, after, y), (i2, o2, _b, _a, _y) = examples(axis_val)
        note = (f"the output digit right after the first bar is {y}: it uses the input digit"
                f"{'s' if len(before) > 1 else ''} just before the bar ({before})"
                + (f" together with the digit right after it ({after})" if uses_current else ""))
        new = (WORKED_HEAD + "\nWorked examples with bars:\n"
               f"Input {i1} -> Output {o1} ({note})\n"
               f"Input {i2} -> Output {o2}\n")
        assert prompt.count("Constraint:") == 1
        s = prompt.index("Constraint:")
        e = prompt.index("No spaces", s)
        return inp, out, prompt[:s] + new + prompt[e:]
    return build_prompt
VARIANTS = {
    "neutral": "Constraint: The input contains vertical bars (|). Copy each bar to the output at the same position.",
    "across": ("Constraint: The input contains vertical bars (|). They are only visual separators: apply the rule "
               "to the digits as if the bars were not there (the digit right after a bar uses the digit right "
               "before the bar as its previous input), and copy each bar to the output at the same position."),
}


def make_wording(engine, variant):
    new = VARIANTS[variant]

    def build_prompt(N, k, rng, axis_val=None):
        if engine.AXIS_NAME:
            inp, out, prompt = engine.build_prompt(N, k, rng, axis_val=axis_val)
        else:
            inp, out, prompt = engine.build_prompt(N, k, rng)
        assert prompt.count(ORIGINAL) == 1
        return inp, out, prompt.replace(ORIGINAL, new)
    return build_prompt
