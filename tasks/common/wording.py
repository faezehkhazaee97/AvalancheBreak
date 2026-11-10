"""Instruction-wording variants for the marker constraint (tests whether the blind spot
depends on calling the markers "structural boundaries").

  neutral: only asks to copy each bar through, with no "structural boundaries" framing.
  across:  states that bars are only visual separators and that the rule applies across them
           (the digit after a bar uses the digit before it as its previous input).
Everything else in the prompt, and every seed, is unchanged.
"""

ORIGINAL = ("Constraint: The input contains vertical bars (|). These are structural boundaries.\n"
            "DO NOT transform them. Copy them exactly to the output.")
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
