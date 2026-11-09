"""Spaced-letter variants of the position-local letter tasks: the tokenization control.

Without markers, letter strings merge into multi-character tokens (48 letters -> 23-32
tokens across the study's tokenizers); a marker after each letter restores one token per
letter. Separating the letters with spaces gives one token per letter with or without
markers (48 -> 48 on every model; markers add one token each). So, on the same alphabet:
  merged letters, no markers  vs  spaced letters, no markers    -> tokenization alone
  spaced letters, no markers  vs  spaced letters, with markers  -> the marker beyond tokenization

make_spaced(engine) wraps a letter engine's build_prompt without changing what is drawn:
same seed, same letters, same rule, same prompt, except that
  - the input line shows every character (markers included) separated by single spaces,
  - "No spaces, no explanations." becomes an instruction to answer in the same spaced format.
The returned ground truth is the engine's unspaced marked output: scoring strips markers
and spaces from the prediction, so the comparison is character by character as elsewhere.
"""

SPACED_INSTRUCTION = ("Write the output in the same format as the input: every character separated "
                      "by a single space. No explanations.")


def spaced(s):
    return " ".join(s)


def make_spaced(engine):
    def build_prompt(N, k, rng, axis_val):
        inp, out, prompt = engine.build_prompt(N, k, rng, axis_val=axis_val)
        old_input_line = f"Input: {inp}"
        assert prompt.count(old_input_line) == 1 and prompt.count("No spaces, no explanations.") == 1
        prompt = prompt.replace(old_input_line, f"Input: {spaced(inp)}")
        prompt = prompt.replace("No spaces, no explanations.", SPACED_INSTRUCTION)
        return spaced(inp), out, prompt
    return build_prompt
