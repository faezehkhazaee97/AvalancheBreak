"""External Tier-2 (one-step dependency) task: flip-flop language modeling,
per Liu et al. 2306.00946 ("Exposing Attention Glitches with Flip-Flop
Language Modeling"). The original paper studies next-token prediction at
'read' instructions; this study's convention needs a per-position sequence
transduction instead, so it's reframed as: output the REGISTER STATE
STREAM at every position, not just at reads.

Input alphabet (4 symbols, one per position):
  '0' -- write instruction, sets the register to 0
  '1' -- write instruction, sets the register to 1
  'r' -- read instruction (paper's query point; here, a no-op for the
         state -- the register just keeps its current value)
  'i' -- ignore instruction (explicit no-op filler)
Output at position i: the register's value after processing input[0..i]
  (i.e. the value written by the most recent '0'/'1' at or before i, or
  the default '0' if no write has occurred yet). This is a genuine
  one-step recurrence: output[i] = input[i] if input[i] in {'0','1'},
  else output[i] = output[i-1] -- the same shape as running_xor /
  running_sum_local, just with the update event now controlled by an
  explicit input symbol instead of always firing.

WRITE_PROB controls write-instruction density and therefore the effective
dependency length a 'read' or 'ignore' position may need to look back
across: high WRITE_PROB -> writes are frequent -> short average gap since
the last write; low WRITE_PROB -> writes are rare -> long average gap,
i.e. state must be carried across many positions, closer in spirit to a
full-prefix dependency at the token-count scale of this study even though
the underlying recurrence is still formally one-step.
"""

from tasks.common.boundary import insert_boundaries

AXIS_NAME = None
AXIS_VALUES = None
OUTPUT_HAS_BOUNDARIES = True

DEFAULT_REGISTER = "0"


def _compute_register_stream(instr_str):
    register = DEFAULT_REGISTER
    out = []
    for ch in instr_str:
        if ch in ("0", "1"):
            register = ch
        out.append(register)
    return "".join(out)


def make_build_prompt(write_prob, density_label):
    def build_prompt(N, k, rng, axis_val=None):
        # remaining probability mass split evenly between read and ignore
        weights = [write_prob / 2, write_prob / 2, (1 - write_prob) / 2, (1 - write_prob) / 2]
        symbols = ["0", "1", "r", "i"]
        input_str_base = "".join(rng.choices(symbols, weights=weights, k=N))
        output_str_base = _compute_register_stream(input_str_base)

        input_bounded = insert_boundaries(input_str_base, k)
        output_bounded = insert_boundaries(output_str_base, k)

        prompt = f"""You are a deterministic sequence transformer, simulating a flip-flop register ({density_label} write density).
Task: Track a single binary register (starts at 0) as you scan an instruction stream left to right.
Alphabet: 0 (write 0 to register), 1 (write 1 to register), r (read -- register unchanged), i (ignore -- register unchanged)
Rule: At each position, output the register's CURRENT value after processing that position's instruction. Writing '0' or '1' sets the register to that value. 'r' and 'i' do not change the register -- output whatever the register currently holds.
Constraint: The input contains vertical bars (|). These are structural boundaries. DO NOT treat them as instructions. Copy them exactly to the output.
Output length: your output must contain one character for EVERY position in the input, including every 'r' and 'i' -- do not stop early at an 'r'; it is not a final query, just another position to emit a value for. The output is exactly as long as the input.
No spaces, no explanations.

Input: {input_bounded}
Output: """

        return input_bounded, output_bounded, prompt

    return build_prompt
