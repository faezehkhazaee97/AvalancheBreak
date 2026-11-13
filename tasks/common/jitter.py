"""Jittered marker placement for the dependency tasks (reviewer question: how does the blind spot
change when markers are not exactly periodic?).

make_jitter(engine, amount) wraps a task engine: the input string and the plain answer are the
ones the base engine produces for the same seed (so a jittered cell is paired with the exactly
periodic cell of the main grid), and every canonical marker position k, 2k, ... is moved by an
offset drawn uniformly from [-amount, amount] (clipped to stay inside the string, duplicates
merged). Markers are overlaid at the same jittered positions in the input and in the answer, as
in the base engines. At k = 0 the base prompt is returned unchanged.
"""


def jittered_positions(N, k, amount, rng):
    pos = set()
    for p in range(k, N, k):                      # same convention as boundary.insert_boundaries
        pos.add(min(N - 1, max(1, p + rng.randint(-amount, amount))))
    return pos


def overlay(s, pos):
    return "".join(("|" if i in pos else "") + c for i, c in enumerate(s))


def make_jitter(engine, amount=1):
    def build_prompt(N, k, rng, axis_val=None):
        res = engine.build_prompt(N, k, rng, axis_val=axis_val) if engine.AXIS_NAME else engine.build_prompt(N, k, rng)
        if k <= 0:
            return res
        inp_b, out_b, prompt = res
        plain_in, plain_out = inp_b.replace("|", ""), out_b.replace("|", "")
        assert len(plain_in) == len(plain_out) == N
        pos = jittered_positions(N, k, amount, rng)
        inp_j, out_j = overlay(plain_in, pos), overlay(plain_out, pos)
        head, sep, tail = prompt.rpartition("Input: ")
        assert sep and tail == inp_b + "\nOutput: ", "prompt does not end with the input line"
        return inp_j, out_j, head + "Input: " + inp_j + "\nOutput: "
    return build_prompt
