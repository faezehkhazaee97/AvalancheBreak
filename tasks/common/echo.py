"""'Echo' variants of the dependency tasks (reviewer's carry-over assist): after every bar the input
repeats, in square brackets, the input digit just before the bar, so the input the post-marker position
needs is written right next to it. The rule, the expected output (bars, no echoes), and the inputs are
those of the base task: build_prompt calls the base engine with the same rng, so with the runner's
seeds every echo run is paired input for input with the base task's marked run."""

ECHO_NOTE = ("After each bar, the digit in square brackets repeats the input digit just before the bar. "
             "It is not part of the sequence: do not transform it and do not write it in the output.")


def add_echo(inp):
    out, prev = [], None
    for ch in inp:
        out.append(ch)
        if ch == "|":
            out.append(f"[{prev}]")
        else:
            prev = ch
    return "".join(out)


assert add_echo("0110|1011|01") == "0110|[0]1011|[1]01"


def make_echo(engine):
    def build_prompt(N, k, rng, axis_val=None):
        if engine.AXIS_NAME:
            inp, out, prompt = engine.build_prompt(N, k, rng, axis_val=axis_val)
        else:
            inp, out, prompt = engine.build_prompt(N, k, rng)
        new = add_echo(inp)
        assert prompt.count(f"Input: {inp}\n") == 1 and prompt.count("Copy them exactly to the output.") == 1
        prompt = prompt.replace(f"Input: {inp}\n", f"Input: {new}\n")
        prompt = prompt.replace("Copy them exactly to the output.", "Copy them exactly to the output.\n" + ECHO_NOTE)
        return new, out, prompt
    return build_prompt
