import os

# ============================================================
# DIAGNOSTIC: why does the scratchpad addition task only find a
# parseable "Final Answer:" line in ~54% of templated responses?
# ============================================================
# Regenerates a handful of KNOWN-FAILING trials (same seeds as
# addition_scratchpad_grid_templated.csv) and prints the FULL raw
# decoded response plus token-count/truncation info, so we can see
# whether the model is (a) hitting max_new_tokens before writing the
# Final Answer line, or (b) writing the answer in some format the
# regex `Final Answer:\s*(\d+)` doesn't match.

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, random
import re

print(f"Loading {MODEL_ID} into VRAM...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager",
)
model.eval()


class ScratchpadAdditionTaskEngine:
    def _sample_pair(self, length, rng):
        for _ in range(200):
            a = [rng.randrange(10) for _ in range(length)]
            b = [rng.randrange(10) for _ in range(length)]
            if a[0] == 0: a[0] = rng.randrange(1, 10)
            if b[0] == 0: b[0] = rng.randrange(1, 10)
            carry = 0
            total = []
            for i in range(length - 1, -1, -1):
                s = a[i] + b[i] + carry
                total.append(s % 10)
                carry = s // 10
            if carry == 0:
                return a, b, list(reversed(total))
        raise RuntimeError(f"failed to sample a non-overflowing {length}-digit pair")

    def build_prompt(self, length, interval, rng=None):
        rng = rng or random
        a, b, total = self._sample_pair(length, rng)
        a_str = "".join(map(str, a))
        b_str = "".join(map(str, b))
        sum_str = "".join(map(str, total))

        combined = list(a_str) + ["+"] + list(b_str)
        bounded = []
        for i, d in enumerate(combined):
            bounded.append(d)
            if interval > 0 and (i + 1) % interval == 0 and i != len(combined) - 1:
                bounded.append("|")
        input_str = "".join(bounded)

        prompt = f"""You are a careful arithmetic assistant.
Task: Add two {length}-digit numbers.
Constraint: The input may contain vertical bars (|) as structural boundaries -- they carry no numeric meaning, ignore them when computing.
First, work through the addition step by step from the rightmost (least-significant) digit to the left, tracking the carry at each step. Then, on its own line, write exactly:
Final Answer: <the {length}-digit sum, zero-padded if needed>

Input: {input_str}
"""
        return input_str, sum_str, prompt


engine = ScratchpadAdditionTaskEngine()

# (Seq_Length, Interval, Trial) -- picked from KNOWN failures in
# addition_scratchpad_grid_templated.csv
CASES = [
    (8, 0, 1), (8, 0, 2), (8, 0, 3),      # Base, N=8 -- worst cell, 73% failure rate
    (8, 4, 1), (8, 4, 2),                  # Cellular, N=8 -- also failing
    (16, 4, 0),                             # Cellular, N=16 -- worst overall (20% success)
]

for (N, k, t) in CASES:
    local_seed = 150_000 * N + 100 * k + t
    rng = random.Random(local_seed)
    inp, truth, prompt = engine.build_prompt(N, interval=k, rng=rng)

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    inputs = {kk: vv.to(model.device) for kk, vv in enc.items()}

    max_tokens_needed = N * 20 + 60
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens_needed, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_only = outputs[0][inputs["input_ids"].shape[1]:]
    n_new_tokens = gen_only.shape[0]
    ended_naturally = (gen_only[-1].item() == tokenizer.eos_token_id) if n_new_tokens > 0 else False
    pred_full = tokenizer.decode(gen_only, skip_special_tokens=True)
    match = re.search(r'Final Answer:\s*(\d+)', pred_full)

    print("=" * 70)
    print(f"N={N}  k={k}  trial={t}  truth={truth}")
    print(f"Prompt input: {inp}")
    print(f"Tokens generated: {n_new_tokens} / budget {max_tokens_needed}  "
          f"(hit budget cap: {n_new_tokens >= max_tokens_needed}, ended on EOS: {ended_naturally})")
    print(f"Regex 'Final Answer:\\s*(\\d+)' matched: {match is not None}")
    print("--- FULL RAW RESPONSE ---")
    print(pred_full)
    print()

print("Diagnostic complete.")
