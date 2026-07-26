"""Check what the model's actual token probabilities look like at the exact
moment it decides to stop generating early, vs. a healthy earlier position
in the same real generation. Uses a real observed case: Input=10010011,
N=8, Base (no boundaries), the model generated '1111111' (7 digits, one
short of the expected 8) and then emitted its end-of-turn token instead of
continuing.
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import tasks.running_xor.engine as running_xor

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.eval()

    # Reconstruct the exact real prompt (Input_Given confirms this matches: 10010011)
    import random
    from tasks.common.runner import _local_seed
    rng = random.Random(_local_seed(8, 0, None, 0))
    input_str, truth, prompt = running_xor.build_prompt(8, 0, rng)
    assert input_str == "10010011", input_str
    print(f"Input: {input_str}  Ground truth: {truth}")

    real_continuation = "1111111"  # the model's own real generated text, teacher-forced back in

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    cont_ids = tokenizer(real_continuation, add_special_tokens=False, return_tensors="pt")["input_ids"]
    input_ids = torch.cat([enc["input_ids"], cont_ids], dim=1).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[0]  # (seq_len, vocab)
    probs = torch.softmax(logits.float(), dim=-1)

    prompt_len = enc["input_ids"].shape[1]
    n_cont = cont_ids.shape[1]

    eos_ids = set()
    if isinstance(tokenizer.eos_token_id, int):
        eos_ids.add(tokenizer.eos_token_id)
    elif tokenizer.eos_token_id is not None:
        eos_ids.update(tokenizer.eos_token_id)
    eos_ids |= set(tokenizer.all_special_ids)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None:
        eos_ids.add(im_end_id)
    print(f"Treating these token IDs as 'end of turn': {eos_ids}")

    print(f"\n{'Position':<10}{'Token just generated':<25}{'P(that token)':<16}{'P(EOS/end-of-turn)':<20}{'Top-3 predicted next':<40}")
    for i in range(n_cont):
        pos = prompt_len + i - 1  # logits at this position predict the NEXT token (position prompt_len+i)
        token_id = input_ids[0, prompt_len + i].item()
        token_str = tokenizer.decode([token_id])
        p_token = probs[pos, token_id].item()
        p_eos = sum(probs[pos, eid].item() for eid in eos_ids if eid < probs.shape[1])
        top3 = torch.topk(probs[pos], 3)
        top3_str = ", ".join(f"{tokenizer.decode([t.item()])!r}:{p.item():.3f}" for t, p in zip(top3.indices, top3.values))
        marker = "  <-- STOPPING POINT (real EOS came next)" if i == n_cont - 1 else ""
        print(f"{i:<10}{token_str!r:<25}{p_token:<16.4f}{p_eos:<20.4f}{top3_str:<40}{marker}")

    # what did the model actually predict AFTER the last generated digit? (this is what determined stopping)
    final_pos = prompt_len + n_cont - 1
    top5_final = torch.topk(probs[final_pos], 5)
    print(f"\n--- Distribution at the exact stopping decision (after '{real_continuation}') ---")
    for t, p in zip(top5_final.indices, top5_final.values):
        tag = " <- END OF TURN TOKEN" if t.item() in eos_ids else ""
        print(f"  {tokenizer.decode([t.item()])!r}: {p.item():.4f}{tag}")


if __name__ == "__main__":
    main()
