#!/usr/bin/env python3
"""Can a small finetune teach the model to carry a dependency across a marker? (Reviewer question.)

LoRA on Qwen2.5-7B-Instruct with synthetic examples of two dependency tasks, running_xor and
running_sum_local (moduli 2, 4, 8), at N in {8, 16, 32, 64} and every spacing k in
{0, 2, 4, 8, 16} (k < N), with the marked answers as targets, so every cross-boundary step is
supervised. A control adapter (--no-markers) is trained on the same tasks and seeds with k = 0 only,
to separate learning the tasks from learning to cross a marker. Examples are drawn from a seed
stream disjoint from every grid seed. The merged models are saved as ordinary checkpoints so the
usual runners evaluate them on held-out tasks (copy_previous, sliding_window_max,
random_permutation with digits) and on running_xor at the grid's own, unseen inputs.

Output: results/lora_separator/<tag>/ (merged checkpoint + tokenizer), results/lora_separator/<tag>.log
Usage:  python scripts/train_lora_separator.py [--no-markers] [--dry]
"""

import argparse
import importlib
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tasks.common.boundary import valid_intervals  # noqa: E402

BASE = "Qwen/Qwen2.5-7B-Instruct"
TASKS = {"running_xor": [None], "running_sum_local": [2, 4, 8]}
LENGTHS = [8, 16, 32, 64]
KS = [0, 2, 4, 8, 16]
PER_CELL = 30


def examples(no_markers):
    out = []
    for task, axes in TASKS.items():
        e = importlib.import_module(f"tasks.{task}.engine")
        for axis in axes:
            for N in LENGTHS:
                for k in valid_intervals(N, [0] if no_markers else KS):
                    for i in range(PER_CELL):
                        rng = random.Random(f"lora-{task}-{axis}-{N}-{k}-{i}")
                        inp, ans, prompt = (e.build_prompt(N, k, rng, axis_val=axis) if e.AXIS_NAME
                                            else e.build_prompt(N, k, rng))
                        out.append((prompt, ans))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-markers", action="store_true")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    tag = "Qwen2.5-7B-Instruct-" + ("lorabase" if args.no_markers else "lorasep")
    data = examples(args.no_markers)
    print(f"{len(data)} training examples ({tag})")
    if args.dry:
        p, a = data[len(data) // 2]
        print(p[-120:], "->", a)
        return

    import torch
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from tasks.common.runner import load_model

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = load_model(BASE)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    dev = model.device

    def encode(prompt, answer):
        head = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True)
        h = tokenizer(head, add_special_tokens=False)["input_ids"]
        a = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        return h + a, [-100] * len(h) + a

    enc = [encode(p, a) for p, a in data]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps_total = args.epochs * ((len(enc) + args.batch * args.accum - 1) // (args.batch * args.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps_total))
    (REPO / "results" / "lora_separator").mkdir(parents=True, exist_ok=True)
    log = open(REPO / "results" / "lora_separator" / f"{tag}.log", "a")
    model.train()
    rng = random.Random(0)
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        order = list(range(len(enc)))
        rng.shuffle(order)
        for b0 in range(0, len(order), args.batch):
            batch = [enc[i] for i in order[b0:b0 + args.batch]]
            L = max(len(x) for x, _ in batch)
            ids = torch.tensor([x + [pad] * (L - len(x)) for x, _ in batch], device=dev)
            lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in batch], device=dev)
            att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in batch], device=dev)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / args.accum
            loss.backward()
            if (b0 // args.batch + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 10 == 0:
                    msg = f"epoch {epoch} step {step}/{steps_total} loss {loss.item() * args.accum:.4f} {time.time() - t0:.0f}s"
                    print(msg, flush=True); log.write(msg + "\n"); log.flush()
    model.eval()
    out = REPO / "results" / "lora_separator" / tag
    merged = model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    print(f"saved merged checkpoint to {out}")


if __name__ == "__main__":
    main()
