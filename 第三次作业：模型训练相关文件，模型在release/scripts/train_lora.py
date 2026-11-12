#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA SFT for Qwen2.5-0.5B-Instruct on Apple Silicon (MLX).

Why LoRA: 16 GB unified memory. Full FT of 0.5B in bf16 + AdamW would need
~10+ GB and be slow; LoRA (r=16 on the last 16 of 24 blocks) trains ~1% of
parameters, fits in ~5 GB, and keeps the base model untouched.

Pipeline: chat-template tokenisation -> assistant-only loss mask ->
length-bucketed batches -> AdamW + warmup/cosine -> periodic val loss ->
adapter checkpoint (adapters.safetensors + adapter_config.json).
"""
import argparse
import json
import math
import os
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

IGNORE = -100


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def encode_dataset(rows, tok, max_len):
    """Tokenise chat rows and build assistant-only labels."""
    out = []
    for ex in rows:
        msgs = ex["messages"]
        ids = tok.apply_chat_template(msgs, return_dict=False)
        if len(ids) < 8:
            continue
        prefix_len = len(tok.apply_chat_template(
            msgs[:-1], add_generation_prompt=True, return_dict=False))
        ids = ids[:max_len]
        labels = [IGNORE] * len(ids)
        start = min(prefix_len, len(ids) - 1)
        for i in range(start, len(ids)):
            labels[i] = ids[i]
        if all(v == IGNORE for v in labels):
            continue
        out.append((np.array(ids, dtype=np.int32), np.array(labels, dtype=np.int32)))
    return out


def collate(indices, data, pad_id, pad_to=32):
    seqs = [data[i] for i in indices]
    longest = max(len(s[0]) for s in seqs)
    length = 1 + pad_to * ((longest + pad_to - 1) // pad_to)
    x = np.full((len(seqs), length), pad_id, dtype=np.int32)
    y = np.full((len(seqs), length), IGNORE, dtype=np.int32)
    for j, (ids, lab) in enumerate(seqs):
        x[j, : len(ids)] = ids
        y[j, : len(lab)] = lab
    return mx.array(x), mx.array(y)


def make_batches(data, batch_size, rng, pad_id, token_budget=None):
    """Length-bucketed batches; optionally cap tokens/step to bound peak memory."""
    order = sorted(range(len(data)), key=lambda i: len(data[i][0]))
    if token_budget:
        chunks, cur, ntok = [], [], 0
        for i in order:
            L = max(len(data[i][0]), 1)
            if cur and (ntok + L > token_budget or len(cur) >= batch_size):
                chunks.append(cur); cur, ntok = [], 0
            cur.append(i); ntok += L
        if cur:
            chunks.append(cur)
    else:
        chunks = [order[i : i + batch_size] for i in range(0, len(order) - batch_size + 1, batch_size)]
    rng.shuffle(chunks)
    for ch in chunks:
        yield collate(ch, data, pad_id)


def loss_fn(model, x, y):
    logits = model(x)[:, :-1, :]
    targets = y[:, 1:]
    mask = targets != IGNORE
    safe = mx.where(mask, targets, 0)
    ce = nn.losses.cross_entropy(logits, safe, reduction="none")
    ce = ce * mask
    return ce.sum() / mx.maximum(mask.sum(), 1)


def evaluate(model, data, pad_id, batch_size=8, max_batches=16, seed=0):
    model.eval()
    rng = random.Random(seed)
    total, ntok, nb = 0.0, 0, 0
    for x, y in make_batches(data, batch_size, rng, pad_id, 3200):
        mask = y != IGNORE
        loss = loss_fn(model, x, y)
        mx.eval(loss)
        total += float(loss) * int(mask.sum())
        ntok += int(mask.sum())
        nb += 1
        if nb >= max_batches:
            break
    model.train()
    return total / max(ntok, 1)


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / max(args.warmup, 1)
    p = (step - args.warmup) / max(1, args.iters - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(p, 1.0)))


def save_adapter(model, path, args):
    os.makedirs(path, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(os.path.join(path, "adapters.safetensors"), weights)
    cfg = {
        "model": args.model,
        "fine_tune_type": "lora",
        "num_layers": args.num_layers,
        "lora_parameters": {"rank": args.rank, "scale": args.scale, "dropout": args.dropout},
        "mask_prompt": True,
        "max_seq_length": args.max_seq_len,
        "iters": args.iters,
        "learning_rate": args.lr,
    }
    with open(os.path.join(path, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return len(weights)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/base")
    ap.add_argument("--data-dir", default="data/sft")
    ap.add_argument("--train-file", default="train.jsonl")
    ap.add_argument("--valid-file", default="valid.jsonl")
    ap.add_argument("--adapter-path", default="models/adapters/zhihu-lora")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--num-layers", type=int, default=16)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--val-batches", type=int, default=16)
    ap.add_argument("--token-budget", type=int, default=3200)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", default="results/train_log.jsonl")
    args = ap.parse_args()

    mx.random.seed(args.seed)
    mx.set_cache_limit(2 * 1024**3)
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    os.makedirs(args.adapter_path, exist_ok=True)

    print(f"[load] {args.model}", flush=True)
    model, tok = load(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"[tok] vocab={len(tok)} eos={tok.eos_token_id} pad={pad_id}", flush=True)

    t0 = time.time()
    train_rows = read_jsonl(os.path.join(args.data_dir, args.train_file))
    valid_rows = read_jsonl(os.path.join(args.data_dir, args.valid_file))[:400]
    train = encode_dataset(train_rows, tok, args.max_seq_len)
    valid = encode_dataset(valid_rows, tok, args.max_seq_len)
    print(f"[data] train={len(train)} valid={len(valid)} "
          f"tokens={sum(len(x[0]) for x in train):,} ({time.time()-t0:.0f}s)", flush=True)

    model.freeze()
    linear_to_lora_layers(model, args.num_layers, {
        "rank": args.rank, "scale": args.scale, "dropout": args.dropout})
    ntrain = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    ntotal = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"[lora] trainable={ntrain:,}/{ntotal:,} ({100*ntrain/ntotal:.3f}%)", flush=True)
    model.train()

    opt = optim.AdamW(learning_rate=lambda st: mx.array(lr_at(st, args)),
                      weight_decay=args.weight_decay, betas=[0.9, 0.95], eps=1e-8)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    step, epoch, tokens_seen = 0, 0, 0
    best_val = float("inf")
    t_start = time.time()
    rng = random.Random(args.seed)
    logf = open(args.log, "a", encoding="utf-8")

    def stream():
        nonlocal epoch
        while True:
            epoch += 1
            yield from make_batches(train, args.batch_size, rng, pad_id, args.token_budget)

    batches = stream()
    while step < args.iters:
        lr = lr_at(step, args)
        acc, loss = None, None
        for _ in range(args.grad_accum):
            x, y = next(batches)
            loss, grads = loss_and_grad(model, x, y)
            mx.eval(loss, grads)
            acc = grads if acc is None else tree_map(lambda a, b: a + b, acc, grads)
            tokens_seen += int((y != IGNORE).sum())
        acc = tree_map(lambda g: g / args.grad_accum, acc)
        acc, grad_norm = optim.clip_grad_norm(acc, args.grad_clip)
        opt.update(model, acc)
        mx.eval(model.parameters(), opt.state)
        step += 1
        if step % 10 == 0 or step == 1:
            el = time.time() - t_start
            rec = {"step": step, "epoch": epoch, "loss": round(float(loss), 4),
                   "lr": round(lr, 8), "grad_norm": round(float(grad_norm), 4),
                   "tokens": tokens_seen, "tok_s": round(tokens_seen / el, 1),
                   "elapsed_s": round(el, 1)}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"step {step:4d}/{args.iters} ep{epoch} loss={float(loss):.4f} "
                  f"lr={lr:.2e} |g|={float(grad_norm):.2f} {tokens_seen/el:.0f} tok/s", flush=True)

        if args.eval_every and step % args.eval_every == 0:
            v = evaluate(model, valid, pad_id, max_batches=args.val_batches)
            print(f"  [eval] step {step} val_loss={v:.4f} ppl={math.exp(v):.3f}", flush=True)
            logf.write(json.dumps({"step": step, "val_loss": round(v, 4),
                                   "val_ppl": round(math.exp(v), 3)}) + "\n")
            logf.flush()
            if v < best_val:
                best_val = v
                save_adapter(model, args.adapter_path, args)
                print(f"  [save] best adapter @ {step} (val {v:.4f})", flush=True)

        if args.save_every and step % args.save_every == 0:
            save_adapter(model, args.adapter_path, args)

    n = save_adapter(model, args.adapter_path, args)
    meta = {"iters": step, "epochs": epoch, "tokens_seen": tokens_seen,
            "train_seconds": round(time.time() - t_start, 1),
            "best_val_loss": None if best_val == float("inf") else round(best_val, 4),
            "trainable_params": ntrain, "total_params": ntotal,
            "adapter_tensors": n, "train_samples": len(train), "valid_samples": len(valid),
            "config": vars(args)}
    with open(os.path.join(args.adapter_path, "train_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[done]", json.dumps({k: meta[k] for k in ("iters", "tokens_seen", "train_seconds")}), flush=True)
    logf.close()


if __name__ == "__main__":
    main()
