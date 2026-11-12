#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Held-out loss / perplexity on the Zhihu test split (assistant tokens only)."""
import argparse, json, math, time
import mlx.core as mx
from mlx_lm import load
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lora import read_jsonl, encode_dataset, make_batches, loss_fn, IGNORE  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/base")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data/sft/test.jsonl")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.data)[: args.limit]
    model, tok = load(args.model, adapter_path=args.adapter)
    data = encode_dataset(rows, tok, args.max_seq_len)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model.eval()

    total, ntok, nseq = 0.0, 0, 0
    t0 = time.time()
    for x, y in make_batches(data, args.batch_size, __import__("random").Random(0), pad):
        mask = y != IGNORE
        loss = loss_fn(model, x, y)
        mx.eval(loss)
        total += float(loss) * int(mask.sum())
        ntok += int(mask.sum())
        nseq += x.shape[0]
    loss = total / max(ntok, 1)
    res = {"model": args.model, "adapter": args.adapter, "split": args.data,
           "n_samples": nseq, "n_tokens": ntok, "loss": round(loss, 5),
           "ppl": round(math.exp(loss), 3), "seconds": round(time.time() - t0, 1)}
    print(json.dumps(res, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
