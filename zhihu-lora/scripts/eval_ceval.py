#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5-shot C-Eval evaluation (52 subjects, validation split)."""
import argparse, glob, json, os, random
import mlx.core as mx
from mlx_lm import load
import pyarrow.parquet as pq
from mc_core import LETTERS, run


def load_parquet(path):
    out = []
    for r in pq.read_table(path).to_pylist():
        q = (r.get("question") or "").strip()
        opts = [(r.get(k) or "").strip() for k in LETTERS]
        ans = (r.get("answer") or "").strip().upper()
        if q and all(opts) and ans in LETTERS:
            out.append({"q": q, "opts": opts, "ans": LETTERS.index(ans)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/benchmarks/ceval")
    ap.add_argument("--model", default="models/base")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--per-subject", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    mx.set_cache_limit(2 * 1024**3)
    rng = random.Random(0)
    items = []
    for vf in sorted(glob.glob(os.path.join(args.data, "*", "val-*.parquet"))):
        subj = os.path.basename(os.path.dirname(vf)).replace("_", " ")
        dev = glob.glob(os.path.join(os.path.dirname(vf), "dev-*.parquet"))
        shots = load_parquet(dev[0])[: args.shots] if dev else []
        tests = load_parquet(vf)
        rng.shuffle(tests)
        for ex in tests[: args.per_subject]:
            items.append({"subject": subj, "shots": shots, "ex": ex})
    model, tok = load(args.model, adapter_path=args.adapter)
    model.eval()
    print(f"[ceval] {len(items)} questions, {len(glob.glob(os.path.join(args.data,'*','val-*.parquet')))} subjects", flush=True)
    res, preds = run(items, model, tok, args.batch_size, "C-Eval", args.model, args.adapter)
    res["shots"] = args.shots
    print(json.dumps({k: res[k] for k in ("benchmark", "n", "accuracy", "correct", "seconds")}, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    with open(args.out.replace(".json", "_preds.jsonl"), "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
