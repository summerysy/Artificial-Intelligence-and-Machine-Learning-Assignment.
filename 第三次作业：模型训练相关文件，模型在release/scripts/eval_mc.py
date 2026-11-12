#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5-shot CMMLU evaluation (Chinese multiple-choice) with letter log-prob scoring."""
import argparse, csv, glob, json, math, os, random, re, time
import mlx.core as mx
from mlx_lm import load

from mc_core import LETTERS, run


def load_subject(path):
    import csv
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        q = (r.get("Question") or "").strip()
        opts = [(r.get(k) or "").strip() for k in LETTERS]
        ans = (r.get("Answer") or "").strip().upper()
        if q and all(opts) and ans in LETTERS:
            out.append({"q": q, "opts": opts, "ans": LETTERS.index(ans)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/benchmarks/cmmlu_data")
    ap.add_argument("--model", default="models/base")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--per-subject", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    mx.set_cache_limit(2 * 1024**3)
    rng = random.Random(0)
    test_files = sorted(glob.glob(os.path.join(args.data, "test", "*.csv")))
    items = []
    for tf in test_files:
        subj = os.path.basename(tf)[:-4].replace("_", " ")
        devf = os.path.join(args.data, "dev", os.path.basename(tf))
        shots = load_subject(devf)[: args.shots] if os.path.exists(devf) else []
        tests = load_subject(tf)
        rng.shuffle(tests)
        for ex in tests[: args.per_subject]:
            items.append({"subject": subj, "shots": shots, "ex": ex})
    model, tok = load(args.model, adapter_path=args.adapter)
    model.eval()
    print(f"[cmmlu] {len(items)} questions, {len(test_files)} subjects", flush=True)
    res, preds = run(items, model, tok, args.batch_size, "CMMLU", args.model, args.adapter)
    res["shots"] = args.shots
    print(json.dumps({k: res[k] for k in ("benchmark", "n", "accuracy", "correct", "seconds")}, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    with open(args.out.replace(".json", "_preds.jsonl"), "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
