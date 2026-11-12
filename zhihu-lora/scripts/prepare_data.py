#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zhihu-KOL (wangrui6/Zhihu-KOL, Apache-2.0) -> cleaned Chinese chat-SFT dataset.

Pipeline:
  1. stream the parquet shard in batches
  2. normalize / strip HTML, URLs, zero-width chars, Zhihu footer noise
  3. quality filters (length, Chinese ratio, ad/spam patterns, duplicates)
  4. deterministic split: 2,000 test / 2,000 valid / rest train
  5. sample a compute-budgeted training subset (default 40k) and write JSONL

Output records use the OpenAI chat schema consumed by mlx-lm:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
"""
import argparse
import hashlib
import html
import json
import os
import random
import re
import statistics
import sys
import time

import pyarrow.parquet as pq

SYSTEM_PROMPT = "你是一位知识渊博、表达清晰的知乎答主，请用中文认真回答用户的问题。"

URL_RE = re.compile(r"https?://[^\s)】]+|www\.[^\s)】]+")
HTML_RE = re.compile(r"<[^>]{1,200}>")
ZERO_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff\xa0]")
FOOTER_RE = re.compile(r"^\s*(发布于|编辑于|修改于|赞同\s*\d+|收起)\b.*$", re.M)
AD_RE = re.compile(r"微信公众号|公众号[：:]|加微信|微信号[：:]|扫码|二维码|点击链接|私信我|淘宝|优惠券")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
SPACE_RE = re.compile(r"[ \t\u3000]+")


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = ZERO_RE.sub("", s)
    s = HTML_RE.sub("", s)
    s = URL_RE.sub("", s)
    s = FOOTER_RE.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = SPACE_RE.sub(" ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def cjk_ratio(s: str) -> float:
    if not s:
        return 0.0
    compact = re.sub(r"\s", "", s)
    if not compact:
        return 0.0
    return len(CJK_RE.findall(compact)) / len(compact)


def dedup_key(instr: str, resp: str) -> str:
    return hashlib.md5((instr + "\x00" + resp[:64]).encode("utf-8")).hexdigest()


def iter_rows(path: str, batch_size: int = 20000):
    pf = pq.ParquetFile(path)
    cols = [c for c in ("INSTRUCTION", "RESPONSE", "SOURCE") if c in pf.schema.names]
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        d = batch.to_pydict()
        for i in range(len(d[cols[0]])):
            yield {c: (d[c][i] or "") for c in cols}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/raw/zhihu_00000.parquet")
    ap.add_argument("--out-dir", default="data/sft")
    ap.add_argument("--train-samples", type=int, default=40000)
    ap.add_argument("--valid", type=int, default=2000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    kept, seen = [], set()
    reasons = {"raw": 0, "empty": 0, "instr_len": 0, "resp_len": 0, "cjk": 0,
               "ad": 0, "dup": 0, "ok": 0}
    chars_raw = chars_clean = 0

    for r in iter_rows(args.parquet):
        reasons["raw"] += 1
        instr = clean_text(r.get("INSTRUCTION", "")).strip()
        resp = clean_text(r.get("RESPONSE", "")).strip()
        chars_raw += len(r.get("INSTRUCTION") or "") + len(r.get("RESPONSE") or "")
        chars_clean += len(instr) + len(resp)
        if not instr or not resp:
            reasons["empty"] += 1
            continue
        if not (5 <= len(instr) <= 150):
            reasons["instr_len"] += 1
            continue
        if not (50 <= len(resp) <= 2500):
            reasons["resp_len"] += 1
            continue
        if cjk_ratio(instr + resp) < 0.30:
            reasons["cjk"] += 1
            continue
        if AD_RE.search(resp):
            reasons["ad"] += 1
            continue
        k = dedup_key(instr, resp)
        if k in seen:
            reasons["dup"] += 1
            continue
        seen.add(k)
        reasons["ok"] += 1
        kept.append({"instruction": instr, "response": resp})
        if reasons["raw"] % 50000 == 0:
            print(f"  scanned={reasons['raw']:,} kept={reasons['ok']:,} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    test = kept[: args.test]
    valid = kept[args.test : args.test + args.valid]
    train_pool = kept[args.test + args.valid :]
    train = train_pool
    if args.train_samples and args.train_samples < len(train_pool):
        train = rng.sample(train_pool, args.train_samples)

    def to_msg(x):
        return {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["instruction"]},
            {"role": "assistant", "content": x["response"]},
        ]}

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(to_msg(x), ensure_ascii=False) + "\n")

    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "train_full.jsonl"), train_pool)
    write_jsonl(os.path.join(args.out_dir, "valid.jsonl"), valid)
    write_jsonl(os.path.join(args.out_dir, "test.jsonl"), test)

    def lens(rows):
        v = [len(x["response"]) for x in rows]
        return {"n": len(v), "chars": sum(v), "mean": round(statistics.mean(v), 1),
                "p50": statistics.median(v), "p90": sorted(v)[int(0.9 * len(v)) - 1],
                "max": max(v)} if v else {}

    stats = {
        "source": "wangrui6/Zhihu-KOL (HF) / OmniData/Zhihu-KOL (ModelScope), shard 0",
        "license": "Apache-2.0",
        "raw_rows": reasons["raw"],
        "kept_rows": reasons["ok"],
        "drop_reasons": reasons,
        "raw_chars": chars_raw,
        "clean_chars": chars_clean,
        "approx_tokens_cjk": int(chars_clean / 1.55),
        "train": lens(train), "train_full": lens(train_pool),
        "valid": lens(valid), "test": lens(test),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
