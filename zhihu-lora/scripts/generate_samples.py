#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Side-by-side qualitative comparison: base vs LoRA-tuned generations."""
import argparse, json, time
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

SYSTEM = "你是一位知识渊博、表达清晰的知乎答主，请用中文认真回答用户的问题。"
PROMPTS = [
    "为什么现在越来越多的年轻人选择不结婚？",
    "如何评价《三体》中的程心这个角色？",
    "刚毕业的大学生应该如何规划自己的职业？",
    "为什么有些人越努力反而越焦虑？",
    "有哪些看似不起眼但其实很赚钱的行业？",
    "如何在一个月内有效提高英语口语？",
]


def run(model_path, adapter, prompts, max_tokens, temp):
    model, tok = load(model_path, adapter_path=adapter)
    out = []
    for q in prompts:
        p = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
            add_generation_prompt=True, tokenize=False)
        t0 = time.time()
        txt = generate(model, tok, prompt=p, max_tokens=max_tokens,
                       sampler=make_sampler(temp=temp, top_p=0.9), verbose=False)
        out.append({"prompt": q, "output": txt.strip(),
                    "chars": len(txt.strip()), "seconds": round(time.time() - t0, 1)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/base")
    ap.add_argument("--adapter", default="models/adapters/zhihu-qwen05b-lora")
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--out", default="results/generations.json")
    args = ap.parse_args()

    base = run(args.model, None, PROMPTS, args.max_tokens, args.temp)
    tuned = run(args.model, args.adapter, PROMPTS, args.max_tokens, args.temp)
    res = []
    for b, t in zip(base, tuned):
        res.append({"prompt": b["prompt"], "base": b["output"], "tuned": t["output"],
                    "base_chars": b["chars"], "tuned_chars": t["chars"]})
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    for r in res:
        print("=" * 80)
        print("Q:", r["prompt"])
        print(f"[base {r['base_chars']}字]", r["base"][:300].replace("\n", " "))
        print(f"[tuned {r['tuned_chars']}字]", r["tuned"][:300].replace("\n", " "))


if __name__ == "__main__":
    main()
