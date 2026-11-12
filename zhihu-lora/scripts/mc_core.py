# -*- coding: utf-8 -*-
"""Shared prompt building + letter log-prob scoring for Chinese MC benchmarks."""
import time
import mlx.core as mx

LETTERS = ["A", "B", "C", "D"]


def fmt_example(ex, subject):
    lines = [f"以下是一道关于{subject}的单项选择题，请从A、B、C、D中选出唯一正确答案。", ""]
    lines.append(ex["q"])
    for k, o in zip(LETTERS, ex["opts"]):
        lines.append(f"{k}. {o}")
    lines.append(f"答案：{LETTERS[ex['ans']]}")
    return "\n".join(lines)


def build_prompt(subject, shots, test):
    parts = [fmt_example(ex, subject) for ex in shots]
    tail = [f"以下是一道关于{subject}的单项选择题，请从A、B、C、D中选出唯一正确答案。", "", test["q"]]
    tail += [f"{k}. {o}" for k, o in zip(LETTERS, test["opts"])]
    tail.append("答案：")
    return "\n\n".join(parts + ["\n".join(tail)])


def run(items, model, tok, batch_size, name, model_path, adapter):
    prompts, metas = [], []
    for it in items:
        txt = build_prompt(it["subject"], it["shots"], it["ex"])
        chat = tok.apply_chat_template([{"role": "user", "content": txt}],
                                       add_generation_prompt=True, tokenize=False)
        # put "答案：" at the start of the assistant turn, otherwise the model
        # starts a fresh reply ("答案" is a top token) instead of a letter
        chat = chat + "答案："
        prompts.append(tok.encode(chat, add_special_tokens=False))
        metas.append(it)
    letter_ids = [tok.encode(k, add_special_tokens=False)[0] for k in LETTERS]
    pad = tok.pad_token_id
    t0 = time.time()
    correct, per_subject, preds = 0, {}, []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        L = max(len(x) for x in batch)
        arr = mx.full((len(batch), L), pad, dtype=mx.int32)
        for j, ids in enumerate(batch):
            arr[j, L - len(ids):] = mx.array(ids, dtype=mx.int32)
        logits = model(arr)[:, -1, :].astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(logp)
        lp = logp[:, letter_ids].tolist()
        for j, scores in enumerate(lp):
            pred = max(range(4), key=lambda k: scores[k])
            m = metas[i + j]
            ok = pred == m["ex"]["ans"]
            correct += ok
            s = per_subject.setdefault(m["subject"], [0, 0])
            s[0] += ok; s[1] += 1
            preds.append({"subject": m["subject"], "pred": LETTERS[pred],
                          "answer": LETTERS[m["ex"]["ans"]], "ok": bool(ok)})
    return {
        "benchmark": name, "n": len(items), "accuracy": round(correct / max(len(items), 1), 4),
        "correct": correct, "model": model_path, "adapter": adapter,
        "seconds": round(time.time() - t0, 1),
        "per_subject": {k: {"acc": round(v[0] / v[1], 3), "n": v[1]} for k, v in sorted(per_subject.items())},
    }, preds
