#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate report figures from training log and evaluation JSONs."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "figures"
os.makedirs(FIG, exist_ok=True)


def load(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def loss_curve():
    tr, va = [], []
    with open("results/train_log.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "val_loss" in r:
                va.append((r["step"], r["val_loss"]))
            elif "loss" in r:
                tr.append((r["step"], r["loss"]))
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    xs = [x for x, _ in tr]; ys = [y for _, y in tr]
    ax.plot(xs, ys, lw=0.9, color="#3b6ea5", alpha=0.35, label="train loss (per 10 steps)")
    if len(ys) >= 5:
        k = 5
        ma = [sum(ys[max(0, i - k + 1):i + 1]) / len(ys[max(0, i - k + 1):i + 1]) for i in range(len(ys))]
        ax.plot(xs, ma, lw=2.0, color="#1f4e79", label="train loss (5-point moving avg)")
    if va:
        ax.plot([x for x, _ in va], [y for _, y in va], "o-", lw=1.8, ms=4, color="#d1495b", label="validation loss")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Cross-entropy (assistant tokens)")
    ax.set_title("LoRA SFT on Zhihu-KOL — Qwen2.5-0.5B-Instruct")
    ax.axvline(50, color="#2e8b57", ls="--", lw=1)
    ax.annotate("selected\ncheckpoint", xy=(50, ax.get_ylim()[1]), xytext=(58, ax.get_ylim()[1]-0.06),
                fontsize=7, color="#2e8b57", va="top")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/loss_curve.png")
    print("wrote", f"{FIG}/loss_curve.png")


def data_funnel():
    st = load("data/sft/stats.json", {})
    if not st:
        return
    stages = ["Raw rows\n(shard 0)", "After cleaning\n& dedup", "Sampled for\ntraining", "Valid / Test\n(each)"]
    vals = [st["raw_rows"], st["kept_rows"], st["train"]["n"], st["valid"]["n"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
    bars = ax.barh(range(len(vals)), vals, color=["#9fb3c8", "#5d8aa8", "#2e5e8c", "#d1495b"])
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(stages)
    ax.invert_yaxis()
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(v + max(vals) * 0.01, i, f"{v:,}", va="center", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("Samples")
    ax.set_title("Zhihu-KOL data funnel")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{FIG}/data_funnel.png")
    print("wrote", f"{FIG}/data_funnel.png")


def eval_bars():
    cm_b = load("results/cmmlu_base.json"); cm_t = load("results/cmmlu_tuned.json")
    ce_b = load("results/ceval_base.json"); ce_t = load("results/ceval_tuned.json")
    pb = load("results/ppl_base.json"); pt = load("results/ppl_tuned.json")
    if not (cm_b and cm_t):
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=160)
    names = ["Base\n(Qwen2.5-0.5B-Instruct)", "+ LoRA\n(Zhihu-KOL)"]
    colors = ["#9fb3c8", "#2e5e8c"]
    ax = axes[0]
    x = [0, 1]
    labels, bvals, tvals = [], [], []
    for key, b, t in (("CMMLU", cm_b, cm_t), ("C-Eval", ce_b, ce_t)):
        if b and t:
            labels.append(key)
            bvals.append(b["accuracy"] * 100)
            tvals.append(t["accuracy"] * 100)
    w = 0.34
    for i, (bv, tv) in enumerate(zip(bvals, tvals)):
        ax.bar(i - w / 2, bv, w, color=colors[0], label="base" if i == 0 else None)
        ax.bar(i + w / 2, tv, w, color=colors[1], label="+LoRA" if i == 0 else None)
        ax.text(i - w / 2, bv + 1, f"{bv:.1f}", ha="center", fontsize=8)
        ax.text(i + w / 2, tv + 1, f"{tv:.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("5-shot accuracy (%)")
    ax.set_ylim(0, max(bvals + tvals) * 1.35)
    ax.set_title("Chinese multiple-choice benchmarks")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    if pb and pt:
        vals = [pb["ppl"], pt["ppl"]]
        bars = ax.bar([0, 1], vals, 0.5, color=colors)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Base", "+ LoRA"])
        ax.set_ylabel("Perplexity (lower is better)")
        ax.set_title("Held-out Zhihu test perplexity")
        ax.set_ylim(0, max(vals) * 1.2)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{FIG}/eval_summary.png")
    print("wrote", f"{FIG}/eval_summary.png")


if __name__ == "__main__":
    loss_curve()
    data_funnel()
    eval_bars()
