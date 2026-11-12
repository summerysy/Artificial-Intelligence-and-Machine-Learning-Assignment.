# 16 GB M1 Pro 上的 Qwen2.5-0.5B 知乎 LoRA 微调

完整报告见 **[report.md](report.md)**（数据准备 → 训练 → CMMLU/C-Eval/困惑度评测 → 分析）。

## 快速开始

```bash
./run_all.sh          # 下载 → 清洗 → 训练 → 评测 → 出图（约 40 分钟）
```

已生成的结果：

| 路径 | 内容 |
|---|---|
| `report.md` | 完整中文报告 |
| `figures/` | 训练曲线、数据漏斗、评测柱状图 |
| `results/` | 指标 JSON、训练日志、生成样例 |
| `models/adapters/zhihu-qwen05b-lora/` | 选用的 LoRA 适配器（23 MB） |
| `data/sft/` | 清洗后的 train/valid/test JSONL 与统计 |

## 关键数字

- 基座：Qwen2.5-0.5B-Instruct（0.494 B，988 MB bf16）
- 语料：Zhihu-KOL 分片 0（306 MB，201,244 行 → 138,304 行清洗后）
- 训练：LoRA r=16，5.87 M 可训练参数（1.17%），150 步 / 4,800 条 / 25.4 分钟
- 评测：知乎测试集 PPL 49.88 → 48.57；CMMLU 36.12% → 34.93%；C-Eval 37.31% → 37.69%
