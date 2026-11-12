#!/bin/bash
# End-to-end reproduction on Apple Silicon (M1 Pro / 16 GB).
# Prereq: uv (https://docs.astral.sh/uv/), ~4 GB free disk, network.
set -euo pipefail
MODEL_URL="https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-0.5B-Instruct/repo?Revision=master&FilePath=model.safetensors"
DATA_URL="https://modelscope.cn/api/v1/datasets/OmniData/Zhihu-KOL/repo?Revision=master&FilePath=data/train-00000-of-00005-a1278ede4e8c5cdb.parquet"

# 1. Python 3.12 env
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python --index-url https://mirrors.aliyun.com/pypi/simple/ \
  mlx mlx-lm numpy pandas pyarrow matplotlib tqdm huggingface_hub

# 2. Assets (ModelScope CDN; HF mirror alternatives are noted in the report)
mkdir -p models/base data/raw
for f in config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json; do
  curl -sSL -o "models/base/$f" "https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-0.5B-Instruct/repo?Revision=master&FilePath=$f"
done
./scripts/fetch_parallel.sh "$MODEL_URL" models/base/model.safetensors 6
./scripts/fetch_parallel.sh "$DATA_URL" data/raw/zhihu_00000.parquet 4

# 3. Clean + split
.venv/bin/python scripts/prepare_data.py --parquet data/raw/zhihu_00000.parquet --train-samples 40000
head -n 8000 data/sft/train.jsonl > data/sft/train_8k.jsonl

# 4. LoRA SFT (~75 min on M1 Pro 16 GB)
.venv/bin/python scripts/train_lora.py --model models/base --data-dir data/sft \
  --train-file train_8k.jsonl --adapter-path models/adapters/zhihu-qwen05b-lora \
  --iters 150 --grad-accum 4 --batch-size 8 --token-budget 3200 --lr 3e-5 --warmup 10 \
  --eval-every 100 --save-every 100 --val-batches 4 --log results/train_log.jsonl

# 5. Evaluation
mkdir -p results
.venv/bin/python scripts/eval_ppl.py --model models/base --out results/ppl_base.json
.venv/bin/python scripts/eval_ppl.py --model models/base --adapter models/adapters/zhihu-qwen05b-lora --out results/ppl_tuned.json
.venv/bin/python scripts/eval_mc.py --model models/base --out results/cmmlu_base.json
.venv/bin/python scripts/eval_mc.py --model models/base --adapter models/adapters/zhihu-qwen05b-lora --out results/cmmlu_tuned.json
.venv/bin/python scripts/eval_ceval.py --model models/base --out results/ceval_base.json
.venv/bin/python scripts/eval_ceval.py --model models/base --adapter models/adapters/zhihu-qwen05b-lora --out results/ceval_tuned.json
.venv/bin/python scripts/generate_samples.py
.venv/bin/python scripts/make_figures.py
