#!/bin/bash
set -e
AD=models/adapters/zhihu-qwen05b-lora
.venv/bin/python scripts/eval_mc.py --model models/base --out results/cmmlu_base.json
.venv/bin/python scripts/eval_mc.py --model models/base --adapter $AD --out results/cmmlu_tuned.json
.venv/bin/python scripts/eval_ceval.py --model models/base --out results/ceval_base.json
.venv/bin/python scripts/eval_ceval.py --model models/base --adapter $AD --out results/ceval_tuned.json
.venv/bin/python scripts/generate_samples.py
.venv/bin/python scripts/make_figures.py
