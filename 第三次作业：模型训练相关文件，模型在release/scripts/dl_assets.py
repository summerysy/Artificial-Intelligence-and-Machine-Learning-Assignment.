"""Download the base model and the raw Zhihu-KOL shard from the HF mirror."""
import os, sys, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.abspath(".cache/huggingface"))
from huggingface_hub import snapshot_download, hf_hub_download

def main():
    t0 = time.time()
    print("[1/2] model Qwen/Qwen2.5-0.5B-Instruct ...", flush=True)
    p = snapshot_download(
        "Qwen/Qwen2.5-0.5B-Instruct",
        local_dir="models/base",
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"],
    )
    print("model ->", p, f"{time.time()-t0:.0f}s", flush=True)
    print("[2/2] dataset wangrui6/Zhihu-KOL shard 0 ...", flush=True)
    f = hf_hub_download(
        "wangrui6/Zhihu-KOL",
        "data/train-00000-of-00005-a1278ede4e8c5cdb.parquet",
        repo_type="dataset",
        local_dir="data/raw",
    )
    print("dataset ->", f, f"{time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
