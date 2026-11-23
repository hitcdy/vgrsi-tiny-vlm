"""Fetch the backbone into ./models (works behind a mirror, skips the 3 GB of ONNX).

    python -u scripts/fetch_model.py                       # hf-mirror.com (CN default)
    python -u scripts/fetch_model.py --endpoint https://huggingface.co

Only ~520 MB is needed: model.safetensors + tokenizer/processor configs.  The
repository also ships ~3 GB of ONNX exports that HF would download otherwise.
Re-running resumes a partial download.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# HF_HOME must be writable: on this box %USERPROFILE%\.cache is access-denied, and
# huggingface_hub writes xet logs / refs there even when --local_dir is used.
os.environ.setdefault("HF_HOME", os.path.abspath(".hf_home"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO = os.environ.get("VGRSI_MODEL", "HuggingFaceTB/SmolVLM-256M-Instruct")
PATTERNS = ["*.json", "*.txt", "*.safetensors", "*.model"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT",
                                                        "https://hf-mirror.com"))
    ap.add_argument("--local_dir", default=os.path.join("models", "SmolVLM-256M-Instruct"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    from huggingface_hub import snapshot_download

    print(f"[fetch] repo={REPO}\n[fetch] endpoint={args.endpoint}\n[fetch] -> {args.local_dir}")
    t0 = time.time()
    path = snapshot_download(repo_id=REPO, local_dir=args.local_dir,
                             allow_patterns=PATTERNS, max_workers=args.workers,
                             endpoint=args.endpoint)
    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(path) for f in fs)
    print(f"[fetch] done in {time.time() - t0:.1f}s, {size / 1e6:.0f} MB at {path}")
    print("[fetch] now run:  $env:VGRSI_MODEL='" + os.path.abspath(path) + "'")


if __name__ == "__main__":
    sys.exit(main())