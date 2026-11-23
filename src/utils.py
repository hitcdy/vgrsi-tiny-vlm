"""Shared helpers: seeding, JSON IO, timing."""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np


def seed_all(seed: int = 42) -> None:
    """Fix every RNG we touch (Sec. 5 of the report claims seed=42 everywhere)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # CPU SDPA kernels are not bitwise stable
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def merge_json(path: str, patch: dict) -> dict:
    cur = read_json(path, {}) or {}
    cur.update(patch)
    write_json(path, cur)
    return cur


class Timer:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.dt = time.time() - self.t0
        if self.dt > 2.0:
            print(f"[time] {self.name}: {self.dt:.1f}s")
        return False