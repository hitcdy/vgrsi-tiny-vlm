"""Config loading (yaml) with CLI override."""
from __future__ import annotations

import os

import yaml

DEFAULTS = {
    "seed": 42,
    "model_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "img": 256,
    "data": {"out_dir": "data", "n_train": 60, "n_pool": 40, "n_eval": 50,
             "n_obj_min": 1, "n_obj_max": 4},
    "train": {"lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "lr": 1e-3,
              "steps": 60, "extra_steps": 15, "max_new_tokens": 24},
    "rsi": {"rounds": 2, "steps_per_round": 60, "pool_policy": "flip_only"},
    "eval": {"limit": 0},
    "demo": {"port": 8501},
}


def _merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: str = "configs/exp.yaml") -> dict:
    cfg = DEFAULTS
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg