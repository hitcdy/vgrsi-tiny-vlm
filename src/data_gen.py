"""Parameterised synthetic grounding benchmark (self-built dataset).

Why synthetic: the target skill is *absolute* spatial grounding on a fixed grid,
where a ground-truth cell exists by construction and an analytic verifier can
therefore act as judge without any human label.  Two difficulty settings:

    grid=4  cell 64 px, objects 34-54 px   (objects straddle borders often)
    grid=8  cell 32 px, objects 16-28 px   (finer, but per-cell evidence is weaker)

Splits are generated with disjoint seeds so that the RSI pool, the seed SFT set
and the evaluation set never share an image:
    train  seed 2026
    pool   seed 777   (used only by the verifier-gated loop; no gold is read)
    eval   seed 42    (never seen during training)
"""
from __future__ import annotations

import os
import random

from PIL import Image, ImageDraw

from .utils import read_json, write_json
from .verifier import COLORS

IMG = 256
SHAPES = ("圆", "方")
CLASSES = [f"{c}{s}" for c in COLORS for s in SHAPES]

SPLIT_SEEDS = {"train": 2026, "pool": 777, "eval": 42}


def size_range(grid: int):
    return (34, 54) if grid <= 4 else (16, 28)


def draw_scene(rnd: random.Random, grid: int, n_obj: int):
    """Return [(class, box), ...]; classes[0] is guaranteed to be a singleton."""
    lo, hi = size_range(grid)
    pool = CLASSES[:]
    rnd.shuffle(pool)
    singleton = pool[0]
    chosen = [singleton]
    rest = pool[1:]
    if n_obj >= 3 and rnd.random() < 0.65:
        dup = rest.pop(0)
        chosen += [dup, dup]
    while len(chosen) < n_obj and rest:
        chosen.append(rest.pop(0))
    rnd.shuffle(chosen)
    items = []
    for cls in chosen:
        for _ in range(30):  # rejection sampling against overlaps
            s = rnd.randint(lo, hi)
            x, y = rnd.randint(6, IMG - s - 6), rnd.randint(6, IMG - s - 6)
            box = [x, y, x + s, y + s]
            if all(not (box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3])
                   for _, b in items):
                items.append((cls, box))
                break
    return items


def render(items, path: str, width: int = 4) -> None:
    im = Image.new("RGB", (IMG, IMG), "white")
    d = ImageDraw.Draw(im)
    for cls, box in items:
        color, shape = cls[0], cls[1:]
        if shape == "圆":
            d.ellipse(box, outline=COLORS[color], width=width)
        else:
            d.rectangle(box, outline=COLORS[color], width=width)
    im.save(path)


def make_split(split: str, n: int, grid: int, out_dir: str, n_obj=(1, 4)) -> dict:
    seed = SPLIT_SEEDS[split] + 1000 * grid
    rnd = random.Random(seed)
    sub = os.path.join(out_dir, f"g{grid}", split)
    os.makedirs(sub, exist_ok=True)
    gt = {}
    width = 4 if grid <= 4 else 3
    for i in range(n):
        sid = f"{split[0]}{i:03d}"
        items = draw_scene(rnd, grid, rnd.randint(*n_obj))
        render(items, os.path.join(sub, f"{sid}.png"), width=width)
        gt[sid] = {"grid": grid, "items": [{"cls": c, "box": b} for c, b in items]}
    meta = {"split": split, "grid": grid, "seed": seed, "n": n, "dir": sub, "gt": gt}
    write_json(os.path.join(out_dir, f"g{grid}", f"{split}.json"), meta)
    return meta


def ensure_data(out_dir: str, grids=(4,), n_train=60, n_pool=40, n_eval=50) -> dict:
    metas = {}
    for g in grids:
        for split, n in (("train", n_train), ("pool", n_pool), ("eval", n_eval)):
            path = os.path.join(out_dir, f"g{g}", f"{split}.json")
            if os.path.exists(path):
                metas[(g, split)] = read_json(path)
            else:
                metas[(g, split)] = make_split(split, n, g, out_dir)
    return metas