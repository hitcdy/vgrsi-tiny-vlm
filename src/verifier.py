"""Analytic (non-neural) verifier: colour connected components + cell coverage.

This module is the reward source of VGRSI.  It never reads the model's text; it
only reads the image, so it is an independent judge.  Zero training, millisecond
cost, and therefore usable in an on-device self-improvement loop.

Components are cached per (image fingerprint, colour): the same scene is asked
about many times (every cell, several question types), and one BFS per scene is
enough.  Without the cache the verifier audit is O(questions x cells) BFS calls.

Known limitations (measured in Table 4 of the report):
  * same-colour objects that touch merge into one component  -> count under-estimate
  * a ring outline split by occlusion breaks into pieces      -> count over-estimate
  * an object straddling a cell border is accepted for both   -> cross-cell false accept
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np
from PIL import Image

from .codec import cell_box

COLORS = {"红": (220, 50, 50), "蓝": (50, 80, 220), "绿": (40, 170, 70), "黄": (230, 190, 40)}
TOL = 9000
MIN_AREA = 24
_CACHE: dict = {}


def to_array(img) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, str):
        img = Image.open(img)
    return np.asarray(img.convert("RGB"), dtype=np.int32)


def color_mask(arr: np.ndarray, color: str) -> np.ndarray:
    ref = np.array(COLORS[color], dtype=np.int32)
    return ((arr - ref) ** 2).sum(axis=2) < TOL


def components(mask: np.ndarray, min_area: int = MIN_AREA):
    """8-neighbour connected components of a boolean mask (sparse: rings only)."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        pts = []
        while q:
            y, x = q.popleft()
            pts.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
        if len(pts) < min_area:
            continue
        py = [p[0] for p in pts]
        px = [p[1] for p in pts]
        out.append({"box": [min(px), min(py), max(px) + 1, max(py) + 1],
                    "area": len(pts),
                    "center": [float(np.mean(px)), float(np.mean(py))]})
    out.sort(key=lambda c: -c["area"])
    return out


def _fp(arr: np.ndarray, color: str) -> str:
    h = hashlib.md5(arr.tobytes())
    h.update(color.encode())
    return h.hexdigest()


def color_components(img, color: str):
    arr = to_array(img)
    key = _fp(arr, color)
    if key not in _CACHE:
        _CACHE[key] = components(color_mask(arr, color))
    return _CACHE[key]


def clear_cache():
    _CACHE.clear()


def cell_of_point(x: float, y: float, grid: int, img_size: int = 256):
    cell = img_size // grid
    return min(grid - 1, int(y // cell)), min(grid - 1, int(x // cell))


def verify_cell(img, color: str, cell, grid: int = 4):
    """Is `cell` a correct localisation of an object of `color`?

    Accepted when a component centre falls inside the cell, or when the cell
    covers >=50% of a component's area.
    """
    comps = color_components(img, color)
    cb = cell_box(cell[0], cell[1], grid=grid)
    if not comps:
        return False, {"reason": "colour absent", "components": 0, "coverage": 0.0}
    best = 0.0
    for c in comps:
        cx, cy = c["center"]
        if cb[0] <= cx < cb[2] and cb[1] <= cy < cb[3]:
            return True, {"reason": "centre in cell", "components": len(comps),
                          "coverage": 1.0, "area": c["area"]}
        x1, y1 = max(cb[0], c["box"][0]), max(cb[1], c["box"][1])
        x2, y2 = min(cb[2], c["box"][2]), min(cb[3], c["box"][3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        best = max(best, inter / c["area"] if c["area"] else 0.0)
    ok = best >= 0.5
    return ok, {"reason": "coverage>=50%" if ok else "no coverage",
                "components": len(comps), "coverage": round(best, 3)}


def accepted_cells(img, color: str, grid: int = 4):
    """All cells the verifier would accept - one components pass, O(grid^2) cheap."""
    comps = color_components(img, color)
    if not comps:
        return []
    cell = 256 // grid
    hit = set()
    for c in comps:
        cx, cy = c["center"]
        hit.add(cell_of_point(cx, cy, grid))
        box = c["box"]
        for r in range(grid):
            for cc in range(grid):
                cb = [cc * cell, r * cell, (cc + 1) * cell, (r + 1) * cell]
                x1, y1 = max(cb[0], box[0]), max(cb[1], box[1])
                x2, y2 = min(cb[2], box[2]), min(cb[3], box[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                if c["area"] and inter / c["area"] >= 0.5:
                    hit.add((r, cc))
    return sorted(hit)


def verify_count(img, color: str) -> int:
    return len(color_components(img, color))


def verify_exist(img, color: str) -> bool:
    return verify_count(img, color) > 0


def visible_colors(img):
    arr = to_array(img)
    return {name: len(color_components(arr, name)) for name in COLORS
            if color_components(arr, name)}


def evidence_text(img, color: str, grid: int = 4) -> str:
    """Structured evidence fed back to the model (the RSI correction prompt)."""
    comps = color_components(img, color)
    if not comps:
        return f"{color}色像素连通块 0 处（图中没有{color}色图形）"
    parts = []
    for i, c in enumerate(comps[:3], 1):
        r, col = cell_of_point(c["center"][0], c["center"][1], grid)
        parts.append(f"第{i}块：{c['area']} 像素，中心在第{r + 1}行第{col + 1}列")
    return f"{color}色像素连通块 {len(comps)} 处；" + "；".join(parts)