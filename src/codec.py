"""Grid codec: pixel box <-> (row, col), and answer parsing.

The strict/lenient split is a first-class part of this project (Sec. 4.4 of the
report): a tiny VLM frequently emits an enumerated scan ("row 1 col 2 col 3
col 4 ...") instead of a single answer.  A lenient regex then reports a hit by
accident.  Every metric is therefore reported twice:

    acc_lenient : first regex match wins  (what a careless eval script reports)
    acc_strict  : exactly one mention, no question echo - the honest number
    parser_fp   : ok_lenient and not ok_strict, i.e. an evaluation artefact
"""
from __future__ import annotations

import re

IMG = 256


def cell_of(box, grid: int = 4, img: int = IMG):
    """GT box -> (row, col) of its centre, 0-based."""
    cell = img // grid
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    return min(grid - 1, int(cy // cell)), min(grid - 1, int(cx // cell))


def cell_box(r: int, c: int, grid: int = 4, img: int = IMG):
    cell = img // grid
    return [c * cell, r * cell, (c + 1) * cell, (r + 1) * cell]


def cell_text(r: int, c: int) -> str:
    return f"第{r + 1}行第{c + 1}列"


def _mentions(text: str, grid: int):
    """All (row, col) mentions in various surface forms, in order."""
    hi = str(grid)
    t = text.strip()
    out = []
    for m in re.finditer(rf"第\s*([1-{hi}])\s*行\s*第\s*([1-{hi}])\s*列", t):
        out.append((int(m.group(1)) - 1, int(m.group(2)) - 1))
    if not out:
        for m in re.finditer(rf"[rR]\s*([1-{hi}])\s*[cC]\s*([1-{hi}])", t):
            out.append((int(m.group(1)) - 1, int(m.group(2)) - 1))
    if not out:
        for m in re.finditer(rf"([1-{hi}])\s*[,，\-、]\s*([1-{hi}])", t):
            out.append((int(m.group(1)) - 1, int(m.group(2)) - 1))
    return out


def parse_cell(text: str, grid: int = 4):
    """Lenient: first mention wins. Kept for comparability with the pilot."""
    m = _mentions(text, grid)
    return m[0] if m else None


def parse_cell_strict(text: str, grid: int = 4):
    """Strict: a single, unambiguous mention and no question echo.

    Returns (cell | None, reason).  reason in {"ok","ambiguous","echo","empty"}.
    """
    t = (text or "").strip()
    if not t:
        return None, "empty"
    if "？" in t or "?" in t:
        return None, "echo"
    m = _mentions(t, grid)
    if not m:
        return None, "empty"
    # an enumerated scan ("第1行第2列第3列第4列...") has a single regex match but
    # repeats the row/column token - that is the pilot's false-positive pattern.
    if t.count("列") > 1 or t.count("行") > 1:
        return None, "ambiguous"
    if len(set(m)) > 1:
        return None, "ambiguous"
    return m[0], "ok"


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0