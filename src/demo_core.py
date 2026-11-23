"""Inference glue shared by the two demo servers (stdlib + FastAPI).

`predict` is model-optional: with no adapter on disk it answers from the analytic
verifier alone, which is exactly the "verifier-only mode" the paper describes for
cold-start deployment.
"""
from __future__ import annotations

import base64
import io
import re

from PIL import Image, ImageDraw

from .codec import cell_text
from .verifier import (COLORS, accepted_cells, color_components, evidence_text,
                       verify_count, verify_exist)


def parse_question(question: str):
    color = next((c for c in COLORS if f"{c}色" in question), None)
    shape = "圆" if "圆形" in question else ("方" if "方形" in question else None)
    if "第几行" in question:
        qtype = "loc"
    elif "几个" in question:
        qtype = "count"
    else:
        qtype = "exist"
    grid = None
    m = re.search(r"共(\d+)行", question)
    if m:
        grid = int(m.group(1))
    return qtype, color, shape, grid


def overlay(img: Image.Image, grid: int, accepted, comps):
    im = img.copy().convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    cell = im.size[0] // grid
    for i in range(1, grid):
        d.line([(i * cell, 0), (i * cell, im.size[1])], fill=(180, 180, 180, 160))
        d.line([(0, i * cell), (im.size[0], i * cell)], fill=(180, 180, 180, 160))
    for (r, c) in accepted:
        d.rectangle([c * cell, r * cell, (c + 1) * cell - 1, (r + 1) * cell - 1],
                    fill=(60, 120, 220, 55), outline=(30, 80, 200), width=2)
    for comp in comps:
        b = comp["box"]
        d.rectangle(b, outline=(20, 160, 90), width=2)
        cx, cy = comp["center"]
        d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(20, 160, 90))
    return im


def predict(img, question: str, grid: int = 4, model=None, max_new: int = 24) -> dict:
    if isinstance(img, str):
        img = Image.open(img)
    img = img.convert("RGB")
    qtype, color, shape, qgrid = parse_question(question)
    grid = qgrid or grid
    out = {"question": question, "type": qtype, "grid": grid,
           "color": color, "shape": shape, "source": "verifier-only"}
    comps = color_components(img, color) if color else []
    accepted = accepted_cells(img, color, grid) if color else []

    if model is not None:
        out["model_answer"] = model.answer(img, question, max_new=max_new)

    if qtype == "loc" and comps:
        r = min(grid - 1, int(comps[0]["center"][1] // (256 // grid)))
        c = min(grid - 1, int(comps[0]["center"][0] // (256 // grid)))
        out["answer"] = cell_text(r, c)
        out["confidence"] = "high" if len(accepted) == 1 else "ambiguous"
    elif qtype == "count" and color:
        out["answer"] = f"{verify_count(img, color)}个"
        out["confidence"] = "high"
    elif qtype == "exist" and color:
        out["answer"] = "有" if verify_exist(img, color) else "没有"
        out["confidence"] = "high" if shape is None else "colour-only"
    else:
        out["answer"] = "(无法解析问题)"
        out["confidence"] = "n/a"

    out["evidence"] = evidence_text(img, color, grid) if color else ""
    out["accepted_cells"] = [list(a) for a in accepted]
    out["n_components"] = len(comps)
    buf = io.BytesIO()
    overlay(img, grid, accepted, comps).save(buf, format="PNG")
    out["overlay_png"] = base64.b64encode(buf.getvalue()).decode()
    return out