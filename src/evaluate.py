"""Evaluation: per-question records, lenient+strict accuracy, optional
inference-time verifier correction (arm C / the VGRSI inner loop)."""
from __future__ import annotations

import os
import time

from PIL import Image

from .codec import cell_text, parse_cell, parse_cell_strict
from .metrics import aggregate, judge
from .verifier import evidence_text, verify_cell, verify_count, verify_exist


def split_dir(meta: dict) -> str:
    return meta["dir"]


def load_img(meta: dict, sid: str):
    return Image.open(os.path.join(meta["dir"], f"{sid}.png")).convert("RGB")


def verify_answer(q: dict, pred: str, img, grid: int):
    """Analytic verdict on a candidate answer -> (ok, verifier_label | None)."""
    if q["type"] == "loc":
        cell = parse_cell_strict(pred, grid)[0] or parse_cell(pred, grid)
        if cell is None:
            return False, None
        ok, _ = verify_cell(img, q["target"][0], cell, grid)
        return ok, cell_text(*cell)
    if q["type"] == "exist":
        body = (pred or "").replace("有没有", "")
        said = "没有" not in body and "有" in body
        truth = verify_exist(img, q["target"][0])
        return said == truth, ("有" if truth else "没有")
    if q["type"] == "count":
        import re
        nums = re.findall(r"\d+", pred or "")
        n = verify_count(img, q["target"][0])
        return (bool(nums) and int(nums[0]) == n), f"{n}个"
    return False, None


def run_eval(model, meta: dict, questions, grid: int, verifier_mode: bool = False,
             limit: int = 0):
    """verifier_mode=True -> arm C: re-ask with verifier evidence when rejected."""
    if limit:
        questions = questions[:limit]
    records = []
    t0 = time.time()
    for q in questions:
        img = load_img(meta, q["img"])
        t1 = time.time()
        pred = model.answer(img, q["q"])
        rec = judge(q, pred, grid)
        rec["latency_ms"] = round((time.time() - t1) * 1000, 1)
        rec["first_pred"] = pred
        rec["corrected"] = False
        if verifier_mode:
            ok, _ = verify_answer(q, pred, img, grid)
            if not ok:
                hint = evidence_text(img, q["target"][0], grid)
                pred2 = model.answer(img, q["q"], hint=hint)
                rec2 = judge(q, pred2, grid)
                rec2["latency_ms"] = rec["latency_ms"] + round((time.time() - t1) * 1000, 1)
                rec2["first_pred"] = pred
                rec2["final_pred"] = pred2
                rec2["corrected"] = True
                rec2["hint"] = hint
                rec = rec2
        records.append(rec)
    return records, aggregate(records), round(time.time() - t0, 1)