"""Judging and aggregation: lenient vs strict accuracy, pilot-style vs honest."""
from __future__ import annotations

import re

from .codec import iou, parse_cell, parse_cell_strict


def judge(q: dict, pred: str, grid: int = 4) -> dict:
    pred = (pred or "").strip()
    out = {"id": q["id"], "type": q["type"], "gold": q["gold"], "pred": pred,
           "ok_lenient": False, "ok_strict": False, "iou": 0.0,
           "echo": ("？" in pred or "?" in pred)}
    if q["type"] == "loc":
        cl = parse_cell(pred, grid)
        cs, reason = parse_cell_strict(pred, grid)
        out["ok_lenient"] = cl is not None and list(cl) == list(q["cell"])
        out["ok_strict"] = cs is not None and list(cs) == list(q["cell"])
        cell_used = cs or cl
        if cell_used:
            from .codec import cell_box
            out["iou"] = round(iou(cell_box(*cell_used, grid=grid), q["box"]), 4)
        out["reason"] = reason
        return out
    if q["type"] == "count":
        nums = re.findall(r"\d+", pred)
        out["ok_lenient"] = bool(nums) and int(nums[0]) == q["n"]
        # strict: exactly ONE integer in the answer. "1. 1. 1. 1." is the counting
        # analogue of the enumerated scan and must not score as a clean answer.
        out["ok_strict"] = (len(nums) == 1 and int(nums[0]) == q["n"]) and not out["echo"]
        return out
    body = pred.replace("有没有", "")
    said_yes = ("没有" not in body) and ("有" in body or "yes" in body.lower())
    said_no = ("没有" in body) or body.strip().lower() in {"no", "not"}
    if q["gold"] == "有":
        out["ok_lenient"] = said_yes and not said_no
    else:
        out["ok_lenient"] = said_no and not said_yes
    out["ok_strict"] = out["ok_lenient"] and not out["echo"]
    return out


def aggregate(records, types=("loc", "count", "exist")) -> dict:
    agg = {"n": len(records), "parser_fp": 0, "echo": 0}
    for t in types:
        rs = [r for r in records if r["type"] == t]
        if not rs:
            continue
        agg[t] = {
            "n": len(rs),
            "acc_lenient": round(100.0 * sum(r["ok_lenient"] for r in rs) / len(rs), 1),
            "acc_strict": round(100.0 * sum(r["ok_strict"] for r in rs) / len(rs), 1),
        }
        if t == "loc":
            agg[t]["mean_iou"] = round(sum(r["iou"] for r in rs) / len(rs), 4)
    allr = records
    agg["acc_lenient"] = round(100.0 * sum(r["ok_lenient"] for r in allr) / len(allr), 1)
    agg["acc_strict"] = round(100.0 * sum(r["ok_strict"] for r in allr) / len(allr), 1)
    agg["parser_fp"] = sum(1 for r in allr if r["ok_lenient"] and not r["ok_strict"])
    agg["echo"] = sum(1 for r in allr if r["echo"])
    lat = [r.get("latency_ms") for r in allr if r.get("latency_ms")]
    if lat:
        agg["latency_ms"] = round(sum(lat) / len(lat), 1)
    return agg