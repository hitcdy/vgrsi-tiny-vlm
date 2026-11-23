"""Model-free audit of the analytic verifier (Table 4 of the paper).

Runs in seconds, no weights needed, and it is the evidence for RQ1: before a
verifier may supply labels we must know how often it is right.

  loc_recall        P(verifier accepts the ground-truth cell)
  mean_extra        #extra cells accepted besides ground truth (ambiguity)
  random_pass       pass rate of a uniform random guesser  = |accepted|/grid^2
  count_exact/mae   analytic colour-component count vs ground truth
  exist_acc         positive/negative existence verdicts
"""
from __future__ import annotations

import os
import time

from . import data_gen, qa
from .config import load as load_cfg
from .evaluate import load_img
from .utils import write_json
from .verifier import accepted_cells, verify_count, verify_exist


def audit_grid(grid: int, cfg: dict):
    metas = data_gen.ensure_data(cfg["data"]["out_dir"], grids=(grid,),
                                 n_train=cfg["data"]["n_train"],
                                 n_pool=cfg["data"]["n_pool"],
                                 n_eval=cfg["data"]["n_eval"])
    meta = metas[(grid, "eval")]
    qs = qa.build_questions(meta, seed=42)
    out = {"grid": grid, "n_questions": len(qs), "loc": {}, "count": {}, "exist": {}}

    recall = extras = ambig = n = 0
    t0 = time.time()
    for q in [x for x in qs if x["type"] == "loc"]:
        img = load_img(meta, q["img"])
        cells = accepted_cells(img, q["target"][0], grid)
        n += 1
        gt_in = tuple(q["cell"]) in cells
        recall += 1 if gt_in else 0
        extras += len(cells) - (1 if gt_in else 0)
        ambig += 1 if len(cells) > 1 else 0
    out["loc"] = {"n": n,
                  "recall_pct": round(100.0 * recall / n, 1) if n else 0.0,
                  "mean_extra_accepts": round(extras / n, 2) if n else 0.0,
                  "ambiguous_pct": round(100.0 * ambig / n, 1) if n else 0.0,
                  "random_pass_pct": round(100.0 * (extras + recall) / (n * grid * grid), 2) if n else 0.0}

    exact = tot = err = fp_absent = absent_tot = 0
    for sid, rec in meta["gt"].items():
        img = load_img(meta, sid)
        truth = {}
        for it in rec["items"]:
            truth[it["cls"][0]] = truth.get(it["cls"][0], 0) + 1
        for color in ("红", "蓝", "绿", "黄"):
            pred = verify_count(img, color)
            if color in truth:
                tot += 1
                err += abs(pred - truth[color])
                exact += 1 if pred == truth[color] else 0
            else:
                absent_tot += 1
                fp_absent += 1 if pred > 0 else 0
    out["count"] = {"n": tot,
                    "exact_pct": round(100.0 * exact / tot, 1) if tot else 0.0,
                    "mae": round(err / tot, 2) if tot else 0.0,
                    "absent_fp_pct": round(100.0 * fp_absent / absent_tot, 1) if absent_tot else 0.0}

    eok = etot = 0
    for q in [x for x in qs if x["type"] == "exist"]:
        img = load_img(meta, q["img"])
        eok += 1 if verify_exist(img, q["target"][0]) == (q["gold"] == "有") else 0
        etot += 1
    out["exist"] = {"n": etot, "acc_pct": round(100.0 * eok / etot, 1) if etot else 0.0}
    out["seconds"] = round(time.time() - t0, 1)
    print(f"[audit] grid {grid}: {out}")
    return out


def main():
    cfg = load_cfg("configs/exp.yaml")
    res = {"grids": [audit_grid(g, cfg) for g in (4, 8)]}
    write_json("runs/verifier_audit.json", res)
    lines = ["| Grid | loc recall | mean extra accepts | ambiguous | random-pass | count exact | count MAE | count FP (absent) | exist acc |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in res["grids"]:
        lines.append(f"| {r['grid']}×{r['grid']} | {r['loc']['recall_pct']}% | "
                     f"{r['loc']['mean_extra_accepts']} | {r['loc']['ambiguous_pct']}% | "
                     f"{r['loc']['random_pass_pct']}% | {r['count']['exact_pct']}% | "
                     f"{r['count']['mae']} | {r['count']['absent_fp_pct']}% | {r['exist']['acc_pct']}% |")
    md = "\n".join(lines)
    os.makedirs("paper", exist_ok=True)
    with open("paper/table4_verifier.md", "w", encoding="utf-8") as f:
        f.write("# Table 4. Analytic verifier reliability (model-free audit, no weights needed)\n\n" + md + "\n")
    print(md)


if __name__ == "__main__":
    main()