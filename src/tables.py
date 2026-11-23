"""Build Table 1 / Table 2 from runs/ (mean +- std over seeds).

Usage:  python -m src.tables --backend real --grid 4
Outputs paper/table1_main.md|.tex, paper/table2_ablation.md, paper/numbers.json.
Numbers produced from the mock backend are watermarked and must never be pasted
into the report.
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics as st

from .utils import read_json, write_json

ARM_LABEL = {
    "zero_shot": "SmolVLM-256M zero-shot (A)",
    "sft": "+ LoRA SFT on seed split (B)",
    "sft_infer_verifier": "+ inference-time verifier (C)",
    "seed_only_cont": "+ seed-only continued SFT (D, control)",
    "vgrs": "+ VGRSI self-training (E, ours)",
}
COLS = ["acc_lenient", "acc_strict", "loc", "count", "exist"]


SEED_DIR = __import__("re").compile(r"^s\d+$")


def collect(backend: str, grid: int):
    """Only seed directories (s42, s7, ...) - ablation dirs must not pollute the mean."""
    paths = sorted(p for p in glob.glob(os.path.join("runs", backend, f"g{grid}", "*", "results.json"))
                   if SEED_DIR.match(os.path.basename(os.path.dirname(p))))
    per_arm = {}
    meta = {}
    for p in paths:
        r = read_json(p, {}) or {}
        meta = r.get("meta", meta)
        params = r.get("params", {})
        for arm, v in (r.get("arms") or {}).items():
            if arm == "sft_train" or "metrics" not in v:
                continue
            m = v["metrics"]
            row = {c: (m[c]["acc_strict"] if c in ("loc", "count", "exist") else m.get(c))
                   for c in COLS}
            row["acc_lenient"] = m.get("acc_lenient", 0.0)
            row["acc_strict"] = m.get("acc_strict", 0.0)
            row["parser_fp"] = m.get("parser_fp", 0)
            row["iou"] = m.get("loc", {}).get("mean_iou", 0.0)
            row["latency_ms"] = m.get("latency_ms", 0.0)
            row["params_total"] = params.get("total", 0)
            row["params_train"] = params.get("trainable", 0)
            row["seconds"] = v.get("seconds", 0.0)
            per_arm.setdefault(arm, []).append(row)
    return per_arm, meta, len(paths)


def ms(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.1f}"
    return f"{st.mean(vals):.1f} ± {st.stdev(vals):.1f}"


def build(backend: str, grid: int, watermark: str = ""):
    per_arm, meta, n_seeds = collect(backend, grid)
    order = [a for a in ARM_LABEL if a in per_arm]
    lines = []
    if watermark:
        lines += [f"> **{watermark}**", ""]
    lines += [f"| Model | Acc (lenient) | Acc (strict) | loc | count | exist | Mean IoU | #Params (trainable) | Latency (ms) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for a in order:
        rows = per_arm[a]
        p_tot = rows[0]["params_total"]
        p_tr = rows[0]["params_train"]
        lines.append(f"| {ARM_LABEL[a]} | {ms(rows,'acc_lenient')} | {ms(rows,'acc_strict')} | "
                     f"{ms(rows,'loc')} | {ms(rows,'count')} | {ms(rows,'exist')} | "
                     f"{ms(rows,'iou')} | {p_tot/1e6:.1f} M ({p_tr/1e6:.2f} M) | {ms(rows,'latency_ms')} |")
    md = "\n".join(lines)

    # ---- deltas for the abstract
    num = {"backend": backend, "grid": grid, "seeds": n_seeds,
           "model_id": meta.get("model_id", ""), "n_eval_q": meta.get("n_eval_q", 0)}
    def _mean(arm, key="acc_strict"):
        rows = per_arm.get(arm) or []
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        return round(st.mean(vals), 1) if vals else None

    def _std(arm, key="acc_strict"):
        rows = per_arm.get(arm) or []
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        return round(st.stdev(vals), 1) if len(vals) > 1 else None

    if "zero_shot" in per_arm and "vgrs" in per_arm:
        z = _mean("zero_shot")
        v = _mean("vgrs")
        num["zero_shot_acc"] = z
        num["vgrs_acc"] = v
        num["abs_gain_pts"] = round(v - z, 1)
        num["rel_gain_pct"] = round(100.0 * (v - z) / z, 1) if z else None
    if "sft" in per_arm and "vgrs" in per_arm:
        s = _mean("sft")
        v = _mean("vgrs")
        num["sft_acc"] = s
        num["vgrs_vs_sft_pts"] = round(v - s, 1)
        num["vgrs_vs_sft_rel_pct"] = round(100.0 * (v - s) / s, 1) if s else None
    if "seed_only_cont" in per_arm and "sft" in per_arm:
        num["seed_only_acc"] = _mean("seed_only_cont")
        num["sft_acc_for_control"] = _mean("sft")
    if "sft_infer_verifier" in per_arm:
        num["arm_c_acc"] = _mean("sft_infer_verifier")
    for arm in ("zero_shot", "sft", "sft_infer_verifier", "seed_only_cont", "vgrs"):
        sd = _std(arm)
        if sd is not None:
            num[f"std_{arm}"] = sd
    if "vgrs" in per_arm:
        num["parser_fp"] = per_arm["vgrs"][0].get("parser_fp", 0)

    os.makedirs("paper", exist_ok=True)
    with open(f"paper/table1_main_g{grid}.md", "w", encoding="utf-8") as f:
        f.write(f"# Table 1. Main results, grid {grid}x{grid}, mean over {n_seeds} seed(s)\n\n" + md + "\n")
    with open(f"paper/table1_main_g{grid}.tex", "w", encoding="utf-8") as f:
        f.write(tex_table(per_arm, order, grid, n_seeds))
    write_json(f"paper/numbers_g{grid}.json", num)
    if grid == 4:                      # the docx builder reads this one
        write_json("paper/numbers.json", num)
    print(md)
    print("\n[numbers]", num)
    return num


def tex_table(per_arm, order, grid, n_seeds):
    head = ("\\begin{table}[t]\n\\centering\n\\small\n"
            "\\caption{Main results on the self-built $%dx%d$ grounding benchmark "
            "(mean over %d seed(s)).}\n\\label{tab:main}\n"
            "\\begin{tabular}{lcccccc}\n\\toprule\n"
            "Model & Acc$_{len}$ & Acc$_{strict}$ & loc & count & exist & IoU \\\\\n\\midrule\n"
            % (grid, grid, n_seeds))
    rows = []
    for a in order:
        r = per_arm[a][0]
        rows.append("%s & %.1f & %.1f & %.1f & %.1f & %.1f & %.3f \\\\" %
                    (ARM_LABEL[a].replace("&", "\\&"), r["acc_lenient"], r["acc_strict"],
                     r["loc"], r["count"], r["exist"], r["iou"]))
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"




def build_table2(backend: str, grid: int, watermark: str = ""):
    """Ablation: every run directory under runs/<backend>/g<grid>/ that carries a tag."""
    rows = []
    for p in sorted(glob.glob(os.path.join("runs", backend, f"g{grid}", "*", "results.json"))):
        r = read_json(p, {}) or {}
        meta = r.get("meta") or {}
        v = (r.get("arms") or {}).get("vgrs")
        if not v or "metrics" not in v:
            continue
        tag = meta.get("tag") or os.path.basename(os.path.dirname(p))
        seed = os.path.basename(os.path.dirname(p))
        if not meta.get("tag"):
            # untagged seed dirs: show the round axis, that is the ablation worth reading
            for rd in (v.get("per_round") or []):
                m2 = rd.get("metrics") or {}
                rows.append((f"round {rd['round']} (seed {seed})", rd["round"],
                             meta.get("pool_policy", ""), m2.get("acc_strict"),
                             (m2.get("loc") or {}).get("acc_strict"), m2.get("parser_fp")))
            continue
        m = v["metrics"]
        rows.append((tag, meta.get("rounds", ""), meta.get("pool_policy", ""),
                     m.get("acc_strict"), m.get("loc", {}).get("acc_strict"),
                     m.get("parser_fp")))
    if not rows:
        print("[table2] no tagged ablation runs yet")
        return None
    lines = ([f"> **{watermark}**", ""] if watermark else []) + [
        "| Variant | rounds | pool policy | Acc (strict) | loc (strict) | parser FP |",
        "|---|---|---|---|---|---|"]
    for tag, rds, pol, acc, loc, fp in rows:
        lines.append(f"| {tag} | {rds} | {pol} | {acc} | {loc} | {fp} |")
    md = "\n".join(lines)
    with open("paper/table2_ablation.md", "w", encoding="utf-8") as f:
        f.write(f"# Table 2. Ablation on grid {grid}x{grid}\n\n" + md + "\n")
    print(md)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="real")
    ap.add_argument("--grid", type=int, default=4)
    args = ap.parse_args()
    wm = "MOCK BACKEND - SIMULATED NUMBERS, NOT FOR SUBMISSION" if args.backend == "mock" else ""
    build(args.backend, args.grid, wm)
    build_table2(args.backend, args.grid, wm)


if __name__ == "__main__":
    main()