"""Figures 2-5: per-type accuracy, training curves, pool growth, verifier overlay.

Usage: python -m src.figures --backend real --grid 4
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from .utils import read_json
from .verifier import accepted_cells, color_components


def _wm(fig, wm):
    if wm:
        fig.text(0.5, 0.5, "MOCK - NOT FOR SUBMISSION", fontsize=26, color="red",
                 ha="center", va="center", alpha=0.25, rotation=25)


def fig_accuracy(per_arm, grid, out, wm=""):
    labels = ["zero_shot", "sft", "sft_infer_verifier", "seed_only_cont", "vgrs"]
    names = {"zero_shot": "A zero-shot", "sft": "B SFT", "sft_infer_verifier": "C SFT+verifier",
             "seed_only_cont": "D seed-only", "vgrs": "E VGRSI"}
    types = ["loc", "count", "exist", "acc_strict"]
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4))
    present = [l for l in labels if l in per_arm]
    for i, t in enumerate(types):
        vals = [per_arm[a][0].get(t, 0.0) for a in present]
        ax.bar([x + i * width for x in range(len(present))], vals, width, label=t)
    ax.set_xticks([x + 1.5 * width for x in range(len(present))])
    ax.set_xticklabels([names[a] for a in present], rotation=15)
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"Per-type accuracy, grid {grid}x{grid}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _wm(fig, wm)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_curves(backend, grid, out, wm=""):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for p in sorted(glob.glob(os.path.join("runs", backend, f"g{grid}", "s*", "results.json"))):
        r = read_json(p, {}) or {}
        curve = ((r.get("arms") or {}).get("sft_train") or {}).get("train", {}).get("curve")
        if curve:
            ax.plot([c[0] for c in curve], [c[1] for c in curve], label="SFT loss")
        for rd in ((r.get("arms") or {}).get("vgrs") or {}).get("per_round", []) or []:
            c = rd.get("train", {}).get("curve") or []
            if c:
                ax.plot([c[0] for c in c], [c[1] for c in c], "--", label=f"VGRSI r{rd['round']}")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("training loss (answer tokens only)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _wm(fig, wm)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_pool(backend, grid, out, wm=""):
    p = os.path.join("runs", backend, f"g{grid}", "s42", "results.json")
    r = read_json(p, {}) or {}
    curve = ((r.get("arms") or {}).get("vgrs") or {}).get("pool_curve") or []
    fig, ax = plt.subplots(figsize=(5, 3.4))
    if curve:
        ax.plot(range(len(curve)), curve, "o-")
        for i, v in enumerate(curve):
            ax.annotate(str(v), (i, v), textcoords="offset points", xytext=(0, 6), fontsize=8)
    ax.set_xlabel("self-training round")
    ax.set_ylabel("training pool size")
    ax.set_title("Verifier-gated pool growth")
    ax.grid(alpha=0.3)
    _wm(fig, wm)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_overlay(backend, grid, out, n=4, wm=""):
    """Show accepted-cell ambiguity: the verifier is colour-only, so a straddling
    object is accepted in two cells.  This is the visual for Table 4."""
    meta_path = os.path.join("data", f"g{grid}", "eval.json")
    meta = read_json(meta_path, {})
    if not meta:
        return
    sids = list(meta["gt"])[:n]
    fig, axes = plt.subplots(1, len(sids), figsize=(3.2 * len(sids), 3.4))
    if len(sids) == 1:
        axes = [axes]
    cell = 256 // grid
    for ax, sid in zip(axes, sids):
        img = Image.open(os.path.join(meta["dir"], f"{sid}.png")).convert("RGB")
        rec = meta["gt"][sid]
        color = rec["items"][0]["cls"][0]
        cells = accepted_cells(img, color, grid)
        d = ImageDraw.Draw(img)
        for (r, c) in cells:
            d.rectangle([c * cell, r * cell, (c + 1) * cell, (r + 1) * cell],
                        outline=(120, 120, 120), width=2)
        for (r, c) in cells:
            d.line([c * cell, r * cell, (c + 1) * cell, (r + 1) * cell], fill=(150, 150, 150))
        ax.imshow(img)
        ax.set_title(f"{sid}: {len(cells)} accepted cells", fontsize=9)
        ax.axis("off")
    _wm(fig, wm)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_failures(backend, grid, out, wm=""):
    p = os.path.join("runs", backend, f"g{grid}", "s42", "results.json")
    r = read_json(p, {}) or {}
    arms = r.get("arms") or {}
    names, fp, echo = [], [], []
    for a in ("zero_shot", "sft", "sft_infer_verifier", "vgrs"):
        if a in arms:
            m = arms[a]["metrics"]
            names.append(a)
            fp.append(m.get("parser_fp", 0))
            echo.append(m.get("echo", 0))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], fp, 0.4, label="parser false positives")
    ax.bar([i + 0.2 for i in x], echo, 0.4, label="question echo")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylabel("#questions")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _wm(fig, wm)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="real")
    ap.add_argument("--grid", type=int, default=4)
    args = ap.parse_args()
    wm = "MOCK" if args.backend == "mock" else ""
    from .tables import collect
    per_arm, _, _ = collect(args.backend, args.grid)
    tag = ("mock_" if wm else "") + f"g{args.grid}_"
    os.makedirs("paper/figs", exist_ok=True)
    fig_accuracy(per_arm, args.grid, f"paper/figs/{tag}fig2_accuracy.png", wm)
    fig_curves(args.backend, args.grid, f"paper/figs/{tag}fig3_loss.png", wm)
    fig_pool(args.backend, args.grid, f"paper/figs/{tag}fig4_pool.png", wm)
    fig_overlay(args.backend, args.grid, f"paper/figs/{tag}fig5_verifier.png", 4, wm)
    fig_failures(args.backend, args.grid, f"paper/figs/{tag}fig6_failures.png", wm)
    print("[figs] -> paper/figs/")


if __name__ == "__main__":
    main()