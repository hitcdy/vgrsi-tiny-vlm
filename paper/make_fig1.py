"""Fig. 1: the VGRSI loop, drawn programmatically so it can be regenerated."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "paper/figs/fig1_loop.png"
BLUE, GREEN, ORANGE, GREY, PURPLE = "#1b3a6b", "#1a7f4b", "#b5761a", "#4a5361", "#5b3a8c"


def box(ax, x, y, w, h, text, fc="#eef3fb", ec=BLUE, fs=8.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.03",
                                linewidth=1.15, facecolor=fc, edgecolor=ec))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#14202e", linespacing=1.35)


def arrow(ax, p1, p2, color=GREY, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.2, color=color, linestyle=ls,
                                 shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.set_xlim(0, 10.2); ax.set_ylim(0, 7.0); ax.axis("off")

    cx, w = 2.55, 4.9          # main column
    h = 0.78
    steps = [
        (5.95, "unlabelled pool split: 40 scenes (seed 777)", "#eef3fb", BLUE),
        (4.90, "SmolVLM-256M + LoRA answers the prompt\n(greedy, 24 new tokens)", "#eef3fb", BLUE),
        (3.85, "analytic verifier: colour components\n+ cell coverage  ->  ok / not ok", "#f1f7f2", GREEN),
        (2.80, "GATE: admit only wrong -> right flips\n(flip_only)", "#fdf3e7", ORANGE),
        (1.75, "pool grows 180 -> 200-218 samples;\nre-train the LoRA adapter, 60 steps", "#eef3fb", BLUE),
    ]
    for y, text, fc, ec in steps:
        box(ax, cx, y, w, h, text, fc=fc, ec=ec)

    for i in range(len(steps) - 1):
        arrow(ax, (cx + w / 2, steps[i][0]), (cx + w / 2, steps[i + 1][0] + h))
    ax.text(cx + w / 2 - 0.15, 5.60, "answer a", fontsize=7.3, color=GREY, ha="right", bbox=dict(fc="white", ec="none", pad=1))
    ax.text(cx + w / 2 - 0.15, 3.58, "verdict", fontsize=7.3, color=GREY, ha="right", bbox=dict(fc="white", ec="none", pad=1))
    ax.text(cx + w / 2 - 0.15, 2.53, "accepted samples", fontsize=7.3, color=ORANGE, ha="right", bbox=dict(fc="white", ec="none", pad=1))

    # reject branch: verifier -> re-ask -> back to the model
    rbx, rby, rbw, rbh = 7.75, 3.72, 2.25, 1.05
    box(ax, rbx, rby, rbw, rbh, "reject: re-ask with\nstructured evidence\n(arm C / round correction)",
        fc="#f7eef0", ec="#a33", fs=7.6)
    arrow(ax, (cx + w, 4.24), (rbx, 4.24), color="#a33", ls="--")
    arrow(ax, (rbx + rbw / 2, rby + rbh), (cx + w / 2 + 1.3, 4.90), color="#a33", ls="--")
    ax.text(cx + w + 0.05, 4.38, "not ok", fontsize=7.2, color="#a33", bbox=dict(fc="white", ec="none", pad=1))

    # round loop back to the model
    lx = cx - 0.55
    arrow(ax, (cx, 2.14), (lx, 2.14), color=PURPLE)
    arrow(ax, (lx, 2.14), (lx, 5.29), color=PURPLE)
    arrow(ax, (lx, 5.29), (cx, 5.29), color=PURPLE)
    ax.text(lx - 0.12, 3.7, "round r+1", fontsize=7.4, color=PURPLE, rotation=90,
            va="center", ha="center")

    # held-out eval split
    box(ax, 0.35, 0.35, 9.5, 0.72,
        "held-out evaluation split: 50 scenes (seed 42) - 150 questions, reported only, never trained on",
        fc="#f4f6f9", ec=GREY, fs=7.9)
    arrow(ax, (cx + w - 0.4, 1.75), (cx + w - 0.4, 1.07), color=GREY, ls=":")
    ax.text(cx + w - 0.3, 1.38, "evaluate every round", fontsize=7.1, color=GREY)

    fig.tight_layout()
    fig.savefig(OUT, dpi=170)
    print("[fig1] ->", OUT)


if __name__ == "__main__":
    main()