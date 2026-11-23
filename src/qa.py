"""Question construction: loc (grounding) / count / exist.

Three deliberate fixes over the pilot (lab06), all of them visible in Table 1:

1. loc targets only classes that occur exactly once, so the question is
   well-posed (the pilot sometimes had two instances in different cells).
2. count targets are drawn from classes with 2-3 instances whenever the scene
   has one, otherwise the pilot's "always 1" saturation reappears.
3. half of the exist questions are *negative* (an absent colour), which removes
   the "answer = 有" degenerate strategy that made the pilot's exist subset
   uninformative.

Each evaluation image therefore yields 3 questions; 50 images -> 150 questions.
"""
from __future__ import annotations

import random

from .codec import cell_of, cell_text
from .verifier import COLORS


def name_of(cls: str) -> str:
    return f"{cls[0]}色{cls[1:]}形"


def _counts(items):
    out = {}
    for it in items:
        out[it["cls"]] = out.get(it["cls"], 0) + 1
    return out


def build_questions(meta: dict, limit: int = None, seed: int = 42):
    grid = meta["grid"]
    rnd = random.Random(seed)
    qs = []
    for sid, rec in meta["gt"].items():
        items = rec["items"]
        cnt = _counts(items)
        singles = [c for c, n in cnt.items() if n == 1]
        multis = [c for c, n in cnt.items() if n >= 2]
        if not singles:
            continue
        # --- loc: unique instance only, so that gold is unambiguous
        cls = singles[rnd.randrange(len(singles))]
        box = next(it["box"] for it in items if it["cls"] == cls)
        r, c = cell_of(box, grid)
        qs.append(dict(id=f"L{sid}", type="loc", img=sid, target=cls,
                       q=(f"图中的{name_of(cls)}位于第几行第几列？"
                          f"请用「第X行第Y列」回答（共{grid}行{grid}列）。"),
                       gold=cell_text(r, c), cell=[r, c], box=box))
        # --- count: prefer a class with 2-3 instances
        ccls = multis[rnd.randrange(len(multis))] if multis else cls
        qs.append(dict(id=f"C{sid}", type="count", img=sid, target=ccls,
                       q=f"图中有几个{name_of(ccls)}？", gold=f"{cnt[ccls]}个", n=cnt[ccls]))
        # --- exist: alternate positive / negative
        if len(qs) // 3 % 2 == 0:
            ecls, gold = cls, "有"
        else:
            absent = [f"{c}{s}" for c in COLORS for s in ("圆", "方") if f"{c}{s}" not in cnt]
            if not absent:
                ecls, gold = cls, "有"
            else:
                ecls, gold = absent[rnd.randrange(len(absent))], "没有"
        qs.append(dict(id=f"E{sid}", type="exist", img=sid, target=ecls,
                       q=f"图里有没有{name_of(ecls)}？", gold=gold,
                       present=(gold == "有")))
    if limit:
        qs = qs[:limit]
    return qs


def build_pool(meta: dict, seed: int = 7):
    """Unlabelled scenario pool for the self-improvement loop.

    Only the image and the target class are used at training time; the verifier
    supplies the label.  `gold` is kept solely for post-hoc analysis and is never
    read by `src.experiments` when building training samples.
    """
    grid = meta["grid"]
    rnd = random.Random(seed)
    qs = []
    for sid, rec in meta["gt"].items():
        cnt = _counts(rec["items"])
        singles = [c for c, n in cnt.items() if n == 1]
        for cls in singles[:2]:
            box = next(it["box"] for it in rec["items"] if it["cls"] == cls)
            r, c = cell_of(box, grid)
            qs.append(dict(id=f"P{sid}_{cls}", type="loc", img=sid, target=cls,
                           q=(f"图中的{name_of(cls)}位于第几行第几列？"
                              f"请用「第X行第Y列」回答（共{grid}行{grid}列）。"),
                           gold=cell_text(r, c), cell=[r, c], box=box))
        cls = singles[rnd.randrange(len(singles))] if singles else None
        if cls:
            absent = [f"{c}{s}" for c in COLORS for s in ("圆", "方") if f"{c}{s}" not in cnt]
            for ecls in ([cls] + absent[:1]):
                qs.append(dict(id=f"Q{sid}_{ecls}", type="exist", img=sid, target=ecls,
                               q=f"图里有没有{name_of(ecls)}？",
                               gold="有" if ecls in cnt else "没有", present=ecls in cnt))
    return qs