"""Deterministic stand-in for the backbone.

Purpose: exercise the whole pipeline (arms -> tables -> figures -> demo) on a
machine with no model weights / no network, and in CI.  It is NOT a model: the
answer distributions below are hand-set constants.  Everything produced with
this backend lands in `runs_mock/` and every table/图 it feeds is watermarked
"MOCK - NOT FOR SUBMISSION" so simulated numbers can never leak into the paper.
"""
from __future__ import annotations

import hashlib
import time


def _u(key: str) -> float:
    """Deterministic uniform [0,1) from a string."""
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    return int(h, 16) / 0xFFFFFFFF


class MockVLM:
    backend = "mock"

    def __init__(self, seed: int = 0, lora_r: int = 8, **kw):
        self.seed = seed
        self.skill = 0.0
        self.stats = {"total": 256_484_928, "trainable": 1_363_968, "ratio_pct": 0.529}
        self.latency = 0.004

    # ---------- inference ----------
    def answer(self, img, question: str, hint: str = None, max_new: int = 24) -> str:
        time.sleep(self.latency)
        key = question + "|" + str(self.seed) + ("|hint" if hint else "")
        u = _u(key)
        if hint:
            self.skill = min(0.9, self.skill + 0.02)  # evidence nudges the mock
        if "第几行第几列" in question:
            p = 0.05 + 0.55 * self.skill
            if u < p:
                grid = 8 if "共8行" in question else 4
                return f"第{1 + int(_u(key + 'r') * grid)}行第{1 + int(_u(key + 'c') * grid)}列"
            if u < p + 0.18:  # the pilot's enumerated-scan artefact
                return "第1行第2列第3列第4列"
            return "第1行第1列"
        if "几个" in question:
            p = 0.25 + 0.5 * self.skill
            if u < p:
                return "1个"
            return question  # prompt echo
        p = 0.3 + 0.45 * self.skill
        if u < p:
            return "有" if "有没有" in question and _u(key + "y") < 0.5 else "没有"
        return "有"

    # ---------- training ----------
    def sft(self, samples, steps: int = 60, lr: float = 1e-3, seed: int = 0, log_every=10):
        t0 = time.time()
        curve = []
        for i in range(steps):
            self.skill += (0.85 - self.skill) * 0.012
            curve.append([i + 1, round(2.6 * (1 - self.skill) + 0.4, 4)])
        return {"curve": curve, "steps": steps, "epochs": 1,
                "seconds": round(time.time() - t0, 1), "pool": len(samples)}

    def nudge(self, amount: float):
        self.skill = min(0.9, self.skill + amount)

    def snapshot(self):
        return {"skill": self.skill}

    def restore(self, snap):
        self.skill = snap.get("skill", 0.0)

    # ---------- io ----------
    def save(self, path: str):
        return None

    def load(self, path: str):
        return self