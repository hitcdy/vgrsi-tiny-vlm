"""Table 2 ablation: three variants of the VGRSI loop on grid 4 / seed 42, in parallel.

    python -u scripts/run_ablation.py           # parallel, ~25-35 min on CPU
    python -u scripts/run_ablation.py --seq

Variants
    r1        one round  (does the second round pay for itself?)
    allver    admit every verifier-correct sample, not only wrong->right flips
              (does the gate matter, or is any verifier-approved data fine?)
    firsttry  no evidence re-prompt: admit only samples the model already got right
              (does the correction path matter?)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS = [
    ("r1", "1 round", ["--rounds", "1"]),
    ("allver", "all_verified pool", ["--policy", "all_verified"]),
    ("firsttry", "no evidence re-prompt", ["--policy", "first_try_only"]),
    ("lr3e4", "rounds lr=3e-4", ["--lr_rsi", "0.0003"]),
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    os.chdir(ROOT)
    seq = "--seq" in sys.argv
    procs = []
    for short, label, extra in VARIANTS:
        out = os.path.join("runs", "real", "g4", f"s42_{short}")
        cmd = [sys.executable, "-u", "-m", "src.experiments", "--backend", "real",
               "--arm", "vgrs", "--grid", "4", "--seed", "42", "--out", out,
               "--tag", label] + extra
        log("RUN " + " ".join(cmd[2:]))
        if seq:
            t0 = time.time()
            rc = subprocess.run(cmd, cwd=ROOT).returncode
            log(f"{short} exit={rc} {time.time() - t0:.0f}s")
        else:
            fh = open(os.path.join(ROOT, "runs", f"ablation_{short}.log"), "wb", buffering=0)
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
            procs.append((p, short, fh, time.time()))
    for p, short, fh, t0 in procs:
        rc = p.wait(); fh.close()
        log(f"{short} exit={rc} elapsed={time.time() - t0:.0f}s")
    subprocess.run([sys.executable, "-u", "-m", "src.tables", "--backend", "real",
                    "--grid", "4"], cwd=ROOT)
    log("ABLATION DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())