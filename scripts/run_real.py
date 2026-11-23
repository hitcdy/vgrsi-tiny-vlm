"""Driver for the REAL backbone runs.

Three runs, executed **in parallel** (8 torch threads each on a 32-core box):
    (grid 4, seed 42)  main result           (grid 8, seed 42)  hardness (RQ3)
    (grid 4, seed 7)   second seed for mean+-std

    python -u scripts/run_real.py            # parallel (default)
    python -u scripts/run_real.py --seq      # one at a time
    python -u scripts/launch_real.py         # detached wrapper

Each run is its own process with its own log (runs/real_g{G}_s{S}.log), so one
failure cannot poison the others; experiments.py writes results.json after every
arm, so a re-run only redoes what is missing.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [(4, 42), (8, 42), (4, 7)]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cmd_for(grid: int, seed: int):
    return [sys.executable, "-u", "-m", "src.experiments", "--backend", "real",
            "--arm", "all", "--grid", str(grid), "--seed", str(seed)]


def main() -> int:
    os.chdir(ROOT)
    seq = "--seq" in sys.argv
    t_all = time.time()

    if seq:
        for grid, seed in RUNS:
            log(f"RUN grid={grid} seed={seed}")
            t0 = time.time()
            rc = subprocess.run(cmd_for(grid, seed), cwd=ROOT).returncode
            log(f"grid={grid} seed={seed} exit={rc} elapsed={time.time() - t0:.0f}s")
    else:
        procs = []
        for grid, seed in RUNS:
            path = os.path.join(ROOT, "runs", f"real_g{grid}_s{seed}.log")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fh = open(path, "wb", buffering=0)
            p = subprocess.Popen(cmd_for(grid, seed), cwd=ROOT, stdout=fh,
                                 stderr=subprocess.STDOUT)
            log(f"START grid={grid} seed={seed} pid={p.pid} -> {os.path.basename(path)}")
            procs.append((p, grid, seed, fh, time.time()))
        for p, grid, seed, fh, t0 in procs:
            rc = p.wait()
            fh.close()
            log(f"grid={grid} seed={seed} exit={rc} elapsed={time.time() - t0:.0f}s")

    for grid in (4, 8):
        subprocess.run([sys.executable, "-u", "-m", "src.tables", "--backend", "real",
                        "--grid", str(grid)], cwd=ROOT)
        subprocess.run([sys.executable, "-u", "-m", "src.figures", "--backend", "real",
                        "--grid", str(grid)], cwd=ROOT)
    log(f"ALL DONE total={time.time() - t_all:.0f}s")
    with open(os.path.join(ROOT, "runs", "real_done.txt"), "w", encoding="utf-8") as f:
        f.write(f"ok {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())