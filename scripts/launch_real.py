"""Start scripts/run_real.py fully detached (survives the calling shell).

    python -u scripts/launch_real.py
    # then watch: Get-Content runs\\real_run.log -Tail 20 -Wait
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200


def main():
    log_path = os.path.join(ROOT, "runs", "real_run.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "ab", buffering=0)
    p = subprocess.Popen([sys.executable, "-u", os.path.join("scripts", "run_real.py")],
                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                         creationflags=0,
                         close_fds=True)
    print(f"[launch] pid={p.pid} -> {log_path}")


if __name__ == "__main__":
    main()