"""Smoke test for the web demo: boots the stdlib server in-process, hammers the
REST surface, and writes paper/figs/demo_screenshot_data.json.

    python -u scripts/smoke_demo.py --grid 4
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import sys
import threading
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_demo():
    spec = importlib.util.spec_from_file_location("demoapp", os.path.join(ROOT, "demo", "app.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=4)
    args = ap.parse_args()
    mod = load_demo()
    cfg = mod.load_cfg(os.path.join(ROOT, "configs", "exp.yaml"))
    mod.STATE["model"] = mod.build_state(cfg, args.grid, "verifier", "")

    srv = mod.ThreadingHTTPServer(("127.0.0.1", 0), mod.Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    page = urllib.request.urlopen(base + "/", timeout=10).read().decode()
    print(f"[demo] GET /            -> {len(page)} bytes, title ok: {'VGRSI demo' in page}")

    samples = json.loads(urllib.request.urlopen(base + "/api/samples", timeout=10).read())
    print(f"[demo] GET /api/samples -> {len(samples['samples'])} scenes, grid {samples['grid']}")

    sid = samples["samples"][0]["sid"]
    img = urllib.request.urlopen(f"{base}/img/{sid}", timeout=10).read()
    print(f"[demo] GET /img/{sid}   -> {len(img)} bytes PNG: {img[:4] == b'\x89PNG'}")

    out = []
    for q in samples["samples"][0]["questions"]:
        body = json.dumps({"img": sid, "question": q["q"]}).encode()
        req = urllib.request.Request(base + "/predict", data=body,
                                     headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=20).read())
        assert base64.b64decode(d["overlay_png"])[:4] == b"\x89PNG"
        out.append({"q": q["q"], "gold": q["gold"], "answer": d["answer"],
                    "evidence": d["evidence"], "accepted": d["accepted_cells"]})
        print(f"[demo] POST /predict {q['type']:5s} gold={q['gold']:8s} -> {d['answer']}  "
              f"(accepted cells: {len(d['accepted_cells'])})")
    os.makedirs(os.path.join(ROOT, "paper"), exist_ok=True)
    with open(os.path.join(ROOT, "paper", "demo_trace.json"), "w", encoding="utf-8") as f:
        json.dump({"scene": sid, "calls": out}, f, ensure_ascii=False, indent=1)
    srv.shutdown()
    print("[demo] smoke test passed -> paper/demo_trace.json")


if __name__ == "__main__":
    main()