"""Produce Fig. 6 (demo) for the report.

    python -u scripts/capture_demo.py --grid 4

Two paths:
  1. a real headless browser screenshot of the running demo (best);
  2. if no browser / not enough memory, a panel composed from the actual
     /predict responses (scene + verifier overlay + answer + evidence).
Both call the same demo_core.predict, so the second is the real output, not a mock-up.
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import threading
import time

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


def load_demo():
    spec = importlib.util.spec_from_file_location("demoapp", os.path.join(ROOT, "demo", "app.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def try_browser(url: str, out: str) -> bool:
    browser = next((p for p in CANDIDATES if os.path.exists(p)), "")
    if not browser:
        return False
    prof = tempfile.mkdtemp(prefix="edgeprof_")
    cmd = [browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
           "--window-size=1360,820", f"--user-data-dir={prof}",
           "--virtual-time-budget=8000", f"--screenshot={out}", url]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    except Exception as e:
        print(f"[capture] browser failed: {e}")
        return False
    return os.path.exists(out)


def compose_panel(rows, out: str, width=960):
    """rows = [(png_bytes, title, answer, evidence)]"""
    pad, ih, iw = 14, 256, 256
    th = 190
    h = pad + len(rows) * (max(ih, th) + pad)
    canvas = Image.new("RGB", (width, h), "white")
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
        small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 14)
    except Exception:
        font = small = ImageFont.load_default()
    y = pad
    for png, title, answer, ev in rows:
        im = Image.open(io.BytesIO(png)).convert("RGB").resize((iw, ih))
        canvas.paste(im, (pad, y))
        d.rectangle([pad, y, pad + iw, y + ih], outline=(210, 216, 224))
        tx = pad + iw + 18
        d.text((tx, y + 4), title, font=font, fill=(15, 63, 140))
        d.text((tx, y + 34), f"answer: {answer}", font=font, fill=(20, 20, 20))
        d.text((tx, y + 66), "verifier evidence:", font=small, fill=(90, 96, 105))
        d.multiline_text((tx, y + 88), ev, font=small, fill=(60, 66, 75), spacing=4)
        y += max(ih, th) + pad
    d.text((pad, h - 20), "VGRSI demo - real /predict output (analytic verifier, model-free mode)",
           font=small, fill=(120, 126, 135))
    canvas.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--out", default="paper/figs/fig6_demo.png")
    args = ap.parse_args()
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    mod = load_demo()
    cfg = mod.load_cfg(os.path.join(ROOT, "configs", "exp.yaml"))
    mod.STATE["model"] = mod.build_state(cfg, args.grid, "verifier", "")
    srv = mod.ThreadingHTTPServer(("127.0.0.1", 0), mod.Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.8)
    url = f"http://127.0.0.1:{port}/"

    if try_browser(url, out):
        srv.shutdown()
        print(f"[capture] browser screenshot -> {args.out} ({os.path.getsize(out)} B)")
        return

    rows = []
    for s in mod.STATE["samples"][:3]:
        q = next(x for x in s["questions"] if x["type"] == "loc")
        r = mod.predict(os.path.join(mod.STATE["meta"]["dir"], f"{s['sid']}.png"),
                        q["q"], args.grid)
        rows.append((base64.b64decode(r["overlay_png"]),
                     f"{s['sid']}  loc  gold={q['gold']}  "
                     f"accepted={len(r['accepted_cells'])}",
                     r["answer"], r["evidence"]))
    srv.shutdown()
    compose_panel(rows, out)
    print(f"[capture] panel composed from real /predict output -> {args.out} "
          f"({os.path.getsize(out)} B)")


if __name__ == "__main__":
    main()