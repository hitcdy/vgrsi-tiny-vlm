"""Self-contained web demo (stdlib only, no pip install, works offline).

    python demo/app.py --port 8501 --grid 4
    # browser: http://localhost:8501

Endpoints
    GET  /              interactive page
    GET  /api/samples   eval scenes + their generated questions
    GET  /img/<sid>     scene PNG
    POST /predict       {"img": "e003", "question": "..."} -> answer + verifier overlay

The FastAPI variant exposing the identical REST surface is demo/api_fastapi.py;
this file exists so the demonstration runs on a machine with no network at all.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_gen, qa          # noqa: E402
from src.config import load as load_cfg  # noqa: E402
from src.demo_core import predict    # noqa: E402
from src.utils import read_json      # noqa: E402

STATE = {"grid": 4, "meta": None, "samples": [], "model": None}

PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>VGRSI demo - verifier-gated grounding</title>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,"Microsoft YaHei",sans-serif;margin:0;background:#f6f7f9;color:#1b1f24}
header{background:#16233a;color:#fff;padding:14px 22px;font-size:17px;font-weight:600}
.wrap{display:flex;gap:18px;padding:18px 22px;flex-wrap:wrap}
.card{background:#fff;border:1px solid #e2e6ec;border-radius:10px;padding:14px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.left{flex:0 0 420px}.right{flex:1 1 420px;min-width:380px}
select,input,button{font-size:14px;padding:7px 9px;border:1px solid #cfd6df;border-radius:7px}
button{background:#2563eb;color:#fff;border:none;cursor:pointer}
button.ghost{background:#eef2f8;color:#1b3a6b}
img{max-width:100%;border-radius:8px;border:1px solid #e2e6ec}
.row{display:flex;gap:8px;margin:8px 0;flex-wrap:wrap}
.mono{font-family:Consolas,monospace;font-size:12.5px;background:#f4f6f9;border-radius:6px;padding:8px;white-space:pre-wrap}
.big{font-size:22px;font-weight:700;color:#0f3f8c}
.tag{display:inline-block;font-size:12px;background:#e8f0fe;color:#174ea6;border-radius:5px;padding:2px 7px;margin-left:6px}
</style></head><body>
<header>VGRSI demo &nbsp;·&nbsp; SmolVLM-256M + analytic verifier &nbsp;·&nbsp; grid <span id="hd">?</span></header>
<div class="wrap">
 <div class="card left">
  <div class="row"><label>Scene</label><select id="scene"></select>
   <button class="ghost" onclick="pick()">random</button></div>
  <img id="pic" src="">
  <div class="row" id="qbtns"></div>
  <div class="row"><input id="q" style="flex:1" value="图中的红色圆形位于第几行第几列？请用「第X行第Y列」回答（共4行4列）。">
   <button onclick="ask()">Ask</button></div>
  <div class="mono" id="meta"></div>
 </div>
 <div class="card right">
  <div>Answer</div><div class="big" id="ans">-</div>
  <div class="tag" id="src"></div><div class="tag" id="conf"></div>
  <div class="row" style="margin-top:12px"><img id="ov" src=""></div>
  <div class="mono" id="ev"></div>
 </div>
</div>
<script>
let S={};
fetch('/api/samples').then(r=>r.json()).then(d=>{
  S=d; document.getElementById('hd').textContent=d.grid+'x'+d.grid;
  const sel=document.getElementById('scene');
  d.samples.forEach(s=>{const o=document.createElement('option');o.value=s.sid;o.textContent=s.sid;sel.appendChild(o)});
  sel.onchange=()=>show(sel.value); show(d.samples[0].sid);
});
function show(sid){
  const s=S.samples.find(x=>x.sid===sid);
  document.getElementById('pic').src='/img/'+sid;
  document.getElementById('q').value=s.questions[0].q;
  const box=document.getElementById('qbtns'); box.innerHTML='';
  s.questions.forEach(q=>{const b=document.createElement('button');b.className='ghost';
    b.textContent=q.type; b.onclick=()=>{document.getElementById('q').value=q.q;ask()};box.appendChild(b)});
  document.getElementById('ans').textContent='-'; document.getElementById('ov').src='';
  document.getElementById('ev').textContent=''; document.getElementById('meta').textContent='gold: '+JSON.stringify(s.questions.map(q=>q.type+'='+q.gold));
}
function ask(){
  const sid=document.getElementById('scene').value, q=document.getElementById('q').value;
  fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({img:sid,question:q})})
   .then(r=>r.json()).then(d=>{
     document.getElementById('ans').textContent=d.answer;
     document.getElementById('src').textContent=d.source;
     document.getElementById('conf').textContent=d.confidence;
     document.getElementById('ov').src='data:image/png;base64,'+d.overlay_png;
     document.getElementById('ev').textContent=d.evidence+'\naccepted cells: '+JSON.stringify(d.accepted_cells)
        +(d.model_answer?'\nmodel says: '+d.model_answer:'');
   });
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if path == "/api/samples":
            return self._send(200, json.dumps({"grid": STATE["grid"],
                                               "samples": STATE["samples"]}))
        if path.startswith("/img/"):
            sid = path.split("/")[-1].replace(".png", "")
            p = os.path.join(STATE["meta"]["dir"], f"{sid}.png")
            if os.path.exists(p):
                return self._send(200, open(p, "rb").read(), "image/png")
            return self._send(404, "not found", "text/plain")
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/predict":
            return self._send(404, "not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        sid = req.get("img")
        p = os.path.join(STATE["meta"]["dir"], f"{sid}.png")
        res = predict(p, req.get("question", ""), STATE["grid"], STATE["model"])
        return self._send(200, json.dumps(res))


def build_state(cfg, grid, backend, adapter):
    metas = data_gen.ensure_data(cfg["data"]["out_dir"], grids=(grid,),
                                 n_train=cfg["data"]["n_train"],
                                 n_pool=cfg["data"]["n_pool"],
                                 n_eval=cfg["data"]["n_eval"])
    meta = metas[(grid, "eval")]
    qs = qa.build_questions(meta, seed=42)
    by_img = {}
    for q in qs:
        by_img.setdefault(q["img"], []).append({"type": q["type"], "q": q["q"], "gold": q["gold"]})
    STATE.update({"grid": grid, "meta": meta,
                  "samples": [{"sid": k, "questions": v} for k, v in list(by_img.items())]})
    model = None
    if backend in ("real", "auto") and adapter and os.path.exists(adapter):
        try:
            from src.vlm import VLM

            model = VLM(lora_r=cfg["train"]["lora_r"], image_splitting=False).load(adapter)
            print(f"[demo] model + adapter loaded: {adapter}")
        except Exception as e:
            print(f"[demo] model unavailable ({e}); verifier-only mode")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--backend", default="auto", choices=["auto", "verifier", "real"])
    ap.add_argument("--adapter", default="runs/real/g4/s42/adapter_vgrs")
    ap.add_argument("--config", default="configs/exp.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    STATE["model"] = build_state(cfg, args.grid, args.backend, args.adapter)
    print(f"[demo] http://localhost:{args.port}  (grid {args.grid}, "
          f"{'model+verifier' if STATE['model'] else 'verifier-only'})")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()