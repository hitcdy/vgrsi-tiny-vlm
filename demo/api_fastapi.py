"""FastAPI variant of the demo (same REST surface as demo/app.py).

    pip install fastapi uvicorn
    uvicorn demo.api_fastapi:app --port 8501
"""
from __future__ import annotations

import base64
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI            # noqa: E402
from fastapi.responses import HTMLResponse, Response  # noqa: E402
from pydantic import BaseModel         # noqa: E402

from src import data_gen, qa           # noqa: E402
from src.config import load as load_cfg  # noqa: E402
from src.demo_core import predict      # noqa: E402

CFG = load_cfg("configs/exp.yaml")
GRID = int(os.environ.get("VGRSI_GRID", 4))
META = data_gen.ensure_data(CFG["data"]["out_dir"], grids=(GRID,),
                            n_train=CFG["data"]["n_train"],
                            n_pool=CFG["data"]["n_pool"],
                            n_eval=CFG["data"]["n_eval"])[(GRID, "eval")]
QS = qa.build_questions(META, seed=42)

app = FastAPI(title="VGRSI demo")


class Req(BaseModel):
    img: str
    question: str


@app.get("/api/samples")
def samples():
    by = {}
    for q in QS:
        by.setdefault(q["img"], []).append({"type": q["type"], "q": q["q"], "gold": q["gold"]})
    return {"grid": GRID, "samples": [{"sid": k, "questions": v} for k, v in by.items()]}


@app.get("/img/{sid}")
def image(sid: str):
    p = os.path.join(META["dir"], f"{sid}.png")
    return Response(open(p, "rb").read(), media_type="image/png")


@app.post("/predict")
def predict_route(r: Req):
    p = os.path.join(META["dir"], f"{r.img}.png")
    return predict(p, r.question, GRID)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse("<h1>VGRSI demo</h1><p>POST /predict</p>")