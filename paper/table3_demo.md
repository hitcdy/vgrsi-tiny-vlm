| Component | Technology | Function |
|---|---|---|
| Backend | Python stdlib http.server (FastAPI variant in demo/api_fastapi.py) | model inference + analytic-verifier REST API |
| Frontend | single-page HTML/JS, no build step | scene picker, question input, answer + overlay visualisation |
| Inference | SmolVLM-256M + LoRA adapter (optional) | answers the question; falls back to verifier-only mode when no adapter is present |
| Verifier | colour components + cell coverage (src/verifier.py) | independent judgement, structured evidence, overlay rendering |
| Deployment | localhost / container | python demo/app.py --port 8501 (uvicorn demo.api_fastapi:app as the FastAPI route) |