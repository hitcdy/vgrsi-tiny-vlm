# VGRSI — Verifier-Gated Recursive Self-Improvement for Sub-Billion-Parameter VLMs

Code for the final report *"Verifier-Gated Recursive Self-Improvement for
Sub-Billion-Parameter Vision-Language Models: A Compute-Matched Study on Spatial
Grounding"* (HITSZ, 大模型基础与应用探索, 2026 秋).

The claim under test: a **256 M-parameter VLM** can acquire absolute spatial
grounding **without human labels**, if an *analytic, non-neural verifier* is put
in charge of deciding which self-generated samples may enter the training pool.
Everything runs on a laptop — RTX 4060 8 GB, or CPU only.

## 0. Results at a glance (grid 4x4, 2 seeds, 150 questions/seed)

| arm | Acc strict | Acc lenient | loc | count | exist |
|---|---|---|---|---|---|
| A zero-shot | 6.3 +- 0.5 | 10.3 | 0.0 | 14.0 | 5.0 |
| B LoRA SFT | 18.0 +- 6.6 | 27.7 | 0.0 | 30.0 | 24.0 |
| C inference-time verifier | **29.6 +- 9.0** | 50.6 | 0.0 (39.0 lenient) | 29.0 | 21.0 |
| D seed-only continuation (control) | 7.7 +- 5.2 | 25.7 | 0.0 | 3.0 | 20.0 |
| E VGRSI self-training | 18.3 +- 12.7 | 33.0 | 0.0 | 21.0 | 34.0 |

Grid 8x8 (1 seed): A 4.7 / B 3.3 / C 4.0 / D 0.0 / **E 18.0**: self-training only pays where
supervised fitting fails. Strict localisation is 0 % everywhere once the evaluation leak of the
pilot study is removed, while lenient parsing still credits arm C with 39 % - the parser
false-positive counter is the point, not a detail.

Regenerate everything: `python -u -m src.tables --backend real --grid 4` and
`python -u paper/build_docx.py` (needs `paper/report.md`).

Extra scripts: `paper/make_fig1.py` (Fig. 1 loop diagram), `paper/build_docx.py` (report ->
submission .docx), `scripts/run_ablation.py` (Table 2 variants),
`scripts/capture_demo.py` (Fig. 6), `scripts/fetch_model.py` (weights, mirror-friendly).


---

## 1. What is in the box

```
project/
├── configs/exp.yaml          # every hyper-parameter and budget (single source of truth)
├── data/                     # generated: scenes + gt.json (train / pool / eval splits)
├── src/
│   ├── data_gen.py           # parameterised grounding benchmark (grid 4x4 and 8x8)
│   ├── codec.py              # cell <-> pixel codec + strict/lenient answer parser
│   ├── verifier.py           # analytic verifier: colour components + coverage (the reward source)
│   ├── qa.py                 # loc / count / exist question construction
│   ├── metrics.py            # judging, lenient vs strict accuracy, parser-false-positive count
│   ├── vlm.py                # SmolVLM-256M + LoRA + frozen vision-tower cache (real backend)
│   ├── mock.py               # deterministic stand-in used to smoke-test with no weights
│   ├── train.py-free         # SFT lives in vlm.VLM.sft (labels masked to answer tokens)
│   ├── evaluate.py           # inference, optional verifier-in-the-loop correction
│   ├── experiments.py        # the five arms A-E
│   ├── verifier_audit.py     # Table 4: verifier reliability, no weights required
│   ├── tables.py             # Table 1 / Table 2 (markdown + LaTeX + numbers.json)
│   ├── figures.py            # Fig. 2-6
│   └── demo_core.py          # shared inference glue for the two demo servers
├── demo/app.py               # zero-dependency web demo (stdlib http.server)
├── demo/api_fastapi.py       # identical REST surface on FastAPI
├── paper/                    # tables + figures + numbers.json (generated)
└── scripts/run_all.{sh,ps1}  # one-command reproduction
```

## 2. Quick start

```powershell
# 0) install (CPU is enough; a CUDA build only makes it faster)
pip install -r requirements.txt

# 1) full study, real backbone
$env:HF_ENDPOINT = "https://hf-mirror.com"   # only if huggingface.co is unreachable
.\scripts\run_all.ps1            # or: bash scripts/run_all.sh

# 2) smoke test with NO weights and NO network (writes runs_mock/, watermarked)
.\scripts\run_all.ps1 mock

# 3) demo
python demo/app.py --port 8501 --grid 4
#   open http://localhost:8501  → pick a scene, press loc/count/exist, read the answer
```

Runs without any model weights (seconds each, useful on a fresh machine):

```powershell
python -m src.verifier_audit                    # Table 4
python -m src.experiments --backend mock --arm all --grid 4
python -m src.tables  --backend mock --grid 4
python -m src.figures --backend mock --grid 4
```

## 3. The five arms

| arm | id | what it does | what it isolates |
|---|---|---|---|
| A | `zero_shot` | no adaptation | the backbone's prior |
| B | `sft` | LoRA SFT, 60 steps, labelled seed split | supervised adaptation |
| C | `sft_infer_verifier` | B + verifier evidence re-prompt, no re-training | inference-time correction |
| D | `seed_only_cont` | B + 15 more steps on the *same* seed data | "is it just more compute?" |
| E | `vgrs` | B + 2 rounds of verifier-gated self-training | ours |

Arm E trains on a **separate unlabelled pool split (seed 777)**; the labels come
only from the verifier. The 150-question evaluation split (seed 42) is never
trained on. This is the leak that the lab06 pilot had (it admitted corrected
samples of the very 15 questions it then reported on) and it is fixed here.

## 4. Evaluation protocol (why two accuracies)

`metrics.judge` returns both:

* `acc_lenient` — first regex match wins (what a careless eval script reports)
* `acc_strict` — exactly one row/column mention, no question echo
* `parser_fp` — `ok_lenient and not ok_strict`, i.e. an **evaluation artefact**

A tiny VLM typically degenerates to an enumerated scan (`第1行第2列第3列第4列…`)
or echoes the question. A lenient parser then scores accidental hits as successes;
`acc_strict` does not. Every table reports both, and Table 4 quantifies how much
of the lenient number is fake.

## 5. Data

Synthetic scenes, three disjoint splits, all deterministic:

| split | seed | images | used for |
|---|---|---|---|
| train | 2026 | 60 | labelled seed set (180 QA) |
| pool | 777 | 40 | self-training prompts (labels = verifier only) |
| eval | 42 | 50 | 150 held-out questions (loc / count / exist) |

`grid=4` uses 34–54 px objects on a 256 px canvas, `grid=8` shrinks them to
16–28 px so that per-cell evidence stays comparable. Scene generation guarantees
(1) at least one *singleton* class per scene, so loc questions are well-posed;
(2) a 2–3 instance class in most scenes, so count questions are not saturated at
"1"; (3) half of the exist questions are negative (absent colour).

## 6. Reproducibility claims for Section 5 of the report

* all RNGs seeded from `configs/exp.yaml:seed` (`src/utils.seed_all`)
* vision tower frozen + feature cache keyed on the md5 of `pixel_values`, so a
  re-run of the same image is a dict lookup
* `run_all.ps1` regenerates every table/figure from scratch
* cost reference (measure on your machine and quote it): one `--arm all` pass at
  `grid=4` is ~6 VLM generations per question plus 60–180 SFT steps; on CPU with
  4 threads the pilot equivalent took ~9 min per full pipeline.

## 7. Mock backend: read this before pasting numbers

`src/mock.py` is a *fake* model with hand-set answer distributions. It exists to
prove the pipeline runs end-to-end on a machine with no weights. Its runs land in
`runs_mock/`, its tables start with a `MOCK … NOT FOR SUBMISSION` line, and its
figures are watermarked. Never copy those numbers into the report.