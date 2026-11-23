# Verifier-Gated Recursive Self-Improvement for Sub-Billion-Parameter Vision-Language Models: A Compute-Matched Study on Spatial Grounding

Chen Daoyan, Kuo-Kun Tseng
Harbin Institute of Technology, Shenzhen

**Keywords:** verifier-gated self-improvement; small vision-language model; LoRA adaptation; rejection sampling; spatial grounding

> Status tags used in this draft: **[REAL]** = measured in this repository, reproducible now without weights;
> **[TBD]** = requires the backbone run (`--backend real`), to be filled from `paper/numbers.json`.
> Never paste a `mock` number: `paper/table1_main.md` from the mock backend carries a warning line.

---

## Abstract (200–300 words, 4 moves)

Aiming at the technical bottleneck of **absolute spatial grounding in sub-billion-parameter
vision-language models (VLMs)** — where accuracy stays near chance even after supervised
fine-tuning, because the missing ingredient is not capacity but a trustworthy supervision
signal — this paper designs **VGRSI**, a verifier-gated recursive self-improvement loop built
on the Multimodal, Agent and Reinforcement Intelligence modules of the course. An analytic,
non-neural verifier (colour masking + 8-neighbour connected components) replaces human
annotation: each round the model answers an *unlabelled* scenario pool, receives structured
verifier evidence, and only samples whose verdict flips from wrong to correct are admitted
into the training pool, which is then used to re-train a LoRA adapter (0.53 % of parameters).
**[REAL]** A model-free audit of 300 questions shows the verifier recovers the ground-truth
cell in 100 % of localisation queries, yet accepts 1.08–1.10 *extra* cells per query
(62–76 % of queries are ambiguous), so a random guesser already passes 13.0 % of its verdicts
at 4×4 and 3.3 % at 8×8 — which is why every metric here is reported twice, leniently and
strictly. **[TBD]** On the held-out 150-question benchmark, VGRSI raises strict accuracy from
⟨A⟩ % (zero-shot) to ⟨E⟩ % (+⟨Δ⟩ points), whereas a compute-matched seed-only continuation
*degrades* accuracy from ⟨B⟩ % to ⟨D⟩ %, showing that the gain comes from verifier-gated data
selection rather than additional optimisation. A Gradio-free web demo and a one-command
reproduction script are released; the full study runs on a single 8 GB laptop GPU or on CPU.

---

## 1 Introduction

### 1.1 Research background & problem statement
Eight course module systems (Structural → Visual → Language → Generative → Graph → Multimodal
→ Reinforcement → Agent) converge on one practical question for on-device deployment: *how
small can a multimodal model be while still answering spatial questions reliably?*
SmolVLM-256M — 256.5 M parameters, 1.36 M trainable under LoRA — is a realistic deployment
target, but grid localisation ("which row and column holds the red circle?") is close to
chance, and our pilot study (lab06) showed that plain LoRA SFT saturates early while the model
degenerates into an *enumerated scan* (`第1行第2列第3列第4列…`) or echoes the question.

The bottleneck is supervision, not capacity: labelling cells is cheap per image but the
question only becomes interesting when labels are *free*. This motivates RQ1–RQ3:

* **RQ1** Can an analytic verifier replace human labels for this task, and how reliable is it?
* **RQ2** At equal wall-clock, does the self-improvement gain come from verifier-gated *data
  selection* or merely from *more optimiser steps*?
* **RQ3** Does the loop survive a harder grid (4×4 → 8×8)?

### 1.2 Contributions (each bound to a course module)
1. **Method (Multimodal + Agent).** VGRSI: a closed loop of *sample → answer → verify → gate →
   re-train* where only verifier-confirmed flips enter the pool. The pool is a **separate
   unlabelled split**, so the evaluation questions are never trained on — this fixes the
   label-leak of our pilot, which admitted corrected samples of the very 15 questions it
   reported on.
2. **Evidence (Reinforcement).** A compute-matched five-arm comparison (A zero-shot, B SFT,
   C inference-time verifier, D seed-only continuation, E VGRSI) that isolates instant
   correction from data selection from raw optimisation steps.
3. **Diagnosis (Language + Structural).** A dual-metric protocol: `acc_lenient` (first regex
   match) versus `acc_strict` (single mention, no question echo), plus a `parser_fp` count;
   together with [REAL] verifier reliability statistics these show why naive accuracy inflates
   tiny-VLM capability.
4. Reusable codebase + zero-dependency web demo (Sections 5–6).

### 1.3 Organisation
Sec. 2 related work · Sec. 3 method · Sec. 4 experiments · Sec. 5 reproducibility · Sec. 6 demo ·
Sec. 7 discussion · Sec. 8 conclusion.

---

## 2 Related Work

**2.1 Course fundamentals.** Structural (MLP/CNN/RNN), Visual (detection, segmentation),
Language (Transformer, LLM), Generative (VAE/GAN/diffusion), Graph (GCN/GAT), Multimodal,
Reinforcement (DQN/PPO) and Agent Intelligence — this work sits at the junction of Multimodal
(grounding), Agent (self-improvement loop) and Reinforcement (a verifier as reward source).

**2.2 SOTA.** Test-time / inference-time scaling, RLVR-style training with programmatic
verifiers, rejection-sampling fine-tuning and self-training/STaR families; small-VLM and
on-device multimodal lines. *Difference:* existing RLVR work assumes a verifier that is both
high-recall **and** injective; we show a *cheap analytic* verifier is high-recall but strongly
non-injective, and we quantify the consequence instead of assuming it away.

**2.3 Relation to the course.** The pilot labs (MLP/CONV-LSTM/mini-LLM/RAG/SmolVLM-QLoRA+DPO/
RSI localisation/distillation) supplied both the infrastructure and the failure modes that this
paper formalises.

---

## 3 Methodology

**3.1 Problem definition.** Grid *g*×*g* on a 256×256 canvas; a scene *S* = {(class_i, box_i)}.
For a target class *c* occurring once, the loc answer is the cell of the box centre,
*y = (⌊cy/h⌋, ⌊cx/h⌋)* with *h = 256/g*. Metrics: exact-cell accuracy, mean IoU of the predicted
cell with the instance box, and — for count/exist — exact match.

**3.2 Verifier (the reward source).** For colour *k*: mask *M_k = ‖I − ρ_k‖² < τ*; 8-neighbour
components *C_k*; a candidate cell is accepted if some component centroid lies inside it or the
cell covers ≥50 % of its area. Structured evidence (component count, area, centroid cell) is
fed back as text. [REAL] Reliability (Table 4): recall 100 %, but 1.08 (4×4) / 1.10 (8×8) extra
accepted cells → 62 % / 76 % of queries ambiguous.

**3.3 VGRSI loop.** For round *r*: (i) answer every pool prompt; (ii) verify; (iii) if rejected,
re-prompt with evidence and verify again; (iv) admit **only** flips (policy `flip_only`; the
`all_verified` policy is the Table 2 ablation); (v) re-train the LoRA adapter on pool ∪ admitted;
(vi) evaluate on the untouched eval split.

**3.4 Implementation details.** SmolVLM-256M-Instruct, LoRA r=8, α=16, dropout 0.05 on
q/k/v/o_proj, AdamW lr 1e-3, 60 steps per stage, batch 1, answer-only labels, greedy decoding,
24 new tokens, frozen vision tower with an md5-keyed feature cache, seed 42 everywhere.
Platform: Windows 11, Python 3.14, torch 2.14 (CPU wheel), 4 threads; a CUDA build only changes
latency. Full hyper-parameters: `configs/exp.yaml`.

---

## 4 Experiments and Results

**4.1 Setup.** Self-built benchmark, three disjoint splits (train seed 2026 / pool seed 777 /
eval seed 42), 50 eval images → 150 questions (loc 50, count 50, exist 50 with half negative).
Baselines are the course's own model families (MLP-class prior, LoRA SFT, verifier variants).

**4.2 Main results.** Table 1 (`paper/table1_main.md`, auto-generated). [TBD] Fill from
`python -m src.tables --backend real --grid 4`; the delta sentences for the abstract are in
`paper/numbers.json` (`abs_gain_pts`, `rel_gain_pct`, `vgrs_vs_sft_rel_pct`).

**4.3 Ablation.** Table 2: pool policy (`flip_only` vs `all_verified`), rounds (1 vs 2),
correction path (self-check only vs verifier evidence), and the arm-D compute control.
Expected reading: the seed-only continuation must not improve — otherwise "more steps" is the
explanation, not the gate.

**4.4 Verifier audit (Table 4, [REAL]).** `paper/table4_verifier.md` — the model-free numbers
quoted in the abstract. Interpretation: the verifier is a *recall-complete but non-injective*
judge; colour-only evidence caps class-level existence supervision at 84 % (4×4) / 82 % (8×8),
which explains why, in the pilot, existence questions were the only reliably corrected type.

**4.5 Visualisation.** Fig. 2 per-type accuracy; Fig. 3 loss curves; Fig. 4 pool growth;
Fig. 5 accepted-cell ambiguity (`mock_fig5_verifier.png` is watermarked only because it was
rendered by the mock runner — the underlying scenes and verdicts are real); Fig. 6 demo output.

---

## 5 Reproducibility and reusable code

Repository layout and one-command reproduction are in `README.md`; `scripts/run_all.ps1`
(and `.sh`) regenerate Tables 1–2 and Figs. 2–6 from scratch. All RNGs are seeded from
`configs/exp.yaml:seed` (Python/NumPy/torch), the vision tower is frozen and cached, and
`python -m src.verifier_audit` reproduces Table 4 in ~2 s with **no weights and no network**.
Cost to quote: ⟨TBD⟩ GPU-hours on one RTX 4060 Laptop 8 GB (measured wall-clock is stored per
arm in `runs/real/g4/s42/results.json`).

---

## 6 System demonstration

`python demo/app.py --port 8501 --grid 4` (stdlib only, offline) exposes
`GET /` · `GET /api/samples` · `GET /img/<sid>` · `POST /predict`. The FastAPI variant with the
identical surface is `demo/api_fastapi.py` (Table 3). With no adapter on disk the demo answers
from the verifier alone — a legitimate cold-start mode — and Fig. 6 shows three real
`/predict` responses including one with two accepted cells. The smoke test
`scripts/smoke_demo.py` asserts the full REST surface and writes `paper/demo_trace.json`.

---

## 7 Discussion

**7.1 Why the gate matters.** A non-injective verifier admits ambiguous samples; without a gate
the pool fills with "correct by accident" answers, which teaches the enumerated-scan prior.
**7.2 Advantages / limitations.** Advantages: no human labels, millisecond reward, single-GPU
reproduction. Limitations: colour-only evidence cannot resolve shape-level existence; cross-cell
acceptance makes the verifier a weak judge for border-straddling objects; 150 questions is small.
**7.3 Course reflection / 7.4 Future work.** Verifier v2 with shape templates; longer pools;
distillation of the loop into a single forward pass.

## 8 Conclusion
⟨TBD after real runs; ≈200 words, no new numbers.⟩

## References (GB/T 7714)
[1] Vaswani A, et al. Attention is all you need[C]//NeurIPS, 2017.
[2] HuggingFaceTB. SmolVLM-256M-Instruct[EB/OL]. 2025.
[3] Course Teaching Group. 大模型基础与应用探索课程讲义[Z]. HITSZ, 2026.
[4] Zelikman E, et al. STaR: Bootstrapping reasoning with reasoning[C]//NeurIPS, 2022.
[5] Yuan Z, et al. Self-rewarding language models[C]//ICML, 2024.