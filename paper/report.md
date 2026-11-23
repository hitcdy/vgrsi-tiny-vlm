# Verifier-Gated Recursive Self-Improvement for Sub-Billion-Parameter Vision-Language Models: When Does It Help?
Chen Daoyan (Student ID 2024312149) — Instructor: Kuo-Kun Tseng
Harbin Institute of Technology, Shenzhen — 大模型基础与应用探索 · Final Report
**Keywords:** verifier-gated self-improvement; small vision-language model; LoRA adaptation; rejection sampling; spatial grounding
---
## Abstract
Sub-billion-parameter vision-language models (VLMs) are attractive for on-device deployment,
yet their spatial grounding is close to chance and the labels needed to fix it are expensive.
This paper asks when an analytic, non-neural verifier can replace human annotation. We design
VGRSI, a loop in which the model answers an unlabelled scenario pool, receives structured
verifier evidence, and only samples whose verdict flips from wrong to correct enter the
training pool before a LoRA adapter (0.53 % of parameters) is re-trained. The loop is evaluated
with a leak-free protocol: the self-training pool and the 150-question benchmark come from
disjoint splits, unlike our pilot study, which admitted corrected samples of the very questions
it reported on. A model-free audit shows the verifier recovers the ground-truth cell in 100 %
of localisation queries but accepts 1.08-1.10 extra cells per query, so 62-76 % of queries are
ambiguous; every metric is therefore reported both leniently and strictly. Over two seeds and
two grid resolutions we find that (i) once the leak is removed, strict localisation accuracy is
0 % in every arm - the pilot's apparent grounding gain was an evaluation artefact, and lenient
parsing still credits one arm with 39 %; (ii) inference-time verification is the only reliable
lever (+11.6 points over LoRA SFT, 18.0 -> 29.6 %); (iii) retraining-based self-training is
unstable at low difficulty (18.3 +- 12.7 %) but decisive where supervised fitting fails
(8x8 grid: 3.3 -> 18.0 %); and (iv) a compute-matched control degrades accuracy (18.0 -> 7.7 %),
so the gains are not bought with extra steps. The complete study takes 1.9 hours of wall-clock across
three runs on eight CPU threads (about 45 minutes when run in parallel), with no GPU, and a
zero-dependency web demo is released.
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
**3.3 VGRSI loop.** For round *r*: (i) answer every pool prompt; (ii) verify the answer;
(iii) if rejected, re-prompt with structured evidence and verify again; (iv) admit **only**
wrong-to-right flips (policy `flip_only`; `all_verified` and `first_try_only` are the Table 2
ablation); (v) re-train the LoRA adapter on pool + admitted; (vi) evaluate on the untouched
evaluation split. The pool comes from a split that is disjoint from the evaluation split, which
is exactly what the pilot study got wrong. Fig. 1 is the block diagram and Algorithm 1 the
reference implementation; both map one-to-one onto `src/experiments.py:self_training_round`.

{{FIG1}}

```
Algorithm 1  Verifier-gated recursive self-improvement (VGRSI)
Input : labelled seed set S (60 scenes), unlabelled pool P (40 scenes),
        held-out evaluation set E (50 scenes, 150 questions),
        rounds R = 2, steps K = 60, SFT learning rate eta, round learning rate eta_r
Output: adapter theta_R, pool-growth curve, per-round metrics

 1  theta <- LoRA-SFT(theta_0, S, K, eta)                    # arm B
 2  pool  <- S ;  assert P intersect E = empty               # no evaluation leak
 3  for r = 1 .. R do
 4      admitted <- empty
 5      for each prompt q in questions(P) do
 6          a <- Decode(theta, q)                             # greedy, 24 new tokens
 7          ok, label <- Verify(img(q), a)                    # analytic, no network
 8          if not ok then
 9              hint <- Evidence(img(q))                      # "N comps; block 1 at row r col c"
10              a2 <- Decode(theta, q, hint)
11              ok2, label2 <- Verify(img(q), a2)
12              if ok2 then admitted <- admitted + {(q, label2)}    # flip only
13          else if policy = all_verified then admitted <- admitted + {(q, label)}
14      pool <- pool + admitted                              # 180 -> 200-218 samples
15      theta <- SFT(theta, pool, K, eta_r)
16      report Acc_strict(theta, E), Acc_lenient(theta, E)
17  end for
```

Lines 2 and 12 carry the two claims of the paper: the pool never touches the evaluation split,
and the gate - not the raw amount of self-generated data - decides what is learned.

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
**4.2 Main results.** Table 1 reports the five arms on the held-out 150-question benchmark
(grid 4x4, mean +- s.d. over seeds 42 and 7). Three observations dominate.
First, strict localisation accuracy is **0 % in every arm** (0/50 questions, both seeds),
although the lenient parser credits arm C with 39.0 % of localisation questions: the entire
localisation "signal" of the pilot study is a parsing artefact once the evaluation leak is
removed. Second, the only arm that improves reliably over supervised fine-tuning is the
inference-time verifier (C): 18.0 -> 29.6 % strict (+11.6 points, +64 % relative), at the cost
of 7.9 s versus 3.3 s per question because rejected answers are re-asked with evidence. Third,
retraining-based self-training (E) is a coin flip at this difficulty: 27.3 % on seed 42 and
9.3 % on seed 7 (18.3 +- 12.7), i.e. indistinguishable from arm B on average while being far
less stable.
{{TABLE1}}
The latency column needs a caveat: it is the mean per-question time of the whole arm, and it
includes the one-off vision-tower encoding the first time a scene is seen (2.1 s on 8 CPU
threads); later questions on the same scene drop to about 1 s. Arm C is 2.4x slower than arm B
because roughly four out of five rejected answers are re-asked with evidence, which is the
actual price of inference-time verification.

The compute control is what rules out the trivial explanation: arm D spends 15 further
optimiser steps on exactly the same seed data as arm B and *loses* 10 points (18.0 -> 7.7 %,
and 3.3 -> 0.0 % at 8x8). Additional optimisation on a 60-sample seed set destroys the format
it just learned, so no part of arm E's behaviour can be attributed to extra compute.
**4.3 Ablation.** Table 2 collects the ablations. All variants except the two-round main runs
see exactly the same compute (SFT plus one self-training round), so the columns are comparable.

Three readings. (i) *The correction path is not optional.* With the evidence re-prompt disabled
(`first_try_only`), the verifier never observes a wrong-to-right flip, the pool stays at its 180
seed samples, and the extra round collapses accuracy to 3.3 % - the same collapse the compute
control shows. The gain therefore comes from the loop's second half (evidence -> rewrite ->
re-verify), not from the verifier as a passive filter. (ii) *The gate is cheap insurance rather
than the engine.* Admitting every verifier-correct sample (`all_verified`, pool 236) gives
20.0 %, inside the run-to-run noise: the same `flip_only` configuration re-run from scratch
yields 17.3 % and 10.0 %, i.e. about seven points of noise on a 150-question benchmark. The gate
matters for label purity, yet at this data scale it is not what moves the metric. (iii) *Lowering
the round learning rate does not fix the instability*: with 3e-4 the first round reaches only
4.7 %, so the model moves less and learns less; the two-round 3e-4 variant is left untested and
is called out in Section 7.2.

The round axis is the loudest result in the table: round 1 lifts both seeds to 16.7-17.3 % and
round 2 then diverges (27.3 % on seed 42, 9.3 % on seed 7). At 1e-3 with 60 steps per round the
second round is not a refinement but a fresh random walk on an ever-growing pool: the gate
controls *which* samples enter, not *how far* the adapter moves. On the harder 8x8 grid, where
supervised fitting fails outright (B = 3.3 %, D = 0.0 %), the same loop is the only arm that
works (18.0 %), so self-training pays exactly in the regime where the seed supervision cannot fit
the task.

{{TABLE2}}

**4.4 Verifier audit (Table 4).** Before trusting the verifier as a label source we
audited it against ground truth on both grids with no model in the loop. It accepts the correct
cell in 100 % of localisation queries, but also accepts 1.08-1.10 extra cells per query, so
62-76 % of queries admit more than one "valid" answer and a uniform random guesser already
passes 13.0 % (4x4) / 3.28 % (8x8) of its verdicts. Colour-only evidence also caps class-level
existence supervision at 84 % / 82 %, because a red square also satisfies "is there a red
circle?".
{{TABLE4}}
**4.5 Visualisation.** Fig. 2 breaks accuracy down by question type, Fig. 3 shows the
training loss of SFT and of each VGRSI round, Fig. 4 how the gated pool grows, Fig. 5 the
accepted-cell ambiguity quantified in Table 4, and Fig. 6 three real `/predict` responses of the
demonstration system.
{{FIG2}}
{{FIG3}}
{{FIG4}}
{{FIG5}}
{{FIG6}}
---
## 5 Reproducibility and reusable code
Repository layout and one-command reproduction are in `README.md`; `scripts/run_all.ps1`
(and `.sh`) regenerate Tables 1–2 and Figs. 2–6 from scratch. All RNGs are seeded from
`configs/exp.yaml:seed` (Python/NumPy/torch), the vision tower is frozen and cached, and
`python -m src.verifier_audit` reproduces Table 4 in ~2 s with **no weights and no network**.
Measured cost: one full five-arm run at one grid/seed takes 2275 s of wall-clock
on eight CPU threads (240 s of optimisation, 851 s for the verifier-in-the-loop
arm); the whole paper - three runs plus the model-free audit - is 1.9 hours of wall-clock on eight CPU threads (about 45 minutes in parallel)
on a laptop CPU, no GPU required.
---
## 6 System demonstration
`python demo/app.py --port 8501 --grid 4` (stdlib only, offline) exposes
`GET /` · `GET /api/samples` · `GET /img/<sid>` · `POST /predict`. The FastAPI variant with the
identical surface is `demo/api_fastapi.py` (Table 3). With no adapter on disk the demo answers
from the verifier alone — a legitimate cold-start mode — and Fig. 6 shows three real
`/predict` responses including one with two accepted cells. The smoke test
`scripts/smoke_demo.py` asserts the full REST surface and writes `paper/demo_trace.json`.
{{TABLE3}}
---
## 7 Discussion
**7.1 Why the gate matters - and why it is not enough.** A non-injective verifier admits
ambiguous samples; the `flip_only` gate filters them, and the audit shows why that filter is
necessary: 62-76 % of localisation queries accept more than one cell. But the gate is necessary,
not sufficient: it fixes the *label noise* while leaving the *optimisation* uncontrolled, and
Table 2 shows the second round drifting either way. Where the verifier's evidence is decisive
(existence: 24 % -> 34 % on 4x4, 0 % -> 42 % on 8x8) the loop transfers; where it is not
(localisation, which needs deictic reasoning and answer formatting rather than colour evidence)
no arm moves at all.
**7.2 Advantages and limitations.** Advantages: no human labels, millisecond reward, a
leak-free reproducible protocol and a total cost of 1.9 wall-clock hours on eight CPU threads. Limitations, stated plainly:
150 questions is a small benchmark; strict localisation is 0 %, so this is a study of existence
and counting, not of grounding; the verifier is colour-only, which caps class-level existence
supervision at 84 %; the second round is unstable and we probed only one alternative round
learning rate (3e-4, one round only), so part of that instability may still be a tuning
artefact; and identical configurations re-run from scratch differ by up to seven accuracy points
(17.3 vs 10.0), which is the resolution limit of a 150-question benchmark.
**7.3 Course reflection / 7.4 Future work.** Verifier v2 with shape templates; longer pools;
distillation of the loop into a single forward pass.
## 8 Conclusion

We asked when an analytic verifier can replace human labels for a 256 M-parameter VLM, and
answered it with a leak-free, compute-matched five-arm study over two seeds and two grid
resolutions. The verifier turns out to be a high-recall but non-injective judge: it recovers the
ground-truth cell in 100 % of localisation queries while accepting 1.08-1.10 extra cells, so
62-76 % of queries admit more than one "valid" answer and every claim has to be reported under
both strict and lenient parsing. Under that protocol the pilot study's localisation gain
disappears - 0 % strict accuracy in every arm, while lenient parsing still credits one arm with
39 % - inference-time verification is the reliable lever (+11.6 points over supervised
fine-tuning), retraining-based self-improvement is unstable at low difficulty (18.3 +- 12.7 %)
yet decisive where supervised fitting fails (3.3 -> 18.0 % on the 8x8 grid), and the gain cannot
be bought with extra optimisation steps (18.0 -> 7.7 % for the compute control). The practical
takeaway is that a cheap verifier is best used as an inference-time judge first and as a label
source second, and that any self-training result must be defended against a compute-matched
control and a leak-free split.


## References (GB/T 7714)
[1] Vaswani A, et al. Attention is all you need[C]//NeurIPS, 2017.
[2] HuggingFaceTB. SmolVLM-256M-Instruct[EB/OL]. 2025.
[3] Course Teaching Group. 大模型基础与应用探索课程讲义[Z]. HITSZ, 2026.
[4] Zelikman E, et al. STaR: Bootstrapping reasoning with reasoning[C]//NeurIPS, 2022.
[5] Yuan Z, et al. Self-rewarding language models[C]//ICML, 2024.
## Appendix
### A. Submission status
Not submitted at the time of writing; the reproducibility package (code, generated data,
adapters and logs) is included with this report. If an arXiv preprint is created, cite it as
arXiv:xxxx.xxxxx here and add the same sentence to the footnote of the first page.
### B. Core code snippet (the verifier gate)
```python
# src/experiments.py, self_training_round() - only wrong->right flips enter the pool
ok1, label = verify_answer(q, pred, img, grid)         # analytic verifier, no network
kind = "first_try" if ok1 else None
if not ok1 and policy != "first_try_only":
    hint = evidence_text(img, q["target"][0], grid)    # "N components; block 1 at row r col c"
    pred2 = model.answer(img, q["q"], hint=hint)
    ok2, label2 = verify_answer(q, pred2, img, grid)
    if ok2:
        kind, label = "flip", label2
if kind == "flip" or (kind == "first_try" and policy == "all_verified"):
    pool.append({"prompt": q["q"], "answer": label})   # label comes from the verifier
```
### C. Supplementary results
Per-question records for every arm are stored as `runs/real/g<grid>/s<seed>/records_*.json`
(question id, gold, first prediction, final prediction, strict/lenient verdict, IoU, latency).
The grid-8 run (`runs/real/g8/s42/results.json`) is the evidence for the difficulty-regime claim
in Section 4.3: B = 3.3 %, D = 0.0 %, E = 18.0 %.
[6] Laurencon H, et al. SmolVLM: Redefining small and efficient multimodal models[EB/OL]. arXiv:2504.05299, 2025.
[7] Guo D, et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning[EB/OL]. arXiv:2501.12948, 2025.
[8] Yuan W, et al. Self-rewarding language models[C]//ICML, 2024.
[9] Johnson J, et al. CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning[C]//CVPR, 2017.
