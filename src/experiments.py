"""The five arms of the study.

    A  zero_shot            no adaptation
    B  sft                  LoRA SFT on the labelled seed split (60 images / 180 QA)
    C  sft_infer_verifier   B + inference-time verifier correction (no re-training)
    D  seed_only_cont       B + 15 extra steps on the same seed data  (compute control)
    E  vgrs                 B + N rounds of verifier-gated self-training  (ours)

E is the point of the paper: the self-training pool comes from an *unlabelled*
scenario split (seed 777).  The verifier supplies every label, and the 150-question
evaluation split (seed 42) is never trained on -- this is the leak fixed relative
to the lab06 pilot, which admitted corrected samples of the very same 15 questions
it then reported on.
"""
from __future__ import annotations

import argparse
import os
import time

from . import data_gen, qa
from .config import load as load_cfg
from .evaluate import load_img, run_eval, verify_answer
from .utils import read_json, seed_all, write_json
from .verifier import evidence_text


def build_backend(args, cfg):
    if args.backend == "mock":
        from .mock import MockVLM

        return MockVLM(seed=args.seed)
    from .vlm import VLM, resolve_model_id

    return VLM(model_id=resolve_model_id(cfg["model_id"]), lora_r=cfg["train"]["lora_r"],
               lora_alpha=cfg["train"]["lora_alpha"],
               lora_dropout=cfg["train"]["lora_dropout"],
               image_splitting=bool(args.image_splitting), dtype=args.dtype)


def seed_samples(meta: dict, grid: int):
    """Supervised seed set: gold answers only (this is the 'human labels' budget)."""
    out = []
    for q in qa.build_questions(meta, seed=1):
        out.append({"img": os.path.join(meta["dir"], f"{q['img']}.png"),
                    "prompt": q["q"], "answer": q["gold"], "id": q["id"], "src": "seed"})
    return out


def self_training_round(model, pool_qs, pool_meta, grid, pool, policy="flip_only"):
    """One VGRSI round: sample -> answer -> verify -> correct -> gate -> retrain."""
    admitted, seen = [], []
    t0 = time.time()
    for q in pool_qs:
        img = load_img(pool_meta, q["img"])
        pred = model.answer(img, q["q"])
        ok1, lab = verify_answer(q, pred, img, grid)
        kind = "first_try" if ok1 else None
        final_pred = pred
        if not ok1 and policy != "first_try_only":
            hint = evidence_text(img, q["target"][0], grid)
            pred2 = model.answer(img, q["q"], hint=hint)
            ok2, lab2 = verify_answer(q, pred2, img, grid)
            if ok2:
                kind, lab, final_pred = "flip", lab2, pred2
        rec = {"id": q["id"], "type": q["type"], "first": pred, "final": final_pred,
               "verifier_ok": bool(kind), "kind": kind, "label": lab}
        seen.append(rec)
        if kind == "flip" or (kind == "first_try" and policy == "all_verified"):
            admitted.append({"img": os.path.join(pool_meta["dir"], f"{q['img']}.png"),
                             "prompt": q["q"], "answer": lab, "id": q["id"], "src": kind})
    new_pool = pool + admitted
    return {"admitted": admitted, "seen": seen, "pool": new_pool,
            "seconds": round(time.time() - t0, 1),
            "n_flip": sum(1 for a in admitted if a["src"] == "flip"),
            "n_first": sum(1 for a in admitted if a["src"] == "first_try")}


def arm_zero_shot(model, eval_meta, eval_qs, grid, limit=0):
    recs, agg, sec = run_eval(model, eval_meta, eval_qs, grid, limit=limit)
    return {"records": recs, "metrics": agg, "seconds": sec}


def arm_sft(model, seed_data, steps, lr, seed):
    logs = model.sft(seed_data, steps=steps, lr=lr, seed=seed)
    return {"train": logs}


def arm_seed_only_cont(model, seed_data, extra_steps, lr, seed):
    """Compute control: roll the adapter back to the post-SFT state afterwards, so
    the VGRSI arm that follows still starts from arm B and not from a drifted model."""
    snap = model.snapshot() if hasattr(model, "snapshot") else None
    logs = model.sft(seed_data, steps=extra_steps, lr=lr, seed=seed + 500)
    return {"train": logs, "snapshot": snap}


def arm_vgrs(model, pool_qs, pool_meta, eval_meta, eval_qs, grid, pool, cfg, seed,
             rounds=2):
    out = {"rounds": [], "pool_curve": [len(pool)]}
    for r in range(1, rounds + 1):
        res = self_training_round(model, pool_qs, pool_meta, grid, pool,
                                  policy=cfg["rsi"]["pool_policy"])
        pool = res["pool"]
        train = model.sft(pool, steps=cfg["rsi"]["steps_per_round"],
                          lr=cfg["rsi"].get("lr") or cfg["train"]["lr"],
                          seed=seed + 100 * r)
        recs, agg, sec = run_eval(model, eval_meta, eval_qs, grid)
        out["rounds"].append({"round": r, "n_admitted": len(res["admitted"]),
                              "n_flip": res["n_flip"], "n_first": res["n_first"],
                              "pool_size": len(pool), "seconds": res["seconds"],
                              "train": train, "metrics": agg, "records": recs})
        out["pool_curve"].append(len(pool))
        print(f"[vgrs] round {r}: pool {len(pool)} (flip {res['n_flip']}) "
              f"acc_strict {agg['acc_strict']}%")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="VGRSI experiments")
    ap.add_argument("--config", default="configs/exp.yaml")
    ap.add_argument("--backend", default="real", choices=["real", "mock"])
    ap.add_argument("--arm", default="all",
                    choices=["zero_shot", "sft", "sft_infer_verifier",
                             "seed_only_cont", "vgrs", "all"])
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--policy", default="", choices=["", "flip_only", "all_verified",
                                                     "first_try_only"])
    ap.add_argument("--image_splitting", type=int, default=0)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lr_rsi", type=float, default=0.0,
                    help="learning rate for the self-training rounds only (arm E)")
    args = ap.parse_args(argv)

    cfg = load_cfg(args.config)
    cfg["train"]["steps"] = args.steps or cfg["train"]["steps"]
    cfg["rsi"]["rounds"] = args.rounds or cfg["rsi"]["rounds"]
    if args.policy:
        cfg["rsi"]["pool_policy"] = args.policy
    if args.lr_rsi:
        cfg["rsi"]["lr"] = args.lr_rsi
    seed_all(args.seed)
    grid = args.grid
    metas = data_gen.ensure_data(cfg["data"]["out_dir"], grids=(grid,),
                                 n_train=cfg["data"]["n_train"],
                                 n_pool=cfg["data"]["n_pool"],
                                 n_eval=cfg["data"]["n_eval"])
    train_meta, pool_meta, eval_meta = metas[(grid, "train")], metas[(grid, "pool")], metas[(grid, "eval")]
    eval_qs = qa.build_questions(eval_meta, seed=args.seed)
    pool_qs = qa.build_pool(pool_meta, seed=args.seed)
    print(f"[data] grid={grid} train={len(train_meta['gt'])} pool={len(pool_meta['gt'])} "
          f"eval={len(eval_meta['gt'])} imgs -> {len(eval_qs)} questions")

    out_dir = args.out or os.path.join("runs", args.backend, f"g{grid}", f"s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "results.json")
    res = read_json(res_path, {}) or {}
    res["meta"] = {"backend": args.backend, "grid": grid, "seed": args.seed,
                   "model_id": cfg["model_id"], "n_eval_q": len(eval_qs),
                   "n_pool_q": len(pool_qs), "steps": cfg["train"]["steps"],
                   "rounds": cfg["rsi"]["rounds"],
                   "pool_policy": cfg["rsi"]["pool_policy"],
                   "tag": args.tag, "rsi_lr": cfg["rsi"].get("lr")}
    res.setdefault("arms", {})

    model = build_backend(args, cfg)
    res["params"] = model.stats
    seed_data = seed_samples(train_meta, grid)
    do = (lambda a: args.arm in ("all", a))

    if do("zero_shot"):
        r = arm_zero_shot(model, eval_meta, eval_qs, grid, args.limit)
        res["arms"]["zero_shot"] = {"metrics": r["metrics"], "seconds": r["seconds"]}
        write_json(os.path.join(out_dir, "records_zero_shot.json"), r["records"])
        print(f"[A] zero-shot acc_strict={r['metrics']['acc_strict']}%")

    trained = False
    if any(do(a) for a in ("sft", "sft_infer_verifier", "vgrs", "seed_only_cont")):
        t = arm_sft(model, seed_data, cfg["train"]["steps"], cfg["train"]["lr"], args.seed)
        trained = True
        res["arms"]["sft_train"] = t
        model.save(os.path.join(out_dir, "adapter_sft"))
        print(f"[B] sft {t['train']['steps']} steps in {t['train']['seconds']}s")

    if do("sft"):
        r = arm_zero_shot(model, eval_meta, eval_qs, grid, args.limit)
        res["arms"]["sft"] = {"metrics": r["metrics"], "seconds": r["seconds"]}
        write_json(os.path.join(out_dir, "records_sft.json"), r["records"])
        print(f"[B] sft acc_strict={r['metrics']['acc_strict']}%")

    if do("sft_infer_verifier"):
        recs, agg, sec = run_eval(model, eval_meta, eval_qs, grid, verifier_mode=True,
                                  limit=args.limit)
        res["arms"]["sft_infer_verifier"] = {"metrics": agg, "seconds": sec}
        write_json(os.path.join(out_dir, "records_sft_infer_verifier.json"), recs)
        print(f"[C] sft+infer-verifier acc_strict={agg['acc_strict']}%")

    if do("seed_only_cont"):
        saved = model.snapshot() if hasattr(model, "snapshot") else None
        skill_before = getattr(model, "skill", None)
        t = arm_seed_only_cont(model, seed_data, cfg["train"]["extra_steps"],
                              cfg["train"]["lr"], args.seed)
        r = arm_zero_shot(model, eval_meta, eval_qs, grid, args.limit)
        res["arms"]["seed_only_cont"] = {"metrics": r["metrics"], "seconds": r["seconds"],
                                        "train": t["train"]}
        write_json(os.path.join(out_dir, "records_seed_only_cont.json"), r["records"])
        if saved is not None:
            model.restore(saved)
        elif skill_before is not None:
            model.nudge(skill_before - getattr(model, "skill", 0.0))
        print(f"[D] seed-only cont acc_strict={r['metrics']['acc_strict']}% "
              f"(adapter rolled back to arm-B state)")

    if do("vgrs"):
        if not trained:
            model.load(os.path.join(out_dir, "adapter_sft"))
        pool = list(seed_data)
        out = arm_vgrs(model, pool_qs, pool_meta, eval_meta, eval_qs, grid, pool,
                       cfg, args.seed, rounds=cfg["rsi"]["rounds"])
        res["arms"]["vgrs"] = {
            "metrics": out["rounds"][-1]["metrics"],
            "per_round": [{k: v for k, v in r.items() if k != "records"} for r in out["rounds"]],
            "pool_curve": out["pool_curve"],
        }
        write_json(os.path.join(out_dir, "records_vgrs.json"), out["rounds"][-1]["records"])
        model.save(os.path.join(out_dir, "adapter_vgrs"))
        print(f"[E] vgrs acc_strict={out['rounds'][-1]['metrics']['acc_strict']}%")

    write_json(res_path, res)
    print(f"[done] -> {res_path}")
    return res


if __name__ == "__main__":
    main()