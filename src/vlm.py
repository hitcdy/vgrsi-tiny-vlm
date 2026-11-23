"""Real backbone: SmolVLM-256M-Instruct + LoRA, with a frozen vision-tower cache.

The cache is what makes a full 5-arm study feasible on one CPU/8 GB GPU: the
vision tower is frozen, so its output for a given image is a pure function of
the pixels.  Keying a dict on the md5 of `pixel_values` turns a ~2-25 s encode
into a dict lookup for every later question on the same image.
"""
from __future__ import annotations

import hashlib
import os
import time

import torch
from PIL import Image

# Backbone resolution order:
#   1) $VGRSI_MODEL (a hub id or a local folder)
#   2) ./models/SmolVLM-256M-Instruct  (fetched by scripts/fetch_model.py)
#   3) the hub id (needs network, or a warm cache)
# On this machine %USERPROFILE%\.cache is not writable, so HF_HOME is redirected
# into the project; with a local folder present we also go fully offline.
_LOCAL_DIR = os.path.join("models", "SmolVLM-256M-Instruct")
_HAS_LOCAL = os.path.isdir(_LOCAL_DIR) and os.path.exists(
    os.path.join(_LOCAL_DIR, "model.safetensors"))
MODEL_ID = os.environ.get("VGRSI_MODEL") or (_LOCAL_DIR if _HAS_LOCAL
                                             else "HuggingFaceTB/SmolVLM-256M-Instruct")
# 32 logical cores thrash on the small tensors of a 256 M model; the pilot used 4.
# Override with VGRSI_THREADS.
try:
    torch.set_num_threads(int(os.environ.get("VGRSI_THREADS", "8")))
except Exception:
    pass
os.environ.setdefault("HF_HOME", os.path.abspath(".hf_home"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
if _HAS_LOCAL and not os.environ.get("VGRSI_MODEL"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def resolve_model_id(preferred: str = None) -> str:
    """Explicit local folder > auto-detected local folder > whatever was asked for."""
    if preferred and os.path.isdir(preferred):
        return preferred
    if _HAS_LOCAL:
        return _LOCAL_DIR
    return preferred or MODEL_ID


def _pick_cls():
    import transformers as T
    for name in ("AutoModelForVision2Seq", "AutoModelForImageTextToText"):
        if hasattr(T, name):
            return getattr(T, name)
    raise RuntimeError("transformers has no vision2seq auto class")


def _fingerprint(pixel_values):
    x = pixel_values.detach().to("cpu").contiguous()
    h = hashlib.md5()
    h.update(str(tuple(x.shape)).encode())
    h.update(x.numpy().tobytes())
    return h.hexdigest()


VISION_CACHE_DIR = os.environ.get("VGRSI_VISION_CACHE", ".cache_vision")


def _disk_load(key):
    path = os.path.join(VISION_CACHE_DIR, key + ".pt")
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None


def _disk_save(key, out):
    """torch.save round-trips whatever get_image_features returns (tensor, tuple,
    or a model-output dataclass), which numpy cannot."""
    try:
        os.makedirs(VISION_CACHE_DIR, exist_ok=True)
        torch.save(out, os.path.join(VISION_CACHE_DIR, key + ".pt"))
    except Exception as e:
        print(f"[vision] disk cache write failed: {e}", flush=True)


def enable_vision_cache(model, tag="vision"):
    from peft import PeftModel

    outer = model.base_model.model if isinstance(model, PeftModel) else model
    inner = getattr(outer, "model", None)
    targets = [outer] + ([inner] if isinstance(inner, torch.nn.Module) else [])
    store = {}
    for tgt in targets:
        if getattr(tgt, "_vgrs_cache_on", False):
            continue
        orig = tgt.get_image_features

        def wrap(_orig):
            def cached(pixel_values, pixel_attention_mask=None, **kw):
                key = _fingerprint(pixel_values)
                if key not in store:
                    hit = _disk_load(key)
                    if hit is not None:
                        store[key] = hit
                        return store[key]
                    with torch.no_grad():
                        store[key] = _orig(pixel_values, pixel_attention_mask, **kw)
                    _disk_save(key, store[key])
                    print(f"[{tag}] vision encoded ({len(store)} in memory)", flush=True)
                return store[key]
            return cached

        tgt.get_image_features = wrap(orig)
        tgt._vgrs_cache_on = True
    print(f"[{tag}] vision cache on: {[type(t).__name__ for t in targets]}")
    return store


class VLM:
    """Uniform interface shared with `mock.MockVLM` so the pipeline is backend-agnostic."""

    backend = "real"

    def __init__(self, model_id: str = MODEL_ID, attach_lora: bool = True, lora_r: int = 8,
                 lora_alpha: int = 16, lora_dropout: float = 0.05,
                 image_splitting: bool = False, dtype: str = "auto", vision_cache: bool = True):
        from transformers import AutoProcessor

        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.do_image_splitting = bool(image_splitting)
        if dtype == "auto":
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)
        cls = _pick_cls()
        try:
            model = cls.from_pretrained(model_id, dtype=torch_dtype)
        except TypeError:
            model = cls.from_pretrained(model_id, torch_dtype=torch_dtype)
        self.stats = {"total": sum(p.numel() for p in model.parameters()),
                      "trainable": 0, "ratio_pct": 0.0}
        if attach_lora:
            from peft import LoraConfig, get_peft_model

            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                             task_type="CAUSAL_LM")
            model = get_peft_model(model, cfg)
            total = sum(p.numel() for p in model.parameters())
            train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.stats = {"total": total, "trainable": train,
                          "ratio_pct": round(100.0 * train / total, 3)}
        self.model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.device = next(self.model.parameters()).device
        if vision_cache:
            enable_vision_cache(self.model)
        print(f"[params] total {self.stats['total']:,} / trainable {self.stats['trainable']:,} "
              f"({self.stats['ratio_pct']}%)")

    # ---------- inference ----------
    def _chat(self, text: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return self.processor.apply_chat_template(msgs, add_generation_prompt=True)

    @torch.no_grad()
    def answer(self, img, question: str, hint: str = None, max_new: int = 24) -> str:
        text = question if hint is None else f"{question}\n（校验提示：{hint}）"
        images = [as_pil(img)]
        inputs = self.processor(text=[self._chat(text)], images=images,
                                return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        return self.processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()

    # ---------- training ----------
    def sft(self, samples, steps: int = 60, lr: float = 1e-3, seed: int = 0,
            log_every: int = 10):
        import random as _r

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("no trainable parameters: adapter is frozen "
                               "(load it with is_trainable=True)")
        opt = torch.optim.AdamW(trainable, lr=lr)
        self.model.train()
        curve, done, epoch, t0 = [], 0, 0, time.time()
        while done < steps:
            order = list(range(len(samples)))
            _r.Random(1000 + seed + epoch).shuffle(order)
            for idx in order:
                if done >= steps:
                    break
                s = samples[idx]
                img = as_pil(s["img"])
                prompt = self._chat(s["prompt"])
                full = self.processor(text=[prompt + s["answer"]], images=[img],
                                      return_tensors="pt").to(self.device)
                with torch.no_grad():
                    plen = self.processor(text=[prompt], images=[img],
                                          return_tensors="pt")["input_ids"].shape[1]
                labels = full["input_ids"].clone()
                labels[:, :plen] = -100
                full["labels"] = labels
                loss = self.model(**full).loss
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                done += 1
                curve.append([done, round(float(loss.item()), 4)])
                if done % log_every == 0 or done == steps:
                    print(f"[sft] step {done}/{steps} loss={loss.item():.4f} "
                          f"({time.time() - t0:.1f}s)")
            epoch += 1
        self.model.eval()
        return {"curve": curve, "steps": done, "epochs": epoch,
                "seconds": round(time.time() - t0, 1), "pool": len(samples)}

    # ---------- io ----------
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)

    def snapshot(self):
        """Copy the adapter weights so a compute-control arm can be rolled back
        without reloading the backbone (or re-encoding 110 images)."""
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()
                if "lora_" in k}

    def restore(self, snap) -> None:
        missing = self.model.load_state_dict(snap, strict=False)
        print(f"[restore] adapter rolled back ({len(snap)} tensors, "
              f"{len(getattr(missing, 'unexpected_keys', []))} unexpected)", flush=True)

    def load(self, path: str):
        from peft import PeftModel

        if not os.path.exists(path):
            raise FileNotFoundError(path)
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        # is_trainable=True is essential: the default freezes the adapter and any
        # later optimiser gets an empty parameter list.
        self.model = PeftModel.from_pretrained(base, path, is_trainable=True).to(self.device)
        self.model.eval()
        enable_vision_cache(self.model)
        return self


def as_pil(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.open(img).convert("RGB")