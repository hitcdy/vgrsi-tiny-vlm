# Table 1. Main results, grid 8×8, mean over 1 seed(s)

| Model | Acc (lenient) | Acc (strict) | loc | count | exist | Mean IoU | #Params (trainable) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| SmolVLM-256M zero-shot (A) | 9.3 | 4.7 | 0.0 | 14.0 | 0.0 | 0.0 | 257.8 M (1.36 M) | 3074.3 |
| + LoRA SFT on seed split (B) | 22.7 | 3.3 | 0.0 | 10.0 | 0.0 | 0.0 | 257.8 M (1.36 M) | 3207.3 |
| + inference-time verifier (C) | 26.7 | 4.0 | 6.0 | 6.0 | 0.0 | 0.1 | 257.8 M (1.36 M) | 7896.5 |
| + seed-only continued SFT (D, control) | 17.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 257.8 M (1.36 M) | 2458.2 |
| + VGRSI self-training (E, ours) | 31.3 | 18.0 | 0.0 | 12.0 | 42.0 | 0.0 | 257.8 M (1.36 M) | 3289.1 |
