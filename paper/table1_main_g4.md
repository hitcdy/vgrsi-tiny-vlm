# Table 1. Main results, grid 4x4, mean over 2 seed(s)

| Model | Acc (lenient) | Acc (strict) | loc | count | exist | Mean IoU | #Params (trainable) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| SmolVLM-256M zero-shot (A) | 10.3 ± 0.5 | 6.3 ± 0.5 | 0.0 ± 0.0 | 14.0 ± 0.0 | 5.0 ± 1.4 | 0.0 ± 0.0 | 257.8 M (1.36 M) | 3146.9 ± 0.1 |
| + LoRA SFT on seed split (B) | 27.7 ± 4.2 | 18.0 ± 6.6 | 0.0 ± 0.0 | 30.0 ± 42.4 | 24.0 ± 22.6 | 0.0 ± 0.0 | 257.8 M (1.36 M) | 3317.7 ± 7.4 |
| + inference-time verifier (C) | 50.6 ± 10.4 | 29.6 ± 9.0 | 39.0 ± 49.5 | 29.0 ± 41.0 | 21.0 ± 18.4 | 0.2 ± 0.1 | 257.8 M (1.36 M) | 7912.1 ± 26.2 |
| + seed-only continued SFT (D, control) | 25.7 ± 1.4 | 7.7 ± 5.2 | 0.0 ± 0.0 | 3.0 ± 1.4 | 20.0 ± 17.0 | 0.0 ± 0.0 | 257.8 M (1.36 M) | 3020.2 ± 10.9 |
| + VGRSI self-training (E, ours) | 33.0 ± 2.4 | 18.3 ± 12.7 | 0.0 ± 0.0 | 21.0 ± 29.7 | 34.0 ± 8.5 | 0.0 ± 0.0 | 257.8 M (1.36 M) | 3566.2 ± 32.3 |
