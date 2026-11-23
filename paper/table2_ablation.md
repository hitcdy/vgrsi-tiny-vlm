# Table 2. Ablation on grid 4x4

| Variant | rounds | pool policy | Acc (strict) | loc (strict) | parser FP |
|---|---|---|---|---|---|
| round 1 (seed s42) | 1 | flip_only | 17.3 | 0.0 | 32 |
| round 2 (seed s42) | 2 | flip_only | 27.3 | 0.0 | 11 |
| 1 round, all_verified pool | 1 | all_verified | 20.0 | 0.0 | 15 |
| 1 round, no evidence re-prompt | 1 | first_try_only | 3.3 | 0.0 | 31 |
| 1 round, round lr=3e-4 | 1 | flip_only | 4.7 | 0.0 | 34 |
| 1 round | 1 | flip_only | 10.0 | 0.0 | 34 |
| round 1 (seed s7) | 1 | flip_only | 16.7 | 0.0 | 28 |
| round 2 (seed s7) | 2 | flip_only | 9.3 | 0.0 | 33 |
