# Project 6 — Classifier Two-Sample Test: Physics Features vs Raw Pixels

**What it does.** Answers "how good must a fake be to fool a discriminator?" —
trains a small classifier to separate real vs synthetic showers using either
**physics features** (depth, width, sparsity, cell count, profiles) or **raw
pixels**, for four fake types (flow, blurred, mean-image, pure noise).

**Model.** 407k-param MLP discriminator over physics features vs pixels.

**Result** (from `outputs/metrics.json`):

| fake     | AUC (physics) | AUC (pixels) |
|----------|--------------:|-------------:|
| flow     | 0.9994        | 0.9928       |
| blurred  | 0.9999        | 0.9998       |
| mean     | **0.9568**    | **0.5926**   |
| noise    | 1.0000        | 1.0000       |

Key finding: on the hard **mean-image fake**, physics features detect it easily
(AUC 0.957) while raw pixels barely can (AUC 0.593) — feature choice dominates
generated-data evaluation. Includes a permutation test (p = 0.048, null
AUC = 0.501 ± 0.023) and a power-vs-n study (128 → 1024 events).

**How to run:**

```
cd project-6-c2st-test
python src/main.py        # or: python run.py
```

Outputs → `outputs/figures/` (ROC curves, permutation null, power curve,
fake-vs-data images) and `outputs/metrics.json`.
