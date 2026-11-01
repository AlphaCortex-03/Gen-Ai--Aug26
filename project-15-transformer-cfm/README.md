# Project 15 — DiT-Style Transformer Velocity Field (CFM)

**What it does.** Replaces the MLP velocity network with a **transformer**
(DiT-style blocks with AdaLN conditioning on time and energy) for conditional
flow matching on shower images.

**Model.** DiT velocity net, **156k parameters**.

**Result** (from `outputs/metrics.json`):

- final CFM training loss **0.0129**
- generated mean-image MSE: 1.4e-5 (E=10), 1.5e-5 (E=50), 9.6e-6 (E=100)

**How to run:**

```
cd project-15-transformer-cfm
python src/train.py                    # full run
GENAI_ITERS=100 python src/train.py    # quick smoke test
```

Outputs → `outputs/` (loss curve, DiT vs baseline samples, comparison figure,
`metrics.json`).
