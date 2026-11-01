# Project 13 — Latent-Space Flow Matching (VAE + CFM)

**What it does.** Trains a VAE on shower images, then runs conditional flow
matching **in the compressed latent space** — a two-stage pipeline that cuts
sampling cost while keeping pixel-level fidelity.

**Model.** Conv VAE (encoder/decoder) → 566k-param conditional CFM in latent space.

**Result** (from `outputs/metrics.json`):

- VAE reconstruction MSE **1.4e-4**
- generated vs reference mean-image MSE **1.4e-6**

**How to run:**

```
cd project-13-latent-cfm
python src/train.py                    # full run
GENAI_ITERS=100 python src/train.py    # quick smoke test
```

Outputs → `outputs/` (latent pipeline reconstruction, latent-space scatter,
`metrics.json`).
