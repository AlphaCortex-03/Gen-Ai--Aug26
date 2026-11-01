# Project 7 — Ensemble Flow Matching with Epistemic Uncertainty

**What it does.** Trains an independent **3-member ensemble** of conditional flow
models and maps where sampled showers disagree, separating epistemic (model)
from aleatoric (intrinsic) uncertainty.

**Model.** 3 × ResMLP velocity nets (407k params each), independently seeded; mean
and per-pixel std across members.

**Result** (from `outputs/metrics.json`, E = 50 GeV, 512 events):

- member C2ST AUCs: 0.9995 / 0.9999 / 0.9999 (spread ±0.0002)
- ensemble mean vs data: pixel correlation **0.9993**, mean abs err 8.4e-4
- mean epistemic std 1.7e-4 vs aleatoric 3.7e-3 — uncertainty map highlights
  the shower core where models disagree most

**How to run:**

```
cd project-7-ensemble-flow
python src/main.py        # or: python run.py
```

Outputs → `outputs/figures/` (ensemble maps, member-vs-data, pixel closure,
relative uncertainty) and `outputs/metrics.json`.
