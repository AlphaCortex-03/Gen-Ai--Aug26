# Project 8 — Attention U-Net vs ResMLP Velocity Field

**What it does.** Compares a plain **ResMLP** (works on flattened pixels) against a
convolutional **attention U-Net** velocity field for the same conditional flow
matching task, measuring fidelity, speed, and parameter count.

**Model.** ResMLP (566k params) vs AttnUNet (**131k params**) on identical data.

**Result** (from `outputs/metrics.json`):

| model     | params | C2ST AUC | mean rel. err | ms/event |
|-----------|-------:|---------:|--------------:|---------:|
| resmlp    | 565,888| 0.9997   | 0.2162        | 1.16     |
| attn_unet | 131,025| 0.6951   | **0.0174**    | 19.5     |

Trade-off made explicit: the U-Net cuts observable relative error **~12×**
(0.216 → 0.017, closely matching data depth 3.45 / width 2.36) at 4× lower
parameter count, but samples ~17× slower per event. Realistic architecture
budgeting for detector simulation.

**How to run:**

```
cd project-8-attn-unet-flow
python src/main.py        # or: python run.py
```

Outputs → `outputs/figures/` (training + quality curves, sample heatmaps,
profiles for both nets) and `outputs/metrics.json`.
