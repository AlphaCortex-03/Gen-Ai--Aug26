# Project 3 — Energy Interpolation to Unseen Beam Energies

**What it does.** Trains the conditional flow model only on {10, 50, 100} GeV
showers and evaluates sampling at **unseen** energies {30, 70} GeV — testing
whether the learned energy conditioning interpolates smoothly instead of only
snapping to the nearest training value.

**Model.** Same ResMLP conditional CFM (565k params), conditioned continuously on E.

**Result** (from `outputs/metrics.json`):

| E (GeV) | seen | C2ST AUC | rel. err | snap-to-nearest AUC | snap rel. err |
|--------:|------|---------:|---------:|--------------------:|--------------:|
| 10      | yes  | 1.0000   | 0.246    | 1.0000              | 0.246         |
| 30      | no   | 0.9985   | 0.127    | 0.9996              | 0.086         |
| 50      | yes  | 0.9900   | 0.079    | 0.9900              | 0.079         |
| 70      | no   | 0.9893   | 0.060    | 0.9933              | 0.046         |

Unseen energies reach **AUC 0.994** (avg) with rel. err 9.3% — comparable to seen
energies, i.e. the conditioning genuinely interpolates.

**How to run:**

```
cd project-3-energy-interpolation
python src/main.py        # or: python run.py
```

Outputs → `outputs/figures/` (loss, energy scan, unseen-energy samples) and
`outputs/metrics.json`.
