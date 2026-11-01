# Project 1 — Calo Demo: Synthetic Calorimeter Shower Simulation (NumPy)

**What it does.** A dependency-light ShowerMC-style simulator for 2D electromagnetic
calorimeter showers: per-event beam energy sampling, layer-wise energy deposition,
longitudinal/lateral profiles, event displays and observable summary statistics
(shower depth, depth RMS, width, cell multiplicity, sparsity, peak fraction) for
8192 events in the 10–100 GeV range.

**Model / method.** Pure NumPy parametric shower model — no neural network. This is
the physics data generator shared (conceptually) by all later projects.

**Result** (from `outputs/metrics.json`):

| energy band | events | shower depth (layers) | width | sparsity |
|-------------|-------:|----------------------:|------:|---------:|
| 10–25 GeV   | 3289   | 2.86 ± 2.05           | 2.21  | 0.34     |
| 25–55 GeV   | 2788   | 3.61 ± 2.55           | 2.40  | 0.41     |
| 55–100 GeV  | 2115   | 4.20 ± 2.92           | 2.53  | 0.46     |

Showers deepen and sparsen with energy as expected; these bands are the conditioning
targets of the generative projects.

**How to run** (from the repo root):

```
cd project-1-calo-demo
python run.py            # or: python src/main.py
```

Outputs → `outputs/figures/` (event display, mean images, observables, energy
profiles) and `outputs/metrics.json`.
