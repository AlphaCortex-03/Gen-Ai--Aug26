# Project 2 — Conditional Flow Matching on Beam Energy

**What it does.** Trains a conditional continuous normalizing flow (velocity-field
ODE) to generate 2D calorimeter shower images conditioned on beam energy, then
evaluates sample fidelity with physics observables and a classifier two-sample
test (C2ST).

**Model.** ResMLP velocity network (565k params), Euler ODE sampler (32 steps),
CFM loss on Gaussian-path pairs.

**Result** (from `outputs/metrics.json`):

- C2ST AUC **0.998** (per-energy bands 0.984–0.999)
- mean relative error of physics observables **13%**
- depth profile L1 error 0.115, lateral profile L1 error 0.122
- training: ~40 s on CPU

**How to run:**

```
cd project-2-conditional-cfm
python src/main.py        # or: python run.py
```

Outputs → `outputs/figures/` (loss, generated-vs-data, profiles, conditional
quality by band) and `outputs/metrics.json`.
