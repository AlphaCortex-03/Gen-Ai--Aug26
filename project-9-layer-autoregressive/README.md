# Project 9 — Layer-Wise Autoregressive Shower Generation

**What it does.** Generates calorimeter showers **layer by layer** with an
autoregressive chain conditioned on (energy, previously deposited layers),
versus a joint model that generates the whole image at once.

**Model.** 4-stage autoregressive chain (997k params total) vs joint ResMLP (566k).

**Result** (from `outputs/metrics.json`):

| model          | params | C2ST AUC | rel. err | sample time |
|----------------|-------:|---------:|---------:|------------:|
| autoregressive | 996,608| 0.9548   | **0.0374** | 0.497 s   |
| joint          | 565,888| 0.99997  | 0.2375   | 0.136 s    |

Trade-off profile: the AR chain reproduces physics observables ~6× more
accurately (rel. err 0.037) but is somewhat less indistinguishable overall
(AUC 0.955) and slower to sample — the sequential structure is a fidelity lever.

**How to run:**

```
cd project-9-layer-autoregressive
python src/main.py
```

Outputs → `outputs/figures/` (AR samples, joint samples, stage losses/costs,
profiles) and `outputs/metrics.json`.
