# Project 10 — Quantization & Pruning of a Trained Flow (Size/Quality Pareto)

**What it does.** Compresses a trained flow-matching sampler with **INT8 dynamic
quantization** and magnitude **pruning** (50/70/90%), tracing the model-size vs
sample-quality Pareto frontier.

**Model.** 566k-param ResMLP velocity net; size measured per exported variant.

**Result** (from `outputs/metrics.json`):

| variant      | size   | C2ST AUC | rel. err |
|--------------|-------:|---------:|---------:|
| fp32         | 2223 KB| 0.9983   | 0.157    |
| int8         | 592 KB | 0.9982   | 0.157    |
| prune90+int8 | 592 KB | 1.000    | 0.763    |

**3.8× compression** with **0.0001 AUC loss** via INT8. Pruning alone, however,
destroys physical fidelity (rel. err 0.157 → 0.763 even though the pixel-level
AUC looks perfect) — a cautionary result: C2ST on pixels alone hides physics
damage. Quantization, not pruning, is the safe lever here.

**How to run:**

```
cd project-10-quantized-flow
python src/main.py
```

Outputs → `outputs/figures/` (Pareto curve, fp32 vs pruned samples) and
`outputs/metrics.json` with the full variant table.
