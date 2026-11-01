# Project 4 — Flow Distillation: 50-Step Teacher → 1-Step Student

**What it does.** Distills a 50-step Euler sampler teacher CFM into a student that
generates in a **single network evaluation**, trading a small fidelity cost for a
large latency win.

**Model.** Teacher and student share the same ResMLP velocity architecture (565k
params); the student is trained on (x_t, t) → teacher velocity pairs.

**Result** (from `outputs/metrics.json`):

| model   | NFE | C2ST AUC | rel. err | ms/event |
|---------|----:|---------:|---------:|---------:|
| teacher | 50  | 0.9962   | 0.126    | 0.485    |
| student | 1   | **0.9985** | 0.164  | **0.0099** |

**49× latency speedup** (0.485 → 0.0099 ms/event) while C2ST AUC *improves*
slightly; observable relative error rises 0.126 → 0.164.

**How to run:**

```
cd project-4-flow-distillation
python src/main.py        # or: python run.py
```

Outputs → `outputs/figures/` (teacher/student loss, quality vs NFE, latency vs
NFE, 1-step-vs-data) and `outputs/metrics.json`.
