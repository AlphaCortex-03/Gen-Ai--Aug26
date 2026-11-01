# Project 4 — Flow Distillation: 50-Step Teacher → 1-Step Student

**What it does.** Distills a 50-step Euler sampler teacher CFM into a student that
generates in a **single network evaluation**, trading a small fidelity cost for a
large latency win.

**Model.** Teacher and student share the same ResMLP velocity architecture (565k
params); the student is trained on (x_t, t) → teacher velocity pairs.

**Result** (from `outputs/metrics.json`):

| model   | NFE | C2ST AUC | rel. err | ms/event |
|---------|----:|---------:|---------:|---------:|
| teacher | 50  | 0.9939   | 0.117    | 1.089    |
| student | 1   | **0.9975** | 0.153  | **0.0243** |

**45× latency speedup** (1.089 → 0.0243 ms/event) while C2ST AUC *improves*
slightly; observable relative error rises 0.117 → 0.153.

**How to run:**

```
cd project-4-flow-distillation
python src/main.py        # or: python run.py
```

Outputs → `outputs/figures/` (teacher/student loss, quality vs NFE, latency vs
NFE, 1-step-vs-data) and `outputs/metrics.json`.
