# Project 12 — Anomaly Detection with Flow Log-Likelihood (Negative Result)

**What it does.** Tests whether a trained conditional flow's **log-probability**
can act as an anomaly score: nominal Standard-Model-like showers should be
high-density, Beyond-Standard-Model (BSM) modified showers low-density.

**Model.** Conditional CFM velocity net (RealNVP-style density estimate for scoring).

**Result** (from `outputs/metrics.json`):

- anomaly ROC AUC = **0.486** (mean score nominal −124.9 vs BSM −125.5)

**Honest negative result**: with this score family the two classes are not
separable (AUC ≈ 0.49, no better than chance). The README and figures document
what was tried and why it fails here — recorded as a baseline so the failure
mode is understood, without overclaiming.

**How to run:**

```
cd project-12-anomaly-flow
python src/train.py                    # full run
GENAI_ITERS=100 python src/train.py    # quick smoke test
```

Outputs → `outputs/` (anomaly examples, ROC curve, `metrics.json`).
