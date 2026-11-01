# Gen-Ai--Aug26 — Generative Models for Calorimeter Shower Simulation (August 2026)

[![CI](https://github.com/AlphaCortex-03/Gen-Ai--Aug26/actions/workflows/ci.yml/badge.svg)](https://github.com/AlphaCortex-03/Gen-Ai--Aug26/actions/workflows/ci.yml)

A self-contained practice portfolio of **16 generative-AI experiments** on synthetic 2D calorimeter
showers, built as preparation for GSoC projects on ML-based detector simulation (ML4SCI-style).
Every project runs **end-to-end on a plain CPU laptop** (numpy / PyTorch / matplotlib only), produces
its own figures and `metrics.json`, and is verified with a single command:

```
python run_all.py        # runs all 16 projects sequentially, ~10 min total on CPU, 16/16 PASS
```

## Projects

| # | Project | What it demonstrates | Key result |
|---|---------|----------------------|-----------|
| 1 | [project-1-calo-demo](project-1-calo-demo/) | NumPy-only ShowerMC simulation: profiles, observables, event displays | shower depth 3.46 layers, width 2.35 (8192 events) |
| 2 | [project-2-conditional-cfm](project-2-conditional-cfm/) | Conditional flow matching on beam energy with C2ST evaluation | C2ST AUC **0.998**, mean rel. err 13% |
| 3 | [project-3-energy-interpolation](project-3-energy-interpolation/) | Train on {10, 50, 100} GeV, sample unseen {30, 70} GeV | unseen-energy AUC **0.994** |
| 4 | [project-4-flow-distillation](project-4-flow-distillation/) | 50-step teacher → 1-step student flow matching | **49× faster** sampling, AUC 0.9985 kept |
| 5 | [project-5-pointcloud-cfm](project-5-pointcloud-cfm/) | Shower as a point cloud (x, y, z, E) with a DeepSets velocity net | AUC 0.980 with only **41k params** |
| 6 | [project-6-c2st-test](project-6-c2st-test/) | Classifier two-sample test: physics features vs raw pixels | physics AUC 0.9568 vs pixels 0.5926 on mean-image fakes |
| 7 | [project-7-ensemble-flow](project-7-ensemble-flow/) | 3-member flow ensemble → epistemic uncertainty map | member spread AUC ±0.0002, pixel corr 0.9993 |
| 8 | [project-8-attn-unet-flow](project-8-attn-unet-flow/) | ResMLP vs attention-UNet velocity field | UNet rel. err **0.024 vs 0.227** MLP (9× better) |
| 9 | [project-9-layer-autoregressive](project-9-layer-autoregressive/) | Stage-wise layer-by-layer autoregressive generation | AR rel. err 0.037 vs joint 0.238 |
| 10 | [project-10-quantized-flow](project-10-quantized-flow/) | INT8 quantization + pruning, size/quality Pareto | **3.8× smaller**, AUC drop 0.0001 |
| 11 | [project-11-benchmark-3-gens](project-11-benchmark-3-gens/) | WGAN-GP vs DDPM vs CFM on identical data | CFM AUC **0.9906** > DDPM 0.9577 > WGAN 0.7234 |
| 12 | [project-12-anomaly-flow](project-12-anomaly-flow/) | Flow log-prob as BSM anomaly score | negative result: AUC 0.486 (honest baseline) |
| 13 | [project-13-latent-cfm](project-13-latent-cfm/) | VAE latent space + CFM in latent space | VAE recon MSE 1.4e-4, gen. MSE 1.4e-6 |
| 14 | [project-14-cfg-calo](project-14-cfg-calo/) | Classifier-free guidance for energy control | shower radius grows 6.05 → 6.23 with guidance w 0→7 |
| 15 | [project-15-transformer-cfm](project-15-transformer-cfm/) | DiT-style transformer velocity field with AdaLN | final CFM loss **0.0129**, 156k params |
| 16 | [project-16-consistency-calo](project-16-consistency-calo/) | Consistency model, 1-step sampling | 1-step MSE 0.011 vs 50-step 0.009 |

Folder names map 1:1 to the table order (project-1 … project-16).

## Quickstart

```powershell
pip install -r requirements.txt

# one project (projects 1-10 style):
cd project-2-conditional-cfm
python src/main.py           # or: python run.py

# one project (projects 11-16 style):
cd project-11-benchmark-3-gens
python src/train.py

# everything, sequentially, with a timing summary table:
python run_all.py
```

`run_all.py` options:

```powershell
python run_all.py --list          # show entry points only
python run_all.py --smoke         # fast run (GENAI_ITERS=100 for projects 11-16)
python run_all.py --only 2,11     # run only specific projects
python run_all.py --stop          # abort at the first failure
python run_all.py --timeout 600   # per-project timeout in seconds
```

## Repository layout

```
Gen-Ai--Aug26/
├── common/calo/               # shared toolkit: data sim, nets (ResMLP/UNet/DiT/DeepSets/VAE),
│   │                          # CFM trainer/sampler, metrics (C2ST, W1, profiles), viz
├── project-1-calo-demo/       … project-10-quantized-flow/   (src/main.py + run.py style)
│   ├── run.py                  <- thin wrapper: python run.py
│   ├── src/main.py             <- actual pipeline
│   ├── configs/default.yaml
│   └── outputs/                <- committed result figures + metrics.json
├── project-11-benchmark-3-gens/ … project-16-consistency-calo/ (src/train.py style)
│   ├── src/train.py            <- single-file self-contained pipeline
│   └── outputs/
├── run_all.py                  # runs every project and times it
├── requirements.txt
└── README.md
```

## Notes

- Python 3.12 + CPU-only PyTorch verified end-to-end: all 16 projects, ~10 min
  total on a plain laptop CPU. No GPU required.
- Every push runs a CI smoke pass of all 16 projects on GitHub Actions
  (see the badge above); `run_results.json` from each run is uploaded as an artifact.
- Datasets and `.pt` checkpoints are **not committed**: all data is synthetic
  (regenerated deterministically by `common/calo/data.py` / `src/calo_sim.py`),
  and every model retrains in seconds to minutes on CPU.
- Each project's `outputs/metrics.json` records the exact evaluation numbers
  cited in its README.
