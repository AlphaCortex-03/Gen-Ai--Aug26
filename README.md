# Gen-Ai--Aug26

August practice: 16 CPU-only generative-model projects on synthetic calorimeter showers — flow matching, VAE-style latents, evaluation benchmarks.

[![CI](https://github.com/AlphaCortex-03/Gen-Ai--Aug26/actions/workflows/ci.yml/badge.svg)](https://github.com/AlphaCortex-03/Gen-Ai--Aug26/actions/workflows/ci.yml)

## Overview

This repository contains 16 standalone CPU-only projects (p5–p20) exploring generative modeling on synthetic calorimeter shower data. All projects run on NumPy / PyTorch / matplotlib and are validated through GitHub Actions CI on every push.

| # | Project | Highlight |
|---|---------|-----------|
| p5 | [calo-demo](p5-calo-demo/) | NumPy calorimeter shower simulator: profiles, observables, event displays |
| p6 | [conditional-cfm](p6-conditional-cfm/) | Conditional flow matching on beam energy, C2ST AUC 0.998 |
| p7 | [energy-interpolation](p7-energy-interpolation/) | Sample unseen beam energies (30/70 GeV) from a flow trained on 10/50/100 GeV |
| p8 | [flow-distillation](p8-flow-distillation/) | Distilled 50-step flow sampler into a 1-step student, 45× faster |
| p9 | [pointcloud-cfm](p9-pointcloud-cfm/) | Point-cloud shower generation with a DeepSets velocity net (41k params) |
| p10 | [c2st-test](p10-c2st-test/) | Classifier two-sample test: physics features vs raw pixels |
| p11 | [ensemble-flow](p11-ensemble-flow/) | 3-member flow ensemble with epistemic uncertainty maps |
| p12 | [attn-unet-flow](p12-attn-unet-flow/) | Attention U-Net vs ResMLP velocity field comparison |
| p13 | [layer-autoregressive](p13-layer-autoregressive/) | Layer-by-layer autoregressive shower generation (4 stages) |
| p14 | [quantized-flow](p14-quantized-flow/) | INT8 quantisation + pruning: 3.8× smaller with ~zero AUC loss |
| p15 | [benchmark-3-gens](p15-benchmark-3-gens/) | WGAN-GP vs DDPM vs CFM on identical data |
| p16 | [anomaly-flow](p16-anomaly-flow/) | Flow log-prob as anomaly score (negative result, AUC 0.486) |
| p17 | [latent-cfm](p17-latent-cfm/) | VAE latent space + flow matching in latent space |
| p18 | [cfg-calo](p18-cfg-calo/) | Classifier-free guidance for energy control (w sweep 0-7) |
| p19 | [transformer-cfm](p19-transformer-cfm/) | DiT-style transformer velocity field (AdaLN) |
| p20 | [consistency-calo](p20-consistency-calo/) | Consistency model, 1-step sampling |

## Installation

```bash
git clone https://github.com/AlphaCortex-03/Gen-Ai--Aug26.git
cd Gen-Ai--Aug26
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Usage

Run a single project:
```bash
cd p6-conditional-cfm
python src/main.py        # or src/train.py depending on project
```

Run all projects sequentially (smoke mode, fast):
```bash
python run_all.py --smoke
```

See `python run_all.py --help` for `--only`, `--skip`, `--stop`, `--list`, and `--timeout` options.

## CI

GitHub Actions smoke-runs all 16 projects on every push and pull request. Results are uploaded as `run_results.json`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to report bugs, suggest features, and submit pull requests.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
