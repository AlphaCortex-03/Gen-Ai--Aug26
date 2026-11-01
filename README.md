# August Practice - Projects 5 to 20

[![CI](https://github.com/AlphaCortex-03/Gen-Ai--Aug26/actions/workflows/ci.yml/badge.svg)](https://github.com/AlphaCortex-03/Gen-Ai--Aug26/actions/workflows/ci.yml)

| No | Project | What I did |
|----|---------|------------|
| p5 | [calo-demo](p5-calo-demo/) | NumPy calorimeter shower simulator: profiles, observables, event displays |
| p6 | [conditional-cfm](p6-conditional-cfm/) | Conditional flow matching on beam energy, C2ST AUC 0.998 |
| p7 | [energy-interpolation](p7-energy-interpolation/) | Sample unseen beam energies (30/70 GeV) from a flow trained on 10/50/100 GeV |
| p8 | [flow-distillation](p8-flow-distillation/) | Distilled 50-step flow sampler into a 1-step student, 45x faster |
| p9 | [pointcloud-cfm](p9-pointcloud-cfm/) | Point-cloud shower generation with a DeepSets velocity net (41k params) |
| p10 | [c2st-test](p10-c2st-test/) | Classifier two-sample test: physics features vs raw pixels |
| p11 | [ensemble-flow](p11-ensemble-flow/) | 3-member flow ensemble with epistemic uncertainty maps |
| p12 | [attn-unet-flow](p12-attn-unet-flow/) | Attention U-Net vs ResMLP velocity field comparison |
| p13 | [layer-autoregressive](p13-layer-autoregressive/) | Layer-by-layer autoregressive shower generation (4 stages) |
| p14 | [quantized-flow](p14-quantized-flow/) | INT8 quantisation + pruning: 3.8x smaller with ~zero AUC loss |
| p15 | [benchmark-3-gens](p15-benchmark-3-gens/) | WGAN-GP vs DDPM vs CFM on identical data |
| p16 | [anomaly-flow](p16-anomaly-flow/) | Flow log-prob as anomaly score (negative result, AUC 0.486) |
| p17 | [latent-cfm](p17-latent-cfm/) | VAE latent space + flow matching in latent space |
| p18 | [cfg-calo](p18-cfg-calo/) | Classifier-free guidance for energy control (w sweep 0-7) |
| p19 | [transformer-cfm](p19-transformer-cfm/) | DiT-style transformer velocity field (AdaLN) |
| p20 | [consistency-calo](p20-consistency-calo/) | Consistency model, 1-step sampling |

Everything runs on CPU (numpy / PyTorch / matplotlib):
- `cd p5-calo-demo && python src/main.py` (pipeline projects: also `python src/run.py`)
- `cd p15-benchmark-3-gens && python src/train.py`
- `python run_all.py --smoke` runs all 16 sequentially (GitHub Actions badge above checks on every push)

Install: `pip install -r requirements.txt`
