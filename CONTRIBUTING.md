# Contributing to Gen-Ai--Aug26

Thank you for your interest in contributing! This repo contains 16 CPU-only
generative-model projects on synthetic calorimeter showers (flow matching,
VAE latents, consistency models, quantisation, and benchmarks).

## How to Contribute

1. **Fork** the repository and create your branch from `main`.
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
3. **Run the smoke tests:**
   ```bash
   python run_all.py --smoke
   ```
4. **Make your changes** and ensure all projects still pass.
5. **Commit** with a clear message and **open a pull request**.

## Project Structure

| Folder | Description |
|--------|-------------|
| `p5-calo-demo` … `p20-consistency-calo` | Individual generative-model projects |
| `common/calo` | Shared calorimeter utilities |
| `.github/workflows/ci.yml` | GitHub Actions smoke test for all 16 projects |

Each project folder contains either:
* `src/main.py` — pipeline-style entry point (p5–p14)
* `src/train.py` — self-contained training entry point (p15–p20)

## Reporting Issues

Use the issue templates under `.github/ISSUE_TEMPLATE/` to report bugs or
suggest features. Please include:
* Which project number (e.g., p6, p14)
* Your operating system and Python version
* Steps to reproduce the issue

## License

By contributing, you agree that your contributions will be licensed under the
MIT License (`LICENSE`).
