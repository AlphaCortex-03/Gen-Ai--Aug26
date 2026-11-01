"""Uniform CLI + config handling for every project.

Each ``src/main.py`` starts with::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
    from calo.cli import setup
    args, cfg = setup(__file__, "short description")

which gives it ``--smoke``, ``--seed``, ``--steps``, ``--n``, ``--config``
and a merged config dict with a ready ``cfg["outdir"]``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .util import outputs_dir, seed_all


def setup(main_file: str, description: str) -> tuple[argparse.Namespace, dict]:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--smoke", action="store_true", help="tiny/fast run for CI")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--n", type=int, default=None, help="number of training events")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--tag", type=str, default="", help="suffix for output filenames")
    args = p.parse_args()

    project = Path(main_file).resolve().parents[1]
    cfg_path = Path(args.config) if args.config else project / "configs" / "default.yaml"
    cfg: dict = {}
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.n is not None:
        cfg["n_train"] = args.n
    if args.smoke:
        cfg["steps"] = min(int(cfg.get("steps", 400)), 80)
        cfg["n_train"] = min(int(cfg.get("n_train", 4096)), 512)
        cfg["sample_steps"] = min(int(cfg.get("sample_steps", 50)), 10)
        cfg["n_eval"] = min(int(cfg.get("n_eval", 1024)), 128)
        cfg["smoke"] = True

    cfg.setdefault("seed", 0)
    cfg.setdefault("n_train", 4096)
    cfg.setdefault("steps", 400)
    cfg.setdefault("batch", 128)
    cfg.setdefault("lr", 1e-3)
    cfg.setdefault("sample_steps", 50)
    cfg.setdefault("n_eval", 1024)
    cfg["smoke"] = bool(cfg.get("smoke", False))
    cfg["project"] = project.name
    cfg["tag"] = args.tag
    cfg["outdir"] = Path(args.outdir) if args.outdir else outputs_dir(project.name)
    (cfg["outdir"] / "figures").mkdir(parents=True, exist_ok=True)
    seed_all(cfg["seed"])
    return args, cfg


def fig_path(cfg: dict, name: str) -> Path:
    tag = cfg.get("tag", "")
    if tag:
        name = f"{name}_{tag}"
    return cfg["outdir"] / "figures" / f"{name}.png"


def metric_path(cfg: dict, name: str = "metrics.json") -> Path:
    return cfg["outdir"] / name
