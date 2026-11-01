"""Shared utilities: paths, seeding, config loading, timing, small helpers."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / ".cache"
OUTPUT_ROOT = REPO_ROOT


def seed_all(seed: int = 0) -> None:
    """Seed python, numpy and torch. CPU-only, so determinism is cheap."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def project_dir(project: str) -> Path:
    return REPO_ROOT / project


def figures_dir(project: str) -> Path:
    p = project_dir(project) / "outputs" / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def outputs_dir(project: str) -> Path:
    p = project_dir(project) / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def hash_params(**params) -> str:
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


class Timer:
    """Context manager measuring wall-clock seconds."""

    def __init__(self):
        self.dt = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self._t0
        return False


def count_params(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def human_int(n: int) -> str:
    for unit in ("", "k", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.0f}{unit}" if unit == "" else f"{n:.1f}{unit}"
        n /= 1000.0
    return f"{n:.1f}T"


def fmt_table(rows: list[dict], keys: list[str]) -> str:
    """Render a small markdown table from list-of-dicts."""
    if not rows:
        return ""
    widths = {k: max(len(k), *(len(_cell(r.get(k))) for r in rows)) for k in keys}
    head = "| " + " | ".join(k.ljust(widths[k]) for k in keys) + " |"
    sep = "|" + "|".join("-" * (widths[k] + 2) for k in keys) + "|"
    body = [
        "| " + " | ".join(_cell(r.get(k)).ljust(widths[k]) for k in keys) + " |"
        for r in rows
    ]
    return "\n".join([head, sep, *body])


def _cell(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)
