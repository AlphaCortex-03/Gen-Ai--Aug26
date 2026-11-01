"""Small pipeline helpers that several projects share.

Keeps each ``src/main.py`` focused on *its* idea instead of boilerplate.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .cfm import CFMTrainer
from .data import Scaler, summary_vector
from .metrics import c2st_auc, compare_observables


def prep(cfg: dict, data: dict, mode: str = "sqrt"):
    """Fit the scaler on the training set and return (scaler, flat tensor)."""
    scaler = Scaler.fit(data["images"], mode=mode)
    z = scaler.transform(data["images"]).reshape(len(data["images"]), -1)
    return scaler, torch.as_tensor(z, dtype=torch.float32)


def fit(
    cfg: dict,
    net: torch.nn.Module,
    scaler: Scaler,
    data: dict,
    cond=None,
    cond_dropout: float = 0.0,
    x1=None,
    verbose: bool = True,
    lr: float | None = None,
) -> CFMTrainer:
    """Train a CFM velocity network on ``data`` conditioned on its energy."""
    if cond is None:
        cond = data["energy"]
    trainer = CFMTrainer(
        net, scaler,
        cond_mean=float(np.mean(data["energy"])),
        cond_std=float(np.std(data["energy"])),
        lr=float(lr if lr is not None else cfg.get("lr", 8e-4)),
        cond_dropout=cond_dropout,
        ema_decay=cfg.get("ema_decay", 0.999),
    )
    if x1 is None:
        x1 = scaler.transform(data["images"]).reshape(len(data["images"]), -1)
    trainer.train(
        x1, cond,
        steps=int(cfg["steps"]),
        batch=int(cfg["batch"]),
        log_every=max(1, int(cfg["steps"]) // 4),
        verbose=verbose,
    )
    return trainer


def sample_images(
    trainer: CFMTrainer,
    energies: np.ndarray,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    net=None,
) -> np.ndarray:
    """Sample and decode to physical (N, 16, 16) images."""
    if seed is not None:
        torch.manual_seed(seed)
    steps = steps if steps is not None else 32
    z = trainer.sample(len(energies), energies, steps=steps, guidance=guidance, net=net)
    return trainer.decode(z)


def evaluate(real: np.ndarray, gen: np.ndarray, seed: int = 0, iters: int = 300) -> dict:
    """C2ST AUC plus physics closure, computed identically for every project."""
    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    n = min(len(real), len(gen))
    out = {"c2st_auc": c2st_auc(summary_vector(real[:n]), summary_vector(gen[:n]), seed=seed, iters=iters)}
    out.update(compare_observables(real[:n], gen[:n]))
    return out


def timed_sample(fn, *args, **kwargs) -> tuple[object, float]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def n_eval(cfg: dict, test: dict) -> int:
    """Never ask for more test events than the split actually has."""
    return int(min(int(cfg.get("n_eval", 1024)), len(test["images"])))
