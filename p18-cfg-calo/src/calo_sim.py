"""
calo_sim.py -- synthetic 2D calorimeter shower generator (numpy only).

Shared by all projects in gen-ai-practice/aug. Each project keeps its own
copy of this file so every project folder is self-contained.

A "shower" is a normalized 16x16 energy-deposit map (cells sum to 1).
The total beam energy (10..100 GeV) is a separate conditioning value.
"""
import numpy as np

GRID = 16
ALLOWED_ENERGIES = (10.0, 30.0, 50.0, 70.0, 100.0)


def sample_shower(rng, energy):
    """One nominal shower: normalized (sum = 1) 16x16 deposit map."""
    xx, yy = np.mgrid[0:GRID, 0:GRID].astype(np.float32)
    cx, cy = rng.uniform(5.0, GRID - 5.0, size=2).astype(np.float32)
    sigma = (0.7 + 0.012 * energy) * rng.uniform(0.85, 1.15)
    img = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2)))
    # secondary EM sub-blob
    sx, sy = cx + rng.uniform(-2, 2), cy + rng.uniform(-2, 2)
    img = img + 0.35 * np.exp(-(((xx - sx) ** 2 + (yy - sy) ** 2) / (2.0 * (0.6 * sigma) ** 2)))
    # electronics noise
    img = img + rng.normal(0.0, 0.01, img.shape).astype(np.float32)
    img = np.clip(img, 0.0, None)
    s = img.sum()
    if s <= 0:
        img = np.ones_like(img)
        s = img.sum()
    return (img / s).astype(np.float32)


def bsm_shower(rng, energy):
    """Anomalous / BSM-like shower: two separated cores, very wide profile."""
    xx, yy = np.mgrid[0:GRID, 0:GRID].astype(np.float32)
    img = (np.exp(-(((xx - 3.5) ** 2 + (yy - 3.5) ** 2) / 18.0))
           + np.exp(-(((xx - GRID - 2.5) ** 2 + (yy - GRID - 2.5) ** 2) / 18.0)))
    img = img + 0.4 * np.exp(-(((xx - 7.5) ** 2 + (yy - 7.5) ** 2) / 8.0))
    img = img + rng.normal(0.0, 0.01, img.shape).astype(np.float32)
    img = np.clip(img, 0.0, None)
    s = img.sum()
    if s <= 0:
        img = np.ones_like(img)
        s = img.sum()
    return (img / s).astype(np.float32)


def dataset(n, energies, seed=0):
    """Returns (images (n,1,16,16) float32, energies (n,) float32)."""
    rng = np.random.default_rng(seed)
    e = np.asarray(energies, dtype=np.float32)
    pick = e[rng.integers(0, len(e), size=n)]
    imgs = np.stack([sample_shower(rng, float(en)) for en in pick])[:, None, :, :]
    return imgs.astype(np.float32), pick.astype(np.float32)


def cond_of(energies):
    """Conditioning value: log10(E / 10 GeV), maps 10 GeV -> 0, 100 GeV -> 1."""
    return np.log10(np.asarray(energies, np.float32) / 10.0).astype(np.float32)
