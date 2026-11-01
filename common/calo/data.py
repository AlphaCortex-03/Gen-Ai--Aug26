"""Toy calorimeter shower simulator (numpy) + dataset plumbing.

Physics model (deliberately simple, but with real structure):

* A shower of energy ``E`` is sampled as ``N`` independent hits.
* Hit depth ``t`` follows a Gamma distribution whose mean grows like
  ``2.6 + 2.4 * log10(E/10)`` (longitudinal shower development).
* Transverse spread grows with depth (``sigma = 0.85 + 0.28 t`` core) and a
  wide halo component is added for ~22% of hits, so the transverse profile is
  multimodal.
* Hit weights are log-normal, so the voxel image is sparse and heavy-tailed.

The image is returned as *fractional* energy (``sum == 1``): the model learns
the shower *shape* conditioned on the total energy ``E``.  All physics
observables (depth, width, multiplicity, sparsity) still depend strongly on
``E``, which is what the conditional models are asked to reproduce.

The optional ``shift``/``stretch``/``extra_core`` knobs produce the
out-of-distribution (BSM-like) showers used by the anomaly-detection project.
"""
from __future__ import annotations

import numpy as np

from .util import CACHE_DIR, hash_params

GRID = 16
PMAX = 256
DATA_VERSION = 2


class Scaler:
    """Variance-stabilising transform to roughly unit variance.

    ``mode="sqrt"`` maps ``x -> (sqrt(x) - mu) / sigma``.  Calorimeter images
    are extremely sparse (most cells are exact zeros); the square-root map
    compresses the heavy tail and shrinks the noise around the zero atom,
    which measurably reduces the bias of clipping negatives after sampling.
    """

    def __init__(self, mu: float, sigma: float, mode: str = "sqrt"):
        self.mu = float(mu)
        self.sigma = float(max(sigma, 1e-8))
        self.mode = mode

    @classmethod
    def fit(cls, images: np.ndarray, mode: str = "sqrt") -> "Scaler":
        x = np.sqrt(np.clip(images, 0, None)) if mode == "sqrt" else images
        return cls(float(x.mean()), float(max(x.std(), 1e-8)), mode)

    def transform(self, images: np.ndarray) -> np.ndarray:
        x = np.sqrt(np.clip(images, 0, None)) if self.mode == "sqrt" else images
        return ((x - self.mu) / self.sigma).astype(np.float32)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        x = z * self.sigma + self.mu
        if self.mode == "sqrt":
            x = np.clip(x, 0, None) ** 2
        return np.clip(x, 0, None).astype(np.float32)

    def state(self) -> dict:
        return {"mu": self.mu, "sigma": self.sigma, "mode": self.mode}

    @classmethod
    def from_state(cls, s: dict) -> "Scaler":
        return cls(s["mu"], s["sigma"], s.get("mode", "sqrt"))


def normalize_images(images: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Clip to non-negative and renormalise each event to sum 1."""
    x = np.clip(images, 0.0, None)
    s = x.reshape(len(x), -1).sum(1, keepdims=True)
    s = np.maximum(s, eps)
    return (x.reshape(len(x), -1) / s).reshape(x.shape)


def _splat(xs, ys, ws, mask, grid):
    """Bilinear splat of hits onto a grid x grid image."""
    b, p = xs.shape
    imgs = np.zeros((b, grid * grid), dtype=np.float64)
    ev = np.repeat(np.arange(b), p).reshape(b, p)

    xc = np.clip(xs, 0.0, grid - 1e-6)
    yc = np.clip(ys, 0.0, grid - 1e-6)
    x0 = np.floor(xc)
    y0 = np.floor(yc)
    fx = xc - x0
    fy = yc - y0

    for dx, dy in ((0, 0), (0, 1), (1, 0), (1, 1)):
        wx = fx if dx else (1.0 - fx)
        wy = fy if dy else (1.0 - fy)
        ix = (x0 + dx).astype(np.int64)
        iy = (y0 + dy).astype(np.int64)
        ok = mask & (ix >= 0) & (ix < grid) & (iy >= 0) & (iy < grid)
        np.add.at(imgs, (ev[ok], (iy * grid + ix)[ok]), (ws * wx * wy)[ok])
    return imgs.reshape(b, grid, grid)


def simulate(
    n_events: int = 8192,
    seed: int = 0,
    grid: int = GRID,
    e_min: float = 10.0,
    e_max: float = 100.0,
    energies: np.ndarray | list | None = None,
    pmax: int = PMAX,
    halo_frac: float = 0.22,
    with_points: bool = False,
    shift: float = 0.0,
    stretch: float = 1.0,
    extra_core: float = 0.0,
) -> dict:
    """Generate ``n_events`` toy showers.

    Returns ``images`` (N,grid,grid) summing to 1, ``energy`` (N,), and - when
    ``with_points`` - ``points`` (N,pmax,4) = (x_norm, y_norm, layer_norm, e)
    plus ``mask`` (N,pmax).
    """
    rng = np.random.default_rng(seed)
    if energies is not None:
        levels = np.asarray(energies, dtype=np.float64).ravel()
        E = levels[rng.integers(0, len(levels), size=n_events)]
        E = E * np.exp(rng.normal(0.0, 0.004, size=n_events))  # tiny jitter
    else:
        logE = rng.uniform(np.log10(e_min), np.log10(e_max), size=n_events)
        E = 10.0 ** logE

    l10 = np.log10(E / 10.0)
    depth_mean = 2.6 + 2.4 * l10 + shift                # rows
    hits_mu = 190.0 + 110.0 * l10                       # ~190 @10, ~300 @100
    n_hits = np.clip(rng.poisson(hits_mu), 8, pmax).astype(np.int64)

    t = rng.gamma(shape=2.0, scale=(depth_mean / 2.0)[:, None], size=(n_events, pmax))
    t = np.clip(t, 0.0, grid - 1.0)

    sig_core = (0.85 + 0.28 * t) * (1.0 + 0.14 * l10[:, None])
    sig_wide = (2.60 + 0.35 * t) * (1.0 + 0.14 * l10[:, None])
    wide = rng.random((n_events, pmax)) < halo_frac
    sigma = np.where(wide, sig_wide, sig_core) * stretch
    if extra_core > 0:
        narrow = rng.random((n_events, pmax)) < extra_core
        sigma = np.where(narrow, 0.25, sigma)

    x = rng.normal(grid / 2.0, sigma)
    w = rng.lognormal(mean=0.0, sigma=0.65, size=(n_events, pmax))
    w *= np.exp(-0.045 * t)                             # longitudinal attenuation

    mask = np.arange(pmax)[None, :] < n_hits[:, None]
    w = w * mask

    images = _splat(x, t, w, mask, grid)
    tot = images.reshape(n_events, -1).sum(1)
    images = images / np.maximum(tot, 1e-12)[:, None, None]

    out = {
        "images": images.astype(np.float32),
        "energy": E.astype(np.float32),
        "meta": {
            "grid": grid, "pmax": pmax, "n_events": int(n_events),
            "shift": shift, "stretch": stretch, "extra_core": extra_core,
        },
    }
    if with_points:
        sw = np.maximum((w * mask).sum(1), 1e-12)
        e_frac = (w / sw[:, None]) * mask
        out["points"] = np.stack(
            [
                np.clip(x / grid, 0.0, 1.0),
                np.clip(t / grid, 0.0, 1.0),
                t / (grid - 1.0),
                e_frac,
            ],
            axis=-1,
        ).astype(np.float32)
        out["mask"] = mask.astype(np.float32)
    return out


def get_data(
    n_events: int = 8192,
    seed: int = 0,
    energies=None,
    e_min: float = 10.0,
    e_max: float = 100.0,
    with_points: bool = False,
    cache: bool = True,
    **sim_kw,
) -> dict:
    """Simulate, with an on-disk cache keyed by the generator parameters."""
    key = hash_params(
        v=DATA_VERSION, n=n_events, seed=seed, e_min=e_min, e_max=e_max,
        energies=None if energies is None else list(np.asarray(energies).ravel()),
        pts=with_points, **sim_kw,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"showers_{key}.npz"
    if cache and path.exists():
        z = np.load(path, allow_pickle=True)
        out = {
            "images": z["images"],
            "energy": z["energy"],
            "meta": z["meta"].item(),
        }
        if "points" in z.files:
            out["points"] = z["points"]
            out["mask"] = z["mask"]
        return out
    data = simulate(
        n_events=n_events, seed=seed, e_min=e_min, e_max=e_max,
        with_points=with_points, **sim_kw,
    )
    if cache:
        np.savez_compressed(path, **{k: v for k, v in data.items() if k != "meta"},
                            meta=np.array(data["meta"], dtype=object))
    return data


def split_data(data: dict, frac: float = 0.8, seed: int = 0) -> tuple[dict, dict]:
    n = len(data["images"])
    idx = np.random.default_rng(seed).permutation(n)
    ntr = int(frac * n)
    keys = ["images", "energy"] + [k for k in ("points", "mask") if k in data]

    def take(ii):
        return {k: data[k][ii] for k in keys}

    return take(idx[:ntr]), take(idx[ntr:])


# ---------------------------------------------------------------------------
# point clouds -> images
# ---------------------------------------------------------------------------
def points_to_images(points: np.ndarray, mask: np.ndarray, grid: int = GRID) -> np.ndarray:
    """Splat point clouds ``(N, P, 4)`` = (x, y, layer, energy) into images."""
    points = np.asarray(points, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    xs = np.clip(points[..., 0] * grid, 0, grid - 1e-6)
    ys = np.clip(points[..., 1] * grid, 0, grid - 1e-6)
    ws = np.clip(points[..., 3], 0, None)
    imgs = _splat(xs, ys, ws, mask, grid)
    return normalize_images(imgs.astype(np.float32))


# ---------------------------------------------------------------------------
# physics observables
# ---------------------------------------------------------------------------
def physics_features(images: np.ndarray) -> dict:
    """Per-event observables used for validation and C2ST features."""
    b, g, _ = images.shape
    frac = images.reshape(b, g, g)
    frac = frac / np.maximum(frac.reshape(b, -1).sum(1), 1e-12)[:, None, None]
    rows = np.arange(g, dtype=np.float64)
    cols = np.arange(g, dtype=np.float64)

    depth_prof = frac.sum(2)                       # (b, g)
    lat_prof = frac.sum(1)                         # (b, g)
    depth = (depth_prof * rows).sum(1)
    depth_rms = np.sqrt(np.maximum((depth_prof * (rows - depth[:, None]) ** 2).sum(1), 0))
    lat = (lat_prof * cols).sum(1)
    width = np.sqrt(np.maximum((lat_prof * (cols - lat[:, None]) ** 2).sum(1), 0))
    flat = frac.reshape(b, -1)
    sparsity = (flat > 1e-3).mean(1)
    n_cells = (flat > 5e-3).sum(1).astype(np.float64)
    peak = flat.max(1)
    return {
        "depth": depth,
        "depth_rms": depth_rms,
        "width": width,
        "sparsity": sparsity,
        "n_cells": n_cells,
        "peak": peak,
    }


def layer_profiles(images: np.ndarray) -> np.ndarray:
    """Mean longitudinal profile, shape (g,)."""
    b = len(images)
    return images.reshape(b, images.shape[1], -1).sum(2).mean(0)


def lateral_profiles(images: np.ndarray) -> np.ndarray:
    b = len(images)
    return images.reshape(b, images.shape[1], -1).sum(1).mean(0)


def summary_vector(images: np.ndarray) -> np.ndarray:
    """Per-event feature vector: 16 depth + 16 lateral bins + 6 observables."""
    f = physics_features(images)
    obs = np.stack(
        [f["depth"], f["depth_rms"], f["width"], f["sparsity"], f["n_cells"], f["peak"]],
        axis=1,
    )
    depth_prof = images.sum(2)   # (b, g) per-event longitudinal bins
    lat_prof = images.sum(1)     # (b, g) per-event lateral bins
    return np.concatenate([depth_prof, lat_prof, obs], axis=1)
