"""Plotting helpers (Agg backend, deterministic, no GUI needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 130,
    "font.size": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})

C_REAL = "#2c7fb8"
C_GEN = "#d95f02"


def save(fig, path, tight=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def image_grid(images, path, title="", ncols=6, vmax_pct=99.5):
    n = min(len(images), 24)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    vmax = float(np.percentile(images, vmax_pct)) if vmax_pct else float(images.max())
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.45 * ncols, 1.45 * nrows + 0.4))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i], cmap="inferno", vmin=0, vmax=max(vmax, 1e-9), aspect="auto")
    fig.suptitle(title, fontsize=9)
    return save(fig, path)


def compare_grid(real, gen, path, title="", n=6):
    n = min(n, len(real), len(gen))
    fig, axes = plt.subplots(2, n, figsize=(1.35 * n, 2.9))
    axes = np.atleast_2d(axes)
    vmax = max(float(np.percentile(real, 99.5)), 1e-9)
    for i in range(n):
        axes[0, i].imshow(real[i], cmap="inferno", vmin=0, vmax=vmax)
        axes[1, i].imshow(gen[i], cmap="inferno", vmin=0, vmax=vmax)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("data", fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.text(0.08, 0.47, "data", rotation=90, va="center", fontsize=8)
    fig.text(0.08, 0.16, "generated", rotation=90, va="center", fontsize=8)
    return save(fig, path)


def profile_compare(real, gen, path, title=""):
    g = real.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    layers = np.arange(g)
    axes[0].plot(layers, real.sum(2).mean(0), "o-", ms=3, color=C_REAL, label="data")
    axes[0].plot(layers, gen.sum(2).mean(0), "s--", ms=3, color=C_GEN, label="generated")
    axes[0].set_xlabel("layer (depth)")
    axes[0].set_ylabel("mean energy fraction")
    axes[0].set_title("longitudinal profile")
    axes[0].legend(frameon=False, fontsize=7)

    axes[1].plot(layers, real.sum(1).mean(0), "o-", ms=3, color=C_REAL, label="data")
    axes[1].plot(layers, gen.sum(1).mean(0), "s--", ms=3, color=C_GEN, label="generated")
    axes[1].set_xlabel("cell (lateral)")
    axes[1].set_title("lateral profile")
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=9)
    return save(fig, path)


def hist_compare(a, b, path, xlabel="", title="", bins=40, labels=("data", "generated")):
    fig, ax = plt.subplots(figsize=(3.9, 2.7))
    lo = min(np.min(a), np.min(b))
    hi = max(np.max(a), np.max(b))
    bins = np.linspace(lo, hi, bins)
    ax.hist(a, bins=bins, density=True, histtype="step", lw=1.4, color=C_REAL, label=labels[0])
    ax.hist(b, bins=bins, density=True, histtype="step", lw=1.4, color=C_GEN, label=labels[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title, fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    return save(fig, path)


def loss_curve(history, path, title="training loss"):
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    steps = [h["step"] for h in history]
    loss = [h["loss"] for h in history]
    ax.plot(steps, loss, color=C_REAL, lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.set_title(title, fontsize=8)
    return save(fig, path)


def line_plot(xs, series: dict, path, xlabel="", ylabel="", title="", logx=False, logy=False):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for name, (ys, style) in series.items():
        ax.plot(xs, ys, style, label=name, lw=1.3, ms=3.5)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    return save(fig, path)


def scatter3d(points, path, title="", max_events=3, colors=None):
    """points: (P, 4) or list of clouds, columns (x, y, layer, e)."""
    clouds = points if isinstance(points, (list, tuple)) else [points]
    fig = plt.figure(figsize=(6.2, 3.2))
    for k, p in enumerate(clouds[:max_events]):
        ax = fig.add_subplot(1, len(clouds[:max_events]), k + 1, projection="3d")
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[1] >= 4:
            m = p[:, 3] > 0
            ax.scatter(p[m, 0], p[m, 1], p[m, 2], s=2 + 40 * p[m, 3] / (p[m, 3].max() + 1e-9),
                       c=np.log10(p[m, 3] + 1e-6), cmap="inferno", alpha=0.85)
        ax.set_xlabel("x", fontsize=7)
        ax.set_ylabel("y", fontsize=7)
        ax.set_zlabel("layer", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(f"event {k}", fontsize=7)
    fig.suptitle(title, fontsize=8)
    return save(fig, path)


def bar_plot(names, values, path, ylabel="", title="", color=C_REAL, yerr=None, hline=None):
    fig, ax = plt.subplots(figsize=(max(3.6, 0.55 * len(names) + 1.6), 2.8))
    ax.bar(names, values, color=color, yerr=yerr, capsize=3)
    if hline is not None:
        ax.axhline(hline, color="k", ls=":", lw=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    return save(fig, path)


def heatmap(mat, path, title="", xlabel="", ylabel="", cbar_label="", cmap="viridis"):
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    im = ax.imshow(mat, aspect="auto", cmap=cmap)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label, fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    return save(fig, path)
