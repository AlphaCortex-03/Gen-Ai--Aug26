"""05 - 2D calorimeter shower visualisation with numpy only.

No torch, no training: this project just generates toy showers and turns them
into the diagnostic figures every later project is judged against.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from calo.cli import fig_path, metric_path, setup
from calo.data import get_data, lateral_profiles, layer_profiles, physics_features
from calo.util import save_json
from calo.viz import image_grid, save

BINS = [(10, 25, "10-25 GeV"), (25, 55, "25-55 GeV"), (55, 100, "55-100 GeV")]


def ascii_event(img: np.ndarray, width: int = 16) -> str:
    """Render one event as text, so the demo works even in a bare terminal."""
    chars = " .:-=+*#%@"
    v = img / (img.max() + 1e-12)
    v = np.clip(v, 0, 1)
    rows = []
    for r in range(img.shape[0]):
        line = "".join(chars[min(len(chars) - 1, int(v[r, c] * (len(chars) - 1)))] for c in range(width))
        rows.append(f"|{line}|")
    return "\n".join(rows)


def main() -> dict:
    args, cfg = setup(__file__, "p5-calo-demo: numpy shower visualisation")
    data = get_data(n_events=int(cfg.get("n_events", 8192)), seed=int(cfg["seed"]))
    imgs, E = data["images"], data["energy"]
    print(f"generated {len(imgs)} showers, E in [{E.min():.1f}, {E.max():.1f}] GeV")
    print("one event:")
    print(ascii_event(imgs[0]))

    image_grid(imgs[:12], fig_path(cfg, "event_display"), "12 individual showers (16x16 cells)", ncols=6)

    # mean image per energy slice
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.9))
    vmax = float(np.percentile(imgs.mean(0), 99.5))
    panels = [("all energies", imgs.mean(0))]
    for lo, hi, label in BINS:
        m = (E >= lo) & (E < hi)
        panels.append((f"{label} (n={m.sum()})", imgs[m].mean(0)))
    for ax, (title, img) in zip(axes, panels):
        im = ax.imshow(img, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    cax = fig.add_axes([0.28, 0.04, 0.44, 0.03])
    fig.colorbar(im, cax=cax, orientation="horizontal", label="mean energy fraction")
    save(fig, fig_path(cfg, "mean_images"))

    # longitudinal / lateral profiles per energy slice
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(BINS)))
    for (lo, hi, label), col in zip(BINS, colors):
        m = (E >= lo) & (E < hi)
        axes[0].plot(layer_profiles(imgs[m]), "o-", ms=2.5, color=col, label=label)
        axes[1].plot(lateral_profiles(imgs[m]), "o-", ms=2.5, color=col, label=label)
    axes[0].set_xlabel("layer (depth)")
    axes[0].set_ylabel("mean energy fraction")
    axes[0].set_title("longitudinal shower profile")
    axes[1].set_xlabel("cell (lateral)")
    axes[1].set_title("lateral shower profile")
    for ax in axes:
        ax.legend(frameon=False, fontsize=6.5)
    save(fig, fig_path(cfg, "profiles_by_energy"))

    # observables vs energy
    f = physics_features(imgs)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))
    axes[0, 0].scatter(E, f["depth"], s=3, alpha=0.25, color="#2c7fb8")
    axes[0, 0].set(xlabel="E [GeV]", ylabel="mean depth [layers]", title="shower depth vs energy")
    axes[0, 1].scatter(E, f["width"], s=3, alpha=0.25, color="#d95f02")
    axes[0, 1].set(xlabel="E [GeV]", ylabel="lateral width [cells]", title="shower width vs energy")
    
    axes[1, 0].hist(E, bins=40, color="#7570b3")
    axes[1, 0].set(xlabel="E [GeV]", ylabel="events", title="energy spectrum (log-uniform)")
    axes[1, 1].hist(np.log10(f["n_cells"]), bins=40, color="#1b9e77")
    axes[1, 1].set(xlabel="log10 cells above 0.5% energy", ylabel="events", title="shower multiplicity")
    save(fig, fig_path(cfg, "observables"))

    # low vs high energy difference image
    lo_m = E < 30
    hi_m = E > 70
    diff = imgs[hi_m].mean(0) - imgs[lo_m].mean(0)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    lim = float(np.abs(diff).max())
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_title("mean(>70 GeV) - mean(<30 GeV)", fontsize=8)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.045, label="energy fraction difference")
    save(fig, fig_path(cfg, "energy_difference"))

    stats = {"n_events": int(len(imgs)), "energy_min": float(E.min()), "energy_max": float(E.max())}
    for lo, hi, label in BINS:
        m = (E >= lo) & (E < hi)
        stats[label] = {
            "n": int(m.sum()),
            "depth": float(f["depth"][m].mean()),
            "depth_rms": float(f["depth_rms"][m].mean()),
            "width": float(f["width"][m].mean()),
            "n_cells": float(f["n_cells"][m].mean()),
            "sparsity": float(f["sparsity"][m].mean()),
            "peak": float(f["peak"][m].mean()),
        }
    stats["all"] = {
        "depth": float(f["depth"].mean()),
        "width": float(f["width"].mean()),
        "n_cells": float(f["n_cells"].mean()),
        "sparsity": float(f["sparsity"].mean()),
    }
    save_json(metric_path(cfg), stats)
    print(f"mean depth {stats['all']['depth']:.2f} layers, width {stats['all']['width']:.2f} cells, "
          f"{stats['all']['n_cells']:.0f} cells/event")
    print(f"wrote figures to {cfg['outdir'] / 'figures'}")
    return stats


if __name__ == "__main__":
    main()
