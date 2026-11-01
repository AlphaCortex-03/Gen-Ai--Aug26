"""06 — Conditional Flow Matching on 2D showers, conditioned on energy only.

Baseline for every conditional project that follows: a residual MLP learns the
velocity field ``v(x, t | E)`` on the sqrt-scaled 16x16 image and is integrated
with an Euler ODE solver.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from calo import pipeline as P
from calo.cli import fig_path, metric_path, setup
from calo.data import get_data, split_data
from calo.nets import ResMLP
from calo.util import count_params, save_json
from calo.viz import compare_grid, hist_compare, loss_curve, profile_compare, save

import matplotlib.pyplot as plt

EVAL_ENERGIES = [15.0, 30.0, 50.0, 75.0, 95.0]


def main() -> dict:
    _, cfg = setup(__file__, "project-2-conditional-cfm: energy-conditioned flow matching")
    data = get_data(n_events=int(cfg["n_train"]), seed=int(cfg["seed"]))
    tr, te = split_data(data, 0.8, seed=int(cfg["seed"]))
    n_eval = P.n_eval(cfg, te)

    scaler, x1 = P.prep(cfg, tr)
    net = ResMLP(dim=256, hidden=int(cfg.get("hidden", 192)), depth=int(cfg.get("depth", 4)), cond_dim=1)
    print(f"training {cfg['project']} | {count_params(net):,} params | {len(tr['images'])} events")
    trainer = P.fit(cfg, net, scaler, tr, x1=x1)

    real_e, real_img = te["energy"][:n_eval], te["images"][:n_eval]
    gen = P.sample_images(trainer, real_e, steps=int(cfg["sample_steps"]), seed=1)

    metrics = P.evaluate(real_img, gen, seed=int(cfg["seed"]))
    metrics["n_params"] = count_params(net)
    metrics["train_seconds"] = round(trainer.train_seconds, 2)
    metrics["sample_steps"] = int(cfg["sample_steps"])

    # --- conditioning quality per energy band --------------------------
    bands = [(10, 25), (25, 45), (45, 65), (65, 100)]
    per_band = {}
    for lo, hi in bands:
        m_real = (real_e >= lo) & (real_e < hi)
        if m_real.sum() < 32:
            continue
        n = int(m_real.sum())
        e_band = np.full(n, float(np.mean([lo, hi])))
        g = P.sample_images(trainer, e_band, steps=int(cfg["sample_steps"]), seed=7)
        res = P.evaluate(real_img[m_real], g, seed=0)
        per_band[f"{lo}-{hi} GeV"] = {
            "n": n,
            "c2st_auc": res["c2st_auc"],
            "mean_rel_err": res["mean_rel_err"],
            "depth_gen": res["depth_gen"],
            "width_gen": res["width_gen"],
        }
    metrics["per_band"] = per_band

    # --- figures -------------------------------------------------------
    loss_curve(trainer.history, fig_path(cfg, "loss"), "flow-matching MSE")
    show_e = [15.0, 40.0, 70.0, 95.0]
    gen_show = P.sample_images(trainer, np.array(show_e), steps=int(cfg["sample_steps"]), seed=11)
    real_show = np.stack([real_img[np.argmin(np.abs(real_e - e))] for e in show_e])
    compare_grid(real_show, gen_show, fig_path(cfg, "samples_vs_data"), "top: data, bottom: generated", n=4)

    fig, axes = plt.subplots(1, len(show_e), figsize=(2.0 * len(show_e), 2.2))
    for ax, e in zip(axes, show_e):
        ax.axis("off")
        ax.imshow(gen_show[show_e.index(e)], cmap="inferno")
        ax.set_title(f"E={e:.0f} GeV", fontsize=7)
    save(fig, fig_path(cfg, "generated_by_energy"))

    profile_compare(real_img, gen, fig_path(cfg, "profiles"), "all energies")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    for i, (lo, hi, c) in enumerate([(10, 30, "#2c7fb8"), (60, 100, "#d95f02")]):
        mr = (real_e >= lo) & (real_e < hi)
        mg = (real_e >= lo) & (real_e < hi)
        axes[0].plot(real_img[mr].sum(2).mean(0), "--" if i else "-", color=c, lw=1.4,
                     label=f"data {lo}-{hi} GeV")
        axes[0].plot(gen[mg].sum(2).mean(0), ":", color=c, lw=1.8, label=f"gen {lo}-{hi} GeV")
    axes[0].set(xlabel="layer", ylabel="energy fraction", title="longitudinal profile by energy")
    axes[0].legend(frameon=False, fontsize=6)
    pe = np.array([float(k.split(" ")[0].split("-")[1]) for k in per_band])
    pa = np.array([v["c2st_auc"] for v in per_band.values()])
    axes[1].plot(pe, pa, "o-", color="#1b9e77")
    axes[1].axhline(0.5, color="k", ls=":", lw=1)
    axes[1].set(xlabel="upper edge of E band [GeV]", ylabel="C2ST AUC", title="conditional quality vs energy")
    axes[1].set_ylim(0.45, 1.02)
    save(fig, fig_path(cfg, "conditional_quality"))

    from calo.data import physics_features

    fr, fg = physics_features(real_img), physics_features(gen)
    hist_compare(fr["depth"], fg["depth"], fig_path(cfg, "depth_hist"), "mean depth [layers]", "shower depth")
    hist_compare(fr["width"], fg["width"], fig_path(cfg, "width_hist"), "lateral width [cells]", "shower width")

    np.savez_compressed(
        cfg["outdir"] / "samples.npz",
        images=gen.astype(np.float32), energy=real_e.astype(np.float32),
        real_images=real_img.astype(np.float32),
    )
    save_json(metric_path(cfg), metrics)

    print(f"C2ST AUC {metrics['c2st_auc']:.3f} | mean physics rel. error {metrics['mean_rel_err']:.3f}")
    print(f"depth {metrics['depth_real']:.2f} -> {metrics['depth_gen']:.2f} | "
          f"width {metrics['width_real']:.2f} -> {metrics['width_gen']:.2f}")
    return metrics


if __name__ == "__main__":
    main()
