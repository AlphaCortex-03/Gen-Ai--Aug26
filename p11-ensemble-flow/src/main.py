"""11 - Deep ensemble of flows: where is the model uncertain?

Three independently initialised flows are trained on the same data.  Their
disagreement at fixed energy is an *epistemic* uncertainty estimate; the mean
within-member spread is the *aleatoric* part.  We map both and check whether
the ensemble's error bars actually cover the data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from calo import pipeline as P
from calo.cli import fig_path, metric_path, setup
from calo.data import get_data, split_data, summary_vector
from calo.metrics import c2st_auc, coverage
from calo.nets import ResMLP
from calo.util import count_params, fmt_table, save_json
from calo.viz import compare_grid, save

import matplotlib.pyplot as plt


def main() -> dict:
    args, cfg = setup(__file__, "p11-ensemble-flow")
    seed = int(cfg["seed"])
    data = get_data(n_events=int(cfg["n_train"]), seed=seed)
    tr, te = split_data(data, 0.8, seed=seed)
    n_members = int(cfg.get("members", 3))
    n_eval = P.n_eval(cfg, te)
    e_fixed = float(cfg.get("energy", 50.0))

    scaler, x1 = P.prep(cfg, tr)
    members, member_imgs = [], []
    for m in range(n_members):
        net = ResMLP(dim=256, hidden=int(cfg.get("hidden", 160)), depth=int(cfg.get("depth", 4)), cond_dim=1)
        tr_m = P.fit(cfg, net, scaler, tr, x1=x1, verbose=(m == 0))
        members.append(tr_m)
        member_imgs.append(P.sample_images(tr_m, np.full(n_eval, e_fixed),
                                           steps=int(cfg["sample_steps"]), seed=seed + 100 + m))
        print(f"  member {m}: {count_params(net):,} params")

    stack = np.stack(member_imgs)                       # (M, N, 16, 16)
    member_mean = stack.mean(1)                         # (M, 16, 16)
    ens_mean = member_mean.mean(0)                      # (16, 16)
    ens_std = member_mean.std(0, ddof=1)                # epistemic
    aleatoric = stack.std(1).mean(0)                    # within-member event spread
    rel_std = ens_std / np.maximum(ens_mean, 1e-9)

    # held-out data at the same energy
    m = np.abs(te["energy"] - e_fixed) < 6.0
    real = te["images"][m][:n_eval]
    if len(real) < 64:
        real = te["images"][:n_eval]
    data_mean = real.mean(0)

    cov = coverage(ens_mean, ens_std, data_mean)
    aucs = [c2st_auc(summary_vector(real), summary_vector(im), seed=seed) for im in member_imgs]
    ens_auc = float(np.mean(aucs))
    metrics = {
        "energy": e_fixed,
        "n_members": n_members,
        "n_events": n_eval,
        "member_auc": aucs,
        "auc_mean": ens_auc,
        "auc_spread": float(np.std(aucs)),
        "mean_epistemic_std": float(ens_std.mean()),
        "mean_aleatoric_std": float(aleatoric.mean()),
        "mean_rel_std": float(rel_std.mean()),
        "max_rel_std": float(rel_std.max()),
        "pixel_corr_ens_vs_data": float(np.corrcoef(ens_mean.ravel(), data_mean.ravel())[0, 1]),
        "mean_abs_err_ens_vs_data": float(np.abs(ens_mean - data_mean).mean()),
        "n_params_per_member": count_params(members[0].net),
    }
    metrics.update(cov)
    print(f"ensemble AUC {ens_auc:.3f} +/- {metrics['auc_spread']:.3f} | "
          f"epistemic std {metrics['mean_epistemic_std']:.2e} | coverage 1sigma {cov['cov_1sigma']:.2f}")

    # ---------------- figures -----------------------------------------
    fig, axes = plt.subplots(2, n_members + 2, figsize=(2.0 * (n_members + 2), 3.9))
    vmax = float(np.percentile(np.stack([*member_mean, data_mean]), 99.5))
    for i, im in enumerate(member_mean):
        axes[0, i].imshow(im, cmap="inferno", vmin=0, vmax=vmax)
        axes[0, i].set_title(f"member {i}", fontsize=7)
    axes[0, n_members].imshow(ens_mean, cmap="inferno", vmin=0, vmax=vmax)
    axes[0, n_members].set_title("ensemble mean", fontsize=7)
    axes[0, n_members + 1].imshow(data_mean, cmap="inferno", vmin=0, vmax=vmax)
    axes[0, n_members + 1].set_title("data", fontsize=7)
    im = axes[1, 0].imshow(ens_std, cmap="magma")
    axes[1, 0].set_title("epistemic std", fontsize=7)
    axes[1, 1].imshow(aleatoric, cmap="magma")
    axes[1, 1].set_title("aleatoric std", fontsize=7)
    for i in range(2, n_members + 2):
        axes[1, i].axis("off")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[1, :2], fraction=0.03, label="std of mean image")
    fig.suptitle(f"ensemble at E = {e_fixed:.0f} GeV", fontsize=9)
    save(fig, fig_path(cfg, "ensemble_maps"))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    axes[0].imshow(ens_std, cmap="magma")
    axes[0].set_title("epistemic uncertainty map", fontsize=8)
    axes[0].set_xticks([]) , axes[0].set_yticks([])
    im2 = axes[1].imshow(rel_std, cmap="viridis", vmax=float(np.percentile(rel_std, 99)))
    axes[1].set_title("relative uncertainty (std/mean)", fontsize=8)
    axes[1].set_xticks([]), axes[1].set_yticks([])
    fig.colorbar(im2, ax=axes[1], fraction=0.045)
    save(fig, fig_path(cfg, "relative_uncertainty"))

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.scatter(data_mean.ravel(), ens_mean.ravel(), s=4, alpha=0.6)
    lim = [0, max(data_mean.max(), ens_mean.max())]
    ax.plot(lim, lim, "k:", lw=1)
    ax.set(xlabel="data mean image", ylabel="ensemble mean image", title=f"pixel closure (r={metrics['pixel_corr_ens_vs_data']:.3f})")
    save(fig, fig_path(cfg, "pixel_closure"))

    compare_grid(real, member_imgs[0], fig_path(cfg, "member_vs_data"), "member 0 vs data", n=6)
    save_json(metric_path(cfg), metrics)
    print(fmt_table([{"member": i, "auc": a, "params": metrics["n_params_per_member"]}
                     for i, a in enumerate(aucs)], ["member", "auc", "params"]))
    return metrics


if __name__ == "__main__":
    main()
