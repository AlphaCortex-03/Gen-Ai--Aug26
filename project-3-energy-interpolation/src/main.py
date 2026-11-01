"""07 — Energy interpolation: train on 10/50/100 GeV, test on 30/70 GeV.

Question: does a continuous energy conditioning learn the *shape* of the
energy dependence, or does it just memorise the training energies?  We compare
against a "binning" baseline that snaps unseen energies to the nearest training
point, which is what a discrete conditioning scheme would do.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from calo import pipeline as P
from calo.cli import fig_path, metric_path, setup
from calo.data import get_data
from calo.nets import ResMLP
from calo.util import count_params, fmt_table, save_json
from calo.viz import compare_grid, hist_compare, loss_curve, save

import matplotlib.pyplot as plt

TRAIN_E = [10.0, 50.0, 100.0]
TEST_E = [10.0, 30.0, 50.0, 70.0, 100.0]


def main() -> dict:
    args, cfg = setup(__file__, "project-3-energy-interpolation")
    seed = int(cfg["seed"])
    train = get_data(n_events=int(cfg["n_train"]), seed=seed, energies=TRAIN_E)
    test = get_data(n_events=2500, seed=seed + 5, energies=[30.0, 70.0] + TRAIN_E)

    scaler, x1 = P.prep(cfg, train)
    net = ResMLP(dim=256, hidden=int(cfg.get("hidden", 192)), depth=int(cfg.get("depth", 4)), cond_dim=1)
    print(f"training on E in {TRAIN_E} | {count_params(net):,} params")
    trainer = P.fit(cfg, net, scaler, train, x1=x1)
    steps = int(cfg["sample_steps"])

    n_per = int(cfg.get("n_per_energy", 256))
    metrics, rows = {}, []
    gen_at = {}
    for e in TEST_E:
        gen = P.sample_images(trainer, np.full(n_per, e), steps=steps, seed=int(e))
        gen_at[e] = gen
        m = (np.abs(test["energy"] - e) < 2.0)
        real = test["images"][m][:n_per]
        if len(real) < 32:
            continue
        res = P.evaluate(real, gen, seed=seed)
        # nearest-training-energy baseline (what discrete binning would give)
        near = min(TRAIN_E, key=lambda t: abs(t - e))
        base = gen_at.get(near)
        if base is None:
            base = P.sample_images(trainer, np.full(len(real), near), steps=steps, seed=int(near) + 1)
        bres = P.evaluate(real, base[: len(real)], seed=seed)
        rows.append({
            "E": e, "seen": e in TRAIN_E,
            "auc": res["c2st_auc"], "rel": res["mean_rel_err"],
            "depth": res["depth_gen"], "depth_data": res["depth_real"],
            "width": res["width_gen"], "width_data": res["width_real"],
            "auc_snap": bres["c2st_auc"], "rel_snap": bres["mean_rel_err"],
            "near": near,
        })
        metrics[f"E{e:g}"] = rows[-1]
        print(f"  E={e:5.0f} GeV  AUC {res['c2st_auc']:.3f}  rel {res['mean_rel_err']:.3f}"
              f"  depth {res['depth_real']:.2f}/{res['depth_gen']:.2f}  (snap->{near:g}: AUC {bres['c2st_auc']:.3f})")

    seen = [r for r in rows if r["seen"]]
    unseen = [r for r in rows if not r["seen"]]
    metrics["summary"] = {
        "seen_auc": float(np.mean([r["auc"] for r in seen])),
        "unseen_auc": float(np.mean([r["auc"] for r in unseen])),
        "seen_rel": float(np.mean([r["rel"] for r in seen])),
        "unseen_rel": float(np.mean([r["rel"] for r in unseen])),
        "snap_unseen_auc": float(np.mean([r["auc_snap"] for r in unseen])),
        "snap_unseen_rel": float(np.mean([r["rel_snap"] for r in unseen])),
        "n_params": count_params(net),
        "train_seconds": round(trainer.train_seconds, 2),
    }
    metrics["table"] = fmt_table(rows, ["E", "seen", "auc", "rel", "auc_snap", "rel_snap"])

    # --- figures -------------------------------------------------------
    loss_curve(trainer.history, fig_path(cfg, "loss"))

    # smooth energy scan: measured depth vs requested energy
    scan_e = np.arange(10.0, 100.1, 7.5)
    depths, widths = [], []
    for e in scan_e:
        g = P.sample_images(trainer, np.full(256, e), steps=steps, seed=int(e * 7))
        from calo.data import physics_features

        f = physics_features(g)
        depths.append(f["depth"].mean())
        widths.append(f["width"].mean())
    from calo.data import physics_features

    fr = physics_features(test["images"])
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    axes[0].scatter(test["energy"], fr["depth"], s=2, alpha=0.2, color="#cccccc", label="test data")
    axes[0].plot(scan_e, depths, "o-", color="#d95f02", ms=4, label="generated")
    axes[0].axvline(30, color="k", ls=":", lw=1)
    axes[0].axvline(70, color="k", ls=":", lw=1)
    axes[0].set(xlabel="E [GeV]", ylabel="mean depth [layers]", title="interpolation of shower depth")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].plot(scan_e, widths, "s-", color="#2c7fb8", ms=4, label="generated")
    axes[1].scatter(test["energy"], fr["width"], s=2, alpha=0.2, color="#cccccc", label="test data")
    axes[1].axvline(30, color="k", ls=":", lw=1)
    axes[1].axvline(70, color="k", ls=":", lw=1)
    axes[1].set(xlabel="E [GeV]", ylabel="lateral width [cells]", title="interpolation of shower width")
    axes[1].legend(frameon=False, fontsize=7)
    save(fig, fig_path(cfg, "energy_scan"))

    compare_grid(gen_at[30.0], gen_at[70.0], fig_path(cfg, "unseen_energies"), "unseen energies", n=4)
    hist_compare(fr["depth"][np.abs(test["energy"] - 30) < 2], physics_features(gen_at[30.0])["depth"],
                 fig_path(cfg, "depth_at_30"), "mean depth [layers]", "interpolated 30 GeV")

    save_json(metric_path(cfg), metrics)
    print(metrics["table"])
    return metrics


if __name__ == "__main__":
    main()
