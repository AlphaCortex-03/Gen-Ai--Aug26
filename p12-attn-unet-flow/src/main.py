"""12 - Velocity architecture bake-off: ResMLP vs attention U-Net.

Both nets are trained as flow-matching velocity fields on the same data with
the same step budget.  The U-Net gets conv inductive bias plus a self-attention
bottleneck; the MLP gets speed.  We report params, wall-clock and physics
closure for both.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from calo import pipeline as P
from calo.cli import fig_path, metric_path, setup
from calo.data import get_data, split_data
from calo.nets import AttnUNet, ResMLP
from calo.util import count_params, fmt_table, save_json
from calo.viz import compare_grid, profile_compare, save

import matplotlib.pyplot as plt


def main() -> dict:
    args, cfg = setup(__file__, "p12-attn-unet-flow")
    seed = int(cfg["seed"])
    data = get_data(n_events=int(cfg["n_train"]), seed=seed)
    tr, te = split_data(data, 0.8, seed=seed)
    n_eval = P.n_eval(cfg, te)
    scaler, x1 = P.prep(cfg, tr)

    builders = {
        "resmlp": lambda: ResMLP(dim=256, hidden=int(cfg.get("hidden", 192)), depth=int(cfg.get("depth", 4)), cond_dim=1),
        "attn_unet": lambda: AttnUNet(dim=256, grid=16, base=int(cfg.get("base", 16)), cond_dim=1, heads=4),
    }
    results, samples, histories = {}, {}, {}
    for name, build in builders.items():
        net = build()
        print(f"training {name} | {count_params(net):,} params")
        trainer = P.fit(cfg, net, scaler, tr, x1=x1)
        t0 = time.perf_counter()
        gen = P.sample_images(trainer, te["energy"][:n_eval], steps=int(cfg["sample_steps"]), seed=seed + 3)
        sample_s = time.perf_counter() - t0
        res = P.evaluate(te["images"][:n_eval], gen, seed=seed)
        res.update({"params": count_params(net), "train_seconds": round(trainer.train_seconds, 2),
                    "ms_per_event": round(1000 * sample_s / n_eval, 3)})
        results[name], samples[name], histories[name] = res, gen, trainer.history
        print(f"  AUC {res['c2st_auc']:.3f} | rel {res['mean_rel_err']:.3f} | "
              f"train {res['train_seconds']}s | {res['ms_per_event']:.2f} ms/event")

    real = te["images"][:n_eval]
    metrics = {
        "rows": [{"model": k, **{kk: v[kk] for kk in ("params", "c2st_auc", "mean_rel_err",
                                                      "train_seconds", "ms_per_event", "depth_gen", "width_gen")}}
                 for k, v in results.items()],
    }
    metrics["table"] = fmt_table(metrics["rows"], ["model", "params", "c2st_auc", "mean_rel_err",
                                                   "train_seconds", "ms_per_event"])
    metrics["data_reference"] = {"depth": results["resmlp"]["depth_real"], "width": results["resmlp"]["width_real"]}

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.7))
    for name, hist in histories.items():
        axes[0].plot([h["step"] for h in hist], [h["loss"] for h in hist], label=name)
    axes[0].set(xlabel="step", ylabel="MSE", title="training loss")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=7)

    names = list(results)
    xpos = np.arange(len(names))
    axes[1].bar(xpos, [results[n]["c2st_auc"] for n in names], color="#2c7fb8", width=0.6)
    axes[1].set_xticks(xpos, names, fontsize=7)
    axes[1].set_ylim(0.4, 1.02)
    axes[1].axhline(0.5, color="k", ls=":", lw=1)
    axes[1].set(ylabel="C2ST AUC", title="sample quality")
    save(fig, fig_path(cfg, "training_and_quality"))

    compare_grid(real, samples["resmlp"], fig_path(cfg, "resmlp_samples"), "ResMLP velocity", n=5)
    compare_grid(real, samples["attn_unet"], fig_path(cfg, "unet_samples"), "attention U-Net velocity", n=5)
    profile_compare(real, samples["attn_unet"], fig_path(cfg, "unet_profiles"), "attention U-Net")
    profile_compare(real, samples["resmlp"], fig_path(cfg, "resmlp_profiles"), "ResMLP")

    np.savez_compressed(cfg["outdir"] / "samples.npz",
                        resmlp=samples["resmlp"].astype(np.float32),
                        unet=samples["attn_unet"].astype(np.float32))
    save_json(metric_path(cfg), metrics)
    print(metrics["table"])
    return metrics


if __name__ == "__main__":
    main()
