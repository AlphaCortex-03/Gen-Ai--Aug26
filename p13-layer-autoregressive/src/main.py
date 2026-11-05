"""13 - Layer-wise autoregressive generation.

The 16 layers are grouped into 4 blocks of 4 layers.  A separate flow models
each block conditioned on the energy and on everything generated before it, so
sampling is sequential: block 0 -> block 1 -> block 2 -> block 3.  Compared
against a single joint flow over all 256 cells.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from calo import pipeline as P
from calo.cfm import CFMTrainer
from calo.cli import fig_path, metric_path, setup
from calo.data import get_data, split_data
from calo.nets import ResMLP
from calo.util import count_params, fmt_table, save_json
from calo.viz import compare_grid, loss_curve, profile_compare, save

import matplotlib.pyplot as plt

N_STAGES = 4
BLOCK = 4          # rows per stage
W = 16             # cells per row


class Std:
    def __init__(self, mu, sd):
        self.mu, self.sd = mu, np.maximum(sd, 1e-6)

    def __call__(self, x):
        return (x - self.mu) / self.sd


def main() -> dict:
    args, cfg = setup(__file__, "p13-layer-autoregressive")
    seed = int(cfg["seed"])
    data = get_data(n_events=int(cfg["n_train"]), seed=seed)
    tr, te = split_data(data, 0.8, seed=seed)
    n_eval = P.n_eval(cfg, te)
    scaler, x1 = P.prep(cfg, tr)
    x1n = x1.numpy()
    E_tr = tr["energy"][:, None].astype(np.float32)
    steps = int(cfg["steps"])

    def make_cond(E, z_prev_cols):
        parts = [E]
        if z_prev_cols is not None:
            parts.append(z_prev_cols)
        return np.concatenate(parts, axis=1).astype(np.float32)

    stages, stds = [], []
    for k in range(N_STAGES):
        sl = slice(k * BLOCK * W, (k + 1) * BLOCK * W)
        cond = make_cond(E_tr, x1n[:, : k * BLOCK * W] if k else None)
        std = Std(cond.mean(0), cond.std(0))
        net = ResMLP(dim=BLOCK * W, hidden=int(cfg.get("hidden", 128)),
                     depth=int(cfg.get("depth", 3)), cond_dim=cond.shape[1])
        trainer = CFMTrainer(net, scaler, 0.0, 1.0, lr=float(cfg["lr"]))
        trainer.train(x1n[:, sl], std(cond),
                      steps=steps, batch=int(cfg["batch"]),
                      log_every=max(1, steps // 3), verbose=(k == 0))
        stages.append(trainer)
        stds.append(std)
        print(f"  stage {k}: rows {k*BLOCK}-{(k+1)*BLOCK-1}, {count_params(net):,} params")

    # ---------------- sequential sampling -----------------------------
    E_te = te["energy"][:n_eval, None].astype(np.float32)
    z_gen = np.zeros((n_eval, 256), dtype=np.float32)
    stage_times = []
    torch.manual_seed(seed + 5)
    for k in range(N_STAGES):
        cond = make_cond(E_te, z_gen[:, : k * BLOCK * W] if k else None)
        t0 = time.perf_counter()
        zk = stages[k].sample(n_eval, stds[k](cond), steps=int(cfg["sample_steps"]), net=stages[k].ema)
        stage_times.append(time.perf_counter() - t0)
        z_gen[:, k * BLOCK * W:(k + 1) * BLOCK * W] = zk.numpy()

    ar_imgs = scaler.inverse(z_gen)
    ar_imgs = ar_imgs / np.maximum(ar_imgs.reshape(n_eval, -1).sum(1, keepdims=True), 1e-12)
    ar_imgs = ar_imgs.reshape(n_eval, 16, 16)

    # ---------------- joint baseline ----------------------------------
    joint_net = ResMLP(dim=256, hidden=int(cfg.get("joint_hidden", 192)),
                       depth=int(cfg.get("joint_depth", 4)), cond_dim=1)
    joint = P.fit(cfg, joint_net, scaler, tr, x1=x1)
    t0 = time.perf_counter()
    joint_imgs = P.sample_images(joint, te["energy"][:n_eval], steps=int(cfg["sample_steps"]), seed=seed + 6)
    joint_time = time.perf_counter() - t0

    real = te["images"][:n_eval]
    r_ar = P.evaluate(real, ar_imgs, seed=seed)
    r_joint = P.evaluate(real, joint_imgs, seed=seed)

    rows = [
        {"model": "autoregressive", "params": sum(count_params(s.net) for s in stages),
         "auc": r_ar["c2st_auc"], "rel": r_ar["mean_rel_err"],
         "sample_s": round(sum(stage_times), 3)},
        {"model": "joint", "params": count_params(joint_net),
         "auc": r_joint["c2st_auc"], "rel": r_joint["mean_rel_err"],
         "sample_s": round(joint_time, 3)},
    ]
    metrics = {
        "rows": rows,
        "table": fmt_table(rows, ["model", "params", "auc", "rel", "sample_s"]),
        "stage_seconds": [round(t, 3) for t in stage_times],
        "n_stages": N_STAGES,
    }
    print(metrics["table"])

    # ---------------- figures -----------------------------------------
    compare_grid(real, ar_imgs, fig_path(cfg, "ar_samples"), "autoregressive (bottom) vs data (top)", n=6)
    compare_grid(real, joint_imgs, fig_path(cfg, "joint_samples"), "joint flow (bottom) vs data (top)", n=6)
    profile_compare(real, ar_imgs, fig_path(cfg, "ar_profiles"), "layer-autoregressive")
    loss_curve(stages[0].history, fig_path(cfg, "stage0_loss"), "stage 0 MSE")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    axes[0].bar(np.arange(1, N_STAGES + 1), stage_times, color="#2c7fb8")
    axes[0].set(xlabel="stage", ylabel="seconds", title="sequential sampling cost")
    blocks_r = real.reshape(len(real), N_STAGES, -1).sum(2).mean(0)
    blocks_a = ar_imgs.reshape(len(ar_imgs), N_STAGES, -1).sum(2).mean(0)
    blocks_j = joint_imgs.reshape(len(joint_imgs), N_STAGES, -1).sum(2).mean(0)
    idx = np.arange(N_STAGES)
    axes[1].bar(idx - 0.2, blocks_r, width=0.2, label="data")
    axes[1].bar(idx, blocks_a, width=0.2, label="AR")
    axes[1].bar(idx + 0.2, blocks_j, width=0.2, label="joint")
    axes[1].set_xticks(idx, [f"rows {4*i}-{4*i+3}" for i in idx], fontsize=6)
    axes[1].set(ylabel="energy fraction", title="energy per layer block")
    axes[1].legend(frameon=False, fontsize=6)
    save(fig, fig_path(cfg, "stage_costs"))

    np.savez_compressed(cfg["outdir"] / "samples.npz", ar=ar_imgs.astype(np.float32),
                        joint=joint_imgs.astype(np.float32))
    save_json(metric_path(cfg), metrics)
    return metrics


if __name__ == "__main__":
    main()
