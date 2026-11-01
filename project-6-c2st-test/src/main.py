"""10 — Classifier Two-Sample Test (C2ST) as a rigorous quality metric.

We test four "generators" against held-out data:

* ``flow``    — the trained conditional flow-matching model
* ``blurred`` — real showers smoothed with a Gaussian (a near-miss fake)
* ``mean``    — the dataset mean image plus noise
* ``noise``   — random sparse Dirichlet showers

and report AUC with bootstrap CIs, a permutation-test p-value, ROC curves and
a sample-size power study.  Two feature sets are compared to show that the
answer depends on *what* you feed the classifier.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from calo import pipeline as P
from calo.cli import fig_path, metric_path, setup
from calo.data import get_data, physics_features, split_data
from calo.metrics import auc_score, c2st_auc, c2st_scores, permutation_test, roc_curve
from calo.nets import ResMLP
from calo.util import count_params, fmt_table, save_json
from calo.viz import compare_grid, save

import matplotlib.pyplot as plt

PHYS_KEYS = ("depth", "depth_rms", "width", "sparsity", "n_cells", "peak")


def physics_matrix(images: np.ndarray) -> np.ndarray:
    f = physics_features(images)
    return np.stack([f[k] for k in PHYS_KEYS], axis=1)


def make_fakes(train_imgs: np.ndarray, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    mean_img = train_imgs.mean(0)
    mean_img = mean_img / mean_img.sum()
    return {
        "blurred": np.stack([gaussian_filter(im, sigma=1.0) for im in train_imgs[:n]]) / 1.0,
        "mean": np.clip(mean_img[None] * (1 + 0.25 * rng.standard_normal((n, 16, 16))), 0, None),
        "noise": rng.dirichlet(np.full(256, 0.6), size=n).reshape(n, 16, 16),
    }


def fix_norm(x):
    x = np.clip(x, 0, None).reshape(len(x), -1)
    return (x / np.maximum(x.sum(1, keepdims=True), 1e-12)).reshape(-1, 16, 16)


def main() -> dict:
    args, cfg = setup(__file__, "project-6-c2st-test")
    seed = int(cfg["seed"])
    data = get_data(n_events=int(cfg["n_train"]), seed=seed)
    tr, te = split_data(data, 0.8, seed=seed)

    scaler, x1 = P.prep(cfg, tr)
    net = ResMLP(dim=256, hidden=int(cfg.get("hidden", 160)), depth=int(cfg.get("depth", 4)), cond_dim=1)
    trainer = P.fit(cfg, net, scaler, tr, x1=x1)
    n = P.n_eval(cfg, te)
    flow = P.sample_images(trainer, te["energy"][:n], steps=int(cfg["sample_steps"]), seed=seed + 1)

    real = te["images"][:n]
    fakes = {"flow": flow}
    fakes.update(make_fakes(tr["images"], n, seed + 2))
    fakes = {k: fix_norm(v) for k, v in fakes.items()}

    # ---------------- AUC table ---------------------------------------
    rows, results = [], {}
    for name, fake in fakes.items():
        auc_phys = c2st_auc(physics_matrix(real), physics_matrix(fake), seed=seed)
        auc_pix = c2st_auc(real.reshape(n, -1), fake.reshape(n, -1), seed=seed)
        labels, scores = c2st_scores(physics_matrix(real), physics_matrix(fake), seed=seed)
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(200):
            idx = rng.integers(0, len(labels), len(labels))
            if len(np.unique(labels[idx])) < 2:
                continue
            boot.append(auc_score(labels[idx], scores[idx]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"fake": name, "auc_physics": auc_phys, "auc_pixels": auc_pix,
                     "ci_low": lo, "ci_high": hi})
        results[name] = {"auc_physics": auc_phys, "auc_pixels": auc_pix,
                         "bootstrap_ci": [float(lo), float(hi)], "labels": labels, "scores": scores}
        print(f"  {name:8s} AUC(physics) {auc_phys:.3f} [{lo:.3f},{hi:.3f}]  AUC(pixels) {auc_pix:.3f}")

    # ---------------- permutation test on the flow --------------------
    print("permutation test (flow, physics features, 20 permutations)...")
    perm = permutation_test(physics_matrix(real), physics_matrix(fakes["flow"]),
                            n_perm=int(cfg.get("n_perm", 20)), seed=seed, iters=200)
    print(f"  observed AUC {perm['auc']:.3f}, p = {perm['p_value']:.3f}, "
          f"null {perm['null_mean']:.3f} +/- {perm['null_std']:.3f}")

    # ---------------- power study -------------------------------------
    power = []
    for nn in (128, 256, 512, 1024, n):
        nn = min(nn, n)
        if nn < 64:
            continue
        power.append({
            "n": nn,
            "flow": c2st_auc(physics_matrix(real[:nn]), physics_matrix(fakes["flow"][:nn]), seed=seed),
            "blurred": c2st_auc(physics_matrix(real[:nn]), physics_matrix(fakes["blurred"][:nn]), seed=seed),
        })
        print(f"  n={nn:5d}  AUC flow {power[-1]['flow']:.3f}  blurred {power[-1]['blurred']:.3f}")

    metrics = {
        "table": fmt_table(rows, ["fake", "auc_physics", "auc_pixels", "ci_low", "ci_high"]),
        "rows": rows,
        "permutation": {k: v for k, v in perm.items() if k != "null"},
        "permutation_null": perm["null"],
        "power": power,
        "n_params": count_params(net),
        "n_eval": n,
    }

    # ---------------- figures -----------------------------------------
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for name, res in results.items():
        fpr, tpr = roc_curve(res["labels"], res["scores"])
        ax.plot(fpr, tpr, lw=1.3, label=f"{name} (AUC {res['auc_physics']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(xlabel="false positive rate", ylabel="true positive rate", title="C2ST ROC (physics features)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    save(fig, fig_path(cfg, "roc_curves"))

    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.hist(perm["null"], bins=12, color="#bdbdbd", label="permutation null")
    ax.axvline(perm["auc"], color="#d95f02", lw=2, label=f"observed {perm['auc']:.3f}")
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.set(xlabel="C2ST AUC", ylabel="count", title=f"permutation test (p = {perm['p_value']:.3f})")
    ax.legend(frameon=False, fontsize=7)
    save(fig, fig_path(cfg, "permutation_null"))

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ns = [p["n"] for p in power]
    ax.plot(ns, [p["flow"] for p in power], "o-", label="flow", color="#d95f02")
    ax.plot(ns, [p["blurred"] for p in power], "s--", label="blurred", color="#2c7fb8")
    ax.set_xscale("log")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set(xlabel="test set size", ylabel="C2ST AUC", title="test power vs sample size")
    ax.legend(frameon=False, fontsize=7)
    save(fig, fig_path(cfg, "power_vs_n"))

    compare_grid(real, fakes["flow"], fig_path(cfg, "flow_vs_data"), "flow (bottom) vs data (top)", n=6)
    compare_grid(real, fakes["blurred"], fig_path(cfg, "blurred_vs_data"), "blurred fake (bottom)", n=6)

    save_json(metric_path(cfg), metrics)
    print(metrics["table"])
    return metrics


if __name__ == "__main__":
    main()
