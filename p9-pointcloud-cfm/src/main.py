"""09 - Point-cloud flow matching with DeepSets.

Same physics, different representation: each shower is a variable-length set of
hits ``(x, y, layer, E)`` instead of a 16x16 image.  A permutation-equivariant
DeepSets velocity field is trained with flow matching; at sampling time the
cardinality is drawn from the data marginal and the generated points are
voxelised for comparison with the image-based projects.
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from calo import pipeline as P
from calo.cli import fig_path, metric_path, setup
from calo.data import get_data, points_to_images, split_data
from calo.nets import DeepSets
from calo.util import count_params, save_json
from calo.viz import compare_grid, profile_compare, save, scatter3d

import matplotlib.pyplot as plt


class PointScaler:
    """Per-feature standardisation of the hit set, with sqrt on the energy."""

    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = np.asarray(mu, dtype=np.float32)
        self.sigma = np.maximum(np.asarray(sigma, dtype=np.float32), 1e-6)

    @classmethod
    def fit(cls, points: np.ndarray, mask: np.ndarray) -> "PointScaler":
        p = cls._pre(points)
        valid = mask.astype(bool)
        flat = p[valid]
        return cls(flat.mean(0), flat.std(0))

    @staticmethod
    def _pre(points):
        p = points.copy()
        p[..., 3] = np.sqrt(np.clip(p[..., 3], 0, None))
        return p

    def transform(self, points: np.ndarray) -> np.ndarray:
        return ((self._pre(points) - self.mu) / self.sigma).astype(np.float32)

    def inverse(self, z: np.ndarray) -> np.ndarray:
        p = z * self.sigma + self.mu
        p[..., 3] = np.clip(p[..., 3], 0, None) ** 2
        p[..., 0:3] = np.clip(p[..., 0:3], 0.0, 1.0)
        return p


@torch.no_grad()
def sample_points(net, scaler, energies, multiplicity, steps=32, seed=0, cond_mean=0.0, cond_std=1.0):
    torch.manual_seed(seed)
    n = len(energies)
    pmax = int(multiplicity.max())
    mask = (torch.arange(pmax)[None, :] < torch.as_tensor(multiplicity)[:, None]).float()
    x = torch.randn(n, pmax, 4) * mask[..., None]
    c = torch.as_tensor((energies - cond_mean) / (cond_std + 1e-9), dtype=torch.float32)[:, None]
    ts = torch.linspace(0, 1, steps + 1)
    for i in range(steps):
        t = ts[i].expand(n, 1)
        x = x + (ts[i + 1] - ts[i]) * net(x, t, c, mask)
    p = scaler.inverse(x.numpy())
    e = p[..., 3] * mask.numpy()
    e = e / np.maximum(e.sum(1, keepdims=True), 1e-12)
    p[..., 3] = e
    return p, mask.numpy(), c


def main() -> dict:
    args, cfg = setup(__file__, "p9-pointcloud-cfm: DeepSets flow matching")
    seed = int(cfg["seed"])
    data = get_data(n_events=int(cfg["n_train"]), seed=seed, with_points=True, pmax=320)
    tr, te = split_data(data, 0.8, seed=seed)

    scaler = PointScaler.fit(tr["points"], tr["mask"])
    p1 = scaler.transform(tr["points"]) * tr["mask"][..., None]
    p1 = torch.as_tensor(p1, dtype=torch.float32)
    m1 = torch.as_tensor(tr["mask"], dtype=torch.float32)
    e_mean, e_std = float(tr["energy"].mean()), float(tr["energy"].std())
    cond = torch.as_tensor((tr["energy"] - e_mean) / e_std, dtype=torch.float32)[:, None]

    net = DeepSets(feat=4, hidden=int(cfg.get("hidden", 96)), cond_dim=1, depth=3)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(cfg["lr"]), total_steps=int(cfg["steps"]), pct_start=0.15)
    print(f"training DeepSets | {count_params(net):,} params | points/event {int(tr['mask'].sum(1).mean())}")

    bs = int(cfg["batch"])
    n = len(p1)
    t0 = time.perf_counter()
    hist = []
    for step in range(int(cfg["steps"])):
        idx = torch.randint(0, n, (min(bs, n),))
        pb, mb, cb = p1[idx], m1[idx], cond[idx]
        x0 = torch.randn_like(pb) * mb[..., None]
        t = torch.rand(pb.shape[0], 1)
        xt = (1 - t)[..., None] * x0 + t[..., None] * pb
        pred = net(xt, t, cb, mb)
        w = mb[..., None]
        loss = (((pred - (pb - x0)) ** 2) * w).sum() / (w.sum() * 4).clamp(min=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pp in zip(ema.parameters(), net.parameters()):
                pe.mul_(0.99).add_(pp, alpha=0.01)
        if step % max(1, int(cfg["steps"]) // 4) == 0:
            hist.append({"step": step, "loss": float(loss.detach())})
            print(f"  step {step:4d}  loss {float(loss.detach()):.4f}  {time.perf_counter() - t0:5.1f}s")
    train_s = time.perf_counter() - t0
    net.eval()

    # ---------------- sample ------------------------------------------
    n_eval = P.n_eval(cfg, te)
    mult = tr["mask"].sum(1).astype(int)
    mult_draw = np.random.default_rng(seed).choice(mult, size=n_eval)
    t0 = time.perf_counter()
    pts, mask, _ = sample_points(ema, scaler, te["energy"][:n_eval], mult_draw,
                                 steps=int(cfg["sample_steps"]), seed=seed + 1,
                                 cond_mean=e_mean, cond_std=e_std)
    sample_s = time.perf_counter() - t0

    gen = points_to_images(pts, mask.astype(bool))
    real = te["images"][:n_eval]
    metrics = P.evaluate(real, gen, seed=seed)
    metrics.update({
        "n_params": count_params(net),
        "train_seconds": round(train_s, 2),
        "sample_seconds_per_event": round(sample_s / n_eval, 4),
        "mean_points_real": float(mult_draw.mean()),
        "mean_points_gen": float(mask.sum(1).mean()),
        "voxel_c2st": metrics["c2st_auc"],
    })
    print(f"C2ST AUC {metrics['c2st_auc']:.3f} | rel err {metrics['mean_rel_err']:.3f} | "
          f"depth {metrics['depth_real']:.2f}/{metrics['depth_gen']:.2f}")

    # ---------------- figures -----------------------------------------
    scatter3d([te["points"][0], pts[0]], fig_path(cfg, "pointcloud_3d"), "data (left) vs generated (right)")
    compare_grid(real, gen, fig_path(cfg, "voxelised_vs_data"), "voxelised point clouds vs data", n=6)
    profile_compare(real, gen, fig_path(cfg, "profiles"), "point-cloud model")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    axes[0].hist(mult_draw, bins=40, alpha=0.55, label="data (drawn)", color="#2c7fb8")
    axes[0].hist(mask.sum(1), bins=40, alpha=0.55, label="generated used", color="#d95f02")
    axes[0].set(xlabel="points per event", ylabel="events", title="cardinality")
    axes[0].legend(frameon=False, fontsize=7)
    ep = pts[..., 3][mask.astype(bool)]
    er = te["points"][..., 3][te["mask"].astype(bool)]
    axes[1].hist(np.log10(er + 1e-6), bins=50, density=True, histtype="step", color="#2c7fb8", label="data")
    axes[1].hist(np.log10(ep + 1e-6), bins=50, density=True, histtype="step", color="#d95f02", label="generated")
    axes[1].set(xlabel="log10 hit energy fraction", ylabel="density", title="hit energy spectrum")
    axes[1].legend(frameon=False, fontsize=7)
    save(fig, fig_path(cfg, "multiplicity_energy"))

    np.savez_compressed(cfg["outdir"] / "samples.npz", points=pts.astype(np.float32),
                        mask=mask.astype(np.float32), images=gen.astype(np.float32))
    save_json(metric_path(cfg), metrics)
    return metrics


if __name__ == "__main__":
    main()
