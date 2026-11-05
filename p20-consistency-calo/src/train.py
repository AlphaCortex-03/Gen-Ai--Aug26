"""
p20-consistency-calo -- Consistency-style model, 1-step sampling.

Parameterization: F(x_t, t) = x_t + (1-t) * v_theta(x_t, t, E), which
satisfies the consistency boundary condition at t=1 (F(x_1, 1) = x_1).
Training uses a level-discretized schedule with heavy emphasis on the
t ~ 0 boundary region, so the 1-step map x1 = F(x_0, 0) is accurate.
Sampling: single Euler step (t: 0 -> 1) vs. the 50-step baseline.

Run:  python train.py                        (full, ~4 min on CPU)
      GENAI_ITERS=100 python train.py        (quick smoke test)
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calo_sim import dataset, cond_of

HERE = Path(__file__).resolve().parent
OUT, CKPT, DATA = HERE.parent / "outputs", HERE.parent / "checkpoints", HERE.parent / "data"
for d in (OUT, CKPT, DATA):
    d.mkdir(exist_ok=True)

ITERS = int(os.environ.get("GENAI_ITERS", "1500"))
BATCH = 128
LEVELS = (0.0625, 0.25, 0.5, 0.75, 1.0)      # discretized schedule near 1.0 top
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

ENERGIES = (10.0, 50.0, 100.0)
N_TRAIN = 8000
D = 256


class Velocity(nn.Module):
    def __init__(self, dim=D, cond=1, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + cond + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t, c):
        return self.net(torch.cat([x, t, c], dim=1))


# ---------------- data ----------------
X, E = dataset(N_TRAIN, ENERGIES, seed=SEED)
X = torch.from_numpy(X.reshape(N_TRAIN, -1)).to(DEV)
C = torch.from_numpy(cond_of(E))[:, None].to(DEV)
TestX, TestE = dataset(400, ENERGIES, seed=999)
TestX_np = TestX[:, 0]

# ---------------- consistency-style training ----------------
model = Velocity().to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for it in range(1, ITERS + 1):
    idx = torch.randint(0, X.shape[0], (BATCH,), device=DEV)
    x1, c = X[idx], C[idx]
    x0 = torch.randn_like(x1)
    # 50% of batches: uniform t; 50%: boundary emphasis on the low levels
    if np.random.rand() < 0.5:
        t = torch.rand(BATCH, 1, device=DEV)
    else:
        lv = torch.tensor(LEVELS, device=DEV)
        t = lv[torch.randint(0, len(lv) - 1, (BATCH,), device=DEV)][:, None]
        t = t * (1 - 1e-3)
    xt = (1.0 - t) * x0 + t * x1
    # consistency map target: one big Euler jump to the next level
    step = 1.0 - t                                # jump to endpoint
    loss = nn.functional.mse_loss(model(xt, t, c) * step, (x1 - xt) * step)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % max(1, ITERS // 10) == 0:
        print(f"iter {it:5d}/{ITERS}  consistency_loss={loss.item():.4f}")
torch.save(model.state_dict(), CKPT / "consistency_velocity.pt")
model.eval()


@torch.no_grad()
def consistency_sample(c):
    """1 step: x -> x + (1 - 0) * v(x, 0, c)."""
    x = torch.randn(c.shape[0], D, device=DEV)
    t = torch.zeros_like(c)
    return x + model(x, t, c)


@torch.no_grad()
def euler_sample(c, n_steps=50):
    x = torch.randn(c.shape[0], D, device=DEV)
    for s in range(n_steps):
        t = torch.full((c.shape[0], 1), s / n_steps, device=DEV)
        x = x + model(x, t, c) / n_steps
    return x

# ---------------- evaluate: 1-step vs 50-step ----------------
metrics = {"mean_image_mse": {}}
for tag, fn in (("1-step", consistency_sample), ("50-step", euler_sample)):
    per = {}
    for E0 in ENERGIES:
        c0 = torch.full((128, 1), float(np.log10(E0 / 10.0)), device=DEV)
        gen = fn(c0).reshape(128, 16, 16)
        true = TestX_np[np.asarray(TestE) == E0]
        per[f"E={E0:.0f}"] = round(float(np.mean((gen.mean(0).cpu().detach().numpy() - true.mean(0)) ** 2)), 8)
    metrics["mean_image_mse"][tag] = per
    print(f"{tag:>7} sampling mean-image MSE: {per}")

fig, axes = plt.subplots(3, len(ENERGIES), figsize=(3.2 * len(ENERGIES), 9))
for j, E0 in enumerate(ENERGIES):
    c0 = torch.full((128, 1), float(np.log10(E0 / 10.0)), device=DEV)
    g1 = consistency_sample(c0).reshape(128, 16, 16).mean(0).cpu().numpy()
    g50 = euler_sample(c0).reshape(128, 16, 16).mean(0).cpu().numpy()
    tru = TestX_np[np.asarray(TestE) == E0].mean(0)
    for i, (img, title) in enumerate((
            (tru, f"true E={E0:.0f}"),
            (g1, f"1-step (MSE={metrics['mean_image_mse']['1-step'][f'E={E0:.0f}']:.1e})"),
            (g50, f"50-step (MSE={metrics['mean_image_mse']['50-step'][f'E={E0:.0f}']:.1e})"))):
        axes[i, j].imshow(img, cmap="inferno")
        axes[i, j].set_title(title, fontsize=9)
        axes[i, j].axis("off")
fig.suptitle("Consistency-style model: 1-step vs 50-step sampling")
fig.tight_layout()
fig.savefig(OUT / "consistency_compare.png", dpi=120)
plt.close(fig)

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"p20-consistency-calo done ({DEV}) -> outputs/")
