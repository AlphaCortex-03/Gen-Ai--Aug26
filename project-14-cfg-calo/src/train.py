"""
project-14-cfg-calo -- Classifier-Free Guidance for energy control.

Training uses cond-dropout: 10% of samples get the "null" condition (c=0).
At sampling, v = v_null + w * (v_cond - v_null) for guidance scales
w in {0, 1, 2, 4, 7}. Because shower width grows with energy in the data,
w > 1 should sharpen the effective energy response: the measured RMS
radius of generated 50 GeV samples should move toward / past the true value.

Run:  python train.py                        (full, ~3 min on CPU)
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

from calo_sim import dataset, cond_of, GRID

HERE = Path(__file__).resolve().parent
OUT, CKPT, DATA = HERE.parent / "outputs", HERE.parent / "checkpoints", HERE.parent / "data"
for d in (OUT, CKPT, DATA):
    d.mkdir(exist_ok=True)

ITERS = int(os.environ.get("GENAI_ITERS", "1500"))
BATCH = 128
P_NULL = 0.10
GUIDANCE = (0.0, 1.0, 2.0, 4.0, 7.0)
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
TestX, TestE = dataset(300, ENERGIES, seed=999)
TestX_np = TestX[:, 0]


def eff_radius(img):
    """Centroid-weighted RMS radius (cells) of a shower map."""
    yy, xx = np.mgrid[0:GRID, 0:GRID].astype(np.float64)
    tot = img.sum()
    cx = (img * xx).sum() / tot
    cy = (img * yy).sum() / tot
    return float(np.sqrt((img * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum() / tot))


# ---------------- train with cond-dropout ----------------
model = Velocity().to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for it in range(1, ITERS + 1):
    idx = torch.randint(0, X.shape[0], (BATCH,), device=DEV)
    x1, c = X[idx], C[idx].clone()
    null = torch.rand(BATCH, 1, device=DEV) < P_NULL
    c[null] = 0.0                                  # null condition
    x0 = torch.randn_like(x1)
    t = torch.rand(BATCH, 1, device=DEV)
    xt = (1.0 - t) * x0 + t * x1
    loss = nn.functional.mse_loss(model(xt, t, c), x1 - x0)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % max(1, ITERS // 10) == 0:
        print(f"iter {it:5d}/{ITERS}  cfm_loss={loss.item():.4f}")
torch.save(model.state_dict(), CKPT / "velocity_cond_dropout.pt")
model.eval()

# ---------------- guided sampling ----------------
@torch.no_grad()
def sample_cfg(c, w, n_steps=50):
    x = torch.randn(c.shape[0], D, device=DEV)
    null = torch.zeros_like(c)
    for s in range(n_steps):
        t = torch.full((c.shape[0], 1), s / n_steps, device=DEV)
        v_c = model(x, t, c)
        v_u = model(x, t, null)
        v = v_u + w * (v_c - v_u)
        x = x + v / n_steps
    return x


true_radius = {E0: eff_radius(TestX_np[np.asarray(TestE) == E0].mean(0))
               for E0 in (10.0, 50.0, 100.0)}
true_r50 = eff_radius(TestX_np[np.asarray(TestE) == 50.0].mean(0))

radius_by_w = []
for w in GUIDANCE:
    c50 = torch.full((128, 1), float(np.log10(0.5)), device=DEV)
    gen = sample_cfg(c50, w).reshape(128, 16, 16).mean(0).cpu().numpy()
    gen = np.clip(gen, 0.0, None)                      # CFG extrapolation can go negative
    r = eff_radius(gen)
    radius_by_w.append(r)
    print(f"w = {w:>4.1f}: generated 50 GeV sample radius = {r:.3f} cells (true = {true_r50:.3f})")

fig, axes = plt.subplots(2, len(GUIDANCE), figsize=(3 * len(GUIDANCE), 6))
for j, w in enumerate(GUIDANCE):
    c100 = torch.full((64, 1), float(np.log10(10.0)), device=DEV)
    g_hi = np.clip(sample_cfg(c100, w).reshape(64, 16, 16).mean(0).cpu().numpy(), 0.0, None)
    c10 = torch.full((64, 1), 0.0, device=DEV)
    g_lo = np.clip(sample_cfg(c10, w).reshape(64, 16, 16).mean(0).cpu().numpy(), 0.0, None)
    axes[0, j].imshow(g_lo, cmap="inferno")
    axes[0, j].set_title(f"w={w} @ 10 GeV")
    axes[1, j].imshow(g_hi, cmap="inferno")
    axes[1, j].set_title(f"w={w} @ 100 GeV")
    for ax in (axes[0, j], axes[1, j]):
        ax.axis("off")
fig.suptitle("CFG w sharpens the conditioning contrast")
fig.tight_layout()
fig.savefig(OUT / "cfg_samples.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(GUIDANCE, radius_by_w, "o-", label="generated 50 GeV")
ax.axhline(true_r50, ls="--", c="k", label=f"true 50 GeV = {true_r50:.2f}")
for E0, col in ((10.0, "C0"), (100.0, "C3")):
    ax.axhline(true_radius[E0], ls=":", c=col, label=f"true {E0:.0f} GeV = {true_radius[E0]:.2f}")
ax.set_xlabel("guidance scale w")
ax.set_ylabel("RMS shower radius [cells]")
ax.set_title("CFG controls the energy response sharpness")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "cfg_radius_curve.png", dpi=120)
plt.close(fig)

(OUT / "metrics.json").write_text(json.dumps(
    {"guidance_scales": list(GUIDANCE),
     "radius_by_w": [round(r, 4) for r in radius_by_w],
     "true_radius": {f"E={k:.0f}": round(v, 4) for k, v in true_radius.items()}},
    indent=2), encoding="utf-8")
print(f"project-14-cfg-calo done ({DEV}) -> outputs/")
