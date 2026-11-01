"""
p19-transformer-cfm -- DiT velocity with AdaLN-Zero conditioning.

Showers are tokenized into 4x4 = 16 patches of 4x4x1 = 16 dims. A stack of
3 DiT blocks (self-attention + MLP, AdaLN-Zero modulation by (t, E)) maps
tokens to per-token velocity. Trained with the CFM loss; 50-step Euler
sampling.

Run:  python train.py                        (full, ~5 min on CPU)
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
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

ENERGIES = (10.0, 50.0, 100.0)
N_TRAIN = 8000
DIM, TOKENS = 64, 16


def patchify(x):
    """(B,) flat 256 -> (B, 16, 16) tokens: [b, hb, wb, hi*wi]."""
    B = x.shape[0]
    x = x.reshape(B, 16, 16).reshape(B, 4, 4, 4, 4)
    return x.permute(0, 1, 3, 2, 4).reshape(B, TOKENS, 16)


def unpatchify(x):
    """(B, 16, 16) tokens -> (B, 256) flat image."""
    B = x.shape[0]
    x = x.reshape(B, 4, 4, 4, 4).permute(0, 1, 3, 2, 4)
    return x.reshape(B, 256)


class DiTBlock(nn.Module):
    def __init__(self, dim=DIM, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))
        self.ada = nn.Linear(2, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, cond):
        s1, sc1, g1, s2, sc2, g2 = self.ada(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + sc1[:, None, :]) + s1[:, None, :]
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1[:, None, :] * a
        h = self.norm2(x) * (1 + sc2[:, None, :]) + s2[:, None, :]
        return x + g2[:, None, :] * self.mlp(h)


class DiT(nn.Module):
    def __init__(self, depth=3):
        super().__init__()
        self.proj = nn.Linear(16, DIM)
        self.pos = nn.Parameter(torch.randn(1, TOKENS, DIM) * 0.02)
        self.blocks = nn.ModuleList([DiTBlock() for _ in range(depth)])
        self.out = nn.Linear(DIM, 16)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_flat, t, c):
        h = self.proj(patchify(x_flat)) + self.pos
        cond = torch.cat([t, c], dim=1)
        for blk in self.blocks:
            h = blk(h, cond)
        return unpatchify(self.out(h))


# ---------------- data ----------------
X, E = dataset(N_TRAIN, ENERGIES, seed=SEED)
X = torch.from_numpy(X.reshape(N_TRAIN, -1)).to(DEV)
C = torch.from_numpy(cond_of(E))[:, None].to(DEV)
TestX, TestE = dataset(300, ENERGIES, seed=999)
TestX_np = TestX[:, 0]


def cfm_loss(model, x1, c):
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], 1, device=x1.device)
    xt = (1.0 - t) * x0 + t * x1
    return nn.functional.mse_loss(model(xt, t, c), x1 - x0)


@torch.no_grad()
def sample(model, c, n_steps=50):
    x = torch.randn(c.shape[0], 256, device=DEV)
    for s in range(n_steps):
        t = torch.full((c.shape[0], 1), s / n_steps, device=DEV)
        x = x + model(x, t, c) / n_steps
    return x


# ---------------- train ----------------
model = DiT().to(DEV)
n_par = sum(p.numel() for p in model.parameters())
print(f"DiT parameters: {n_par / 1e3:.1f}k")
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []
for it in range(1, ITERS + 1):
    idx = torch.randint(0, X.shape[0], (BATCH,), device=DEV)
    loss = cfm_loss(model, X[idx], C[idx])
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if it % max(1, ITERS // 10) == 0:
        print(f"iter {it:5d}/{ITERS}  cfm_loss={loss.item():.4f}")
torch.save(model.state_dict(), CKPT / "dit_velocity.pt")
np.save(DATA / "loss_curve.npy", np.asarray(losses))

# ---------------- generate + evaluate ----------------
model.eval()
metrics = {"params_k": round(n_par / 1e3, 1), "final_cfm_loss": losses[-1],
           "mean_image_mse": {}}
fig, axes = plt.subplots(2, len(ENERGIES), figsize=(3.2 * len(ENERGIES), 6))
for j, E0 in enumerate(ENERGIES):
    c0 = torch.full((128, 1), float(np.log10(E0 / 10.0)), device=DEV)
    gen = sample(model, c0).reshape(128, 16, 16)
    true = TestX_np[np.asarray(TestE) == E0]
    mse = float(np.mean((gen.mean(0).cpu().detach().numpy() - true.mean(0)) ** 2))
    metrics["mean_image_mse"][f"E={E0:.0f}"] = round(mse, 8)
    axes[0, j].imshow(true.mean(0), cmap="inferno")
    axes[0, j].set_title(f"true E={E0:.0f}")
    axes[1, j].imshow(gen.mean(0).cpu().numpy(), cmap="inferno")
    axes[1, j].set_title(f"DiT-CFM E={E0:.0f}\nMSE={mse:.1e}")
    for ax in (axes[0, j], axes[1, j]):
        ax.axis("off")
print("mean-image MSE:", metrics["mean_image_mse"])
fig.suptitle("DiT velocity (AdaLN-Zero): true vs generated mean shower")
fig.tight_layout()
fig.savefig(OUT / "dit_compare.png", dpi=120)
plt.close(fig)

c50 = torch.full((16, 1), float(np.log10(0.5)), device=DEV)
g = sample(model, c50).cpu().numpy()
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for ax, im in zip(axes.flat, g):
    ax.imshow(im.reshape(16, 16), cmap="inferno")
    ax.axis("off")
fig.suptitle("DiT-CFM generated showers @ 50 GeV")
fig.tight_layout()
fig.savefig(OUT / "dit_samples.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(np.asarray(losses))
ax.set_xlabel("iteration")
ax.set_ylabel("CFM loss")
ax.set_title("DiT training loss")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "loss_curve.png", dpi=120)
plt.close(fig)

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"p19-transformer-cfm done ({DEV}) -> outputs/")
