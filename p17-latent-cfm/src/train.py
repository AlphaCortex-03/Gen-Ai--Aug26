"""
p17-latent-cfm -- VAE + Flow in latent space.

Stage 1: train a small MLP-VAE that encodes 16x16 showers into 16-d latents.
Stage 2: train a CFM velocity field in latent space conditioned on E.
Sampling: CFM generates a latent, the VAE decoder maps it back to a shower.

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
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

ENERGIES = (10.0, 50.0, 100.0)
N_TRAIN = 8000
D, Z = 256, 16


# ---------------- VAE ----------------
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, 256), nn.SiLU(),
                                 nn.Linear(256, 128), nn.SiLU())
        self.mu = nn.Linear(128, Z)
        self.lv = nn.Linear(128, Z)
        self.dec = nn.Sequential(nn.Linear(Z, 128), nn.SiLU(),
                                 nn.Linear(128, 256), nn.SiLU(),
                                 nn.Linear(256, D))

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.lv(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)   # reparam
        return self.dec(z), mu, lv


X, E = dataset(N_TRAIN, ENERGIES, seed=SEED)
C = cond_of(E)
Xf = torch.from_numpy(X.reshape(N_TRAIN, -1)).to(DEV)
C = torch.from_numpy(C)[:, None].to(DEV)
TestX, TestE = dataset(300, ENERGIES, seed=999)
TestX_np = TestX[:, 0]
TestXf = torch.from_numpy(TestX.reshape(300, -1)).to(DEV)

VAE_BETA = 5e-2
vae = VAE().to(DEV)
opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
for it in range(1, ITERS + 1):
    idx = torch.randint(0, N_TRAIN, (BATCH,), device=DEV)
    xb = Xf[idx]
    rec, mu, lv = vae(xb)
    recon = nn.functional.mse_loss(rec, xb)
    kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp())).sum(dim=1).mean()
    loss = recon + VAE_BETA * kl
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % max(1, ITERS // 10) == 0:
        print(f"[vae] iter {it:5d}/{ITERS}  recon={recon.item():.2e} kl={kl.item():.2e}")
torch.save(vae.state_dict(), CKPT / "vae.pt")
vae.eval()

# ---------------- latent CFM ----------------
with torch.no_grad():
    ZT, _ = vae.encode(Xf)                                 # train on mu (mean path)
    Z_eval = ZT.clone()

class Velocity(nn.Module):
    def __init__(self, dim=Z, cond=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + cond + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t, c):
        return self.net(torch.cat([x, t, c], dim=1))


def cfm_loss(model, x1, c):
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], 1, device=x1.device)
    xt = (1.0 - t) * x0 + t * x1
    return nn.functional.mse_loss(model(xt, t, c), x1 - x0)


@torch.no_grad()
def sample(model, c, n_steps=50):
    x = torch.randn(c.shape[0], Z, device=DEV)
    for s in range(n_steps):
        t = torch.full((c.shape[0], 1), s / n_steps, device=DEV)
        x = x + model(x, t, c) / n_steps
    return x


cmodel = Velocity().to(DEV)
opt = torch.optim.Adam(cmodel.parameters(), lr=1e-3)
for it in range(1, ITERS + 1):
    idx = torch.randint(0, N_TRAIN, (BATCH,), device=DEV)
    loss = cfm_loss(cmodel, ZT[idx], C[idx])
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % max(1, ITERS // 10) == 0:
        print(f"[latent-cfm] iter {it:5d}/{ITERS}  loss={loss.item():.4f}")
torch.save(cmodel.state_dict(), CKPT / "latent_cfm.pt")
cmodel.eval()

# ---------------- generate + evaluate ----------------
metrics = {}
with torch.no_grad():
    recon = vae.dec(Z_eval[:48]).cpu().numpy()
    c50 = torch.full((48, 1), float(np.log10(0.5)), device=DEV)
    z_gen = sample(cmodel, c50)
    gen = vae.dec(z_gen).cpu().numpy()
    rec_mse = float(((recon - Xf[:48].cpu().numpy()) ** 2).mean())
    gen_mean = gen.mean(0).reshape(16, 16)
    true_mean = TestX_np[np.asarray(TestE) == 50.0].mean(0)
    gen_mse = float(((gen_mean - true_mean) ** 2).mean())
metrics = {"vae_recon_mse": round(rec_mse, 8), "generated_mean_image_mse": round(gen_mse, 8)}
print("recon MSE:", rec_mse, "| generated mean-image MSE:", gen_mse)

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for j in range(8):
    axes[0, j].imshow(TestX_np[j], cmap="inferno")
    axes[0, j].set_title("true" if j == 0 else "")
    axes[1, j].imshow(Xf[j].reshape(16, 16).cpu(), cmap="inferno")
    axes[1, j].set_title("VAE input" if j == 0 else "")
    axes[2, j].imshow(recon[j].reshape(16, 16), cmap="inferno")
    axes[2, j].set_title("VAE recon" if j == 0 else "")
    axes[3, j].imshow(gen[j].reshape(16, 16), cmap="inferno")
    axes[3, j].set_title("latent-CFM gen" if j == 0 else "")
    for ax in axes[:, j]:
        ax.axis("off")
fig.suptitle("Latent-space CFM pipeline: true -> VAE -> reconstruction -> generated")
fig.tight_layout()
fig.savefig(OUT / "latent_cfm_pipeline.png", dpi=120)
plt.close(fig)

# latent space: real mu vs generated latents (first 2 dims)
with torch.no_grad():
    Zt2 = Z_eval.cpu().numpy()
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(Zt2[:, 0], Zt2[:, 1], s=8, alpha=0.5, label="real latents (VAE mu)")
ax.scatter(z_gen[:200, 0].cpu(), z_gen[:200, 1].cpu(), s=8, alpha=0.5, label="CFM latents")
ax.set_xlabel("latent dim 0")
ax.set_ylabel("latent dim 1")
ax.legend()
ax.set_title("Latent space: real vs flow-generated")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "latent_scatter.png", dpi=120)
plt.close(fig)

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"p17-latent-cfm done ({DEV}) -> outputs/")
