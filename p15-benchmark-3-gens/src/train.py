"""
p15-benchmark-3-gens -- WGAN-GP vs DDPM vs CFM on identical synthetic data.

Trains all three generators on exactly the same training set, then compares:
  * mean-image MSE per energy (distribution-level match)
  * a Classifier Two-Sample Test AUC (indistinguishability, lower = better)

Run:  python train.py                        (full, ~6-10 min on CPU)
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
N_TRAIN, N_EVAL = 8000, 600
D = 256
LAM_GP = 10.0

# ---------------- shared data ----------------
X, E = dataset(N_TRAIN, ENERGIES, seed=SEED)
C = cond_of(E)
XT = torch.from_numpy(X.reshape(N_TRAIN, -1)).to(DEV)
CT = torch.from_numpy(C)[:, None].to(DEV)
TestX, TestE = dataset(N_EVAL, ENERGIES, seed=999)
TestX_np = TestX[:, 0]
test_real = TestX.reshape(N_EVAL, -1)


def cond_batch(n, energies=None):
    if energies is None:                      # random energies like training
        i = torch.randint(0, N_TRAIN, (n,), device=DEV)
        return CT[i]
    c = torch.as_tensor(cond_of(energies), dtype=torch.float32, device=DEV)
    c = c.repeat_interleave(n // len(energies) + 1)[:n]
    return c[:, None]


def rank_auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ================= WGAN-GP =================
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(257, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1))

    def forward(self, x, c):
        return self.net(torch.cat([x, c], dim=1))


class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(129, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256))

    def forward(self, c):
        return self.net(torch.cat([torch.randn(c.shape[0], 128, device=DEV), c], dim=1))


critic, wgen = Critic().to(DEV), G().to(DEV)
opt_c = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_g = torch.optim.Adam(wgen.parameters(), lr=2e-4, betas=(0.5, 0.999))
g_iters = max(2, ITERS // 5)
for gi in range(1, g_iters + 1):
    for _ in range(5):
        idx = torch.randint(0, N_TRAIN, (BATCH,), device=DEV)
        real, c = XT[idx], CT[idx]
        with torch.no_grad():
            fake = wgen(c)
        eps = torch.rand(BATCH, 1, device=DEV)
        inter = (eps * real + (1 - eps) * fake).requires_grad_(True)
        grad = torch.autograd.grad(critic(inter, c).sum(), inter, create_graph=True)[0]
        gp = LAM_GP * ((grad.norm(2, dim=1) - 1) ** 2).mean()
        lc = critic(fake, c).mean() - critic(real, c).mean() + gp
        opt_c.zero_grad()
        lc.backward()
        opt_c.step()
    idx = torch.randint(0, N_TRAIN, (BATCH,), device=DEV)
    c = CT[idx]
    lg = -critic(wgen(c), c).mean()
    opt_g.zero_grad()
    lg.backward()
    opt_g.step()
    if gi % max(1, g_iters // 8) == 0:
        print(f"[wgan] g-iter {gi}/{g_iters}  critic={lc.item():.3f} gp={gp.item():.3f}")
wgen.eval()
torch.save(wgen.state_dict(), CKPT / "wgan.pt")


@torch.no_grad()
def wgan_sample(c):
    return wgen(c)

# ================= DDPM (T=200) =================
class Eps(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(258, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256))

    def forward(self, x, t_level, c):
        return self.net(torch.cat([x, t_level, c], dim=1))


T = 200
betas = torch.linspace(1e-4, 0.02, T, device=DEV)
alpha = 1.0 - betas
abar = torch.cumprod(alpha, dim=0)
eps_model = Eps().to(DEV)
opt_d = torch.optim.Adam(eps_model.parameters(), lr=2e-4)
for it in range(1, ITERS + 1):
    idx = torch.randint(0, N_TRAIN, (BATCH,), device=DEV)
    x1, c = XT[idx], CT[idx]
    t0 = torch.randint(0, T, (BATCH,), device=DEV)
    noise = torch.randn_like(x1)
    xt = abar[t0].sqrt()[:, None] * x1 + (1 - abar[t0]).sqrt()[:, None] * noise
    loss = nn.functional.mse_loss(eps_model(xt, (t0 / T)[:, None], c), noise)
    opt_d.zero_grad()
    loss.backward()
    opt_d.step()
    if it % max(1, ITERS // 8) == 0:
        print(f"[ddpm] iter {it:5d}/{ITERS}  eps_loss={loss.item():.4f}")
eps_model.eval()
torch.save(eps_model.state_dict(), CKPT / "ddpm.pt")


@torch.no_grad()
def ddpm_sample(c):
    x = torch.randn(c.shape[0], D, device=DEV)
    for t0 in reversed(range(T)):
        t_lvl = torch.full((c.shape[0], 1), t0 / T, device=DEV)
        pred = eps_model(x, t_lvl, c)
        mean = (1 / alpha[t0].sqrt()) * (x - betas[t0] / (1 - abar[t0]).sqrt() * pred)
        if t0 > 0:
            x = mean + betas[t0].sqrt() * torch.randn_like(x) * 0.9
        else:
            x = mean
    return x

# ================= CFM =================
class Velocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(258, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256))

    def forward(self, x, t, c):
        return self.net(torch.cat([x, t, c], dim=1))


cfm = Velocity().to(DEV)
opt_f = torch.optim.Adam(cfm.parameters(), lr=1e-3)
for it in range(1, ITERS + 1):
    idx = torch.randint(0, N_TRAIN, (BATCH,), device=DEV)
    x1, c = XT[idx], CT[idx]
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], 1, device=DEV)
    xt = (1.0 - t) * x0 + t * x1
    loss = nn.functional.mse_loss(cfm(xt, t, c), x1 - x0)
    opt_f.zero_grad()
    loss.backward()
    opt_f.step()
    if it % max(1, ITERS // 8) == 0:
        print(f"[cfm] iter {it:5d}/{ITERS}  cfm_loss={loss.item():.4f}")
cfm.eval()
torch.save(cfm.state_dict(), CKPT / "cfm.pt")


@torch.no_grad()
def cfm_sample(c):
    x = torch.randn(c.shape[0], D, device=DEV)
    for s in range(50):
        t = torch.full((c.shape[0], 1), s / 50, device=DEV)
        x = x + cfm(x, t, c) / 50
    return x

# ================= metrics =================
samplers = {"WGAN-GP": wgan_sample, "DDPM-200": ddpm_sample, "CFM-50": cfm_sample}
eval_c = cond_batch(N_EVAL, ENERGIES)
results = {}
for name, fn in samplers.items():
    fake = fn(eval_c).cpu().numpy()
    fake = np.clip(fake, fake.min(), fake.max())
    fake_img = fake.reshape(N_EVAL, 16, 16)
    per_mse = {}
    for E0 in ENERGIES:
        sel_t = np.asarray(TestE) == E0
        sel_f = np.asarray(np.repeat(ENERGIES, N_EVAL // len(ENERGIES))) == E0
        per_mse[f"E={E0:.0f}"] = round(
            float(((fake_img[sel_f].mean(0) - TestX_np[sel_t].mean(0)) ** 2).mean()), 8)
    # C2ST: first half train clf, second half evaluate
    half = N_EVAL // 2
    cX = np.concatenate([test_real[:half], fake[:half]])
    cy = np.concatenate([np.ones(half), np.zeros(half)])
    perm = np.random.permutation(len(cX))
    clf = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 1)).to(DEV)
    optc = torch.optim.Adam(clf.parameters(), lr=1e-3)
    clx = torch.from_numpy(cX[perm]).to(DEV)
    cly = torch.from_numpy(cy[perm]).to(DEV).float()
    blc = nn.BCEWithLogitsLoss()
    for it2 in range(300):
        b = torch.randint(0, len(clx), (256,), device=DEV)
        bl = blc(clf(clx[b]).squeeze(1), cly[b])
        optc.zero_grad()
        bl.backward()
        optc.step()
    clf.eval()
    with torch.no_grad():
        sco = torch.sigmoid(clf(torch.cat([torch.from_numpy(
            np.concatenate([test_real[half:], fake[half:]]))]).to(DEV)).squeeze(1)).cpu().numpy()
    lab = np.concatenate([np.ones(N_EVAL - half), np.zeros(N_EVAL - half)])
    auc = rank_auc(sco, lab)
    results[name] = {"mean_image_mse": per_mse, "avg_mse": round(float(np.mean(list(per_mse.values()))), 8),
                     "c2st_auc": round(float(auc), 4)}
    print(f"{name}: avgMSE={results[name]['avg_mse']:.2e}  C2ST AUC={auc:.3f}")

# bar plots
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
names = list(results)
axes[0].bar(names, [results[n]["avg_mse"] for n in names], color=["C1", "C0", "C2"])
axes[0].set_ylabel("avg mean-image MSE")
axes[0].set_title("Distribution match (lower = better)")
axes[1].bar(names, [results[n]["c2st_auc"] for n in names], color=["C1", "C0", "C2"])
axes[1].axhline(0.5, ls="--", c="k", lw=1)
axes[1].set_ylabel("C2ST AUC")
axes[1].set_title("Indistinguishability (closer to 0.5 = better)")
for ax in axes:
    ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "benchmark_bars.png", dpi=120)
plt.close(fig)

# generated sample strips
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
c50 = cond_batch(3, (50.0,))
for r, (name, fn) in enumerate(samplers.items()):
    g = fn(c50).cpu().numpy()
    for k in range(3):
        axes[r, k].imshow(g[k].reshape(16, 16), cmap="inferno")
        axes[r, k].set_title(f"{name}" if k == 0 else "")
        axes[r, k].axis("off")
fig.suptitle("Three generators @ 50 GeV")
fig.tight_layout()
fig.savefig(OUT / "benchmark_samples.png", dpi=120)
plt.close(fig)

(OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"p15-benchmark-3-gens done ({DEV}) -> outputs/")
