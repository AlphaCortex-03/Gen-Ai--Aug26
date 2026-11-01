"""
project-12-anomaly-flow -- log-probability as an anomaly score for BSM search.

A continuous normalizing flow (CFM velocity field) is trained ONLY on
nominal showers. Its log-likelihood log p(x) is evaluated with the ODE
change-of-variable formula, where the divergence integral is estimated
with an antithetic Hutchinson estimator. BSM-like showers (two separated
cores) score low log p -> high anomaly score -> detected by AUC.

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

from calo_sim import dataset, cond_of, bsm_shower

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

TRAIN_EN = (10.0, 50.0, 100.0)
N_TRAIN = 8000
N_EVAL = 300
D = 256
LOG_2PI = float(np.log(2.0 * np.pi))


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


def cfm_loss(model, x1, c):
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], 1, device=x1.device)
    xt = (1.0 - t) * x0 + t * x1
    return nn.functional.mse_loss(model(xt, t, c), x1 - x0)


# ---------------- train on nominal only ----------------
X, E = dataset(N_TRAIN, TRAIN_EN, seed=SEED)
C = cond_of(E)
X = torch.from_numpy(X.reshape(N_TRAIN, -1)).to(DEV)
C = torch.from_numpy(C)[:, None].to(DEV)

model = Velocity().to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for it in range(1, ITERS + 1):
    idx = torch.randint(0, X.shape[0], (BATCH,), device=DEV)
    loss = cfm_loss(model, X[idx], C[idx])
    opt.zero_grad()
    loss.backward()
    opt.step()
    if it % max(1, ITERS // 10) == 0:
        print(f"iter {it:5d}/{ITERS}  cfm_loss={loss.item():.4f}")
torch.save(model.state_dict(), CKPT / "velocity.pt")
model.eval()

# marginal conditioning value (average of c) -> flow over the full sample mix
C_MARGINAL = float(np.log10(np.mean(TRAIN_EN) / 10.0))

# ---------------- log-likelihood via ODE change of variables -----------
@torch.enable_grad()
def log_prob(x, n_steps=20):
    """log p(x): integrate the probability-flow ODE from data (t=1) to the
    base Gaussian (t=0). Divergence via antithetic finite-diff Hutchinson."""
    x = x.clone()
    B = x.shape[0]
    c = torch.full((B, 1), C_MARGINAL, device=DEV)
    h = 1.0 / n_steps
    s_acc = torch.zeros(B, device=DEV)
    for s_i in range(n_steps):
        tau = 1.0 - s_i * h
        t = torch.full((B, 1), tau, device=DEV)
        eps = torch.randn_like(x)
        with torch.no_grad():
            f_p = model(x + 1e-3 * eps, t, c)
            f_m = model(x - 1e-3 * eps, t, c)
        tr_est = ((f_p - f_m) / (2e-3) * eps).sum(dim=1)      # trace estimate
        s_acc = s_acc + tr_est * h
        with torch.no_grad():
            x = x - h * model(x, t, c)
    logp0 = -0.5 * (x ** 2).sum(dim=1) - 0.5 * D * LOG_2PI
    return logp0 - s_acc            # log p(x) = log p0 + (divergence sign per CNF)


# ---------------- evaluation sets ----------------
TestX, TestE = dataset(N_EVAL, TRAIN_EN, seed=999)            # nominal test
nominal = torch.from_numpy(TestX.reshape(N_EVAL, -1)).to(DEV)

rng = np.random.default_rng(123)
bsm_imgs = np.stack([bsm_shower(rng, float(rng.choice(TRAIN_EN))) for _ in range(N_EVAL)])
bsm = torch.from_numpy(bsm_imgs.reshape(N_EVAL, -1)).to(DEV)

with torch.no_grad():
    lp_nom = log_prob(nominal).cpu().numpy()
    lp_bsm = log_prob(bsm).cpu().numpy()

score_nom, score_bsm = -lp_nom, -lp_bsm          # anomaly score = -log p
labels = np.concatenate([np.zeros(N_EVAL), np.ones(N_EVAL)])
scores = np.concatenate([score_nom, score_bsm])
order = np.argsort(scores)
ranks = np.empty_like(order, dtype=np.float64)
ranks[order] = np.arange(1, len(scores) + 1)
n_pos = labels.sum()
auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(labels) - n_pos))
print(f"Anomaly AUC = {auc:.4f}  (1.0 = perfect BSM rejection)")
print(f"mean -logp nominal = {score_nom.mean():.1f} | BSM = {score_bsm.mean():.1f}")

# ---------------- plots ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].hist(score_nom, bins=40, alpha=0.6, density=True, label="nominal")
axes[0].hist(score_bsm, bins=40, alpha=0.6, density=True, label="BSM-like")
axes[0].set_xlabel("anomaly score (-log p)")
axes[0].legend()
axes[0].set_title("Score distributions (AUC = %.3f)" % auc)
ths = np.linspace(scores.min(), scores.max(), 100)
tpr = [np.mean(score_bsm >= th) for th in ths]
fpr = [np.mean(score_nom >= th) for th in ths]
axes[1].plot(fpr, tpr)
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].set_xlabel("FPR")
axes[1].set_ylabel("TPR")
axes[1].set_title("ROC: BSM anomaly detection")
for ax in axes:
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "anomaly_roc.png", dpi=120)
plt.close(fig)

top_anom = np.argsort(score_bsm)[::-1][:2]
fig, axes = plt.subplots(1, 3, figsize=(11, 4))
axes[0].imshow(TestX[0, 0], cmap="inferno")
axes[0].set_title("nominal example")
axes[1].imshow(bsm_imgs[top_anom[0]], cmap="inferno")
axes[1].set_title("top-anomaly BSM #1")
axes[2].imshow(bsm_imgs[top_anom[1]], cmap="inferno")
axes[2].set_title("top-anomaly BSM #2")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "anomaly_examples.png", dpi=120)
plt.close(fig)

(OUT / "metrics.json").write_text(json.dumps(
    {"anomaly_auc": round(float(auc), 4),
     "mean_score_nominal": round(float(score_nom.mean()), 2),
     "mean_score_bsm": round(float(score_bsm.mean()), 2)},
    indent=2), encoding="utf-8")
print(f"project-12-anomaly-flow done ({DEV}) -> outputs/")
