"""Evaluation metrics: C2ST, AUC, 1D Wasserstein, physics closure."""
from __future__ import annotations

import numpy as np
import torch

from .data import physics_features, summary_vector


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Tie-aware ranks (equivalent to scipy.stats.rankdata, no scipy needed)."""
    a = np.asarray(a)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# AUC / classifier two-sample test
# ---------------------------------------------------------------------------
def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC with mid-ranks for ties."""
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores).ravel()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = _rankdata(scores)
    r_pos = ranks[labels == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _fit_logreg(Xtr, ytr, iters=300, lr=0.05, wd=1e-3, seed=0):
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    Xt = torch.as_tensor(Xtr, dtype=torch.float32)
    yt = torch.as_tensor(ytr, dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(iters):
        opt.zero_grad()
        loss = lossf(Xt @ w + b, yt) + wd * (w ** 2).sum()
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


def c2st_scores(X: np.ndarray, Y: np.ndarray, folds: int = 2, seed: int = 0, iters: int = 300):
    """Out-of-fold classifier scores for a two-sample test.

    A logistic-regression classifier is trained on (label, features) and scored
    out-of-fold, so an over-confident classifier cannot inflate the AUC.
    Returns ``(labels, scores)`` with 1 = first sample, 0 = second.
    """
    n = min(len(X), len(Y))
    X = np.asarray(X, dtype=np.float64)[:n]
    Y = np.asarray(Y, dtype=np.float64)[:n]
    data = np.concatenate([X, Y], axis=0)
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))
    data, labels = data[idx], labels[idx]
    mu, sd = data.mean(0), data.std(0) + 1e-9
    data = (data - mu) / sd

    fold_id = np.arange(len(data)) % folds
    scores = np.zeros(len(data))
    for f in range(folds):
        tr = fold_id != f
        te = ~tr
        w, b = _fit_logreg(data[tr], labels[tr], iters=iters, seed=seed + f)
        with torch.no_grad():
            logits = torch.as_tensor(data[te], dtype=torch.float32) @ w + b
        scores[te] = torch.sigmoid(logits).numpy()
    return labels, scores


def c2st_auc(X: np.ndarray, Y: np.ndarray, folds: int = 2, seed: int = 0, iters: int = 300) -> float:
    """C2ST AUC. 0.5 = indistinguishable, 1.0 = trivially separable."""
    labels, scores = c2st_scores(X, Y, folds=folds, seed=seed, iters=iters)
    return auc_score(labels, scores)


def roc_curve(labels, scores):
    """ROC points (fpr, tpr) without sklearn."""
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores).ravel()
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    tps = np.cumsum(labels == 1)
    fps = np.cumsum(labels == 0)
    P = max(int((labels == 1).sum()), 1)
    N = max(int((labels == 0).sum()), 1)
    # collapse ties
    keep = np.r_[np.diff(scores) != 0, True]
    return fps[keep] / N, tps[keep] / P


def permutation_test(X, Y, n_perm: int = 20, seed: int = 0, **kw) -> dict:
    """Null distribution of C2ST AUC under label permutation."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = min(len(X), len(Y))
    pooled = np.concatenate([X[:n], Y[:n]], 0)
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    obs = c2st_auc(X, Y, seed=seed, **kw)
    rng = np.random.default_rng(seed + 1)
    null = []
    for i in range(n_perm):
        perm = rng.permutation(labels)
        null.append(c2st_auc(pooled[perm == 1], pooled[perm == 0], seed=seed + 10 + i, **kw))
    null = np.asarray(null)
    p = (1.0 + float((null >= obs).sum())) / (1.0 + n_perm)
    return {
        "auc": obs,
        "p_value": p,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "null": null,
    }


# ---------------------------------------------------------------------------
# distribution distances
# ---------------------------------------------------------------------------
def wasserstein1(a, b, nq: int = 200) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    q = np.linspace(0.0, 1.0, nq)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def compare_observables(real: np.ndarray, fake: np.ndarray) -> dict:
    """Physics closure: per-observable mean relative error + Wasserstein."""
    fr = physics_features(real)
    ff = physics_features(fake)
    out = {}
    for k in ("depth", "depth_rms", "width", "sparsity", "n_cells", "peak"):
        mr, mf = float(fr[k].mean()), float(ff[k].mean())
        denom = abs(mr) + 1e-9
        out[f"{k}_real"] = mr
        out[f"{k}_gen"] = mf
        out[f"{k}_rel_err"] = abs(mf - mr) / denom
        out[f"{k}_w1"] = wasserstein1(fr[k], ff[k])
    # depth / lateral profile L1
    from .data import lateral_profiles, layer_profiles

    out["depth_profile_l1"] = float(np.abs(layer_profiles(real) - layer_profiles(fake)).sum())
    out["lateral_profile_l1"] = float(np.abs(lateral_profiles(real) - lateral_profiles(fake)).sum())
    out["mean_rel_err"] = float(
        np.mean([out[f"{k}_rel_err"] for k in ("depth", "depth_rms", "width", "sparsity", "n_cells")])
    )
    return out


def radial_moment(images: np.ndarray) -> np.ndarray:
    """Second transverse moment about the shower axis, per event."""
    b, g, _ = images.shape
    frac = images.reshape(b, g, g)
    frac = frac / np.maximum(frac.reshape(b, -1).sum(1), 1e-12)[:, None, None]
    cols = np.arange(g)
    lat = (frac.sum(1) * cols).sum(1)
    return np.sqrt(np.maximum((frac.sum(1) * (cols - lat[:, None]) ** 2).sum(1), 0))


def coverage(pred_mean: np.ndarray, pred_std: np.ndarray, truth: np.ndarray) -> dict:
    """Empirical coverage of the ensemble's predictive bands."""
    z = np.abs(truth - pred_mean) / np.maximum(pred_std, 1e-9)
    return {
        "cov_1sigma": float((z <= 1).mean()),
        "cov_2sigma": float((z <= 2).mean()),
        "mean_z": float(z.mean()),
        "frac_z_gt3": float((z > 3).mean()),
    }


def summary_features(images: np.ndarray) -> np.ndarray:
    """Classifier features used by C2ST for image samples."""
    return summary_vector(np.asarray(images, dtype=np.float64))
