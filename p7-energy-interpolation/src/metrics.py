"""Shared evaluation metrics for the Aug shower projects (numpy, no deps)."""
import numpy as np


def _norm(img):
    img = np.asarray(img, dtype=np.float64)
    s = img.sum()
    return img / s if s > 0 else img


def profile_l2(sample_vecs, data_vecs, n_radial=12):
    """L2 distance between mean depth profile of samples vs data (rows sum 1)."""
    a = _norm(np.asarray(sample_vecs).reshape(len(sample_vecs), -1).mean(0))
    b = _norm(np.asarray(data_vecs).reshape(len(data_vecs), -1).mean(0))
    return float(np.sqrt(((a - b) ** 2).sum()))


def hist2d_l2(a2d, b2d, bins=(20, 12), rng_range=None):
    """L2 between normalized 2D histograms of two point sets (x, y arrays)."""
    a = np.asarray(a2d, dtype=np.float64)
    b = np.asarray(b2d, dtype=np.float64)
    if rng_range is None:
        lo = min(a.min(0).min(), b.min(0).min())
        hi = max(a.max(0).max(), b.max(0).max())
        rng_range = [[lo, hi]] * 2
    ha, _ = np.histogramdd(a, bins=bins, range=rng_range)
    hb, _ = np.histogramdd(b, bins=bins, range=rng_range)
    return float(np.sqrt(((_norm(ha) - _norm(hb)) ** 2).sum()))


def auc(scores_pos, scores_neg):
    """Rank-based area under ROC; scores for positive-class should be larger."""
    s = np.concatenate([np.asarray(scores_pos, float), np.asarray(scores_neg, float)])
    y = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order][i:j + 1].mean())
            ranks[order][i:j + 1] = avg
        i = j + 1
    npos = y.sum()
    nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def deposit_stats(vecs, n_radial=12):
    """Per-depth mean profile (over radial axis) numpy (n_depth,) for plots."""
    return np.asarray(vecs).reshape(len(vecs), -1, n_radial).mean(0).sum(axis=1)
