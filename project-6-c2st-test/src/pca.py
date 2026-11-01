"""PCA into a compact latent space for fast, smooth CFM training (numpy)."""
import numpy as np

def fit_pca(X, k=16):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    C = X - mu
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    comps = Vt[:k]
    Z = C @ comps.T
    scale = Z.std(axis=0) + 1e-8
    return {"mu": mu, "comps": comps, "scale": scale,
            "evr": float((S[:k] ** 2).sum() / max((S ** 2).sum(), 1e-12))}

def encode(pca, X):
    return ((np.asarray(X) - pca["mu"]) @ pca["comps"].T) / pca["scale"]

def decode(pca, Z):
    return (np.asarray(Z) * pca["scale"]) @ pca["comps"] + pca["mu"]
