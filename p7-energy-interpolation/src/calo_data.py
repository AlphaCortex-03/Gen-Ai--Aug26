"""Shared synthetic calorimeter shower generator (physics-lite toy model).

All Aug projects reproduce shape/behaviour consistently with this file so that
numbers are comparable across the project tree. Deterministic per seed.
"""
import numpy as np

N_DEPTH = 20
N_RADIAL = 12
VEC_DIM = N_DEPTH * N_RADIAL


def make_rng(seed):
    return np.random.default_rng(seed)


def _longitudinal_weights(e, n_depth=N_DEPTH):
    """Gamma-like longitudinal shower profile; peak shifts slightly with E."""
    z = np.arange(n_depth) + 0.5
    k = 2.0 + 0.04 * np.sqrt(e)
    theta = 3.0 + 0.35 * np.sqrt(e)
    p = z ** (k - 1.0) * np.exp(-z / theta)
    p /= p.sum()
    return p


def _radial_weights(e, p_z, n_radial=N_RADIAL):
    """Lateral spread grows with depth and is broader for lower energies."""
    zn = p_z / p_z.max()
    sig = (0.45 + 1.1 * zn) * (1.15 - 0.18 * np.log10(e + 2.0))
    r = np.arange(n_radial) + 0.5
    g = np.exp(-0.5 * (r[None, :] / sig[:, None]) ** 2)
    return g / g.sum(axis=1, keepdims=True)


def shower_profile(e, rng, n_depth=N_DEPTH, n_radial=N_RADIAL):
    """Relative per-cell energy fractions (sums to 1), shape (n_depth, n_radial)."""
    p_z = _longitudinal_weights(e, n_depth)
    g = _radial_weights(e, p_z, n_radial)
    base = p_z[:, None] * g
    base /= base.sum()
    sigma = 0.12 / (1.0 + 0.03 * e)          # stochasticity shrinks at high E
    base = base * (1.0 + rng.normal(0.0, sigma, size=base.shape))
    base = np.clip(base, 1e-6, None)
    return base / base.sum()


def make_profiles(energies, n_per, seed=0, n_depth=N_DEPTH, n_radial=N_RADIAL):
    """Dataset of normalized shower vectors. Returns (X, E):
    X shape (N, n_depth*n_radial) rows sum to 1, E of matching energies (GeV)."""
    rng = make_rng(seed)
    xs, es = [], []
    for e in np.atleast_1d(energies):
        for _ in range(int(n_per)):
            xs.append(shower_profile(float(e), rng, n_depth, n_radial).ravel())
            es.append(float(e))
    X = np.asarray(xs, dtype=np.float64)
    E = np.asarray(es, dtype=np.float64)
    idx = rng.permutation(len(E))
    return X[idx], E[idx]


def shower_pointcloud(e, rng, n_particles=48):
    """Shower as point cloud (x, y, z, E_particle); z in depth units, x/y in radial units."""
    prof = shower_profile(e, rng)
    flat = prof.ravel()
    cells = rng.choice(len(flat), size=n_particles, p=flat / flat.sum())
    d = cells // N_RADIAL
    r = cells % N_RADIAL
    z = d + rng.uniform(0.0, 1.0, n_particles)
    rad = (r + rng.uniform(0.0, 1.0, n_particles)) * (0.4 + 0.06 * z)
    ang = rng.uniform(0.0, 2.0 * np.pi, n_particles)
    w = flat[cells] * float(e)
    return np.stack([rad * np.cos(ang), rad * np.sin(ang), z, w], axis=1)


def make_pointclouds(energies, n_per, n_particles=48, seed=0):
    rng = make_rng(seed)
    clouds, es = [], []
    for e in np.atleast_1d(energies):
        for _ in range(int(n_per)):
            clouds.append(shower_pointcloud(float(e), rng, n_particles))
            es.append(float(e))
    idx = rng.permutation(len(es))
    return np.asarray(clouds)[idx], np.asarray(es, dtype=np.float64)[idx]


def summary6(vec, e):
    """6-D physics summary of one shower vector: [E, <z>, rms_z, <r>, front, late]."""
    grid = vec.reshape(-1, N_RADIAL)
    depth = grid.sum(axis=1) + 1e-12
    z = np.arange(len(depth)) + 0.5
    zbar = float((depth * z).sum() / depth.sum())
    zrms = float(np.sqrt((depth * (z - zbar) ** 2).sum() / depth.sum()))
    rprof = grid.sum(axis=0)
    rbar = float((rprof * (np.arange(len(rprof)) + 0.5)).sum() / rprof.sum())
    return np.array([e, zbar, zrms, rbar, float(depth[:5].sum()), float(depth[-5:].sum())])
