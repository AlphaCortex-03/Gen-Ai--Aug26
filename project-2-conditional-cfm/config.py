"""Configuration knobs: QUICK (default, seconds) and FULL (minutes)."""

QUICK = {
    "steps": 3000, "batch": 256, "width": 128, "pca_k": 16,
}

FULL = {
    "steps": 12000, "batch": 256, "width": 256, "pca_k": 24,
}
