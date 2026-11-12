"""Hand-crafted morphological features for the classical baseline.

A CNN result is only convincing if it beats a simple model built on the
quantities an astronomer would measure by hand: peak significance, central
concentration, size and shape. These are computed from the same cutouts.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = ["peak_snr", "aperture_snr", "concentration", "area_half_max_pix",
                 "axis_ratio", "background_level"]


def _radius(n: int) -> np.ndarray:
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[:n, :n]
    return np.hypot(yy - c, xx - c), yy - c, xx - c


def cutout_features(X: np.ndarray) -> np.ndarray:
    """X: (N, 2, H, W) model inputs -> (N, len(FEATURE_NAMES))."""
    n = X.shape[-1]
    r, dy, dx = _radius(n)
    snr = 3.0 * np.sinh(X[:, 0])
    morph = X[:, 1]
    inner, mid, ann = r < 2.0, r < 6.0, (r >= 10) & (r < 15)
    feats = np.zeros((len(X), len(FEATURE_NAMES)), np.float64)
    feats[:, 0] = snr[:, inner].max(1)
    feats[:, 1] = snr[:, mid].sum(1) / np.sqrt(mid.sum())
    feats[:, 2] = morph[:, inner].mean(1) / np.clip(morph[:, mid].mean(1), 1e-3, None)
    feats[:, 3] = ((morph > 0.5) & (r < 12)).sum((1, 2))
    w = np.clip(morph, 0, None) * (r < 8)
    sw = w.sum((1, 2)) + 1e-9
    mxx = (w * dx ** 2).sum((1, 2)) / sw
    myy = (w * dy ** 2).sum((1, 2)) / sw
    mxy = (w * dx * dy).sum((1, 2)) / sw
    tr, det = mxx + myy, mxx * myy - mxy ** 2
    disc = np.sqrt(np.clip(tr ** 2 / 4 - det, 0, None))
    l1, l2 = tr / 2 + disc, np.clip(tr / 2 - disc, 1e-9, None)
    feats[:, 4] = np.sqrt(l2 / l1)
    feats[:, 5] = np.median(morph[:, ann], axis=1)
    return feats
