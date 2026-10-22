"""Stereo tools: equal-power pan (static or automated), M/S width, Haas widening, correlation."""
from __future__ import annotations

import numpy as np

from ..core import SR, stereo


def pan(x, p):
    """Equal-power pan. x mono or stereo; p scalar or per-sample array in [-1, 1].

    For stereo input this is a balance control that keeps the centre at unity (equal power law).
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), -1.0, 1.0)
    th = (p + 1.0) * np.pi / 4.0
    gl, gr = np.cos(th) * np.sqrt(2), np.sin(th) * np.sqrt(2)
    if x.ndim == 1:
        return np.vstack([x * gl / np.sqrt(2), x * gr / np.sqrt(2)])
    return np.vstack([x[0] * np.minimum(gl, 1.0), x[1] * np.minimum(gr, 1.0)])


def width(x, w: float):
    """M/S width: 0 = mono, 1 = unchanged, >1 = wider (side boosted)."""
    x = stereo(x)
    m = 0.5 * (x[0] + x[1])
    s = 0.5 * (x[0] - x[1]) * w
    return np.vstack([m + s, m - s])


def haas(x, delay_ms: float = 12.0, side: str = "right", level: float = 1.0):
    """Haas widening of a mono or stereo source: delay one channel by a few ms."""
    x = stereo(x).copy()
    k = int(delay_ms * 1e-3 * SR)
    c = 1 if side == "right" else 0
    x[c] = np.concatenate([np.zeros(k), x[c][: x.shape[1] - k]]) * level
    return x


def correlation(x):
    """Pearson correlation between L and R (1 = mono, 0 = uncorrelated, <0 = phasey)."""
    x = stereo(x)
    a, b = x[0] - x[0].mean(), x[1] - x[1].mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / den) if den > 0 else 1.0


def side_ratio_db(x):
    """Side energy relative to mid energy in dB."""
    x = stereo(x)
    m = 0.5 * (x[0] + x[1])
    s = 0.5 * (x[0] - x[1])
    return float(10 * np.log10((np.sum(s * s) + 1e-20) / (np.sum(m * m) + 1e-20)))
