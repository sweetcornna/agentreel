"""Additive modal synthesis core shared by bells, glass, gongs, drums and timpani.

Each mode (partial) has frequency, amplitude, T60, start phase, stereo gains, onset smoothing,
an optional "bloom" (delayed rise, used for tam-tams), an optional initial pitch offset that relaxes
exponentially (membrane tension modulation after a strike) and a start delay.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from ..core import SR


@njit(cache=True, fastmath=True)
def _modes(n, freqs, amps, t60s, phases, gl, gr, attack, bloom, glide, glide_tau, delay, floor, sr):
    out = np.zeros((2, n))
    two_pi = 2.0 * np.pi
    for k in range(freqs.shape[0]):
        f = freqs[k]
        if f <= 0.0 or f * (1.0 + max(glide[k], 0.0)) >= sr * 0.48:
            continue
        w = two_pi * f / sr
        dec = np.exp(-6.907755 / (t60s[k] * sr))
        env = amps[k]
        if env == 0.0:
            continue
        thr = floor * abs(amps[k])
        ae = 1.0
        adec = np.exp(-1.0 / (max(attack[k], 1e-5) * sr))
        be = 1.0
        bdec = np.exp(-1.0 / (max(bloom[k], 1e-5) * sr)) if bloom[k] > 0 else 0.0
        ge = 1.0
        gdec = np.exp(-1.0 / (max(glide_tau[k], 1e-5) * sr))
        gk = glide[k]
        ph = phases[k]
        i0 = int(delay[k] * sr)
        for i in range(i0, n):
            ph += w * (1.0 + gk * ge)
            ge *= gdec
            a = env * (1.0 - ae)
            if bloom[k] > 0:
                a *= (1.0 - be)
                be *= bdec
            v = a * np.sin(ph)
            out[0, i] += v * gl[k]
            out[1, i] += v * gr[k]
            env *= dec
            ae *= adec
            if abs(env) < thr and i > i0 + 64 and (bloom[k] <= 0 or be < 0.01):
                break
    return out


def render_modes(n, freqs, amps, t60s, rng=None, pans=None, attack=0.0005, bloom=0.0, glide=0.0,
                 glide_tau=0.05, delay=0.0, phases=None, floor=1e-5, sr=SR):
    """Render a bank of decaying sinusoids to a (2, n) array. Scalars broadcast to every mode."""
    k = len(freqs)
    b = lambda v: np.ascontiguousarray(np.broadcast_to(np.asarray(v, dtype=np.float64), (k,)).copy())  # noqa: E731
    freqs = b(freqs)
    if rng is None:
        rng = np.random.default_rng(0)
    if phases is None:
        phases = np.zeros(k)  # struck modes start as sin(wt): physically correct, no step at t=0
    elif isinstance(phases, str) and phases == "random":
        phases = rng.uniform(0, 2 * np.pi, k)
    if pans is None:
        pans = np.zeros(k)
    th = (np.clip(b(pans), -1, 1) + 1) * np.pi / 4
    return _modes(int(n), freqs, b(amps), np.maximum(b(t60s), 1e-3), b(phases), np.cos(th), np.sin(th),
                  b(attack), b(bloom), b(glide), b(glide_tau), b(delay), float(floor), float(sr))


def doublets(freqs, amps, t60s, split_hz, rng, balance=0.5):
    """Split each mode into a close pair (mode degeneracy) to produce natural slow beating."""
    f2, a2, t2 = [], [], []
    for f, a, t, s in zip(freqs, amps, t60s, np.broadcast_to(split_hz, np.shape(freqs))):
        if s <= 0:
            f2.append(f); a2.append(a); t2.append(t)
            continue
        bal = np.clip(balance + 0.15 * rng.standard_normal(), 0.15, 0.85)
        f2 += [f - s / 2, f + s / 2]
        a2 += [a * bal, a * (1 - bal)]
        t2 += [t * rng.uniform(0.9, 1.1), t * rng.uniform(0.9, 1.1)]
    return np.array(f2), np.array(a2), np.array(t2)
