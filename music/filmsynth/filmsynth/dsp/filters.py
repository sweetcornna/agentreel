"""Filters: RBJ biquads (static), time-varying TPT state-variable filter, one-poles, modal banks.

Static filters accept arrays shaped (n,) or (channels, n) and filter along the last axis.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy import signal

from ..core import SR


# ---------------------------------------------------------------------------
# RBJ cookbook biquads
# ---------------------------------------------------------------------------
def rbj(kind: str, fc: float, q: float = 0.7071, gain_db: float = 0.0, sr: int = SR):
    """Return normalised (b, a) for an RBJ cookbook biquad.

    kind: lowpass, highpass, bandpass (0 dB peak), notch, peak, lowshelf, highshelf, allpass.
    For shelves q is the shelf slope S (1.0 = steepest without overshoot).
    """
    fc = float(np.clip(fc, 1.0, sr * 0.49))
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / sr
    cw, sw = np.cos(w0), np.sin(w0)
    if kind in ("lowshelf", "highshelf"):
        S = q
        alpha = sw / 2 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)
    else:
        alpha = sw / (2 * q)
    if kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bandpass":
        b = [alpha, 0.0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "notch":
        b = [1.0, -2 * cw, 1.0]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "allpass":
        b = [1 - alpha, -2 * cw, 1 + alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowshelf":
        sa = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
        a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
    elif kind == "highshelf":
        sa = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    else:
        raise ValueError(f"unknown biquad kind {kind!r}")
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]
    return b, a


def biquad(x, kind: str, fc: float, q: float = 0.7071, gain_db: float = 0.0, sr: int = SR):
    b, a = rbj(kind, fc, q, gain_db, sr)
    return signal.lfilter(b, a, x, axis=-1)


def lowpass(x, fc, q=0.7071, sr=SR):
    return biquad(x, "lowpass", fc, q, sr=sr)


def highpass(x, fc, q=0.7071, sr=SR):
    return biquad(x, "highpass", fc, q, sr=sr)


def bandpass(x, fc, q=1.0, sr=SR):
    return biquad(x, "bandpass", fc, q, sr=sr)


def peak(x, fc, gain_db, q=1.0, sr=SR):
    return biquad(x, "peak", fc, q, gain_db, sr=sr)


def lowshelf(x, fc, gain_db, s=1.0, sr=SR):
    return biquad(x, "lowshelf", fc, s, gain_db, sr=sr)


def highshelf(x, fc, gain_db, s=1.0, sr=SR):
    return biquad(x, "highshelf", fc, s, gain_db, sr=sr)


def eq(x, bands, sr=SR):
    """Apply a list of (kind, fc, q, gain_db) biquads in series."""
    y = x
    for band in bands:
        kind, fc = band[0], band[1]
        q = band[2] if len(band) > 2 else 0.7071
        g = band[3] if len(band) > 3 else 0.0
        y = biquad(y, kind, fc, q, g, sr)
    return y


def butter(x, kind: str, fc, order: int = 4, sr: int = SR, zero_phase: bool = False):
    sos = signal.butter(order, fc, btype=kind, fs=sr, output="sos")
    if zero_phase:
        return signal.sosfiltfilt(sos, x, axis=-1)
    return signal.sosfilt(sos, x, axis=-1)


def dc_block(x, fc: float = 8.0, sr: int = SR):
    """Remove DC and sub-sonic drift with a 2nd-order Butterworth high-pass."""
    sos = signal.butter(2, fc, btype="highpass", fs=sr, output="sos")
    return signal.sosfilt(sos, x, axis=-1)


# ---------------------------------------------------------------------------
# Time-varying filters (numba)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _svf_tv(x, fc, res, mode, sr):
    """Zavalishin TPT state-variable filter, per-sample cutoff. mode 0 LP, 1 BP, 2 HP, 3 notch, 4 BP(peak=1)."""
    n = x.shape[0]
    y = np.empty(n)
    ic1 = 0.0
    ic2 = 0.0
    k = 1.0 / res
    nyq = sr * 0.49
    for i in range(n):
        f = fc[i]
        if f > nyq:
            f = nyq
        if f < 1.0:
            f = 1.0
        g = np.tan(np.pi * f / sr)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = x[i] - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if mode == 0:
            y[i] = v2
        elif mode == 1:
            y[i] = v1
        elif mode == 2:
            y[i] = x[i] - k * v1 - v2
        elif mode == 3:
            y[i] = x[i] - k * v1
        else:
            y[i] = k * v1
    return y


_SVF_MODES = {"lowpass": 0, "bandpass": 1, "highpass": 2, "notch": 3, "bandpass_norm": 4}


def svf(x, fc, q: float = 0.7071, mode: str = "lowpass", sr: int = SR):
    """Time-varying state-variable filter. fc may be scalar or per-sample array.

    Works on mono or (channels, n). 'bandpass_norm' has unity gain at the centre.
    """
    x = np.asarray(x, dtype=np.float64)
    m = _SVF_MODES[mode]
    if x.ndim == 1:
        f = np.broadcast_to(np.asarray(fc, dtype=np.float64), x.shape).copy()
        return _svf_tv(x, f, float(q), m, float(sr))
    out = np.empty_like(x)
    f = np.broadcast_to(np.asarray(fc, dtype=np.float64), x.shape[-1:]).copy()
    for c in range(x.shape[0]):
        out[c] = _svf_tv(np.ascontiguousarray(x[c]), f, float(q), m, float(sr))
    return out


@njit(cache=True, fastmath=True)
def _onepole_tv(x, coef):
    n = x.shape[0]
    y = np.empty(n)
    s = 0.0
    for i in range(n):
        s = s + coef[i] * (x[i] - s)
        y[i] = s
    return y


def onepole_lp(x, fc, sr: int = SR):
    """One-pole low-pass, fc scalar or per-sample array (6 dB/oct, no resonance)."""
    x = np.asarray(x, dtype=np.float64)
    f = np.broadcast_to(np.asarray(fc, dtype=np.float64), x.shape[-1:])
    coef = 1.0 - np.exp(-2.0 * np.pi * np.clip(f, 1.0, sr * 0.49) / sr)
    if x.ndim == 1:
        return _onepole_tv(x, np.ascontiguousarray(coef))
    return np.vstack([_onepole_tv(np.ascontiguousarray(c), np.ascontiguousarray(coef)) for c in x])


@njit(cache=True, fastmath=True)
def _modal_bank(x, freqs, t60s, gains, sr):
    """Parallel two-pole resonators (impulse-invariant). Output sum. Each resonator ~unit peak gain."""
    n = x.shape[0]
    m = freqs.shape[0]
    y = np.zeros(n)
    for k in range(m):
        f = freqs[k]
        if f <= 0.0 or f >= sr * 0.49:
            continue
        r = np.exp(-6.907755 / (t60s[k] * sr))
        w = 2.0 * np.pi * f / sr
        a1 = -2.0 * r * np.cos(w)
        a2 = r * r
        # normalise so that the peak gain is about gains[k]
        b0 = gains[k] * (1.0 - r) * np.sqrt(1.0 - 2.0 * r * np.cos(2.0 * w) + r * r) / np.sin(w) * 0.5
        s1 = 0.0
        s2 = 0.0
        for i in range(n):
            v = b0 * x[i] - a1 * s1 - a2 * s2
            s2 = s1
            s1 = v
            y[i] += v - s2  # (1 - z^-2) zero pair gives a bandpass shape, no DC
    return y


def modal_bank(x, freqs, t60s, gains, sr: int = SR):
    """Filter x through parallel resonators (freqs Hz, decay T60 seconds, linear gains)."""
    return _modal_bank(np.ascontiguousarray(x, dtype=np.float64), np.asarray(freqs, dtype=np.float64),
                       np.asarray(t60s, dtype=np.float64), np.asarray(gains, dtype=np.float64), float(sr))


def fir_from_modes(freqs, t60s, gains, length_s: float, rng, sr: int = SR):
    """Impulse response made of decaying sinusoids (used for body resonances, convolved by FFT)."""
    n = int(length_s * sr)
    t = np.arange(n) / sr
    h = np.zeros(n)
    for f, d, g in zip(freqs, t60s, gains):
        if f >= sr * 0.45:
            continue
        ph = rng.uniform(0, 2 * np.pi)
        h += g * np.exp(-6.907755 * t / d) * np.sin(2 * np.pi * f * t + ph)
    return h
