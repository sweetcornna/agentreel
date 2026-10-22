"""Shared deterministic DSP helpers (no generative models anywhere in this package)."""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy import signal

SR = 48000


def db2a(db):
    return 10.0 ** (np.asarray(db, dtype=np.float64) / 20.0)


def a2db(a, floor=1e-12):
    return 20.0 * np.log10(np.maximum(np.abs(a), floor))


def midi_to_hz(m, a4=440.0):
    return a4 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


def stereo(x) -> np.ndarray:
    """Return a float32 (n, 2) view/copy of mono or stereo input."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return np.repeat(x[:, None], 2, axis=1)
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    if x.shape[1] > 2:
        return x[:, :2]
    return x


def pan_gains(pan: float):
    """Constant-power pan law; pan in [-1 (left), +1 (right)]."""
    th = (np.clip(pan, -1.0, 1.0) + 1.0) * np.pi / 4.0
    return float(np.cos(th)), float(np.sin(th))


def cos_ramp(n: int, rising=True) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    r = 0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)
    return (r if rising else r[::-1]).astype(np.float32)


def fade_edges(x: np.ndarray, fade_in_s=0.005, fade_out_s=0.005) -> np.ndarray:
    """Raised-cosine fades so that the first/last samples are exactly zero."""
    x = np.array(x, dtype=np.float32, copy=True)
    n = len(x)
    if n == 0:
        return x
    a = min(int(round(fade_in_s * SR)), n)
    b = min(int(round(fade_out_s * SR)), n)
    shape = (-1,) + (1,) * (x.ndim - 1)
    if a:
        x[:a] *= cos_ramp(a).reshape(shape)
    if b:
        x[n - b:] *= cos_ramp(b, rising=False).reshape(shape)
    x[0] = 0.0
    x[-1] = 0.0
    return x


def add_at(buf: np.ndarray, x: np.ndarray, start: int):
    """Mix x into buf at sample index start (clips to buffer bounds)."""
    if start >= len(buf) or start + len(x) <= 0:
        return
    s0 = max(0, -start)
    b0 = max(0, start)
    n = min(len(x) - s0, len(buf) - b0)
    if n > 0:
        buf[b0:b0 + n] += x[s0:s0 + n]


# ---------------------------------------------------------------- resampling
@lru_cache(maxsize=256)
def _rational(ratio: float, tol_cents=0.02):
    for den in (64, 128, 256, 512, 1000, 2000, 4000):
        f = Fraction(ratio).limit_denominator(den)
        if f.numerator and abs(1200 * np.log2(float(f) / ratio)) < tol_cents:
            return f.numerator, f.denominator
    f = Fraction(ratio).limit_denominator(8000)
    return f.numerator, f.denominator


def resample_by(x: np.ndarray, ratio: float) -> np.ndarray:
    """Change playback speed by `ratio` (>1 = higher pitch, shorter).

    Polyphase windowed-sinc (scipy resample_poly, Kaiser beta=9) with a rational
    approximation of 1/ratio that is accurate to < 0.02 cent.
    """
    if abs(ratio - 1.0) < 1e-9:
        return np.asarray(x, dtype=np.float32)
    up, down = _rational(1.0 / ratio)
    y = signal.resample_poly(np.asarray(x, dtype=np.float64), up, down, axis=0,
                             window=('kaiser', 9.0))
    return y.astype(np.float32)


def resample_sr(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    f = Fraction(sr_out, sr_in)
    y = signal.resample_poly(np.asarray(x, dtype=np.float64), f.numerator, f.denominator,
                             axis=0, window=('kaiser', 9.0))
    return y.astype(np.float32)


# ---------------------------------------------------------------- filters (RBJ cookbook)
def biquad(kind: str, f0: float, q: float = 0.7071, gain_db: float = 0.0, sr=SR):
    """Return an (1, 6) SOS section. kind: lowpass highpass bandpass peak lowshelf highshelf notch."""
    f0 = float(np.clip(f0, 5.0, sr * 0.49))
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    c, s = np.cos(w0), np.sin(w0)
    alpha = s / (2 * q)
    if kind == 'lowpass':
        b = [(1 - c) / 2, 1 - c, (1 - c) / 2]
        a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == 'highpass':
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
        a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == 'bandpass':  # constant 0 dB peak gain
        b = [alpha, 0, -alpha]
        a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == 'notch':
        b = [1, -2 * c, 1]
        a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == 'peak':
        b = [1 + alpha * A, -2 * c, 1 - alpha * A]
        a = [1 + alpha / A, -2 * c, 1 - alpha / A]
    elif kind in ('lowshelf', 'highshelf'):
        sq = 2 * np.sqrt(A) * alpha
        if kind == 'lowshelf':
            b = [A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c),
                 A * ((A + 1) - (A - 1) * c - sq)]
            a = [(A + 1) + (A - 1) * c + sq, -2 * ((A - 1) + (A + 1) * c),
                 (A + 1) + (A - 1) * c - sq]
        else:
            b = [A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c),
                 A * ((A + 1) + (A - 1) * c - sq)]
            a = [(A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c),
                 (A + 1) - (A - 1) * c - sq]
    else:
        raise ValueError(kind)
    b = np.asarray(b) / a[0]
    a = np.asarray(a) / a[0]
    return np.concatenate([b, a])[None, :]


def sosfilt(sos, x, zero_phase=False):
    x = np.asarray(x, dtype=np.float64)
    if zero_phase:
        return signal.sosfiltfilt(sos, x, axis=0).astype(np.float32)
    return signal.sosfilt(sos, x, axis=0).astype(np.float32)


def butter(kind: str, fc, order=2, sr=SR):
    """Butterworth SOS; kind lowpass/highpass/bandpass (fc = (lo, hi) for bandpass)."""
    nyq = sr / 2
    if kind == 'bandpass':
        lo, hi = fc
        return signal.butter(order, [max(lo, 1.0) / nyq, min(hi, nyq * 0.98) / nyq],
                             btype='bandpass', output='sos')
    return signal.butter(order, min(max(fc, 1.0), nyq * 0.98) / nyq, btype=kind, output='sos')


def one_pole_lp(x, fc, sr=SR):
    a = np.exp(-2 * np.pi * fc / sr)
    return signal.lfilter([1 - a], [1, -a], np.asarray(x, dtype=np.float64), axis=0)


def smooth_noise(n: int, rate_hz: float, rng, sr=SR, order=2):
    """Unit-RMS low-frequency random signal (filtered white noise), length n."""
    if n <= 0:
        return np.zeros(0)
    pad = int(sr / max(rate_hz, 0.05) * 2)
    w = rng.standard_normal(n + pad)
    sos = signal.butter(order, max(rate_hz, 0.01) / (sr / 2), output='sos')
    y = signal.sosfilt(sos, w)[pad:]
    s = np.std(y)
    return y / s if s > 0 else y
