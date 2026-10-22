"""Algorithmically synthesised impulse responses + FFT convolution.

No recorded IRs. An IR is built from:
  * a pre-delay,
  * sparse early reflections (image-source-like taps, each low-passed and placed in the stereo field),
  * a late tail of independent L/R noise whose decay depends on frequency (STFT-domain shaping),
    with an echo-density build-up at its start,
  * optional cross-feed so the result is "true stereo" (LL, LR, RL, RR).

Presets: room, hall, cathedral, plate, chamber. RT60 values are mid-band (500 Hz-1 kHz).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy import signal

from ..core import SR, raised_cos

# Per preset: rt60 mid, predelay s, ER span s, ER count, ER level, low RT multiplier, high RT
# multipliers at 4k/8k/16k, size (affects ER spacing), build-up ms, brightness of ERs (Hz)
PRESETS = {
    "room":      dict(rt60=0.55, predelay=0.004, er_span=0.035, er_count=14, er_level=0.55, low_mult=1.1,
                      hi=(0.75, 0.5, 0.3), buildup=0.010, er_lp=9000, cross=0.35),
    "chamber":   dict(rt60=1.3, predelay=0.010, er_span=0.050, er_count=16, er_level=0.45, low_mult=1.15,
                      hi=(0.7, 0.45, 0.25), buildup=0.020, er_lp=8000, cross=0.35),
    "hall":      dict(rt60=2.3, predelay=0.022, er_span=0.085, er_count=20, er_level=0.38, low_mult=1.25,
                      hi=(0.62, 0.38, 0.2), buildup=0.035, er_lp=7000, cross=0.3),
    "cathedral": dict(rt60=5.0, predelay=0.040, er_span=0.140, er_count=22, er_level=0.28, low_mult=1.35,
                      hi=(0.55, 0.32, 0.16), buildup=0.070, er_lp=5500, cross=0.3),
    "plate":     dict(rt60=1.8, predelay=0.008, er_span=0.0, er_count=0, er_level=0.0, low_mult=0.9,
                      hi=(0.95, 0.8, 0.55), buildup=0.004, er_lp=16000, cross=0.15),
}


def _rt_curve(freqs, rt_mid, low_mult, hi):
    """RT60 as a function of frequency: longer in the bass, shorter in the air bands."""
    f = np.maximum(freqs, 20.0)
    lf = np.log2(f)
    pts_f = np.log2([20, 125, 250, 1000, 4000, 8000, 16000, 24000])
    pts_m = np.array([low_mult * 1.05, low_mult, (low_mult + 1) / 2, 1.0, hi[0], hi[1], hi[2], hi[2] * 0.8])
    return rt_mid * np.interp(lf, pts_f, pts_m)


@lru_cache(maxsize=16)
def _make_ir_cached(kind, rt60, predelay, seed, sr, length_s, er_level, width, brightness, dense_start):
    p = dict(PRESETS[kind])
    if rt60 is not None:
        p["rt60"] = rt60
    if predelay is not None:
        p["predelay"] = predelay
    if er_level is not None:
        p["er_level"] = er_level
    rng = np.random.default_rng(seed)
    rt_max = p["rt60"] * p["low_mult"] * 1.05
    n = int((length_s if length_s else min(rt_max * 1.15, 12.0)) * sr)
    npd = int(p["predelay"] * sr)

    # ---- late tail: noise with frequency-dependent exponential decay (STFT shaping)
    nper = 1024
    hop = nper // 4
    late = np.zeros((2, n))
    for c in range(2):
        noise = rng.standard_normal(n)
        f, t, Z = signal.stft(noise, fs=sr, nperseg=nper, noverlap=nper - hop, boundary="zeros", padded=True)
        rt = _rt_curve(f, p["rt60"], p["low_mult"], p["hi"])
        if brightness != 1.0:
            rt = rt * np.where(f > 2000, brightness ** np.log2(np.maximum(f, 2000) / 2000), 1.0)
        decay = np.exp(-6.907755 * t[None, :] / rt[:, None])
        Z = Z * decay
        _, y = signal.istft(Z, fs=sr, nperseg=nper, noverlap=nper - hop, boundary=True)
        y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
        late[c] = y

    # echo-density build-up: the tail starts sparse (a few strong spikes) and becomes dense
    nb = max(8, int(p["buildup"] * sr))
    ramp = raised_cos(nb)
    for c in range(2):
        sparse = np.zeros(nb)
        k = max(4, int(nb / sr * 1500))
        pos = np.sort(rng.integers(0, nb, k))
        sparse[pos] = rng.standard_normal(k) * 3.0
        sparse = signal.lfilter([1.0], [1.0, -0.6], sparse)  # soften spikes a little
        mix = (1 - ramp) * sparse * np.std(late[c, :nb] + 1e-12) + ramp * late[c, :nb]
        late[c, :nb] = mix if not dense_start else late[c, :nb]
    # gentle fade-in of the late field itself (no step at its start)
    late[:, :nb] *= raised_cos(nb) ** 0.5

    # ---- early reflections
    er = np.zeros((2, n))
    if p["er_count"] > 0 and p["er_level"] > 0:
        span = int(p["er_span"] * sr)
        times = np.sort(rng.uniform(0.08, 1.0, p["er_count"]) ** 1.3) * span
        for i, tt in enumerate(times):
            g = p["er_level"] * (0.25 + 0.75 * (1 - tt / span)) * rng.uniform(0.6, 1.0) * rng.choice([-1, 1])
            pan = rng.uniform(-0.9, 0.9)
            gl = np.cos((pan + 1) * np.pi / 4)
            gr = np.sin((pan + 1) * np.pi / 4)
            # slight inter-aural delay for each reflection
            itd = int(abs(pan) * 0.0006 * sr)
            il = int(tt) + (itd if pan > 0 else 0)
            ir_ = int(tt) + (itd if pan < 0 else 0)
            if il < n:
                er[0, il] += g * gl
            if ir_ < n:
                er[1, ir_] += g * gr
        # each reflection loses highs (absorption)
        b, a = signal.butter(2, p["er_lp"], fs=sr)
        er = signal.lfilter(b, a, er, axis=-1)

    # normalise late and early energy separately so presets stay comparable
    late /= np.sqrt(np.sum(late ** 2) / 2) + 1e-12
    if np.any(er):
        er *= 0.9 / (np.sqrt(np.sum(er ** 2) / 2) + 1e-12) * p["er_level"]
    ir = late + er
    # pre-delay
    ir = np.pad(ir, ((0, 0), (npd, 0)))[:, :n]

    # stereo width (M/S) on the IR itself
    if width != 1.0:
        m = 0.5 * (ir[0] + ir[1])
        s = 0.5 * (ir[0] - ir[1]) * width
        ir = np.vstack([m + s, m - s])
    # tail fade to make the truncation silent
    k = int(0.05 * n)
    ir[:, n - k:] *= raised_cos(k)[::-1]
    # unit energy per channel
    ir /= np.sqrt(np.mean(np.sum(ir ** 2, axis=1))) + 1e-12
    return ir, p["cross"]


def make_ir(kind: str = "hall", rt60: float | None = None, predelay: float | None = None, seed: int = 7,
            sr: int = SR, length_s: float | None = None, er_level: float | None = None, width: float = 1.0,
            brightness: float = 1.0, dense_start: bool = False):
    """Synthesise a stereo IR. Returns (ir (2, n), cross-feed gain)."""
    if kind not in PRESETS:
        raise ValueError(f"reverb kind must be one of {list(PRESETS)}")
    return _make_ir_cached(kind, rt60, predelay, seed, sr, length_s, er_level, width, brightness, dense_start)


def convolve(x, ir, cross: float = 0.3):
    """True-stereo-ish FFT convolution: L*irL + R*irR, plus cross-feed L*irR', R*irL'."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = np.vstack([x, x])
    n = x.shape[1] + ir.shape[1] - 1
    out = np.zeros((2, n))
    out[0] = signal.oaconvolve(x[0], ir[0])
    out[1] = signal.oaconvolve(x[1], ir[1])
    if cross > 0:
        # cross terms use the opposite IR channel reversed in polarity-free way (different noise = decorrelated)
        out[0] += cross * signal.oaconvolve(x[1], ir[0])
        out[1] += cross * signal.oaconvolve(x[0], ir[1])
        out /= np.sqrt(1 + cross ** 2)
    return out


@dataclass
class Reverb:
    """Convolution reverb effect for buses. wet-only output (use as a send/return)."""
    kind: str = "hall"
    rt60: float | None = None
    predelay: float | None = None
    seed: int = 7
    width: float = 1.0
    brightness: float = 1.0
    lowcut: float = 80.0      # high-pass on the reverb input (keeps the bass clean)
    highcut: float = 12000.0  # low-pass on the reverb input
    gain_db: float = 0.0
    er_level: float | None = None

    def ir(self, sr=SR):
        return make_ir(self.kind, self.rt60, self.predelay, self.seed, sr, None, self.er_level, self.width,
                       self.brightness)

    def process(self, x, ctx=None):
        sr = SR
        ir, cross = self.ir(sr)
        y = x
        if self.lowcut:
            y = signal.sosfilt(signal.butter(2, self.lowcut, "highpass", fs=sr, output="sos"), y, axis=-1)
        if self.highcut:
            y = signal.sosfilt(signal.butter(2, self.highcut, "lowpass", fs=sr, output="sos"), y, axis=-1)
        out = convolve(y, ir, cross)[:, : x.shape[1]]
        return out * 10 ** (self.gain_db / 20)


def reverb(x, kind="hall", wet: float = 0.3, dry: float = 1.0, **kw):
    """Convenience: dry/wet mix of a signal with a synthesised reverb (output length = input length)."""
    rv = Reverb(kind=kind, **kw)
    return dry * np.asarray(x) + wet * rv.process(x)
