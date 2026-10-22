"""Classic subtractive / analog-style synthesis. No samples, no models: oscillators, noise, envelopes and
filters, the way a drum machine and a polysynth make their sounds.

Written for a product-film idiom: a steady pulse (soft kick and bass on the beat), bright plucked chords and airy
hats in the full sections, a thin "thinking" texture where the picture slows, low downsweeps into section changes,
risers, and a heavy sub hit on the reveals. Everything here returns float32 stereo (n, 2) at sampler.SR, first and last sample zero, deterministic by seed.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from ._dsp import SR

__all__ = ['kick', 'clap', 'hat', 'shaker', 'bass', 'pluck', 'keys', 'riser', 'downsweep', 'blip', 'sub_hit',
           'sidechain', 'midi_hz', 'tick']


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def _st(x):
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    return x


def _edge(x, fi=0.001, fo=0.004):
    n = len(x)
    a = min(n, max(1, int(fi * SR)))
    b = min(n, max(1, int(fo * SR)))
    x = x.copy()
    x[:a] *= np.linspace(0, 1, a)[:, None] if x.ndim == 2 else np.linspace(0, 1, a)
    x[n - b:] *= np.linspace(1, 0, b)[:, None] if x.ndim == 2 else np.linspace(1, 0, b)
    return x


def _lp(x, fc, order=2):
    sos = butter(order, min(fc, SR * 0.45), 'low', fs=SR, output='sos')
    return sosfilt(sos, x, axis=0)


def _hp(x, fc, order=2):
    sos = butter(order, max(10.0, fc), 'high', fs=SR, output='sos')
    return sosfilt(sos, x, axis=0)


def _bp(x, lo, hi, order=2):
    sos = butter(order, [max(10.0, lo), min(hi, SR * 0.45)], 'band', fs=SR, output='sos')
    return sosfilt(sos, x, axis=0)


def _tvlp(x, fc_curve, q=0.7, block=64):
    """time-varying resonant 2-pole lowpass: RBJ biquad, coefficients updated every `block` samples, state carried"""
    from scipy.signal import lfilter
    y = np.zeros_like(x)
    zi = np.zeros(2)
    fc_curve = np.clip(fc_curve, 20, SR * 0.45)
    for s0 in range(0, len(x), block):
        s1 = min(len(x), s0 + block)
        w0 = 2 * np.pi * fc_curve[s0] / SR
        alpha = np.sin(w0) / (2 * q)
        cw = np.cos(w0)
        b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
        a = np.array([1 + alpha, -2 * cw, 1 - alpha])
        y[s0:s1], zi = lfilter(b / a[0], a / a[0], x[s0:s1], zi=zi)
    return y


def _saw(freq, n, phase=0.0):
    """band-limited-ish saw by PolyBLEP"""
    t = (phase + np.cumsum(np.full(n, freq / SR))) % 1.0
    dt = freq / SR
    y = 2 * t - 1
    # PolyBLEP
    m1 = t < dt
    tt = t[m1] / dt
    y[m1] -= tt + tt - tt * tt - 1
    m2 = t > 1 - dt
    tt = (t[m2] - 1) / dt
    y[m2] -= tt * tt + tt + tt + 1
    return y


def _tri(freq, n, phase=0.0):
    t = (phase + np.cumsum(np.full(n, freq / SR))) % 1.0
    return 4 * np.abs(t - 0.5) - 1


# ----------------------------------------------------------------------------- drums
def kick(vel: float = 1.0, *, tone: float = 0.5, length: float = 0.42, seed: int = 0) -> np.ndarray:
    """A soft, round kick: a sine whose pitch falls fast from ~140 Hz to ~46 Hz, a felt-mallet click, a little
    saturation. `tone` 0 = deep and soft, 1 = tighter and punchier."""
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    f_hi, f_lo = 120 + 50 * tone, 44 + 6 * tone
    f = f_lo + (f_hi - f_lo) * np.exp(-t / (0.028 + 0.02 * (1 - tone)))
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / (0.16 + 0.12 * (1 - tone)))
    click = rng.standard_normal(n) * np.exp(-t / 0.0025)
    click = _bp(click, 1200, 5000) * (0.25 + 0.3 * tone)
    y = np.tanh((body + click) * (1.2 + 0.8 * tone)) * vel
    return _st(_edge(y.astype(np.float32), 0.0005, 0.02))


def clap(vel: float = 1.0, *, seed: int = 0, length: float = 0.32) -> np.ndarray:
    """A soft hand clap: three quick noise bursts through a band-pass, then a short room tail."""
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    env = np.zeros(n)
    for k, d in enumerate((0.0, 0.009, 0.018)):
        env += (t >= d) * np.exp(-np.maximum(0, t - d) / (0.006 if k < 2 else 0.06))
    noise = rng.standard_normal((n, 2))
    y = _bp(noise, 900, 3200) * env[:, None]
    y += _bp(noise, 1500, 6000) * (np.exp(-t / 0.12) * 0.12)[:, None]
    return _edge((y * vel * 0.8).astype(np.float32))


def hat(vel: float = 1.0, *, open_: bool = False, seed: int = 0, bright: float = 0.6) -> np.ndarray:
    """An analog hi-hat: six detuned square waves (the classic metallic cluster) plus noise, band-passed high."""
    rng = np.random.default_rng(seed)
    length = 0.36 if open_ else 0.08
    n = int(length * SR)
    t = np.arange(n) / SR
    fr = np.array([205.3, 304.4, 369.6, 522.7, 540.0, 800.0]) * (1 + 0.01 * rng.standard_normal(6))
    sq = sum(np.sign(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))) for f in fr) / 6
    y = 0.55 * sq + 0.45 * rng.standard_normal(n)
    y = _hp(_bp(y, 6000 + 2500 * bright, 16000), 7000)
    env = np.exp(-t / (0.11 if open_ else 0.016))
    y = y * env * vel
    st = np.stack([y * (1 + 0.1 * rng.uniform(-1, 1)), y * (1 + 0.1 * rng.uniform(-1, 1))], 1)
    return _edge(st.astype(np.float32), 0.0003, 0.005)


def shaker(vel: float = 1.0, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    env = np.minimum(1, t / 0.012) * np.exp(-np.maximum(0, t - 0.012) / 0.03)
    y = _bp(rng.standard_normal((n, 2)), 4500, 12000) * env[:, None] * vel
    return _edge(y.astype(np.float32))


def tick(vel: float = 1.0, *, freq: float = 2400, seed: int = 0) -> np.ndarray:
    """a tiny woody tick (a rim or a clock)"""
    rng = np.random.default_rng(seed)
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    y = np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.006) + 0.3 * rng.standard_normal(n) * np.exp(-t / 0.002)
    return _st(_edge((_hp(y, 600) * vel).astype(np.float32)))


# ----------------------------------------------------------------------------- tonal
def bass(midi: float, dur: float, vel: float = 1.0, *, bright: float = 0.35, glide_from: float | None = None) -> np.ndarray:
    """A warm synth bass: sine sub plus a filtered saw, plucky envelope, mono in the middle."""
    n = int((dur + 0.12) * SR)
    t = np.arange(n) / SR
    f = np.full(n, midi_hz(midi))
    if glide_from is not None:
        f = midi_hz(midi) + (midi_hz(glide_from) - midi_hz(midi)) * np.exp(-t / 0.04)
    ph = 2 * np.pi * np.cumsum(f) / SR
    sub = np.sin(ph)
    saw = _saw(midi_hz(midi), n)
    fc = 180 + (500 + 1600 * bright) * np.exp(-t / 0.09)
    body = _tvlp(saw * 0.55, fc, q=0.9)
    env = np.minimum(1, t / 0.004) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.03)) * (0.75 + 0.25 * np.exp(-t / 0.25))
    y = np.tanh((sub * 0.9 + body) * 1.3) * env * vel
    return _st(_edge(y.astype(np.float32), 0.001, 0.02))


def pluck(midis, vel: float = 1.0, *, decay: float = 0.45, bright: float = 0.55, detune: float = 0.08,
          spread: float = 0.6, seed: int = 0, length: float | None = None) -> np.ndarray:
    """A polysynth pluck: per note two detuned saws and a triangle an octave down, through a lowpass whose cutoff
    snaps shut; notes spread across the stereo field."""
    rng = np.random.default_rng(seed)
    midis = [midis] if np.isscalar(midis) else list(midis)
    length = length or decay * 3.2 + 0.1
    n = int(length * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for k, m in enumerate(midis):
        f = midi_hz(m)
        osc = (_saw(f * (1 + detune / 100 * 1.0), n, rng.uniform()) + _saw(f * (1 - detune / 100), n, rng.uniform())) * 0.5
        osc += 0.35 * _tri(f / 2, n, rng.uniform())
        fc = 300 + f * 0.8 + (1500 + 6000 * bright) * np.exp(-t / (decay * 0.35))
        y = _tvlp(osc, fc, q=0.8)
        env = np.minimum(1, t / 0.003) * np.exp(-t / decay)
        y = y * env
        p = (k / max(1, len(midis) - 1) - 0.5) * spread if len(midis) > 1 else 0.0
        out[:, 0] += y * np.cos((p + 1) * np.pi / 4)
        out[:, 1] += y * np.sin((p + 1) * np.pi / 4)
    out *= vel / max(1, np.sqrt(len(midis)))
    return _edge(out.astype(np.float32), 0.0005, 0.03)


def keys(midis, dur: float, vel: float = 1.0, *, bright: float = 0.35, seed: int = 0) -> np.ndarray:
    """A soft electric-piano-like tone (sine + a touch of FM bell on the attack) for held chords."""
    rng = np.random.default_rng(seed)
    midis = [midis] if np.isscalar(midis) else list(midis)
    n = int((dur + 1.2) * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for k, m in enumerate(midis):
        f = midi_hz(m)
        mod = np.sin(2 * np.pi * f * 14 * t) * (1.2 * bright) * np.exp(-t / 0.05)
        y = np.sin(2 * np.pi * f * t + mod + rng.uniform(0, 6.28)) + 0.18 * np.sin(4 * np.pi * f * t) * np.exp(-t / 0.4)
        env = np.minimum(1, t / 0.006) * np.exp(-t / 2.2) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.35))
        y = y * env
        p = (k / max(1, len(midis) - 1) - 0.5) * 0.7 if len(midis) > 1 else 0.0
        out[:, 0] += y * np.cos((p + 1) * np.pi / 4)
        out[:, 1] += y * np.sin((p + 1) * np.pi / 4)
    out *= vel / max(1, np.sqrt(len(midis)))
    return _edge(out.astype(np.float32), 0.001, 0.05)


# ----------------------------------------------------------------------------- transitions
def riser(dur: float, *, f0: float = 400, f1: float = 6000, seed: int = 0, tonal_midi: float | None = None) -> np.ndarray:
    """Filtered noise sweeping up, growing into its end; optionally a saw gliding up an octave with it."""
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    u = np.linspace(0, 1, n)
    noise = rng.standard_normal(n)
    fc = f0 * (f1 / f0) ** (u ** 1.6)
    y = _tvlp(noise, fc, q=2.0) * 0.6
    y = y - _tvlp(y, fc * 0.4, q=0.7) * 0.8   # band-ish
    if tonal_midi is not None:
        f = midi_hz(tonal_midi) * 2 ** u
        y += 0.25 * np.sin(2 * np.pi * np.cumsum(f) / SR) * u
    env = u ** 2.4
    y = y * env
    st = np.stack([y, np.roll(y, int(0.007 * SR))], 1)
    st = st / (np.max(np.abs(st)) + 1e-9) * 0.6
    return _edge(st.astype(np.float32), 0.05, 0.004)


def downsweep(dur: float = 1.2, *, f0: float = 220, f1: float = 38, steps: int = 0, drive: float = 1.6) -> np.ndarray:
    """A low sine falling from f0 to f1 (into a section boundary); `steps` > 0 makes it fall in
    stepped notes instead of a smooth slide."""
    n = int(dur * SR)
    u = np.linspace(0, 1, n)
    if steps:
        u_q = np.floor(u * steps) / steps
        f = f0 * (f1 / f0) ** u_q
    else:
        f = f0 * (f1 / f0) ** (u ** 0.7)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR)
    y = np.tanh(y * drive) / np.tanh(drive)
    env = np.minimum(1, u / 0.03) * (1 - u) ** 1.2
    return _st(_edge((y * env * 0.7).astype(np.float32), 0.002, 0.02))


def sub_hit(midi: float = 26, *, length: float = 2.4, drive: float = 1.4, seed: int = 0) -> np.ndarray:
    """The reveal's weight: a kick transient into a long pitched sub (808-like), with a noise breath on top."""
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    f = midi_hz(midi) * (1 + 1.8 * np.exp(-t / 0.03))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (length * 0.38))
    y = np.tanh(y * drive) / np.tanh(drive)
    air = _bp(rng.standard_normal(n), 200, 2500) * np.exp(-t / 0.35) * 0.12
    return _st(_edge((y + air).astype(np.float32) * 0.9, 0.0005, 0.1))


def blip(midi: float = 96, vel: float = 1.0, *, up: float = 5.0, length: float = 0.07) -> np.ndarray:
    """a short rising digital blip (a tiny sound for text appearing)"""
    n = int(length * SR)
    t = np.arange(n) / SR
    f = midi_hz(midi) * 2 ** (up / 12 * np.minimum(1, t / (length * 0.5)))
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (length * 0.35)) * vel
    return _st(_edge(y.astype(np.float32), 0.0005, 0.005))


# ----------------------------------------------------------------------------- mix helper
def sidechain(x: np.ndarray, hits_s, *, depth_db: float = 7.0, release: float = 0.16) -> np.ndarray:
    """duck x under each kick: gain dips by depth_db at each hit and recovers over `release` s"""
    n = len(x)
    g = np.zeros(n)
    t_rel = np.arange(int(release * 3 * SR)) / SR
    shape = np.exp(-t_rel / (release * 0.5))
    for h in hits_s:
        a = int(h * SR)
        if a >= n:
            continue
        b = min(n, a + len(shape))
        g[a:b] = np.maximum(g[a:b], shape[: b - a])
    gain = 10 ** (-depth_db * g / 20)
    return (x * gain[:, None]).astype(np.float32)
