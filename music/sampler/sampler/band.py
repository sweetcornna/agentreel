"""A small synthetic band: tuned percussion by modal synthesis, a whistled lead, a small drum kit and a picked
bass, all classic DSP (oscillators, noise, resonant modes, envelopes, filters). No samples, no models; the sampled
instruments of the score (Salamander piano, VSCO 2 strings, VCSL wood and chimes) stay in piano.py / strings.py /
nature.py.

    marimba(midi, vel, hardness, seed)     rosewood bar over a tuned tube: partials 1 : 4 : 9.9, short
    vibes(midi, vel, dur, trem, seed)      aluminium bar, long ring, optional motor tremolo, damped at dur
    glock(midi, vel, seed)                 small steel bar: free-bar partials 1 : 2.76 : 5.40 : 8.93, bright
    kalimba(midi, vel, seed)               a thumb-piano tine: 1 : 5.4 : 8.9, woody knock on the attack
    whistle(notes, ...)                    one continuous whistled line with glides, breath and late vibrato
    bass(midi, dur, vel, ...)              a round picked bass: sine + filtered saw, short and tuned
    kick / snare / clap / snap / hat / shaker / rim

Every voice returns float32 stereo (n, 2) at sampler.SR with a zero first and last sample and is deterministic by seed.
Tuned percussion is spread across the stereo field by pitch (low left, high right, as a player hears the bars).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from ._dsp import SR

__all__ = ['marimba', 'vibes', 'glock', 'kalimba', 'whistle', 'bass', 'kick', 'snare', 'clap', 'snap', 'hat',
           'shaker', 'rim', 'midi_hz', 'bar_pan']


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def _edge(x, fi=0.0005, fo=0.004):
    n = len(x)
    a = min(n, max(1, int(fi * SR)))
    b = min(n, max(1, int(fo * SR)))
    x = x.copy()
    ra = np.linspace(0, 1, a)
    rb = np.linspace(1, 0, b)
    if x.ndim == 2:
        x[:a] *= ra[:, None]
        x[n - b:] *= rb[:, None]
    else:
        x[:a] *= ra
        x[n - b:] *= rb
    return x


def _sos(kind, fc, order=2):
    if isinstance(fc, (list, tuple)):
        fc = [max(10.0, fc[0]), min(fc[1], SR * 0.45)]
    else:
        fc = min(max(10.0, fc), SR * 0.45)
    return butter(order, fc, kind, fs=SR, output='sos')


def _lp(x, fc, order=2):
    return sosfilt(_sos('low', fc, order), x, axis=0)


def _hp(x, fc, order=2):
    return sosfilt(_sos('high', fc, order), x, axis=0)


def _bp(x, lo, hi, order=2):
    return sosfilt(_sos('band', [lo, hi], order), x, axis=0)


def _reson(x, f, q):
    """two-pole resonator (constant peak gain)"""
    w = 2 * np.pi * f / SR
    r = np.exp(-w / (2 * q))
    b = [1 - r, 0, 0]
    a = [1, -2 * r * np.cos(w), r * r]
    return lfilter(b, a, x)


def bar_pan(midi: float, lo=45, hi=96, spread=0.7) -> float:
    return float(np.clip((midi - lo) / (hi - lo) * 2 - 1, -1, 1) * spread)


def _to_stereo(y, pan, width_ms=0.0):
    p = (pan + 1) * np.pi / 4
    l, r = np.cos(p) * np.sqrt(2), np.sin(p) * np.sqrt(2)
    if width_ms:
        k = int(width_ms / 1000 * SR)
        yr = np.concatenate([np.zeros(k), y[:len(y) - k]])
        return np.stack([y * l, yr * r], 1)
    return np.stack([y * l, y * r], 1)


def _modal(f0, ratios, amps, t60s, n, rng, detune_cents=0.0):
    """a bank of decaying sines (the bar's modes), each with a tiny random detune and phase"""
    t = np.arange(n) / SR
    y = np.zeros(n)
    for r, a, t60 in zip(ratios, amps, t60s):
        f = f0 * r * 2 ** (rng.normal(0, detune_cents) / 1200)
        if f >= SR * 0.45 or a <= 0:
            continue
        y += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 0.3)) * np.exp(-6.9078 * t / t60)
    return y


def _strike(n, hardness, rng, ms=None):
    """the mallet's contact: a half-sine force pulse whose length shrinks with hardness, plus felt noise"""
    ms = ms if ms is not None else 3.2 - 2.4 * hardness
    k = max(8, int(ms / 1000 * SR))
    t = np.arange(n) / SR
    pulse = np.zeros(n)
    pulse[:k] = np.sin(np.linspace(0, np.pi, k))
    noise = rng.standard_normal(n) * np.exp(-t / (0.0015 + 0.002 * (1 - hardness)))
    return pulse, noise


# ----------------------------------------------------------------------------- tuned percussion
def marimba(midi: float, vel: float = 0.8, hardness: float = 0.5, *, seed: int = 0, pan: float | None = None,
            length: float | None = None, dead: float = 0.0) -> np.ndarray:
    """A rosewood marimba bar. The bar is cut so its first overtones sit two octaves and ~ two octaves + a major
    third up (1 : 4 : 9.9); the tube under it reinforces the fundamental. Low bars ring ~1.5 s, high bars ~0.35 s;
    harder mallets bring the overtones and the knock forward. `dead` (0..1) damps it like a dead stroke."""
    rng = np.random.default_rng(seed)
    f0 = midi_hz(midi)
    h = float(np.clip(hardness * (0.7 + 0.5 * vel), 0, 1))
    t60_1 = float(np.clip(2.6 * (220 / f0) ** 0.75, 0.28, 2.4)) * (1 - 0.75 * dead)
    length = length or min(3.0, t60_1 * 0.9 + 0.12)
    n = int(length * SR)
    ratios = [1.0, 3.99, 9.87, 2.68 * 1.0, 6.2]
    amps = [1.0, 0.16 + 0.34 * h, 0.03 + 0.14 * h ** 1.5, 0.02 * h, 0.012 * h]
    t60s = [t60_1, t60_1 * 0.28, t60_1 * 0.11, t60_1 * 0.08, t60_1 * 0.06]
    y = _modal(f0, ratios, amps, t60s, n, rng, detune_cents=1.5)
    # the tube: a slightly delayed, slower-blooming reinforcement of the fundamental
    t = np.arange(n) / SR
    tube = np.sin(2 * np.pi * f0 * t + 0.4) * (1 - np.exp(-t / 0.006)) * np.exp(-6.9078 * t / (t60_1 * 1.15))
    y += 0.35 * tube * (1.0 if f0 < 900 else 0.4)
    pulse, noise = _strike(n, h, rng)
    knock = _bp(noise, 600, 3500) * (0.05 + 0.18 * h) + _lp(pulse, 1800) * 0.12
    y = y * (1 - np.exp(-t / (0.0004 + 0.0012 * (1 - h)))) + knock
    y *= vel ** 1.2 * 0.5
    p = bar_pan(midi) if pan is None else pan
    return _edge(_to_stereo(y, p, width_ms=0.25).astype(np.float32), 0.0003, 0.02)


def vibes(midi: float, vel: float = 0.7, dur: float = 2.0, *, trem: float = 0.0, trem_hz: float = 5.2,
          hardness: float = 0.4, seed: int = 0, pan: float | None = None) -> np.ndarray:
    """An aluminium vibraphone bar (1 : 4 : 10), long ring, damped by the pedal at `dur`; `trem` adds the motor's
    amplitude tremolo (0..1)."""
    rng = np.random.default_rng(seed)
    f0 = midi_hz(midi)
    t60 = float(np.clip(7.0 * (330 / f0) ** 0.5, 2.5, 9.0))
    n = int((dur + 0.35) * SR)
    t = np.arange(n) / SR
    h = hardness
    y = _modal(f0, [1.0, 4.0, 10.0, 2.95], [1.0, 0.12 + 0.3 * h, 0.03 + 0.08 * h, 0.01], [t60, t60 * 0.18, t60 * 0.06, t60 * 0.05],
               n, rng, detune_cents=0.8)
    pulse, noise = _strike(n, h, rng)
    y = y * (1 - np.exp(-t / 0.0008)) + _bp(noise, 1500, 6000) * 0.05 * (0.5 + h)
    if trem > 0:
        y *= 1 - trem * 0.45 * (1 + np.sin(2 * np.pi * trem_hz * t + rng.uniform(0, 6.28))) / 2
    damp = np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.07))
    y *= damp * vel ** 1.1 * 0.45
    p = bar_pan(midi, 53, 89) if pan is None else pan
    return _edge(_to_stereo(y, p, width_ms=0.4).astype(np.float32), 0.0003, 0.02)


def glock(midi: float, vel: float = 0.6, *, seed: int = 0, pan: float | None = None, length: float = 2.2) -> np.ndarray:
    """A glockenspiel bar: untuned free-bar overtones (1 : 2.76 : 5.40 : 8.93), bright and bell-like."""
    rng = np.random.default_rng(seed)
    f0 = midi_hz(midi)
    n = int(length * SR)
    t = np.arange(n) / SR
    y = _modal(f0, [1.0, 2.756, 5.404, 8.933], [1.0, 0.32, 0.12, 0.05], [length * 0.9, 0.5, 0.2, 0.1], n, rng, 2.0)
    pulse, noise = _strike(n, 0.9, rng, ms=0.35)
    y = y * (1 - np.exp(-t / 0.0002)) + _hp(noise, 4000) * 0.04
    y *= vel * 0.4
    p = bar_pan(midi, 72, 108) if pan is None else pan
    return _edge(_to_stereo(y, p, width_ms=0.3).astype(np.float32), 0.0002, 0.03)


def kalimba(midi: float, vel: float = 0.7, *, seed: int = 0, pan: float | None = None, length: float = 1.4) -> np.ndarray:
    """A thumb-piano tine over a small wooden box: the tine's modes (1 : 5.4 : 8.9 ... strongly damped overtones),
    a box resonance near 250 Hz knocked by the thumb, and a soft buzz on the attack."""
    rng = np.random.default_rng(seed)
    f0 = midi_hz(midi)
    n = int(length * SR)
    t = np.arange(n) / SR
    t60 = float(np.clip(2.4 * (300 / f0) ** 0.4, 0.8, 2.6))
    y = _modal(f0, [1.0, 5.4, 8.93, 2.0], [1.0, 0.18, 0.06, 0.05], [t60, 0.12, 0.05, t60 * 0.4], n, rng, 2.5)
    pulse, noise = _strike(n, 0.6, rng, ms=1.5)
    box = _reson(noise * 0.6 + pulse * 0.4, 260 + rng.uniform(-15, 15), 7) * 0.5
    y = y * (1 - np.exp(-t / 0.0006)) + box
    y *= vel * 0.5
    p = bar_pan(midi, 60, 96) if pan is None else pan
    return _edge(_to_stereo(y, p, width_ms=0.3).astype(np.float32), 0.0003, 0.02)


# ----------------------------------------------------------------------------- the whistle
def whistle(notes, *, glide: float = 0.06, vibrato_cents: float = 22.0, vib_hz: float = 5.6, vib_delay: float = 0.22,
            breath: float = 0.22, duration: float | None = None, seed: int = 0, scoop: float = 0.0) -> np.ndarray:
    """One continuous whistled line. notes: [(start_s, dur_s, midi, vel)] (monophonic). A note that starts within
    40 ms of the previous note's end is slurred: the pitch glides (`glide` s, exponential) instead of re-tonguing;
    a gap re-articulates with a soft breathy onset. `scoop` (semitones) bends each tongued note up from below.
    The tone: a sine with a whisper of the 2nd and 3rd harmonics, air through a resonant band-pass that follows the
    pitch, a little broadband hiss, and vibrato that grows in after `vib_delay`."""
    rng = np.random.default_rng(seed)
    notes = sorted(notes, key=lambda n: n[0])
    end = max(s + d for s, d, *_ in notes) + 0.4
    n = int((duration or end) * SR)
    pitch = np.full(n, np.nan)
    amp = np.zeros(n)
    onset_age = np.full(n, 10.0)
    prev_end = -1.0
    prev_m = None
    for i, (s, d, m, v) in enumerate(notes):
        a, b = int(s * SR), min(n, int((s + d) * SR))
        slur = prev_m is not None and s - prev_end < 0.04
        k = np.arange(b - a) / SR
        if slur:
            p = m + (prev_m - m) * np.exp(-k / (glide / 3))
        else:
            p = m - scoop * np.exp(-k / 0.05) if scoop else np.full(len(k), float(m))
        pitch[a:b] = p
        att = 1.0 if slur else 1 - np.exp(-k / 0.025)
        rel = np.clip((d - k) / 0.05, 0, 1)
        amp[a:b] = np.maximum(amp[a:b], v * att * rel ** 0.7)
        age = k if not slur else k + 0.3
        onset_age[a:b] = age
        prev_end, prev_m = s + d, m
    # fill gaps so the oscillator phase stays continuous
    idx = np.flatnonzero(~np.isnan(pitch))
    if len(idx) == 0:
        return np.zeros((n, 2), np.float32)
    pitch = np.interp(np.arange(n), idx, pitch[idx])
    t = np.arange(n) / SR
    vib_env = np.clip((onset_age - vib_delay) / 0.3, 0, 1)
    drift = _lp(rng.standard_normal(n), 2.5, 1)
    drift = drift / (np.std(drift) + 1e-9) * 4.0          # a lip never holds a pitch dead still: +-4 cents, slow
    cents = vibrato_cents * vib_env * np.sin(2 * np.pi * vib_hz * t + 0.4 * np.sin(2 * np.pi * 0.7 * t)) + drift
    f = 440.0 * 2 ** ((pitch - 69) / 12 + cents / 1200)
    ph = 2 * np.pi * np.cumsum(f) / SR
    amp_s = _lp(amp, 60, 1)
    tone = np.sin(ph) + 0.035 * np.sin(2 * ph + 0.3) + 0.012 * np.sin(3 * ph)
    # air: noise through a band following the pitch (block-wise resonator bank)
    noise = rng.standard_normal(n)
    air = np.zeros(n)
    blk = 256
    zi = np.zeros(2)
    for s0 in range(0, n, blk):
        s1 = min(n, s0 + blk)
        fc = float(f[s0])
        w = 2 * np.pi * min(fc, SR * 0.45) / SR
        r = np.exp(-w / (2 * 9.0))
        g = (1 - r * r) / 2
        bb, aa = [g, 0, -g], [1, -2 * r * np.cos(w), r * r]    # band-pass: zeros at DC and Nyquist
        air[s0:s1], zi = lfilter(bb, aa, noise[s0:s1], zi=zi)
    air = air / (np.std(air) + 1e-9) * 0.16
    hiss = _hp(noise, 6000) * 0.012
    onset_air = np.exp(-onset_age / 0.04) * 0.5
    y = (tone * amp_s) + (air + hiss) * amp_s * breath * (1 + 3 * onset_air)
    y *= 0.38
    st = _to_stereo(y, 0.0, width_ms=0.0)
    return _edge(st.astype(np.float32), 0.002, 0.03)


# ----------------------------------------------------------------------------- bass
def bass(midi: float, dur: float, vel: float = 0.9, *, tone: float = 0.5, slide_from: float | None = None,
         seed: int = 0) -> np.ndarray:
    """A round picked bass: sine fundamental and a soft 2nd harmonic for weight, a saw through a snapping low-pass
    for the finger (so it reads on small speakers too), a small pitch drop on the pluck, mono."""
    rng = np.random.default_rng(seed)
    n = int((dur + 0.08) * SR)
    t = np.arange(n) / SR
    f0 = midi_hz(midi)
    bend = 0.25 * np.exp(-t / 0.012)
    if slide_from is not None:
        bend = (slide_from - midi) * np.exp(-t / 0.05)
    f = f0 * 2 ** (bend / 12)
    ph = 2 * np.pi * np.cumsum(f) / SR + rng.uniform(0, 0.2)
    body = np.sin(ph) + 0.35 * np.sin(2 * ph + 0.5)
    saw = 2 * ((ph / (2 * np.pi)) % 1.0) - 1
    fc = 220 + (900 + 2200 * tone) * np.exp(-t / (0.05 + 0.04 * tone))
    from .synth import _tvlp
    finger = _tvlp(saw, fc, q=1.1) * (0.25 + 0.3 * tone)
    env = (1 - np.exp(-t / 0.002)) * np.where(t < dur, 0.72 + 0.28 * np.exp(-t / 0.18), 0.0)
    rel = np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.025))
    y = np.tanh((body + finger) * 1.15) * env * rel * vel * 0.55
    y = np.nan_to_num(y)
    return _edge(np.stack([y, y], 1).astype(np.float32), 0.0008, 0.015)


# ----------------------------------------------------------------------------- drums
def kick(vel: float = 1.0, *, tone: float = 0.5, length: float = 0.34, seed: int = 0) -> np.ndarray:
    """A tight, punchy kick: sine body 150 -> 52 Hz in ~30 ms, decaying in ~0.2 s; a beater click; gentle drive.
    Lower `tone` is softer and rounder (for the quiet sections)."""
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    f_lo = 55 + 7 * tone
    f = f_lo + (135 + 60 * tone - f_lo) * np.exp(-t / (0.022 + 0.012 * (1 - tone)))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (0.09 + 0.07 * (1 - tone)))
    click = _bp(rng.standard_normal(n), 2500, 8000) * np.exp(-t / 0.0012) * (0.15 + 0.35 * tone)
    thump = _lp(rng.standard_normal(n), 400) * np.exp(-t / 0.006) * 0.25
    y = np.tanh((body + click + thump) * (1.4 + 1.2 * tone)) / np.tanh(1.4 + 1.2 * tone) * vel
    return _edge(np.stack([y, y], 1).astype(np.float32), 0.0003, 0.02)


def snare(vel: float = 1.0, *, seed: int = 0, tight: float = 0.6, length: float = 0.28) -> np.ndarray:
    """A light, crisp snare: two tuned shell modes (~190 / 330 Hz) and band-limited wire noise."""
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    shell = (np.sin(2 * np.pi * 188 * t) * 0.8 + np.sin(2 * np.pi * 332 * t) * 0.5) * np.exp(-t / 0.045)
    wires = _bp(rng.standard_normal((n, 2)), 1800, 9000) * np.exp(-t / (0.06 + 0.08 * (1 - tight)))[:, None]
    y = shell[:, None] * 0.6 + wires * 0.9
    y = np.tanh(y * 1.5) * vel * 0.6
    return _edge(y.astype(np.float32), 0.0003, 0.02)


def clap(vel: float = 1.0, *, seed: int = 0, size: float = 0.5) -> np.ndarray:
    """Several hands, not one: four bursts with random spacing, each its own band, then a small room."""
    rng = np.random.default_rng(seed)
    n = int((0.22 + 0.2 * size) * SR)
    t = np.arange(n) / SR
    y = np.zeros((n, 2))
    d = 0.0
    for k in range(4):
        e = (t >= d) * np.exp(-np.maximum(0, t - d) / (0.004 if k < 3 else 0.05 + 0.08 * size))
        lo = 800 + rng.uniform(0, 500)
        burst = _bp(rng.standard_normal((n, 2)), lo, lo * 3.2)
        side = rng.uniform(-0.3, 0.3)
        y[:, 0] += burst[:, 0] * e * (1 - side)
        y[:, 1] += burst[:, 1] * e * (1 + side)
        d += rng.uniform(0.006, 0.011)
    y += _bp(rng.standard_normal((n, 2)), 1200, 7000) * (np.exp(-t / (0.08 + 0.1 * size)) * 0.1)[:, None]
    return _edge((y * vel * 0.7).astype(np.float32), 0.0003, 0.02)


def snap(vel: float = 1.0, *, seed: int = 0) -> np.ndarray:
    """A finger snap: a very short bright crack with a tonal ping near 2 kHz."""
    rng = np.random.default_rng(seed)
    n = int(0.09 * SR)
    t = np.arange(n) / SR
    crack = _bp(rng.standard_normal(n), 2000, 9000) * np.exp(-t / 0.004)
    ping = np.sin(2 * np.pi * (1900 + rng.uniform(-80, 80)) * t) * np.exp(-t / 0.012) * 0.35
    y = (crack + ping) * vel * 0.8
    return _edge(_to_stereo(y, rng.uniform(-0.15, 0.15)).astype(np.float32), 0.0002, 0.01)


def hat(vel: float = 1.0, *, open_: float = 0.0, seed: int = 0, bright: float = 0.5) -> np.ndarray:
    """A hi-hat: the six-square metallic cluster plus noise, band-passed high; `open_` 0..1 lengthens it."""
    rng = np.random.default_rng(seed)
    decay = 0.012 + 0.2 * open_
    n = int((decay * 6 + 0.02) * SR)
    t = np.arange(n) / SR
    fr = np.array([205.3, 304.4, 369.6, 522.7, 540.0, 800.0]) * (1 + 0.006 * rng.standard_normal(6))
    sq = sum(np.sign(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))) for f in fr) / 6
    y = 0.5 * sq + 0.5 * rng.standard_normal(n)
    y = _hp(_bp(y, 5500 + 3000 * bright, 17000), 6500)
    y = y * np.exp(-t / decay) * (1 - np.exp(-t / 0.0003)) * vel * 0.5
    return _edge(_to_stereo(y, 0.0, width_ms=0.2).astype(np.float32), 0.0002, 0.005)


def shaker(vel: float = 1.0, *, seed: int = 0, length: float = 0.11, push: float = 0.5) -> np.ndarray:
    """A seed shaker: many grains in a band, a soft swell into the stroke (`push` shifts the peak later)."""
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    pk = length * (0.2 + 0.3 * push)
    env = np.where(t < pk, (t / pk) ** 2, np.exp(-(t - pk) / (length * 0.22)))
    grains = rng.standard_normal((n, 2)) * (rng.random((n, 2)) < 0.25)
    y = _bp(grains, 4000, 11000) * env[:, None] * vel * 0.9
    return _edge(y.astype(np.float32), 0.0003, 0.005)


def rim(vel: float = 1.0, *, seed: int = 0, pitch: float = 1.0) -> np.ndarray:
    """A rim click / side stick: a woody knock (~500 Hz and 1.6 kHz modes) with a sharp top."""
    rng = np.random.default_rng(seed)
    n = int(0.08 * SR)
    t = np.arange(n) / SR
    y = (np.sin(2 * np.pi * 520 * pitch * t) * 0.7 + np.sin(2 * np.pi * 1650 * pitch * t) * 0.4) * np.exp(-t / 0.012)
    y += _bp(rng.standard_normal(n), 2500, 9000) * np.exp(-t / 0.002) * 0.6
    return _edge(_to_stereo(y * vel * 0.6, rng.uniform(-0.1, 0.1)).astype(np.float32), 0.0002, 0.01)
