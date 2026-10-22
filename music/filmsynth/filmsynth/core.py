"""Core helpers: sample rate, pitch handling, seeded randomness, envelopes.

Everything in filmsynth works at 48 kHz, float64, and stereo arrays shaped (2, n).
"""
from __future__ import annotations

import re
import zlib

import numpy as np

SR = 48000

# ---------------------------------------------------------------------------
# Pitch
# ---------------------------------------------------------------------------
_NOTE_RE = re.compile(r"^\s*([A-Ga-g])\s*(#|♯|b|♭|##|bb)?\s*(-?\d+)\s*([+-]\d+(?:\.\d+)?c)?\s*$")
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACC = {None: 0, "#": 1, "♯": 1, "b": -1, "♭": -1, "##": 2, "bb": -2}
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# The default mode of penta() / snap_penta(): D gong pentatonic {D E F# A B}; the B yu mode uses the same set.
PENTA_PCS = (2, 4, 6, 9, 11)
COLOUR_PCS = {"qingjiao": 7, "biangong": 1}  # G and C#


def to_midi(p) -> float:
    """Accept MIDI number (int/float), note name ('F#3', 'Bb2', 'A4+5c') and return float MIDI."""
    if p is None:
        return float("nan")
    if isinstance(p, (int, float, np.integer, np.floating)):
        return float(p)
    m = _NOTE_RE.match(str(p))
    if not m:
        raise ValueError(f"cannot parse pitch {p!r}")
    letter, acc, octv, cents = m.groups()
    midi = 12 * (int(octv) + 1) + _PC[letter.upper()] + _ACC[acc]
    if cents:
        midi += float(cents[:-1]) / 100.0
    return float(midi)


def midi_to_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


def hz_to_midi(f):
    return 69.0 + 12.0 * np.log2(np.asarray(f, dtype=np.float64) / 440.0)


def note_name(m: float) -> str:
    if m is None or not np.isfinite(m):
        return "-"
    r = int(round(m))
    cents = (m - r) * 100
    s = f"{_NAMES[r % 12]}{r // 12 - 1}"
    if abs(cents) >= 0.5:
        s += f"{cents:+.0f}c"
    return s


def penta(degree: int, octave: int = 4, tonic: str = "D") -> float:
    """Degree in the D-gong pentatonic, 0 = D, 1 = E, 2 = F#, 3 = A, 4 = B, 5 = D an octave up...

    tonic='B' counts degrees from B instead (B yu mode: B D E F# A).
    """
    pcs = list(PENTA_PCS)
    if tonic.upper() == "B":
        pcs = [11, 14, 16, 18, 21]
    elif tonic.upper() != "D":
        raise ValueError("tonic must be 'D' or 'B'")
    o, k = divmod(degree, 5)
    return 12 * (octave + 1 + o) + pcs[k]


def snap_penta(m: float) -> float:
    """Snap a MIDI pitch to the nearest pentatonic pitch."""
    base = int(np.floor(m))
    best, bd = None, 99.0
    for c in range(base - 3, base + 4):
        if c % 12 in PENTA_PCS and abs(c - m) < bd:
            best, bd = c, abs(c - m)
    return float(best)


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
def seed_of(*parts) -> int:
    """Stable 32-bit seed from arbitrary parts (crc32 of their repr)."""
    return zlib.crc32(repr(parts).encode("utf-8")) & 0x7FFFFFFF


def rng_of(*parts) -> np.random.Generator:
    return np.random.default_rng(seed_of(*parts))


# ---------------------------------------------------------------------------
# Signals and envelopes
# ---------------------------------------------------------------------------
def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-12))


def undb(d):
    return 10.0 ** (np.asarray(d, dtype=np.float64) / 20.0)


def vel_amp(vel, range_db: float = 30.0, curve: float = 1.0):
    """Velocity 0..1 to linear amplitude; vel=1 -> 1.0, vel=0 -> -range_db."""
    v = np.clip(np.asarray(vel, dtype=np.float64), 0.0, 1.0)
    return undb(-range_db * (1.0 - v ** curve))


def secs(n: int, sr: int = SR) -> float:
    return n / sr


def nsamp(t: float, sr: int = SR) -> int:
    return int(round(t * sr))


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def raised_cos(n: int):
    """0 -> 1 raised cosine ramp of n samples (zero slope at both ends)."""
    if n <= 0:
        return np.zeros(0)
    return 0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)


def env_asr(n: int, attack: float, release: float, sr: int = SR, release_at: int | None = None,
            attack_shape: float = 1.0):
    """Attack-sustain-release envelope with raised-cosine edges (click free).

    attack/release in seconds. release_at = sample index where release starts (default n - release).
    """
    e = np.ones(n)
    na = min(n, max(1, int(attack * sr)))
    e[:na] = raised_cos(na) ** attack_shape
    nr = max(1, int(release * sr))
    ra = n - nr if release_at is None else release_at
    ra = max(0, min(ra, n))
    tail = n - ra
    if tail > 0:
        r = 1.0 - raised_cos(min(nr, tail))
        e[ra:ra + len(r)] *= r
        e[ra + len(r):] = 0.0
    return e


def fade(x, fade_in: float = 0.0, fade_out: float = 0.0, sr: int = SR):
    """Raised-cosine fade in/out (seconds) on a (…, n) array. Returns a new array."""
    y = np.array(x, dtype=np.float64, copy=True)
    n = y.shape[-1]
    if fade_in > 0:
        k = min(n, int(fade_in * sr))
        y[..., :k] *= raised_cos(k)
    if fade_out > 0:
        k = min(n, int(fade_out * sr))
        y[..., n - k:] *= raised_cos(k)[::-1]
    return y


def smooth_noise(n: int, rate_hz: float, rng: np.random.Generator, sr: int = SR, order: int = 3):
    """Band-limited random signal (~unit std) varying at roughly rate_hz. Used for drift, gusts."""
    if n <= 0:
        return np.zeros(0)
    npts = int(n / sr * rate_hz) + 4
    pts = rng.standard_normal(npts)
    x = np.linspace(0, npts - 3.000001, n)
    # Catmull-Rom style cubic interpolation for smoothness
    i = np.floor(x).astype(int)
    t = x - i
    p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
    y = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
               + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t)
    s = np.std(y)
    return y / s if s > 0 else y


def stereo(x):
    """Return a (2, n) float64 array from mono (n,) or stereo input."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return np.vstack([x, x])
    if x.shape[0] == 2:
        return x
    if x.shape[1] == 2:
        return x.T.copy()
    raise ValueError("expected mono or stereo array")


def pan_gains(pan: float):
    """Equal-power pan gains for pan in [-1, 1]."""
    th = (np.clip(pan, -1.0, 1.0) + 1.0) * np.pi / 4.0
    return np.cos(th), np.sin(th)


def place(mono, pan: float = 0.0):
    gl, gr = pan_gains(pan)
    return np.vstack([mono * gl, mono * gr])


def add_into(buf, x, start: int):
    """Add x (2, m) into buf (2, n) at sample offset start, clipping to bounds."""
    n = buf.shape[-1]
    m = x.shape[-1]
    s0 = max(0, start)
    s1 = min(n, start + m)
    if s1 <= s0:
        return
    buf[..., s0:s1] += x[..., s0 - start:s1 - start]


def trim_tail(x, thresh_db: float = -100.0, min_len: int = 0, fade_s: float = 0.02, sr: int = SR):
    """Cut trailing samples below thresh and apply a short fade so the cut is inaudible."""
    a = np.max(np.abs(np.atleast_2d(x)), axis=0)
    th = undb(thresh_db)
    idx = np.nonzero(a > th)[0]
    end = max(min_len, (idx[-1] + 1) if len(idx) else min_len)
    end = min(end + int(fade_s * sr), x.shape[-1])
    y = x[..., :end].copy()
    return fade(y, 0.0, min(fade_s, end / sr), sr)
