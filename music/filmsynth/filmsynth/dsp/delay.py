"""Feedback delay with time-varying feedback, filtered repeats and optional ping-pong.

A `persistence` curve p in 0..1 (how much of a sound is remembered; a film can share it with the picture,
e.g. as a timeline curve) drives the echo directly:
  feedback  fb = 0.15 + 0.45 * p           (p=0: echoes vanish at once; p=1: ~0.6, a long tail)
  loop low-pass = 1800 + 3200 * p Hz       (low p: darker repeats that disappear faster)
Use persistence_delay(curve) so the music and the picture share this one mapping.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from ..core import SR


def persistence_feedback(p):
    """Feedback gain for persistence p in [0, 1]: fb = 0.15 + 0.45 p."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return 0.15 + 0.45 * p


def persistence_lowpass(p):
    """Low-pass cutoff (Hz) inside the feedback loop: at low persistence the repeats are darker and
    die faster (forgotten); at high persistence they stay clearer for longer (remembered)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return 1800.0 + 3200.0 * p


@njit(cache=True, fastmath=True, inline="always")
def _hread(buf, rp, mask):
    ip = int(np.floor(rp))
    fr = rp - ip
    xm1 = buf[(ip - 1) & mask]
    x0 = buf[ip & mask]
    x1 = buf[(ip + 1) & mask]
    x2 = buf[(ip + 2) & mask]
    c1 = 0.5 * (x1 - xm1)
    c2 = xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2
    c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1)
    return ((c3 * fr + c2) * fr + c1) * fr + x0


@njit(cache=True, fastmath=True)
def _fb_delay(x, dl, dr, fb, lp_coef, hp_coef, pingpong, cross, drive):
    n = x.shape[1]
    size = 1
    mx = 0.0
    for i in range(n):
        if dl[i] > mx:
            mx = dl[i]
        if dr[i] > mx:
            mx = dr[i]
    while size < mx + 8:
        size *= 2
    mask = size - 1
    bl = np.zeros(size)
    br = np.zeros(size)
    y = np.zeros((2, n))
    lpl = 0.0
    lpr = 0.0
    hpl = 0.0
    hpr = 0.0
    w = 0
    for i in range(n):
        ol = _hread(bl, w - dl[i], mask)
        orr = _hread(br, w - dr[i], mask)
        # loop filters: one-pole low-pass then one-pole high-pass
        lpl += lp_coef[i] * (ol - lpl)
        lpr += lp_coef[i] * (orr - lpr)
        hpl += hp_coef * (lpl - hpl)
        hpr += hp_coef * (lpr - hpr)
        fl = lpl - hpl
        fr_ = lpr - hpr
        if drive > 0.0:
            fl = np.tanh(fl * drive) / drive
            fr_ = np.tanh(fr_ * drive) / drive
        g = fb[i]
        if pingpong:
            inl = x[0, i] + g * fr_
            inr = g * fl + cross * x[1, i]
        else:
            inl = x[0, i] + g * (fl * (1 - cross) + fr_ * cross)
            inr = x[1, i] + g * (fr_ * (1 - cross) + fl * cross)
        bl[w & mask] = inl
        br[w & mask] = inr
        y[0, i] = ol
        y[1, i] = orr
        w += 1
    return y


@dataclass
class FeedbackDelay:
    """Stereo feedback delay. Returns wet only from .process(); use mix for dry/wet.

    time_l / time_r: seconds, or beats if `beats=True` (resolved with the session tempo at render).
    feedback: constant, or a callable f(t_seconds_array) -> feedback array (e.g. from a curve).
    lowpass: Hz (constant) or callable f(t) -> Hz.
    pingpong: alternate repeats L/R. wow: slow time modulation depth in ms (tape-like life).
    mix: dry/wet when used as a track insert (1.0 = wet only, 0 = bypass). wet_db scales the wet.
    """
    time_l: float = 0.75
    time_r: float | None = None
    beats: bool = True
    feedback: object = 0.35
    lowpass: object = 4500.0
    highpass: float = 180.0
    pingpong: bool = True
    cross: float = 0.0
    wow: float = 0.6
    drive: float = 0.8
    mix: float = 0.35
    wet_db: float = 0.0
    seed: int = 3

    def _delay_curve(self, t, which, ctx):
        tm = self.time_l if which == 0 else (self.time_r if self.time_r is not None else self.time_l)
        if self.beats:
            if ctx is not None and hasattr(ctx, "beat_seconds_at"):
                sec = ctx.beat_seconds_at(t) * tm
            else:
                sec = np.full_like(t, 60.0 / 72.0 * tm)
        else:
            sec = np.full_like(t, tm)
        return sec

    def wet(self, x, ctx=None, t0: float = 0.0):
        n = x.shape[1]
        t = t0 + np.arange(n) / SR
        dl = self._delay_curve(t, 0, ctx) * SR
        dr = self._delay_curve(t, 1, ctx) * SR
        if self.wow > 0:
            rng = np.random.default_rng(self.seed)
            ph = rng.uniform(0, 2 * np.pi, 2)
            dl = dl + self.wow * 1e-3 * SR * np.sin(2 * np.pi * 0.31 * t + ph[0])
            dr = dr + self.wow * 1e-3 * SR * np.sin(2 * np.pi * 0.27 * t + ph[1])
        fb = self.feedback(t) if callable(self.feedback) else np.full(n, float(self.feedback))
        fb = np.clip(fb, 0.0, 0.95)
        lp = self.lowpass(t) if callable(self.lowpass) else np.full(n, float(self.lowpass))
        lp_coef = 1.0 - np.exp(-2 * np.pi * np.clip(lp, 50, SR * 0.45) / SR)
        hp_coef = 1.0 - np.exp(-2 * np.pi * self.highpass / SR)
        y = _fb_delay(np.ascontiguousarray(x), np.ascontiguousarray(dl), np.ascontiguousarray(dr),
                      np.ascontiguousarray(fb), np.ascontiguousarray(lp_coef), hp_coef, self.pingpong,
                      self.cross, self.drive)
        return y * 10 ** (self.wet_db / 20)

    def process(self, x, ctx=None):
        if self.mix <= 0:
            return x
        return x * (1.0 - 0.5 * self.mix) + self.mix * self.wet(x, ctx)


class _PFeedback:
    def __init__(self, curve):
        self.curve = curve

    def __call__(self, t):
        return persistence_feedback(self.curve(t))


class _PLowpass:
    def __init__(self, curve, scale: float = 1.0):
        self.curve = curve
        self.scale = scale

    def __call__(self, t):
        return persistence_lowpass(self.curve(t)) * self.scale


def persistence_delay(curve, time_beats: float = 0.75, time_r_beats: float | None = 1.0, lp_scale: float = 1.0, **kw):
    """FeedbackDelay whose feedback and loop low-pass follow a persistence curve.

    `curve` is a callable t_seconds -> persistence (Session.curve(keys) returns one; picklable).
    `lp_scale` scales the loop low-pass (persistence_lowpass(p) * lp_scale): below 1 every repeat comes back
    darker than the one before by more (a dark, woody source wants its echoes darker still). The feedback
    mapping fb = 0.15 + 0.45 p is never changed.
    """
    return FeedbackDelay(time_l=time_beats, time_r=time_r_beats, beats=True,
                         feedback=_PFeedback(curve), lowpass=_PLowpass(curve, lp_scale), **kw)


@njit(cache=True, fastmath=True)
def _vardelay(x, d):
    n = x.shape[0]
    y = np.zeros(n)
    for i in range(n):
        rp = i - d[i]
        ip = int(np.floor(rp))
        if ip < 1 or ip + 2 >= n:
            continue
        fr = rp - ip
        xm1 = x[ip - 1]
        x0 = x[ip]
        x1 = x[ip + 1]
        x2 = x[ip + 2]
        c1 = 0.5 * (x1 - xm1)
        c2 = xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2
        c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1)
        y[i] = ((c3 * fr + c2) * fr + c1) * fr + x0
    return y


def vardelay(x, delay_samples):
    """Read x through a time-varying delay (cubic Hermite). delay_samples: per-sample array (>= 1)."""
    x = np.asarray(x, dtype=np.float64)
    d = np.ascontiguousarray(np.asarray(delay_samples, dtype=np.float64))
    if x.ndim == 1:
        return _vardelay(np.ascontiguousarray(x), d)
    return np.vstack([_vardelay(np.ascontiguousarray(c), d) for c in x])
