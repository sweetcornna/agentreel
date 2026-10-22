"""Automation inserts: swell box (expression pedal), gain curves, time-varying filter sweeps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import SR
from .filters import svf


@dataclass
class Swell:
    """Organ swell box / expression pedal: `curve` maps t (s) -> 0 (closed) .. 1 (open).

    Closed = quieter by `range_db` and darker (low-pass `closed_hz`), open = full and bright.
    Works on any track (strings and choir too)."""
    curve: object
    range_db: float = 18.0
    closed_hz: float = 1400.0
    open_hz: float = 18000.0

    def process(self, x, ctx=None):
        n = x.shape[1]
        t = np.arange(n) / SR
        s = np.clip(self.curve(t), 0, 1)
        fc = self.closed_hz * (self.open_hz / self.closed_hz) ** s
        y = svf(x, fc, 0.6, "lowpass")
        g = 10 ** (-self.range_db * (1 - s) / 20)
        return y * g


@dataclass
class GainCurve:
    """Multiply by a gain curve in dB: `curve` maps t (s) -> dB."""
    curve: object

    def process(self, x, ctx=None):
        t = np.arange(x.shape[1]) / SR
        return x * 10 ** (np.asarray(self.curve(t)) / 20)


@dataclass
class FilterSweep:
    """Time-varying SVF: `curve` maps t (s) -> cutoff Hz. mode: lowpass | highpass | bandpass."""
    curve: object
    mode: str = "lowpass"
    q: float = 0.7071

    def process(self, x, ctx=None):
        t = np.arange(x.shape[1]) / SR
        return svf(x, np.asarray(self.curve(t)), self.q, self.mode)
