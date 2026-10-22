"""Track/bus/master inserts: small dataclasses with .process(x, ctx) -> x (stereo (2, n))."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dynamics import limit, saturate
from .filters import dc_block, eq
from .stereo import haas, width


@dataclass
class EQ:
    """Series RBJ biquads: bands = [(kind, fc, q, gain_db), ...], kinds as in filters.rbj
    (lowpass, highpass, bandpass, notch, peak, lowshelf, highshelf)."""
    bands: list = field(default_factory=list)

    def process(self, x, ctx=None):
        return eq(x, self.bands)


@dataclass
class Saturate:
    """4x oversampled saturation (tanh | tape | soft)."""
    drive_db: float = 6.0
    kind: str = "tape"
    mix: float = 1.0

    def process(self, x, ctx=None):
        return saturate(x, self.drive_db, self.mix, self.kind)


@dataclass
class Width:
    """M/S width (0 mono, 1 unchanged, >1 wider)."""
    w: float = 1.0

    def process(self, x, ctx=None):
        return width(x, self.w)


@dataclass
class Haas:
    delay_ms: float = 12.0
    side: str = "right"

    def process(self, x, ctx=None):
        return haas(x, self.delay_ms, self.side)


@dataclass
class Gain:
    db: float = 0.0

    def process(self, x, ctx=None):
        return x * 10 ** (self.db / 20)


@dataclass
class Limiter:
    """Look-ahead true-peak-aware limiter (use on the master or a bus)."""
    ceiling_db: float = -1.0
    lookahead_ms: float = 5.0
    release_ms: float = 120.0

    def process(self, x, ctx=None):
        return limit(x, self.ceiling_db, self.lookahead_ms, self.release_ms)


@dataclass
class DCBlock:
    fc: float = 8.0

    def process(self, x, ctx=None):
        return dc_block(x, self.fc)
