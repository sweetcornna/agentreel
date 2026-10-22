"""Dynamics and level: oversampled saturation, RMS compressor (with side-chain),
look-ahead true-peak limiter, loudness (pyloudnorm), peaks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy import signal

from ..core import SR, stereo

OS = 4  # oversampling factor for saturation and true-peak estimates


# ---------------------------------------------------------------------------
# Saturation
# ---------------------------------------------------------------------------
def saturate(x, drive_db: float = 6.0, mix: float = 1.0, kind: str = "tanh", bias: float = 0.0,
             os: int = OS):
    """Oversampled waveshaper. kind: 'tanh' (symmetric, odd harmonics), 'tape' (soft asymmetric,
    adds a little even harmonic warmth), 'soft' (cubic soft clip). Output level is compensated for
    small signals so drive changes colour more than loudness."""
    x = np.asarray(x, dtype=np.float64)
    g = 10 ** (drive_db / 20)
    up = signal.resample_poly(x, os, 1, axis=-1)
    u = up * g
    if kind == "tanh":
        y = np.tanh(u) / g
    elif kind == "tape":
        y = (np.tanh(u + bias) - np.tanh(bias)) / g
        y += 0.08 * (np.tanh(u) ** 2) / g  # gentle 2nd harmonic
        y -= np.mean(y, axis=-1, keepdims=True)
    elif kind == "soft":
        uc = np.clip(u, -1.5, 1.5)
        y = (uc - uc ** 3 / 6.75) / g
    else:
        raise ValueError(kind)
    down = signal.resample_poly(y, 1, os, axis=-1)[..., : x.shape[-1]]
    return mix * down + (1 - mix) * x


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _comp_gain(det_db, thresh, ratio, knee, att, rel):
    n = det_db.shape[0]
    gr = np.zeros(n)
    env = 0.0
    for i in range(n):
        over = det_db[i] - thresh
        if knee > 0 and 2 * abs(over) <= knee:
            red = (1.0 / ratio - 1.0) * (over + knee / 2) ** 2 / (2 * knee)
        elif over > 0:
            red = (1.0 / ratio - 1.0) * over
        else:
            red = 0.0
        # red is negative dB; smooth with attack when reducing more, release otherwise
        if red < env:
            env = att * env + (1 - att) * red
        else:
            env = rel * env + (1 - rel) * red
        gr[i] = env
    return gr


@dataclass
class Compressor:
    thresh_db: float = -18.0
    ratio: float = 2.5
    attack_ms: float = 15.0
    release_ms: float = 180.0
    knee_db: float = 6.0
    makeup_db: float = 0.0
    rms_ms: float = 12.0
    mix: float = 1.0

    def gain_db(self, det):
        """Gain reduction (dB, <= 0) for a detector signal (mono or stereo)."""
        det = np.atleast_2d(det)
        p = np.max(det ** 2, axis=0)
        k = max(1, int(self.rms_ms * 1e-3 * SR))
        rms = np.sqrt(np.convolve(p, np.ones(k) / k, mode="same") + 1e-20)
        det_db = 20 * np.log10(rms + 1e-12)
        att = np.exp(-1.0 / (self.attack_ms * 1e-3 * SR))
        rel = np.exp(-1.0 / (self.release_ms * 1e-3 * SR))
        return _comp_gain(det_db, self.thresh_db, self.ratio, self.knee_db, att, rel)

    def process(self, x, ctx=None, sidechain=None):
        gr = self.gain_db(x if sidechain is None else sidechain)
        g = 10 ** ((gr + self.makeup_db) / 20)
        return self.mix * x * g + (1 - self.mix) * x


def compress(x, sidechain=None, **kw):
    return Compressor(**kw).process(x, sidechain=sidechain)


def duck(x, sidechain, depth_db: float = 6.0, attack_ms: float = 8.0, release_ms: float = 250.0,
         thresh_db: float = -30.0):
    """Side-chain ducking: lowers x by up to depth_db while the side-chain signal is loud."""
    c = Compressor(thresh_db=thresh_db, ratio=20.0, attack_ms=attack_ms, release_ms=release_ms, knee_db=6.0)
    gr = np.maximum(c.gain_db(sidechain), -depth_db)
    return x * 10 ** (gr / 20)


# ---------------------------------------------------------------------------
# Peaks, true peak, loudness
# ---------------------------------------------------------------------------
def peak_db(x):
    return float(20 * np.log10(np.max(np.abs(x)) + 1e-12))


def true_peak_db(x, os: int = OS):
    """ITU-style true-peak estimate: 4x oversampled absolute maximum (dBTP)."""
    up = signal.resample_poly(np.asarray(x, dtype=np.float64), os, 1, axis=-1)
    return float(20 * np.log10(np.max(np.abs(up)) + 1e-12))


def lufs(x, sr: int = SR):
    import pyloudnorm as pyln

    x = stereo(x)
    if x.shape[1] < int(0.5 * sr):
        x = np.pad(x, ((0, 0), (0, int(0.5 * sr) - x.shape[1])))
    meter = pyln.Meter(sr)
    v = meter.integrated_loudness(x.T)
    return float(v) if np.isfinite(v) else -120.0


def short_term_lufs(x, sr: int = SR, win: float = 3.0, hop: float = 1.0):
    """Short-term loudness series (3 s windows) for a rough loudness range (LRA-like)."""
    import pyloudnorm as pyln

    x = stereo(x)
    meter = pyln.Meter(sr)
    out = []
    n = int(win * sr)
    h = int(hop * sr)
    for s in range(0, max(1, x.shape[1] - n + 1), h):
        v = meter.integrated_loudness(x[:, s:s + n].T)
        out.append(v if np.isfinite(v) else -120.0)
    return np.array(out)


def normalize_lufs(x, target: float = -16.0):
    cur = lufs(x)
    if cur < -100:
        return x, 0.0
    g = target - cur
    return x * 10 ** (g / 20), g


# ---------------------------------------------------------------------------
# Look-ahead limiter (true-peak aware)
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _sliding_min(a, w):
    n = a.shape[0]
    out = np.empty(n)
    # monotone deque
    dq = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0
    for i in range(n):
        while tail > head and a[dq[tail - 1]] >= a[i]:
            tail -= 1
        dq[tail] = i
        tail += 1
        if dq[head] <= i - w:
            head += 1
        out[i] = a[dq[head]]
    return out


@njit(cache=True, fastmath=True)
def _release(g, rel):
    n = g.shape[0]
    out = np.empty(n)
    s = 1.0
    for i in range(n):
        if g[i] < s:
            s = g[i]
        else:
            s = rel * s + (1 - rel) * g[i]
        out[i] = s
    return out


def limit(x, ceiling_db: float = -1.0, lookahead_ms: float = 5.0, release_ms: float = 120.0, os: int = OS):
    """Look-ahead brick-wall limiter driven by a 4x oversampled peak detector (approximate true peak).

    The output is delayed internally and re-aligned, so its timing equals the input.
    """
    x = stereo(x)
    n = x.shape[1]
    la = max(1, int(lookahead_ms * 1e-3 * SR))
    ceil = 10 ** (ceiling_db / 20)
    up = signal.resample_poly(x, os, 1, axis=-1)
    pk = np.max(np.abs(up), axis=0)
    pk = pk[: n * os].reshape(n, os).max(axis=1) if pk.shape[0] >= n * os else np.pad(pk, (0, n * os - pk.shape[0])).reshape(n, os).max(axis=1)
    pk = np.maximum(pk, np.max(np.abs(x), axis=0))
    req = np.minimum(1.0, ceil / np.maximum(pk, 1e-12))
    # gain must reach req before the peak arrives: min over the next `la` samples
    req_ahead = _sliding_min(np.concatenate([req, np.ones(la)]), la + 1)[la: la + n]
    # smooth the attack with a moving average of length la (stays below required gain)
    k = np.ones(la) / la
    sm = np.convolve(np.concatenate([np.ones(la), req_ahead]), k, mode="valid")[: n]
    sm = np.minimum(sm, req_ahead)
    rel = np.exp(-1.0 / (release_ms * 1e-3 * SR))
    g = _release(sm, rel)
    return x * g


def master_chain(x, target_lufs: float = -16.0, ceiling_dbtp: float = -1.0, passes: int = 3):
    """Normalise to the loudness target, then true-peak limit; repeat a couple of times to converge."""
    y = x
    for _ in range(passes):
        y, _ = normalize_lufs(y, target_lufs)
        y = limit(y, ceiling_dbtp - 0.3)
        if true_peak_db(y) <= ceiling_dbtp + 0.05 and abs(lufs(y) - target_lufs) < 0.3:
            break
    return y
