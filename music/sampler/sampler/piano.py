"""Sampled grand piano: Salamander Grand Piano V3 (Alexander Holm, CC-BY 3.0).

    render_piano(notes, pedal=[], felt=False, duration=None, **options) -> (n, 2) float32

notes  = [(start_s, dur_s, midi, velocity_1_127), ...]   (midi may be fractional)
pedal  = [(down_s, up_s), ...]                           (up_s=None -> held to the end)

The SFZ mapping (<samples>/salamander/*.sfz, incl. its #define/#include data files) is
parsed by ``sampler.sfz`` and interpreted here: key/velocity ranges, keyswitched tuning
masters (natural/retuned), per-region tune, volume, amp_veltrack (+ its CC modulation),
ampeg_release (+CC), release-triggered string-resonance and hammer-noise samples with
rt_decay, pedal up/down noise samples with lorand/hirand, note self-masking
(note_polyphony=1 + off_time) and CC-conditioned layers (locc/hicc).

Additions beyond the SFZ: equal-power-ish crossfade between adjacent velocity layers
(onset-aligned), onset-aligned timing (the hammer strike lands exactly on start_s),
the ``felt=True`` model (see ``_felt_voice``) and pedal sympathetic resonance of the
open strings (``sympathetic``, see ``_sympathetic``).
"""
from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from . import sfz as _sfz
from ._dsp import (SR, add_at, biquad, butter, cos_ramp, db2a, fade_edges, midi_to_hz,
                   resample_by, sosfilt)
from ._paths import SAMPLES, log_access, need

SAMPLE_ROOT = SAMPLES / 'salamander'
SFZ_FILE = SAMPLE_ROOT / 'Salamander Grand Piano V3.sfz'

# CC numbers used by this SFZ (see its #define block) and our option names for them
CC = {'string_res': 20, 'hammer_noise': 21, 'pedal_noise': 22, 'release': 72,
      'offset': 98, 'veltrack': 99}

_BUILTIN_CURVES = {
    0: lambda x: x, 1: lambda x: 2 * x - 1, 2: lambda x: 1 - x, 3: lambda x: 1 - 2 * x,
    4: lambda x: x * x, 5: lambda x: np.sqrt(x), 6: lambda x: np.sqrt(max(0.0, 1 - x)),
}


def _curve_fn(curves, idx):
    if idx in _BUILTIN_CURVES:
        return _BUILTIN_CURVES[idx]
    for c in curves:
        if int(c.get('curve_index', -1)) == idx:
            pts = sorted((int(k[1:]) / 127.0, float(v)) for k, v in c.items()
                         if k.startswith('v') and k[1:].isdigit())
            xs, ys = zip(*pts)
            return lambda x, xs=xs, ys=ys: float(np.interp(x, xs, ys))
    return _BUILTIN_CURVES[0]


class Region:
    """One SFZ region with the opcodes we interpret resolved to numbers."""

    def __init__(self, d: dict, curves):
        self.raw = d
        self.sample = d['sample']
        key = d.get('key')
        self.lokey = _sfz.note_number(d.get('lokey', key if key is not None else 0))
        self.hikey = _sfz.note_number(d.get('hikey', key if key is not None else 127))
        self.keycenter = _sfz.note_number(d.get('pitch_keycenter', key if key is not None else 60))
        self.lovel = int(d.get('lovel', 1))
        self.hivel = int(d.get('hivel', 127))
        self.tune = float(d.get('tune', 0.0))
        self.volume = float(d.get('volume', 0.0))
        self.trigger = d.get('trigger', 'attack')
        self.rt_decay = float(d.get('rt_decay', 0.0))
        self.off_time = float(d.get('off_time', 0.006))
        self.group = int(d.get('group', 0))
        self.lorand = float(d.get('lorand', 0.0))
        self.hirand = float(d.get('hirand', 1.0))
        self.sw_label = d.get('sw_label')
        self.cond, self.on_cc = {}, {}
        self.mods = {'amp_veltrack': [], 'ampeg_release': [], 'amplitude': []}
        for k, v in d.items():
            if k.startswith(('locc', 'hicc')) and k[4:].isdigit():
                lo, hi = self.cond.get(int(k[4:]), (0.0, 127.0))
                self.cond[int(k[4:])] = (float(v), hi) if k.startswith('lo') else (lo, float(v))
            elif k.startswith(('on_locc', 'on_hicc')) and k[7:].isdigit():
                lo, hi = self.on_cc.get(int(k[7:]), (0.0, 127.0))
                self.on_cc[int(k[7:])] = (float(v), hi) if k.startswith('on_lo') else (lo, float(v))
            else:
                for base in self.mods:
                    pre = base + '_oncc'
                    if k.startswith(pre) and k[len(pre):].isdigit():
                        n = int(k[len(pre):])
                        cidx = int(d.get(f'{base}_curvecc{n}', 0))
                        self.mods[base].append((n, float(v), _curve_fn(curves, cidx)))
        self.amp_veltrack = float(d.get('amp_veltrack', 100.0))
        self.ampeg_release = float(d.get('ampeg_release', 0.0))

    def active(self, cc) -> bool:
        return all(lo <= cc.get(n, 0.0) * 127.0 <= hi for n, (lo, hi) in self.cond.items())

    def veltrack(self, cc) -> float:
        v = self.amp_veltrack + sum(a * f(cc.get(n, 0.0)) for n, a, f in self.mods['amp_veltrack'])
        return float(np.clip(v, -100.0, 100.0)) / 100.0

    def release_time(self, cc) -> float:
        return self.ampeg_release + sum(a * f(cc.get(n, 0.0)) for n, a, f in self.mods['ampeg_release'])

    def amp_factor(self, cc) -> float:
        g = 1.0
        for n, a, f in self.mods['amplitude']:
            if n in (7, 10):          # channel volume / pan: identical for all regions
                continue
            g *= a / 100.0 * f(cc.get(n, 0.0))
        return g

    def vel_gain(self, vel, cc) -> float:
        """SFZ amp_veltrack with the default vel^2 curve: g = 1 - |V| (1 - (v/127)^2)."""
        V = self.veltrack(cc)
        c = (vel / 127.0) ** 2
        if V < 0:                      # inverted tracking (not used by Salamander)
            c = 1.0 - c
        return 1.0 - abs(V) * (1.0 - c)


def _trim_late_spike(x, block=240, jump_db=20.0):
    """Cut recording artefacts near a sample's end.

    A few library files end with a stray tick (found by scanning all release/pedal
    samples: harmV3C5.flac has a +27 dB jump 27 ms before its end). If the level jumps
    by more than `jump_db` from one 5 ms block to the next in the last half of the
    file, the sample is truncated just before it with a 10 ms fade.
    """
    e = np.abs(x).max(axis=1)
    n = len(e) // block
    if n < 8:
        return x
    db = 20 * np.log10(e[:n * block].reshape(n, block).max(axis=1) + 1e-12)
    st = n // 2
    j = np.flatnonzero(np.diff(db[st:]) > jump_db)
    if not len(j):
        return x
    cut = (st + int(j[0]) + 1) * block - int(0.005 * SR)
    y = np.array(x[:cut], copy=True)
    f = int(0.01 * SR)
    y[-f:] *= cos_ramp(f, rising=False)[:, None]
    return y


ALIGN = {'on': False}            # render_piano(align_stereo=True) sets it for the call


def _align_lr(x, max_lag_s=0.0025):
    """Time-align R to L (the Salamander pair is spaced: around C5-E5 the two channels arrive almost in
    opposite phase, correlation down to -0.8, so the note is hollow in stereo and cancels in mono). The lag
    that best correlates the two channels over the first 250 ms after the attack is removed; the room is kept."""
    env = np.abs(x).max(axis=1)
    on = int(np.argmax(env > 0.1 * env.max())) if env.size and env.max() > 0 else 0
    a, b = x[on:on + int(0.25 * SR), 0], x[on:on + int(0.25 * SR), 1]
    k = int(max_lag_s * SR)
    if len(a) < 4 * k:
        return x
    c = np.array([np.dot(a[k:-k], b[k + j:len(b) - k + j]) for j in range(-k, k + 1)])
    j = int(np.argmax(c)) - k
    y = x.copy()
    if j > 0:
        y[:-j, 1] = x[j:, 1]
        y[-j:, 1] = 0
    elif j < 0:
        y[-j:, 1] = x[:j, 1]
        y[:-j, 1] = 0
    return y


class _Cache:
    """Sample cache: decoded FLAC and pitch-shifted versions (LRU by bytes)."""

    def __init__(self, sample_dir: Path, max_bytes=3e9):
        self.dir = sample_dir
        self.max = max_bytes
        self.raw: dict[str, np.ndarray] = {}
        self.shifted: OrderedDict = OrderedDict()
        self.bytes = 0

    def load(self, name):
        if ALIGN['on']:
            x = self.raw.get((name, 'aligned'))
            if x is None:
                x = _align_lr(self._load(name))
                self.raw[(name, 'aligned')] = x
            return x
        return self._load(name)

    def _load(self, name):
        x = self.raw.get(name)
        if x is None:
            log_access(self.dir / name)
            x, sr = sf.read(need(self.dir / name, 'Salamander 钢琴采样'), dtype='float32', always_2d=True)
            if x.shape[1] == 1:
                x = np.repeat(x, 2, axis=1)
            if sr != SR:
                from ._dsp import resample_sr
                x = resample_sr(x, sr, SR)
            x = _trim_late_spike(x)
            self.raw[name] = x
        return x

    def get(self, name, ratio):
        key = (name, round(ratio, 7), ALIGN['on'])
        hit = self.shifted.get(key)
        if hit is not None:
            self.shifted.move_to_end(key)
            return hit
        y = resample_by(self.load(name), ratio)
        env = np.abs(y[:int(0.6 * SR)]).max(axis=1)
        thr = 0.1 * env.max() if env.size else 0.0
        on = int(np.argmax(env > thr)) if thr > 0 else 0
        hit = (y, on)
        self.shifted[key] = hit
        self.bytes += y.nbytes
        while self.bytes > self.max and len(self.shifted) > 1:
            _, (old, _) = self.shifted.popitem(last=False)
            self.bytes -= old.nbytes
        return hit


class Instrument:
    def __init__(self, sfz_path=SFZ_FILE):
        s = _sfz.parse(need(sfz_path, 'Salamander 钢琴 SFZ'))
        self.cc_default = s.cc_defaults()
        self.regions = [Region(r, s.curves) for r in s.regions]
        self.cache = _Cache(Path(sfz_path).parent / s.control.get('default_path', ''))
        self.sw_labels = sorted({r.sw_label for r in self.regions if r.sw_label})

    def attack_regions(self, key, tuning, cc):
        return sorted((r for r in self.regions
                       if r.trigger == 'attack' and r.lokey <= key <= r.hikey and r.lokey >= 0
                       and (r.sw_label is None or r.sw_label.lower() == tuning)
                       and r.active(cc)), key=lambda r: r.lovel)

    def release_regions(self, key, vel, cc):
        return [r for r in self.regions if r.trigger == 'release' and r.lokey <= key <= r.hikey
                and r.lovel <= vel <= r.hivel and r.active(cc)]

    def pedal_regions(self, down: bool, cc):
        val = 127.0 if down else 0.0
        return [r for r in self.regions if 64 in r.on_cc
                and r.on_cc[64][0] <= val <= r.on_cc[64][1] and r.active(cc)]


_INSTRUMENT: Instrument | None = None


def instrument() -> Instrument:
    global _INSTRUMENT
    if _INSTRUMENT is None:
        _INSTRUMENT = Instrument()
    return _INSTRUMENT


# ----------------------------------------------------------------------------- helpers
def _layer_weights(regs, vel, xfade):
    """Adjacent-layer crossfade. xfade=0 -> hard SFZ switching; 1 -> full-width blend."""
    if len(regs) == 1:
        return [(regs[0], 1.0)]
    centers = [(r.lovel + r.hivel) / 2.0 for r in regs]
    if vel <= centers[0]:
        return [(regs[0], 1.0)]
    if vel >= centers[-1]:
        return [(regs[-1], 1.0)]
    i = int(np.searchsorted(centers, vel, side='right')) - 1
    lo, hi = regs[i], regs[i + 1]
    b = lo.hivel + 0.5                   # the SFZ's hard switch point
    c0, c1 = centers[i], centers[i + 1]
    t = 0.5 * (vel - c0) / (b - c0) if vel < b else 0.5 + 0.5 * (vel - b) / (c1 - b)
    if xfade <= 0:
        t = 0.0 if vel < b else 1.0
    else:
        t = float(np.clip((t - 0.5) / xfade + 0.5, 0.0, 1.0))
    # onset-aligned layers of the same string are strongly correlated (measured: an
    # equal-power law overshoots by ~+1.5 dB mid-fade), so weights sum to one in amplitude
    t = t * t * (3 - 2 * t)
    w0, w1 = 1.0 - t, t
    return [(r, w) for r, w in ((lo, w0), (hi, w1)) if w > 1e-4]


def _exp_tail(n_from, n_len, time_s, floor_db=-80.0):
    """Exponential decay reaching -60 dB after time_s; returns (env, stop_len)."""
    time_s = max(time_s, 0.01)
    tau = time_s * SR / np.log(1000.0)
    stop = int(tau * (-floor_db / 20.0) * np.log(10.0))
    k = np.arange(n_len - n_from, dtype=np.float64)
    return np.exp(-k / tau), stop


def _felt_voice(y, onset, midi, vel, amount, rng):
    """Felt ('Klavier-Moderator') model applied to one voice.

    * dynamic low-pass: cutoff tracks pitch and velocity (softer -> darker),
      2nd-order Butterworth + 1st-order at 1.8x (about -18 dB/oct above),
    * transient softening: raised-cosine attack over 4-14 ms + 60 ms dip of the
      first peak (felt hammer = longer contact time),
    * key-bed thump: short low-passed noise + 90-140 Hz damped sine at the onset.
    """
    v = vel / 127.0
    f0 = float(midi_to_hz(midi))
    fc = max(f0 * (1.8 + 5.0 * v ** 1.6), 350.0 + 1500.0 * v ** 1.5)
    fc = fc * (1.0 / amount) ** 0.5 if amount > 0 else 20000.0
    fc = float(min(fc, 14000.0))
    z = sosfilt(butter('lowpass', fc, order=2), y)
    z = sosfilt(butter('lowpass', min(1.8 * fc, 20000.0), order=1), z)
    y = (1 - amount) * y + amount * z if amount < 1 else z
    # transient softening: smooth gain dip over the hammer transient (no gain steps)
    att = (0.004 + 0.010 * (1 - v)) * max(amount, 0.2)
    t = (np.arange(len(y)) - onset) / SR
    rise = np.clip((t + 0.001) / 0.002, 0.0, 1.0)
    rise = rise * rise * (3 - 2 * rise)                      # smoothstep 0->1 around onset
    bump = np.where(t < 0.001, rise, np.exp(-np.maximum(t - 0.001, 0.0) / att))
    tail = rise * np.exp(-np.maximum(t, 0.0) / 0.06)
    gain = 1.0 - amount * (0.55 * bump + 0.30 * tail)
    y = y * gain[:, None].astype(np.float32)
    # key-bed / hammer-felt thump (mechanical noise), a few dB under the note
    n = int(0.08 * SR)
    t = np.arange(n) / SR
    fth = rng.uniform(90, 140)
    thump = np.sin(2 * np.pi * fth * t) * np.exp(-t / 0.025)
    nz = rng.standard_normal(n) * np.exp(-t / 0.012)
    nz = sosfilt(butter('lowpass', 900.0, order=2), nz).astype(np.float64)
    th = (0.7 * thump + 0.5 * nz / (np.abs(nz).max() + 1e-9))
    lvl = amount * (0.0035 + 0.006 * v)
    th = (th / (np.abs(th).max() + 1e-9) * lvl).astype(np.float32)
    ts = max(0, onset - int(0.004 * SR))
    stereo_th = np.stack([th * 0.9, th], axis=1)
    add_at(y, stereo_th, ts)
    return y


def _una_corda_voice(y, onset, midi, vel, amount, rng):
    """Soft-pedal colour for one voice (see render_piano(una_corda=...))."""
    v = vel / 127.0
    f0 = float(midi_to_hz(midi))
    t = (np.arange(len(y)) - onset) / SR
    # softer hammer: take part of the first transient off (lighter than the felt model)
    att = 0.003 + 0.006 * (1 - v)
    rise = np.clip((t + 0.001) / 0.002, 0.0, 1.0)
    rise = rise * rise * (3 - 2 * rise)
    bump = np.where(t < 0.001, rise, np.exp(-np.maximum(t - 0.001, 0.0) / att))
    y = y * (1.0 - amount * 0.35 * bump)[:, None].astype(np.float32)
    # darker: pitch / velocity tracking low-pass, mixed in
    fc = float(min(max(f0 * (3.0 + 6.0 * v), 1200.0 + 2500.0 * v), 14000.0))
    z = sosfilt(butter('lowpass', fc, order=2), y)
    y = (1 - 0.6 * amount) * y + 0.6 * amount * z
    # the string the shifted hammer misses rings along, a cent or two apart (read at a slightly
    # different speed = constant detune), rising over ~0.15 s, low-passed
    cents = rng.uniform(0.8, 1.8) * (1 if rng.random() < 0.5 else -1)
    r = 2.0 ** (cents / 1200.0)
    idx = np.arange(len(y), dtype=np.float64)
    src = onset + (idx - onset) * r
    ok = (src >= 0) & (src <= len(y) - 1)
    free = np.zeros_like(y)
    for c in range(y.shape[1]):
        free[ok, c] = np.interp(src[ok], idx, z[:, c]).astype(np.float32)
    grow = (1 - np.exp(-np.maximum(t, 0.0) / 0.15)).astype(np.float32)
    free = sosfilt(butter('lowpass', min(2500.0, 4 * f0), order=2), free) * grow[:, None]
    y = y + free * np.float32(db2a(-16.0) * amount)
    # key-bed knock
    n = int(0.06 * SR)
    tt = np.arange(n) / SR
    knock = np.sin(2 * np.pi * rng.uniform(100, 150) * tt) * np.exp(-tt / 0.02)
    nz = sosfilt(butter('lowpass', 1200.0, order=2), rng.standard_normal(n) * np.exp(-tt / 0.01))
    th = 0.7 * knock + 0.5 * nz / (np.abs(nz).max() + 1e-9)
    th = (th / (np.abs(th).max() + 1e-9) * amount * (0.002 + 0.004 * v)).astype(np.float32)
    add_at(y, np.stack([th * 0.9, th], axis=1), max(0, onset - int(0.003 * SR)))
    return y.astype(np.float32)


# ----------------------------------------------------------------------------- sympathetic resonance
SYMP_PARTIALS = 12          # partial modes per string
SYMP_FMAX = 9000.0          # highest modelled partial
SYMP_CAL = 0.060            # measured: sympathetic=0.35 -> resonance ~ -21 dB re. the dry pedalled chord


def _string_table(tuning):
    """key -> (f0 Hz, inharmonicity B, {partial n: measured Hz}) from lib/_piano_tuning.py
    (written by tools/measure_piano_tuning.py); falls back to 12-TET and a generic B."""
    try:
        from . import _piano_tuning as pt
        tab = getattr(pt, tuning.upper(), {})
    except ImportError:
        tab = {}
    out = {}
    for key in range(21, 109):
        c, B, meas = tab.get(key, (0.0, 10 ** (-3.6 + 0.022 * max(0, key - 50)), ()))
        meas = dict(meas) if isinstance(meas, tuple) else {}
        out[key] = (440.0 * 2 ** ((key - 69) / 12 + c / 1200), float(B), meas)
    return out


@lru_cache(maxsize=2)
def _symp_modes(tuning, t60_mid):
    """Partial modes of all 88 open strings: key -> arrays (freq, T60, amp, phase_L, phase_R).
    Mode phases differ per channel (two pick-up points on the soundboard -> decorrelated
    stereo), amplitudes fall as 1/n, higher partials decay faster."""
    rng = np.random.default_rng(88)
    modes = {}
    for key, (f0, B, meas) in _string_table(tuning).items():
        n = np.arange(1, SYMP_PARTIALS + 1, dtype=np.float64)
        fn = n * f0 * np.sqrt(1 + B * n * n)
        fn = np.array([meas.get(int(k), f) for k, f in zip(n, fn)])   # measured partials win
        ok = fn < SYMP_FMAX
        n, fn = n[ok], fn[ok]
        T = float(np.clip(t60_mid * (262.0 / f0) ** 0.35, 0.8, 4.0)) / (1 + (fn / 2500.0) ** 2)
        modes[key] = (fn, T, 1.0 / n, rng.uniform(0, 2 * np.pi, len(n)), rng.uniform(0, 2 * np.pi, len(n)))
    return modes


def _modes_on_grid(fn, T, amp, phi, N):
    """Exact DTFT of sum_m amp_m r_m^k sin(theta_m k + phi_m) (k >= 0) on the rfft bins of an
    N-point grid, evaluated only within +-W of each mode (W = max(40 Hz, 30 bandwidths),
    tapered over the outer 30 %). Returns (bin indices, complex values)."""
    df = SR / N
    idx, val = [], []
    for f, T60, a, ph in zip(fn, T, amp, phi):
        tau = T60 / 6.908
        r = np.exp(-1.0 / (tau * SR))
        th = 2 * np.pi * f / SR
        W = max(40.0, 30.0 / (np.pi * tau))
        k = np.arange(max(1, int((f - W) / df)), min(N // 2, int((f + W) / df) + 1))
        z1 = np.exp(-2j * np.pi * k / N)
        H = (np.exp(1j * ph) / (1 - r * np.exp(1j * th) * z1)
             - np.exp(-1j * ph) / (1 - r * np.exp(-1j * th) * z1)) / 2j * a
        u = np.abs(k * df - f) / W
        H *= 0.5 + 0.5 * np.cos(np.pi * np.clip((u - 0.7) / 0.3, 0, 1))
        idx.append(k)
        val.append(H)
    return np.concatenate(idx), np.concatenate(val)


@lru_cache(maxsize=3)
def _symp_grid(tuning, t60_mid, N):
    """Total resonator-bank response on an N-point grid (L, R), each key's own share (for
    excluding the struck string) and the unit-energy normalisation."""
    modes = _symp_modes(tuning, t60_mid)
    HL = np.zeros(N // 2 + 1, dtype=np.complex128)
    HR = np.zeros_like(HL)
    own = {}
    for key, (fn, T, amp, pl, pr) in modes.items():
        il, vl = _modes_on_grid(fn, T, amp, pl, N)
        _, vr = _modes_on_grid(fn, T, amp, pr, N)
        np.add.at(HL, il, vl)
        np.add.at(HR, il, vr)
        own[key] = (il, vl, vr)
    e = 0.0
    for H in (HL, HR):
        p2 = np.abs(H) ** 2
        e += (p2[0] + 2 * p2[1:-1].sum() + p2[-1]) / N
    return HL, HR, own, 1.0 / np.sqrt(e / 2)


def _sympathetic(n_out, attack_voices, peds, tuning, amount, t60_mid, damp_s):
    """Sympathetic resonance of the undamped strings while the sustain pedal is down.

    For each pedal interval, every struck note inside it (mono, gated by the pedal: 50 ms
    damper lift, 20 ms fall) drives the resonator bank minus its own string; the result is
    released with the dampers (exp(-t / damp_s) after pedal-up), so a later pedal-down
    starts from silent strings."""
    res = np.zeros((n_out, 2))
    up_r = int(0.05 * SR)
    dn_r = int(0.02 * SR)
    tail = int(8 * damp_s * SR)
    for d, u in peds:
        a = int(round(d * SR))
        b = int(round(min(u, n_out / SR) * SR))
        if b <= a or a >= n_out:
            continue
        L = b - a + dn_r
        gate = np.ones(L)
        gate[:min(up_r, L)] = cos_ramp(min(up_r, L))
        gate[b - a:] = cos_ramp(dn_r, rising=False)[:L - (b - a)]
        xk = {}
        for s_idx, key, v in attack_voices:
            lo, hi = max(s_idx, a), min(s_idx + len(v), a + L)
            if hi <= lo:
                continue
            buf = xk.setdefault(key, np.zeros(L))
            buf[lo - a:hi - a] += v[lo - s_idx:hi - s_idx].mean(axis=1) * gate[lo - a:hi - a]
        if not xk:
            continue
        L_out = L + tail
        N = 1 << int(np.ceil(np.log2(L + 4.5 * SR)))
        HL, HR, own, norm = _symp_grid(tuning, round(t60_mid, 3), N)
        X = np.fft.rfft(sum(xk.values()), N)
        YL, YR = HL * X, HR * X
        for key, x in xk.items():
            if key in own:
                il, vl, vr = own[key]
                Xk = np.fft.rfft(x, N)[il]
                np.add.at(YL, il, -vl * Xk)
                np.add.at(YR, il, -vr * Xk)
        # the band-limited mode windows smear each IR by ~1/W (25 ms) to both sides; the
        # part before t=0 sits at the end of the circular buffer -> take it as a pre-roll
        P = int(0.04 * SR)
        yl, yr = np.fft.irfft(YL, N), np.fft.irfft(YR, N)
        y = np.stack([np.concatenate([yl[N - P:], yl[:L_out]]),
                      np.concatenate([yr[N - P:], yr[:L_out]])], axis=1)
        D = np.ones(P + L_out)
        D[:P] = cos_ramp(P)
        k = np.arange(L_out - (b - a))
        D[P + b - a:] = np.exp(-k / (damp_s * SR))
        y *= (D * norm * SYMP_CAL * amount)[:, None]
        s0 = a - P
        lo = max(0, -s0)
        m = min(len(y), n_out - s0)
        res[s0 + lo:s0 + m] += y[lo:m]
    return res.astype(np.float32)


# ----------------------------------------------------------------------------- main API
def render_piano(notes, pedal=(), felt=False, duration=None, *, align_stereo=False, **kw) -> np.ndarray:
    """See _render_piano. align_stereo=True time-aligns each sample's two channels first (see _align_lr);
    default False leaves the output unchanged sample for sample."""
    prev = ALIGN['on']
    ALIGN['on'] = bool(align_stereo)
    try:
        return _render_piano(notes, pedal, felt, duration, **kw)
    finally:
        ALIGN['on'] = prev


def _render_piano(notes, pedal=(), felt=False, duration=None, *, tuning='natural',
                 string_res=0.5, hammer_noise=0.5, pedal_noise=0.35, release=0.5,
                 veltrack=0.73, velocity_xfade=1.0, felt_amount=1.0, preroll_s=None,
                 gain_db=-6.0, sympathetic=0.35, sympathetic_t60=2.5, damper_s=0.18,
                 una_corda=0.0, seed=0) -> np.ndarray:
    """Render piano notes to a (n, 2) float32 array at 48 kHz.

    notes : [(start_s, dur_s, midi, velocity)]; velocity 1..127 (0 = skipped)
    pedal : [(down_s, up_s)] sustain pedal; a key released while the pedal is down
            keeps sounding until the pedal comes up, then follows the release envelope.
    felt  : felt-piano model (dark, soft attack, more mechanism noise).
    duration : output length in s (None = until the last tail dies away).
    tuning : 'natural' (as recorded, stretched) or 'retuned' (Markus Fiedler retune).
    string_res / hammer_noise / pedal_noise / release / veltrack : the SFZ's own
            controls (0..1, SFZ default 0.5 / 0.5 / 0.5 / 0.5 / 0.73). pedal_noise
            defaults to 0.35 here (about -3 dB under the SFZ default) as requested.
    velocity_xfade : 0 = hard layer switching as in the SFZ, 1 = widest blend.
    gain_db : output trim; default -6 dB leaves headroom (a single ff note peaks
            around -7 dBFS, dense ff chords can still exceed 0 dBFS in float).
    sympathetic : pedal sympathetic resonance amount (0 = off, default 0.35 = light;
            1 ~ resonance 12 dB under the dry pedalled sound). While the pedal is down the
            struck notes drive a bank of modal resonators tuned to the other 87 open
            strings (measured tuning and inharmonicity); dampers stop it `damper_s` after
            pedal-up. Without pedal the output is unchanged.
    sympathetic_t60 : resonator decay time around C4 (s); bass longer, treble shorter.
    una_corda : soft-pedal colour, 0..1 (default 0 = off, output unchanged sample for sample).
            Per note: a softer hammer (part of the first 3-9 ms taken off), a darker tone
            (pitch/velocity-tracking low-pass, mixed in), the string the hammer now misses
            ringing along a cent or two apart (a slowly rising, low-passed copy ~16 dB down),
            and a key-bed knock; plus more mechanism: release/damper samples +4 dB and pedal
            noise +3 dB at 1.0, overall level -2 dB at 1.0. Uses its own random stream, so
            note/noise choices are the same as without it.
    """
    inst = instrument()
    rng = np.random.default_rng(seed)
    tuning = tuning.lower()
    if tuning not in [s.lower() for s in inst.sw_labels]:
        raise ValueError(f'tuning must be one of {inst.sw_labels}')
    cc = dict(inst.cc_default)
    cc.update({CC['string_res']: string_res, CC['hammer_noise']: hammer_noise,
               CC['pedal_noise']: pedal_noise, CC['release']: release, CC['veltrack']: veltrack})
    if preroll_s is None:
        preroll_s = 0.03 if felt else 0.012
    uc_rng = np.random.default_rng([int(seed), 0x0C0DA]) if una_corda > 0 else None

    evs = []
    for n in notes:
        st, du, m = float(n[0]), float(n[1]), float(n[2])
        vel = float(n[3]) if len(n) > 3 else 80.0
        if vel <= 0 or du <= 0:
            continue
        evs.append((st, du, m, float(np.clip(vel, 1, 127))))
    evs.sort(key=lambda e: e[0])
    peds = sorted((float(d), float(u) if u is not None else np.inf) for d, u in pedal)

    def damper_time(t_up):
        for d, u in peds:
            if d <= t_up < u:
                return u
        return t_up

    # self-masking: the next strike of the same key cuts the previous voice
    next_same = {}
    last_idx = {}
    for i, (st, du, m, vel) in enumerate(evs):
        k = int(round(m))
        if k in last_idx:
            next_same[last_idx[k]] = st
        last_idx[k] = i

    voices = []   # (start_index, stereo array)
    attack_voices = []   # (start_index, key, stereo array): drives the sympathetic resonance
    end_s = 0.0
    release_cut = {}
    for i, (st, du, m, vel) in enumerate(evs):
        key = int(round(m))
        td = damper_time(st + du)
        cut = next_same.get(i)
        regs = inst.attack_regions(key, tuning, cc)
        if not regs:
            continue
        layers = [(r, w) for r, w in _layer_weights(regs, vel, velocity_xfade)]
        pre = int(preroll_s * SR)
        acc = None
        onset_in_acc = pre
        for r, w in layers:
            ratio = 2.0 ** ((m - r.keycenter) / 12.0 + r.tune / 1200.0)
            y, on = inst.cache.get(r.sample, ratio)
            p = min(pre, on)
            seg = y[on - p:]
            g = w * r.vel_gain(vel, cc) * db2a(r.volume) * r.amp_factor(cc)
            if acc is None:
                acc = np.zeros((len(seg) + pre - p, 2), dtype=np.float32)
            n_ = min(len(seg), len(acc) - (pre - p))
            acc[pre - p:pre - p + n_] += (g * seg[:n_]).astype(np.float32)
        rel_t = max(r.release_time(cc) for r, _ in layers)
        off_t = min(r.off_time for r, _ in layers)
        # envelope: pre-roll fade-in, sustain, damper release, optional self-mask cut
        n_tot = len(acc)
        env = np.ones(n_tot, dtype=np.float64)
        fi = min(pre, int(0.003 * SR)) or 1
        env[:fi] *= cos_ramp(fi)
        stop = n_tot
        kd = onset_in_acc + int(round((td - st) * SR))
        if kd < n_tot:
            e, s_ = _exp_tail(kd, n_tot, rel_t)
            env[kd:] *= e
            stop = min(stop, kd + s_)
        if cut is not None:
            kc = onset_in_acc + int(round((cut - st) * SR))
            if kc < n_tot:
                e, s_ = _exp_tail(kc, n_tot, off_t)
                env[kc:] *= e
                stop = min(stop, kc + s_)
        stop = max(min(stop, n_tot), 16)
        v_ = acc[:stop] * env[:stop, None].astype(np.float32)
        v_ = fade_edges(v_, 0.0, 0.005)
        if felt:
            v_ = _felt_voice(v_, onset_in_acc, m, vel, felt_amount, rng)
        if una_corda > 0:
            v_ = _una_corda_voice(v_, onset_in_acc, m, vel, float(una_corda), uc_rng)
        s_idx = int(round(st * SR)) - onset_in_acc
        voices.append((s_idx, v_))
        attack_voices.append((s_idx, key, v_))
        end_s = max(end_s, (s_idx + len(v_)) / SR)

        # release-triggered layers (string resonance + hammer/key-release noise)
        for r in inst.release_regions(key, int(round(vel)), cc):
            ratio = 2.0 ** ((m - r.keycenter) / 12.0 + r.tune / 1200.0)
            y = inst.cache.load(r.sample) if abs(ratio - 1) < 1e-9 else inst.cache.get(r.sample, ratio)[0]
            g = (r.vel_gain(vel, cc) * db2a(r.volume - r.rt_decay * (td - st)) * r.amp_factor(cc))
            if felt and 'rel' in r.sample and 'harm' not in r.sample:
                g *= db2a(10.0 * felt_amount)
            if una_corda > 0 and 'rel' in r.sample and 'harm' not in r.sample:
                g *= db2a(4.0 * una_corda)
            if g < 1e-5:
                continue
            seg = fade_edges(y * np.float32(g), 0.001, 0.01)
            ks = (r.group, key)
            ri = int(round(td * SR))
            if ks in release_cut:                     # note_polyphony=1 in its group
                pi_, pv = release_cut[ks]
                kc = ri - pi_
                if 0 <= kc < len(pv):
                    e, s_ = _exp_tail(kc, len(pv), r.off_time)
                    pv[kc:] *= e[:, None].astype(np.float32)
                    pv[min(len(pv), kc + s_):] = 0
            voices.append((ri, seg))
            release_cut[ks] = (ri, seg)
            end_s = max(end_s, td + len(seg) / SR)

    # pedal mechanism noise
    last_ped = None
    for d, u in peds:
        for t, down in ((d, True), (u, False)):
            if not np.isfinite(t):
                continue
            regs = inst.pedal_regions(down, cc)
            x = rng.random()
            pick = [r for r in regs if r.lorand <= x < r.hirand] or regs
            if not pick:
                continue
            r = pick[0]
            g = db2a(r.volume) * r.amp_factor(cc) * (db2a(4.0 * felt_amount) if felt else 1.0)
            if una_corda > 0:
                g *= db2a(3.0 * una_corda)
            seg = fade_edges(inst.cache.load(r.sample) * np.float32(g), 0.001, 0.01)
            ri = int(round(t * SR))
            if last_ped is not None:                  # polyphony=1, off_time
                pi_, pv = last_ped
                kc = ri - pi_
                if 0 <= kc < len(pv):
                    e, s_ = _exp_tail(kc, len(pv), r.off_time)
                    pv[kc:] *= e[:, None].astype(np.float32)
                    pv[min(len(pv), kc + s_):] = 0
            voices.append((ri, seg))
            last_ped = (ri, seg)
            end_s = max(end_s, t + len(seg) / SR)

    if sympathetic > 0 and peds:
        end_s = max(end_s, max(min(u, end_s) for _, u in peds) + 8 * damper_s)
    total = duration if duration is not None else end_s + 0.05
    out = np.zeros((max(1, int(round(total * SR))), 2), dtype=np.float32)
    for s_idx, v_ in voices:
        add_at(out, v_, s_idx)
    if sympathetic > 0 and peds and attack_voices:
        out += _sympathetic(len(out), attack_voices, peds, tuning, float(sympathetic),
                            float(sympathetic_t60), float(damper_s))
    if felt:
        out = sosfilt(biquad('peak', 180.0, 0.8, 2.0 * felt_amount), out)
        out = sosfilt(biquad('highshelf', 5500.0, 0.7, -3.0 * felt_amount), out)
    out *= np.float32(db2a(gain_db - (2.0 * una_corda if una_corda > 0 else 0.0)))
    return fade_edges(out, 0.002, 0.02 if duration is not None else 0.005)
