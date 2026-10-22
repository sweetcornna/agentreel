"""瓷瓯 - porcelain bowls struck with a chopstick, tuned to a pentatonic set.

    bowl(note, vel=60, hardness=0.5, *, water=0.0, ...)       -> (n, 2) float32   one stroke
    render_bowls(bowls, *, duration=None, ...)                 -> (n, 2) float32   score list
    BOWL_SCALE  = D E F# A B over two octaves, D5..B6 (MIDI 74..95)
    bowl_scale(low='D5', octaves=2)                            -> [midi]

bowls (score.BOWLS): [(start_s, midi, vel, hardness 0..1)]

model='rec' (default): real recordings, CC0, see <samples>/objects/manifest.json:
  'bowl'  3D_Blanco, a ceramic bowl, 4 clean taps (fundamental 821 Hz, partials x2.36, x4.18)
  'plate' Chilsville, a porcelain plate, 3 hits (571 Hz, x2.12, x3.35), long ring
  'cup'   that_Finn, a mug tapped with a fingernail (2667 Hz), short and bright
Every note is ONE recorded stroke played back at another speed (varispeed = the same bowl
scaled in size: all modal frequencies move together, and with a constant material loss factor
the decay times scale with the period too, which is what varispeed does). A set tuned this
way sounds like a family of bowls of one make. source='bowl' uses that one bowl for every
note (a consistent set); 'mixed' takes the recording nearest in pitch.
  hardness  picks the brighter or duller of the recorded taps and tilts the first ~0.1 s
            (the contact) brighter / duller; low velocity also softens the contact.
  water     cents of pitch glide after the stroke (a bowl with water tilted / swirled);
            water_shape 'bend' (glide and stay), 'dip' (down and back), 'wobble' (sloshing).
  dur       None = let it ring; a number = damped by a hand after dur seconds.

model='modal': modal synthesis with the ratios, initial levels and decay times MEASURED on the
'bowl' recording (each mode a close doublet, as in a real bowl); hardness changes the
contact time of a half-sine force (1.4 ms soft .. 0.25 ms hard) and the modes are weighted by
the smooth envelope of that force spectrum relative to the measured (medium) stroke; plus a short band-passed click
for the chopstick itself. No recording involved; useful when many strokes must never repeat.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import soundfile as sf
from scipy import signal

from ._dsp import SR, butter, cos_ramp, db2a, fade_edges, midi_to_hz, pan_gains, resample_by
from ._paths import SAMPLES, log_access, need

OBJ = SAMPLES / 'objects'

_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
PENTA_PCS = (2, 4, 6, 9, 11)          # D E F# A B (D gong)


def _midi(n):
    if isinstance(n, str):
        s = n.strip()
        pc = _PC[s[0].upper()] + s[1:].count('#') - s[1:].count('b')
        octv = int(s.lstrip('ABCDEFGabcdefg#b'))
        return pc + 12 * (octv + 1)
    return float(n)


def bowl_scale(low='D5', octaves=2):
    """Pentatonic D-gong set from `low` upwards (only D E F# A B pitch classes)."""
    lo = int(_midi(low))
    return [m for m in range(lo, lo + 12 * octaves) if m % 12 in PENTA_PCS]


BOWL_SCALE = bowl_scale('D5', 2)      # [74, 76, 78, 81, 83, 86, 88, 90, 93, 95]

# recorded strokes: source -> (file, [(approx. onset s, end s)], highpass Hz, nominal fundamental Hz)
# (the ends stop before the next event in the recording: scrapes / the next hit)
_SOURCES = {
    'bowl': ('bowl_ceramic_3dblanco', [(1.45, 6.2), (6.27, 10.1), (10.16, 14.1), (19.6, None)],
             420.0, 821.0),
    'plate': ('plate_porcelain_chilsville', [(2.71, 7.6), (8.11, 13.3), (13.74, None)], 300.0, 571.0),
    'cup': ('cup_ceramic_that_finn', [(0.03, None)], 1200.0, 2667.0),
}
# measured on the 'bowl' recording (per-mode band envelope, exponential fit extrapolated to the
# stroke; medium strokes 1 and 2): ratio, INITIAL level dB re. the fundamental, T60 s at 821 Hz
_MODES = [(1.0, 0.0, 3.7), (2.36, 2.0, 1.45), (4.18, 0.0, 0.47), (6.36, -18.0, 0.40),
          (8.90, -24.0, 0.30)]
_MODAL_F0 = 821.0
_LEVEL_DB = -4.0         # calibration (measured over BOWL_SCALE, vel 127): 0.4 s RMS ~ -28 dBFS,
                         # peak -13 (hardness 0.25) .. -4 dBFS (hardness 1)


@lru_cache(maxsize=8)
def _raw(name):
    log_access(OBJ / f'{name}.ogg')
    x, sr = sf.read(str(need(OBJ / f'{name}.ogg', f'objects 录音 {name}')), dtype='float64', always_2d=True)
    return x.mean(axis=1), sr


def _fundamental(seg, sr, lo=250.0, hi=4000.0):
    """Frequency of the strongest spectral peak in [lo, hi] (the bowl's (2,0) mode) with
    parabolic interpolation, measured 50-600 ms after the stroke."""
    a, b = int(0.05 * sr), min(len(seg), int(0.6 * sr))
    z = seg[a:b] * np.hanning(b - a)
    nfft = 1 << 18
    S = np.abs(np.fft.rfft(z, nfft))
    f = np.fft.rfftfreq(nfft, 1 / sr)
    band = (f > lo) & (f < hi)
    k = int(np.flatnonzero(band)[np.argmax(S[band])])
    y0, y1, y2 = np.log(S[k - 1:k + 2] + 1e-12)
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return float((k + d) * sr / nfft)


@lru_cache(maxsize=16)
def _hits(source):
    """[(audio at source sr, sr, f0 Hz, brightness)] - trimmed, high-passed, level-normalised."""
    name, onsets, hp, nominal = _SOURCES[source]
    x, sr = _raw(name)
    x = signal.sosfiltfilt(signal.butter(4, hp, 'highpass', fs=sr, output='sos'), x)
    out = []
    for t0, t1 in onsets:
        a = max(0, int((t0 - 0.05) * sr))
        b = int(t1 * sr) if t1 else len(x)
        seg = x[a:b]
        env = np.abs(seg)
        on = int(np.argmax(env > env.max() * 0.3))              # the stroke rises in < 1 ms
        on = max(0, on - int(0.0015 * sr))
        seg = seg[on:].copy()
        seg[:int(0.0015 * sr)] *= cos_ramp(int(0.0015 * sr))
        f0 = _fundamental(seg, sr, nominal * 0.85, nominal * 1.15)
        n30 = int(0.03 * sr)
        hi = signal.sosfilt(signal.butter(4, 3000, 'highpass', fs=sr, output='sos'), seg[:n30])
        bright = float(10 * np.log10(np.sum(hi ** 2) / (np.sum(seg[:n30] ** 2) + 1e-20) + 1e-12))
        # level = mean of the ring level (50-300 ms RMS, dB) and the contact peak (dB): a harder
        # recorded stroke keeps part of its taller transient, but not all of it
        ring = 20 * np.log10(np.sqrt(np.mean(seg[int(0.05 * sr):int(0.3 * sr)] ** 2)) + 1e-12)
        peak = 20 * np.log10(np.abs(seg).max() + 1e-12)
        seg = seg * db2a(-13.0 - 0.5 * (ring + peak))
        seg[-int(0.4 * sr):] *= cos_ramp(int(0.4 * sr), rising=False)
        out.append((seg.astype(np.float32), sr, f0, bright))
    return out


def _vel_gain_db(vel):
    return -30.0 * (1 - float(np.clip(vel, 1, 127)) / 127) ** 1.3


def _tilt(x, amount, sr, tau=0.1):
    """Brighten (amount>0) / dull (amount<0) the stroke's contact: a high shelf / low-pass whose
    effect fades out with time constant tau (the ring itself keeps the recorded colour)."""
    if abs(amount) < 1e-3:
        return x
    if amount > 0:
        b, a = signal.butter(2, 2500, 'highpass', fs=sr)
        wet = x + signal.lfilter(b, a, x) * (db2a(6 * amount) - 1)
    else:
        b, a = signal.butter(2, 1800 + 3000 * (1 + amount), 'lowpass', fs=sr)
        wet = signal.lfilter(b, a, x)
    t = np.arange(len(x)) / sr
    k = np.exp(-t / tau)
    return x * (1 - k) + wet * k


def _warp(x, cents_curve):
    """Time-varying playback speed 2**(c/1200) (cubic Hermite read), same length as x."""
    rate = 2.0 ** (cents_curve / 1200.0)
    pos = np.concatenate([[0.0], np.cumsum(rate[:-1])])
    pos = pos[pos < len(x) - 3]
    i = pos.astype(int)
    f = pos - i
    xm1 = x[np.maximum(i - 1, 0)]
    x0, x1, x2 = x[i], x[i + 1], x[i + 2]
    c1 = 0.5 * (x1 - xm1)
    c2 = xm1 - 2.5 * x0 + 2 * x1 - 0.5 * x2
    c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1)
    return (((c3 * f + c2) * f + c1) * f + x0).astype(np.float32)


def _water_curve(n, cents, shape, time_s, delay_s):
    t = np.arange(n) / SR
    u = np.clip((t - delay_s) / max(time_s, 1e-3), 0, None)
    if shape == 'bend':
        s = np.where(u < 1, u * u * (3 - 2 * u), 1.0)
    elif shape == 'dip':
        s = np.sin(np.pi * np.clip(u, 0, 1)) ** 2
    elif shape == 'wobble':
        s = np.sin(2 * np.pi * u) * np.exp(-u / 1.5) * (u > 0)
    else:
        raise ValueError("water_shape: 'bend' | 'dip' | 'wobble'")
    return cents * s


def _modal_stroke(f0, hardness, rng, length_s):
    n = int(length_s * SR)
    t = np.arange(n) / SR
    tc = (1.4 - 1.15 * np.clip(hardness, 0, 1)) * 1e-3           # contact time 1.4 .. 0.25 ms
    y = np.zeros(n)

    def force(f, tc):
        # upper envelope of a half-sine contact force spectrum |cos(pi f tc)| / |1 - (2 f tc)^2|:
        # flat below ~1/(2 tc), -12 dB/oct above. The notches of the exact spectrum are left out
        # (a real contact is not a perfect half-sine); dividing by them made a softer stroke
        # louder in mode 2 on some notes (up to +14 dB) - found by selfcheck bowl, 2026-09-27.
        return 1.0 / (1.0 + (2 * f * tc) ** 2)

    tc_ref = (1.4 - 1.15 * 0.5) * 1e-3          # the measured levels belong to a medium stroke
    g0 = force(f0, tc) / force(f0, tc_ref)
    for ratio, lvl, t60 in _MODES:
        f = f0 * ratio * (1 + (rng.normal(0, 0.003) if ratio > 1 else 0.0))   # fundamental exact
        if f > 0.45 * SR:
            continue
        g = force(f, tc) / force(f, tc_ref) / g0     # harder/softer than the measured stroke
        g = float(np.clip(g, db2a(-30.0), db2a(20.0)))
        T = t60 * _MODAL_F0 / f0
        # each mode is a close doublet (the two degenerate modes of a not-quite-round bowl); the
        # fundamental's stronger half sits exactly on f0 (tuning), its weaker half beats slowly
        if ratio > 1:
            split = f * rng.uniform(0.0008, 0.003)
            w = rng.uniform(0.4, 0.6)
            pair = ((-split / 2, w), (split / 2, 1 - w))
        else:
            pair = ((0.0, 0.75), (f * 0.0004, 0.25))
        for df, wk in pair:
            y += db2a(lvl) * g * wk * np.sin(2 * np.pi * (f + df) * t + rng.uniform(0, 2 * np.pi)) \
                * np.exp(-6.9 * t / T)
    # the chopstick: a short band-passed click, brighter when harder
    k = int(0.004 * SR)
    click = rng.standard_normal(k) * np.exp(-np.arange(k) / (0.0007 * SR))
    click = signal.sosfilt(butter('bandpass', (1500.0 + 2500 * hardness, 9000.0), 2), click)
    y[:k] += click * (0.05 + 0.6 * hardness)
    att = int(max(tc, 0.0004) * SR)
    y[:att] *= np.sin(0.5 * np.pi * np.arange(att) / att)
    return y


def bowl(note, vel=60, hardness=0.5, *, water=0.0, water_shape='bend', water_time=0.6,
         water_delay=0.05, dur=None, damp_s=0.08, model='rec', source='bowl', pan=0.0,
         width=0.25, seed=0, rr=None) -> np.ndarray:
    """One porcelain-bowl stroke at `note` (MIDI number or name, any pitch; the pentatonic set
    is BOWL_SCALE). Returns stereo float32, 48 kHz, first and last sample 0.
    Level at vel 127: 0.4 s RMS about -28 dBFS, peak -13..-4 dBFS (harder = taller contact);
    velocity follows -30*(1-vel/127)**1.3 dB."""
    rng = np.random.default_rng(seed)
    m = _midi(note)
    ft = float(midi_to_hz(m))
    hardness = float(np.clip(hardness, 0, 1))
    soft = hardness - 0.35 * (1 - float(np.clip(vel, 1, 127)) / 127)    # softer contact when quiet
    if model == 'modal':
        y = _modal_stroke(ft, soft, rng, length_s=min(8.0, 3.7 * _MODAL_F0 / ft * 1.3 + 0.3))
        ring = 20 * np.log10(np.sqrt(np.mean(y[int(0.05 * SR):int(0.3 * SR)] ** 2)) + 1e-12)
        peak = 20 * np.log10(np.abs(y).max() + 1e-12)
        y = y * db2a(-15.5 - 0.5 * (ring + peak))            # same 0.4 s RMS as the recordings
    elif model == 'rec':
        if source == 'mixed':
            cands = [(s, h) for s in _SOURCES for h in _hits(s)]
            best = min(abs(12 * np.log2(ft / h[2])) for _, h in cands)
            cands = [c for c in cands if abs(12 * np.log2(ft / c[1][2])) <= best + 0.01]
        else:
            cands = [(source, h) for h in _hits(source)]
        # hardness chooses among the recorded taps by their measured brightness
        br = np.array([h[3] for _, h in cands])
        rank = (br - br.min()) / (np.ptp(br) + 1e-9) if len(br) > 1 else np.full(1, 0.5)
        cost = np.abs(rank - hardness) + rng.uniform(0, 0.35, len(cands))
        if rr is not None:
            cost = cost + 0.5 * np.array([rr.get((s, h[2], h[3]), 0) for s, h in cands])
        s, (x, sr, f0, _) = cands[int(np.argmin(cost))]
        if rr is not None:
            rr[(s, f0, _)] = rr.get((s, f0, _), 0) + 1
        x = _tilt(x.astype(np.float64), 1.2 * (soft - 0.5), sr)
        y = resample_by(x, ft / f0 * sr / SR).astype(np.float64)
    else:
        raise ValueError("model: 'rec' | 'modal'")
    if water:
        y = _warp(y, _water_curve(len(y), float(water), water_shape, water_time, water_delay))
    if dur is not None:
        k = int((dur + damp_s) * SR)
        y = y[:k].copy() if len(y) > k else y.copy()
        d = int(damp_s * SR)
        a = max(0, len(y) - d)
        y[a:] *= np.exp(-np.arange(len(y) - a) / (d / 5.0))
    # end the ring where it sinks 50 dB under its (30 ms RMS) peak - below that the recordings'
    # noise floor (~55 dB under the peak) would take over - with a fade over the last 0.25 s
    from scipy.ndimage import uniform_filter1d
    env = np.sqrt(np.maximum(uniform_filter1d(y * y, int(0.03 * SR)), 0) + 1e-20)
    i0 = int(np.argmax(env))
    low = np.flatnonzero(env[i0:] < env[i0] * db2a(-50))
    if len(low):
        end = min(len(y), i0 + int(low[0]))
        y = y[:end].copy()
        f = min(int(0.25 * SR), end // 3)
        y[end - f:] *= np.cos(0.5 * np.pi * np.arange(f) / f) ** 2
    y = y * db2a(_LEVEL_DB + _vel_gain_db(vel))
    y = fade_edges(y.astype(np.float32), 0.0, 0.02)
    y[0] = 0.0
    # a close mono source; width adds two dull early reflections (table, near wall) that differ
    # between left and right
    gl, gr = pan_gains(pan)
    yl, yr = y.copy(), y.copy()
    if width > 0:
        dull = signal.sosfilt(butter('lowpass', 5000.0, 2), y).astype(np.float32)
        for ch, d_ms in ((yl, 5.3), (yr, 7.9)):
            d = int(d_ms / 1000 * SR)
            ch[d:] += 0.3 * width * dull[:-d]
    out = np.stack([yl * gl, yr * gr], axis=1).astype(np.float32)
    return fade_edges(out, 0.0, 0.01)


def render_bowls(bowls, *, duration=None, pan_spread=0.5, width=0.25, model='rec', source='bowl',
                 water=0.0, water_shape='bend', dur=None, gain_db=0.0, seed=0) -> np.ndarray:
    """score.BOWLS [(start_s, midi, vel, hardness)] -> one stereo track. Bowls are panned by
    pitch across +-pan_spread (low left, high right, like a row of bowls on a table); the
    recorded taps rotate (round-robin) so repeated notes never reuse the same stroke twice
    in a row. An optional 5th tuple element dict may override water / dur / pan per stroke."""
    evs = sorted(bowls, key=lambda b: b[0])
    if not evs:
        return np.zeros((int((duration or 0.1) * SR), 2), np.float32)
    pitches = [_midi(b[1]) for b in evs]
    lo, hi = min(pitches + [74]), max(pitches + [95])
    rr = {}
    parts = []
    for i, b in enumerate(evs):
        st, note, vel, hard = b[:4]
        kw = b[4] if len(b) > 4 and isinstance(b[4], dict) else {}
        m = _midi(note)
        p = kw.get('pan', -pan_spread + 2 * pan_spread * (m - lo) / max(1.0, hi - lo))
        x = bowl(m, vel, hard, water=kw.get('water', water), water_shape=kw.get('water_shape', water_shape),
                 dur=kw.get('dur', dur), model=kw.get('model', model), source=kw.get('source', source),
                 pan=p, width=width, seed=seed * 1000 + i, rr=rr)
        parts.append((int(round(st * SR)), x))
    end = max(s + len(x) for s, x in parts)
    n = int(duration * SR) if duration else end
    out = np.zeros((max(n, 1), 2), np.float32)
    for s, x in parts:
        if s >= n:
            continue
        k = min(len(x), n - s)
        out[s:s + k] += x[:k]
    out *= db2a(gain_db)
    return fade_edges(out, 0.0, 0.01) if duration else out
