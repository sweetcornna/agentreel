"""Mixing toolkit (48 kHz float32 stereo arrays, time in seconds).

place / place_end / reverb / eq / lowpass / highpass / compress / limit / fade /
normalize_lufs / master / write, plus small helpers (new_track, gain, pan, width, delay, mono).

Reverb uses impulse responses synthesised here (decorrelated L/R noise with
per-band exponential decay, early reflections, build-up), convolved with
scipy.signal.oaconvolve -- no third-party reverb.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

from ._dsp import SR, a2db, biquad, butter, cos_ramp, db2a, pan_gains, sosfilt, stereo


# ----------------------------------------------------------------------------- basics
def new_track(duration_s: float) -> np.ndarray:
    return np.zeros((int(round(duration_s * SR)), 2), dtype=np.float32)


def gain(x, db):
    return (stereo(x) * np.float32(db2a(db))).astype(np.float32)


def mono(x):
    x = stereo(x)
    return x.mean(axis=1).astype(np.float32)


def pan(x, p: float):
    """Mono input: constant-power pan. Stereo input: balance (constant-power)."""
    x = np.asarray(x, dtype=np.float32)
    gl, gr = pan_gains(p)
    if x.ndim == 1:
        return np.stack([x * gl, x * gr], axis=1).astype(np.float32)
    x = stereo(x)
    # balance: centre (p=0) keeps unity on both sides
    k = np.sqrt(2.0)
    return np.stack([x[:, 0] * min(1.0, gl * k), x[:, 1] * min(1.0, gr * k)], axis=1).astype(np.float32)


def width(x, w: float):
    """Mid/side stereo width: 0 = mono, 1 = unchanged, >1 = wider."""
    x = stereo(x)
    m = (x[:, 0] + x[:, 1]) * 0.5
    s = (x[:, 0] - x[:, 1]) * 0.5 * w
    return np.stack([m + s, m - s], axis=1).astype(np.float32)


def delay(x, seconds: float):
    n = int(round(seconds * SR))
    x = stereo(x)
    return np.concatenate([np.zeros((n, 2), np.float32), x]) if n > 0 else x


def place(track, x, start_s: float, gain_db: float = 0.0, pan: float = 0.0):
    """Add clip x into track at start_s (seconds). Returns the track (extended if needed).

    Use as ``track = place(track, clip, 12.5, gain_db=-6, pan=-0.3)``.
    pan: -1..1 (constant-power for mono clips, balance for stereo clips).
    """
    track = stereo(track) if track is not None else np.zeros((0, 2), np.float32)
    x = globals()['pan'](x, pan) if pan else stereo(x)
    x = x * np.float32(db2a(gain_db))
    s = int(round(start_s * SR))
    if s < 0:
        x = x[-s:]
        s = 0
    end = s + len(x)
    if end > len(track):
        track = np.concatenate([track, np.zeros((end - len(track), 2), np.float32)])
    elif not track.flags.writeable:
        track = track.copy()
    track[s:end] += x
    return track


def place_end(track, x, end_s: float, gain_db: float = 0.0, pan: float = 0.0):
    """Like place(), but the clip's LAST sample lands at end_s (for reverse_swell and other
    clips that must arrive exactly on a downbeat)."""
    n = len(stereo(x))
    return place(track, x, end_s - n / SR, gain_db=gain_db, pan=pan)


def fade(x, in_s: float = 0.0, out_s: float = 0.0, shape: str = 'cos'):
    """Fade in/out. shape: 'cos' (raised cosine), 'lin', 'exp' (fast-slow, natural decay)."""
    x = np.array(stereo(x), copy=True)
    n = len(x)

    def curve(m):
        u = np.arange(m) / max(m - 1, 1)          # exact 0 at the outer end
        if shape == 'lin':
            return u
        if shape == 'exp':
            return (np.exp(4 * u) - 1) / (np.exp(4) - 1)
        return 0.5 - 0.5 * np.cos(np.pi * u)
    a = min(n, int(round(in_s * SR)))
    b = min(n, int(round(out_s * SR)))
    if a:
        x[:a] *= curve(a)[:, None]
    if b:
        x[n - b:] *= curve(b)[::-1, None]
    return x.astype(np.float32)


# ----------------------------------------------------------------------------- filters
def lowpass(x, fc: float, order: int = 2):
    return sosfilt(butter('lowpass', fc, order), stereo(x))


def highpass(x, fc: float, order: int = 2):
    return sosfilt(butter('highpass', fc, order), stereo(x))


def eq(x, *bands, low_shelf=None, high_shelf=None, peak=None):
    """Parametric EQ (RBJ biquads).

    eq(x, ('peak', 3200, 1.0, -3), ('highshelf', 8000, 0.7, 2))
    eq(x, low_shelf=(120, +2), high_shelf=(6000, -3), peak=(3000, 1.2, -2))
    band tuple: (kind, freq_hz, q, gain_db); kinds: peak lowshelf highshelf lowpass
    highpass bandpass notch.
    """
    y = stereo(x)
    spec = list(bands)
    if low_shelf:
        spec.append(('lowshelf', low_shelf[0], 0.7, low_shelf[1]))
    if high_shelf:
        spec.append(('highshelf', high_shelf[0], 0.7, high_shelf[1]))
    if peak:
        spec.append(('peak',) + tuple(peak))
    for b in spec:
        kind, f, q = b[0], b[1], b[2] if len(b) > 2 else 0.7071
        g = b[3] if len(b) > 3 else 0.0
        y = sosfilt(biquad(kind, f, q, g), y)
    return y


# ----------------------------------------------------------------------------- dynamics
def _release_hold_db(level_db, release_db_per_s):
    """y[n] = max(x[n], y[n-1] - r): instant rise, linear-in-dB fall (vectorised)."""
    r = release_db_per_s / SR
    k = np.arange(len(level_db)) * r
    return np.maximum.accumulate(level_db + k) - k


def compress(x, threshold_db=-18.0, ratio=3.0, attack_ms=15.0, release_ms=180.0,
             knee_db=6.0, makeup_db=0.0, sidechain=None, detector='peak', rms_ms=30.0,
             return_gr=False):
    """Feed-forward, stereo-linked, soft-knee compressor (offline, vectorised).

    release_ms is the time for 20 dB of gain recovery (linear-in-dB release).
    sidechain: optional signal used for detection (e.g. ducking the ambience by the piano).
    detector : 'peak' (sample peak of L/R, default) or 'rms' (stereo power averaged over
               rms_ms, the smoother 'bus glue' behaviour).
    return_gr : also return the gain-reduction curve in dB (positive = reduction).
    """
    x = stereo(x)
    det = stereo(sidechain) if sidechain is not None else x
    if len(det) < len(x):
        det = np.concatenate([det, np.zeros((len(x) - len(det), 2), np.float32)])
    det = det[:len(x)].astype(np.float64)
    if detector == 'rms':
        pw = ndimage.uniform_filter1d((det ** 2).mean(axis=1), max(1, int(rms_ms / 1000 * SR)),
                                      mode='nearest')
        lvl = 10 * np.log10(np.maximum(pw, 1e-18)) + 3.01      # sine RMS -> peak scale
    elif detector == 'peak':
        lvl = a2db(np.abs(det).max(axis=1), 1e-9)
    else:
        raise ValueError("detector must be 'peak' or 'rms'")
    # static curve with soft knee -> gain reduction (positive dB)
    over = lvl - threshold_db
    gr = np.where(over <= -knee_db / 2, 0.0,
                  np.where(over >= knee_db / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee_db / 2) ** 2 / (2 * max(knee_db, 1e-6))))
    gr = _release_hold_db(gr, 20.0 / (release_ms / 1000.0))
    a = np.exp(-1.0 / (attack_ms / 1000.0 * SR))
    gr = signal.lfilter([1 - a], [1, -a], gr)
    g = db2a(-gr + makeup_db).astype(np.float32)
    y = (x * g[:, None]).astype(np.float32)
    return (y, gr) if return_gr else y


def _true_peak_env(x, oversample=4):
    up = signal.resample_poly(x.astype(np.float64), oversample, 1, axis=0)
    pk = np.abs(up).max(axis=1)
    n = len(x)
    pk = pk[:n * oversample].reshape(n, oversample).max(axis=1)
    return np.maximum(pk, np.abs(x).max(axis=1))


def limit(x, ceiling_db=-1.0, lookahead_ms=5.0, release_ms=120.0, true_peak=True):
    """Look-ahead brick-wall limiter (offline, zero added latency).

    The gain needed at every (4x oversampled) peak is held over the look-ahead
    window, smoothed with a same-length moving average (so it is already reached
    when the peak arrives) and released at 20 dB per `release_ms`.
    """
    x = stereo(x)
    ceil = db2a(ceiling_db)
    pk = _true_peak_env(x) if true_peak else np.abs(x).max(axis=1)
    need = np.minimum(1.0, ceil / np.maximum(pk, 1e-12))
    L = max(1, int(lookahead_ms / 1000 * SR))
    held = ndimage.minimum_filter1d(need, size=2 * L + 1, mode='nearest')
    gr = -a2db(held)
    gr = _release_hold_db(gr, 20.0 / (release_ms / 1000.0))
    gr = ndimage.uniform_filter1d(gr, size=L, mode='nearest')
    gr = np.maximum(gr, -a2db(held))          # never less reduction than needed
    y = x * db2a(-gr)[:, None].astype(np.float32)
    # final guard (should be a no-op); counts are reported by analysis.stats
    return np.clip(y, -ceil, ceil).astype(np.float32)


def normalize_lufs(x, target=-16.0, true_peak_db=-1.0):
    """Gain to integrated loudness `target` (ITU-R BS.1770 via pyloudnorm); if the
    true peak would exceed `true_peak_db`, the look-ahead limiter catches it."""
    import pyloudnorm as pyln
    x = stereo(x)
    meter = pyln.Meter(SR)
    cur = meter.integrated_loudness(x.astype(np.float64))
    if not np.isfinite(cur):
        return x
    y = x * np.float32(db2a(target - cur))
    if a2db(_true_peak_env(y).max()) > true_peak_db:
        y = limit(y, true_peak_db)
        # limiting lowers loudness slightly; one correction pass
        cur2 = meter.integrated_loudness(y.astype(np.float64))
        y = limit(y * np.float32(db2a(target - cur2)), true_peak_db)
    return y.astype(np.float32)


def _lf_side_ratio_db(x, fc=120.0):
    """Side/mid energy ratio (dB) below fc: how much stereo content the low end carries."""
    x = stereo(x).astype(np.float64)
    m = (x[:, 0] + x[:, 1]) * 0.5
    s_ = (x[:, 0] - x[:, 1]) * 0.5
    sos = butter('lowpass', fc, 4)
    em = (signal.sosfiltfilt(sos, m) ** 2).sum()
    es = (signal.sosfiltfilt(sos, s_) ** 2).sum()
    return float(10 * np.log10((es + 1e-30) / (em + 1e-30)))


def master(x, target_lufs=-16.0, *, true_peak_db=-1.0, glue_ratio=1.5, glue_threshold_db=None,
           glue_percentile=85.0, glue_attack_ms=30.0, glue_release_ms=300.0, glue_knee_db=6.0,
           tilt_db=0.0, tilt_pivot_hz=1000.0, mono_below_hz=120.0, width=1.1,
           subsonic_hz=22.0, report=False):
    """Mastering chain for the final stereo mix (replaces a bare compress + normalize).

    1. subsonic high-pass (zero-phase, 4th order effective) at `subsonic_hz`;
    2. low end to mono: the side signal is high-passed at `mono_below_hz` (4th-order
       Butterworth run forwards and backwards: zero phase, -48 dB/oct);
    3. stereo width: side x `width` above that frequency (1 = unchanged);
    4. tilt EQ around `tilt_pivot_hz`: +tilt/2 dB high shelf, -tilt/2 dB low shelf
       (positive = brighter);
    5. bus 'glue' compression: stereo-linked RMS detector, `glue_ratio`:1, soft knee,
       slow attack / release. The threshold defaults to the `glue_percentile`-th
       percentile of the detector level after pre-normalising to the target, i.e. only
       the loudest ~15 % of the programme is compressed, by a dB or two;
    6. loudness to `target_lufs` (BS.1770) with the true-peak limiter at `true_peak_db`.

    report=True returns (y, info) with in/out loudness, true peak, gain-reduction stats
    and the low-frequency side/mid ratio before and after.
    """
    import pyloudnorm as pyln
    x = stereo(x).astype(np.float64)
    meter = pyln.Meter(SR)
    info = {'in_lufs': round(float(meter.integrated_loudness(x)), 2),
            'in_true_peak_dbtp': round(float(a2db(_true_peak_env(x.astype(np.float32)).max())), 2),
            'in_lf_side_over_mid_db': round(_lf_side_ratio_db(x, mono_below_hz), 1)}
    y = x
    if subsonic_hz:
        y = signal.sosfiltfilt(butter('highpass', subsonic_hz, 2), y, axis=0)
    m = (y[:, 0] + y[:, 1]) * 0.5
    s_ = (y[:, 0] - y[:, 1]) * 0.5
    if mono_below_hz:
        s_ = signal.sosfiltfilt(butter('highpass', mono_below_hz, 4), s_)
    s_ = s_ * width
    y = np.stack([m + s_, m - s_], axis=1)
    if tilt_db:
        y = signal.sosfilt(biquad('lowshelf', tilt_pivot_hz, 0.5, -tilt_db / 2), y, axis=0)
        y = signal.sosfilt(biquad('highshelf', tilt_pivot_hz, 0.5, tilt_db / 2), y, axis=0)
    cur = meter.integrated_loudness(y)
    if np.isfinite(cur):
        y = y * db2a(target_lufs - cur)
    gr_stats = {'glue_threshold_db': None, 'gr_max_db': 0.0, 'gr_p95_db': 0.0, 'gr_mean_db': 0.0}
    if glue_ratio and glue_ratio > 1:
        thr = glue_threshold_db
        if thr is None:
            pw = ndimage.uniform_filter1d((y ** 2).mean(axis=1), int(0.03 * SR), mode='nearest')
            lvl = 10 * np.log10(np.maximum(pw, 1e-18)) + 3.01
            active = lvl > lvl.max() - 40
            thr = float(np.percentile(lvl[active], glue_percentile)) if active.any() else -20.0
        y, gr = compress(y.astype(np.float32), threshold_db=thr, ratio=glue_ratio,
                         attack_ms=glue_attack_ms, release_ms=glue_release_ms, knee_db=glue_knee_db,
                         detector='rms', return_gr=True)
        y = y.astype(np.float64)
        gr_stats = {'glue_threshold_db': round(float(thr), 1), 'gr_max_db': round(float(gr.max()), 2),
                    'gr_p95_db': round(float(np.percentile(gr, 95)), 2),
                    'gr_mean_db': round(float(gr.mean()), 2)}
    y = normalize_lufs(y.astype(np.float32), target_lufs, true_peak_db)
    info.update(gr_stats)
    info.update({'out_lufs': round(float(meter.integrated_loudness(y.astype(np.float64))), 2),
                 'out_true_peak_dbtp': round(float(a2db(_true_peak_env(y).max())), 2),
                 'out_lf_side_over_mid_db': round(_lf_side_ratio_db(y, mono_below_hz), 1),
                 'clipped_samples': int((np.abs(y) >= 0.999).sum())})
    return (y, info) if report else y


def write(path, x, sr=SR, subtype='PCM_24'):
    """Write a 24-bit WAV (48 kHz). Values beyond +-1 are clipped (and counted)."""
    import soundfile as sf
    x = stereo(x)
    n_clip = int((np.abs(x) > 1.0).sum())
    if n_clip:
        print(f'write: {n_clip} samples clipped in {path}')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype=subtype)
    return n_clip


# ----------------------------------------------------------------------------- reverb
REVERBS = {
    # rt60 per band (Hz band edges below), pre-delay, early reflections, build-up
    'room':   dict(rt=[0.9, 0.8, 0.7, 0.6, 0.45, 0.3], pre=0.004, er=(0.003, 0.035, 18, -6.0),
                   build=0.004, length=1.6, width=0.8),
    'hall':   dict(rt=[3.4, 3.1, 2.9, 2.5, 1.9, 1.2], pre=0.022, er=(0.012, 0.09, 14, -8.0),
                   build=0.03, length=5.5, width=1.0),
    'valley': dict(rt=[7.0, 6.5, 5.8, 4.6, 3.0, 1.6], pre=0.07, er=None,
                   echoes=[(0.19, -13, -0.6), (0.31, -16, 0.7), (0.47, -19, -0.2), (0.66, -22, 0.5),
                           (0.93, -25, -0.8), (1.27, -28, 0.3)],
                   build=0.35, length=11.0, width=1.0),
    # smooth, dense, no pre-delay / early reflections / build-up: the source for reverse
    # swells (reversed, its energy rises monotonically into the end) and shimmer
    'bloom':  dict(rt=[5.6, 5.3, 4.9, 4.2, 3.2, 2.0], pre=0.0, er=None,
                   build=0.002, length=8.0, width=1.0),
}
_BAND_EDGES = [250, 700, 2000, 5000, 10000]   # -> 6 bands


@lru_cache(maxsize=16)
def reverb_ir(kind='hall', seed=0):
    """Synthesised true-stereo IR: returns (2 inputs, n, 2 outputs) float32."""
    if kind not in REVERBS:
        raise ValueError(f'kind must be one of {list(REVERBS)}')
    return _make_ir(REVERBS[kind], 1000 + seed + 17 * list(REVERBS).index(kind))


# decay-time profile of 'bloom' relative to its low band, reused by custom_ir()
_BLOOM_SHAPE = tuple(r / REVERBS['bloom']['rt'][0] for r in REVERBS['bloom']['rt'])


@lru_cache(maxsize=16)
def custom_ir(rt60: float, seed: int = 0, length_s=None):
    """'bloom'-type IR (no pre-delay / early reflections, 2 ms build-up) with the
    low-band RT60 set to rt60 s (higher bands shorter in the same proportions)."""
    rt60 = float(rt60)
    p = dict(REVERBS['bloom'], rt=[rt60 * k for k in _BLOOM_SHAPE],
             length=float(length_s or min(12.0, 1.3 * rt60 + 0.2)))
    return _make_ir(p, 5000 + seed + int(round(rt60 * 100)))


def _make_ir(p, rng_seed):
    rng = np.random.default_rng(rng_seed)
    n = int(p['length'] * SR)
    t = np.arange(n) / SR
    edges = [0] + _BAND_EDGES + [SR / 2]
    irs = np.zeros((2, n, 2), dtype=np.float64)
    for ci in range(2):          # input channel
        for co in range(2):      # output channel
            w = rng.standard_normal(n)
            tail = np.zeros(n)
            for bi in range(len(edges) - 1):
                lo, hi = edges[bi], edges[bi + 1]
                if lo == 0:
                    sos = signal.butter(4, hi / (SR / 2), 'lowpass', output='sos')
                elif hi >= SR / 2:
                    sos = signal.butter(4, lo / (SR / 2), 'highpass', output='sos')
                else:
                    sos = signal.butter(2, [lo / (SR / 2), hi / (SR / 2)], 'bandpass', output='sos')
                band = signal.sosfiltfilt(sos, w)
                tail += band * np.exp(-6.9078 * t / p['rt'][bi])
            tail *= 1 - np.exp(-t / p['build'])
            # same-side output a little stronger than cross-feed
            tail *= 1.0 if ci == co else 0.8
            irs[ci, :, co] = tail
    irs /= np.sqrt((irs ** 2).sum() / 2)          # unit energy per input channel (average)
    pre = int(p['pre'] * SR)
    irs = np.concatenate([np.zeros((2, pre, 2)), irs], axis=1)[:, :n]
    tot = np.sqrt((irs ** 2).sum() / 2)
    if p.get('er'):
        t0, t1, count, lvl = p['er']
        for ci in range(2):
            for _ in range(count):
                tt = rng.uniform(t0, t1)
                k = int((p['pre'] + tt) * SR)
                g = db2a(lvl) * tot * rng.uniform(0.4, 1.0) * np.exp(-tt / (t1 * 1.5))
                side = rng.uniform(-1, 1)
                irs[ci, k, 0] += g * np.sqrt(0.5 * (1 - side)) * rng.choice([-1, 1])
                irs[ci, k, 1] += g * np.sqrt(0.5 * (1 + side)) * rng.choice([-1, 1])
    for tt, lvl, side in p.get('echoes', []):   # sparse distant-wall echoes (valley)
        k = int((p['pre'] + tt) * SR)
        blip = np.zeros(int(0.03 * SR))
        blip[0] = 1.0
        blip = signal.sosfilt(signal.butter(2, 2500 / (SR / 2), output='sos'), blip)
        blip /= np.sqrt((blip ** 2).sum())
        for ci in range(2):
            gl, gr = pan_gains(np.clip(side + (0.15 if ci else -0.15), -1, 1))
            m = min(len(blip), n - k)
            irs[ci, k:k + m, 0] += db2a(lvl) * tot * gl * blip[:m]
            irs[ci, k:k + m, 1] += db2a(lvl) * tot * gr * blip[:m]
    f = int(0.05 * SR)
    irs[:, -f:, :] *= cos_ramp(f, rising=False)[None, :, None]
    return irs.astype(np.float32)


def reverb(x, kind='hall', mix=0.25, seed=0, predelay_s=0.0, tail=True, lowcut_hz=150.0,
           highcut_hz=None):
    """Convolution reverb with a synthesised IR.

    kind : 'room' (~0.7 s), 'hall' (~3 s), 'valley' (5-7 s open valley: long pre-delay,
           sparse distant echoes, slow build-up), 'bloom' (~5 s smooth tail with no
           pre-delay, used by fx.reverse_swell / fx.shimmer), or an IR array
           (2, n, 2) such as custom_ir(rt60).
    mix  : 0 = dry, 1 = fully wet (equal-power crossfade).
    tail : keep the reverb tail beyond the input length.
    lowcut_hz / highcut_hz : filter the wet return (keeps low end clean).
    """
    x = stereo(x)
    ir = reverb_ir(kind, seed) if isinstance(kind, str) else np.asarray(kind, dtype=np.float32)
    if predelay_s:
        ir = np.concatenate([np.zeros((2, int(predelay_s * SR), 2), np.float32), ir], axis=1)
    n_out = len(x) + (ir.shape[1] if tail else 0)
    wet = np.zeros((len(x) + ir.shape[1] - 1, 2), dtype=np.float64)
    for ci in range(2):
        for co in range(2):
            wet[:, co] += signal.oaconvolve(x[:, ci].astype(np.float64), ir[ci, :, co].astype(np.float64))
    if lowcut_hz:
        wet = signal.sosfilt(butter('highpass', lowcut_hz, 2), wet, axis=0)
    if highcut_hz:
        wet = signal.sosfilt(butter('lowpass', highcut_hz, 2), wet, axis=0)
    dry = np.zeros_like(wet)
    dry[:len(x)] = x
    th = np.clip(mix, 0, 1) * np.pi / 2
    y = np.cos(th) * dry + np.sin(th) * wet
    return y[:n_out].astype(np.float32)
