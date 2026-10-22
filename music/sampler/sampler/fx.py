"""Sound-design effects (deterministic DSP only, no generative models).

    pitch_shift(x, semitones)                     phase-vocoder transposition, same length
    reverse_swell(x, length_s, ...)               reversed 100 %-wet reverb rising into a downbeat
    shimmer(x, mix=0.35, feedback=0.55, ...)      reverb with an octave-up shifter in its loop
    granular(x, duration, grain_ms, density, pitch, spread, seed, chord=..., tonal=...)
    tape(x, drive=3.0, wow=0.05, flutter=0.02, hiss=None, seed=0)
    impact(kind='soft_low', key='D', ...)         restrained low hit with a long tail

All functions take / return float32 stereo (n, 2) at 48 kHz (mono input is accepted).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from ._dsp import (SR, biquad, butter, cos_ramp, db2a, fade_edges, midi_to_hz, pan_gains,
                   resample_by, smooth_noise, sosfilt, stereo)


def _peak_to(x, peak_db):
    if peak_db is None:
        return x.astype(np.float32)
    pk = float(np.abs(x).max())
    return (x * (db2a(peak_db) / pk)).astype(np.float32) if pk > 0 else x.astype(np.float32)


def _env_db(x, win_s=0.02):
    p = ndimage.uniform_filter1d((stereo(x).astype(np.float64) ** 2).mean(axis=1),
                                 max(1, int(win_s * SR)), mode='nearest')
    return 10 * np.log10(np.maximum(p, 1e-20))


def _trim_tail(y, floor_db=-75.0, fade_s=0.3):
    """Cut a decaying tail where its 50 ms envelope falls below floor_db re. its peak."""
    e = _env_db(y, 0.05)
    above = np.flatnonzero(e > e.max() + floor_db)
    if not len(above):
        return y[:1]
    end = min(len(y), int(above[-1]) + int(fade_s * SR))
    y = np.array(y[:end], copy=True)
    f = min(end, int(fade_s * SR))
    y[end - f:] *= cos_ramp(f, rising=False)[:, None]
    return y


# ----------------------------------------------------------------------------- pitch shift
def _pv_stretch(x, r, n_fft=2048, ha=256):
    """Phase-vocoder time stretch of a 1-D signal by factor r (output ~ r x longer).

    Analysis hop ha, synthesis hop hs = round(ha r); the instantaneous frequency of
    every bin (from the frame-to-frame phase advance) drives the synthesis phase, so
    sinusoids stay continuous across frames. Returns (y, hs)."""
    hs = int(round(ha * r))
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    pad = n_fft
    xp = np.pad(np.asarray(x, dtype=np.float64), (pad, pad + n_fft))
    n_frames = 1 + (len(xp) - n_fft) // ha
    idx = np.arange(n_fft)[None, :] + ha * np.arange(n_frames)[:, None]
    X = np.fft.rfft(xp[idx] * win, axis=1)
    mag = np.abs(X)
    ph = np.angle(X)
    nb = n_fft // 2 + 1
    omega = 2 * np.pi * np.arange(nb) * ha / n_fft
    dphi = np.diff(ph, axis=0) - omega
    dphi = np.mod(dphi + np.pi, 2 * np.pi) - np.pi
    inst = (omega + dphi) / ha                                    # rad / sample
    # identity phase locking (Laroche & Dolson 1999): only spectral peaks integrate their
    # instantaneous frequency; every other bin keeps its analysis phase offset from the
    # peak whose region it belongs to, so the partials stay vertically coherent.
    phs = np.empty_like(ph)
    phs[0] = ph[0]
    k = np.arange(nb)
    for t in range(1, n_frames):
        m = mag[t]
        if m.max() <= 0:
            phs[t] = phs[t - 1] + omega * hs / ha
            continue
        pk = np.flatnonzero((m[1:-1] > m[:-2]) & (m[1:-1] >= m[2:]) & (m[1:-1] > m.max() * 1e-5)) + 1
        if not len(pk):
            phs[t] = phs[t - 1] + inst[t - 1] * hs
            continue
        bounds = (pk[:-1] + pk[1:]) // 2
        owner = pk[np.searchsorted(bounds, k, side='right')]
        peak_ph = phs[t - 1, pk] + inst[t - 1, pk] * hs
        phs[t, pk] = peak_ph
        phs[t] = phs[t, owner] + (ph[t] - ph[t, owner])
    frames = np.fft.irfft(mag * np.exp(1j * phs), n_fft, axis=1) * win
    out_len = hs * (n_frames - 1) + n_fft
    y = np.zeros(out_len)
    wsum = np.zeros(out_len)
    w2 = win * win
    for t in range(n_frames):
        a = t * hs
        y[a:a + n_fft] += frames[t]
        wsum[a:a + n_fft] += w2
    y /= np.maximum(wsum, 1e-3 * wsum.max())
    return y, hs


def pitch_shift(x, semitones: float, n_fft: int = 2048, hop: int = 256) -> np.ndarray:
    """Transpose by `semitones` keeping the duration (phase vocoder + resampling).

    Good on sustained / dense material (reverb tails, pads); transients get smeared by
    about n_fft samples (43 ms), which is why it is meant for the inside of a reverb.
    Output has the input's length and timing (group delay compensated)."""
    x = stereo(x)
    if abs(semitones) < 1e-9:
        return x.copy()
    r = 2.0 ** (semitones / 12.0)
    out = np.zeros_like(x, dtype=np.float32)
    for ch in range(2):
        y, hs = _pv_stretch(x[:, ch], r, n_fft, hop)
        r_eff = hs / hop
        z = resample_by(y.astype(np.float32), r_eff)
        # input sample j sits at stretched index (pad + j) * r_eff minus the frame-centre
        # offset n_fft/2 (r_eff - 1); after resampling by r_eff that is pad + j - D
        d = n_fft - int(round(n_fft / 2 * (1 - 1 / r_eff)))
        seg = z[max(d, 0):max(d, 0) + len(x)]
        out[:len(seg), ch] = seg
    return out


# ----------------------------------------------------------------------------- reverse swell
def reverse_swell(x, length_s: float, *, rt60: float | None = None, source_s: float = 0.3,
                  source_decay_s: float = 0.015, gap_s: float = 0.0, fade_in: float = 0.35,
                  end_fade_ms: float = 5.0, lowcut_hz: float = 150.0,
                  highcut_hz: float | None = 9000.0, peak_db: float | None = -6.0,
                  seed: int = 0) -> np.ndarray:
    """Reversed 100 %-wet reverb of `x` that swells into its own end.

    The returned clip is exactly round(length_s * SR) samples long and its END is the
    landing point: ``mix.place_end(track, clip, t_downbeat)`` puts the downbeat right
    after its last sample. First and last samples are 0 (cosine fade-in over the first
    `fade_in` fraction, `end_fade_ms` fade-out at the very end).

    How: the first `source_s` of x, weighted by exp(-t / source_decay_s) so that the
    excitation is impulse-like (the attack of the chord), goes 100 % wet through a smooth
    IR without pre-delay or build-up (mix.custom_ir, low-band RT60 = `rt60`, default
    1.6 x length_s: the swell rises by ~37 dB across the clip whatever its length). The
    wet signal is reversed and its last length_s seconds are kept.
    gap_s : silence left before the landing point (0 = arrives exactly on it).
    """
    from .mix import custom_ir, reverb
    x = stereo(x).astype(np.float64)
    m = int(round(source_s * SR)) if source_s else len(x)
    x = x[:m]
    if source_decay_s:
        x = x * np.exp(-np.arange(len(x)) / (source_decay_s * SR))[:, None]
    f = min(len(x), int(0.01 * SR))
    x[len(x) - f:] *= cos_ramp(f, rising=False)[:, None]
    rt = float(np.clip(rt60 if rt60 else 1.6 * length_s, 0.5, 9.0))
    wet = reverb(x.astype(np.float32), custom_ir(round(rt, 2), seed), mix=1.0,
                 lowcut_hz=lowcut_hz, highcut_hz=highcut_hz)
    a = np.abs(wet).max(axis=1)
    on = int(np.argmax(a > a.max() * 1e-4))                        # drop leading silence
    # start at the wet signal's energy peak (5 ms RMS, within the first 150 ms): reversed,
    # the swell then reaches its maximum at the landing point instead of slightly before
    e5 = _env_db(wet[on:on + int(0.15 * SR)], 0.005)
    on += int(np.argmax(e5))
    rev = wet[on:][::-1].astype(np.float64)
    n = int(round(length_s * SR))
    g = int(round(gap_s * SR))
    body_n = max(1, n - g)
    body = rev[-body_n:] if len(rev) >= body_n else np.concatenate(
        [np.zeros((body_n - len(rev), 2)), rev])
    body = np.array(body, copy=True)
    fi = int(np.clip(fade_in, 0.0, 1.0) * body_n)
    if fi > 1:
        body[:fi] *= cos_ramp(fi)[:, None]
    fo = min(body_n, max(2, int(end_fade_ms / 1000 * SR)))
    body[-fo:] *= cos_ramp(fo, rising=False)[:, None]
    out = np.concatenate([body, np.zeros((g, 2))]) if g else body
    out[0] = 0.0
    out[-1] = 0.0
    return _peak_to(out, peak_db)


# ----------------------------------------------------------------------------- shimmer
def shimmer(x, mix: float = 0.35, feedback: float = 0.55, interval: float = 12.0, *,
            reverb_kind: str = 'bloom', damping_hz: float = 2200.0, lowcut_hz: float = 250.0,
            generations: int = 4, predelay_s: float = 0.02, seed: int = 0) -> np.ndarray:
    """Shimmer reverb: a reverb whose feedback path contains a +`interval` semitone
    pitch shifter (default octave) and a damping low-pass.

    Rendered offline as the Neumann series of that loop (each 'generation' is one more
    pass through shifter + damping + reverb, scaled by `feedback`), which is exactly the
    loop's output for linear processing; stops after `generations` passes or when a
    pass is 60 dB down. `mix` is the equal-power dry/wet balance. Returns input length
    + the reverb tail (trimmed at -75 dB).
    damping_hz : low-pass on the wet signal and in the loop. The default 2.2 kHz keeps
    2.5-4 kHz within ~2 dB of a plain reverb (measured on choir and high piano); raise it
    for a brighter, glassier shimmer. The shifter is pitch_shift(): octave exact, other
    intervals as the ratio round(256 r) / 256 (a fifth comes out pure, +2 cents).
    """
    from .mix import reverb
    x = stereo(x)
    fb = float(np.clip(feedback, 0.0, 0.95))

    def R(s, i):
        return reverb(s, reverb_kind, mix=1.0, seed=seed + 7 * i, predelay_s=predelay_s,
                      lowcut_hz=lowcut_hz, highcut_hz=damping_hz).astype(np.float64)

    cur = _trim_tail(R(x, 0))
    wet = cur.copy()
    e0 = float((cur ** 2).sum()) + 1e-30
    for i in range(1, int(generations) + 1):
        if fb <= 0:
            break
        s_in = pitch_shift(cur.astype(np.float32), interval).astype(np.float64) * fb
        s_in = signal.sosfilt(butter('lowpass', damping_hz, 2), s_in, axis=0)
        cur = _trim_tail(R(s_in, i))
        if len(cur) > len(wet):
            wet = np.concatenate([wet, np.zeros((len(cur) - len(wet), 2))])
        wet[:len(cur)] += cur
        if float((cur ** 2).sum()) < e0 * 1e-6:
            break
    n = max(len(wet), len(x))
    dry = np.zeros((n, 2))
    dry[:len(x)] = x
    w = np.zeros((n, 2))
    w[:len(wet)] = wet
    th = np.clip(mix, 0, 1) * np.pi / 2
    y = np.cos(th) * dry + np.sin(th) * w
    return fade_edges(y.astype(np.float32), 0.0, 0.05)


# ----------------------------------------------------------------------------- granular
def _interp_read(src, start, ratio, n_out, a=8):
    """Band-limited fractional read: n_out samples from src starting at `start`, advancing
    `ratio` source samples per output sample (Lanczos-windowed sinc, cutoff scaled by
    1/ratio when reading faster so nothing aliases)."""
    s = max(1.0, ratio)
    half = int(np.ceil(a * s))
    pos = start + ratio * np.arange(n_out)
    base = np.floor(pos).astype(np.int64)
    k = np.arange(-half + 1, half + 1)
    idx = base[:, None] + k[None, :]
    d = (pos[:, None] - idx) / s
    w = np.sinc(d) * np.sinc(d / a) * (np.abs(d) < a) / s
    idx = np.clip(idx, 0, len(src) - 1)
    return (src[idx] * w).sum(axis=1)


def _pitch_classes(chord):
    from .sfz import note_number
    out = []
    for c in chord:
        m = note_number(c) if isinstance(c, str) else float(c)
        out.append(float(m) % 12.0)
    return np.array(sorted(set(out)))


def granular(x, duration: float, grain_ms: float = 80.0, density: float = 30.0, pitch=None,
             spread: float = 0.8, seed: int = 0, *, chord=None, tonal: float = 0.0,
             tonal_q: float = 45.0, position=(0.0, 1.0), grain_jitter: float = 0.3,
             pitch_jitter_cents: float = 0.0, quantize_band=(250.0, 5000.0),
             band=(120.0, 14000.0), level_lufs: float | None = -30.0) -> np.ndarray:
    """Grain cloud from a recording (e.g. a brook): 'glitter of water' beds.

    x        : array (mono/stereo) or an ambience kind name ('stream', 'creek', 'wind_synth' ...)
    duration : output length in s.
    grain_ms : grain length (each grain randomly +-grain_jitter of it), Hann window.
    density  : grains per second (jittered regular spacing: an even texture).
    pitch    : None (no transposition), a number of semitones, or a transposition table
               (list of semitones, one picked at random per grain), e.g. [0, 7, 12, 19].
    spread   : stereo spread 0..1: each grain is panned at random in [-spread, spread] and
               its right channel is blended towards an independent grain (read from another
               source position), so spread also lowers the L/R correlation.
    chord    : optional list of MIDI notes / names. Each grain's strongest spectral peak
               (within quantize_band) is measured and the grain is resampled so that the
               peak lands on the nearest chord tone (any octave) -> a pitched shimmer
               that follows the harmony. Without chord the cloud stays unpitched.
    tonal    : 0..1, adds narrow resonators (Q = tonal_q) on the target note and its
               octave to make the pitch clearer (only with chord).
    position : (a, b) fraction of the source the grains are read from.
    level_lufs : loudness of the result (None = unnormalised).
    """
    rng = np.random.default_rng(seed)
    if isinstance(x, str):
        from .nature import ambience
        x = ambience(x, max(30.0, duration * 0.5), seed=seed, level_lufs=-30.0)
    src = stereo(x).astype(np.float64).mean(axis=1)
    n = int(round(duration * SR))
    out = np.zeros((n + int(0.5 * SR), 2))
    pcs = _pitch_classes(chord) if chord is not None else None
    table = ([0.0] if pitch is None else [float(pitch)] if np.isscalar(pitch)
             else [float(p) for p in pitch])
    n_gr = max(1, int(np.ceil(duration * density)))
    lo_f, hi_f = quantize_band
    p0, p1 = position
    nfft = 8192
    for i in range(n_gr):
        t = (i + rng.uniform(-0.45, 0.45)) / density
        L = int(grain_ms / 1000 * SR * (1 + grain_jitter * rng.uniform(-1, 1)))
        L = max(L, 64)
        st = int(round(t * SR)) - L // 2
        if st + L <= 0 or st >= n:
            continue
        semis = float(rng.choice(table)) + pitch_jitter_cents / 100 * rng.standard_normal()
        r = 2.0 ** (semis / 12.0)
        span = int(np.ceil(L * 2.2)) + 64
        a0 = int(p0 * max(1, len(src) - span))
        a1 = int(p1 * max(1, len(src) - span))
        pos = int(rng.integers(a0, max(a0 + 1, a1)))
        target = None
        if pcs is not None:
            seg = src[pos:pos + int(L * r) + 1]
            seg = seg * np.hanning(len(seg))
            S = np.abs(np.fft.rfft(seg, nfft))
            f = np.fft.rfftfreq(nfft, 1 / SR)
            band_ = (f * r >= lo_f) & (f * r <= hi_f)
            if band_.any() and S[band_].max() > 0:
                j = np.flatnonzero(band_)[np.argmax(S[band_])]
                if 0 < j < len(S) - 1:              # parabolic peak interpolation
                    al, be, ga = np.log(S[j - 1] + 1e-12), np.log(S[j] + 1e-12), np.log(S[j + 1] + 1e-12)
                    j = j + 0.5 * (al - ga) / (al - 2 * be + ga + 1e-12)
                fd = j * SR / nfft * r
                md = 69 + 12 * np.log2(fd / 440.0)
                cand = np.concatenate([pcs + 12 * o for o in range(-1, 11)])
                mt = cand[np.argmin(np.abs(cand - md))]
                r = r * 2 ** ((mt - md) / 12.0)
                target = float(midi_to_hz(mt))
        # second read from an unrelated source position for the right channel: spread
        # also decorrelates L/R (spread 0 -> identical, 1 -> independent grains)
        pos2 = int(rng.integers(a0, max(a0 + 1, a1)))
        pair = []
        for pp in (pos, pos2):
            g = _interp_read(src, pp, r, L)
            if target is not None and tonal > 0:
                res = np.zeros(L)
                for h, amp in ((1, 1.0), (2, 0.45)):
                    fh = target * h
                    if fh >= SR * 0.45:
                        continue
                    bw = fh / tonal_q
                    rr = np.exp(-np.pi * bw / SR)
                    b1, b2 = 2 * rr * np.cos(2 * np.pi * fh / SR), -rr * rr
                    res += amp * signal.lfilter([1 - rr], [1, -b1, -b2], g)
                rms_g = np.sqrt((g ** 2).mean()) + 1e-12
                rms_r = np.sqrt((res ** 2).mean()) + 1e-12
                g = (1 - tonal) * g + tonal * res * (rms_g / rms_r)
            pair.append(g * np.hanning(L))
        th = float(np.clip(spread, 0, 1)) * np.pi / 2
        gL, gR = pair[0], np.cos(th) * pair[0] + np.sin(th) * pair[1]
        gl, gr = pan_gains(rng.uniform(-spread, spread))
        a = max(0, st)
        b = min(len(out), st + L)
        out[a:b, 0] += gL[a - st:b - st] * gl
        out[a:b, 1] += gR[a - st:b - st] * gr
    out = out[:n]
    lo, hi = band
    out = signal.sosfilt(butter('highpass', max(lo, 20.0), 2), out, axis=0)
    if hi and hi < SR / 2 * 0.95:
        out = signal.sosfilt(butter('lowpass', hi, 2), out, axis=0)
    y = out.astype(np.float32)
    if level_lufs is not None:
        from .analysis import lufs
        cur = lufs(y)
        if np.isfinite(cur):
            y = (y * db2a(level_lufs - cur)).astype(np.float32)
    return fade_edges(y, 0.01, 0.01)


# ----------------------------------------------------------------------------- tape
def _catmull(x, pos):
    i = np.floor(pos).astype(np.int64)
    f = pos - i
    n = len(x)

    def at(k):
        return x[np.clip(k, 0, n - 1)]
    xm1, x0, x1, x2 = at(i - 1), at(i), at(i + 1), at(i + 2)
    c1 = 0.5 * (x1 - xm1)
    c2 = xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2
    c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1)
    return ((c3 * f + c2) * f + c1) * f + x0


def _speed_curve(n, fs, wow, flutter, rng):
    """Relative speed deviation (fraction) at rate fs: wow ~0.55 Hz + slow random,
    flutter ~7.3 Hz capstan + 4-20 Hz random; each scaled so its peak = the setting."""
    t = np.arange(n) / fs
    eps = np.zeros(n)
    if wow:
        w = 0.8 * np.sin(2 * np.pi * rng.uniform(0.45, 0.65) * t + rng.uniform(0, 2 * np.pi))
        w += 0.2 * smooth_noise(n, 0.4, rng, sr=fs)
        eps += w / (np.percentile(np.abs(w), 99.9) + 1e-12) * wow / 100.0
    if flutter:
        f = 0.6 * np.sin(2 * np.pi * rng.uniform(6.8, 7.8) * t + rng.uniform(0, 2 * np.pi))
        nz = signal.sosfilt(signal.butter(2, [4.0 / (fs / 2), 20.0 / (fs / 2)], 'bandpass', output='sos'),
                            rng.standard_normal(n))
        f += 0.4 * nz / (np.std(nz) + 1e-12) / 2.5
        eps += f / (np.percentile(np.abs(f), 99.9) + 1e-12) * flutter / 100.0
    return eps


def tape(x, drive: float = 3.0, wow: float = 0.05, flutter: float = 0.02, hiss=None, *,
         seed: int = 0, emphasis_db: float = 6.0, bias: float = 0.03, head_bump_db: float = 1.0,
         hf_rolloff_hz: float = 16000.0, oversample: int = 4) -> np.ndarray:
    """Tape character: soft saturation, very light wow / flutter, optional hiss.

    drive   : dB into the saturator; 0 = clean. Small signals keep unity gain, peaks are
              rounded (asymmetric tanh -> mostly 3rd, a little 2nd harmonic). The
              saturator runs 4x oversampled between pre-/de-emphasis shelves (+-emphasis_db
              at 3 kHz), like a record/playback EQ, so highs saturate earlier and stay smooth.
    wow     : peak speed deviation in % around ~0.55 Hz (0.05 = very light; a good deck
              measures ~0.03-0.1 %).
    flutter : peak speed deviation in % around ~7.3 Hz plus 4-20 Hz random.
    hiss    : None (off) or hiss level in dBFS RMS (e.g. -68).
    Deterministic for a given seed; same length as the input.
    """
    rng = np.random.default_rng(seed)
    x = stereo(x).astype(np.float64)
    n = len(x)
    y = x
    if emphasis_db:
        y = signal.sosfilt(biquad('highshelf', 3000.0, 0.7, emphasis_db), y, axis=0)
    os_ = int(max(1, oversample))
    u = signal.resample_poly(y, os_, 1, axis=0, window=('kaiser', 9.0)) if os_ > 1 else y
    k = float(db2a(drive) - 1.0)
    if k > 1e-4:
        tb = np.tanh(k * bias)
        u = (np.tanh(k * (u + bias)) - tb) / (k * (1 - tb * tb))
    if wow or flutter:
        fs = SR * os_
        eps = _speed_curve(len(u), fs, wow, flutter, rng)
        disp = np.cumsum(eps)
        disp -= disp.mean()
        pos = np.clip(np.arange(len(u)) + disp, 0, len(u) - 1)
        u = np.stack([_catmull(u[:, c], pos) for c in range(2)], axis=1)
    y = signal.resample_poly(u, 1, os_, axis=0, window=('kaiser', 9.0))[:n] if os_ > 1 else u
    if emphasis_db:
        y = signal.sosfilt(biquad('highshelf', 3000.0, 0.7, -emphasis_db), y, axis=0)
    y = signal.sosfilt(butter('highpass', 22.0, 2), y, axis=0)          # DC from the bias
    if head_bump_db:
        y = signal.sosfilt(biquad('peak', 70.0, 1.0, head_bump_db), y, axis=0)
    if hf_rolloff_hz:
        y = signal.sosfilt(butter('lowpass', hf_rolloff_hz, 2), y, axis=0)
    if hiss is not None:
        h = rng.standard_normal((n, 2))
        h = signal.sosfilt(butter('bandpass', (150.0, 14000.0), 1), h, axis=0)
        h = signal.sosfilt(biquad('peak', 5000.0, 0.8, 3.0), h, axis=0)
        h *= db2a(hiss) / (np.sqrt((h ** 2).mean()) + 1e-20)
        y = y + fade_edges(h.astype(np.float32), 0.005, 0.005)
    return y.astype(np.float32)


# ----------------------------------------------------------------------------- impact
def _onset(x, rel_db=-30.0):
    a = np.abs(stereo(x)).max(axis=1)
    return int(np.argmax(a > a.max() * db2a(rel_db))) if a.max() > 0 else 0


def _lufs_to(x, target):
    from .analysis import lufs
    cur = lufs(x)
    return (x * db2a(target - cur)).astype(np.float32) if np.isfinite(cur) else x


IMPACT_KINDS = {
    'soft_low': 'piano low octave + fifth (pedal, sympathetic resonance, low-passed) + VCSL gong p '
                'a fourth lower + VCSL low slit drum + short sub, valley tail',
    'felt_low': 'felt piano low cluster (root, fifth, octave) + VCSL low slit drum, hall tail',
    'gong_low': 'VCSL gong mf a minor third lower + sub swell, valley tail',
}


def impact(kind: str = 'soft_low', *, key='D', intensity: float = 0.5, tail_s: float = 5.0,
           seed: int = 0, peak_db: float | None = -6.0) -> np.ndarray:
    """Restrained low 'landing' hit for scene transitions (see IMPACT_KINDS).

    key       : pitch class (e.g. 'D', 'F#') - the root sits at octave 1 (D1 = 36.7 Hz).
    intensity : 0..1, piano velocity / gong level / brightness (0.5 = soft).
    tail_s    : total length after the hit (the reverb tail is faded out by then).
    The hit is the clip's first sample (1 ms fade-in), so place(track, clip, t_hit) lands
    it exactly; peak-normalised to peak_db like sfx().
    """
    from .mix import reverb
    from .nature import sfx
    from .pads import sub
    from .piano import render_piano
    from .sfz import note_number
    if kind not in IMPACT_KINDS:
        raise ValueError(f'kind must be one of {list(IMPACT_KINDS)}')
    v = float(np.clip(intensity, 0.0, 1.0))
    root = note_number(f'{key}1') if isinstance(key, str) else int(key)
    pre = 0.1
    hit = int(pre * SR)
    n = int((pre + tail_s) * SR)
    parts = []

    def put(x, lufs_target, onset_db=-30.0):
        x = _lufs_to(stereo(x), lufs_target)
        on = _onset(x, onset_db)
        buf = np.zeros((n, 2), np.float32)
        s0 = hit - on
        a, b = max(0, s0), min(n, s0 + len(x))
        buf[a:b] = x[a - s0:b - s0]
        parts.append(buf)

    if kind in ('soft_low', 'felt_low'):
        felt = kind == 'felt_low'
        chord = [root, root + 12, root + 19] if not felt else [root + 12, root + 19, root + 24]
        notes = [(pre + 0.006 * i, 3.0, m, int(40 + 30 * v - 6 * i)) for i, m in enumerate(chord)]
        pn = render_piano(notes, pedal=[(0.0, pre + 4.0)], felt=felt, duration=pre + tail_s,
                          sympathetic=0.5, seed=seed)
        pn = sosfilt(butter('lowpass', 1800.0 + 2200.0 * v, 2), pn)
        put(pn, -20.0)
        drum = sfx('log_drum', pitch='lo', vel=1 if v < 0.6 else 2, normalize=False)
        drum = sosfilt(butter('lowpass', 900.0 + 600.0 * v, 2), drum)
        put(drum, -29.0 + 4 * v)
    if kind in ('soft_low', 'gong_low'):
        g = sfx('gong', vel='p' if kind == 'soft_low' else 'mf', normalize=False)
        g = resample_by(g, 2 ** ((-5 if kind == 'soft_low' else -3) / 12))
        g = sosfilt(butter('lowpass', 1200.0 + 1500.0 * v, 2), g)
        put(g, (-27.0 if kind == 'soft_low' else -21.0) + 4 * v, onset_db=-20.0)
    if kind in ('soft_low', 'gong_low'):
        sb = sub(root + 12, 0.25 if kind == 'soft_low' else 0.8, attack=0.03,
                 release=2.0 if kind == 'soft_low' else 3.0, gain_db=-12.0)
        put(sb, -26.0 + 3 * v, onset_db=-40.0)
    dry = sum(parts)
    wet_kind = 'hall' if kind == 'felt_low' else 'valley'
    y = reverb(dry, wet_kind, mix=0.4 if kind != 'felt_low' else 0.3, seed=seed, lowcut_hz=60.0,
               highcut_hz=4000.0, tail=False)
    y = y[hit - int(0.001 * SR):]
    y = fade_edges(y, 0.001, min(1.0, tail_s * 0.3))
    y = sosfilt(butter('highpass', 25.0, 2), y)
    return _peak_to(fade_edges(y, 0.001, 0.02), peak_db)
