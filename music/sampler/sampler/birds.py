"""Real birds for the bird-piano dialogue.

    bird_calls(clear=True, max_dur=None, min_dur=None, source=None)   -> [dict]   the catalogue
    bird_call(call_id=None, *, tune='D', denoise=True, ...)            -> (n, 2) float32
    transcribe_call(call, scale='D', register=84, stretch=2.0, ...)    -> [(t, dur, midi, vel)]
    render_birds(birds, duration=None, ...)                            -> (n, 2)   score.BIRDS
    echo_notes(echo, birds=(), ...)                                    -> (notes, pedal) score.ECHO
    resolve_calls(birds, echo)                                         -> catalogue ids for None / labels

Catalogue: every call is cut from the four bird recordings already in <samples>/field
(public domain / CC0, see CREDITS.md): birds_garden_sparse 'gs', birds_fontainebleau 'fb',
birds_evening_waidachswald 'ew', dawn_chorus_dorset 'dd'. tools/catalog_birds.py runs
build_catalog() once and writes <samples>/field/bird_calls.json. Method (classic DSP):
STFT at 24 kHz (21 ms window, 5 ms hop); per-bin noise floor = 20th percentile over 2 s
blocks; in 1.2-10.5 kHz the strongest bin of each frame is followed (parabolic
interpolation) - that is the pitch contour. A frame is active when that bin stands >= 12 dB
over the floor; syllables closer than 120 ms form one call. Per call: duration, contour
(10 ms), level, SNR, tonal purity (energy within +-2 bins of the track / band energy),
isolation (the track vs. anything else in the band at the same time: other birds),
quiet margins before/after. 'clear' = SNR >= 22 dB, purity >= 0.35, isolation >= 10 dB,
quiet margins >= 10 dB under the call (inside the call's band), no broadband haze (<= 8 %
of frames with > 25 % of the band lit) and no clicks (spectral: a frame with > 50 % of the
band lit; waveform: 0.5 ms spikes of the >5 kHz band 20 dB over the local median),
0.08 s <= duration <= 2.5 s. All measured, none listened to.

bird_call: the cut (+150 ms of natural tail), spectral-subtraction denoise (noise PSD from
the 3 s around the call; gain floor -18 dB so the place's air stays; time/frequency
smoothed gains against musical noise) and a soft band-pass around the call; then the whole
call is played a little faster/slower (varispeed, at most +-150 cents for the pentatonic
scale; duration changes by the same ratio, <= 9 %) so its central pitch lands on a scale
tone. No pitch-shifting algorithm touches the bird.

transcribe_call (after Messiaen's practice): the contour is transposed down by octaves into
the piano register (`register` = lowest MIDI note allowed), slowed by `stretch`, cut into
notes at gaps and at jumps, every note quantised to the nearest scale tone; a fast sweep
becomes a run through the scale tones it crosses (glide='run') or its two ends
(glide='ends'); velocity follows the call's level.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter

from ._dsp import SR, db2a, fade_edges, pan_gains, resample_by, stereo
from ._paths import SAMPLES, log_access, need

FIELD = SAMPLES / 'field'
CATALOG = FIELD / 'bird_calls.json'
SOURCES = {'gs': 'birds_garden_sparse', 'fb': 'birds_fontainebleau',
           'ew': 'birds_evening_waidachswald', 'dd': 'dawn_chorus_dorset'}

A_SR = 24000                  # analysis rate (birds here sit in 1.2-10.5 kHz)
A_NFFT, A_HOP = 512, 120      # 21 ms window, 5 ms hop
BAND = (1200.0, 10500.0)
ACT_DB = 15.0                 # frame active when the tracked bin is this far over the floor
SYL_GAP = 0.12                # syllables closer than this form one call

_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
SCALES = {'D': (2, 4, 6, 9, 11),                 # D gong pentatonic (D E F# A B)
          'D_major': (1, 2, 4, 6, 7, 9, 11), 'G': (7, 9, 11, 2, 4), 'A': (9, 11, 1, 4, 6),
          'chromatic': tuple(range(12))}


# ----------------------------------------------------------------------------- analysis
def _stft_power(x, sr=A_SR, nfft=A_NFFT, hop=A_HOP):
    n = 1 + max(0, (len(x) - nfft) // hop)
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx].astype(np.float32) * np.hanning(nfft).astype(np.float32)
    P = (np.abs(np.fft.rfft(fr, axis=1)) ** 2).astype(np.float32)
    f = np.fft.rfftfreq(nfft, 1 / sr)
    t = (np.arange(n) * hop + nfft / 2) / sr
    return t, f, P


def _noise_floor(P, block_frames=400, pct=50):
    """Per-bin floor per frame: median of 2 s blocks (in a chorus: the chorus bed), interpolated."""
    nb = max(1, int(np.ceil(len(P) / block_frames)))
    cent, vals = [], []
    for b in range(nb):
        seg = P[b * block_frames:(b + 1) * block_frames]
        if len(seg) < 20:
            continue
        cent.append(b * block_frames + len(seg) / 2)
        vals.append(np.percentile(seg, pct, axis=0))
    vals = np.array(vals)
    if len(vals) == 1:
        return np.repeat(vals, len(P), axis=0)
    fi = np.arange(len(P))
    out = np.empty_like(P)
    for k in range(P.shape[1]):
        out[:, k] = np.interp(fi, cent, vals[:, k])
    return np.maximum(out, 1e-20)


def _track(P, N, f):
    """Strongest in-band bin per frame -> (freq Hz, SNR dB, purity, other dB)."""
    band = np.flatnonzero((f >= BAND[0]) & (f <= BAND[1]))
    Pb, Nb = P[:, band], N[:, band]
    S = 10 * np.log10((Pb + 1e-20) / Nb)
    k = np.argmax(Pb, axis=1)
    rows = np.arange(len(P))
    snr = S[rows, k]
    kk = np.clip(k, 1, len(band) - 2)
    y0, y1, y2 = (np.log(Pb[rows, kk - 1] + 1e-20), np.log(Pb[rows, kk] + 1e-20),
                  np.log(Pb[rows, kk + 1] + 1e-20))
    den = y0 - 2 * y1 + y2
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0)
    freq = f[band[0]] + (kk + np.clip(d, -0.5, 0.5)) * (f[1] - f[0])
    ex = np.maximum(Pb - Nb, 0)
    lo, hi = np.clip(k - 2, 0, None), np.clip(k + 3, None, len(band))
    csum = np.concatenate([np.zeros((len(P), 1), np.float32), np.cumsum(ex, axis=1)], axis=1)
    near = csum[rows, hi] - csum[rows, lo]
    purity = near / (csum[:, -1] + 1e-20)
    # anything else in the band: exclude +-15 % around the track and around its 2nd harmonic
    fb = f[band][None, :]
    ft = freq[:, None]
    mask = (np.abs(fb - ft) < 0.15 * ft) | (np.abs(fb - 2 * ft) < 0.1 * ft) | (np.abs(fb - ft / 2) < 0.08 * ft)
    Sx = np.where(mask, -99.0, S).astype(np.float32)          # the band without the tracked bird
    wide = (S > 10).mean(axis=1).astype(np.float32)           # broadband frames: clicks, haze
    return (freq.astype(np.float32), snr.astype(np.float32), purity.astype(np.float32), Sx,
            f[band].astype(np.float32), wide)


def _runs(active):
    a = np.concatenate([[0], active.astype(np.int8), [0]])
    d = np.diff(a)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _fill(active, max_gap):
    act = active.copy()
    r = _runs(act)
    for (a0, a1), (b0, _) in zip(r[:-1], r[1:]):
        if b0 - a1 <= max_gap:
            act[a1:b0] = True
    return act


@lru_cache(maxsize=8)
def _source_mono(name):
    import soundfile as sf
    log_access(FIELD / f'{name}.flac')
    x, sr = sf.read(str(need(FIELD / f'{name}.flac', f'鸟鸣录音 {name}')), dtype='float32', always_2d=True)
    return x, sr


def _analyse_mono(x, sr):
    from ._dsp import resample_sr
    m = x.mean(axis=1) if x.ndim == 2 else x
    y = resample_sr(m, sr, A_SR)
    t, f, P = _stft_power(y)
    N = _noise_floor(P)
    return t, f, P, N


def _calls_from(t, freq, snr, purity, Sx, fb, wide, max_dur=2.5):
    """Group active frames into calls and measure them. Margins and isolation are taken inside
    the call's own band (f_lo / 1.5 .. f_hi x 1.6 - the band bird_call keeps), because that
    is what remains audible after the band-pass."""
    hop = A_HOP / A_SR
    active = snr >= ACT_DB
    active = _fill(active, int(0.03 / hop))
    syl = [(a, b) for a, b in _runs(active) if (b - a) * hop >= 0.025]
    calls, cur = [], []
    for a, b in syl:
        if cur and (a - cur[-1][1]) * hop <= SYL_GAP and (b - cur[0][0]) * hop <= max_dur:
            cur.append((a, b))
        else:
            if cur:
                calls.append(cur)
            cur = [(a, b)]
    if cur:
        calls.append(cur)
    rows = []
    w0, w1 = int(0.03 / hop), int(0.25 / hop)
    for c in calls:
        a, b = c[0][0], c[-1][1]
        on = np.zeros(b - a, bool)
        for s0, s1 in c:
            on[s0 - a:s1 - a] = True
        fr = np.where(on, freq[a:b], np.nan)
        lv = np.where(on, snr[a:b], np.nan)
        fv = fr[on]
        f_lo, f_hi = float(np.percentile(fv, 5)), float(np.percentile(fv, 95))
        kb = (fb >= f_lo / 1.5) & (fb <= f_hi * 1.6)
        s90 = float(np.nanpercentile(lv, 90))
        around = np.concatenate([Sx[max(0, a - w1):max(0, a - w0)], Sx[min(len(Sx), b + w0):b + w1]])
        # before/after: the loudest thing in the call's band (the call itself is not there yet)
        full = np.concatenate([snr[max(0, a - w1):max(0, a - w0)], snr[min(len(snr), b + w0):b + w1]])
        quiet = float(np.percentile(np.maximum(around[:, kb].max(axis=1), 0), 90)) if len(around) else 0.0
        oth = float(np.percentile(Sx[a:b][on][:, kb].max(axis=1), 90)) if kb.any() else 0.0
        pur = float(np.median(purity[a:b][on]))
        wd = wide[max(0, a - w0):b + w0]
        haze = float((wide[a:b][on] > 0.25).mean())
        clicks = int((wd > 0.5).sum())
        dur = (b - a) * hop
        row = dict(t0=round(float(t[a] - hop / 2), 3), t1=round(float(t[b - 1] + hop / 2), 3),
                   dur=round(float(dur), 3), n_syll=len(c),
                   f_med=round(float(np.median(fv)), 1), f_lo=round(f_lo, 1), f_hi=round(f_hi, 1),
                   snr_db=round(s90, 1), purity=round(pur, 3), isolation_db=round(s90 - oth, 1),
                   margin_db=round(s90 - quiet, 1),
                   margin_full_db=round(s90 - float(np.percentile(full, 90)) if len(full) else s90, 1),
                   haze=round(haze, 3), click_frames=clicks,
                   contour_hz=[None if not np.isfinite(v) else round(float(v), 1) for v in fr[::2]],
                   level_db=[None if not np.isfinite(v) else round(float(v), 1) for v in lv[::2]])
        row['clear'] = bool(s90 >= 22 and pur >= 0.35 and s90 - oth >= 10 and s90 - quiet >= 10
                            and haze <= 0.08 and clicks == 0 and 0.08 <= dur <= 2.5)
        row['score'] = round(float(s90 / 10 + min(s90 - oth, 30) / 10 + 3 * pur
                                   + min(s90 - quiet, 30) / 15 - 0.5 * max(0.0, dur - 1.2)
                                   - 6 * haze - 0.5 * min(clicks, 4)), 2)
        rows.append(row)
    return rows


def _wave_clicks(seg, sr):
    """Clicks in a cut: 0.5 ms peaks of the >5 kHz band standing 20 dB over the 40 ms local median."""
    from scipy.ndimage import maximum_filter1d, median_filter
    hp = signal.sosfilt(signal.butter(4, 5000, 'highpass', fs=sr, output='sos'), seg.astype(np.float64))
    k = max(1, int(0.0005 * sr))
    e = np.abs(hp[:len(hp) // k * k]).reshape(-1, k).max(axis=1)
    if len(e) < 10:
        return 0
    med = median_filter(e, size=int(0.04 * sr / k) | 1, mode='nearest')
    spike = (e > 10 * (med + 1e-7)) & (e == maximum_filter1d(e, 9)) & (e > 10 ** (-60 / 20))
    return int(spike.sum())


def build_catalog(sources=SOURCES, verbose=True):
    """Analyse the bird recordings and return the catalogue (list of dicts, ids like 'gs-007')."""
    out = []
    for prefix, name in sources.items():
        if not (FIELD / f'{name}.flac').exists():
            continue
        x, sr = _source_mono(name)
        t, f, P, N = _analyse_mono(x, sr)
        freq, snr, pur, Sx, fb, wide = _track(P, N, f)
        rows = _calls_from(t, freq, snr, pur, Sx, fb, wide)
        mono = x.mean(axis=1)
        for i, r in enumerate(rows):
            r['id'] = f'{prefix}-{i:03d}'
            r['source'] = name
            seg = mono[max(0, int((r['t0'] - 0.04) * sr)):int((r['t1'] + 0.15) * sr)]
            r['wave_clicks'] = _wave_clicks(seg, sr)
            if r['wave_clicks']:
                r['clear'] = False
                r['score'] = round(r['score'] - 0.5 * min(r['wave_clicks'], 4), 2)
        out += rows
        if verbose:
            nc = sum(r['clear'] for r in rows)
            print(f'{name:28} {len(x) / sr:6.1f} s  calls {len(rows):4d}  clear {nc:4d}')
    return out


def write_catalog(path=CATALOG):
    rows = build_catalog()
    meta = dict(method=__doc__.split('Catalogue:')[1].split('bird_call:')[0].strip(),
                analysis_sr=A_SR, nfft=A_NFFT, hop_s=A_HOP / A_SR, contour_step_s=2 * A_HOP / A_SR,
                band_hz=BAND, sources={k: v for k, v in SOURCES.items()})
    Path(path).write_text(json.dumps(dict(meta=meta, calls=rows), ensure_ascii=False, separators=(',', ':')))
    return rows


@lru_cache(maxsize=1)
def _catalog():
    if not CATALOG.exists():
        write_catalog()
    return json.loads(CATALOG.read_text())['calls']


def _row(call_id):
    for r in _catalog():
        if r['id'] == call_id:
            return r
    raise KeyError(f'unknown bird call {call_id!r}')


def bird_calls(clear=True, max_dur=None, min_dur=None, source=None, sort='score'):
    """Catalogue rows (dicts: id, source, t0, t1, dur, f_med, f_lo, f_hi, snr_db, purity,
    isolation_db, margin_db, clear, score, contour_hz (10 ms, None = silent), level_db).
    Default: the clear ones, best first."""
    rows = [r for r in _catalog() if (not clear or r['clear'])
            and (max_dur is None or r['dur'] <= max_dur) and (min_dur is None or r['dur'] >= min_dur)
            and (source is None or r['source'] == source or r['id'].startswith(source))]
    if sort == 'score':
        rows = sorted(rows, key=lambda r: -r['score'])
    return rows


def pick_calls(n, *, max_dur=1.2, min_dur=0.12, seed=0, exclude=(), top=24):
    """n distinct clear calls, drawn (seeded) from the `top` best that fit the length."""
    pool = [r['id'] for r in bird_calls(True, max_dur, min_dur) if r['id'] not in exclude][:top]
    if not pool:
        pool = [r['id'] for r in bird_calls(False, max_dur, min_dur)][:top]
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(pool)))
    return [pool[order[i % len(pool)]] for i in range(n)]


# ----------------------------------------------------------------------------- one call
def _scale_pcs(scale):
    if isinstance(scale, str):
        return SCALES[scale]
    return tuple(int(p) % 12 for p in scale)


def _nearest_in_scale(m, pcs):
    base = int(np.floor(m))
    cands = [k for k in range(base - 7, base + 8) if k % 12 in pcs]
    return min(cands, key=lambda k: (abs(k - m), k))


def _denoise(x, noise_psd, nfft=1024, hop=256, floor_db=-18.0, over=2.0, band=None):
    """Spectral subtraction with a gain floor, gains smoothed over 3 frames x 3 bins, the same
    gain on both channels (keeps the stereo image)."""
    win = np.sqrt(np.hanning(nfft + 1)[:-1])
    xs = stereo(x).astype(np.float64)
    pad = np.pad(xs, ((nfft, nfft), (0, 0)))
    n = 1 + (len(pad) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    F = [np.fft.rfft(pad[idx, c] * win, axis=1) for c in range(2)]
    Pm = 0.5 * (np.abs(F[0]) ** 2 + np.abs(F[1]) ** 2)
    G = np.clip(1 - over * noise_psd[None, :] / (Pm + 1e-20), db2a(floor_db) ** 2, 1.0) ** 0.5
    G = uniform_filter(G, size=(3, 3), mode='nearest')
    if band is not None:
        f = np.fft.rfftfreq(nfft, 1 / SR)
        lo, hi = band
        bm = np.clip(np.minimum((f / lo) ** 3, (hi / np.maximum(f, 1)) ** 3), db2a(floor_db), 1.0)
        G = G * bm[None, :]
    out = np.zeros_like(pad)
    wsum = np.zeros(len(pad))
    for c in range(2):
        fr = np.fft.irfft(F[c] * G, nfft, axis=1) * win
        for i in range(n):
            out[i * hop:i * hop + nfft, c] += fr[i]
    for i in range(n):
        wsum[i * hop:i * hop + nfft] += win ** 2
    out /= np.maximum(wsum, 1e-9)[:, None]
    return out[nfft:nfft + len(xs)].astype(np.float32)


def _noise_psd_around(src, a, b, nfft=1024, hop=256, span_s=1.5):
    """Noise PSD (20th percentile per bin) from the source around [a, b] (samples, 48 kHz)."""
    lo, hi = max(0, a - int(span_s * SR)), min(len(src), b + int(span_s * SR))
    m = src[lo:hi].mean(axis=1).astype(np.float64)
    win = np.sqrt(np.hanning(nfft + 1)[:-1])
    n = 1 + max(0, (len(m) - nfft) // hop)
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    P = np.abs(np.fft.rfft(m[idx] * win, axis=1)) ** 2
    return np.percentile(P, 20, axis=0)


@lru_cache(maxsize=8)
def _source48(name):
    from ._dsp import resample_sr
    x, sr = _source_mono(name)
    return resample_sr(x, sr, SR) if sr != SR else x


def bird_call(call_id=None, *, tune='D', denoise=True, floor_db=-18.0, tail_s=0.15, peak_db=-6.0,
              pan=0.0, mono=True, seed=0, return_info=False):
    """One real call, cleaned and (optionally) tuned. call_id None -> a clear short call
    picked by `seed`. tune: scale name ('D' = D E F# A B, see SCALES), a list of pitch
    classes, or None (as recorded). mono=True keeps the one channel of a stereo source that
    carries more of the call (a point source, then panned by `pan`); mono=False keeps the
    recording's stereo. Returns stereo float32 peak-normalised to peak_db."""
    if call_id is None:
        call_id = pick_calls(1, seed=seed)[0]
    r = _row(call_id)
    src = stereo(_source48(r['source']))
    a = max(0, int((r['t0'] - 0.04) * SR))
    b = min(len(src), int((r['t1'] + tail_s) * SR))
    x = src[a:b]
    if mono:        # one bird = one point: the channel with more energy in the call's band
        sos = signal.butter(4, [max(200.0, r['f_lo'] / 1.5), min(r['f_hi'] * 1.6, 0.45 * SR)], 'bandpass',
                            fs=SR, output='sos')
        e = (signal.sosfilt(sos, x, axis=0) ** 2).sum(axis=0)
        x = x[:, [int(np.argmax(e))]].repeat(2, axis=1)
    if denoise:
        psd = _noise_psd_around(src, a, b)
        x = _denoise(x, psd, floor_db=floor_db, band=(r['f_lo'] / 1.5, min(r['f_hi'] * 1.6, 20000.0)))
    x = x.astype(np.float64)
    x -= x.mean(axis=0)
    shift = 0.0
    if tune is not None:
        m = 69 + 12 * np.log2(r['f_med'] / 440)
        target = _nearest_in_scale(m, _scale_pcs(tune))
        shift = float(target - m)
        x = resample_by(x, 2 ** (shift / 12)).astype(np.float64)
    x = stereo(x)
    x = fade_edges(x.astype(np.float32), 0.02, min(0.12, tail_s * 0.8))
    pk = float(np.abs(x).max())
    x = x * (db2a(peak_db) / pk if pk > 0 else 1.0)
    if pan:
        gl, gr = pan_gains(pan)
        x = x * np.array([gl, gr], np.float32) * np.sqrt(2)
    x = x.astype(np.float32)
    if return_info:
        return x, dict(r, shift_semitones=round(shift, 3), tuned_to=None if tune is None else
                       round(69 + 12 * np.log2(r['f_med'] / 440) + shift, 2))
    return x


# ----------------------------------------------------------------------------- transcription
def _contour_of(call):
    """(times s, midi float or nan, level dB) at 10 ms from a catalogue id / row / audio."""
    if isinstance(call, str):
        call = _row(call)
    if isinstance(call, dict):
        f = np.array([np.nan if v is None else v for v in call['contour_hz']], float)
        lv = np.array([np.nan if v is None else v for v in call['level_db']], float)
    else:                                   # an audio clip: analyse it the same way
        x = stereo(np.asarray(call, np.float32))
        from ._dsp import resample_sr
        y = resample_sr(x.mean(axis=1), SR, A_SR)
        t, fq, P = _stft_power(y)
        N = np.repeat(np.percentile(P, 50, axis=0)[None, :], len(P), axis=0) + 1e-20
        fr, snr, _, _, _, _ = _track(P, N, fq)
        act = _fill(snr >= ACT_DB, 6)
        f = np.where(act, fr, np.nan)[::2]
        lv = np.where(act, snr, np.nan)[::2]
    step = 2 * A_HOP / A_SR
    with np.errstate(divide='ignore', invalid='ignore'):
        m = 69 + 12 * np.log2(f / 440)
    return np.arange(len(f)) * step, m, lv


def _dp_split(y, a, b, tol, out):
    """Douglas-Peucker: split y[a:b] into straight pieces within `tol` semitones."""
    if b - a <= 2:
        out.append((a, b))
        return
    line = y[a] + (y[b - 1] - y[a]) * (np.arange(a, b) - a) / max(1, b - 1 - a)
    d = np.abs(y[a:b] - line)
    k = int(np.argmax(d))
    if d[k] > tol and 0 < k < b - a - 1:
        _dp_split(y, a, a + k + 1, tol, out)
        _dp_split(y, a + k, b, tol, out)
    else:
        out.append((a, b))


def transcribe_call(call, scale='D', register=84, stretch=2.0, *, top=108, quant=None, bpm=None,
                    grid=0.25, detail=0.5, glide='run', sweep_st_s=12.0, run_max=4, grace_s=0.05,
                    min_dur=0.08, vel=(34, 58), max_notes=None):
    """Messiaen-style transcription of a bird call into piano notes.

    call     : catalogue id, catalogue row, or an audio clip (48 kHz)
    scale    : 'D' (D E F# A B), another SCALES name, or pitch classes
    register : lowest MIDI note; the call moves down by whole octaves as far as it can while its
               10th-percentile pitch stays at or above it (birds sing 2-8 kHz = MIDI 95-120);
               single notes still below it go up an octave, notes above `top` down one
    stretch  : time factor (2 = half speed)
    detail   : 0..1. The contour is cut at gaps and at jumps, then into straight pieces
               (tolerance 1.2 - 0.9*detail semitones); steady pieces shorter than
               (40 - 30*detail) ms of the bird's time are absorbed by their neighbour
    glide    : a piece sweeping faster than sweep_st_s semitones/s becomes 'run' (the scale
               tones it crosses, at most run_max, as grace notes landing on the last one) or
               'ends' (grace note + landing note)
    quant    : grid (s) for the main notes (None = the bird's own timing, stretched); or bpm +
               grid in beats. Grace notes stay unquantised, grace_s apart, before their landing
    vel      : (low, high), from the call's level; grace notes 6 lower
    Returns [(t, dur, midi, vel)], t from 0, sorted."""
    pcs = _scale_pcs(scale)
    t, m, lv = _contour_of(call)
    step = t[1] - t[0] if len(t) > 1 else 0.01
    ok = np.isfinite(m)
    if ok.sum() < 2:
        return []
    p10 = float(np.nanpercentile(m, 10))
    m = m - 12 * np.floor((p10 - register) / 12)
    lv_ok = lv[ok]
    l_lo, l_hi = np.nanpercentile(lv_ok, 10), np.nanpercentile(lv_ok, 95)

    def v_of(level):
        u = np.clip((level - l_lo) / max(1e-6, l_hi - l_lo), 0, 1)
        return int(round(vel[0] + (vel[1] - vel[0]) * u))

    tol = 1.2 - 0.9 * float(np.clip(detail, 0, 1))
    min_src = (40 - 30 * float(np.clip(detail, 0, 1))) / 1000
    pieces = []                              # (a, b) frame ranges
    for a, b in _runs(ok):
        ys = m.copy()
        if b - a >= 3:                       # 3-point median against single-frame jitter
            from scipy.ndimage import median_filter
            ys[a:b] = median_filter(m[a:b], 3, mode='nearest')
        cuts = [a] + [i for i in range(a + 1, b) if abs(ys[i] - ys[i - 1]) > 2.5] + [b]
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            _dp_split(ys, c0, c1, tol, pieces)
    # notes: [start, dur, midi, vel, grace?]
    notes = []
    for a, b in pieces:
        seg = m[a:b]
        dur = (b - a) * step
        level = float(np.nanmean(lv[a:b]))
        ext = seg[-1] - seg[0]
        slope = ext / max(dur, step)
        q0, q1 = _nearest_in_scale(seg[0], pcs), _nearest_in_scale(seg[-1], pcs)
        if abs(slope) >= sweep_st_s and abs(ext) >= 1.5 and q0 != q1:
            lo_, hi_ = sorted((q0, q1))
            tones = [k for k in range(lo_, hi_ + 1) if k % 12 in pcs]
            if q1 < q0:
                tones = tones[::-1]
            if glide == 'ends' or len(tones) <= 2:
                tones = [tones[0], tones[-1]]
            elif len(tones) > run_max:
                idx = np.round(np.linspace(0, len(tones) - 1, run_max)).astype(int)
                tones = [tones[i] for i in idx]
            land = t[a] + dur * 0.75           # the landing tone sounds where the sweep ends
            for j, p in enumerate(tones[:-1]):
                notes.append([land, 0.0, p, v_of(level) - 6, len(tones) - 1 - j])
            notes.append([land, max(dur * 0.25, step), tones[-1], v_of(level), 0])
        else:
            notes.append([t[a], dur, _nearest_in_scale(float(np.median(seg)), pcs), v_of(level), 0])
    notes.sort(key=lambda n: (n[0], -n[4]))
    # absorb very short steady notes into the previous main note; tie repeats
    main = []
    for n in notes:
        prev = next((q for q in reversed(main) if q[4] == 0), None)
        if n[4] == 0 and prev is not None:
            gap = n[0] - (prev[0] + prev[1])
            if n[2] == prev[2] and gap < 0.03:
                prev[1] = n[0] + n[1] - prev[0]
                continue
            if n[1] < min_src and gap < 0.03 and not any(q[4] for q in main[-1:]):
                prev[1] = n[0] + n[1] - prev[0]
                continue
        main.append(n)
    for n in main:
        while n[2] > top:
            n[2] -= 12
        while n[2] < register:
            n[2] += 12
    # time: stretch, quantise the main notes, hang the grace notes before their landing note
    if bpm is not None:
        quant = 60.0 / bpm * grid
    out, last = [], None
    for n in main:
        st, du, p, v, g = n
        if g:
            continue
        st, du = st * stretch, du * stretch
        if quant:
            q = float(quant)
            s0 = round(st / q) * q
            if last is not None and s0 <= last + 1e-9:
                s0 = last + q
            du = max(q, round(du / q) * q)
            st = s0
        du = max(du, min_dur)
        last = st
        n.append((st, du))
    prev_on = -1e9
    for i, n in enumerate(main):
        if n[4]:
            land = next((q for q in main[i + 1:] if q[4] == 0), None)
            if land is None:
                continue
            gs = land[5][0] - n[4] * grace_s
            if gs <= prev_on + 0.03:              # no room after the previous note: leave it out
                continue
            out.append((gs, grace_s * 1.4, n[2], max(1, n[3]), 1))
        else:
            out.append((n[5][0], n[5][1], n[2], n[3], 0))
            prev_on = n[5][0]
    # one key cannot be struck twice within 60 ms: keep the first (a main note wins over a grace)
    out.sort(key=lambda o: (o[0], o[4]))
    kept = []
    for o in out:
        if any(k[2] == o[2] and o[0] - k[0] < 0.06 for k in kept[-6:]):
            continue
        kept.append(o)
    out = [o[:4] for o in kept]
    # make durations legato up to the next main onset at most (a line, not a cluster)
    out.sort()
    res = []
    for i, (st, du, p, v) in enumerate(out):
        nxt = next((o[0] for o in out[i + 1:] if o[0] > st + 1e-6), None)
        if nxt is not None and du > (nxt - st) * 1.5 and du > grace_s * 2:
            du = max(min_dur, (nxt - st) * 1.5)
        res.append([st, du, p, v])
    if max_notes:
        res = res[:max_notes]
    t0 = min(r[0] for r in res) if res else 0.0
    return [(round(st - t0, 4), round(du, 4), int(p), int(v)) for st, du, p, v in res]


# ----------------------------------------------------------------------------- score helpers
FIRST_LABELS = ('prologue', 'first', 'opening')      # an echo label meaning "the first bird"
LAST_LABELS = ('previous', 'last', 'latest')          # "the latest bird before this echo"


def resolve_calls(birds=(), echo=(), *, seed=0, max_dur=1.2):
    """Turn the call ids of score.BIRDS / score.ECHO into catalogue ids, deterministically.

    - a catalogue id ('gs-002') is used as given;
    - None: a bird gets a clear short call no other entry uses; an echo learns the latest bird
      placed before it (or gets a call of its own if there is none);
    - any other string is a LABEL: every entry with the same label gets the same call. In an
      echo, 'prologue' / 'first' / 'opening' mean the first bird of the piece and 'previous' /
      'last' / 'latest' the latest bird before the echo, unless a bird carries that label.
    Returns (bird_ids in time order, echo_ids in the given order)."""
    birds = sorted(birds, key=lambda b: b[0])
    cat = {r['id'] for r in _catalog()}
    fixed = {b[1] for b in birds if b[1] in cat} | {e[1] for e in echo if e[1] in cat}
    fresh = iter(pick_calls(len(birds) + len(echo) + 1, max_dur=max_dur, seed=seed, exclude=fixed))
    labels = {}
    bird_ids = []
    for b in birds:
        cid = b[1]
        if cid is None:
            bird_ids.append(next(fresh))
        elif cid in cat:
            bird_ids.append(cid)
        else:
            if cid not in labels:
                labels[cid] = next(fresh)
            bird_ids.append(labels[cid])
    echo_ids = []
    for e in echo:
        cid = e[1]
        prev = [bid for b, bid in zip(birds, bird_ids) if b[0] <= e[0]]
        if cid in cat:
            echo_ids.append(cid)
        elif cid is not None and cid in labels:
            echo_ids.append(labels[cid])
        elif cid is not None and str(cid).lower() in FIRST_LABELS and bird_ids:
            echo_ids.append(bird_ids[0])
        elif cid is None or str(cid).lower() in LAST_LABELS:
            echo_ids.append(prev[-1] if prev else next(fresh))
        else:
            labels[cid] = next(fresh)
            echo_ids.append(labels[cid])
    return bird_ids, echo_ids


def render_birds(birds, duration=None, *, tune='D', denoise=True, echo=(), seed=0,
                 gain_db=0.0) -> np.ndarray:
    """score.BIRDS [(start_s, call_id|None, gain_db, pan)] -> stereo track (each call peaks at
    -6 dBFS + its gain_db). Pass the score's ECHO too so that None ids resolve the same way
    as in echo_notes()."""
    birds = sorted(birds, key=lambda b: b[0])
    ids, _ = resolve_calls(birds, echo, seed=seed)
    parts = []
    for (st, _, g, p), cid in zip(birds, ids):
        x = bird_call(cid, tune=tune, denoise=denoise, pan=p)
        parts.append((int(round(st * SR)), x * db2a(g + gain_db)))
    end = max([s + len(x) for s, x in parts], default=int(0.1 * SR))
    n = int(duration * SR) if duration else end
    out = np.zeros((n, 2), np.float32)
    for s, x in parts:
        if s < n:
            k = min(len(x), n - s)
            out[s:s + k] += x[:k]
    return out


def echo_notes(echo, birds=(), *, scale='D', seed=0, glide='run', quant=None, res_vel=0.55,
               res_hold_s=1.5, pedal=True):
    """score.ECHO [(start_s, call_id|None, stretch, low_midi, vel, [resonance midi])] ->
    (piano notes [(start, dur, midi, vel)], pedal [(down, up)]). The call's transcription
    starts at start_s; `vel` becomes the loudest note; the resonance chord is struck softly
    (res_vel x vel) under the first note and held through the phrase; the sustain pedal is down
    for the phrase + res_hold_s."""
    _, ids = resolve_calls(birds, echo, seed=seed)
    notes, ped = [], []
    for e, cid in zip(echo, ids):
        st, _, stretch, low, vel, res = e[:6]
        tr = transcribe_call(cid, scale=scale, register=low, stretch=stretch, glide=glide, quant=quant)
        if not tr:
            continue
        vmax = max(v for *_, v in tr)
        end = max(t + d for t, d, _, _ in tr)
        for t, d, p, v in tr:
            notes.append((st + t, d, p, max(1, int(round(vel * v / vmax)))))
        for p in res:
            notes.append((st, end + res_hold_s, int(p), max(1, int(round(vel * res_vel)))))
        if pedal:
            ped.append((round(st + 0.01, 4), round(float(st + end + res_hold_s), 4)))
    return sorted(notes), ped
