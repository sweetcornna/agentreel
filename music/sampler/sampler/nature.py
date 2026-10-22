"""Nature sounds: real field recordings first, classic procedural synthesis as fallback.

    ambience(kind, duration, seed=0, loop=False, level_lufs=-30) -> (n, 2)
    sfx(kind, seed=0, **kw)                                        -> (n, 2)
    distant(x, amount=0.7, ...)                                    -> (n', 2)

KINDS_AMBIENCE / KINDS_SFX list every kind with its source (recording + licence, or
'synth'). Recordings live in <samples>/field (Wikimedia Commons, CC0 / public domain /
CC-BY, see CREDITS.md), <samples>/vcsl (VCSL, CC0) and <samples>/objects (Freesound, CC0).

Arbitrary-length ambiences from recordings: random segments of the source, cut at
low-energy moments and joined with long equal-power crossfades (segments are
uncorrelated, so the level stays constant); mono sources become stereo from two
independent segment streams with controlled L/R correlation. loop=True makes the
end crossfade back into the start, so the result can be tiled without a seam.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import soundfile as sf
from scipy import signal

from ._dsp import (SR, add_at, butter, cos_ramp, db2a, fade_edges, midi_to_hz, pan_gains,
                   resample_by, resample_sr, smooth_noise, sosfilt, stereo)
from ._paths import SAMPLES, log_access, need

FIELD = SAMPLES / 'field'
VCSL = SAMPLES / 'vcsl'

# ----------------------------------------------------------------------------- catalogue
# recorded ambiences: kind -> (library files, segment length range s, crossfade s,
#                              L/R correlation for mono sources, description)
_REC_AMB = {
    'stream': (['stream_molln_bachlauf'], (7, 12), 2.5, 0.35,
               'babbling mountain brook, close (DrTrumpet, CC0)'),
    'stream_gentle': (['rivulet_molln_rinnsal'], (7, 12), 2.5, 0.35,
                      'small trickling rivulet (DrTrumpet, CC0)'),
    'creek': (['creek_forest_lawn'], (8, 16), 3.0, 0.3,
              'forest creek, 7.5 min source = little repetition (Dsw4, public domain)'),
    'river': (['river_shallow_stony'], (6, 9), 2.5, None,
              'shallow stony river, stereo (stephan / pdsounds, public domain)'),
    'forest_wind': (['forest_ambience_nille'], (8, 16), 3.0, None,
                    'soft wind in trees, occasional distant birds (nille / pdsounds, PD)'),
    'pine_forest': (['pine_forest_dobroide'], (8, 14), 3.0, None,
                    'summer pine forest: wind in pines, birds, cicadas (dobroide, CC BY 3.0)'),
    'birds_morning': (['birds_fontainebleau', 'dawn_chorus_dorset'], (5, 9), 2.0, None,
                      'forest bird chorus (Barracuda1983, PD) layered with a dawn-chorus excerpt '
                      '(Silas S. Brown, PD; ~6 kHz bandwidth)'),
    'birds_sparse': (['birds_garden_sparse'], (8, 14), 2.0, 0.2,
                     'sparse garden bird phrases, quiet (ezwa / pdsounds, PD)'),
    'birds_evening': (['birds_evening_waidachswald'], (8, 16), 3.0, None,
                      'summer evening forest birdsong, true stereo (Evillan, CC0)'),
    'crickets': (['cricket_gryllus_campestris'], (6, 12), 2.0, None,
                 'field crickets: one NHM/BioAcoustica recording (CC BY 3.0) layered as '
                 '4 individuals at different distance / pitch / pan'),
}
_SYN_AMB = {
    'stream_synth': 'Minnaert bubbles (van den Doel damping) over a low-passed noise bed',
    'wind_synth': 'coloured noise, slowly wandering band-pass, gust envelope, leaf rustle',
    'bamboo_wind_synth': 'wind_synth with dry bamboo-leaf rustle and rare soft culm knocks',
    'birds_synth': 'FM chirp / whistle / trill phrases from 4 synthetic "species"',
    'insects_synth': 'pulse-modulated ~4.5 kHz carriers (crickets) + a tree-cricket trill',
    'night_insects': 'crickets (recorded, layered) + insects_synth bed',
}
KINDS_AMBIENCE = {**{k: v[4] for k, v in _REC_AMB.items()},
                  **{k: ('mixed: ' if k == 'night_insects' else 'synth: ') + v
                     for k, v in _SYN_AMB.items()}}

KINDS_SFX = {
    'rooster': 'rooster crow, clean (DenisChardonnet, CC0); seed picks one of 3 recordings '
               '(the other two: alys / pdsounds, PD, with farm background)',
    'dog': 'dog bark(s): even seed = Edo.pt2 (CC0, 3 barks, de-clipped), odd seed = '
           '"George" (Broadbeer, PD, 1 bark repeated with variation); kw barks=1|2|3, '
           "source='edo'|'george', synth=True for the synthetic fallback. Meant for distant()",
    'woodblock': 'VCSL woodblock (CC0); kw vel in pp|mp|f|ff',
    'hand_chime': 'VCSL hand chime (CC0); kw note = SOUNDING pitch (MIDI or name; VCSL files are '
                  'named an octave below the sounding pitch - measured - and this is compensated), '
                  'range C4-C7, whole-tone sampled + resampled',
    'tubular_bell': 'VCSL tubular bell (CC0); kw note = VCSL written name (C3-A#3..E4 sampled; the '
                    'strongest partial measures 2 octaves above the name), vel in p|f',
    'gong': 'VCSL gong (CC0); kw vel in p|mf|f|fff',
    'gong_scrape': 'VCSL gong scrape (CC0)',
    'log_drum': 'VCSL slit drum (CC0); kw pitch in hi|lo, vel 1..3 - closest recorded bamboo-like knock',
    'water_drop': 'synth: single drip (bubble with fast upward chirp + tiny splash)',
    'wind_chime': 'synth: modal free-free bar partials (1, 2.76, 5.40, 8.93), kw notes, strikes',
    'paper': 'synth: page turn (crackling band-passed noise swoosh + soft landing flap)',
    'seal_stamp': 'synth: seal pressed on paper (low contact thump + paper friction + ink release)',
    'bamboo_knock': 'synth: 竹梆 hollow bamboo knock (3 damped modes + strike click), kw pitch_hz',
    'bird_call': 'synth: one FM bird phrase',
    'chirp': 'REAL short bird call (<= 0.45 s) cut from the field recordings via sampler.birds '
             '(PD / CC0), denoised and tuned into the key; kw call_id, tune (default "D" = D E F# A B, '
             'None = as recorded), max_dur; seed picks among the clearest',
    'flutter': 'REAL wing flaps (Freesound, CC0): seed 0 = pigeon take-off, close, true stereo '
               '(Kinoton); 1 = pigeon take-off outside a bar (gerardcatala); 2.. = single wing '
               'bursts of a bird close to the mic (21100495); kw take_off=True forces seed 0, dur trims',
    'pluck': 'REAL pizzicato (VSCO 2 CE, CC0) via sampler.strings.pluck; kw note (default D3), '
             "vel (70), sec ('vc'; vn va vc cb vn_solo), ring_s",
    'bowl': 'porcelain bowl struck with a chopstick (real ceramic bowl, Freesound CC0, varispeed) '
            'via sampler.porcelain.bowl; kw note (default A5), vel (90), hardness (0.5), water (cents), '
            "water_shape, model ('rec' | 'modal'), source ('bowl' | 'plate' | 'cup' | 'mixed')",
}


# ----------------------------------------------------------------------------- loading
@lru_cache(maxsize=64)
def _load(path: str):
    log_access(path)
    x, sr = sf.read(need(path), dtype='float32', always_2d=True)
    if sr != SR:
        x = resample_sr(x, sr, SR)
    return x


def _field(name):
    p = FIELD / f'{name}.flac'
    return _load(str(p)) if p.exists() else None


def _lufs(x):
    import pyloudnorm as pyln
    x = stereo(x).astype(np.float64)
    if len(x) < SR // 2:
        x = np.pad(x, ((0, SR // 2 - len(x)), (0, 0)))
    return pyln.Meter(SR).integrated_loudness(x)


def _to_lufs(x, target):
    cur = _lufs(x)
    if not np.isfinite(cur):
        return x
    return (x * db2a(target - cur)).astype(np.float32)


def _peak_norm(x, peak_db=-6.0):
    pk = float(np.abs(x).max())
    return (x * db2a(peak_db) / pk).astype(np.float32) if pk > 0 else x


# ----------------------------------------------------------------------------- concatenation
def _env_db(x, win=0.1):
    from scipy.ndimage import uniform_filter1d
    m = np.abs(stereo(x)).mean(axis=1).astype(np.float64)
    e = np.sqrt(uniform_filter1d(m ** 2, max(1, int(win * SR)), mode='nearest')) + 1e-9
    return 20 * np.log10(e)


def _quiet_near(env, idx, radius):
    lo, hi = max(0, idx - radius), min(len(env) - 1, idx + radius)
    if hi <= lo:
        return idx
    return lo + int(np.argmin(env[lo:hi]))


def _concat(src, n_out, rng, seg_s, xf_s):
    """Random-segment overlap-add with equal-power crossfades at quiet moments."""
    n_src = len(src)
    xf = int(min(xf_s * SR, n_src // 4))
    env = _env_db(src)
    out = np.zeros((n_out + xf, src.shape[1]), dtype=np.float32)
    w_in = np.sin(0.5 * np.pi * (np.arange(xf) + 0.5) / xf).astype(np.float32)[:, None]
    w_out = w_in[::-1]
    pos, first, last = 0, True, -10 ** 9
    while pos < n_out:
        L = int(min(rng.uniform(*seg_s) * SR, n_src - 2 * xf - 1))
        L = max(L, xf + 1)
        for _ in range(8):     # avoid reusing the neighbourhood of the previous segment
            st = int(rng.integers(0, max(1, n_src - L - xf)))
            if abs(st - last) > L // 2:
                break
        st = _quiet_near(env, st + xf // 2, int(0.8 * SR)) - xf // 2
        st = int(np.clip(st, 0, n_src - L - xf))
        end = _quiet_near(env, st + L + xf // 2, int(0.8 * SR)) - xf // 2
        end = int(np.clip(end, st + xf + 1, n_src - xf))
        piece = src[st:end + xf].copy()
        if not first:
            piece[:xf] *= w_in
        piece[-xf:] *= w_out
        m = min(len(piece), len(out) - pos)
        out[pos:pos + m] += piece[:m]
        pos += len(piece) - xf
        first, last = False, st
    return out, xf


def _recorded_stream(srcs, n, rng, seg_s, xf_s, corr):
    """Stereo stream of length n (+xf tail) from one or more sources."""
    parts = []
    for s in srcs:
        if s.shape[1] == 2 and np.corrcoef(s[::7, 0], s[::7, 1])[0, 1] > 0.95:
            s = s.mean(axis=1, keepdims=True)      # near-mono 'stereo': decorrelate instead
        if s.shape[1] == 2:
            y, xf = _concat(s, n, rng, seg_s, xf_s)
        else:
            u, xf = _concat(s, n, rng, seg_s, xf_s)
            v, _ = _concat(s, n, rng, seg_s, xf_s)
            th = 0.5 * np.arcsin(np.clip(corr if corr is not None else 0.3, 0, 1))
            c, sn = np.cos(th), np.sin(th)
            y = np.concatenate([c * u + sn * v, sn * u + c * v], axis=1)
        parts.append(y)
    m = min(len(p) for p in parts)
    y = sum(p[:m] / np.sqrt(len(parts)) for p in parts).astype(np.float32)
    return y, xf


def _loopify(y, n, xf):
    """Wrap: the extra xf samples after n are crossfaded into the start."""
    out = y[:n].copy()
    if xf > 0 and len(y) >= n + xf:
        w = np.sin(0.5 * np.pi * (np.arange(xf) + 0.5) / xf)[:, None]
        out[:xf] = y[:xf] * w + y[n:n + xf] * np.sqrt(1 - w ** 2)
    return out


# ----------------------------------------------------------------------------- synthesis
def _bubble(f0, rng, rise=0.1, amp=1.0, dur_cap=0.08):
    """Minnaert bubble after van den Doel (2005): damped sine, damping
    d = 0.13 f + 0.0072 f^1.5, frequency rising with time (f = f0 (1 + rise*d*t))."""
    d = 0.13 * f0 + 0.0072 * f0 ** 1.5
    n = int(min(6.9 / d, dur_cap) * SR) + 1
    t = np.arange(n) / SR
    f = f0 * (1 + rise * d * t)
    ph = 2 * np.pi * np.cumsum(f) / SR
    att = 1 - np.exp(-t / 0.0003)        # 0.3 ms onset: no sample-level corner
    return (amp * np.sin(ph) * np.exp(-d * t) * att).astype(np.float32)


def _syn_stream(n, rng):
    dur = n / SR
    out = np.zeros((n, 2), dtype=np.float32)
    # noise bed: low-passed brownish noise with slow swells
    for ch in range(2):
        w = rng.standard_normal(n)
        bed = sosfilt(butter('lowpass', 900.0, 2), np.cumsum(w) * 0.02 + w * 0.3)
        bed = sosfilt(butter('highpass', 80.0, 2), bed)
        sw = 0.7 + 0.3 * np.tanh(smooth_noise(n, 0.4, rng))
        out[:, ch] += (bed * sw * 0.05).astype(np.float32)
        hiss = sosfilt(butter('bandpass', (2500.0, 9000.0), 2), rng.standard_normal(n))
        spark = np.abs(smooth_noise(n, 12.0, rng)) ** 2
        out[:, ch] += (hiss * spark * 0.004).astype(np.float32)
    # bubbles: Poisson onsets, rate modulated by 'gurgle' clusters
    rate = 220.0
    lam = rate * (0.5 + np.exp(0.8 * smooth_noise(int(dur * 50) + 2, 0.8, rng, sr=50)))
    lam = lam / lam.mean() * rate
    count = rng.poisson(rate * dur)
    cdf = np.cumsum(lam)
    times = np.interp(rng.random(count) * cdf[-1], cdf, np.arange(len(lam)) / 50.0)
    for t0 in times:
        f0 = float(np.exp(rng.uniform(np.log(450), np.log(3800))))
        a = 0.02 * (f0 / 1000) ** -0.6 * float(np.exp(rng.normal(0, 0.6)))
        b = _bubble(f0, rng, rise=rng.uniform(0.05, 0.25), amp=a)
        gl, gr = pan_gains(rng.uniform(-0.9, 0.9))
        i = int(t0 * SR)
        add_at(out, np.stack([b * gl, b * gr], axis=1), i)
    return out


def _tv_bandpass(x, fc, q):
    from .choir import tv_resonator
    bw = fc / q
    y = tv_resonator(x.astype(np.float64), fc, bw)
    return sosfilt(butter('highpass', 60.0, 2), y).astype(np.float64)


def _syn_wind(n, rng, bamboo=False):
    gust = smooth_noise(n, 0.12, rng)
    gust = np.exp(0.9 * gust)
    gust = gust / np.percentile(gust, 90)
    out = np.zeros((n, 2), dtype=np.float32)
    lag = int(0.25 * SR)
    for ch in range(2):
        g = np.roll(gust, lag * ch)
        pink = sosfilt(butter('lowpass', 1500.0, 1), rng.standard_normal(n)).astype(np.float64)
        fc = 250 + 650 * np.clip(g, 0, 2) + 60 * smooth_noise(n, 0.5, rng)
        whoosh = _tv_bandpass(pink, np.clip(fc, 120, 2500), 1.6)
        whoosh = whoosh / (np.std(whoosh) + 1e-12)
        out[:, ch] += (whoosh * g ** 1.5 * 0.03).astype(np.float32)
        # leaf rustle: dense tiny noise bursts whose rate follows the gusts
        lo, hi = (2500.0, 11000.0) if bamboo else (1200.0, 7000.0)
        hiss = sosfilt(butter('bandpass', (lo, hi), 2), rng.standard_normal(n)).astype(np.float64)
        k = int(0.004 * SR)
        imp = (rng.random(n) < (40 + 260 * np.clip(g, 0, 2)) / SR).astype(np.float64)
        imp *= np.exp(rng.normal(0, 0.7, n))
        kt = np.arange(k) / SR
        kern = (1 - np.exp(-kt / 0.0004)) * np.exp(-kt / 0.0012)
        crack = signal.fftconvolve(imp, kern)[:n]
        rust = hiss * (0.25 * g + crack) * (0.012 if bamboo else 0.008)
        out[:, ch] += rust.astype(np.float32)
    if bamboo:   # rare, soft culm knocks and creaks
        for t0 in np.cumsum(rng.exponential(7.0, int(n / SR / 3) + 3)):
            if t0 * SR >= n - SR:
                break
            k_ = _bamboo(rng, pitch_hz=rng.uniform(380, 700), soft=True) * 0.05
            gl, gr = pan_gains(rng.uniform(-0.8, 0.8))
            add_at(out, np.stack([k_[:, 0] * gl, k_[:, 1] * gr], axis=1), int(t0 * SR))
    return out


_BIRDS = [  # (f_lo, f_hi, kind, syllables, mean gap s)
    (2600, 4200, 'whistle', (2, 5), 4.0),
    (3500, 6500, 'trill', (6, 14), 6.0),
    (2000, 3200, 'warble', (3, 7), 5.0),
    (4500, 7500, 'chirp', (2, 6), 3.0),
]


def _bird_phrase(rng, spec):
    f_lo, f_hi, kind, syl, _ = spec
    n_syl = int(rng.integers(*syl))
    pieces = []
    base = rng.uniform(f_lo, f_hi)
    for i in range(n_syl):
        if kind == 'trill':
            d = rng.uniform(0.025, 0.045)
        elif kind == 'chirp':
            d = rng.uniform(0.03, 0.07)
        else:
            d = rng.uniform(0.08, 0.28)
        m = int(d * SR)
        u = np.linspace(0, 1, m)
        if kind == 'whistle':
            f = base * (1 + rng.uniform(-0.25, 0.25) * u)
        elif kind == 'trill':
            f = base * (1 + 0.15 * (1 - u)) * (1 + 0.02 * i)
        elif kind == 'warble':
            f = base * (1 + 0.12 * np.sin(2 * np.pi * rng.uniform(15, 30) * u * d)
                        + rng.uniform(-0.15, 0.15) * u)
        else:
            f = base * (0.7 + 0.6 * u ** 0.7)
        ph = 2 * np.pi * np.cumsum(f) / SR
        env = np.sin(np.pi * u) ** (0.6 if kind != 'trill' else 1.5)
        s = (np.sin(ph) + 0.12 * np.sin(2 * ph)) * env
        pieces.append(s.astype(np.float32))
        gap = rng.uniform(0.01, 0.03) if kind == 'trill' else rng.uniform(0.04, 0.16)
        pieces.append(np.zeros(int(gap * SR), np.float32))
    y = np.concatenate(pieces)
    return fade_edges(y, 0.002, 0.005)


def _syn_birds(n, rng):
    out = np.zeros((n, 2), dtype=np.float32)
    for bi in range(6):
        spec = _BIRDS[bi % len(_BIRDS)]
        pan = rng.uniform(-0.9, 0.9)
        dist = rng.uniform(0.0, 0.8)
        g = db2a(-18 - 16 * dist)
        t = rng.uniform(0, spec[4])
        while t * SR < n:
            ph = _bird_phrase(rng, spec)
            ph = sosfilt(butter('lowpass', 12000 * (1 - 0.6 * dist), 2), ph)
            gl, gr = pan_gains(pan)
            add_at(out, np.stack([ph * gl * g, ph * gr * g], axis=1), int(t * SR))
            t += len(ph) / SR + rng.exponential(spec[4])
    return out


def _syn_insects(n, rng):
    out = np.zeros((n, 2), dtype=np.float32)
    t = np.arange(n) / SR
    for i in range(5):                          # field crickets: chirps of 3-5 pulses
        fc = rng.uniform(4200, 5200)
        pulse_rate = rng.uniform(26, 34)
        chirp_rate = rng.uniform(1.6, 3.2)
        npul = int(rng.integers(3, 6))
        cph = np.mod(t * chirp_rate + rng.random(), 1.0) / chirp_rate   # time within chirp
        pidx = np.floor(cph * pulse_rate)
        pfrac = cph * pulse_rate - pidx
        on = (pidx < npul) & (pfrac < 0.6)
        env = np.where(on, np.sin(np.pi * np.clip(pfrac / 0.6, 0, 1)) ** 2, 0.0)
        # each pulse glides down ~2 %: integrate the instantaneous frequency (no t*f(t) bug)
        ph = 2 * np.pi * np.cumsum(fc * (1 - 0.02 * pfrac)) / SR
        car = np.sin(ph) + 0.18 * np.sin(2 * ph)
        dist = rng.uniform(0.2, 1.0)
        s = car * env * db2a(-24 - 14 * dist) * (1 + 0.1 * smooth_noise(n, 0.3, rng))
        gl, gr = pan_gains(rng.uniform(-0.9, 0.9))
        out += np.stack([s * gl, s * gr], axis=1).astype(np.float32)
    # a distant tree-cricket trill (continuous, ~50 Hz pulses)
    fc = rng.uniform(2600, 3200)
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 48 * t)) ** 3
    s = np.sin(2 * np.pi * fc * t) * env * db2a(-36) * (0.7 + 0.3 * np.tanh(smooth_noise(n, 0.1, rng)))
    out += np.stack([s * 0.8, s], axis=1).astype(np.float32)
    return sosfilt(butter('highpass', 1500.0, 2), out)


def _crickets_recorded(n, rng):
    src = _field('cricket_gryllus_campestris')
    if src is None:
        return None
    out = np.zeros((n, 2), dtype=np.float32)
    for i in range(4):
        ratio = rng.uniform(0.94, 1.06)
        need = int(n * ratio) + SR
        y, _ = _concat(src, need, rng, (6, 12), 2.0)
        y = resample_by(y[:, 0], ratio)[:n]
        dist = [0.15, 0.4, 0.6, 0.85][i]
        y = sosfilt(butter('lowpass', 16000 * (1 - dist) ** 1.3 + 3500 * dist, 2), y)
        y = sosfilt(butter('highpass', 900.0, 2), y)
        gl, gr = pan_gains(rng.uniform(-0.85, 0.85))
        g = db2a(-10 * dist)
        m = min(n, len(y))
        out[:m, 0] += y[:m] * gl * g
        out[:m, 1] += y[:m] * gr * g
    return out


# ----------------------------------------------------------------------------- public: ambience
def ambience(kind: str, duration: float, seed: int = 0, loop: bool = False,
             level_lufs: float = -30.0) -> np.ndarray:
    """Ambient bed of any length (s). Output is normalised to `level_lufs` integrated
    loudness (default -30 LUFS: a bed, not a foreground). loop=True -> seamless tiling
    (no edge fades); otherwise 10 ms edge fades. See KINDS_AMBIENCE."""
    rng = np.random.default_rng(seed)
    n = int(round(duration * SR))
    xf = 0
    if kind in _REC_AMB and kind != 'crickets':
        files, seg, xfs, corr, _ = _REC_AMB[kind]
        srcs = [s for s in (_field(f) for f in files) if s is not None]
        if not srcs:
            raise FileNotFoundError(f'no recording for {kind!r} in {FIELD}')
        y, xf = _recorded_stream(srcs, n, rng, seg, xfs, corr)
        y = _loopify(y, n, xf) if loop else y[:n]
    else:
        m = n + (int(3 * SR) if loop else 0)
        if kind == 'crickets':
            y = _crickets_recorded(m, rng)
            if y is None:
                raise FileNotFoundError('cricket recording missing')
        elif kind == 'night_insects':
            y = (_crickets_recorded(m, rng) if _field('cricket_gryllus_campestris') is not None
                 else np.zeros((m, 2), np.float32))
            y = y + _to_lufs(_syn_insects(m, rng), _lufs(y) - 8 if np.abs(y).max() > 0 else -40)
        elif kind == 'stream_synth':
            y = _syn_stream(m, rng)
        elif kind == 'wind_synth':
            y = _syn_wind(m, rng)
        elif kind == 'bamboo_wind_synth':
            y = _syn_wind(m, rng, bamboo=True)
        elif kind == 'birds_synth':
            y = _syn_birds(m, rng)
        elif kind == 'insects_synth':
            y = _syn_insects(m, rng)
        else:
            raise ValueError(f'unknown ambience {kind!r}; one of {sorted(KINDS_AMBIENCE)}')
        if loop:
            y = _loopify(y, n, m - n)
        else:
            y = y[:n]
    y = stereo(y).astype(np.float64)
    y = signal.sosfilt(butter('highpass', 20.0, 2), y, axis=0).astype(np.float32)   # DC block
    y = _to_lufs(y, level_lufs)
    return y if loop else fade_edges(y, 0.01, 0.01)


# ----------------------------------------------------------------------------- synth sfx
def _modal(freqs, amps, t60s, dur, rng, strike_ms=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for f, a, T in zip(freqs, amps, t60s):
        if f >= SR * 0.45:
            continue
        for det in (-0.4, 0.4):                 # slight beating between the two polarisations
            y += a * 0.5 * np.sin(2 * np.pi * (f + det) * t + rng.uniform(0, 2 * np.pi)) * np.exp(-6.9 * t / T)
    k = max(2, int(strike_ms / 1000 * SR))
    click = rng.standard_normal(k) * np.hanning(k) * 0.3
    y[:k] += click
    att = int(0.0015 * SR)
    y[:att] *= cos_ramp(att)
    return y


def _bamboo(rng, pitch_hz=None, soft=False):
    f0 = pitch_hz or rng.uniform(600, 950)
    ratios = np.array([1.0, 2.32, 3.95]) * (1 + rng.normal(0, 0.01, 3))
    y = _modal(f0 * ratios, [1.0, 0.45, 0.2], [0.18, 0.09, 0.05], 0.4, rng, strike_ms=1.5)
    hit = sosfilt(butter('bandpass', (1500.0, 6000.0), 2), rng.standard_normal(int(0.004 * SR)))
    y[:len(hit)] += hit * (0.15 if soft else 0.35)
    y = sosfilt(butter('highpass', 200.0, 2), y)
    return np.stack([y, y], axis=1).astype(np.float32)


def _water_drop(rng):
    f0 = rng.uniform(900, 2200)
    b = _bubble(f0, rng, rise=rng.uniform(0.3, 0.8), amp=1.0, dur_cap=0.15)
    y = np.zeros(int(0.35 * SR))
    splash = sosfilt(butter('highpass', 3000.0, 2), rng.standard_normal(int(0.0015 * SR)))
    y[:len(splash)] += splash * 0.2
    s = int(0.002 * SR)
    y[s:s + len(b)] += b
    for _ in range(int(rng.integers(0, 3))):   # a couple of tiny secondary bubbles
        b2 = _bubble(rng.uniform(2500, 5000), rng, rise=0.2, amp=rng.uniform(0.05, 0.2))
        i = int(rng.uniform(0.01, 0.08) * SR)
        y[i:i + len(b2)] += b2[:len(y) - i]
    gl, gr = pan_gains(rng.uniform(-0.2, 0.2))
    return np.stack([y * gl, y * gr], axis=1).astype(np.float32)


def _wind_chime(rng, notes=None, strikes=None, dur=6.0):
    notes = notes or [74, 76, 78, 81, 83]           # D5 E5 F#5 A5 B5 pentatonic
    strikes = strikes or int(rng.integers(4, 9))
    out = np.zeros((int(dur * SR), 2), dtype=np.float32)
    pans = np.linspace(-0.6, 0.6, len(notes))
    t = 0.0
    for _ in range(strikes):
        i = int(rng.integers(len(notes)))
        f = float(midi_to_hz(notes[i]))
        ratios = np.array([1.0, 2.756, 5.404, 8.933])
        T = np.array([3.5, 2.0, 1.1, 0.6]) * (880 / f) ** 0.3
        a = np.array([1.0, 0.55, 0.3, 0.15]) * rng.uniform(0.6, 1.0, 4)
        y = _modal(f * ratios, a, T, dur - t, rng) * rng.uniform(0.3, 1.0)
        gl, gr = pan_gains(pans[i])
        add_at(out, np.stack([y * gl, y * gr], axis=1).astype(np.float32), int(t * SR))
        t += rng.exponential(0.35)
        if t > dur * 0.5:
            break
    return out


def _paper(rng):
    d = rng.uniform(0.5, 0.85)
    n = int((d + 0.25) * SR)
    t = np.arange(n) / SR
    env = np.clip(t / d, 0, 1)
    env = np.sin(np.pi * env) ** 1.5 * (t < d)
    hiss = sosfilt(butter('bandpass', (900.0, 9000.0), 2), rng.standard_normal(n)).astype(np.float64)
    imp = (rng.random(n) < 900 / SR) * np.exp(rng.normal(0, 0.8, n))
    kt = np.arange(int(0.003 * SR)) / SR
    crack = signal.fftconvolve(imp, (1 - np.exp(-kt / 0.0003)) * np.exp(-kt / 0.0008))[:n]
    # stereo: same envelope and crackle events, independent hiss per channel (no comb
    # filtering in mono, unlike a delayed copy)
    hiss2 = sosfilt(butter('bandpass', (900.0, 9000.0), 2), rng.standard_normal(n)).astype(np.float64)
    k = int(0.06 * SR)
    flap = sosfilt(butter('lowpass', 500.0, 2), rng.standard_normal(k)) * np.exp(-np.arange(k) / (0.012 * SR))
    i = int(d * SR)
    chans = []
    for hs in (hiss, 0.6 * hiss + 0.8 * hiss2):
        y = hs * env * (0.3 + crack) * 0.3
        y[i:i + k] += flap * 0.8                       # landing flap
        chans.append(sosfilt(butter('highpass', 90.0, 2), y))
    return np.stack(chans, axis=1).astype(np.float32)


def _seal_stamp(rng):
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    thump = (np.sin(2 * np.pi * rng.uniform(85, 120) * t) * np.exp(-t / 0.05)
             + 0.4 * np.sin(2 * np.pi * rng.uniform(210, 260) * t) * np.exp(-t / 0.03))
    body = sosfilt(butter('lowpass', 700.0, 2), rng.standard_normal(n)) * np.exp(-t / 0.02) * 0.6
    y = thump + body
    att = int(0.002 * SR)
    y[:att] *= cos_ramp(att)
    # paper friction while the seal is rocked
    f0, f1 = int(0.03 * SR), int(0.22 * SR)
    fr = sosfilt(butter('bandpass', (900.0, 5000.0), 2), rng.standard_normal(f1 - f0))
    fr *= np.sin(np.linspace(0, np.pi, f1 - f0)) ** 2 * 0.08
    y[f0:f1] += fr
    # ink release 'tack' when lifting
    i = int(rng.uniform(0.45, 0.6) * SR)
    k = int(0.006 * SR)
    tack = sosfilt(butter('highpass', 2000.0, 2), rng.standard_normal(k)) * np.hanning(k) * 0.05
    y[i:i + k] += tack
    y = sosfilt(butter('highpass', 40.0, 2), y)
    return np.stack([y, y * 0.97], axis=1).astype(np.float32)


# ----------------------------------------------------------------------------- recorded sfx
_NOTE_RE = re.compile(r'([A-G]#?)(-?\d)')


def _note_num(s):
    from .sfz import note_number
    return note_number(s)


def _trim(x, thr_db=-45.0, pre=0.01, tail_db=-65.0):
    e = np.abs(stereo(x)).max(axis=1)
    pk = e.max()
    if pk <= 0:
        return x
    on = int(np.argmax(e > pk * db2a(thr_db)))
    on = max(0, on - int(pre * SR))
    above = np.flatnonzero(e > pk * db2a(tail_db))
    end = min(len(x), int(above[-1]) + int(0.02 * SR)) if len(above) else len(x)
    return fade_edges(x[on:end], 0.002, 0.03)


def _vcsl(sub, pattern):
    d = need(VCSL / sub, f'VCSL {sub}')
    files = sorted(p for p in d.glob('*.wav') if re.search(pattern, p.name))
    return files


def _pick_pitched(sub, note, extra='', name_offset=0):
    """Nearest sample by pitch, then shifted by resampling. name_offset converts the
    file's note name to the pitch actually sounding (hand chimes: +12, measured)."""
    files = _vcsl(sub, extra)
    best, bd = None, None
    for p in files:
        m = _NOTE_RE.search(p.name)
        if not m:
            continue
        k = _note_num(m.group(0)) + name_offset
        if bd is None or abs(k - note) < bd:
            best, bd, bk = p, abs(k - note), k
    if best is None:
        raise FileNotFoundError(sub)
    x = _load(str(best))
    if bk != note:
        x = resample_by(x, 2 ** ((note - bk) / 12))
    return x, best.name


def sfx(kind: str, seed: int = 0, normalize: bool = True, **kw) -> np.ndarray:
    """One-shot sound. Peak-normalised to -6 dBFS unless normalize=False. See KINDS_SFX."""
    rng = np.random.default_rng(seed)
    if kind == 'rooster':
        names = ['rooster_chardonnet', 'rooster_alys_medium', 'rooster_alys_small']
        src = [n for n in names if _field(n) is not None]
        name = src[seed % len(src)]
        x = _trim(stereo(_field(name)))
        if name != 'rooster_chardonnet':      # farm background starts under the crow: soft fade-in
            x = fade_edges(x, 0.04, 0.03)
    elif kind == 'dog':
        edo, george = _field('dog_bark_edo'), _field('dog_bark_george')
        barks = int(kw.get('barks') or 1 + (seed // 2) % 2)
        which = kw.get('source') or ('edo' if seed % 2 == 0 else 'george')
        if kw.get('synth') or (edo is None and george is None):
            x = _syn_dog(rng, barks)
        elif (which == 'george' and george is not None) or edo is None:
            one = _trim(stereo(george))
            parts = []
            for i in range(barks):     # repeat with small natural variation
                v = resample_by(one, float(rng.uniform(0.97, 1.03))) * rng.uniform(0.7, 1.0)
                parts += [v, np.zeros((int(rng.uniform(0.25, 0.45) * SR), 2), np.float32)]
            x = np.concatenate(parts[:-1])
        else:
            x = _dog_barks(stereo(edo), barks, seed // 2)
    elif kind == 'woodblock':
        vel = kw.get('vel', 'mp')
        files = _vcsl('Woodblock', rf'wood_click_{vel}') or _vcsl('Woodblock', r'wood_click')
        x = _trim(stereo(_load(str(files[seed % len(files)]))))
    elif kind == 'hand_chime':
        note = kw.get('note', 74)
        note = _note_num(note) if isinstance(note, str) else int(note)
        x, _ = _pick_pitched('Hand Chimes', note, name_offset=12)
        x = _trim(stereo(x), tail_db=-70)
    elif kind == 'tubular_bell':
        note = kw.get('note', 62)
        note = _note_num(note) if isinstance(note, str) else int(note)
        dyn = kw.get('vel', 'p')
        pat = r'_(pp|p)_' if dyn in ('p', 'pp') else r'_(f|ff|fff)_'
        x, _ = _pick_pitched('Tubular Bells 1', note, pat)
        x = _trim(stereo(x), tail_db=-70)
    elif kind in ('gong', 'gong_scrape'):
        if kind == 'gong_scrape':
            files = _vcsl('Gong 1', r'scrape')
        else:
            vel = kw.get('vel', 'mf')
            files = _vcsl('Gong 1', rf'^gong_{vel}\.wav$') or _vcsl('Gong 1', r'^gong_mf')
        x = _trim(stereo(_load(str(files[0]))), tail_db=-70)
    elif kind == 'log_drum':
        pitch = kw.get('pitch', 'Hi').capitalize()
        vel = int(kw.get('vel', 2))
        files = _vcsl('Slit Drum', rf'LogDrum{pitch}_MedM_v{vel}')
        x = _trim(stereo(_load(str(files[0]))))
    elif kind == 'water_drop':
        x = _water_drop(rng)
    elif kind == 'wind_chime':
        x = _wind_chime(rng, kw.get('notes'), kw.get('strikes'), kw.get('dur', 6.0))
    elif kind == 'paper':
        x = _paper(rng)
    elif kind == 'seal_stamp':
        x = _seal_stamp(rng)
    elif kind == 'bamboo_knock':
        x = _bamboo(rng, kw.get('pitch_hz'))
    elif kind == 'bird_call':
        spec = _BIRDS[int(kw.get('species', seed)) % len(_BIRDS)]
        y = _bird_phrase(rng, spec)
        x = np.stack([y, y], axis=1)
    elif kind == 'chirp':
        from . import birds
        cid = kw.get('call_id')
        if cid is None:
            cid = birds.pick_calls(1, max_dur=kw.get('max_dur', 0.45), min_dur=0.06, seed=seed)[0]
        x = birds.bird_call(cid, tune=kw.get('tune', 'D'))
    elif kind == 'flutter':
        x = _flutter(0 if kw.get('take_off') else seed, kw.get('dur'))
    elif kind == 'pluck':
        from .strings import pluck
        note = kw.get('note', 50)
        note = _note_num(note) if isinstance(note, str) else note
        x = pluck(note, kw.get('vel', 70), kw.get('sec', 'vc'), seed=seed, ring_s=kw.get('ring_s', 0.8))
    elif kind == 'bowl':
        from .porcelain import bowl
        x = bowl(kw.get('note', 81), kw.get('vel', 90), kw.get('hardness', 0.5),
                 water=kw.get('water', 0.0), water_shape=kw.get('water_shape', 'bend'),
                 model=kw.get('model', 'rec'), source=kw.get('source', 'bowl'), seed=seed)
    else:
        raise ValueError(f'unknown sfx {kind!r}; one of {sorted(KINDS_SFX)}')
    x = stereo(x).astype(np.float64)
    x = signal.sosfiltfilt(butter('highpass', 20.0, 2), x, axis=0)   # DC (VCSL bells carry some)
    x = fade_edges(x.astype(np.float32), 0.001, 0.01)
    return _peak_norm(x) if normalize else x


OBJECTS = SAMPLES / 'objects'
_WINGS = [('wings_pigeon_kinoton', 0.50, 2.40), ('wings_pigeon_gerardcatala', 0.17, 0.80)]
_WING_BURSTS = ('wings_bird_21100495', [(0.22, 0.56), (3.28, 3.66), (5.25, 5.56), (5.83, 6.47),
                                       (8.90, 9.28), (10.10, 10.80)])


def _flutter(seed, dur=None):
    """Real wing flaps: seed 0/1 = the two pigeon take-offs, 2.. = the six bursts of the close bird
    (cut points: bursts >= 0.15 s at 30 dB under the file's peak, measured)."""
    items = _WINGS + [(_WING_BURSTS[0], a, b) for a, b in _WING_BURSTS[1]]
    name, a, b = items[seed % len(items)]
    p = OBJECTS / f'{name}.ogg'
    x = _load(str(p))
    x = stereo(x[int(a * SR):int(b * SR)]).astype(np.float64)
    x = signal.sosfiltfilt(butter('highpass', 150.0, 2), x, axis=0)   # wind / handling rumble
    if dur is not None and len(x) > int(dur * SR):
        x = x[:int(dur * SR)]
        x = fade_edges(x.astype(np.float32), 0.0, min(0.15, dur / 3))
    return fade_edges(x.astype(np.float32), 0.01, 0.08)


def _dog_barks(src, barks, seed):
    """Cut `barks` consecutive barks out of the recording (onsets from the envelope)."""
    e = _env_db(src, 0.01)
    on = e > e.max() - 25
    starts = np.flatnonzero(np.diff(on.astype(int)) == 1)
    ev = []
    for st in starts:
        if not ev or st - ev[-1] > int(0.25 * SR):
            ev.append(st)
    barks = int(np.clip(barks, 1, len(ev)))
    first = seed % max(1, len(ev) - barks + 1)
    a = max(0, ev[first] - int(0.02 * SR))
    b = ev[first + barks] - int(0.05 * SR) if first + barks < len(ev) else len(src)
    return _trim(src[a:b], thr_db=-40, tail_db=-60)


def _syn_dog(rng, barks=None):
    """Synthetic bark (fallback when no licensed dog recording is present).

    Harsh voiced source (band-limited pulse train with 4 % jitter and period-doubling
    subharmonic) + turbulence, rise-fall pitch contour (~450 -> 650 -> 380 Hz), fixed
    'muzzle' formants at 900 / 1800 / 3000 Hz, 5 ms attack and ~0.1 s decay.
    Plausible only when far away / quiet; use with distant()."""
    from .choir import tv_resonator
    barks = barks or int(rng.integers(1, 3))
    out = []
    for _ in range(barks):
        d = rng.uniform(0.14, 0.22)
        n = int(d * SR)
        t = np.arange(n) / SR
        u = t / d
        f0 = rng.uniform(380, 520) * (1 + 0.45 * np.sin(np.pi * np.clip(u / 0.7, 0, 1)) - 0.25 * u)
        f0 *= 1 + 0.04 * rng.standard_normal(n // 64 + 1).repeat(64)[:n]
        ph = np.cumsum(f0) / SR
        src = np.zeros(n)
        for k in range(1, int(9000 / f0.max())):        # band-limited pulse train
            src += np.cos(2 * np.pi * k * ph) / k ** 0.6
        src += 0.5 * np.cos(np.pi * ph)                  # period doubling (roughness)
        src = src / np.abs(src).max() + 0.35 * rng.standard_normal(n)
        env = (1 - np.exp(-t / 0.005)) * np.exp(-np.maximum(t - 0.03, 0) / 0.07)
        y = src * env
        for f, bw in ((900, 180), (1800, 260), (3000, 400)):
            y = tv_resonator(y, np.full(n, float(f)), np.full(n, float(bw))) * (f / 900) ** 0.5
        y = fade_edges(y.astype(np.float32), 0.001, 0.01)
        out.append(y)
        out.append(np.zeros(int(rng.uniform(0.18, 0.35) * SR), np.float32))
    y = np.concatenate(out)
    y = sosfilt(butter('highpass', 250.0, 2), y)
    return np.stack([y, y], axis=1).astype(np.float32)


# ----------------------------------------------------------------------------- distance
def distant(x, amount: float = 0.7, reverb_kind: str = 'valley', predelay_s=None,
            seed: int = 0) -> np.ndarray:
    """Place a sound far away. amount 0 (near, unchanged) .. 1 (very far, very quiet).

    * air absorption: low-pass from 16 kHz (0) down to ~1.8 kHz (1), plus a gentle
      high-pass (60 -> 200 Hz: small, far sources lose their low end),
    * direct sound drops by 4 + 22*amount dB and narrows towards a point source,
    * reverb (default 'valley') gets relatively louder: wet at -(14 + 8*amount) dB,
    * pre-delay: extra reverb pre-delay (default 0.03*amount s) - the output still
      starts at the original time, so sync points do not move.
    Returned array is longer by the reverb tail.
    """
    from .mix import reverb, width
    a = float(np.clip(amount, 0.0, 1.0))
    x = stereo(x)
    if a <= 0:
        return x
    fc = 16000 * (1 - a) ** 1.5 + 1800 * a
    d = sosfilt(butter('lowpass', fc, 2), x)
    d = sosfilt(butter('highpass', 60 + 140 * a, 2), d)
    d = width(d, 1 - 0.85 * a)
    pre = 0.03 * a if predelay_s is None else predelay_s
    wet = reverb(d, reverb_kind, mix=1.0, seed=seed, predelay_s=pre)
    out = wet * np.float32(db2a(-(14 + 8 * a)))
    out[:len(d)] += d * np.float32(db2a(-(4 + 22 * a)))
    return fade_edges(out, 0.0, 0.05)
