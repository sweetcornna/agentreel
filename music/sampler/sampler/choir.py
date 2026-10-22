"""Formant-synthesis female voice / small choir (pure DSP, no samples, no models).

    render_choir(notes, voices=6, vowel='a', breath=0.15, width=0.8, solo=False, ...)

notes = [(start_s, dur_s, midi, velocity, vowel_or_None[, part]), ...]

Signal chain per singer (see DESIGN.md §2.2):

  LF glottal-flow-derivative source (Fant 1995, Rd parameterisation), rendered by
  additive synthesis from its exact Fourier coefficients -> strictly band-limited
  (every harmonic fades out before 16 kHz, so nothing can alias)
  + per-cycle jitter / shimmer, slow pitch drift, vibrato, onset scoop, phrase-end fall
  + aspiration noise gated by the glottal open phase
  -> cascade of 5 time-varying formant resonators (Klatt-style, unity DC gain),
     vowel interpolation, F1/F2 tuning to f0 for high notes, nasal zero + lip
     low-pass for the hum 'm'
  -> constant-power pan into the stereo field.

The time-varying resonators are evaluated without a Python per-sample loop: each is
a complex one-pole y[n] = p[n] y[n-1] + x[n] solved in closed form per block
(prefix products / prefix sums) with a scalar carry between blocks.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import signal
from scipy.optimize import brentq

from ._dsp import SR, add_at, biquad, butter, db2a, fade_edges, pan_gains, smooth_noise, sosfilt

# --------------------------------------------------------------------------- vowel tables
# Formant frequencies / bandwidths (Hz) after the Csound manual's "Formant Values"
# appendix (soprano / alto). The dB amplitudes of that table are NOT used: with a
# cascade synthesiser the relative formant levels follow from F and BW (checked in
# selfcheck.py against the table).
VOWELS = {
    'soprano': {
        'a': ([800, 1150, 2900, 3900, 4950], [80, 90, 120, 130, 140]),
        'e': ([350, 2000, 2800, 3600, 4950], [60, 100, 120, 150, 200]),
        'i': ([270, 2140, 2950, 3900, 4950], [60, 90, 100, 120, 120]),
        'o': ([450, 800, 2830, 3800, 4950], [70, 80, 100, 130, 135]),
        'u': ([325, 700, 2700, 3800, 4950], [50, 60, 170, 180, 200]),
        # closed-mouth hum: nasal murmur (low pole) + weak higher resonances
        'm': ([280, 1250, 2300, 3100, 4200], [70, 220, 300, 380, 460]),
    },
    'alto': {
        'a': ([800, 1150, 2800, 3500, 4950], [80, 90, 120, 130, 140]),
        'e': ([400, 1600, 2700, 3300, 4950], [60, 80, 120, 150, 200]),
        'i': ([350, 1700, 2700, 3700, 4950], [50, 100, 120, 150, 200]),
        'o': ([450, 800, 2830, 3500, 4950], [70, 80, 100, 130, 135]),
        'u': ([325, 700, 2530, 3500, 4950], [50, 60, 170, 180, 200]),
        'm': ([260, 1150, 2200, 3000, 4100], [70, 220, 300, 380, 460]),
    },
}
# nasal anti-resonance of /m/ (bilabial nasal zero) and lip low-pass, blended in by
# the 'nasality' of the current vowel (1 for 'm', 0 for open vowels)
NASAL_ZERO = (1000.0, 180.0)
NASAL_LP = 1600.0
ALIASES = {'ah': 'a', 'aa': 'a', 'oh': 'o', 'oo': 'u', 'ee': 'i', 'eh': 'e', 'mm': 'm',
           'hum': 'm', 'n': 'm', '嗯': 'm', '啊': 'a', '哦': 'o', '噢': 'o', '呜': 'u', '乌': 'u'}
VOWEL_NAMES = sorted(set(VOWELS['soprano']) | set(ALIASES))

F_CUT = 16000.0      # every harmonic is faded out between F_CUT - F_TAPER and F_CUT
F_TAPER = 3000.0
# Higher-pole correction: a 5-pole cascade drops -12 dB/oct per pole above F5, far
# darker than a real tract. The missing poles of a ~14.5 cm tract (F6..F8) are added
# as fixed wide resonators (Fant's higher-pole correction, truncated).
HIGHER_POLES = ((6400.0, 500.0), (7600.0, 650.0), (8800.0, 800.0))
# Spectral tilt correction after the tract. Measured without it, the formant-region
# harmonics at F3-F5 came out 10-30 dB under the Csound soprano table (selfcheck
# prints the comparison); a broad +TILT_DB shelf from ~2 kHz closes most of that gap.
TILT = (2200.0, 0.6)
OUT_CAL = 0.58        # output calibration (see selfcheck: 1 singer, 'a', A4, vel 100 ~ -18 dBFS RMS)

# Spectrum model v2 (round 2). The 3 fixed poles above left a -96 dB/oct cliff above
# 9 kHz (measured: no harmonics above 8 kHz) and o/u 20-26 dB under the Csound table at
# F3-F5. v2: (1) a ladder of 15 wide higher poles from 5.9 to 20.6 kHz, spaced ~1 kHz
# like the formants of a ~16 cm tube, so the tract response falls gently (a: about
# -4 dB at 9 kHz, -18 dB at 13 kHz re. DC) instead of being truncated; with it the old
# +10 dB tilt shelf is no longer needed (tilt 0); (2) per-vowel parallel high-pass boost
# y += k * HP_2.5k(y) for the back vowels, sized to bring o/u F3-F5 to within ~7-8 dB of
# the Csound table (not all the way: brightness there is what makes a synthetic choir
# harsh); (3) the 'air' noise goes through resonators at the phrase's own F3-F5 plus a soft
# 7-14 kHz top instead of a flat 2.5-11 kHz band. Chosen by measurement (DESIGN_R2.md).
HIGHER_POLES_V2 = tuple((5900.0 + 1050.0 * i, 250.0 + 0.05 * (5900.0 + 1050.0 * i)) for i in range(15))
HF_FIX_HP = 2500.0
SPECTRA = {
    'v1': dict(poles=HIGHER_POLES, tilt_db=10.0, hf_fix={}, air='flat', air_gain=0.08, top_db=0.0),
    'v2': dict(poles=HIGHER_POLES_V2, tilt_db=0.0, hf_fix={'o': 3.0, 'u': 1.5}, air='formant',
               air_gain=0.08, top_db=0.0),
}
TOP_SHELF = (7500.0, 0.7)   # v2 'top_db' shelf: harmonics above 8 kHz (radiation / open tube)


# --------------------------------------------------------------------------- LF source
@lru_cache(maxsize=64)
def lf_coeffs(rd: float, kmax: int = 160, m: int = 8192):
    """Complex Fourier coefficients c_1..c_kmax of one period of the LF glottal
    flow derivative for shape parameter Rd (T0 = 1, Ee = 1), normalised to unit
    power over the returned harmonics. Also returns Te (open-phase end)."""
    rd = float(np.clip(rd, 0.3, 2.7))
    ra = (-1.0 + 4.8 * rd) / 100.0
    rk = (22.4 + 11.8 * rd) / 100.0
    rg = rk / (4.0 * (0.11 * rd / (0.5 + 1.2 * rk) - ra))
    tp = 1.0 / (2.0 * rg)
    te = tp * (1.0 + rk)
    ta = ra
    tc = 1.0
    # return phase constant: eps*Ta = 1 - exp(-eps (Tc - Te))
    eps = 1.0 / ta
    for _ in range(200):
        eps = (1.0 - np.exp(-eps * (tc - te))) / ta
    wg = np.pi / tp
    t = (np.arange(m) + 0.5) / m
    ret = -(1.0 / (eps * ta)) * (np.exp(-eps * (t - te)) - np.exp(-eps * (tc - te)))

    def wave(alpha):
        e0 = -1.0 / (np.exp(alpha * te) * np.sin(wg * te))
        op = e0 * np.exp(alpha * t) * np.sin(wg * t)
        return np.where(t <= te, op, ret)

    def area(alpha):
        return wave(alpha).mean()

    lo, hi = -5.0, 5.0
    while area(lo) * area(hi) > 0 and hi < 400:
        lo, hi = lo - 5.0, hi * 1.6
    alpha = brentq(area, lo, hi, xtol=1e-10)
    e = wave(alpha)
    e -= e.mean()
    c = np.fft.rfft(e)[1:kmax + 1] / m
    c = c / np.sqrt(2.0 * np.sum(np.abs(c) ** 2))
    return c.astype(np.complex128), float(te)


def _harmonic_source(phase, f0, beta, ca, cb, amp):
    """Band-limited sum of harmonics: sum_k 2 Re(c_k(t) w_k(t) e^{i k phase}).

    c_k(t) = ca_k + beta(t) (cb_k - ca_k) (blend of two LF shapes), w_k(t) is the
    anti-alias taper for harmonic k at the instantaneous f0. Horner scheme, complex64.
    """
    n = len(phase)
    kmax = int(np.ceil(F_CUT / max(float(f0.min()), 20.0)))
    mag = np.maximum(np.abs(ca), np.abs(cb))
    sig = np.flatnonzero(mag > 1e-4 * mag.max())
    kmax = int(min(kmax, (sig[-1] + 1) if len(sig) else 1, len(ca)))
    z = np.exp(1j * np.mod(phase, 2 * np.pi)).astype(np.complex64)
    f0_32 = f0.astype(np.float32)
    beta = beta.astype(np.float32)
    acc = np.zeros(n, dtype=np.complex64)
    inv_taper = np.float32(1.0 / F_TAPER)
    for k in range(kmax, 0, -1):
        w = np.clip((F_CUT - k * f0_32) * inv_taper, 0.0, 1.0)
        a = np.complex64(ca[k - 1]) + beta * np.complex64(cb[k - 1] - ca[k - 1])
        acc = acc * z + a * w
    acc = acc * z
    return 2.0 * acc.real.astype(np.float64) * amp


# --------------------------------------------------------------------------- filters
def tv_onepole(x, p):
    """y[n] = p[n] y[n-1] + x[n] for complex time-varying p, vectorised by blocks."""
    n = len(x)
    rmin = float(np.abs(p).min())
    # keep |prod p| over a block >= 1e-6 so the closed form stays well conditioned
    block = int(np.clip(np.log(1e-6) / np.log(max(min(rmin, 0.999999), 1e-3)), 16, 512))
    nb = -(-n // block)
    pad = nb * block - n
    X = np.pad(np.asarray(x, dtype=np.complex128), (0, pad)).reshape(nb, block)
    P = np.pad(np.asarray(p, dtype=np.complex128), (0, pad), constant_values=1.0).reshape(nb, block)
    C = np.cumprod(P, axis=1)
    Y = C * np.cumsum(X / C, axis=1)
    last = Y[:, -1].copy()
    cl = C[:, -1]
    carry = np.empty(nb, dtype=np.complex128)
    c = 0.0 + 0.0j
    for b in range(nb):           # scalar recurrence over blocks only
        carry[b] = c
        c = last[b] + cl[b] * c
    Y += C * carry[:, None]
    return Y.reshape(-1)[:n]


def tv_resonator(x, freq, bw):
    """Unity-DC-gain 2-pole resonator with per-sample centre frequency / bandwidth."""
    r = np.exp(-np.pi * bw / SR)
    th = 2 * np.pi * freq / SR
    p = r * np.exp(1j * th)
    y = tv_onepole(x, p)
    g = np.abs(1 - p) ** 2
    # partial fractions of 1/((1-p z^-1)(1-p* z^-1)) for real input
    return g * np.real(p * y / (1j * np.imag(p)))


def _static_resonator(x, freq, bw):
    r = np.exp(-np.pi * bw / SR)
    b1 = 2 * r * np.cos(2 * np.pi * freq / SR)
    b2 = -r * r
    return signal.lfilter([1 - b1 - b2], [1, -b1, -b2], x)


def tv_zero(x, freq, bw):
    """Unity-DC-gain 2-zero anti-resonance (FIR, per-sample coefficients)."""
    r = np.exp(-np.pi * bw / SR)
    c = 2 * r * np.cos(2 * np.pi * freq / SR)
    g = 1.0 - c + r * r
    x1 = np.concatenate([[0.0], x[:-1]])
    x2 = np.concatenate([[0.0, 0.0], x[:-2]])
    return (x - c * x1 + r * r * x2) / g


# --------------------------------------------------------------------------- helpers
def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _parse_vowel(v, default):
    v = default if v is None else v
    seq = [s.strip().lower() for s in str(v).replace('→', '>').split('>') if s.strip()]
    out = []
    for s in seq:
        s = ALIASES.get(s, s)
        if s not in VOWELS['soprano']:
            raise ValueError(f'unknown vowel {s!r}; use one of {VOWEL_NAMES}')
        out.append(s)
    return out or ['a']


def _assign_parts(notes, tol=0.06):
    """Greedy voice-leading assignment of (possibly polyphonic) notes to monophonic parts."""
    parts: list[list[dict]] = []
    explicit: dict = {}
    for n in sorted(notes, key=lambda n: (n['start'], -n['midi'])):
        if n.get('part') is not None:
            explicit.setdefault(n['part'], []).append(n)
            continue
        best, bd = None, None
        for p in parts:
            last = p[-1]
            if last['start'] + last['dur'] <= n['start'] + tol:
                d = abs(last['midi'] - n['midi']) + 0.001 * (n['start'] - last['start'] - last['dur'])
                if bd is None or d < bd:
                    best, bd = p, d
        if best is None:
            parts.append([n])
        else:
            best.append(n)
    return list(explicit.values()) + parts


def _phrases(part, legato, gap):
    out = []
    for n in part:
        if out and legato and n['start'] - (out[-1][-1]['start'] + out[-1][-1]['dur']) <= gap:
            out[-1].append(n)
        else:
            out.append([n])
    return out


# loudness targets per vowel relative to 'a' (intrinsic vowel intensity; hum is soft)
VOWEL_TRIM_DB = {'a': 0.0, 'o': -1.0, 'e': -1.0, 'i': -1.5, 'u': -2.5, 'm': -4.0}


def _tuned_formants(F, B, f0):
    """F1/F2 tuning towards f0 (sopranos open the jaw on high notes) + F1 damping."""
    F = np.array(F, dtype=np.float64, copy=True)
    B = np.array(B, dtype=np.float64, copy=True)
    F[0] = (F[0] ** 6 + (1.08 * f0) ** 6) ** (1 / 6)
    F[1] = (F[1] ** 6 + (1.45 * F[0]) ** 6) ** (1 / 6)
    B[0] = B[0] * (1 + 0.5 * np.clip(f0 / F[0] - 0.7, 0, 1))
    return F, B


def _tract_mag2(freqs, F, B, nasal, poles=HIGHER_POLES):
    z1 = np.exp(-2j * np.pi * np.asarray(freqs) / SR)
    H = np.ones_like(z1)
    for fi, bi in list(zip(F, B)) + list(poles):
        p = np.exp(-np.pi * bi / SR) * np.exp(2j * np.pi * fi / SR)
        H *= np.abs(1 - p) ** 2 / ((1 - p * z1) * (1 - np.conj(p) * z1))
    if nasal:
        q = np.exp(-np.pi * NASAL_ZERO[1] / SR) * np.exp(2j * np.pi * NASAL_ZERO[0] / SR)
        H *= (1 - q * z1) * (1 - np.conj(q) * z1) / np.abs(1 - q) ** 2
        a = np.exp(-2 * np.pi * NASAL_LP / SR)
        H *= 1.6 * (1 - a) / (1 - a * z1)
    return np.abs(H) ** 2


def _tilt_mag2(freqs, tilt_db):
    if not tilt_db:
        return np.ones(len(freqs))
    sos = biquad('highshelf', TILT[0], TILT[1], tilt_db)
    _, h = signal.sosfreqz(sos, worN=np.asarray(freqs, dtype=np.float64), fs=SR)
    return np.abs(h) ** 2


def _shelf_mag2(freqs, fq, g_db):
    if not g_db:
        return np.ones(len(freqs))
    _, h = signal.sosfreqz(biquad('highshelf', fq[0], fq[1], g_db), worN=np.asarray(freqs, dtype=np.float64), fs=SR)
    return np.abs(h) ** 2


def _hf_fix_mag2(freqs, k):
    if not k:
        return np.ones(len(freqs))
    _, h = signal.sosfreqz(butter('highpass', HF_FIX_HP, 2), worN=np.asarray(freqs, dtype=np.float64), fs=SR)
    return np.abs(1 + k * h) ** 2


def _est_level_db(vowel, voice, f0, rd, fscale, bwscale, tilt_db=0.0, spec=None):
    """Expected output power (dB) of the voiced source through the tract at this f0."""
    spec = spec or SPECTRA['v1']
    F, B = VOWELS[voice][vowel]
    F, B = _tuned_formants(np.asarray(F) * fscale, np.asarray(B) * bwscale, f0)
    c, _ = lf_coeffs(round(rd, 2))
    k = np.arange(1, len(c) + 1)
    ok = k * f0 < F_CUT - F_TAPER / 2
    fk = k[ok] * f0
    P = np.sum(2 * np.abs(c[ok]) ** 2 * _tract_mag2(fk, F, B, vowel == 'm', spec['poles'])
               * _tilt_mag2(fk, tilt_db) * _hf_fix_mag2(fk, spec['hf_fix'].get(vowel, 0.0))
               * _shelf_mag2(fk, TOP_SHELF, spec.get('top_db', 0.0)))
    return 10 * np.log10(P + 1e-30)


def _vowel_track(t, notes, voice, default, trans, sp, rd, a4, tilt_db=0.0, spec=None):
    """Per-sample formant frequencies, bandwidths, nasality and loudness-normalising
    gain (dB) for one phrase; keyframes per note/vowel, smoothed transitions."""
    spec = spec or SPECTRA['v1']
    keys_t, keys_v, keys_g = [], [], []
    for n in notes:
        seq = _parse_vowel(n.get('vowel'), default)
        seg = n['dur'] / len(seq)
        f0 = a4 * 2 ** ((n['midi'] - 69) / 12)
        for j, v in enumerate(seq):
            keys_t.append(n['start'] + j * seg)
            keys_v.append(v)
            keys_g.append(VOWEL_TRIM_DB[v] - _est_level_db(v, voice, f0, rd, sp['fscale'],
                                                           sp['bwscale'], tilt_db, spec))
    F = np.empty((5, len(t)))
    B = np.empty((5, len(t)))
    idx = np.clip(np.searchsorted(keys_t, t, side='right') - 1, 0, len(keys_t) - 1)
    tabF = np.array([np.log(VOWELS[voice][v][0]) for v in keys_v])
    tabB = np.array([VOWELS[voice][v][1] for v in keys_v], dtype=np.float64)
    tabN = np.array([1.0 if v == 'm' else 0.0 for v in keys_v])
    tabK = np.array([spec['hf_fix'].get(v, 0.0) for v in keys_v])
    F[:] = tabF[idx].T
    B[:] = tabB[idx].T
    nas = tabN[idx]
    hfk = tabK[idx]
    gdb = np.array(keys_g)[idx]
    if len(keys_t) > 1:
        # zero-phase smoothing -> transitions of about `trans` seconds centred on the change
        fc = 1.0 / (np.pi * max(trans, 0.02))
        sos = signal.butter(2, fc / (SR / 2), output='sos')
        padn = int(trans * SR * 3)

        def sm(a):
            a = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(padn, padn)], mode='edge')
            return signal.sosfiltfilt(sos, a, axis=-1)[..., padn:-padn]
        F, B, nas, gdb, hfk = sm(F), sm(B), np.clip(sm(nas), 0, 1), sm(gdb), np.maximum(sm(hfk), 0)
    return np.exp(F), B, nas, gdb, hfk


# --------------------------------------------------------------------------- one singer
def _sing_phrase(notes, sp, opts, rng):
    """Render one monophonic phrase for one singer. Returns (start_index, mono)."""
    solo = opts['solo']
    pre = 0.12                      # breath before voicing
    rel = opts['release']
    t_on = notes[0]['start'] + sp['delay']
    t_end = notes[-1]['start'] + notes[-1]['dur'] + sp['delay']
    tail = rel + 1.2
    t0 = t_on - pre
    n = int((t_end + tail - t0) * SR)
    t = t0 + np.arange(n) / SR

    # ---------------- pitch (in MIDI units, fractional)
    m0 = notes[0]['midi']
    mid = np.full(n, float(m0))
    g = opts['portamento']
    for a, b in zip(notes[:-1], notes[1:]):
        tb = b['start'] + sp['delay']
        mid += (b['midi'] - a['midi']) * _smoothstep((t - (tb - 0.6 * g)) / max(g, 1e-3))
    f0_target = opts['a4'] * 2 ** ((mid - 69) / 12)
    cents = np.full(n, sp['detune'])
    scoop = sp['scoop']
    cents += np.where(t >= t_on, -scoop * np.exp(-(t - t_on) / 0.028), -scoop)
    # vibrato: builds ~0.3 s after each note start, relaxes smoothly at note changes
    starts = np.array([x['start'] + sp['delay'] for x in notes])
    since = t - starts[np.clip(np.searchsorted(starts, t, side='right') - 1, 0, len(starts) - 1)]
    venv = _smoothstep((since - sp['vib_delay']) / sp['vib_build'])
    venv = signal.lfilter([1 - np.exp(-1 / (0.08 * SR))], [1, -np.exp(-1 / (0.08 * SR))], venv)
    vrate = sp['vib_rate'] * (1 + 0.03 * smooth_noise(n, 0.4, rng))
    vph = sp['vib_phase'] + 2 * np.pi * np.cumsum(vrate) / SR
    vdepth = sp['vib_depth'] * (1 + 0.15 * smooth_noise(n, 0.3, rng))
    cents += vdepth * venv * np.sin(vph)
    cents += sp['drift'] * smooth_noise(n, 0.5, rng)
    cents -= sp['fall'] * _smoothstep((t - (t_end - 0.08)) / 0.4)
    f0 = f0_target * 2 ** (cents / 1200)
    # jitter: independent period perturbation per glottal cycle
    ph0 = np.cumsum(f0) / SR
    cyc = ph0.astype(np.int64)
    ncyc = int(cyc[-1]) + 2
    f0 = f0 * (1 + opts['jitter'] * rng.standard_normal(ncyc))[cyc]
    phase = 2 * np.pi * np.cumsum(f0) / SR
    frac_c = ph0 - cyc
    shim = 1 + opts['shimmer'] * rng.standard_normal(ncyc + 1)
    shimmer = shim[cyc] * (1 - frac_c) + shim[cyc + 1] * frac_c

    # ---------------- amplitude
    vel = np.array([x['vel'] for x in notes]) / 127.0
    lv = db2a(-26.0 * (1 - vel) ** 1.2)
    level = np.full(n, lv[0])
    for i in range(1, len(notes)):
        tb = notes[i]['start'] + sp['delay']
        level += (lv[i] - lv[i - 1]) * _smoothstep((t - (tb - 0.6 * g)) / max(g, 1e-3))
    att = sp['attack']
    env = _smoothstep((t - t_on) / att) ** 1.3
    env *= 1 - _smoothstep((t - t_end) / rel)
    # gentle messa di voce on long notes
    for x in notes:
        if x['dur'] > 1.8:
            s0 = x['start'] + sp['delay']
            u = np.clip((t - s0) / x['dur'], 0, 1)
            env *= db2a(sp['swell'] * np.sin(np.pi * u) ** 2 * ((t >= s0) & (t <= s0 + x['dur'])))
    amp_v = level * env * shimmer
    if sp.get('ldrift'):
        amp_v = amp_v * db2a(sp['ldrift'] * smooth_noise(n, 0.2, rng))
    # voice quality: breathy (Rd 2.6) at soft onsets/ends -> Rd_note in the sustain
    rd_note = float(np.clip(1.6 - 0.7 * vel.mean(), 0.9, 1.8)) + (0.1 if solo else 0.0)
    ca, te_a = lf_coeffs(2.4)
    cb, te_b = lf_coeffs(round(rd_note, 2))
    beta = np.clip(env, 0, 1) ** 0.7
    voiced = _harmonic_source(phase, f0, beta, ca, cb, amp_v)

    # ---------------- aspiration noise (open-phase gated) + breath onset / tail
    u = np.mod(phase / (2 * np.pi), 1.0)
    te = te_b
    gate = np.where(u < te, np.sin(np.pi * u / te) ** 2, 0.0)
    noise = rng.standard_normal(n)
    noise2 = rng.standard_normal(n)
    noise = sosfilt(butter('highpass', 350.0, order=2), noise).astype(np.float64)
    noise = sosfilt(butter('lowpass', 9000.0, order=1), noise).astype(np.float64)
    puff = _smoothstep((t - (t_on - pre)) / (pre * 0.8)) * (1 - _smoothstep((t - t_on) / 0.25))
    btail = (_smoothstep((t - t_end) / 0.08) * np.exp(-np.maximum(t - t_end, 0) / (0.12 + 0.08 * solo)))
    b_env = level * (env * (0.55 + 0.45 * gate) * (1.0 + 0.8 * (1 - beta)) + 0.9 * puff + 1.2 * btail)
    breath = opts['breath'] * 0.9 * noise * b_env

    src = voiced + breath

    # ---------------- vocal tract
    spec = opts['spec']
    F, Bw, nas, gdb, hfk = _vowel_track(t - sp['delay'], notes, sp['voice'], opts['vowel'],
                                        opts['vowel_time'], sp, rd_note, opts['a4'], opts['tilt_db'],
                                        spec)
    if sp.get('fdrift'):
        # slow per-singer formant drift (tongue / jaw never hold perfectly still)
        F = F * (1 + sp['fdrift'] * smooth_noise(n, 0.25, rng))[None, :]
    F, Bw = _tuned_formants(F * sp['fscale'][:, None], Bw * sp['bwscale'], f0_target)
    y = src * db2a(gdb)
    for i in range(5):
        y = tv_resonator(y, F[i], Bw[i])
    fsc = float(np.mean(sp['fscale']))
    for fh, bh in spec['poles']:
        if fh * fsc < SR * 0.45:
            y = _static_resonator(y, fh * fsc, bh)
    if opts['tilt_db']:
        y = sosfilt(biquad('highshelf', TILT[0], TILT[1], opts['tilt_db']), y).astype(np.float64)
    if hfk.max() > 1e-3:
        y = y + hfk * sosfilt(butter('highpass', HF_FIX_HP, 2), y).astype(np.float64)
    if spec.get('top_db'):
        y = sosfilt(biquad('highshelf', TOP_SHELF[0], TOP_SHELF[1], spec['top_db']), y).astype(np.float64)
    # 'air': part of the turbulence radiates with little tract shaping (Klatt's parallel
    # branch). v1: flat 2.5-11 kHz band; v2: resonators at this phrase's own F3-F5 plus a
    # soft >7 kHz top. Follows the breath envelope.
    air = noise2 * b_env * opts['breath'] * spec['air_gain']
    if spec['air'] == 'formant':
        Fm = F[2:5].mean(axis=1)
        Bm = Bw[2:5].mean(axis=1) * 1.6
        shaped = sum(g * _static_resonator(air, f_, b_) * (b_ / f_)
                     for g, f_, b_ in zip((1.0, 0.8, 0.6), Fm, Bm))
        top = sosfilt(butter('highpass', 7000.0, 2), air).astype(np.float64)
        top = sosfilt(butter('lowpass', 14000.0, 2), top).astype(np.float64)
        air = 2.2 * shaped + 0.35 * top
    else:
        air = sosfilt(butter('bandpass', (2500.0, 11000.0), order=2), air).astype(np.float64)
    y = y + air * (1 - 0.8 * nas)
    if nas.max() > 1e-3:
        z = tv_zero(y, np.full(n, NASAL_ZERO[0]), np.full(n, NASAL_ZERO[1]))
        z = sosfilt(butter('lowpass', NASAL_LP, order=1), z).astype(np.float64)
        y = (1 - nas) * y + nas * z * 1.6
    y = fade_edges((y * OUT_CAL).astype(np.float32), 0.005, 0.02)
    return int(round(t0 * SR)), y


# --------------------------------------------------------------------------- public API
# style='pad': a bed of voices that should not expose the synthesis (long, staggered
# entries, wide image, shallow slow vibrato, 8 singers per part, slow formant / level
# drift per singer, softer top). Replaces only arguments left at their defaults.
PAD_PRESET = dict(voices=8, breath=0.18, width=1.0, attack=1.8, release=2.5,
                  vibrato_depth=(6.0, 12.0), vibrato_rate=(4.6, 5.4), vibrato_delay=0.8,
                  detune=(6.0, 12.0), onset_spread=0.25, portamento=0.25, legato_gap=0.3,
                  brightness_db=-5.0, presence_cut_db=-4.0, formant_drift=0.025,
                  level_drift_db=1.5)


def choir_pad(notes, **kw):
    """render_choir(notes, style='pad', **kw): long-attack, wide choir bed."""
    return render_choir(notes, style='pad', **kw)


def render_choir(notes, voices=6, vowel='a', breath=0.15, width=0.8, solo=False, *,
                 voice_type='auto', legato=True, legato_gap=0.08, portamento=0.12,
                 vibrato_rate=(5.0, 5.8), vibrato_depth=None, vibrato_delay=0.3,
                 detune=(8.0, 15.0), onset_spread=0.03, attack=None, release=0.35,
                 vowel_time=0.15, jitter=0.004, shimmer=0.035, tilt_db=None, brightness_db=0.0,
                 presence_cut_db=-2.0, a4=440.0, duration=None, gain_db=0.0, seed=0,
                 spectrum='v2', style=None, formant_drift=0.0, level_drift_db=0.0) -> np.ndarray:
    """Render sung notes. Returns float32 (n, 2) at 48 kHz.

    notes : [(start_s, dur_s, midi, velocity, vowel_or_None[, part])]
            vowel: 'a' 'o' 'u' 'e' 'i' 'm' (hum) or a morph like 'm>a', 'a>o>u'.
            Simultaneous notes become separate parts automatically (voice leading),
            or give an explicit part id as 6th element.
    voices : singers per part (choir). Ignored when solo=True (1 singer).
    vowel : default vowel for notes whose vowel is None.
    breath : aspiration level (0 = none, 0.15 default, solo adds 60 %).
    width : stereo spread of the singers (0 = mono centre, 1 = full L/R).
    solo : one expressive singer: more breath, deeper vibrato, larger scoops / falls.
    voice_type : 'auto' (per part: soprano if median >= G4, else alto), 'soprano', 'alto'.
    legato / legato_gap / portamento : notes of a part that start within legato_gap
            after the previous one ends are joined with a glide of `portamento` s.
    vibrato_rate (Hz range), vibrato_depth (cents range, default (15,25) choir,
            (18,30) solo), vibrato_delay (s): per-singer random within the range.
    detune : per-singer detune magnitude range in cents (signs alternate).
    onset_spread : width (s) of the per-singer onset scatter; each singer starts at a
            random offset in [-w/2, +w/2] around the written time (default 30 ms).
    attack : attack (crescendo) time in s; default depends on velocity.
    release : decay time after the note end (s); breath continues a little longer.
    brightness_db : high shelf at 3 kHz; presence_cut_db : bell at 3.2 kHz (tames 2.5-4 kHz).
    tilt_db : spectral tilt shelf after the tract (None = the spectrum model's default).
    spectrum : 'v2' (default: higher-pole ladder to 16 kHz, back-vowel HF correction,
            formant-shaped air) or 'v1' (round-1 sound, for reproducing old renders).
    style : None, or 'pad' = long attacks, wide image, shallow vibrato, larger ensemble and
            slow per-singer formant / level drift, for beds that should not expose the
            synthesis. The preset only replaces arguments left at their defaults.
    formant_drift : per-singer slow formant drift (fraction, e.g. 0.025 = +-2.5 %).
    level_drift_db : per-singer slow level drift (dB).
    """
    if style is not None:
        if style != 'pad':
            raise ValueError("style must be None or 'pad'")
        import inspect
        args = dict(locals())
        defaults = {k: v.default for k, v in inspect.signature(render_choir).parameters.items()
                    if v.default is not inspect.Parameter.empty}
        kw = {k: args[k] for k in defaults if k != 'style'}
        for k, v in PAD_PRESET.items():
            if kw[k] == defaults[k]:
                kw[k] = v
        return render_choir(notes, style=None, **kw)
    if spectrum not in SPECTRA:
        raise ValueError(f'spectrum must be one of {list(SPECTRA)}')
    spec = SPECTRA[spectrum]
    if tilt_db is None:
        tilt_db = spec['tilt_db']
    rng = np.random.default_rng(seed)
    ns = []
    for x in notes:
        d = {'start': float(x[0]), 'dur': float(x[1]), 'midi': float(x[2]),
             'vel': float(np.clip(x[3] if len(x) > 3 else 80, 1, 127)),
             'vowel': x[4] if len(x) > 4 else None, 'part': x[5] if len(x) > 5 else None}
        if d['dur'] > 0:
            _parse_vowel(d['vowel'], vowel)     # validate early
            ns.append(d)
    if not ns:
        return np.zeros((int((duration or 0.1) * SR), 2), dtype=np.float32)
    parts = _assign_parts(ns)
    n_sing = 1 if solo else max(1, int(voices))
    if vibrato_depth is None:
        vibrato_depth = (18.0, 30.0) if solo else (15.0, 25.0)
    opts = dict(solo=solo, release=release * (1.3 if solo else 1.0),
                portamento=portamento, a4=a4, jitter=jitter * (1.2 if solo else 1.0),
                shimmer=shimmer, breath=breath * (1.6 if solo else 1.0), vowel=vowel,
                vowel_time=vowel_time, tilt_db=tilt_db, spec=spec)
    rendered = []
    end = max(x['start'] + x['dur'] for x in ns) + opts['release'] + 1.0
    total = duration if duration is not None else end
    out = np.zeros((int(total * SR) + 1, 2), dtype=np.float32)
    n_parts = len(parts)
    for pi, part in enumerate(parts):
        med = float(np.median([x['midi'] for x in part]))
        vt = voice_type if voice_type != 'auto' else ('soprano' if med >= 67 else 'alto')
        phrases = _phrases(part, legato, legato_gap)
        # stereo positions: interleave parts across the field
        base = np.linspace(-1, 1, n_sing) if n_sing > 1 else np.zeros(1)
        base = base + (pi - (n_parts - 1) / 2) / max(n_parts, 1) * (2.0 / max(n_sing, 1)) * 0.5
        rng.shuffle(base)
        signs = np.where(np.arange(n_sing) % 2 == 0, 1.0, -1.0)
        rng.shuffle(signs)
        for si in range(n_sing):
            srng = np.random.default_rng(rng.integers(1 << 62))
            sp = {
                'voice': vt,
                'detune': 0.0 if n_sing == 1 else float(signs[si] * srng.uniform(*detune)),
                'vib_rate': float(srng.uniform(*vibrato_rate)),
                'vib_depth': float(srng.uniform(*vibrato_depth)),
                'vib_phase': float(srng.uniform(0, 2 * np.pi)),
                'vib_delay': float(vibrato_delay * srng.uniform(0.85, 1.25)),
                'vib_build': float(srng.uniform(0.35, 0.6)),
                'drift': float(srng.uniform(2.0, 4.0)),
                'scoop': float(srng.uniform(60, 100) if solo else srng.uniform(50, 80)),
                'fall': float(srng.uniform(25, 45) if solo else srng.uniform(15, 30)),
                'swell': float(1.5 if solo else 0.8),
                'fscale': np.exp(np.log(srng.uniform(0.955, 1.045)) + srng.normal(0, 0.02, 5)),
                'bwscale': float(srng.uniform(0.95, 1.25)),
                'fdrift': float(formant_drift),
                'ldrift': float(level_drift_db),
            }
            pan = float(np.clip(base[si] * width, -1, 1)) if n_sing > 1 else 0.0
            gl, gr = pan_gains(pan)
            itd = int(abs(pan) * 0.0004 * SR)            # up to 0.4 ms towards the far ear
            lvl = db2a(srng.normal(0, 1.0)) / np.sqrt(n_sing)
            for ph in phrases:
                vmean = np.mean([x['vel'] for x in ph]) / 127
                sp['delay'] = float(srng.uniform(-0.5, 0.5) * onset_spread) if n_sing > 1 else 0.0
                sp['attack'] = float(attack if attack is not None else
                                     (0.18 + 0.25 * (1 - vmean)) * (0.8 if solo else 1.0) * srng.uniform(0.9, 1.15))
                s_idx, y = _sing_phrase(ph, sp, opts, srng)
                y = y * np.float32(lvl * 0.5)
                st = np.zeros((len(y) + itd, 2), dtype=np.float32)
                if pan >= 0:
                    st[itd:, 0] = y * gl
                    st[:len(y), 1] = y * gr
                else:
                    st[:len(y), 0] = y * gl
                    st[itd:, 1] = y * gr
                add_at(out, st, s_idx)
    # gentle tone shaping of the whole ensemble
    if presence_cut_db:
        out = sosfilt(biquad('peak', 3200.0, 1.0, presence_cut_db), out)
    if brightness_db:
        out = sosfilt(biquad('highshelf', 3000.0, 0.7, brightness_db), out)
    out = sosfilt(butter('highpass', 45.0, order=2), out)
    out *= np.float32(db2a(gain_db))
    return fade_edges(out, 0.005, 0.02)
