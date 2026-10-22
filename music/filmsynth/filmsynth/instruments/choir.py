"""Formant-synthesis choir: distant, soft, devout pad. Each singer is simulated.

Singer = band-limited Rosenberg glottal-flow-derivative source (polyBLEP at glottal closure),
open quotient and spectral tilt from vocal effort, aspiration noise gated by the glottal flow,
f0 with individual detune (+-10 c), slow drift, fast jitter, delayed 4.8-6.2 Hz vibrato with random
depth, onset scoop, amplitude shimmer, individual timing and level.
Singers are summed in 3 sub-groups per section; each sub-group has its own vocal-tract scaling
(+-4 % formants) and stereo position, and goes through 5 time-varying parallel formant resonators
(vowel morphs ah/oh/oo/eh/ee/mm). Sopranos/altos tune F1 up to f0 on high notes.
Contiguous notes in a voice are sung legato (one continuous source with portamento).
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy import signal

from ..core import SR, fade, midi_to_hz, raised_cos, seed_of, smooth_noise, to_midi
from .base import Instrument, Note, allocate_voices, vel_curve

# formant tables: freq (Hz), level (dB), bandwidth (Hz) for F1..F5
_V = {
    "soprano": {
        "ah": ([800, 1150, 2900, 3900, 4950], [0, -6, -32, -20, -50], [80, 90, 120, 130, 140]),
        "oh": ([450, 800, 2830, 3800, 4950], [0, -11, -22, -22, -50], [70, 80, 100, 130, 135]),
        "oo": ([325, 700, 2700, 3800, 4950], [0, -16, -35, -40, -60], [50, 60, 170, 180, 200]),
        "eh": ([350, 2000, 2800, 3600, 4950], [0, -20, -15, -40, -56], [60, 100, 120, 150, 200]),
        "ee": ([270, 2140, 2950, 3900, 4950], [0, -12, -26, -26, -44], [60, 90, 100, 120, 120]),
        "mm": ([300, 1100, 2400, 3300, 4300], [0, -30, -40, -46, -56], [70, 200, 260, 300, 300]),
    },
    "alto": {
        "ah": ([800, 1150, 2800, 3500, 4950], [0, -4, -20, -36, -60], [80, 90, 120, 130, 140]),
        "oh": ([450, 800, 2830, 3500, 4950], [0, -9, -16, -28, -55], [70, 80, 100, 130, 135]),
        "oo": ([325, 700, 2530, 3500, 4950], [0, -12, -30, -40, -64], [50, 60, 170, 180, 200]),
        "eh": ([400, 1600, 2700, 3300, 4950], [0, -24, -30, -35, -60], [60, 80, 120, 150, 200]),
        "ee": ([350, 1700, 2700, 3700, 4950], [0, -20, -30, -36, -60], [50, 100, 120, 150, 200]),
        "mm": ([280, 1050, 2300, 3200, 4200], [0, -30, -40, -46, -56], [70, 200, 260, 300, 300]),
    },
    "tenor": {
        "ah": ([650, 1080, 2650, 2900, 3250], [0, -6, -7, -8, -22], [80, 90, 120, 130, 140]),
        "oh": ([400, 800, 2600, 2800, 3000], [0, -10, -12, -12, -26], [40, 80, 100, 120, 120]),
        "oo": ([350, 600, 2700, 2900, 3300], [0, -20, -17, -14, -26], [40, 60, 100, 120, 120]),
        "eh": ([400, 1700, 2600, 3200, 3580], [0, -14, -12, -14, -20], [70, 80, 100, 120, 120]),
        "ee": ([290, 1870, 2800, 3250, 3540], [0, -15, -18, -20, -30], [40, 90, 100, 120, 120]),
        "mm": ([260, 1000, 2200, 2900, 3500], [0, -30, -38, -44, -54], [70, 200, 250, 300, 300]),
    },
    "bass": {
        "ah": ([600, 1040, 2250, 2450, 2750], [0, -7, -9, -9, -20], [60, 70, 110, 120, 130]),
        "oh": ([400, 750, 2400, 2600, 2900], [0, -11, -21, -20, -40], [40, 80, 100, 120, 120]),
        "oo": ([350, 600, 2400, 2675, 2950], [0, -20, -32, -28, -36], [40, 80, 100, 120, 120]),
        "eh": ([400, 1620, 2400, 2800, 3100], [0, -12, -9, -12, -18], [40, 80, 100, 120, 120]),
        "ee": ([250, 1750, 2600, 3050, 3340], [0, -30, -16, -22, -28], [60, 90, 100, 120, 120]),
        "mm": ([240, 950, 2100, 2800, 3400], [0, -30, -38, -44, -54], [70, 200, 250, 300, 300]),
    },
}
_RANGES = {"soprano": (60, 81), "alto": (53, 74), "tenor": (48, 69), "bass": (40, 62)}
_PANS = {"soprano": -0.35, "alto": 0.35, "tenor": -0.15, "bass": 0.15}


@njit(cache=True, fastmath=True)
def _glottal(f0, amp, oq, tilt_coef, breath, noise, sr):
    """Rosenberg glottal flow derivative with polyBLEP-corrected closure + flow-gated aspiration."""
    n = f0.shape[0]
    y = np.zeros(n)
    ph = noise[0] * 0.5 + 0.5  # random start phase in 0..1
    if ph < 0.0:
        ph = 0.0
    if ph >= 1.0:
        ph = 0.999
    lp = 0.0
    for i in range(n):
        dt = f0[i] / sr
        te = oq[i]
        tp = te * 0.62
        tn = te - tp
        p = ph
        if p < tp:
            d = (np.pi / (2.0 * tp)) * np.sin(np.pi * p / tp)
            flow = 0.5 * (1.0 - np.cos(np.pi * p / tp))
        elif p < te:
            d = -(np.pi / (2.0 * tn)) * np.sin(0.5 * np.pi * (p - tp) / tn)
            flow = np.cos(0.5 * np.pi * (p - tp) / tn)
        else:
            d = 0.0
            flow = 0.0
        # polyBLEP for the step of height h = +pi/(2 tn) at p = te
        h = np.pi / (2.0 * tn)
        rel = p - te
        if rel >= 0.0 and rel < dt:
            x = rel / dt
            d += 0.5 * h * (2.0 * x - x * x - 1.0)
        elif rel < 0.0 and rel > -dt:
            x = rel / dt
            d += 0.5 * h * (x * x + 2.0 * x + 1.0)
        # scale to time derivative and limit the steep pulse with the effort tilt
        v = d * 0.022 * (dt * sr / 220.0) ** 0.3
        lp += tilt_coef[i] * (v - lp)
        y[i] = amp[i] * (lp + breath[i] * (0.25 + flow) * noise[i])
        ph += dt
        if ph >= 1.0:
            ph -= 1.0
    return y


@njit(cache=True, fastmath=True)
def _formants(x, a1, a2, b0, g):
    """Parallel two-pole resonators with per-sample coefficients; g: per-sample linear gain."""
    m, n = a1.shape
    y = np.zeros(n)
    for k in range(m):
        s1 = 0.0
        s2 = 0.0
        x1 = 0.0
        x2 = 0.0
        for i in range(n):
            v = b0[k, i] * (x[i] - x2) - a1[k, i] * s1 - a2[k, i] * s2
            x2 = x1
            x1 = x[i]
            s2 = s1
            s1 = v
            y[i] += g[k, i] * v
    return y


# natural level of each vowel relative to an open 'ah' (dB): closed vowels and humming are softer
_VOWEL_LEVEL = {"ah": 0.0, "oh": -1.0, "eh": -1.0, "ee": -2.0, "oo": -3.0, "mm": -6.0}
_VNORM = {}


def _vowel_norm(section, vowel, sr=SR):
    """Gain that gives every vowel the same output power for a typical glottal spectrum, then the
    natural vowel level offset. Without it 'mm'/'oo' on low voices come out louder than 'ah'."""
    key = (section, vowel)
    if key not in _VNORM:
        f = np.geomspace(60, 8000, 400)
        src = 1.0 / (1.0 + (f / 250.0) ** 2)  # rough glottal tilt (-12 dB/oct above ~250 Hz)
        z = np.exp(-2j * np.pi * f / sr)

        def power(name):
            F, L, B = _V[section][name]
            H = np.zeros_like(f, dtype=complex)
            for fk, lk, bk in zip(F, L, B):
                r = np.exp(-np.pi * bk / sr)
                a1 = -2 * r * np.cos(2 * np.pi * fk / sr)
                H += 10 ** (lk / 20) * (1 - r * r) / 2 * (1 - z ** 2) / (1 + a1 * z + r * r * z ** 2)
            return np.sum(np.abs(H * src) ** 2)

        _VNORM[key] = np.sqrt(power("ah") / power(vowel)) * 10 ** (_VOWEL_LEVEL[vowel] / 20)
    return _VNORM[key]


def _vowel_arrays(section, vowel_curve_names, weights, scale, f0, sr):
    """Build per-sample F, BW, gain arrays from a list of vowel names and per-sample blend weights."""
    n = f0.shape[0]
    F = np.zeros((5, n))
    B = np.zeros((5, n))
    G = np.zeros((5, n))
    for name, w in zip(vowel_curve_names, weights):
        f, l, b = _V[section][name]
        F += np.array(f)[:, None] * w[None, :]
        B += np.array(b)[:, None] * w[None, :]
        G += (10 ** (np.array(l) / 20.0) * _vowel_norm(section, name, sr))[:, None] * w[None, :]
    F *= scale
    if section in ("soprano", "alto"):
        F[0] = np.maximum(F[0], f0 * 1.08)  # formant tuning on high notes
        F[1] = np.maximum(F[1], F[0] * 1.35)
    F = np.minimum(F, 0.45 * sr)
    r = np.exp(-np.pi * B / sr)
    a1 = -2 * r * np.cos(2 * np.pi * F / sr)
    a2 = r * r
    b0 = (1 - r * r) / 2
    return a1, a2, b0, G


class Choir(Instrument):
    """Choir section (soprano / alto / tenor / bass) of 8-16 simulated singers, formant synthesis."""
    name = "choir"
    legato_phrases = True
    PARAMS = {
        "section": ("alto", "soprano | alto | tenor | bass"),
        "singers": (12, "number of singers (8-16 recommended)"),
        "vowel": ("oo", "ah | oh | oo | eh | ee | mm, or (from, to) to morph across the note"),
        "attack": (0.45, "phrase onset time (s)"),
        "release": (0.7, "phrase release time (s)"),
        "vib_cents": (22.0, "mean vibrato depth (cents); each singer 0.4-1.5x"),
        "vib_rate": (5.4, "mean vibrato rate (Hz); singers spread 4.8-6.2"),
        "detune": (6.0, "per-singer tuning std (cents), clipped to +-12"),
        "timing": (0.035, "per-singer entry spread std (s)"),
        "breath": (0.3, "aspiration noise 0..1"),
        "portamento": (0.12, "legato glide time between notes (s)"),
        "distance": (0.5, "0 close .. 1 far (softer highs, slower edges)"),
        "spread": (0.5, "stereo width of the section"),
        "center": (None, "stereo centre (default by section)"),
        "humanize": (1.0, "scales jitter, drift, shimmer and scoops (0 = machine-steady, for tests only)"),
    }

    def render_notes(self, notes, n_total: int, sr: int = SR):
        buf = np.zeros((2, n_total))
        for phrase in allocate_voices(notes):
            y, start = self._phrase(phrase, sr)
            s0 = max(0, start)
            s1 = min(n_total, start + y.shape[1])
            if s1 > s0:
                buf[:, s0:s1] += y[:, s0 - start:s1 - start]
        return buf

    def render(self, note: Note, rng):
        y, start = self._phrase([note], SR)
        return y

    def _phrase(self, phrase, sr):
        first = phrase[0]
        P = lambda k: self.P(first, k)  # noqa: E731
        section = P("section")
        nsing = int(P("singers"))
        att = float(P("attack")) * (1 + 0.5 * P("distance"))
        rel = float(P("release"))
        pre = 0.15
        t_start = first.t0 - pre
        t_end = phrase[-1].t0 + phrase[-1].dur + rel + 0.3
        n = int((t_end - t_start) * sr)
        t = t_start + np.arange(n) / sr
        prng = np.random.default_rng(first.seed)

        # --- shared (score) curves: target pitch steps, velocity, vowel weights
        tgt = np.full(n, to_midi(first.midi))
        velc = np.zeros(n)
        vowels = []
        vweights = []
        names = list(_V[section])
        wv = {k: np.zeros(n) for k in names}
        bounds = []
        for j, nt in enumerate(phrase):
            a = int((nt.t0 - t_start) * sr)
            b = int((nt.t0 + nt.dur - t_start) * sr) if j == len(phrase) - 1 else int((phrase[j + 1].t0 - t_start) * sr)
            b = min(b, n)
            tgt[a:] = nt.midi
            velc[a:b] = vel_curve(nt.vel, b - a)
            if j == len(phrase) - 1:
                velc[b:] = velc[b - 1] if b > a else 0.5
            vw = self.P(nt, "vowel")
            if isinstance(vw, (tuple, list)):
                v0, v1 = vw
                x = np.linspace(0, 1, b - a)
                wv[v0][a:b] += 1 - x
                wv[v1][a:b] += x
                if j == len(phrase) - 1:
                    wv[v1][b:] += 1
            else:
                wv[vw][a:b] += 1
                if j == len(phrase) - 1:
                    wv[vw][b:] += 1
            if j == 0:
                wv_first = vw if not isinstance(vw, (tuple, list)) else vw[0]
            bounds.append(a)
        a0 = int(pre * sr)
        for k in names:
            wv[k][:a0] = 1.0 if k == wv_first else 0.0
        # smooth vowel transitions (~120 ms) and velocity (~60 ms)
        sm = lambda x, tc: signal.filtfilt([1 - np.exp(-1 / (tc * sr))], [1, -np.exp(-1 / (tc * sr))], x)  # noqa: E731
        for k in names:
            wv[k] = np.clip(sm(wv[k], 0.06), 0, None)
        tot = sum(wv.values()) + 1e-9
        for k in names:
            wv[k] /= tot
        used = [k for k in names if wv[k].max() > 1e-3]
        velc[:a0] = velc[a0] if a0 < n else 0.5
        velc = sm(velc, 0.04)

        # phrase envelope (score timing); singers shift it individually
        vel_mean = float(np.mean(velc[a0:]))
        oq = 0.78 - 0.22 * np.clip(velc, 0, 1)
        tilt_fc = (700 + 3800 * np.clip(velc, 0, 1) ** 1.4) * (1.0 - 0.45 * P("distance"))
        if "mm" in used:
            tilt_fc = tilt_fc * (1 - 0.6 * wv["mm"])
        tilt2 = 1 - np.exp(-2 * np.pi * tilt_fc / (2 * sr))

        groups = 3
        out = np.zeros((2, n))
        center = P("center")
        if center is None:
            center = _PANS[section]
        for g in range(groups):
            grng = np.random.default_rng(seed_of(first.seed, "group", g))
            # vocal-tract size of this sub-group + slow articulatory drift (+-1.5 %)
            scale = (1 + 0.04 * grng.uniform(-1, 1)) * (1 + 0.015 * smooth_noise(n, 0.3, grng, sr))
            src = np.zeros((2, 2 * n))
            members = range(g, nsing, groups)
            for s in members:
                srng = np.random.default_rng(seed_of(first.seed, "singer", s))
                # each singer is the same person across the piece: tuning/vibrato style from section seed
                prs = np.random.default_rng(seed_of(section, "person", s))
                detune = float(np.clip(prs.normal(0, P("detune")), -12, 12))
                vib_depth = P("vib_cents") * prs.uniform(0.4, 1.5)
                vib_rate = float(np.clip(P("vib_rate") + prs.normal(0, 0.4), 4.8, 6.2))
                level = 10 ** (prs.normal(0, 1.5) / 20)
                dt_on = float(np.clip(srng.normal(0.012, P("timing")), -0.06, 0.1))
                dt_off = float(np.clip(srng.normal(0.0, P("timing")), -0.08, 0.08))
                # pitch: glide between targets with individual portamento
                port = P("portamento") * srng.uniform(0.6, 1.5)
                shift = int(dt_on * sr)
                tg = np.roll(tgt, shift)
                if shift > 0:
                    tg[:shift] = tgt[0]
                k = max(1, int(port * sr))
                ker = np.hanning(k + 2)[1:-1]
                tg = np.convolve(np.concatenate([np.full(k, tg[0]), tg, np.full(k, tg[-1])]), ker / ker.sum(), "same")[k:k + n]
                vd = float(np.clip(pre + dt_on + srng.uniform(0.15, 0.6), 0, None))
                ph = srng.uniform(0, 2 * np.pi)
                rate = vib_rate * (1 + 0.05 * smooth_noise(n, 0.4, srng, sr))
                venv = np.clip((np.arange(n) / sr - vd) / 0.5, 0, 1)
                vib = vib_depth * venv * (1 + 0.2 * smooth_noise(n, 0.3, srng, sr)) * np.sin(2 * np.pi * np.cumsum(rate) / sr + ph)
                hum = float(P("humanize"))
                drift = 4.0 * hum * smooth_noise(n, 0.25, srng, sr)
                jit = 3.5 * hum * smooth_noise(n, 35.0, srng, sr)
                scoop = 0.0
                if srng.random() < 0.5 * min(hum, 1.0):
                    tt = np.maximum(np.arange(n) / sr - pre - dt_on, 0)
                    scoop = -srng.uniform(15, 45) * np.exp(-tt / 0.07) * (np.arange(n) / sr >= pre + dt_on)
                m = tg + (detune + vib + drift + jit + scoop) / 100.0
                f0 = midi_to_hz(m)
                # amplitude: phrase envelope with individual edges, shimmer
                on = pre + dt_on
                off = (t_end - rel - 0.3 - t_start) + dt_off
                tt = np.arange(n) / sr
                a_sing = att * srng.uniform(0.7, 1.4)
                env = np.clip((tt - on) / a_sing, 0, 1)
                env = env * env * (3 - 2 * env)
                r_sing = rel * srng.uniform(0.7, 1.3)
                renv = np.clip(1 - (tt - off) / r_sing, 0, 1)
                env *= renv * renv * (3 - 2 * renv)
                shimmer = 1 + hum * (0.06 * smooth_noise(n, 3.0, srng, sr) + 0.025 * smooth_noise(n, 25.0, srng, sr))
                amp = env * shimmer * level * (0.25 + 0.75 * np.clip(np.roll(velc, shift), 0, 1)) ** 1.3
                # re-articulate repeated pitches with a small dip
                for j in range(1, len(phrase)):
                    if abs(phrase[j].midi - phrase[j - 1].midi) < 0.01:
                        c = bounds[j] + shift
                        w = int(0.07 * sr)
                        lo, hi = max(0, c - w), min(n, c + w)
                        amp[lo:hi] *= 1 - 0.35 * np.hanning(hi - lo)
                noise = srng.standard_normal(2 * n)
                bpar = float(P("breath"))
                breath = np.full(n, (0.04 + 0.2 * bpar) if bpar > 0 else 0.0) * (1.3 - 0.6 * np.clip(velc, 0, 1))
                if "mm" in used:
                    breath *= 1 - 0.85 * wv["mm"]  # humming: closed mouth, little air noise
                # aspirated onset for some singers
                if srng.random() < 0.4 and bpar > 0:
                    breath += 0.6 * np.exp(-np.maximum(tt - on, 0) / 0.08) * (tt >= on - 0.03)
                # source runs at 2x (summed per group, decimated once): aliasing < -60 dB below 10 kHz
                r2 = lambda a: np.ascontiguousarray(np.repeat(a, 2))  # noqa: E731
                v = _glottal(r2(f0), r2(amp), r2(oq), r2(tilt2), r2(breath), noise, float(2 * sr))
                sp = float(np.clip(center + P("spread") * prs.uniform(-1, 1), -1, 1))
                th = (sp + 1) * np.pi / 4
                src[0] += v * np.cos(th)
                src[1] += v * np.sin(th)
            src = signal.resample_poly(src, 1, 2, axis=-1)[:, :n]
            f0_mean = midi_to_hz(tgt)
            a1, a2, b0, G = _vowel_arrays(section, used, [wv[k] for k in used], scale, f0_mean, sr)
            for c in range(2):
                out[c] += _formants(np.ascontiguousarray(src[c]), a1, a2, b0, G)
        # distance: soften the top
        dist = float(P("distance"))
        if dist > 0:
            fc = 9000 - 5000 * dist
            out = signal.sosfilt(signal.butter(2, fc, fs=sr, output="sos"), out, axis=-1)
        out = signal.sosfilt(signal.butter(2, 70, "highpass", fs=sr, output="sos"), out, axis=-1)
        out *= 6.9 / np.sqrt(nsing)
        out[:, :64] *= raised_cos(64)
        out = fade(out, 0, 0.1)
        return out, int(round(t_start * sr))
