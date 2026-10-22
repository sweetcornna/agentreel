"""Pipe organ (earth and sky): additive synthesis, one sinusoid per pipe harmonic.

Every pipe is a fixed object: its detune (+-1..2 cents) and harmonic voicing are seeded by
(stop, key), so the same key always sounds the same pipe. Speech: a smooth build-up whose time
grows with pipe length, a pitched 'chiff' (breathy noise near the 2nd/3rd harmonic plus a short
octave overshoot), mouth noise while sounding, and a short release. A shared wind supply adds a
tiny common pitch/amplitude waver. Pipes are laid out in the stereo field like a C/C# chest.

Stops (key -> footage):
  '32'   Contra Bourdon 32'  (two octaves down, stopped; 2nd+3rd harmonics reinforced so the
         pitch is felt on small speakers)
  '16'   Subbass 16' (stopped)        '16p'  Principal 16' (open)
  '8p'   Principal 8'                 '8g'   Gedackt 8' (stopped flute, soft)
  '4p'   Octave 4'                    '2.67' Quint 2 2/3'
  '2p'   Superoctave 2'               'mix'  Mixture IV (1 1/3', 1', 2/3', 1/2' with breaks)
  '16r'  Posaune 16' (reed)           '8r'   Trompete 8' (reed)
Written pitch is the 8' pitch: '32' sounds two octaves below the written note.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy import signal

from ..core import SR, fade, midi_to_hz, raised_cos, seed_of, smooth_noise, vel_amp
from ..dsp.delay import vardelay
from ..dsp.filters import svf
from .base import Instrument, Note


@njit(cache=True, fastmath=True)
def _pipes(n, freqs, amps, phases, tau_att, chiff, tau_chiff, rel_i, tau_rel, gl, gr, sr):
    out = np.zeros((2, n))
    for k in range(freqs.shape[0]):
        f = freqs[k]
        if f <= 0.0 or f >= sr * 0.46:
            continue
        w = 2.0 * np.pi * f / sr
        cw = np.cos(w)
        sw = np.sin(w)
        c = np.cos(phases[k])
        s = np.sin(phases[k])
        ae = 1.0
        adec = np.exp(-1.0 / (tau_att[k] * sr))
        q = 1.0
        qdec = np.exp(-1.0 / (tau_chiff[k] * sr))
        ch = chiff[k]
        re = 1.0
        rdec = np.exp(-1.0 / (tau_rel[k] * sr))
        a = amps[k]
        ri = rel_i[k]
        for i in range(n):
            e1 = 1.0 - ae
            env = e1 * e1
            if ch != 0.0:
                tt = i / (tau_chiff[k] * sr)
                env += ch * tt * q * 2.718281828
                q *= qdec
            if i >= ri:
                env *= re
                re *= rdec
                if re < 1e-5:
                    break
            v = a * env * s
            out[0, i] += v * gl[k]
            out[1, i] += v * gr[k]
            c2 = c * cw - s * sw
            s = s * cw + c * sw
            c = c2
            ae *= adec
            if (i & 4095) == 0:
                r = 1.0 / np.sqrt(c * c + s * s)
                c *= r
                s *= r
    return out


# stop table: (pitch multiplier relative to 8', family, level)
STOPS = {
    "32": (0.25, "bourdon32", 0.6),
    "16": (0.5, "stopped", 0.6),
    "16p": (0.5, "principal", 0.55),
    "8p": (1.0, "principal", 1.0),
    "8g": (1.0, "stopped", 0.75),
    "4p": (2.0, "principal", 0.6),
    "2.67": (3.0, "principal", 0.3),
    "2p": (4.0, "principal", 0.36),
    "mix": (None, "mixture", 0.32),
    "16r": (0.5, "reed", 0.62),
    "8r": (1.0, "reed", 0.7),
}
_MIX_RANKS = (6.0, 8.0, 12.0, 16.0)


def _spectrum(family: str, f: float, rng, max_h=60):
    ks = np.arange(1, max_h + 1)
    if family == "principal":
        a = np.where(ks == 1, 1.0, 0.62 * ks ** -1.45)
        a[1] = 0.55
        a[2] = 0.36
    elif family == "stopped":
        a = np.where(ks % 2 == 1, ks ** -1.7, 0.04 * ks ** -1.8)
        a[0] = 1.0
        a[2] = 0.2
    elif family == "bourdon32":
        a = np.zeros(max_h)
        a[:6] = [1.0, 0.75, 0.5, 0.14, 0.12, 0.04]
    elif family == "reed":
        fk = ks * f
        form = 1 + 2.2 * np.exp(-(np.log2(np.maximum(fk, 1) / 1600.0)) ** 2 / 0.6)
        a = ks ** -0.55 * form
        a /= a[0]
    elif family == "mixture_rank":
        a = np.where(ks == 1, 1.0, 0.4 * ks ** -1.6)
    else:
        raise ValueError(family)
    fk = ks * f
    # gentle roll-off of the top octave and hard cut below Nyquist
    a = a * np.clip((16500 - fk) / 3500, 0, 1)
    a = a * np.exp(0.17 * rng.standard_normal(max_h))  # +-1.5 dB pipe voicing
    keep = (a > 1e-3) & (fk < 0.45 * SR)
    return fk[keep], a[keep], ks[keep]


class Organ(Instrument):
    """Additive pipe organ with registration, chiff, wind, C/C# chest layout."""
    name = "organ"
    PARAMS = {
        "stops": (("8p", "8g", "4p"), "registration: tuple of stop keys (see module doc)"),
        "chiff": (0.5, "pipe speech transient amount 0..1"),
        "wind": (0.5, "wind noise / waver amount 0..1"),
        "attack": (1.0, "multiplies pipe speech time (larger = slower, softer onset)"),
        "release": (1.0, "multiplies pipe release time"),
        "brightness": (0.5, "voicing tilt 0 dark .. 1 bright"),
        "detune": (1.4, "pipe tuning spread in cents (std)"),
        "spread": (0.65, "stereo width of the pipe layout"),
        "tremulant": (0.0, "tremulant depth 0..1 (5.6 Hz)"),
        "wind_seed": (11, "seed of the shared wind supply (keep equal across organ tracks)"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        vel = float(np.clip(note.vel_scalar(), 0.0, 1.0))
        f8 = float(midi_to_hz(note.midi))
        rel = note.dur
        tail = 0.35 + 3.0 / max(f8 * 0.25, 20.0) * P("release")
        n = int((rel + tail) * sr)
        rel_i_note = int(rel * sr)
        F, A, PH, TA, CH, TC, RI, TR, GL, GR = ([] for _ in range(10))
        noise_bursts = []
        bright = float(P("brightness"))
        key = int(round(note.midi))
        for stop in P("stops"):
            if stop not in STOPS:
                raise ValueError(f"unknown organ stop {stop!r}; known {list(STOPS)}")
            mult, fam, lvl = STOPS[stop]
            ranks = [(m, "mixture_rank") for m in _MIX_RANKS] if fam == "mixture" else [(mult, fam)]
            for ri, (m, fam2) in enumerate(ranks):
                fp = f8 * m
                if fam == "mixture":
                    while fp > 5200:
                        fp /= 2.0  # mixture breaks back
                    lv = lvl / len(ranks) * 1.6
                else:
                    lv = lvl
                prng = np.random.default_rng(seed_of("pipe", stop, ri, key))  # the pipe itself
                det = P("detune") * prng.standard_normal() / 1200.0
                fp = fp * 2 ** det
                fk, ak, ks = _spectrum(fam2, fp, prng)
                ak = ak * ks ** ((bright - 0.5) * 0.8)
                # voicing: lower pipes a bit stronger in fundamental, treble slightly tamed
                ak = ak * lv * (fp / 262.0) ** -0.12
                ta = (0.012 + 3.0 / fp) * P("attack") * (1 + 0.1 * rng.standard_normal())
                ta_k = ta * np.where(ks == 1, 1.0, 0.65) * (1.0 if fam2 != "reed" else 0.5)
                if fam2 == "reed":
                    ta_k = ta_k * (1 + 0.06 * ks)  # brassy: upper harmonics bloom slightly later
                chf = np.zeros(len(fk))
                if fam2 in ("principal", "mixture_rank") and P("chiff") > 0:
                    chf[(ks == 2)] = 0.55 * P("chiff")
                    chf[(ks == 3)] = 0.25 * P("chiff")
                tc = np.full(len(fk), max(0.012, ta * 0.7))
                tr = (0.03 + 4.0 / fp) * P("release")  # long pipes die away more slowly
                # chest layout: C side / C# side, low pipes outside, high pipes toward the centre
                side = 1.0 if key % 2 == 0 else -1.0
                if fam in ("bourdon32", "stopped") and m <= 0.5:
                    pan = side * 0.12
                else:
                    pan = side * P("spread") * np.clip(0.85 - (note.midi - 36) / 80.0, 0.15, 0.85)
                    pan += 0.08 * prng.standard_normal()
                th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
                F.append(fk); A.append(ak); PH.append(rng.uniform(0, 2 * np.pi, len(fk)))
                TA.append(ta_k); CH.append(chf); TC.append(tc)
                RI.append(np.full(len(fk), rel_i_note + int(abs(0.004 * rng.standard_normal()) * sr)))
                TR.append(np.full(len(fk), tr)); GL.append(np.full(len(fk), np.cos(th)))
                GR.append(np.full(len(fk), np.sin(th)))
                if fam2 != "reed":
                    noise_bursts.append((fp, ta, pan, lv))
        cat = lambda L: np.ascontiguousarray(np.concatenate(L))  # noqa: E731
        y = _pipes(n, cat(F), cat(A), cat(PH), cat(TA), cat(CH), cat(TC), cat(RI).astype(np.int64),
                   cat(TR), cat(GL), cat(GR), float(sr))

        # pitched chiff noise and mouth noise
        if P("chiff") > 0 or P("wind") > 0:
            t = np.arange(n) / sr
            sustain = np.clip(t / 0.05, 0, 1) * np.where(t < rel, 1.0, np.exp(-(t - rel) / 0.05))
            for fp, ta, pan, lv in noise_bursts[:6]:
                fc = min(fp * (2.0 + rng.uniform(0, 1.0)), 0.4 * sr)
                nz = svf(rng.standard_normal(n), fc, 12.0, "bandpass_norm")
                burst = (t / (ta * 0.8)) * np.exp(1 - t / (ta * 0.8))
                env = 0.10 * P("chiff") * burst + 0.008 * P("wind") * sustain
                mouth = svf(rng.standard_normal(n), min(fp * 6 + 2500, 0.4 * sr), 1.2, "bandpass_norm")
                sig = lv * (env * nz + 0.006 * P("wind") * sustain * mouth)
                th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
                y[0] += sig * np.cos(th) * 0.5
                y[1] += sig * np.sin(th) * 0.5

        # shared wind supply: common, tiny pitch/amplitude waver (time-warp + gain)
        t_abs = note.t0 + np.arange(n) / sr
        wseed = int(P("wind_seed"))
        wob = _global_noise(t_abs, 0.9, wseed)
        trem = P("tremulant")
        dsec = (0.00006 + 0.0001 * P("wind")) * wob
        if trem > 0:
            dsec = dsec + trem * 0.00025 * np.sin(2 * np.pi * 5.6 * t_abs)
        y = vardelay(y, 4.0 + dsec * sr)
        amp_w = 1 + (0.004 + 0.006 * P("wind")) * _global_noise(t_abs, 2.3, wseed + 1)
        if trem > 0:
            amp_w *= 1 + 0.12 * trem * np.sin(2 * np.pi * 5.6 * t_abs + 0.8)
        y *= amp_w
        y = np.roll(y, -4, axis=1)
        y[:, -4:] = 0
        y *= vel_amp(0.55 + 0.45 * vel, 30.0) * 0.12
        y[:, :32] *= raised_cos(32)
        return fade(y, 0, 0.05)

    def render_notes(self, notes, n_total: int, sr: int = SR):
        buf = super().render_notes(notes, n_total, sr)
        # blower and wind-chest noise while the organ plays
        if notes and self.defaults["wind"] > 0:
            act = np.zeros(n_total)
            for nt in notes:
                a = int(nt.t0 * sr)
                b = min(n_total, int((nt.t0 + nt.dur) * sr))
                act[max(0, a):b] += 1
            act = np.minimum(act, 4) / 4
            act = signal.lfilter([1 - np.exp(-1 / (0.3 * sr))], [1, -np.exp(-1 / (0.3 * sr))], act)
            rng = np.random.default_rng(seed_of("blower", self.defaults["wind_seed"]))
            rumble = signal.sosfilt(signal.butter(2, [30, 180], "bandpass", fs=sr, output="sos"),
                                    rng.standard_normal((2, n_total)), axis=-1)
            hiss = signal.sosfilt(signal.butter(2, [1500, 6000], "bandpass", fs=sr, output="sos"),
                                  rng.standard_normal((2, n_total)), axis=-1)
            lvl = np.sqrt(np.mean(buf ** 2) + 1e-20)
            buf += self.defaults["wind"] * lvl * act * (0.05 * rumble + 0.012 * hiss)
        return buf


_GN_CACHE = {}


def _global_noise(t_abs, rate, seed):
    """Deterministic smooth noise defined on absolute time (shared by all notes of the organ)."""
    key = (rate, seed)
    if key not in _GN_CACHE:
        rng = np.random.default_rng(seed)
        _GN_CACHE[key] = rng.standard_normal(int(400 * rate) + 8)
    pts = _GN_CACHE[key]
    x = np.clip(t_abs * rate, 0, len(pts) - 4)
    i = np.floor(x).astype(int)
    u = x - i
    p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)
