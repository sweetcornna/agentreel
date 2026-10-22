"""Metals: gongs and tam-tams (hundreds of inharmonic modes that bloom slowly), suspended cymbal
swells (mallet roll driving a dense resonator bank), and small tuned handbells."""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..core import SR, fade, midi_to_hz, raised_cos, smooth_noise, vel_amp
from ..dsp.filters import modal_bank, svf
from ..dsp.modal import render_modes
from .base import Instrument, Note
from .bells import Bell


class Gong(Instrument):
    """Gong family. presets: 'tamtam' (unpitched, blooming wash), 'chau' (big Chinese gong with a
    pitched centre that sags slightly), 'opera_rise' (small gong whose pitch rises after the hit).
    Pitch sets the centre tone for 'chau'/'opera_rise' and the overall register for 'tamtam'."""
    name = "gong"
    PARAMS = {
        "preset": ("tamtam", "tamtam | chau | opera_rise"),
        "modes": (220, "number of modes"),
        "bloom": (1.0, "how slowly the upper modes swell after the hit (0 = none)"),
        "decay": (1.0, "multiplies T60s"),
        "brightness": (0.5, "0 dark .. 1 bright"),
        "mallet": (0.5, "0 very soft (barely touching) .. 1 hard"),
        "damp": (False, "choke at the written end"),
        "width": (0.8, "stereo width"),
        "max_ring": (16.0, "cap on ring length (s)"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        pre = P("preset")
        vel = float(np.clip(note.vel_scalar(), 0.02, 1))
        f_ref = float(midi_to_hz(note.midi)) if np.isfinite(note.midi) else 110.0
        nm = int(P("modes"))
        lo = f_ref * 0.45
        hi = min(12000.0, f_ref * 110)
        u = rng.random(nm)
        freqs = np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * u ** 0.75)  # denser toward the top
        br = P("brightness")
        amps = (freqs / f_ref) ** (-0.55 + 0.5 * br) * np.exp(0.5 * rng.standard_normal(nm))
        t60 = np.clip(16.0 * P("decay") * (freqs / f_ref) ** -0.3 * np.exp(0.25 * rng.standard_normal(nm)), 0.6, 30)
        mal = P("mallet")
        # bloom: upper modes take longer to receive energy (nonlinear cascade), soft mallets bloom more
        bl = P("bloom") * np.clip(0.22 * (freqs / f_ref) ** 0.55, 0.0, 3.5) * (1.4 - 0.8 * mal) * (1.3 - 0.6 * vel)
        amps *= 1 / np.sqrt(1 + (freqs / (900 + 7000 * mal * vel)) ** 2) ** (1 - 0.6 * np.minimum(bl, 1))
        if pre == "tamtam":
            # energy cascades upward after a soft hit: the bloomed upper modes end up louder than the
            # initial low ones, so the sound swells for a second or two before it decays
            hi = freqs > 4 * f_ref
            amps[hi] *= 1 + 4.0 * P("bloom") * (1.2 - mal)
            amps[~hi] *= 0.6 + 0.4 * mal
        glide = np.zeros(nm)
        gtau = np.full(nm, 0.5)
        if pre in ("chau", "opera_rise"):
            # centre tone cluster
            cf = f_ref * np.array([1.0, 1.003, 2.01, 2.97])
            freqs = np.concatenate([freqs, cf])
            amps = np.concatenate([amps, np.array([2.5, 1.5, 0.6, 0.3]) * amps.max()])
            t60 = np.concatenate([t60, np.array([10, 9, 5, 3.0]) * P("decay")])
            bl = np.concatenate([bl, np.zeros(4)])
            g = 0.025 if pre == "chau" else -0.06
            glide = np.concatenate([glide, np.full(4, g)])
            gtau = np.concatenate([gtau, np.full(4, 0.6 if pre == "chau" else 0.25)])
        length = min(P("max_ring"), float(np.percentile(t60, 90)) * 1.1)
        if P("damp"):
            length = note.dur + 0.4
        n = int(length * sr)
        pans = P("width") * rng.uniform(-1, 1, len(freqs))
        y = render_modes(n, freqs, amps, t60, rng, pans, attack=0.001 + 0.004 * (1 - mal), bloom=bl,
                         glide=glide, glide_tau=gtau, floor=3e-5)
        # strike: low thump + mallet contact
        k = int(0.06 * sr)
        t = np.arange(k) / sr
        th = svf(rng.standard_normal(k), 90, 0.9, "lowpass") * np.exp(-t / 0.02)
        y[:, :k] += 0.4 * th / (np.max(np.abs(th)) + 1e-9) * np.max(np.abs(y[:, :int(0.3 * sr)]))
        if P("damp"):
            s = int(note.dur * sr)
            y[:, s:] *= np.exp(-np.arange(n - s) / sr / 0.12)
        y /= np.sqrt(nm) * 0.25
        y *= vel_amp(vel, 32) * 0.09
        y[:, :32] *= raised_cos(32)
        return fade(y, 0, min(1.5, 0.3 * length))


class CymbalSwell(Instrument):
    """Suspended cymbal rolled with soft mallets: crescendo over the written duration, then rings.
    vel may be a ramp (v0, v1); default ramps from near silence to vel."""
    name = "cymbal"
    pitched = False
    PARAMS = {
        "modes": (160, "resonator count per channel"),
        "roll_rate": (14.0, "mallet strokes per second"),
        "ring": (3.5, "ring time after the swell (s)"),
        "brightness": (0.5, "0 dark .. 1 bright"),
        "size": (0.5, "0 small/high .. 1 large/low"),
        "choke": (False, "stop dead at the written end (for a cut)"),
        "curve": (2.2, "crescendo shape exponent"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        swell = max(note.dur, 0.2)
        ring = 0.15 if P("choke") else P("ring")
        n = int((swell + ring) * sr)
        t = np.arange(n) / sr
        v = note.vel
        v0, v1 = (v if isinstance(v, (tuple, list)) else (0.02, float(v)))[:2]
        x = np.clip(t / swell, 0, 1) ** P("curve")
        env = (v0 + (v1 - v0) * x) * (t <= swell)
        # roll: alternating strokes with jitter
        strokes = np.zeros(n)
        tt = 0.0
        while tt < swell:
            i = int(tt * sr)
            strokes[i] += (1 + 0.15 * rng.standard_normal())
            tt += 1 / P("roll_rate") * (1 + 0.08 * rng.standard_normal())
        ker = np.hanning(int(0.008 * sr))
        exc_pulse = np.convolve(strokes, ker, "same")
        out = np.zeros((2, n))
        sz = P("size")
        for c in range(2):
            fr = np.exp(rng.uniform(np.log(300 * (1.4 - 0.8 * sz)), np.log(14000), int(P("modes"))))
            t60 = np.clip(2.5 * (fr / 2000) ** -0.35 * np.exp(0.3 * rng.standard_normal(len(fr))), 0.4, 6)
            # cymbal energy sits around 2-6 kHz; brightness moves the centre
            fcen = 2200 * 2 ** (1.4 * P("brightness"))
            g = np.exp(-np.log2(fr / fcen) ** 2 / (2 * 1.4 ** 2)) * np.exp(0.35 * rng.standard_normal(len(fr)))
            low = fr < 2500
            ex = svf(rng.standard_normal(n), 5000, 0.5, "lowpass") * exc_pulse * env
            # upper modes are fed by a brightening excitation (they grow faster than the low ones)
            ex_hi = svf(rng.standard_normal(n), 9000, 0.5, "lowpass") * exc_pulse * env ** 1.5
            y = modal_bank(ex, fr[low], t60[low], g[low]) + modal_bank(ex_hi, fr[~low], t60[~low], g[~low])
            wash = svf(rng.standard_normal(n), 5000, 0.7, "bandpass_norm") * signal.lfilter([0.002], [1, -0.998], env)
            out[c] = y / (np.std(y) + 1e-12) + 0.15 * wash / (np.std(wash) + 1e-12)
        out = signal.sosfilt(signal.butter(2, 250, "highpass", fs=sr, output="sos"), out, axis=-1)
        out = signal.sosfilt(signal.butter(2, 12000, "lowpass", fs=sr, output="sos"), out, axis=-1)
        out /= np.max(np.abs(out)) + 1e-9
        out *= 0.37 * vel_amp(max(v0, v1), 24)
        if P("choke"):
            s = int(swell * sr)
            out[:, s:] *= np.exp(-np.arange(n - s) / sr / 0.02)
        out[:, :16] *= raised_cos(16)
        return fade(out, 0, min(0.5, ring * 0.5))


class LampBell(Bell):
    """Small tuned handbell (Bell with preset='lamp'). Tune it to a chord tone."""
    name = "lamp_bell"

    def __init__(self, **kw):
        kw.setdefault("preset", "lamp")
        super().__init__(**kw)
