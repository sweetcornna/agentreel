"""Bells and glass (cool, glassy colours): modal synthesis with per-partial decay (higher partials die faster),
strike hardness shaping the high spectrum, doublet beating, partial-level stereo placement.

Presets
  glass     crystal / glass bowl struck softly: near-harmonic, slightly stretched partials
  church    church bell: hum, prime, tierce (minor third), quint, nominal + upper partials
  bianzhong Chinese chime bell: two strike tones a minor third apart (centre = A tone, side = B tone)
  celesta   struck tuned plate over a resonator: strong fundamental, faint 2.76/5.40 bar partials
  bowed     bowed glass: slow onset, sustained while bowed, beating partials, faint friction
  lamp      small tuned handbell: fundamental + twelfth, bright, short
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..core import SR, fade, midi_to_hz, raised_cos, smooth_noise, vel_amp
from ..dsp.modal import doublets, render_modes
from .base import Instrument, Note

# ratios, relative amplitudes, T60 (s) at the reference pitch, reference Hz, decay exponent
PRESETS = {
    "glass": dict(
        ratios=[1.0, 2.012, 3.041, 4.093, 5.172, 6.282, 7.43],
        amps=[1.0, 0.42, 0.24, 0.12, 0.07, 0.035, 0.02],
        t60=[9.0, 5.0, 3.2, 2.0, 1.3, 0.9, 0.6], ref=880.0, dexp=0.55, split=0.35, hard=0.35,
        noise=0.12, inharm=0.0006),
    "church": dict(
        ratios=[0.5, 1.0, 1.188, 1.505, 2.0, 2.52, 2.66, 3.01, 4.07, 5.02, 6.0],
        amps=[0.55, 0.5, 0.55, 0.22, 1.0, 0.35, 0.3, 0.32, 0.22, 0.12, 0.07],
        t60=[22.0, 12.0, 9.0, 5.0, 6.5, 3.5, 3.0, 2.6, 1.8, 1.2, 0.8], ref=262.0, dexp=0.75, split=0.6,
        hard=0.7, noise=0.35, inharm=0.003),
    "bianzhong": dict(
        ratios=[1.0, 1.189, 2.07, 2.58, 3.24, 4.12, 4.97, 5.86, 7.1],
        amps=[1.0, 0.3, 0.38, 0.3, 0.22, 0.16, 0.1, 0.07, 0.04],
        t60=[4.2, 3.8, 1.7, 1.3, 0.95, 0.7, 0.5, 0.38, 0.28], ref=440.0, dexp=0.6, split=0.5,
        hard=0.6, noise=0.3, inharm=0.004),
    "celesta": dict(
        ratios=[1.0, 2.0, 2.756, 5.404, 8.933],
        amps=[1.0, 0.07, 0.11, 0.045, 0.015],
        t60=[3.4, 1.2, 0.55, 0.25, 0.12], ref=523.0, dexp=0.7, split=0.15, hard=0.3, noise=0.18,
        inharm=0.0015),
    "bowed": dict(
        ratios=[1.0, 2.006, 3.022, 4.05, 5.1],
        amps=[1.0, 0.3, 0.12, 0.05, 0.02],
        t60=[6.0, 3.5, 2.2, 1.4, 1.0], ref=880.0, dexp=0.5, split=0.9, hard=0.2, noise=0.0, inharm=0.0008),
    "lamp": dict(
        ratios=[1.0, 2.98, 5.05, 6.9, 8.6],
        amps=[1.0, 0.55, 0.22, 0.12, 0.06],
        t60=[2.6, 1.4, 0.7, 0.45, 0.3], ref=1046.0, dexp=0.6, split=0.4, hard=0.75, noise=0.25,
        inharm=0.003),
}


# level calibration: a single vel-0.7 note peaks near -24 LUFS momentary for every preset
_PRESET_GAIN_DB = {"glass": -0.9, "church": 1.2, "bianzhong": 4.7, "celesta": 6.3, "bowed": 2.1, "lamp": 4.3}


class Bell(Instrument):
    """Modal bell / glass family (cool colours). Pitch = perceived strike pitch (for bianzhong: centre tone)."""
    name = "bell"
    PARAMS = {
        "preset": ("glass", "glass | church | bianzhong | celesta | bowed | lamp"),
        "hardness": (None, "mallet hardness 0 soft .. 1 hard (default from preset); scales high partials"),
        "decay": (1.0, "multiplies all T60s"),
        "brightness": (0.5, "tilts partial amplitudes up (1) or down (0)"),
        "strike": ("center", "bianzhong only: 'center' (A tone) or 'side' (B tone a minor third up)"),
        "beat": (1.0, "doublet beating amount (0 = none)"),
        "spread": (0.5, "stereo spread of partials"),
        "center": (None, "stereo centre of the partial cloud (default: from pitch)"),
        "damp": (False, "True: stop (choke) at the written end"),
        "attack": (None, "bowed only: onset time in seconds (default 0.8)"),
        "release": (None, "bowed only: fade after the bow stops (default: natural ring)"),
        "tremble": (0.0, "bowed only: slow amplitude waver 0..1"),
        "max_ring": (12.0, "cap on ring length (s)"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        pre = PRESETS[P("preset")]
        vel = float(np.clip(note.vel_scalar() + 0.025 * rng.standard_normal(), 0.02, 1))
        f0 = float(midi_to_hz(note.midi)) * (1 + 0.0005 * rng.standard_normal())
        ratios = np.array(pre["ratios"], dtype=float)
        # small geometry irregularities: each partial off by a little (stretched upwards)
        ratios = ratios * (1 + pre["inharm"] * rng.standard_normal(len(ratios)))
        amps = np.array(pre["amps"], dtype=float) * np.exp(0.12 * rng.standard_normal(len(ratios)))
        if P("preset") == "bianzhong" and P("strike") == "side":
            amps[0], amps[1] = 0.35, 1.0
            amps[2:] *= 1.15
        freqs = f0 * ratios
        t60 = np.array(pre["t60"], dtype=float) * (f0 / pre["ref"]) ** (-pre["dexp"]) * P("decay")
        t60 = np.clip(t60, 0.05, 40)
        hard = pre["hard"] if P("hardness") is None else float(P("hardness"))
        hard = np.clip(hard * (0.6 + 0.6 * vel), 0.02, 1.2)
        fc = 500 + 9000 * hard ** 2 + 3 * f0
        amps = amps / np.sqrt(1 + (freqs / fc) ** 2)
        bright = float(P("brightness"))
        amps = amps * (ratios ** ((bright - 0.5) * 1.2))

        bowed = P("preset") == "bowed"
        if bowed or P("damp"):
            length = note.dur + (min(t60.max(), 6.0) if bowed and P("release") is None else
                                 (P("release") or 0.25))
        else:
            length = min(P("max_ring"), float(t60.max()) * 1.05)
        length = max(length, 0.1)
        n = int(length * sr)

        split = pre["split"] * P("beat") * (0.6 + 0.8 * rng.random(len(freqs))) * np.sqrt(ratios)
        fr, am, tt = doublets(freqs, amps, t60, split, rng)
        pan0 = P("center")
        if pan0 is None:
            pan0 = float(np.clip((note.midi - 72) / 30.0, -0.6, 0.6))
        pans = np.clip(pan0 + P("spread") * 0.6 * rng.uniform(-1, 1, len(fr)), -1, 1)

        if bowed:
            att = P("attack") or 0.8
            y = render_modes(n, fr, am, np.full(len(fr), 1e3), rng, pans, attack=att * 0.35, phases="random")
            env = np.ones(n)
            t = np.arange(n) / sr
            stop = int(note.dur * sr)
            if stop < n:
                # after the bow leaves, each partial rings with its own decay: approximate with the
                # amplitude-weighted mean T60 (the lower partials dominate)
                trel = P("release") or float(np.sum(am * tt) / np.sum(am))
                env[stop:] = np.exp(-6.9 * (t[stop:] - t[stop]) / trel)
            if P("tremble") > 0:
                env *= 1 + 0.25 * P("tremble") * smooth_noise(n, 1.3, rng)
            y *= env
            # friction: faint band-limited noise riding on the fundamental
            fn = signal.lfilter(*signal.butter(2, [f0 * 0.9, f0 * 1.1], "bandpass", fs=sr), rng.standard_normal(n))
            fn *= 0.04 * env * np.clip(t / att, 0, 1)
            y += np.vstack([fn, np.roll(fn, 37)])
        else:
            att_s = 0.00025 + 0.002 * (1 - min(hard, 1))
            y = render_modes(n, fr, am, tt, rng, pans, attack=att_s)
            # strike transient: short filtered noise, brighter for hard mallets
            nz = pre["noise"] * hard
            if nz > 0:
                k = int(0.03 * sr)
                tn = np.arange(k) / sr
                bp_lo = min(1500 + 2500 * hard, 0.4 * sr)
                noise = signal.lfilter(*signal.butter(2, [bp_lo, min(bp_lo * 4, 0.45 * sr)], "bandpass", fs=sr),
                                       rng.standard_normal(k))
                noise *= np.exp(-tn / (0.002 + 0.004 * (1 - min(hard, 1)))) * raised_cos(k) ** 0.02
                noise *= nz * 0.5 * np.max(np.abs(y[:, : int(0.05 * sr)]) + 1e-9)
                y[:, :k] += np.vstack([noise, noise * 0.8])
            if P("damp"):
                stop = int(note.dur * sr)
                if stop < n:
                    tr = (np.arange(n - stop)) / sr
                    y[:, stop:] *= np.exp(-tr / 0.05)
        y *= vel_amp(vel, 30.0) * 0.18 * 10 ** (_PRESET_GAIN_DB[P("preset")] / 20)
        y[:, :16] *= raised_cos(16)
        return fade(y, 0, min(0.3, 0.25 * length))
