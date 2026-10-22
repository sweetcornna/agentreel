"""Keyboard instruments.

FeltPiano  a felt-muted upright: a soft felt hammer on three unison strings, rendered as modes
           (stiff-string partials, unison mistuning, two-stage decay, hammer-position comb, felt
           contact low-pass), with the soft 'thock' of the felt mechanism and a wooden soundboard.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..core import SR, fade, midi_to_hz, raised_cos, vel_amp
from ..dsp.filters import svf
from ..dsp.modal import render_modes
from .base import Instrument, Note
from .pluck import _body_irs


class FeltPiano(Instrument):
    """Felt piano: a felt-muted hammer on unison strings (modal synthesis). Warm and dark, with a soft onset, a
    long two-stage decay (prompt sound, then aftersound) and the soft 'thock' of the felt mechanism. Its partials
    are harmonic (slightly stretched), so it reads as a struck string, not as a bell. pitch = the string; C2-C6."""
    name = "felt_piano"
    PARAMS = {
        "felt": (0.7, "felt mute 0 bare hammer .. 1 thick felt: darker, softer onset, more mechanism noise"),
        "decay": (1.0, "multiplies the natural T60s (C4: aftersound ~14 s, prompt sound ~4 s)"),
        "strings": (3, "unison strings 1..3"),
        "detune": (0.7, "unison mistuning std (cents): the slow beating of a real unison"),
        "hammer_pos": (0.12, "strike point as a fraction of the string (1/8 is the classic)"),
        "inharm": (1.0, "scales the string stiffness (stretch of the upper partials)"),
        "body": (0.45, "soundboard resonance level"),
        "noise": (0.5, "mechanism noise: felt on the string, hammer shank, key bed"),
        "let_ring": (True, "True: pedal down, the note rings on; False: the damper falls at the written end"),
        "damp_time": (0.18, "T60 (s) once the damper is down"),
        "max_ring": (12.0, "cap on ring length (s)"),
        "spread": (0.3, "stereo placement by pitch (low left, high right, as the player hears it)"),
        "center": (0.0, "stereo centre"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        vel = float(np.clip(note.vel_scalar() + 0.015 * rng.standard_normal(), 0.02, 1.0))
        f0 = float(midi_to_hz(note.midi)) * (1 + 0.0004 * rng.standard_normal())
        felt = float(np.clip(P("felt"), 0.0, 1.0))

        # stiff string: f_k = k f0 sqrt(1 + B k^2), B grows up the compass
        B = float(np.clip(3.8e-4 * (f0 / 261.6) ** 1.1 * P("inharm"), 2e-5, 0.02))
        kmax = 1
        while kmax < 48 and (kmax + 1) * f0 * np.sqrt(1 + B * (kmax + 1) ** 2) < min(12000.0, 0.45 * sr):
            kmax += 1
        k = np.arange(1, kmax + 1, dtype=np.float64)
        fk = k * f0 * np.sqrt(1 + B * k * k)

        # felt hammer: a low-pass whose corner rises with velocity and falls with the felt
        fc = (400.0 + 1800.0 * (1 - felt)) * 2 ** (1.6 * (vel - 0.5))
        fc = max(fc, 2.2 * f0)
        hammer = 1.0 / (1.0 + (fk / fc) ** 2.5)
        x0 = float(np.clip(P("hammer_pos") + 0.004 * rng.standard_normal(), 0.04, 0.5))
        comb = 0.08 + np.abs(np.sin(np.pi * k * x0))          # a hammer has width: the nulls are not perfect
        radiate = 1.0 / np.sqrt(1.0 + (110.0 / fk) ** 2)        # the soundboard radiates little below ~110 Hz
        # k^-0.5: the felt's mass and softness favour the fundamental (soft notes are fundamental-led)
        amp = hammer * comb * k ** -0.5 * radiate * np.exp(0.12 * rng.standard_normal(kmax))

        # decays: aftersound long in the bass, the upper partials die faster; the prompt sound is ~1/3 of it
        t_after = float(np.clip(14.0 * (f0 / 261.6) ** -0.55, 2.5, 30.0)) * float(P("decay"))
        t_k = t_after * (fk / f0) ** -0.8
        t_prompt = t_k * 0.3

        # unison strings, each with its own mistuning; two decay stages per partial (the polarisations)
        ns = int(np.clip(P("strings"), 1, 3))
        det = float(P("detune"))
        cents = np.array([0.0]) if ns == 1 else np.linspace(-1, 1, ns) * det + 0.3 * det * rng.standard_normal(ns)
        base_pan = float(P("center")) + float(P("spread")) * float(np.clip((note.midi - 60) / 30.0, -1, 1))
        freqs, amps, t60s, pans = [], [], [], []
        for c in cents:
            g = (1.0 / ns) * np.exp(0.25 * rng.standard_normal())   # unequal strings: the beats never cancel
            fs_ = fk * 2 ** (c / 1200)
            sp = base_pan + 0.06 * rng.uniform(-1, 1)
            for share, tt, df in ((0.55, t_prompt, 0.0), (0.45, t_k, rng.uniform(0.02, 0.07))):
                freqs.append(fs_ + df)
                amps.append(amp * g * share)
                t60s.append(tt * np.exp(0.06 * rng.standard_normal(kmax)))
                pans.append(sp + 0.05 * rng.uniform(-1, 1, kmax))
        freqs, amps, t60s, pans = (np.concatenate(v) for v in (freqs, amps, t60s, pans))

        if P("let_ring"):
            length = min(float(P("max_ring")), t_after * 1.05)
            damp_at = None
        else:
            damp_at = max(0.03, note.dur)
            length = min(float(P("max_ring")), damp_at + float(P("damp_time")) * 1.3 + 0.05)
        n = int(length * sr)
        tc = (0.0025 + 0.004 * felt) * (1.25 - 0.5 * vel)        # contact time: soft felt stays on the string
        y = render_modes(n, freqs, amps, t60s, rng, pans, attack=tc * 0.35, floor=3e-5)

        # soundboard: the same wooden body family as the pluck (large, dark), seen from two sides
        body = float(P("body"))
        if body > 0:
            irs = _body_irs(int(note.seed % 7), 0.85)
            y = np.vstack([y[0] + body * 0.6 * signal.oaconvolve(y[0], irs[0])[:n],
                           y[1] + body * 0.6 * signal.oaconvolve(y[1], irs[1])[:n]])

        # level (after the body, so that the body only colours): equal loudness across the compass
        k0 = min(n, int(0.3 * sr))
        y = y / (np.sqrt(np.mean(y[:, :k0] ** 2)) + 1e-12) * 0.25 * (f0 / 247.0) ** -0.1

        # the felt mechanism: a dark contact noise and the damped knock of the hammer shank and key bed
        nz_amt = float(P("noise")) * (0.3 + 0.7 * felt) * (0.5 + 0.5 * vel)
        if nz_amt > 0:
            kk = min(n, int(0.06 * sr))
            t = np.arange(kk) / sr
            burst = svf(rng.standard_normal(kk), 900.0 + 600.0 * vel, 0.7, "lowpass")
            burst = svf(burst, 140.0, 0.7, "highpass")
            burst *= (1 - np.exp(-t / 0.0015)) * np.exp(-t / 0.012)
            knock = np.sin(2 * np.pi * (165.0 + 25.0 * rng.standard_normal()) * t) * np.exp(-t / 0.018)
            knock *= 1 - np.exp(-t / 0.002)
            mech = burst / (np.max(np.abs(burst)) + 1e-12) + 0.6 * knock
            pk = np.max(np.abs(y[:, :kk])) + 1e-12
            th = (np.clip(base_pan, -1, 1) + 1) * np.pi / 4
            y[0, :kk] += nz_amt * 0.22 * pk * mech * np.cos(th) * np.sqrt(2)
            y[1, :kk] += nz_amt * 0.22 * pk * mech * np.sin(th) * np.sqrt(2)

        if damp_at is not None:
            i0 = int(damp_at * sr)
            tt = np.arange(n - i0) / sr
            y[:, i0:] *= np.exp(-6.907755 * tt / max(float(P("damp_time")), 0.01))

        y *= vel_amp(vel, 30.0) * 0.58                          # vel 0.7: ~-24 LUFS (0.4 s), as the library
        y[:, :32] *= raised_cos(32)
        y = fade(y, 0.0, min(0.3, 0.2 * length))
        return signal.lfilter([1, -1], [1, -0.9995], y, axis=-1)   # no DC from the mechanism noise
