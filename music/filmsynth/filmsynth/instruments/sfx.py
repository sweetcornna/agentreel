"""Sound-effect primitives (all synthesised, all tunable where they have a pitch).

Tick       a very short tuned click (dry), e.g. a cursor tick; a warm one low, a cool one a fifth up.
FeltTick   a softer tick: felt or cork on wood, one low tuned resonance with a short ring
WoodKey    keyboard key: woody transient + 2-3 small resonances + bottom-out and release clicks
Sparkle    rising granular glitter (glyphs unravel into light)
FlamePop   soft 'puff' of a flame catching (lamp lights)
Puff       the small falling breath of a flame going out
Crunch     frost / grass / snow under a step
Hiss       moving hiss with Doppler-ish filter (light racing along a road)
Tap        a fingertip on paper or wood: pitchless, very short and dry (typing without a 'ding')
Breath     breath: an inhale that rises to the written end, a short 'hu' pulse, or a swell; with a pitch
           the air is coloured by a soft resonance there (air across a pipe, no tone)
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..core import SR, fade, midi_to_hz, raised_cos, smooth_noise, vel_amp
from ..dsp.filters import onepole_lp, svf
from .base import Instrument, Note


def _pan(mono, p):
    th = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return np.vstack([mono * np.cos(th), mono * np.sin(th)])


class Tick(Instrument):
    """Very short tuned click. pitch = the click's tone (e.g. B6 warm, F#7 cool)."""
    name = "tick"
    PARAMS = {
        "decay": (0.018, "tone decay time constant (s)"),
        "click": (0.35, "broadband edge amount"),
        "body": (0.3, "low wooden body under the tone"),
        "center": (0.0, "stereo position"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        f = float(midi_to_hz(note.midi)) * (1 + 0.001 * rng.standard_normal())
        n = int(0.15 * sr)
        t = np.arange(n) / sr
        dcy = P("decay") * (1 + 0.08 * rng.standard_normal())
        tone = np.sin(2 * np.pi * f * t) * np.exp(-t / dcy)
        tone += 0.25 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t / (dcy * 0.4))
        att = np.minimum(t / 0.0004, 1)  # 0.4 ms onset: crisp but not a digital step
        cl = svf(rng.standard_normal(n), min(4 * f, 15000), 1.0, "bandpass_norm") * np.exp(-t / 0.0012)
        body = svf(rng.standard_normal(n), 900, 2.0, "bandpass_norm") * np.exp(-t / 0.006)
        mono = att * (tone + P("click") * cl + P("body") * body)
        mono *= vel_amp(note.vel_scalar(), 24) * 0.3
        return fade(_pan(mono, P("center")), 0, 0.03)


class FeltTick(Instrument):
    """Soft cork / felt 'da': a dull noise transient into one tuned resonance with a short ring.

    pitch = the resonance (e.g. B5 for a warm tick); a faint wooden overtone at 2.76 x pitch.
    Rounder and lower than Tick; `ring` sets how long the resonance sounds."""
    name = "felt_tick"
    PARAMS = {
        "ring": (0.2, "T60 of the tuned resonance (s)"),
        "hardness": (0.3, "0 felt .. 1 hard cork: brightness, level and shortness of the transient"),
        "overtone": (0.3, "level of the wooden overtone (2.76 x pitch, decays 3x faster)"),
        "thump": (0.2, "low body thump (160 Hz, 8 ms)"),
        "center": (0.0, "stereo position"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        f = float(midi_to_hz(note.midi)) * (1 + 0.0015 * rng.standard_normal())
        ring = max(0.03, float(P("ring"))) * (1 + 0.05 * rng.standard_normal())
        hard = float(np.clip(P("hardness"), 0.0, 1.0))
        n = int((1.2 * ring + 0.02) * sr)
        t = np.arange(n) / sr
        tau = ring / 6.91                                   # T60 -> time constant
        on = 1.0 - np.exp(-t / (0.0015 - 0.001 * hard))    # felt stays on the wood a little longer
        tone = np.sin(2 * np.pi * f * t) * np.exp(-t / tau)
        tone += P("overtone") * np.sin(2 * np.pi * 2.76 * f * t + rng.uniform(0, 2 * np.pi)) * np.exp(-3 * t / tau)
        tone += P("thump") * np.sin(2 * np.pi * 160.0 * t) * np.exp(-t / 0.008)
        L = max(16, int((0.004 - 0.0025 * hard) * sr))     # mallet contact noise, 1.5-4 ms
        burst = np.zeros(n)
        burst[:L] = rng.standard_normal(L) * np.hanning(L)
        burst = svf(burst, 1500.0 + 5500.0 * hard, 0.7071, "lowpass")
        burst /= np.sqrt(np.mean(burst[:L] ** 2)) + 1e-12
        mono = on * tone + (0.15 + 0.35 * hard) * burst
        mono *= vel_amp(note.vel_scalar(), 24) * 0.3
        return fade(_pan(mono, P("center")), 0, min(0.02, 0.2 * ring))


class WoodKey(Instrument):
    """Wooden key press. pitch optional: tunes the strongest resonance (keeps typing in the key)."""
    name = "wood_key"
    pitched = False
    PARAMS = {
        "size": (0.5, "0 small/high .. 1 large/low"),
        "release_click": (0.5, "level of the key-up click"),
        "hold": (0.09, "seconds before the key-up click"),
        "center": (0.0, "stereo position"),
        "spread": (0.2, "random pan spread per key"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        n = int(0.3 * sr)
        t = np.arange(n) / sr
        sz = P("size")
        if np.isfinite(note.midi):
            f1 = float(midi_to_hz(note.midi))
        else:
            f1 = rng.uniform(900, 1500) * (1.5 - sz)
        res = [(f1, 0.02), (f1 * rng.uniform(2.3, 2.9), 0.008), (f1 * rng.uniform(4.1, 5.3), 0.004)]
        def hit(level, shift):
            y = np.zeros(n)
            exc = rng.standard_normal(int(0.002 * sr)) * np.hanning(int(0.002 * sr))
            x = np.zeros(n)
            s = int(shift * sr)
            x[s:s + len(exc)] = exc
            for f, d in res:
                y += svf(x, f * (1 + 0.02 * rng.standard_normal()), f * d * 40, "bandpass_norm") * 0.8
            y += svf(x, 3500, 0.8, "bandpass_norm") * 0.6  # wood tick
            thump = np.zeros(n)
            thump[s:] = np.sin(2 * np.pi * 180 * (t[: n - s])) * np.exp(-t[: n - s] / 0.006) * 0.4
            return level * (y + thump * np.max(np.abs(y) + 1e-9))
        mono = hit(1.0, 0.0) + hit(P("release_click") * 0.35, P("hold") * (1 + 0.25 * rng.standard_normal()))
        mono = signal.sosfilt(signal.butter(2, 60, "highpass", fs=sr, output="sos"), mono)
        mono /= np.max(np.abs(mono)) + 1e-9
        mono *= vel_amp(note.vel_scalar(), 24) * 0.5
        mono[:8] *= raised_cos(8)
        p = P("center") + P("spread") * rng.uniform(-1, 1)
        return fade(_pan(mono, p), 0, 0.05)


class Sparkle(Instrument):
    """Rising granular glitter over the written duration; pitch = lowest grain pitch."""
    name = "sparkle"
    PARAMS = {
        "density": (60.0, "grains per second"),
        "span": (24.0, "semitones the grain cloud rises across the note"),
        "scale": ((0, 2, 4, 7, 9), "pitch classes offsets for grains (pentatonic by default)"),
        "width": (0.9, "stereo width"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        d = note.dur
        n = int((d + 0.4) * sr)
        out = np.zeros((2, n))
        k = rng.poisson(P("density") * d)
        base = note.midi if np.isfinite(note.midi) else 86
        scale = np.array(P("scale"))
        for tg in np.sort(rng.uniform(0, d, k)):
            fr = tg / d
            m = base + P("span") * fr + rng.choice(scale) + 12 * rng.integers(0, 2)
            f = float(midi_to_hz(m))
            if f > 16000:
                continue
            L = rng.uniform(0.02, 0.08)
            g = int(L * sr)
            tt = np.arange(g) / sr
            s = np.sin(2 * np.pi * f * tt) * np.minimum(tt / 0.002, 1) * np.exp(-tt / (L * 0.35))
            s *= np.cos(0.5 * np.pi * np.clip((tt - 0.7 * L) / (0.3 * L), 0, 1)) ** 2  # grain ends at zero
            s *= rng.uniform(0.3, 1.0) * (0.3 + 0.7 * fr)
            i = int(tg * sr)
            p = P("width") * rng.uniform(-1, 1)
            out[:, i:i + g] += _pan(s, p)[:, : n - i]
        out *= vel_amp(note.vel_scalar(), 24) * 0.12
        return fade(out, 0.01, 0.3)


class FlamePop(Instrument):
    """Soft 'puff' of a flame catching: low breathy burst, quick downward filter, a few crackles."""
    name = "flame_pop"
    pitched = False
    PARAMS = {"size": (0.5, "0 small .. 1 large"), "crackle": (0.4, "crackle amount"), "center": (0.0, "pan")}

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        n = int(0.6 * sr)
        t = np.arange(n) / sr
        sz = P("size")
        env = (1 - np.exp(-t / 0.004)) * np.exp(-t / (0.06 + 0.08 * sz))
        fc = 300 + 2500 * np.exp(-t / 0.05) * (1.2 - 0.6 * sz)
        puff = svf(rng.standard_normal(n), fc, 0.9, "lowpass") * env
        thump = np.sin(2 * np.pi * (90 + 60 * (1 - sz)) * t) * np.exp(-t / 0.03) * (1 - np.exp(-t / 0.003)) * 0.5
        cr = np.zeros(n)
        for _ in range(rng.poisson(6 * P("crackle"))):
            i = int(rng.uniform(0.01, 0.35) * sr)
            cr[i] += rng.uniform(0.2, 1.0)
        cr = svf(cr, 3000, 2.0, "bandpass_norm") * 0.6
        mono = puff / (np.max(np.abs(puff)) + 1e-9) + thump + cr
        mono = signal.sosfilt(signal.butter(2, 30, "highpass", fs=sr, output="sos"), mono)
        mono *= vel_amp(note.vel_scalar(), 24) * 0.4
        return fade(_pan(mono, P("center")), 0, 0.1)


class Puff(Instrument):
    """Flame going out: a short falling breath."""
    name = "puff"
    pitched = False
    PARAMS = {"length": (0.25, "s"), "center": (0.0, "pan")}

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        L = P("length") * (1 + 0.1 * rng.standard_normal())
        n = int((L + 0.1) * sr)
        t = np.arange(n) / sr
        env = np.sin(np.pi * np.clip(t / L, 0, 1)) ** 1.5 * np.exp(-t / (L * 0.6))
        fc = 2200 * (1 - 0.7 * np.clip(t / L, 0, 1)) + 300
        mono = svf(rng.standard_normal(n), fc, 1.2, "bandpass_norm") * env
        mono *= vel_amp(note.vel_scalar(), 24) * 0.36
        return fade(_pan(mono, P("center")), 0, 0.03)


class Crunch(Instrument):
    """Step on frost / grass / snow: dense micro-cracks shaped by the material."""
    name = "crunch"
    pitched = False
    PARAMS = {"material": ("frost", "frost | grass | snow"), "length": (0.16, "s"), "center": (0.0, "pan")}

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        mat = P("material")
        L = P("length") * (1 + 0.15 * rng.standard_normal())
        n = int((L + 0.08) * sr)
        t = np.arange(n) / sr
        env = np.clip(t / 0.01, 0, 1) * np.exp(-t / (L * 0.45))
        rate = {"frost": 2500, "grass": 900, "snow": 1800}[mat]
        imp = (rng.random(n) < rate / sr) * rng.pareto(2.5, n)
        if mat == "frost":
            y = svf(imp, 7000, 1.2, "bandpass_norm") + 0.4 * svf(imp, 3500, 1.5, "bandpass_norm")
        elif mat == "grass":
            sw = svf(rng.standard_normal(n), 2500, 0.6, "bandpass_norm") * 0.3
            y = svf(imp, 2200, 1.0, "bandpass_norm") + sw
        else:
            y = svf(imp, 1300, 2.5, "bandpass_norm") + 0.5 * svf(imp, 4500, 1.0, "bandpass_norm")
            y += 0.2 * svf(rng.standard_normal(n), 900, 4.0, "bandpass_norm")  # squeak
        mono = y * env
        mono /= np.max(np.abs(mono)) + 1e-9
        mono *= vel_amp(note.vel_scalar(), 24) * 0.38
        mono[:16] *= raised_cos(16)
        return fade(_pan(mono, P("center")), 0, 0.03)


class Hiss(Instrument):
    """Moving hiss for light travelling along a road: pans from -> to with a Doppler-like filter."""
    name = "hiss"
    pitched = False
    PARAMS = {"pan_from": (-0.8, "start pan"), "pan_to": (0.8, "end pan"), "centre_hz": (4000.0, "band centre"),
              "doppler": (0.5, "filter sweep amount as it passes")}

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        d = max(note.dur, 0.05)
        n = int((d + 0.15) * sr)
        t = np.arange(n) / sr
        x = np.clip(t / d, 0, 1)
        env = np.sin(np.pi * x) ** 1.2
        fc = P("centre_hz") * 2 ** (P("doppler") * (0.6 - 1.2 * x))
        mono = svf(rng.standard_normal(n), np.clip(fc, 200, 16000), 1.5, "bandpass_norm") * env
        p = P("pan_from") + (P("pan_to") - P("pan_from")) * x
        th = (np.clip(p, -1, 1) + 1) * np.pi / 4
        out = np.vstack([mono * np.cos(th), mono * np.sin(th)])
        out = signal.sosfilt(signal.butter(2, 11000, fs=sr, output="sos"), out, axis=-1)
        out *= vel_amp(note.vel_scalar(), 24) * 0.11
        return fade(out, 0.005, 0.05)


class Tap(Instrument):
    """A fingertip on paper or wood: pitchless, very short and dry. A few milliseconds of shaped noise
    and a damped knock below 1 kHz that is over before it can ring (no tuned resonance, no 'ding')."""
    name = "tap"
    pitched = False
    PARAMS = {
        "material": (0.5, "0 paper (a soft papery brush) .. 1 wood (a duller knock)"),
        "tone": (1600.0, "centre of the tap's colour (Hz)"),
        "length": (0.012, "length of the contact noise (s)"),
        "center": (0.0, "stereo position"),
        "spread": (0.15, "random pan spread per tap"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        mat = float(np.clip(P("material"), 0, 1))
        L = max(0.003, float(P("length")) * (1 + 0.15 * rng.standard_normal()))
        n = int((L + 0.06) * sr)
        t = np.arange(n) / sr
        env = np.minimum(t / 0.0008, 1.0) * np.exp(-t / (0.3 * L))
        fc = float(P("tone")) * (1 - 0.4 * mat) * (1 + 0.08 * rng.standard_normal())
        nz = svf(rng.standard_normal(n), fc, 0.6, "bandpass_norm") * env
        # paper: a soft brush of fibre after the contact; wood: a knock (heavily damped, < 1 kHz)
        brush = svf(rng.standard_normal(n), 3200.0, 0.5, "bandpass_norm") * np.exp(-t / (0.006 + 0.01 * (1 - mat)))
        brush *= np.minimum(t / 0.002, 1.0) * (1 - mat) * 0.35
        fk = 320.0 + 260.0 * (1 - mat) + 40 * rng.standard_normal()
        knock = np.sin(2 * np.pi * fk * t) * np.exp(-t / 0.0045) * np.minimum(t / 0.0006, 1.0) * mat * 0.8
        mono = nz / (np.max(np.abs(nz)) + 1e-12) + brush / (np.max(np.abs(nz)) + 1e-12) * 0.5 + knock
        mono = signal.sosfilt(signal.butter(2, 90, "highpass", fs=sr, output="sos"), mono)
        mono *= vel_amp(note.vel_scalar(), 24) * 0.18
        p = float(P("center")) + float(P("spread")) * rng.uniform(-1, 1)
        return fade(_pan(mono, p), 0, 0.02)


class Breath(Instrument):
    """Breath. shape='inhale': air drawn in, rising in level and colour, stopping at the written end
    (like a breath before a word); 'pulse': a short 'hu' (breath pulses of a crowd); 'swell': in and out
    over the note. With a pitch, the air is coloured by a soft resonance there (and an octave above),
    like air across the mouth of a pipe: a hint of pitch, no tone. Without a pitch the colour is set by
    `formant`."""
    name = "breath"
    PARAMS = {
        "shape": ("pulse", "inhale | pulse | swell"),
        "attack": (0.015, "pulse / swell: onset time (s)"),
        "decay": (0.08, "pulse: decay time constant (s)"),
        "formant": (1100.0, "centre of the breath colour (Hz) when no pitch is given"),
        "rise": (7.0, "inhale: semitones the colour rises across the note"),
        "resonance": (0.5, "0 plain air .. 1 strongly coloured by the pitch (still no tone)"),
        "brightness": (0.4, "0 dark (mouth) .. 1 airy (teeth, hiss)"),
        "center": (0.0, "stereo position"),
        "width": (0.3, "stereo width"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        shape = P("shape")
        d = max(note.dur, 0.03)
        if shape == "pulse":
            dec = max(0.01, float(P("decay"))) * (1 + 0.1 * rng.standard_normal())
            n = int((float(P("attack")) + 7 * dec) * sr)
        else:
            n = int((d + 0.06) * sr)
        t = np.arange(n) / sr
        if shape == "inhale":
            x = np.clip(t / d, 0, 1)
            env = x ** 1.8 * (1 + 0.08 * smooth_noise(n, 3.0, rng, sr))
            env *= np.clip(1 - (t - d) / 0.025, 0, 1)                # stops at the written end
            col = 2 ** (float(P("rise")) * (x - 1) / 12)               # the colour rises to its end value
        elif shape == "swell":
            att = max(0.02, float(P("attack")))
            env = np.sin(np.pi * np.clip(t / d, 0, 1)) ** 1.5 * np.minimum(t / att, 1.0)
            col = np.ones(n)
        else:
            att = max(0.003, float(P("attack")))
            env = (1 - np.exp(-t / (att / 3))) * np.exp(-np.maximum(t - att, 0) / dec)
            col = 1 - 0.06 * np.clip(t / (4 * dec), 0, 1)              # falls a little as the breath runs out
        br = float(np.clip(P("brightness"), 0, 1))
        out = np.zeros((2, n))
        pitched = np.isfinite(note.midi)
        fp = float(midi_to_hz(note.midi)) if pitched else float(P("formant"))
        res = float(np.clip(P("resonance"), 0, 1)) if pitched else 0.0
        for c in range(2):
            src = rng.standard_normal(n)
            air = svf(src, np.clip(fp * 1.6 * col, 300, 9000), 0.55, "bandpass_norm")
            hiss = svf(rng.standard_normal(n), np.clip(3500 * col, 1500, 12000), 0.7, "bandpass_norm") * br
            y = air + 0.6 * hiss
            if res > 0:
                q = 4 + 10 * res
                y = (1 - 0.6 * res) * y + res * 1.2 * (svf(src, np.clip(fp * col, 60, 12000), q, "bandpass_norm")
                                                     + 0.4 * svf(src, np.clip(2 * fp * col, 60, 15000), q, "bandpass_norm"))
            out[c] = y
        w = float(P("width"))
        mid, side = 0.5 * (out[0] + out[1]), 0.5 * (out[0] - out[1])
        out = np.vstack([mid + w * side, mid - w * side]) * env
        out /= np.sqrt(np.mean(out[:, env > 0.5 * env.max()] ** 2)) + 1e-12
        cth = (np.clip(float(P("center")), -1, 1) + 1) * np.pi / 4
        out = np.vstack([out[0] * np.cos(cth), out[1] * np.sin(cth)]) * np.sqrt(2)
        out *= vel_amp(note.vel_scalar(), 24) * 0.05
        return fade(out, 0.002, 0.02)
