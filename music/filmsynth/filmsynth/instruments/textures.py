"""Texture layers. Each 'note' renders the texture for its written duration (fades included).

Wind        filtered noise, gust envelope, occasional whistling resonances (tunable to a pitch)
Blizzard    broadband storm noise + granular ice grains, gusting
AirRush     band sweeping upward with a Doppler whistle and moving pan (cloud punch-through)
Shimmer     granular high partials on chord tones (sunrise glitter)
Drone       sustained layer of slowly beating partials (pitch = root)
Riser       noise + tone rise toward the end of the note (lands exactly at the written end)
ReverseSwell reverse-reverb inhale that peaks exactly at the written end
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..core import SR, fade, midi_to_hz, raised_cos, smooth_noise, to_midi, vel_amp
from ..dsp.delay import vardelay
from ..dsp.filters import onepole_lp, svf
from ..dsp.reverb import convolve, make_ir
from .base import Instrument, Note, vel_curve


def _pink(n, rng):
    """Approximate pink noise (Voss-McCartney-like via filtering white noise)."""
    w = rng.standard_normal(n)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    p = signal.lfilter(b, a, w)
    return p / (np.std(p) + 1e-12)


def _edges(y, fade_in, fade_out):
    return fade(y, fade_in, fade_out)


class Wind(Instrument):
    """Night wind: two decorrelated filtered noises with gusts and rare whistles."""
    name = "wind"
    pitched = False
    PARAMS = {
        "intensity": (0.5, "0 still air .. 1 strong wind (level, brightness, gust depth)"),
        "gustiness": (0.5, "gust depth and rate"),
        "whistle": (0.25, "amount of whistling resonances (0 none)"),
        "whistle_pitch": (None, "tune whistles to this pitch (and its octave/fifth); None = free"),
        "brightness": (0.5, "0 dull rumble .. 1 airy hiss"),
        "fade_in": (1.5, "s"),
        "fade_out": (1.5, "s"),
        "intensity_end": (None, "if set, intensity ramps to this value across the note"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        n = int(note.dur * sr)
        inten = np.full(n, P("intensity"))
        if P("intensity_end") is not None:
            inten = np.linspace(P("intensity"), P("intensity_end"), n)
        g = P("gustiness")
        gust = 0.5 + 0.5 * np.tanh(1.3 * smooth_noise(n, 0.12 + 0.25 * g, rng) + 0.6 * smooth_noise(n, 0.6 + g, rng))
        level = (0.25 + 0.75 * gust * (0.4 + 0.6 * g)) * inten
        out = np.zeros((2, n))
        br = P("brightness")
        for c in range(2):
            src = _pink(n, rng)
            fc = (250 + 2500 * br) * (0.5 + 1.2 * gust * inten) * 2 ** (0.3 * smooth_noise(n, 0.3, rng))
            body = svf(src, np.clip(fc, 80, 12000), 0.9, "lowpass")
            hiss = svf(rng.standard_normal(n), np.clip(fc * 3, 500, 7000), 0.7, "bandpass_norm") * 0.15 * br
            out[c] = (body + hiss) * level
            # whistles: narrow resonances whose pitch rides the gust
            if P("whistle") > 0:
                if P("whistle_pitch") is not None:
                    base = float(midi_to_hz(to_midi(P("whistle_pitch"))))
                    choices = [base, base * 2, base * 1.5]
                else:
                    choices = [rng.uniform(500, 1400)]
                wf = choices[rng.integers(len(choices))] * 2 ** (0.08 * smooth_noise(n, 0.4, rng)) * (0.97 + 0.06 * gust)
                gate = np.clip((gust - 0.6) / 0.3, 0, 1) ** 2 * inten
                ws = svf(rng.standard_normal(n), wf, 60.0, "bandpass_norm")
                out[c] += P("whistle") * 0.5 * ws * gate
        out = signal.sosfilt(signal.butter(2, 30, "highpass", fs=sr, output="sos"), out, axis=-1)
        out *= vel_amp(note.vel_scalar(), 30) * 0.5
        return _edges(out, P("fade_in"), P("fade_out"))


class Blizzard(Instrument):
    """Storm: broadband gusting noise + ice grains hitting (granular clicks, panned)."""
    name = "blizzard"
    pitched = False
    PARAMS = {
        "intensity": (0.7, "0..1"),
        "intensity_end": (None, "ramp target across the note"),
        "grains": (600.0, "ice grains per second at full intensity"),
        "grain_level": (0.5, "ice grain level"),
        "part": ("all", "all | noise | grains: render only the storm noise or only the ice grains (the grains are "
                        "the same either way, so the two can sit on separate tracks)"),
        "fade_in": (1.0, "s"),
        "fade_out": (0.02, "s (storms are usually cut, not faded)"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        n = int(note.dur * sr)
        inten = np.full(n, P("intensity"))
        if P("intensity_end") is not None:
            inten = np.linspace(P("intensity"), P("intensity_end"), n)
        gust = 0.5 + 0.5 * np.tanh(1.5 * smooth_noise(n, 0.35, rng) + 0.8 * smooth_noise(n, 1.4, rng))
        out = np.zeros((2, n))
        for c in range(2):
            src = 0.6 * _pink(n, rng) + 0.4 * rng.standard_normal(n)
            fc = np.clip((700 + 6500 * gust * inten), 200, 8000)
            body = svf(src, fc, 0.8, "lowpass")
            roar = svf(_pink(n, rng), 140 + 200 * gust, 1.5, "bandpass_norm") * 1.2
            out[c] = (body + roar) * (0.3 + 0.7 * gust) * inten
        # ice grains: Poisson clicks, brightness by size, random pan
        rate = P("grains") * np.mean(inten)
        k = rng.poisson(rate * n / sr)
        pos = rng.integers(0, n, k)
        keep = rng.random(k) < (inten[pos] * gust[pos])
        pos = pos[keep]
        grains = np.zeros((2, n))
        amps = rng.pareto(3.0, len(pos)) * 0.3
        pans = rng.uniform(-1, 1, len(pos))
        grains[0, pos] += amps * np.cos((pans + 1) * np.pi / 4)
        grains[1, pos] += amps * np.sin((pans + 1) * np.pi / 4)
        # each grain = tiny resonant tick 4-9 kHz
        for c in range(2):
            grains[c] = svf(grains[c], 6500, 3.0, "bandpass_norm") + 0.3 * svf(grains[c], 3200, 2.0, "bandpass_norm")
        part = P("part")
        if part == "grains":
            out = P("grain_level") * grains * 3.0
        elif part == "noise":
            pass
        else:
            out += P("grain_level") * grains * 3.0
        out = signal.sosfilt(signal.butter(2, 35, "highpass", fs=sr, output="sos"), out, axis=-1)
        out *= vel_amp(note.vel_scalar(), 30) * 0.17
        return _edges(out, P("fade_in"), max(P("fade_out"), 0.003))


class AirRush(Instrument):
    """Huge air rush through clouds: a noise band sweeping upward, a whistle with Doppler shift and
    a pan sweep, sub rumble underneath. Peaks at `peak` (fraction of the note) then falls away."""
    name = "air_rush"
    pitched = False
    PARAMS = {
        "f_start": (180.0, "band centre at the start (Hz)"),
        "f_end": (5000.0, "band centre at the peak (Hz)"),
        "peak": (0.85, "where the rush peaks (fraction of the note); after it, a fast decay"),
        "pan_from": (-0.8, "pan at the start"),
        "pan_to": (0.8, "pan at the end"),
        "doppler": (0.5, "whistle Doppler amount"),
        "sub": (0.5, "sub rumble level"),
        "tail": (0.6, "seconds after the written end"),
        "whistle": (1.0, "level of the narrow Doppler whistle inside the rush (0 = air only)"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        n = int((note.dur + P("tail")) * sr)
        t = np.arange(n) / sr
        pk = P("peak") * note.dur
        x = np.clip(t / max(pk, 1e-3), 0, 1)
        env = np.where(t <= pk, x ** 2.2, np.exp(-(t - pk) / 0.25))
        fc = P("f_start") * (P("f_end") / P("f_start")) ** np.clip(t / max(pk, 1e-3), 0, 1.15)
        src = rng.standard_normal(n)
        band = svf(src, np.clip(fc, 40, 16000), 1.4, "bandpass_norm") + 0.5 * svf(_pink(n, rng), np.clip(fc * 0.5, 30, 12000), 0.8, "lowpass")
        # Doppler whistle: a source passing by at speed -> pitch falls through the pass
        v = 60.0 * P("doppler")
        d0 = 30.0
        xpos = (t - pk) * v
        radial = xpos / np.sqrt(xpos ** 2 + d0 ** 2)
        fw = 900.0 * (343.0 / (343.0 + v * radial)) * (fc / fc.max()) ** 0.25
        wh = svf(rng.standard_normal(n), np.clip(fw, 100, 12000), 40.0, "bandpass_norm") * 0.5 * P("whistle")
        sub = svf(_pink(n, rng), 45, 1.2, "lowpass") * 2.0 * P("sub")
        pan = np.clip(P("pan_from") + (P("pan_to") - P("pan_from")) * np.clip(t / max(note.dur, 1e-3), 0, 1), -1, 1)
        th = (pan + 1) * np.pi / 4
        mono = (band + wh) * env
        out = np.vstack([mono * np.cos(th) * 1.4, mono * np.sin(th) * 1.4]) + (sub * env)[None, :]
        out = signal.sosfilt(signal.butter(2, 9000, fs=sr, output="sos"), out, axis=-1)  # air, not fizz
        out = signal.sosfilt(signal.butter(2, 20, "highpass", fs=sr, output="sos"), out, axis=-1)
        out *= vel_amp(note.vel_scalar(), 30) * 0.21
        out[:, :32] *= raised_cos(32)
        return fade(out, 0, 0.1)


class Shimmer(Instrument):
    """Granular glitter of high partials on chord tones. pitch = lowest tone; chord = intervals."""
    name = "shimmer"
    PARAMS = {
        "chord": ((0, 4, 7, 9, 12, 16, 19, 24), "semitone offsets from the pitch (default D gong-ish)"),
        "density": (18.0, "grains per second"),
        "grain": (0.09, "mean grain length (s)"),
        "octaves": (2, "spread grains up this many octaves"),
        "rise": (0.0, "0 static; >0 grains drift upward over the note (glyphs to light)"),
        "fade_in": (0.6, "s"),
        "fade_out": (1.2, "s"),
        "width": (1.0, "stereo width"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        n = int((note.dur + 0.5) * sr)
        out = np.zeros((2, n))
        dens = P("density")
        k = rng.poisson(dens * note.dur)
        times = np.sort(rng.uniform(0, note.dur, k))
        chord = np.array(P("chord"), dtype=float)
        for tg in times:
            frac = tg / max(note.dur, 1e-3)
            m = note.midi + rng.choice(chord) + 12 * rng.integers(0, P("octaves") + 1)
            m += P("rise") * 12 * frac * rng.uniform(0.3, 1.0)
            f = float(midi_to_hz(m))
            if f > 15000:
                continue
            L = max(0.02, rng.gamma(4, P("grain") / 4))
            g = int(L * sr)
            tt = np.arange(g) / sr
            w = np.sin(np.pi * tt / L) ** 2 * np.exp(-tt / (L * 0.8))
            s = np.sin(2 * np.pi * f * tt + rng.uniform(0, 6.28))
            if 2.01 * f < 0.45 * sr:
                s = s + 0.2 * np.sin(2 * np.pi * 2.01 * f * tt)
            s *= w * rng.uniform(0.3, 1.0) * (f / 2000) ** -0.25
            pan = P("width") * rng.uniform(-1, 1)
            i = int(tg * sr)
            e = min(n, i + g)
            out[0, i:e] += s[: e - i] * np.cos((pan + 1) * np.pi / 4)
            out[1, i:e] += s[: e - i] * np.sin((pan + 1) * np.pi / 4)
        out *= vel_amp(note.vel_scalar(), 30) * 0.24
        return fade(out, P("fade_in"), P("fade_out"))


class Drone(Instrument):
    """Sustained beating drone: several slightly detuned partials per chord tone, slow filter drift."""
    name = "drone"
    PARAMS = {
        "chord": ((0, 7, 12), "semitone offsets from the pitch"),
        "beat_hz": (0.25, "typical beating rate between detuned copies"),
        "harmonics": (5, "harmonics per tone"),
        "brightness": (0.4, "0 dark .. 1 bright"),
        "attack": (2.0, "s"),
        "release": (2.5, "s"),
        "motion": (0.5, "slow timbral motion 0..1"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        n = int((note.dur + P("release")) * sr)
        t = np.arange(n) / sr
        out = np.zeros((2, n))
        for iv in P("chord"):
            f = float(midi_to_hz(note.midi + iv))
            for c in range(3):
                det = (c - 1) * P("beat_hz") * rng.uniform(0.6, 1.4)
                for h in range(1, int(P("harmonics")) + 1):
                    fh = f * h + det * h ** 0.5
                    if fh > 16000:
                        break
                    a = h ** (-1.6 + 0.8 * P("brightness"))
                    a *= 1 + P("motion") * 0.5 * smooth_noise(n, 0.08 + 0.03 * h, rng)
                    s = a * np.sin(2 * np.pi * fh * t + rng.uniform(0, 6.28))
                    pan = rng.uniform(-0.7, 0.7)
                    out[0] += s * np.cos((pan + 1) * np.pi / 4)
                    out[1] += s * np.sin((pan + 1) * np.pi / 4)
        env = np.clip(t / P("attack"), 0, 1) ** 2
        rel = np.clip(1 - (t - note.dur) / P("release"), 0, 1)
        out *= env * rel * rel
        out /= np.sqrt(len(P("chord")) * 3)
        out *= vel_amp(note.vel_scalar(), 30) * 0.134
        return fade(out, 0.01, 0.05)


class Riser(Instrument):
    """Rising noise + tone that lands exactly at the written end (pitch = landing tone)."""
    name = "riser"
    PARAMS = {
        "span": (24.0, "semitones the tone rises over the note"),
        "noise": (0.6, "noise amount"),
        "tone": (0.4, "tone amount"),
        "curve": (2.5, "crescendo exponent"),
        "tail": (0.15, "decay after the end (s)"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        d = note.dur
        n = int((d + P("tail")) * sr)
        t = np.arange(n) / sr
        x = np.clip(t / d, 0, 1)
        env = np.where(t <= d, x ** P("curve"), np.exp(-(t - d) / max(P("tail") / 4, 1e-3)))
        m_end = note.midi if np.isfinite(note.midi) else 86
        m = m_end - P("span") * (1 - x)
        f = midi_to_hz(m)
        ph = 2 * np.pi * np.cumsum(f) / sr
        tone = np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)
        nz = svf(rng.standard_normal(n), np.clip(f * 4, 200, 15000), 1.5, "bandpass_norm")
        mono = (P("tone") * tone * 0.5 + P("noise") * nz) * env
        out = np.vstack([mono, vardelay(mono, np.full(n, 1 + 0.0007 * sr))])
        out *= vel_amp(note.vel_scalar(), 30) * 0.25
        out[:, :16] *= raised_cos(16)
        return fade(out, 0, 0.02)


class ReverseSwell(Instrument):
    """Reverse-reverb inhale: a chord (or noise) is reverberated, reversed, and placed so its peak
    lands exactly at the written end. Use source='chord' with pitch + chord, or source='noise'."""
    name = "reverse_swell"
    PARAMS = {
        "source": ("chord", "chord | noise"),
        "chord": ((0, 7, 12, 16), "semitone offsets from the pitch"),
        "space": ("cathedral", "reverb preset used for the swell"),
        "rt60": (4.0, "reverb time (s)"),
        "brightness": (0.6, "0 dark .. 1 bright"),
        "cut": (True, "True: end abruptly at the peak (classic); False: let the dry hit ring 0.3 s"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        d = note.dur
        k = int(0.25 * sr)
        tt = np.arange(k) / sr
        if P("source") == "chord" and np.isfinite(note.midi):
            src = np.zeros(k)
            for iv in P("chord"):
                f = float(midi_to_hz(note.midi + iv))
                src += np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.08)
        else:
            src = rng.standard_normal(k) * np.exp(-tt / 0.03)
        src = svf(src, 1500 + 9000 * P("brightness"), 0.7, "lowpass")
        ir, cross = make_ir(P("space"), rt60=P("rt60"), seed=int(note.seed % 97))
        wet = convolve(np.vstack([src, src]), ir, cross)
        n = int(d * sr)
        wet = wet[:, :n]
        if wet.shape[1] < n:
            wet = np.pad(wet, ((0, 0), (n - wet.shape[1], 0)))
        rev = signal.sosfilt(signal.butter(2, 25, "highpass", fs=sr, output="sos"), wet[:, ::-1], axis=-1)
        rev = rev / (np.max(np.abs(rev)) + 1e-12)
        # make it start from silence even if the reverb is longer than the note
        rev[:, : int(0.2 * n)] *= raised_cos(int(0.2 * n))
        if not P("cut"):
            hit = np.vstack([src, src])[:, : int(0.3 * sr)]
            rev = np.concatenate([rev, hit / (np.max(np.abs(hit)) + 1e-9) * 0.5], axis=1)
        else:
            rev[:, -384:] *= raised_cos(384)[::-1]  # 8 ms: lands on the downbeat without a click
        return rev * vel_amp(note.vel_scalar(), 30) * 0.42
