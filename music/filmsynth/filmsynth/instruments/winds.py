"""Wind instruments.

Xiao (箫): the end-blown bamboo flute, range about D4-E6. A physical model of the flue (jet drive) and the
bamboo bore, played by a breath envelope:

  * jet: the jet's transverse displacement at the edge is eta = beta * sin(phase) + turbulence; the flow
    that enters the pipe is Q = U * (1 + tanh(eta - y0)), and the pressure that drives the pipe is dQ/dt
    (the jet-drive source of Verge / Fabre). U (jet velocity) follows the square root of the breath
    pressure; beta (how far the jet swings across the edge) grows with the breath, so a harder breath
    gives more harmonics; y0 (where the lips aim the jet) sets the even harmonics;
  * bore: an open-open digital waveguide tuned to the sounding pitch (fractional delay, a loss filter that
    takes more out of the high partials), so the turbulence is coloured by the pipe's resonances and the
    tone rings up out of the air and dies back into it;
  * radiation: the open end radiates the highs a little more than the lows; air that misses the edge is
    radiated straight from the embouchure (the breath you hear around the tone);
  * the player: one breath per phrase (a soft onset in which the air comes before the tone), a soft
    tongued 'du' or a slur inside the phrase, finger changes of a few tens of milliseconds, a slow breath
    vibrato (about 5 Hz) that starts late in a long note, slow drift of breath and pitch, and slides
    (half-holing) that make the tone breathier while the finger moves.

The jet is driven at the written pitch rather than left to oscillate by itself: a free jet-drive loop jumps
between registers and edge tones across this range, and the score needs the pitch, the slides and the
vibrato exactly where they are written.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy import signal

from ..core import SR, fade, midi_to_hz, smooth_noise, vel_amp
from .base import Instrument, Note, allocate_voices, vel_curve


@njit(cache=True, fastmath=True, inline="always")
def _hread(buf, rp, mask):
    ip = int(np.floor(rp))
    fr = rp - ip
    xm1 = buf[(ip - 1) & mask]
    x0 = buf[ip & mask]
    x1 = buf[(ip + 1) & mask]
    x2 = buf[(ip + 2) & mask]
    c1 = 0.5 * (x1 - xm1)
    c2 = xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2
    c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1)
    return ((c3 * fr + c2) * fr + c1) * fr + x0


@njit(cache=True, fastmath=True)
def _bore(src, N, g, pole):
    """Open-open pipe driven at the mouth: y[n] = src[n] + g * LP(y)[n - N]; returns the wave in the pipe."""
    n = src.shape[0]
    size = 1
    mx = 8.0
    for i in range(n):
        if N[i] > mx:
            mx = N[i]
    while size < mx + 8:
        size *= 2
    mask = size - 1
    buf = np.zeros(size)
    out = np.zeros(n)
    lp = 0.0
    for i in range(n):
        r = _hread(buf, i - N[i], mask)
        lp = (1.0 - pole[i]) * r + pole[i] * lp
        y = src[i] + g[i] * lp
        buf[i & mask] = y
        out[i] = y
    return out


def _onepole_phase_delay(pole, w):
    return np.arctan2(pole * np.sin(w), 1 - pole * np.cos(w)) / w


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class Xiao(Instrument):
    """End-blown bamboo flute (箫), jet drive + waveguide bore; legato phrases share one breath."""
    name = "xiao"
    legato_phrases = True
    # voicing (measured with the analysis in the music README, section 5): radiation tilt, jet swing,
    # turbulence, breath noise relative to the tone, output level (~-24 LUFS at vel 0.7)
    RAD_LO, RAD_HI = 500.0, 1500.0
    BETA0, BETA1, BETA_REG = 0.18, 0.55, 0.35
    TURB = 0.10
    PIPE_NOISE_DB, AIR_DB = -21.0, -25.0
    LEVEL = 0.04
    PARAMS = {
        "breath": (0.45, "air heard around the tone 0..1 (missed jet + turbulence)"),
        "attack": (0.09, "breath onset of a phrase (s): 0.05 tongued .. 0.3 a slow 'hu'"),
        "release": (0.16, "breath release at the end of a phrase (s)"),
        "tongue": (0.35, "soft tonguing ('du') of a note inside a phrase 0..1; 0 = slur"),
        "portamento": (0.035, "finger change between slurred notes (s)"),
        "vib_rate": (5.0, "breath vibrato rate (Hz), 4.5-5.5"),
        "vib_depth": (0.12, "breath vibrato depth (fraction of the breath pressure)"),
        "vib_cents": (6.0, "pitch swing that rides on the breath vibrato (cents)"),
        "vib_delay": (0.5, "seconds into a note before the vibrato starts"),
        "vib_attack": (0.6, "seconds for the vibrato to reach full depth"),
        "glide_from": (None, "pitch to slide up or down from at the note's start (half-holing)"),
        "glide_time": (0.18, "slide time (s)"),
        "glide_to": (None, "pitch to slide to later in the note"),
        "glide_start": (0.5, "seconds after the note's start when glide_to begins"),
        "lip": (0.35, "jet offset at the edge: 0 hollow (odd partials) .. 0.8 fuller (even partials)"),
        "edge": (1.0, "how far the jet swings across the edge per breath (harmonic richness)"),
        "bore_t60": (0.09, "ring time of the bore at the fundamental (s)"),
        "bore_dark": (0.35, "loss of the high partials in the bore 0..0.9"),
        "center": (0.0, "stereo position"),
        "width": (0.18, "stereo width of the player's sound"),
        "humanize": (1.0, "scales breath drift, pitch drift and timing of the consonants (0 = steady, tests)"),
    }

    def render_notes(self, notes, n_total: int, sr: int = SR):
        buf = np.zeros((2, n_total))
        for phrase in allocate_voices(notes):
            y, start = self._phrase(phrase, sr)
            s0, s1 = max(0, start), min(n_total, start + y.shape[1])
            if s1 > s0:
                buf[:, s0:s1] += y[:, s0 - start:s1 - start]
        return buf

    def render(self, note: Note, rng):
        return self._phrase([note], SR)[0]

    # ------------------------------------------------------------------------------------------------
    def _phrase(self, phrase, sr):
        first, last = phrase[0], phrase[-1]
        P = lambda k: self.P(first, k)  # noqa: E731
        rng = np.random.default_rng(first.seed)
        hum = float(P("humanize"))
        att = max(0.01, float(P("attack")))
        rel = max(0.02, float(self.P(last, "release")))
        pre = 0.25 * att + 0.02                       # the air starts a little before the written onset
        t_start = first.t0 - pre
        t_end = last.t0 + last.dur + rel + 0.35
        n = int((t_end - t_start) * sr)
        tt = np.arange(n) / sr

        # ---- per-note targets: pitch, breath (velocity), consonants -------------------------------------
        midi = np.full(n, first.midi)
        breath = np.zeros(n)
        cons = np.zeros(n)                            # consonant pulses ('du') inside the phrase
        slide = np.zeros(n)                           # 1 while a finger slides (breathier)
        vib_env = np.zeros(n)
        pc = np.zeros(n)                              # extra pitch (cents): glides
        on_idx = []
        for j, nt in enumerate(phrase):
            a = int((nt.t0 - t_start) * sr)
            b = int((nt.t0 + nt.dur - t_start) * sr) if j == len(phrase) - 1 else int((phrase[j + 1].t0 - t_start) * sr)
            a = int(np.clip(a, 0, n - 1))
            b = int(np.clip(max(b, a + 1), 1, n))
            on_idx.append(a)
            Pn = lambda k: self.P(nt, k)  # noqa: E731,B023
            if j == 0:
                midi[:] = nt.midi
            else:
                pt = max(0.005, float(Pn("portamento")))
                k = int(pt * sr)
                seg = np.arange(n - a)
                ramp = _smoothstep(seg / max(k, 1))
                midi[a:] = midi[a - 1] + (nt.midi - midi[a - 1]) * ramp
            breath[a:b] = vel_curve(nt.vel, b - a)
            if j == len(phrase) - 1:
                breath[b:] = breath[b - 1]
            # glides (half-holing)
            gf, gtgt = Pn("glide_from"), Pn("glide_to")
            gt = max(0.02, float(Pn("glide_time")))
            if gf is not None:
                from ..core import to_midi
                d0 = (to_midi(gf) - nt.midi) * 100.0
                kk = np.arange(b - a)
                g = 1 - _smoothstep(kk / (gt * sr))
                pc[a:b] += d0 * g
                slide[a:b] = np.maximum(slide[a:b], np.clip(g * (1 - g) * 4, 0, 1))
            if gtgt is not None:
                from ..core import to_midi
                d1 = (to_midi(gtgt) - nt.midi) * 100.0
                s0 = a + int(float(Pn("glide_start")) * sr)
                if s0 < n:
                    kk = np.arange(n - s0)
                    g = _smoothstep(kk / (gt * sr))
                    pc[s0:] += d1 * g
                    slide[s0:min(n, s0 + int(gt * sr))] = np.maximum(
                        slide[s0:min(n, s0 + int(gt * sr))], np.clip(g * (1 - g) * 4, 0, 1)[:min(n, s0 + int(gt * sr)) - s0])
            # vibrato envelope of this note: late, slow
            vd, va = float(Pn("vib_delay")), max(0.05, float(Pn("vib_attack")))
            kk = np.arange(b - a) / sr
            ve = _smoothstep((kk - vd) / va)
            vib_env[a:b] = np.maximum(vib_env[a:b], ve)
            if j == len(phrase) - 1 and b < n:
                vib_env[b:] = vib_env[b - 1] if b > a else 0.0
            # tonguing inside the phrase (the first note's onset is the breath attack itself)
            if j > 0:
                tg = float(Pn("tongue"))
                if tg > 0:
                    k = int((0.045 + 0.01 * hum * rng.standard_normal()) * sr)
                    k = max(k, int(0.02 * sr))
                    c0 = max(0, a - int(0.55 * k))           # the tongue closes just before the note speaks
                    seg = np.arange(min(k, n - c0)) / max(k, 1)
                    cons[c0:c0 + len(seg)] = np.maximum(cons[c0:c0 + len(seg)], tg * np.sin(np.pi * seg) ** 2)

        # ---- breath pressure: phrase envelope x note velocities x tonguing x vibrato x drift --------------
        a0 = int(pre * sr)
        on = _smoothstep((tt - (pre - 0.25 * att)) / att) ** 0.8
        off_at = (last.t0 + last.dur - t_start)
        off = 1 - _smoothstep((tt - off_at) / rel)
        env = on * off
        breath = np.clip(breath, 0.0, 1.0)
        breath[:a0] = breath[a0] if a0 < n else breath[:a0]
        bsm = signal.lfilter([1 - np.exp(-1 / (0.03 * sr))], [1, -np.exp(-1 / (0.03 * sr))], breath)
        rate = float(P("vib_rate")) * (1 + 0.04 * smooth_noise(n, 0.6, rng, sr))
        vph = 2 * np.pi * np.cumsum(rate) / sr + rng.uniform(0, 2 * np.pi)
        vib = np.sin(vph) * vib_env
        drift = 1 + 0.025 * hum * smooth_noise(n, 0.7, rng, sr) + 0.012 * hum * smooth_noise(n, 3.0, rng, sr)
        tongue_dip = 1 - np.minimum(0.92, 2.2 * cons)          # the tongue all but stops the air for ~40 ms
        Pr = env * (0.25 + 0.75 * bsm) * drift * tongue_dip * (1 + float(P("vib_depth")) * vib)
        Pr = np.maximum(Pr, 0.0)
        U = np.sqrt(Pr)                                # jet velocity

        # ---- pitch: written + glides + vibrato swing + breath pull + drift -------------------------------
        cents = pc + float(P("vib_cents")) * vib + 8.0 * (bsm - 0.6) * env \
            + hum * (1.5 * smooth_noise(n, 0.4, rng, sr) + 0.6 * smooth_noise(n, 8.0, rng, sr))
        m = midi + cents / 100.0
        f = midi_to_hz(m)

        # ---- the jet: coherent swing builds up out of the air after each onset -----------------------------
        build = np.ones(n)
        for a in on_idx[:1]:
            k = np.arange(n - a) / sr
            build[a:] = 1 - np.exp(-np.maximum(k + 0.25 * att, 0) / 0.035)
            build[:a] = 0.0
        build *= 1 - 0.6 * cons
        build *= 1 - 0.45 * slide
        # the player's jet is slower for low notes (fewer harmonics, less turbulence) and is overblown, faster
        # and swinging harder, in the upper register
        fr_rel = float(np.median(f)) / 440.0
        reg = np.clip((np.median(midi) - 62) / 26.0, 0, 1)
        beta = float(P("edge")) * (self.BETA0 + self.BETA1 * np.sqrt(Pr) + self.BETA_REG * reg) * build \
            * min(1.0, fr_rel) ** 0.5
        phase = 2 * np.pi * np.cumsum(f) / sr + rng.uniform(0, 2 * np.pi)
        # turbulence the jet can follow (around the playing frequency); the faster eddies are in the air below
        turb = signal.lfilter(*signal.butter(2, min(2.5 * float(np.median(f)), 3000.0), "lowpass", fs=sr),
                              rng.standard_normal(n))
        turb *= self.TURB * min(1.0, fr_rel) ** 1.5 / (np.std(turb) + 1e-12)
        eta = beta * np.sin(phase) + turb * np.sqrt(Pr) * (0.5 + float(P("breath")))
        y0 = float(P("lip"))
        Q = U * (1 + np.tanh(eta - y0))
        src = np.diff(Q, prepend=Q[0])

        # ---- the bore: tuned waveguide with losses ------------------------------------------------------------
        dark = float(np.clip(P("bore_dark"), 0.0, 0.9))
        pole = np.full(n, dark)
        w = 2 * np.pi * f / sr
        N = np.ascontiguousarray(sr / f - _onepole_phase_delay(dark, w))
        t60 = max(0.02, float(P("bore_t60")))
        g = np.ascontiguousarray(10 ** (-3.0 / (f * t60)) * (1 - 0.05 * slide))   # a half-covered hole leaks
        za, pb = np.exp(-2 * np.pi * self.RAD_LO / sr), np.exp(-2 * np.pi * self.RAD_HI / sr)

        def radiate(x):
            # the open end radiates the highs a little more (zero at RAD_LO, pole at RAD_HI); DC out
            x = signal.lfilter([1, -za], [1, -pb], x)
            return signal.lfilter([1, -1], [1, -0.995], x)

        tone = radiate(_bore(np.ascontiguousarray(src), N, g, pole))
        k0 = slice(a0, min(n, a0 + int(0.6 * sr)))
        r_tone = np.sqrt(np.mean(tone[k0] ** 2)) + 1e-12
        u_ref = np.sqrt(np.mean(U[k0] ** 2)) + 1e-12
        br = float(P("breath")) / 0.45

        # air that hits the edge goes into the pipe too: noise coloured by the bore's resonances
        en = signal.lfilter(*signal.butter(2, [300, 7000], "bandpass", fs=sr), rng.standard_normal(n))
        en = np.ascontiguousarray(en * (U / u_ref) * (1 + 1.5 * slide + 1.2 * (1 - build) * env))
        piped = radiate(_bore(en, N, g, pole))
        piped *= r_tone / (np.sqrt(np.mean(piped[k0] ** 2)) + 1e-12) * 10 ** (self.PIPE_NOISE_DB / 20) * br

        # air radiated straight from the embouchure (the breath around the tone)
        air = rng.standard_normal(n)
        air = signal.lfilter(*signal.butter(2, [700, 9000], "bandpass", fs=sr), air)
        air = signal.lfilter(*signal.butter(1, 3000, "lowpass", fs=sr), air) * 0.7 + air * 0.3
        air /= np.std(air) + 1e-12
        air_env = (U / u_ref) * (1 + 1.4 * (1 - build) * env + 1.5 * slide + 1.5 * cons)
        air *= air_env * r_tone * 10 ** (self.AIR_DB / 20) * br
        # consonant: a soft 'd' (short, low-passed noise) at tongued notes
        dn = signal.lfilter(*signal.butter(2, 1600, "lowpass", fs=sr), rng.standard_normal(n))
        dn = dn / (np.std(dn) + 1e-12) * cons * 0.25 * r_tone * np.sqrt(Pr + 0.05)
        mono = tone + piped + air + dn

        # ---- level: equal loudness across the range, velocity curve like the other instruments --------------
        ref = np.sqrt(np.mean(mono[k0] ** 2)) + 1e-12
        mono = mono / ref * self.LEVEL * (np.mean(f[k0]) / 440.0) ** -0.12
        # the dynamics inside a phrase: the level follows the (smoothed) velocity curve like the other instruments,
        # on top of what the breath pressure does to the tone
        vref = float(np.clip(np.mean(bsm[k0]), 0.05, 1.0))
        dyn = vel_amp(np.clip(bsm, 0.05, 1.0), 24.0) / vel_amp(vref, 24.0)
        mono *= dyn * vel_amp(vref, 24.0) / vel_amp(0.7, 24.0)

        # ---- stereo: the player's position, a little width (early room), fades ------------------------------
        c = float(np.clip(P("center"), -1, 1))
        wd = float(P("width"))
        dl = int(0.0007 * sr)
        side = np.concatenate([np.zeros(dl), mono[:-dl]]) - mono
        L = mono * np.cos((c + 1) * np.pi / 4) * np.sqrt(2) + wd * 0.5 * side
        R = mono * np.sin((c + 1) * np.pi / 4) * np.sqrt(2) - wd * 0.5 * side
        out = np.vstack([L, R])
        out = fade(out, 0.005, 0.03)
        return out, int(round(t_start * sr))
