"""String sections: violins I/II, violas, cellos, basses; every player simulated.

Player = band-limited Helmholtz-motion source: the difference of two polyBLEP sawtooths offset by
the bow position beta gives the bowed-string spectrum sin(pi k beta)/k without aliasing; beta
wanders slowly (bow placement). Dynamics drive both level and a time-varying low-pass (darker pp,
brighter ff). Bow noise (friction) rides on the tone, stronger at the attack, at bow changes and in
soft playing. Individual tuning, vibrato (delayed, random depth/rate), drift, jitter, timing, level.
Players are summed in four desks-groups per section, each convolved with its own synthesised body
response (section-scaled violin/cello body modes + dense random modes) and placed in the classic
seating (vn1 left ... basses right).

Articulations: 'sustain' (legato phrases, long crescendi), 'tremolo' (unmeasured or measured),
'spiccato' (short bounced strokes, for 16th-note pulses).
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy import signal

from ..core import SR, fade, midi_to_hz, raised_cos, seed_of, smooth_noise, to_midi
from ..dsp.filters import fir_from_modes, svf
from .base import Instrument, Note, allocate_voices, vel_curve


@njit(cache=True, fastmath=True, inline="always")
def _blep(t, dt):
    if t < dt:
        x = t / dt
        return x + x - x * x - 1.0
    if t > 1.0 - dt:
        x = (t - 1.0) / dt
        return x * x + x + x + 1.0
    return 0.0


@njit(cache=True, fastmath=True)
def _bow_osc(f0, beta, ph0, sr):
    n = f0.shape[0]
    y = np.zeros(n)
    ph = ph0
    for i in range(n):
        dt = f0[i] / sr
        if dt > 0.45:
            dt = 0.45
        b = beta[i]
        s1 = 2.0 * ph - 1.0 - _blep(ph, dt)
        p2 = ph - b
        if p2 < 0.0:
            p2 += 1.0
        s2 = 2.0 * p2 - 1.0 - _blep(p2, dt)
        y[i] = s1 - s2
        ph += dt
        if ph >= 1.0:
            ph -= 1.0
    return y


SECTIONS = {
    #       players, pan, body scale, vib rate, vib cents, range (midi lo, hi), level
    "vn1": (14, -0.55, 1.00, 5.8, 18.0, (55, 100), 1.0),
    "vn2": (12, -0.2, 1.00, 5.6, 16.0, (55, 93), 0.95),
    "va": (10, 0.2, 0.80, 5.4, 15.0, (48, 86), 0.88),
    "vc": (8, 0.45, 0.45, 5.2, 14.0, (36, 79), 0.9),
    "cb": (8, 0.65, 0.30, 4.9, 10.0, (28, 60), 0.4),
}

_VIOLIN_MODES = [(275, 0.8), (405, 0.45), (465, 1.0), (540, 0.95), (700, 0.55), (850, 0.5), (1050, 0.5),
                 (1300, 0.45), (1600, 0.45), (2100, 0.6), (2500, 0.75), (2900, 0.6), (3400, 0.4),
                 (4200, 0.28), (5300, 0.16), (6500, 0.08)]
_body_cache = {}


def _body(section, group):
    key = (section, group)
    if key not in _body_cache:
        sc = SECTIONS[section][2]
        rng = np.random.default_rng(seed_of("body", section, group))
        fr = [f * sc * (1 + 0.03 * rng.standard_normal()) for f, _ in _VIOLIN_MODES]
        gn = [g * np.exp(0.2 * rng.standard_normal()) for _, g in _VIOLIN_MODES]
        nr = 50
        rf = np.exp(rng.uniform(np.log(500 * sc), np.log(8000 * max(sc, 0.5)), nr))
        fr += list(rf)
        gn += list(0.3 * rng.random(nr) * np.clip(6000 / rf, 0.2, 1))
        # roughly constant Q (~20-45): a smooth body response without wolf-like level jumps between notes
        t6 = [np.clip(6.9 * rng.uniform(12, 30) / (np.pi * f), 0.006, 0.2) for f in fr]
        # amplitude / T60 so each mode's *peak* response follows its gain (not its decay length)
        gn = [g / t * 0.02 for g, t in zip(gn, t6)]
        # one body per desk group; L/R differ only slightly (radiation direction), so a player stays
        # a point source rather than a diffuse cloud
        hs = []
        for c in range(2):
            h = fir_from_modes(fr, t6, gn, 0.35, np.random.default_rng(seed_of("bodyph", section, group, c)))
            hs.append(h / (np.sqrt(np.sum(h ** 2)) + 1e-12))
        irs = []
        for c in range(2):
            h = hs[0] + (0.3 if c else -0.3) * hs[1]
            h = 0.75 * h / (np.sqrt(np.sum(h ** 2)) + 1e-12)
            h[0] += 1.0  # direct (bridge) sound keeps the colour but limits note-to-note level jumps
            h /= np.sqrt(np.sum(h ** 2)) + 1e-12
            irs.append(h)
        _body_cache[key] = np.array(irs)
    return _body_cache[key]


class Strings(Instrument):
    """Bowed string section; each player simulated (8-14 per section)."""
    name = "strings"
    legato_phrases = True
    PARAMS = {
        "section": ("vn1", "vn1 | vn2 | va | vc | cb"),
        "players": (None, "players in the section (default 14/12/10/8/8)"),
        "articulation": ("sustain", "sustain | tremolo | spiccato"),
        "attack": (0.18, "sustain: onset time (s); long pp entries like 0.6-1.5"),
        "release": (0.4, "sustain: release time (s)"),
        "vib_cents": (None, "vibrato depth (cents), default by section"),
        "vib_rate": (None, "vibrato rate (Hz), default by section"),
        "detune": (4.0, "per-player tuning std (cents)"),
        "timing": (0.025, "per-player timing std (s)"),
        "brightness": (0.5, "overall brightness 0..1 (added to the dynamic brightness)"),
        "bow_noise": (0.5, "friction noise amount 0..1"),
        "portamento": (0.07, "legato shift time (s); a third of the players slide audibly"),
        "trem_rate": (None, "tremolo strokes/s; None = unmeasured (each player ~11-15/s)"),
        "sordino": (False, "muted strings: darker and softer"),
        "tasto": (0.0, "0 normal .. 1 over the fingerboard, light and fast (flautando; with a quiet vibrato it "
                       "also stands for bowed harmonics): fewer upper partials, more air, less vibrato"),
        "spread": (0.25, "stereo spread around the section position"),
        "center": (None, "stereo position (default by seating)"),
        "humanize": (1.0, "scales drift, jitter, attack wobble, shimmer, bow wander (0 = machine-steady, tests)"),
    }

    def render_notes(self, notes, n_total: int, sr: int = SR):
        buf = np.zeros((2, n_total))
        art_sustain = [n for n in notes if self.P(n, "articulation") == "sustain"]
        others = [n for n in notes if self.P(n, "articulation") != "sustain"]
        groups = allocate_voices(art_sustain) + [[n] for n in others]
        for phrase in groups:
            y, start = self._phrase(phrase, sr)
            s0, s1 = max(0, start), min(n_total, start + y.shape[1])
            if s1 > s0:
                buf[:, s0:s1] += y[:, s0 - start:s1 - start]
        return buf

    def render(self, note: Note, rng):
        return self._phrase([note], SR)[0]

    def _phrase(self, phrase, sr):
        first = phrase[0]
        P = lambda k: self.P(first, k)  # noqa: E731
        sec = P("section")
        nplay, pan0, _, vrate0, vdepth0, _, lvl = SECTIONS[sec]
        nplay = int(P("players") or nplay)
        center = pan0 if P("center") is None else float(P("center"))
        art = P("articulation")
        vdepth = P("vib_cents") if P("vib_cents") is not None else vdepth0
        vrate = P("vib_rate") if P("vib_rate") is not None else vrate0
        att = float(P("attack"))
        rel = float(P("release"))
        if art == "spiccato":
            att, rel = 0.006, 0.06
        pre = 0.12
        t_start = first.t0 - pre
        last = phrase[-1]
        tail = rel + 0.35 if art != "spiccato" else 0.45
        t_end = last.t0 + last.dur + tail
        n = int((t_end - t_start) * sr)
        tt = np.arange(n) / sr

        tgt = np.full(n, first.midi)
        velc = np.zeros(n)
        bounds = []
        for j, nt in enumerate(phrase):
            a = int((nt.t0 - t_start) * sr)
            b = int((nt.t0 + nt.dur - t_start) * sr) if j == len(phrase) - 1 else int((phrase[j + 1].t0 - t_start) * sr)
            b = min(max(b, a + 1), n)
            tgt[a:] = nt.midi
            velc[a:b] = vel_curve(nt.vel, b - a)
            if j == len(phrase) - 1:
                velc[b:] = velc[b - 1]
            bounds.append(a)
        a0 = int(pre * sr)
        velc[:a0] = velc[a0]
        velc = np.clip(velc, 0, 1)

        bright = float(P("brightness"))
        sord = bool(P("sordino"))
        tasto = float(np.clip(P("tasto"), 0.0, 1.0))
        f_mean = midi_to_hz(tgt)
        fc_base = (f_mean * 2.5 + 900 + 9000 * velc ** 1.6) * (0.6 + 0.8 * bright) * (0.55 if sord else 1.0)
        fc_base = fc_base * (1 - 0.5 * tasto) + f_mean * 1.5 * tasto
        amp_dyn = 10 ** (-34 * (1 - velc) / 20) * (0.55 if sord else 1.0)

        ngroups = 4
        out = np.zeros((2, n))
        on = pre
        off = last.t0 + last.dur - t_start
        for g in range(ngroups):
            src = np.zeros((2, n))
            for s in range(g, nplay, ngroups):
                prs = np.random.default_rng(seed_of(sec, "player", s))  # the person
                srng = np.random.default_rng(seed_of(first.seed, "pl", s))  # this performance
                detune = float(np.clip(prs.normal(0, P("detune")), -10, 10))
                vd = vdepth * prs.uniform(0.5, 1.4) * (1 - 0.5 * tasto)
                vr = float(vrate * prs.uniform(0.9, 1.12))
                level = 10 ** (prs.normal(0, 1.5) / 20)
                dt_on = float(np.clip(srng.normal(0.008, P("timing") * (0.35 if art == "spiccato" else 1.0)),
                                      -0.05, 0.08))
                shift = int(dt_on * sr)
                tg = np.roll(tgt, shift)
                if shift > 0:
                    tg[:shift] = tgt[0]
                slides = srng.random() < 0.33
                port = (P("portamento") * srng.uniform(0.7, 1.5)) if slides else 0.018
                k = max(1, int(port * sr))
                ker = np.hanning(k + 2)[1:-1]
                tg = np.convolve(np.concatenate([np.full(k, tg[0]), tg, np.full(k, tg[-1])]), ker / ker.sum(), "same")[k:k + n]
                vdel = pre + dt_on + srng.uniform(0.12, 0.35)
                venv = np.clip((tt - vdel) / 0.4, 0, 1)
                rate = vr * (1 + 0.05 * smooth_noise(n, 0.5, srng, sr))
                vib = vd * venv * (1 + 0.25 * smooth_noise(n, 0.4, srng, sr)) * np.sin(
                    2 * np.pi * np.cumsum(rate) / sr + srng.uniform(0, 6.28))
                if art == "spiccato":
                    vib *= 0.3
                hum = float(P("humanize"))
                m = tg + (detune + vib + hum * (3.0 * smooth_noise(n, 0.3, srng, sr) + 1.8 * smooth_noise(n, 30, srng, sr))) / 100
                # attack pitch instability (bow grabbing the string)
                m += hum * (srng.normal(0, 8) / 100) * np.exp(-np.maximum(tt - on - dt_on, 0) / 0.04) * (tt >= on + dt_on)
                f0 = midi_to_hz(m)
                beta = np.clip(0.11 + 0.16 * tasto + hum * (0.012 * prs.standard_normal()
                                                            + 0.008 * smooth_noise(n, 0.2, srng, sr)), 0.05, 0.34)
                # oscillator runs at 2x and is decimated: aliasing < -60 dB below 10 kHz
                osc = _bow_osc(np.ascontiguousarray(np.repeat(f0, 2)), np.ascontiguousarray(np.repeat(beta, 2)),
                               srng.random(), float(2 * sr))
                osc = signal.resample_poly(osc, 1, 2)[:n]

                # envelope by articulation
                o = on + dt_on
                if art == "sustain":
                    a_s = att * srng.uniform(0.75, 1.35)
                    env = np.clip((tt - o) / a_s, 0, 1)
                    env = env * env * (3 - 2 * env)
                    r_s = rel * srng.uniform(0.7, 1.3)
                    renv = np.clip(1 - (tt - off - dt_on * 0.5) / r_s, 0, 1)
                    env *= renv * renv * (3 - 2 * renv)
                    # bow changes on long notes
                    for j, nt in enumerate(phrase):
                        if nt.dur > 3.5:
                            nb = int(nt.dur // srng.uniform(2.8, 4.5))
                            for q in range(nb):
                                c = int((nt.t0 - t_start + (q + 1) * nt.dur / (nb + 1) + srng.normal(0, 0.2)) * sr)
                                w = int(0.06 * sr)
                                lo, hi = max(0, c - w), min(n, c + w)
                                if hi > lo:
                                    env[lo:hi] *= 1 - 0.3 * np.hanning(hi - lo)
                elif art == "spiccato":
                    env = np.zeros(n)
                    for nt in phrase:
                        st = nt.t0 - t_start + dt_on
                        d = min(max(nt.dur * 0.6, 0.07), 0.16) * srng.uniform(0.85, 1.15)
                        x = tt - st
                        e = np.where(x >= 0, np.minimum(x / 0.005, 1) * np.exp(-np.maximum(x, 0) / (d * 0.45)), 0)
                        env = np.maximum(env, e)
                else:  # tremolo
                    rate_t = P("trem_rate") or srng.uniform(11, 15)
                    if P("trem_rate"):
                        rate_t *= 1 + 0.01 * srng.standard_normal()
                    a_s = max(att, 0.05)
                    env0 = np.clip((tt - o) / a_s, 0, 1) * np.clip(1 - (tt - off) / max(rel, 0.08), 0, 1)
                    phs = np.cumsum(rate_t * (1 + 0.08 * smooth_noise(n, 3, srng, sr))) / sr + srng.random()
                    fr = phs % 1.0
                    stroke = 0.62 + 0.38 * np.minimum(fr / 0.12, 1) * (1 - 0.35 * fr)
                    env = env0 * stroke
                shimmer = 1 + 0.04 * hum * smooth_noise(n, 2.5, srng, sr)
                amp = env * shimmer * level * np.roll(amp_dyn, shift)
                # dynamic brightness per sample
                fc = np.clip(np.roll(fc_base, shift) * (1 + 0.1 * hum * smooth_noise(n, 0.7, srng, sr)), 300, 0.45 * sr)
                tone = svf(osc * amp, fc, 0.6, "lowpass")
                # friction noise: band-limited, gated with the tone, louder at onsets and in pp
                bn = P("bow_noise")
                if bn > 0:
                    nz = srng.standard_normal(n)
                    nz = svf(nz, np.clip(fc * 1.3, 800, 0.45 * sr), 0.5, "lowpass")
                    nz = nz - svf(nz, 700, 0.7, "lowpass")
                    burst = np.exp(-np.maximum(tt - o, 0) / 0.05) * (tt >= o)
                    if art == "spiccato":
                        burst = env * 1.5
                    elif art == "tremolo":
                        burst = np.exp(-np.minimum(fr, 1) / 0.03) * env0
                    pp = (1.6 - velc) * (1 + 1.5 * tasto)
                    tone = tone + bn * 0.06 * nz * (np.abs(amp) * pp * 0.6 + burst * amp.max() * 0.8)
                th = (np.clip(center + P("spread") * prs.uniform(-1, 1), -1, 1) + 1) * np.pi / 4
                src[0] += tone * np.cos(th)
                src[1] += tone * np.sin(th)
            h = _body(sec, g)
            out[0] += signal.oaconvolve(src[0], h[0])[:n]
            out[1] += signal.oaconvolve(src[1], h[1])[:n]
        out = signal.sosfilt(signal.butter(2, 28 if sec == "cb" else 45, "highpass", fs=sr, output="sos"), out, axis=-1)
        out *= lvl * 0.17 / np.sqrt(nplay)
        out[:, :64] *= raised_cos(64)
        return fade(out, 0, 0.08), int(round(t_start * sr))


class TautString(Instrument):
    """One long bowed string of light being drawn taut (a sound-design voice): starts flat and slack,
    tightens to the written pitch, its hum beating faster and getting brighter as the tension rises.

    Two band-limited Helmholtz oscillators (the same bowed-string source as Strings), a few cents apart so they
    beat; the pitch rises from `slack_cents` below the written pitch to it over `tighten` of the note (slow at
    first, a smoothstep), the beat rate from `beat_from` to `beat_to` Hz (at most 14 cents) and the low-pass from
    `bright_from` to `bright_to` with the same curve; a little bow friction, a little flutter at high tension.
    No body: these are not wooden instruments. vel may be a ramp or breakpoints (crescendo over the note)."""
    name = "taut_string"
    PARAMS = {
        "slack_cents": (45.0, "how far below the written pitch the string starts (cents)"),
        "tighten": (0.92, "fraction of the note at which it reaches the written pitch"),
        "beat_from": (0.25, "beat rate between the two copies at the start (Hz)"),
        "beat_to": (3.5, "beat rate once taut (Hz)"),
        "bright_from": (0.15, "brightness at the start 0..1"),
        "bright_to": (0.7, "brightness once taut 0..1"),
        "beta": (0.13, "bow position as a fraction of the string: 0.1 bright and peaky .. 0.3 hollow and round"),
        "copy2": (1.0, "level of the second, beating copy (lower = shallower beats, rounder peaks)"),
        "bow": (0.35, "bow friction noise 0..1"),
        "flutter": (0.25, "amplitude flutter at full tension 0..1"),
        "attack": (1.2, "onset time (s): the string materialises out of the dark"),
        "release": (0.25, "release time (s)"),
        "center": (0.0, "stereo position"),
        "width": (0.25, "stereo distance between the two copies"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        rel = float(P("release"))
        n_note = max(1, int(note.dur * sr))
        n = n_note + int((rel + 0.05) * sr)
        tt = np.arange(n) / sr
        x = np.clip(tt / max(1e-3, P("tighten") * note.dur), 0.0, 1.0)
        e = x * x * (3 - 2 * x)
        cents = -float(P("slack_cents")) * (1 - e) + 1.5 * smooth_noise(n, 0.3, rng, sr)
        f = note.hz * 2 ** (cents / 1200)
        # the two copies sit symmetrically around the pitch; the beat is capped at 14 cents, so low strings beat
        # slowly and never sound out of tune
        beat = float(P("beat_from")) + (float(P("beat_to")) - float(P("beat_from"))) * e * e
        beat = np.minimum(beat, f * (2 ** (14 / 1200) - 1))
        f1, f2 = f - 0.5 * beat, f + 0.5 * beat
        bright = float(P("bright_from")) + (float(P("bright_to")) - float(P("bright_from"))) * e
        b0 = float(P("beta"))
        beta = np.clip(b0 + 0.01 * smooth_noise(n, 0.2, rng, sr), 0.6 * b0, 1.5 * b0)
        out = np.zeros((2, n))
        c, w = float(P("center")), float(P("width"))
        for k, f in enumerate((f1, f2)):
            osc = _bow_osc(np.ascontiguousarray(np.repeat(f, 2)), np.ascontiguousarray(np.repeat(beta, 2)),
                           rng.random(), float(2 * sr))
            osc = signal.resample_poly(osc, 1, 2)[:n]
            fc = np.clip(f * (1.5 + 14.0 * bright) + 250.0, 200.0, 0.45 * sr)
            y = svf(osc, fc, 0.6, "lowpass")
            bn = float(P("bow"))
            if bn > 0:
                nz = svf(rng.standard_normal(n), np.clip(fc * 1.3, 800, 0.45 * sr), 0.5, "lowpass")
                nz = nz - svf(nz, 700, 0.7, "lowpass")
                y = y + bn * 0.05 * nz
            if k:
                y = y * float(P("copy2"))
            th = (np.clip(c + (w / 2 if k else -w / 2), -1, 1) + 1) * np.pi / 4
            out[0] += y * np.cos(th)
            out[1] += y * np.sin(th)
        # envelope: attack (smoothstep), the velocity curve over the written note, release
        a = np.clip(tt / max(1e-3, float(P("attack"))), 0, 1)
        env = a * a * (3 - 2 * a)
        r = np.clip(1 - (tt - note.dur) / max(1e-3, rel), 0, 1)
        env *= r * r * (3 - 2 * r)
        vc = vel_curve(note.vel, n_note)
        vc = np.concatenate([vc, np.full(n - n_note, vc[-1])])
        env *= 10 ** (-30 * (1 - np.clip(vc, 0, 1)) / 20)
        env *= 1 + float(P("flutter")) * 0.12 * e * smooth_noise(n, 9.0, rng, sr)
        out *= env * 0.23                    # vel 0.7: about -24 LUFS once sounding, like the rest of the library
        out[:, :64] *= raised_cos(64)
        return fade(out, 0, 0.02)
