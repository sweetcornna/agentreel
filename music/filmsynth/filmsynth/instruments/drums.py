"""Drums: taiko (circular membrane, Bessel mode ratios, strike pitch drop, stick noise, barrel
resonance), crowd footsteps (many soft hits with human timing), timpani (tunable, rolls),
a low 'boom' (sine down-sweep + transient) for structural downbeats, and the heartbeat drum
(felt mallet on a large hand-muffled frame drum: a low, dry 'dum' with no ring)."""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..core import SR, fade, midi_to_hz, raised_cos, seed_of, vel_amp
from ..dsp.filters import onepole_lp, svf
from ..dsp.modal import render_modes
from .base import Instrument, Note

# Ideal membrane mode ratios (Bessel zeros j_mn / j_01) with (m, n) labels
_MEMBRANE = [((0, 1), 1.000), ((1, 1), 1.594), ((2, 1), 2.136), ((0, 2), 2.296), ((3, 1), 2.653),
             ((1, 2), 2.918), ((4, 1), 3.156), ((2, 2), 3.501), ((0, 3), 3.600), ((5, 1), 3.652),
             ((3, 2), 4.060), ((6, 1), 4.154), ((1, 3), 4.230), ((4, 2), 4.601), ((7, 1), 4.654),
             ((2, 3), 4.832), ((0, 4), 4.903), ((8, 1), 5.131)]
_J0Z = [2.405, 5.520, 8.654, 11.79]


def _strike_weights(pos: float):
    """Mode excitation by strike position (0 centre .. 0.9 near rim): approximate J_m(k r) profile."""
    from scipy.special import jv, jn_zeros
    w = []
    for (m, nn), _ in _MEMBRANE:
        z = jn_zeros(m, nn)[-1]
        w.append(abs(jv(m, z * pos)) if (m > 0 or pos > 0) else 1.0)
    w = np.array(w)
    return w / (w.max() + 1e-12)


def _contact_filter(freqs, tc):
    """Spectrum of a half-sine contact force of duration tc (s): soft mallets cut the top."""
    x = 2 * freqs * tc
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.abs(np.cos(np.pi * freqs * tc) / (1 - x ** 2))
    h[~np.isfinite(h)] = np.pi / 4
    return h


TAIKO_SIZES = {
    # fundamental Hz, fundamental T60, barrel resonance Hz, contact time s, stick noise
    "odaiko": (52.0, 1.6, 180.0, 0.006, 0.25),
    "nagado": (92.0, 0.9, 260.0, 0.0035, 0.35),
    "shime": (240.0, 0.35, 900.0, 0.0015, 0.6),
}


class Taiko(Instrument):
    """Taiko-style drum: membrane modes, strike pitch drop, stick, barrel."""
    name = "taiko"
    pitched = False
    PARAMS = {
        "size": ("nagado", "odaiko | nagado | shime"),
        "tune": (0.0, "detune in semitones from the size's natural pitch (or give a pitch)"),
        "position": (0.25, "strike position 0 centre .. 0.8 rim (rim = more overtones, less boom)"),
        "hardness": (0.6, "stick hardness 0 soft (felt) .. 1 hard (bachi)"),
        "decay": (1.0, "multiplies T60s"),
        "drop": (1.0, "pitch drop amount after the strike (tension modulation)"),
        "center": (0.0, "stereo position"),
        "width": (0.35, "stereo width of the modes"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        f_nat, t60f, barrel, tc0, stick = TAIKO_SIZES[P("size")]
        vel = float(np.clip(note.vel_scalar() + 0.04 * rng.standard_normal(), 0.02, 1))
        f0 = float(midi_to_hz(note.midi)) if np.isfinite(note.midi) else f_nat * 2 ** (P("tune") / 12)
        f0 *= 1 + 0.006 * rng.standard_normal()
        ratios = np.array([r for _, r in _MEMBRANE]) * (1 + 0.01 * rng.standard_normal(len(_MEMBRANE)))
        # air loading lowers the fundamental relative to the rest, a real-drum trait
        ratios[0] *= 0.94
        freqs = f0 * ratios
        pos = float(np.clip(P("position") + 0.03 * rng.standard_normal(), 0, 0.85))
        w = _strike_weights(pos)
        hard = np.clip(P("hardness") * (0.6 + 0.6 * vel), 0.05, 1.2)
        tc = tc0 * (1.4 - 0.8 * hard)
        amps = w * _contact_filter(freqs, tc) * (1 / ratios ** 0.6)
        t60 = t60f * P("decay") * (ratios ** -1.1) * np.exp(0.1 * rng.standard_normal(len(ratios)))
        drop = 0.04 + 0.16 * vel * P("drop")
        dur = min(max(t60.max() * 1.1, 0.4), 6)
        n = int(dur * sr)
        pans = P("center") + P("width") * rng.uniform(-1, 1, len(freqs)) * np.minimum(ratios - 0.9, 1)
        y = render_modes(n, freqs, amps, t60, rng, pans, attack=tc * 0.25, glide=drop, glide_tau=0.035 + 0.02 / (1 + hard))
        t = np.arange(n) / sr
        # barrel/cavity resonance excited by the head: band-passed copy of the low modes
        cav = svf(y[0] + y[1], barrel, 4.0, "bandpass_norm") * 0.35
        y += np.vstack([cav, cav])
        # stick noise
        k = int(0.04 * sr)
        nz = svf(rng.standard_normal(k), 1800 + 3000 * hard, 1.2, "bandpass_norm")
        nz *= np.exp(-t[:k] / (0.002 + 0.004 * (1 - hard))) * stick * hard
        # skin slap (low thud)
        th = np.sin(2 * np.pi * f0 * 1.8 * t[:k]) * np.exp(-t[:k] / 0.01) * 0.3
        y[:, :k] += np.vstack([nz + th, nz * 0.9 + th]) * np.max(np.abs(y[:, :int(0.05 * sr)]))
        y *= vel_amp(vel, 34.0) * 0.5
        y = signal.sosfilt(signal.butter(2, 18, "highpass", fs=sr, output="sos"), y, axis=-1)  # no DC
        y[:, :8] *= raised_cos(8)
        return fade(y, 0, 0.1)


class Footsteps(Instrument):
    """'Footsteps of a crowd': each note is one step taken by a crowd of walkers.

    Each walker has a personal drum (size, tuning, position) and steps with human timing and
    force; some walkers skip some beats. Soft and earthy by default. Pitch is ignored."""
    name = "footsteps"
    pitched = False
    PARAMS = {
        "walkers": (24, "number of walkers"),
        "density": (0.8, "probability that a walker takes this step"),
        "timing": (0.018, "timing spread std (s)"),
        "size": (0.5, "0 small/light .. 1 big/heavy"),
        "softness": (0.7, "0 hard sticks .. 1 soft feet/mallets"),
        "width": (0.9, "stereo width of the crowd"),
        "earth": (0.5, "amount of earthy noise (soil, cloth) vs drum tone"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        nw = int(P("walkers"))
        vel = float(note.vel_scalar())
        dur = 0.9
        n = int(dur * sr) + int(0.2 * sr)
        out = np.zeros((2, n))
        off0 = int(0.1 * sr)  # allow early steps
        t = np.arange(int(dur * sr)) / sr
        for w in range(nw):
            prs = np.random.default_rng(seed_of("walker", w))
            if rng.random() > P("density"):
                continue
            f0 = 45 * 2 ** (prs.uniform(0, 1.4) - 0.8 * P("size")) * 2 ** (rng.normal(0, 0.03))
            v = np.clip(vel * prs.uniform(0.5, 1.0) * np.exp(rng.normal(0, 0.25)), 0.02, 1)
            dt = float(np.clip(rng.normal(0, P("timing")) + prs.normal(0, 0.006), -0.09, 0.09))
            ratios = np.array([1.0, 1.59, 2.14, 2.30, 2.65]) * (1 + 0.02 * prs.standard_normal(5))
            soft = P("softness")
            amps = np.array([1.0, 0.5, 0.3, 0.25, 0.12]) * _contact_filter(f0 * ratios, 0.004 + 0.01 * soft)
            t60 = np.array([0.35, 0.2, 0.14, 0.12, 0.08]) * (0.7 + 0.6 * P("size"))
            pan = P("width") * prs.uniform(-1, 1)
            m = render_modes(len(t), f0 * ratios, amps, t60, rng, np.full(5, pan), attack=0.002 + 0.004 * soft,
                             glide=0.08 * v, glide_tau=0.03)
            # earth: low-passed noise thud (sole on soil)
            k = int(0.08 * sr)
            nz = onepole_lp(rng.standard_normal(k), 300 + 900 * (1 - soft)) * np.exp(-t[:k] / 0.015)
            nz *= raised_cos(k) ** 0.05
            gl, gr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
            m[0, :k] += P("earth") * 2.0 * nz * gl
            m[1, :k] += P("earth") * 2.0 * nz * gr
            s = off0 + int(dt * sr)
            out[:, s:s + len(t)] += m[:, : n - s] * vel_amp(v, 30)
        out *= 1.6 / np.sqrt(nw)
        out = signal.sosfilt(signal.butter(2, 28, "highpass", fs=sr, output="sos"), out, axis=-1)
        out[:, :16] *= raised_cos(16)
        return fade(out, 0, 0.1)

    def render_notes(self, notes, n_total, sr=SR):
        buf = np.zeros((2, n_total))
        for nt in notes:
            y = self._post(nt, self.render(nt, np.random.default_rng(nt.seed)))
            s = int(round(nt.t0 * sr)) - int(0.1 * sr)
            a, b = max(0, s), min(n_total, s + y.shape[1])
            if b > a:
                buf[:, a:b] += y[:, a - s:b - s]
        return buf


class Timpani(Instrument):
    """Tuned timpani: air-loaded membrane (near-harmonic (1,1)(2,1)(3,1)... modes), felt mallets,
    thud of the (0,1) mode, single strokes or rolls (roll=True for the note's duration)."""
    name = "timpani"
    PARAMS = {
        "roll": (False, "roll for the whole written duration"),
        "roll_rate": (11.0, "strokes per second for rolls"),
        "hardness": (0.4, "mallet 0 soft felt .. 1 hard wood"),
        "decay": (1.0, "multiplies T60s"),
        "center": (0.1, "stereo position"),
        "damp": (False, "choke at the written end"),
    }
    # (1,1) principal = written pitch; ratios of an air-loaded kettle drum
    _R = np.array([0.66, 1.0, 1.5, 1.98, 2.44, 2.94, 3.3, 3.85])
    _A = np.array([0.35, 1.0, 0.55, 0.32, 0.18, 0.1, 0.06, 0.035])
    _T = np.array([0.4, 3.2, 1.8, 1.3, 0.9, 0.6, 0.45, 0.3])

    def _stroke(self, f0, vel, hard, n, rng, decay):
        ratios = self._R * (1 + 0.004 * rng.standard_normal(len(self._R)))
        tc = 0.006 * (1.5 - hard)
        amps = self._A * _contact_filter(f0 * ratios, tc) * np.exp(0.1 * rng.standard_normal(len(ratios)))
        t60 = self._T * decay * (f0 / 110) ** -0.3
        pans = 0.25 * rng.uniform(-1, 1, len(ratios))
        y = render_modes(n, f0 * ratios, amps, t60, rng, pans, attack=tc * 0.3, glide=0.012 * vel, glide_tau=0.05)
        k = int(0.03 * sr_())
        nz = svf(rng.standard_normal(k), 400 + 1500 * hard, 0.8, "bandpass_norm") * np.exp(-np.arange(k) / sr_() / 0.004)
        y[:, :k] += 0.2 * hard * nz * np.max(np.abs(y[:, :k]))
        return y * vel_amp(vel, 36)

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        f0 = float(midi_to_hz(note.midi))
        hard = P("hardness")
        ring = 4.0 * P("decay")
        n = int((note.dur + ring) * sr)
        y = np.zeros((2, n))
        if P("roll"):
            t = 0.0
            hand = 0
            from .base import vel_curve
            vc = vel_curve(note.vel, 200)
            while t < note.dur:
                v = float(vc[min(199, int(t / max(note.dur, 1e-3) * 199))])
                v = np.clip(v * (0.92 if hand else 1.0) * np.exp(rng.normal(0, 0.08)), 0.02, 1)
                s = int(t * sr)
                st = self._stroke(f0, v, hard * 0.8, n - s, rng, P("decay"))
                y[:, s:] += st * 0.55
                t += (1 / P("roll_rate")) * (1 + rng.normal(0, 0.07))
                hand ^= 1
        else:
            y += self._stroke(f0, note.vel_scalar(), hard, n, rng, P("decay"))
        if P("damp"):
            s = int(note.dur * sr)
            y[:, s:] *= np.exp(-np.arange(n - s) / sr / 0.08)
        c = P("center")
        y = np.vstack([y[0] * np.cos((c + 1) * np.pi / 4) * 1.414, y[1] * np.sin((c + 1) * np.pi / 4) * 1.414])
        y *= 0.5
        y = signal.sosfilt(signal.butter(2, 18, "highpass", fs=sr, output="sos"), y, axis=-1)  # no DC
        y[:, :8] *= raised_cos(8)
        return fade(y, 0, 0.2)


def sr_():
    return SR


class Boom(Instrument):
    """Structural low hit: sine down-sweep with a soft transient. Pitch = where the sweep lands
    (e.g. B0/D1 to sit on the key); the start is `sweep` semitones higher."""
    name = "boom"
    PARAMS = {
        "sweep": (24.0, "semitones the sine falls from"),
        "sweep_time": (0.12, "sweep time constant (s)"),
        "decay": (2.2, "amplitude T60 (s)"),
        "click": (0.25, "transient amount"),
        "harmonics": (0.35, "added 2nd/3rd harmonic so small speakers feel the hit"),
        "attack": (0.002, "onset ramp (s); larger for a soft swell into the hit"),
    }

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        f1 = float(midi_to_hz(note.midi)) if np.isfinite(note.midi) else 36.7
        f0 = f1 * 2 ** (P("sweep") / 12)
        n = int((P("decay") * 1.7 + 0.05) * sr)
        t = np.arange(n) / sr
        tau = P("sweep_time")
        phase = 2 * np.pi * (f1 * t + (f0 - f1) * tau * (1 - np.exp(-t / tau)))
        env = np.exp(-6.9 * t / P("decay")) * np.clip(t / max(P("attack"), 1e-4), 0, 1)
        s = np.sin(phase) * env
        h = P("harmonics")
        s += h * (0.5 * np.sin(2 * phase + 0.3) + 0.25 * np.sin(3 * phase + 1.1)) * env ** 1.5
        k = int(0.05 * sr)
        cl = svf(rng.standard_normal(k), 120, 0.7, "lowpass") * np.exp(-t[:k] / 0.012)
        cl += svf(rng.standard_normal(k), 2500, 0.8, "bandpass_norm") * np.exp(-t[:k] / 0.003) * 0.25
        s[:k] += P("click") * cl / (np.max(np.abs(cl)) + 1e-9) * 0.6
        s *= vel_amp(note.vel_scalar(), 30) * 0.42
        # remove the DC a fast-starting sweep leaves (keeps 30 Hz+)
        s = signal.sosfilt(signal.butter(2, 14, "highpass", fs=sr, output="sos"), s)
        y = np.vstack([s, s])
        y[:, :16] *= raised_cos(16)
        return fade(y, 0, 0.1)


class HeartDrum(Instrument):
    """The heartbeat: a felt mallet on a large frame drum muffled by a hand on the skin. A low, dry 'dum'
    (pitch = the membrane's fundamental, tuned to the key) with its body in the 150-400 Hz band, so that
    laptop speakers carry it too, and no ring: every mode is gone within a few tenths of a second.

    `dry` goes from round (the warm heart: the fundamental breathes out over ~0.4 s) to tight (the cool
    heart: ~0.2 s, a little more knock) without moving the pitch. `dub` adds the soft second beat of a
    heart ('lub-dub') `dub_time` seconds after the first."""
    name = "heart_drum"
    PARAMS = {
        "dry": (0.4, "0 round, the fundamental dies in ~0.45 s .. 1 tight, ~0.18 s (same pitch)"),
        "felt": (0.6, "mallet 0 hard felt (more 300-600 Hz knock) .. 1 soft felt (rounder)"),
        "body": (0.6, "level of the frame and air-cavity resonances (150-400 Hz, very short)"),
        "position": (0.25, "strike position 0 centre .. 0.6 (more of the asymmetric modes, less boom)"),
        "dub": (0.0, "level of a soft second beat, 0 = a single beat"),
        "dub_time": (0.3, "time of the second beat after the first (s)"),
        "drop": (0.05, "pitch drop after the strike (fraction of the pitch, relaxes in ~40 ms)"),
        "center": (0.0, "stereo position"),
        "width": (0.15, "stereo width of the modes"),
    }
    _BODY = np.array([172.0, 251.0, 338.0, 455.0])        # frame / shell and air-cavity modes (Hz)
    _BODY_G = np.array([1.0, 0.8, 0.55, 0.25])

    def _beat(self, f0, vel, n, rng, dry, felt, body, pos, drop, width, center):
        sr = SR
        ratios = np.array([r for _, r in _MEMBRANE]) * (1 + 0.008 * rng.standard_normal(len(_MEMBRANE)))
        ratios[0] *= 0.95                                   # air loading lowers the fundamental
        freqs = f0 * ratios
        tc = (0.0016 + 0.0012 * felt) * (1.15 - 0.3 * vel)  # contact time: soft felt stays longer on the skin
        w = _strike_weights(pos)
        amps = w * _contact_filter(freqs, tc)
        # the hand on the skin: the fundamental decays in 0.45 .. 0.18 s, the rest a little faster
        t60_1 = 0.45 - 0.27 * dry
        t60 = t60_1 * ratios ** -0.8 * np.exp(0.08 * rng.standard_normal(len(ratios)))
        bf = self._BODY * (1 + 0.02 * rng.standard_normal(len(self._BODY)))
        bamp = body * self._BODY_G * _contact_filter(bf, tc) * 1.6
        bt60 = (0.12 - 0.05 * dry) * (bf / 172.0) ** -0.6
        f_all = np.concatenate([freqs, bf])
        a_all = np.concatenate([amps, bamp])
        t_all = np.concatenate([t60, bt60])
        k = len(f_all)
        pans = center + width * rng.uniform(-1, 1, k) * np.minimum(np.concatenate([ratios, bf / f0]) - 0.9, 1)
        glide = np.concatenate([np.full(len(freqs), drop * (0.5 + 0.5 * vel)), np.zeros(len(bf))])
        y = render_modes(n, f_all, a_all, t_all, rng, pans, attack=tc * 0.3, glide=glide, glide_tau=0.04)
        # the felt on the skin: a short, dark contact noise (no click)
        kk = int(0.025 * sr)
        tt = np.arange(kk) / sr
        nz = svf(rng.standard_normal(kk), 700.0 - 250.0 * felt, 0.7, "lowpass") * np.exp(-tt / (0.003 + 0.002 * felt))
        nz *= np.minimum(tt / max(tc * 0.5, 1e-4), 1.0)
        pk = np.max(np.abs(y[:, :int(0.03 * sr)])) + 1e-12
        y[:, :kk] += 0.25 * (1.1 - felt) * nz / (np.max(np.abs(nz)) + 1e-12) * pk
        return y

    def render(self, note: Note, rng):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        vel = float(np.clip(note.vel_scalar() + 0.02 * rng.standard_normal(), 0.02, 1.0))
        f0 = float(midi_to_hz(note.midi)) if np.isfinite(note.midi) else 61.7
        f0 *= 1 + 0.003 * rng.standard_normal()
        dry = float(np.clip(P("dry"), 0, 1))
        args = (dry, float(np.clip(P("felt"), 0, 1)), float(P("body")), float(np.clip(P("position"), 0, 0.6)),
                float(P("drop")), float(P("width")), float(np.clip(P("center"), -1, 1)))
        dub = float(P("dub"))
        dt = float(P("dub_time")) * (1 + 0.03 * rng.standard_normal())
        n = int((0.75 + (dt if dub > 0 else 0.0)) * sr)
        y = self._beat(f0, vel, n, rng, *args)
        if dub > 0:
            s = int(dt * sr)
            y2 = self._beat(f0, vel * 0.85, n - s, rng, min(1.0, dry + 0.25), *args[1:])
            y[:, s:] += dub * y2
        # level: a beat at vel 0.7 has about the short-term loudness of the other instruments' notes
        y *= vel_amp(vel, 30.0) * 0.55
        y = signal.sosfilt(signal.butter(2, 25, "highpass", fs=sr, output="sos"), y, axis=-1)
        y[:, :8] *= raised_cos(8)
        return fade(y, 0, 0.12)
