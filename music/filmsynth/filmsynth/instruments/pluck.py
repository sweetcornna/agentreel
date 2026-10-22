"""Plucked string between guqin and harp: extended Karplus-Strong (Jaffe-Smith) with
  * Thiran all-pass fractional delay for exact tuning (static notes),
    4-point Lagrange interpolation when the pitch moves (vibrato / glides),
  * frequency-dependent loss (one-pole loop filter solved from two T60 targets),
  * dispersion all-passes (string stiffness -> slightly sharp upper partials),
  * pluck-position comb in the excitation, finger-flesh low-passed excitation and thump,
  * two polarisations (slightly detuned, different decay) for beating and a two-stage decay,
  * modal body resonance (decaying-sinusoid IR, separate L/R bodies for width),
  * harmonics (fan yin) by keeping only multiples of n in the excitation spectrum,
  * yin/nao vibrato, glides with finger-slide noise, damping (stop) that speeds the decay.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy import signal

from ..core import SR, fade, midi_to_hz, raised_cos, undb, vel_amp
from ..dsp.filters import fir_from_modes, onepole_lp
from .base import Instrument, Note, pitch_curve


@njit(cache=True, fastmath=True)
def _ks(exc, n_out, dline, gain, bcoef, use_lagr, n_int, eta, disp_a, disp_m):
    """Core string loop. dline: per-sample delay (Lagrange mode) ; n_int/eta: static Thiran mode."""
    mx = n_int + 4.0
    for i in range(n_out):
        if dline[i] + 4 > mx:
            mx = dline[i] + 4
    size = 1
    while size < mx + 8:
        size *= 2
    mask = size - 1
    buf = np.zeros(size)
    out = np.zeros(n_out)
    lp = 0.0
    apx = 0.0
    apy = 0.0
    dx = np.zeros(8)
    dy = np.zeros(8)
    ne = exc.shape[0]
    for i in range(n_out):
        if use_lagr:
            rp = i - dline[i]
            ip = int(np.floor(rp))
            f = rp - ip
            xm1 = buf[(ip - 1) & mask]
            x0 = buf[ip & mask]
            x1 = buf[(ip + 1) & mask]
            x2 = buf[(ip + 2) & mask]
            # 4-point Lagrange around x0..x1 (offsets -1, 0, 1, 2)
            d = f
            v = (-d * (d - 1) * (d - 2) / 6.0) * xm1 + ((d + 1) * (d - 1) * (d - 2) / 2.0) * x0 \
                + (-(d + 1) * d * (d - 2) / 2.0) * x1 + ((d + 1) * d * (d - 1) / 6.0) * x2
        else:
            xin = buf[(i - n_int) & mask]
            v = eta * xin + apx - eta * apy
            apx = xin
            apy = v
        b = bcoef[i]
        lp = (1.0 - b) * v + b * lp
        s = gain[i] * lp
        for k in range(disp_m):
            y = disp_a * s + dx[k] - disp_a * dy[k]
            dx[k] = s
            dy[k] = y
            s = y
        if i < ne:
            s += exc[i]
        buf[i & mask] = s
        out[i] = s
    return out


def _phase_delay_onepole(b, w):
    return np.arctan2(b * np.sin(w), 1 - b * np.cos(w)) / w


def _phase_delay_ap1(a, w):
    z = np.exp(-1j * w)
    h = (a + z) / (1 + a * z)
    ph = np.angle(h)
    if ph > 0:
        ph -= 2 * np.pi
    return -ph / w


def _loop_gain_mag(b, w):
    return (1 - b) / np.sqrt(1 - 2 * b * np.cos(w) + b * b)


def _solve_b(f0, t60_f0, t60_hi, f_hi, sr):
    """Find the one-pole coefficient b so that the loop gives T60(f_hi)/T60(f0) as requested."""
    w0 = 2 * np.pi * f0 / sr
    wh = 2 * np.pi * f_hi / sr
    target_ratio = np.log(10 ** (-3 / (f0 * t60_hi))) / np.log(10 ** (-3 / (f0 * t60_f0)))
    lo, hi = 0.0, 0.98
    for _ in range(50):
        b = 0.5 * (lo + hi)
        G0 = 10 ** (-3 / (f0 * t60_f0))
        g = G0 / _loop_gain_mag(b, w0)
        Gh = g * _loop_gain_mag(b, wh)
        if Gh >= 1:
            lo = b
            continue
        ratio = np.log(Gh) / np.log(G0)
        if ratio < target_ratio:
            lo = b
        else:
            hi = b
    return 0.5 * (lo + hi)


# Body modes: a mid-size wooden box (guqin-ish low resonances + harp soundboard brightness)
_BODY_F = np.array([98, 176, 248, 330, 412, 520, 660, 830, 1040, 1310, 1650, 2100, 2650, 3300, 4200, 5300])
_BODY_T = np.array([0.32, 0.26, 0.22, 0.18, 0.15, 0.12, 0.10, 0.085, 0.07, 0.06, 0.05, 0.04, 0.032, 0.026, 0.02, 0.015])
_BODY_G = np.array([1.0, 0.9, 0.85, 0.7, 0.65, 0.55, 0.5, 0.42, 0.36, 0.3, 0.25, 0.2, 0.16, 0.12, 0.09, 0.06])

_body_cache = {}


def _body_irs(seed: int, wood: float):
    key = (seed, round(wood, 3))
    if key not in _body_cache:
        hs = []
        for c in range(2):
            rng = np.random.default_rng(seed * 7 + c)
            fr = _BODY_F * (1 + 0.035 * rng.standard_normal(len(_BODY_F))) * (0.85 + 0.3 * wood)
            t6 = _BODY_T * 0.5 * (0.8 + 0.4 * rng.random(len(_BODY_F)))
            g = _BODY_G * (0.7 + 0.6 * rng.random(len(_BODY_F)))
            # plus a dense field of weaker soundboard modes so notes between the main modes are not dull
            rf = np.exp(rng.uniform(np.log(150), np.log(6000), 40))
            fr = np.concatenate([fr, rf])
            t6 = np.concatenate([t6, np.clip(6.9 * rng.uniform(8, 20, 40) / (np.pi * rf), 0.004, 0.12)])
            g = np.concatenate([g, 0.35 * rng.random(40) * np.clip(1500 / rf, 0.15, 1.0)])
            h = fir_from_modes(fr, t6, g / t6 * 0.05, 0.45, rng)  # peak response ~ gain
            hs.append(h / (np.sqrt(np.sum(h ** 2)) + 1e-12))
        # one body seen from two directions: mostly common, a little different (natural width)
        irs = [hs[0] - 0.35 * hs[1], hs[0] + 0.35 * hs[1]]
        irs = [h / (np.sqrt(np.sum(h ** 2)) + 1e-12) for h in irs]
        _body_cache[key] = np.array(irs)
    return _body_cache[key]


class Pluck(Instrument):
    """Warm plucked string (guqin/harp hybrid), range B1-B5. Warm and close."""
    name = "pluck"
    PARAMS = {
        "brightness": (0.5, "0 dark/flesh .. 1 bright/nail; also scales with velocity"),
        "pluck_pos": (0.18, "pluck position as a fraction of the string (0.05 near bridge .. 0.5 middle)"),
        "decay": (1.0, "multiplies the natural T60 (low notes ~9 s, high ~2.5 s)"),
        "stiffness": (0.35, "0 ideal string .. 1 stiff (more inharmonic upper partials)"),
        "body": (0.45, "amount of wooden body resonance 0..1"),
        "wood": (0.5, "body size/colour 0 small-bright .. 1 large-dark"),
        "attack": (0.0, "0 normal; >0 softens the onset over that many seconds (brushed / rolled)"),
        "let_ring": (True, "True: ring past the written duration; False: damp at the written end"),
        "max_ring": (7.0, "cap on ring length in seconds"),
        "damp_time": (0.12, "T60 (s) after damping"),
        "harmonic": (0, "0 normal; n>=2 plays the n-th harmonic (fan yin) of the written pitch's string"),
        "vib_cents": (0.0, "yin/nao vibrato depth in cents (yin ~8, nao ~25)"),
        "vib_rate": (2.8, "vibrato rate Hz (yin 3-4, nao 1.5-2.5)"),
        "vib_delay": (0.25, "seconds before vibrato starts"),
        "vib_decay": (1.2, "vibrato depth decay time (s); 0 = constant"),
        "glide_to": (None, "pitch to slide to (shang/xia slide)"),
        "glide_from": (None, "pitch to slide from at the onset"),
        "glide_start": (0.25, "seconds after onset when glide_to begins"),
        "glide_time": (0.35, "glide duration (s)"),
        "slide_noise": (1.0, "finger-slide squeak level during glides"),
        "thump": (0.35, "finger-flesh thump level"),
        "polar": (0.35, "second polarisation level (beating, two-stage decay)"),
        "spread": (0.35, "stereo placement by pitch (low left, high right)"),
        "body_seed": (None, "None: each note gets one of 7 wooden bodies at random (natural variety, +-3 dB); 0..6: always "
                            "that body (an even pulse)"),
    }

    def render(self, note: Note, rng: np.random.Generator):
        sr = SR
        P = lambda k: self.P(note, k)  # noqa: E731
        vel = float(np.clip(note.vel_scalar() + 0.02 * rng.standard_normal(), 0.02, 1.0))
        midi = note.midi + 0.012 * rng.standard_normal()  # +-1 cent
        harm = int(P("harmonic") or 0)
        f_string = float(midi_to_hz(midi))
        f0 = f_string * (harm if harm >= 2 else 1)
        bright = float(np.clip(P("brightness") + 0.04 * rng.standard_normal(), 0, 1))
        hard = np.clip(0.25 + 0.55 * vel * (0.5 + bright), 0.05, 1.0)

        # natural decay: long in the bass, shorter up high
        t60 = float(np.clip(10.5 * (f_string / 55.0) ** -0.42, 1.6, 12.0)) * P("decay")
        if harm >= 2:
            t60 *= 0.8
        t60_hi = t60 * (0.05 + 0.22 * bright)
        f_hi = min(4000.0, 0.42 * sr)
        if f_hi <= f0 * 1.5:
            f_hi = min(f0 * 3.0, 0.45 * sr)
            t60_hi = t60 * 0.35

        written = note.dur
        if P("let_ring"):
            length = min(P("max_ring"), t60 * 1.05)
            damp_at = None
        else:
            damp_at = max(0.03, written)
            length = min(P("max_ring"), damp_at + P("damp_time") * 1.3 + 0.05)
        n = int(length * sr)

        # pitch curve (string frequency)
        moving = bool(P("vib_cents") > 0 or P("glide_to") is not None or P("glide_from") is not None)
        mc = pitch_curve(n, midi, rng, sr, vib_cents=P("vib_cents"), vib_rate=P("vib_rate"),
                         vib_delay=P("vib_delay"), vib_attack=0.3, vib_decay=P("vib_decay"),
                         glide_from=P("glide_from"), glide_to=P("glide_to"), glide_start=P("glide_start"),
                         glide_time=P("glide_time"), drift_cents=0.6, drift_rate=0.4)
        fcurve = midi_to_hz(mc) * (harm if harm >= 2 else 1)

        # loop filter design
        b = _solve_b(f0, t60, t60_hi, f_hi, sr)
        w0 = 2 * np.pi * f0 / sr
        G0 = 10 ** (-3 / (f0 * t60))
        g0 = G0 / _loop_gain_mag(b, w0)
        stiff = float(P("stiffness"))
        disp_m = 2 if f_string < 400 else 1
        disp_a = -float(np.clip(stiff * 0.45 * (110.0 / max(f_string, 40)) ** 0.5, 0.0, 0.7))
        if harm >= 2:
            disp_a *= 0.5

        def total_extra(f):
            w = 2 * np.pi * f / sr
            return _phase_delay_onepole(b, w) + disp_m * _phase_delay_ap1(disp_a, w)

        gain = np.full(n, g0)
        bcurve = np.full(n, b)
        if damp_at is not None:
            i0 = int(damp_at * sr)
            Gd = 10 ** (-3 / (f0 * P("damp_time")))
            gd = Gd / _loop_gain_mag(min(b + 0.25, 0.9), w0)
            ramp = np.clip((np.arange(n) - i0) / (0.02 * sr), 0, 1)
            gain = g0 + (gd - g0) * ramp
            bcurve = b + (min(b + 0.25, 0.9) - b) * ramp

        # ---- excitation: triangle displacement with apex at the pluck point + flesh-filtered noise
        Pn = sr / f0
        Pi = int(np.floor(Pn))
        beta = float(np.clip(P("pluck_pos") + 0.015 * rng.standard_normal(), 0.04, 0.5))
        x = np.arange(Pi) / Pi
        tri = np.where(x < beta, x / beta, (1 - x) / (1 - beta))
        noise = rng.standard_normal(Pi)
        fc_exc = float(np.clip(f0 * (2.0 + 22.0 * hard), 600, 14000))
        noise = signal.lfilter([1 - np.exp(-2 * np.pi * fc_exc / sr)], [1, -np.exp(-2 * np.pi * fc_exc / sr)], noise)
        # pick-position comb on the noise part too
        kd = max(1, int(round(beta * Pi)))
        noise = noise - np.concatenate([np.zeros(kd), noise[:-kd]])
        noise *= 0.35 * hard / (np.std(noise) + 1e-12) * np.std(tri - tri.mean())
        exc = (tri - tri.mean()) + noise
        if harm >= 2:
            # keep only multiples of the harmonic number (finger touches the node)
            E = np.fft.rfft(exc)
            k = np.arange(len(E))
            E[(k % harm) != 0] = 0
            exc = np.fft.irfft(E, Pi)
            # the loop for a harmonic runs at f0 = n * f_string, so keep one period of that
            Pi2 = int(np.floor(sr / f0))
            exc = exc[:Pi2] if harm >= 2 else exc
        exc = exc - exc.mean()
        exc /= np.max(np.abs(exc)) + 1e-12
        att = float(P("attack"))
        if att > 0:
            # soft onset: spread the excitation over `attack` seconds (rolled / brushed)
            na = int(att * sr)
            reps = max(1, na // len(exc))
            exc = np.tile(exc, reps) * np.hanning(len(exc) * reps + 2)[1:-1]

        def run(detune_cents, gscale, bshift):
            fc = fcurve * 2 ** (detune_cents / 1200)
            gl = np.clip(gain * gscale, 0, 0.99999)
            bc = np.clip(bcurve + bshift, 0.0, 0.95)
            if moving:
                ff = np.array([fc.min() * 0.98, f0, fc.max() * 1.02])
                ex = np.array([total_extra(f) for f in ff])
                dl = sr / fc - np.interp(fc, ff, ex)
                return _ks(exc, n, np.ascontiguousarray(dl), gl, bc, True, 0, 0.0, disp_a, disp_m)
            f = f0 * 2 ** (detune_cents / 1200)
            D = sr / f - total_extra(f)
            n_int = int(np.floor(D - 0.5))
            d = D - n_int
            eta = (1 - d) / (1 + d)
            return _ks(exc, n, np.zeros(n), gl, bc, False, n_int, eta, disp_a, disp_m)

        s1 = run(0.0, 1.0, 0.0)
        pol = float(P("polar"))
        if pol > 0:
            det = rng.uniform(0.25, 0.9) * rng.choice([-1, 1])
            # second polarisation decays more slowly (weakly coupled to the bridge)
            s2 = run(det, 1.0 + (1 - g0) * 0.45, 0.02)
            s = s1 + pol * s2
        else:
            s = s1

        # level normalisation: equal loudness across the range (short high strings carry less energy
        # in a KS loop), with a gentle natural tilt
        k0 = min(n, int(0.3 * sr))
        s = s / (np.sqrt(np.mean(s[:k0] ** 2)) + 1e-12) * 0.25 * (f_string / 247.0) ** -0.1
        # finger thump (flesh) + slide noise
        thump = float(P("thump")) * (0.4 + 0.6 * vel)
        if thump > 0:
            nt = int(0.03 * sr)
            tt = np.arange(nt) / sr
            th = np.sin(2 * np.pi * 140 * tt) * np.exp(-tt / 0.006) + 0.6 * onepole_lp(rng.standard_normal(nt), 700) * np.exp(-tt / 0.004)
            th *= raised_cos(min(24, nt)).tolist() + [1.0] * (nt - min(24, nt))
            s[:nt] += thump * 0.6 * th / (np.max(np.abs(th)) + 1e-12)
        if (P("glide_to") is not None or P("glide_from") is not None) and P("slide_noise") > 0:
            speed = np.abs(np.gradient(mc)) * sr  # semitones per second
            env = np.clip(speed / 12.0, 0, 1)
            env = onepole_lp(env, 30)
            sq = signal.lfilter(*signal.butter(2, [900, 3200], "bandpass", fs=sr), rng.standard_normal(n))
            s += P("slide_noise") * 0.05 * env * sq

        # level, body, stereo
        amp = vel_amp(vel, 28.0) * 0.2
        s = s * amp
        body = float(P("body"))
        bs = P("body_seed")
        irs = _body_irs(int(note.seed % 7) if bs is None else int(bs) % 7, float(P("wood")))
        L = s + body * signal.oaconvolve(s, irs[0])[:n] * 0.8
        R = s + body * signal.oaconvolve(s, irs[1])[:n] * 0.8
        pan = float(np.clip((note.midi - 59) / 24.0, -1, 1)) * P("spread")
        gl, gr = np.cos((pan + 1) * np.pi / 4) * np.sqrt(2), np.sin((pan + 1) * np.pi / 4) * np.sqrt(2)
        y = np.vstack([L * min(gl, 1), R * min(gr, 1)])
        # first sample ramp (avoid a step) and tail fade
        y[:, :32] *= raised_cos(32)
        y = fade(y, 0.0, min(0.25, 0.2 * length))
        # remove DC that the triangle can leave
        y = signal.lfilter([1, -1], [1, -0.9995], y, axis=-1)
        return y
