"""Soft sustained pads by additive synthesis, and a sub-bass sine (pure DSP).

    pad(notes, voices=5, ...)       ensemble of slightly detuned additive 'string' voices
    sub(note, dur, ...)             very low sine with slow attack / release (mono centre)

pad is meant to sound like a soft string section holding notes, not like a synth pad:

* every note is `voices` independent players: each has its own detune (signs alternate,
  so the section stays centred on pitch), slow pitch drift, a shallow vibrato that only
  builds after `vibrato_delay`, a random set of harmonic phases (no coherent 'buzz'),
  its own stagger in the attack and a slow +-0.8 dB bow-pressure fluctuation;
* the spectrum is a bowed-string sawtooth (1/k) under a 2nd-order roll-off that moves
  slowly between a dark (900 Hz) and a bright (3.5 kHz) setting: `brightness` sets the
  centre, `motion` how far a slow random control (`motion_rate` Hz) moves it;
* a fixed 'body' EQ (resonances at 250 / 480 / 2700 Hz, dip at 1.25 kHz, soft top), the
  same for every pitch, which is what a real instrument body does and a synth does not;
* a little bow / air noise (band-passed 1.5-7 kHz, following the envelope).

Evaluation: the additive spectrum of each voice is summed once into two single-cycle
tables (dark / bright, only harmonics below f_max) and read with a phase accumulator.
That is exactly the sum of those harmonics (just cheaper), strictly band-limited.
"""
from __future__ import annotations

import numpy as np

from ._dsp import (SR, biquad, butter, cos_ramp, db2a, fade_edges, midi_to_hz, pan_gains,
                   smooth_noise, sosfilt)

TABLE = 16384
BODY_EQ = (('peak', 250.0, 1.4, 3.0), ('peak', 480.0, 1.2, 2.0), ('peak', 1250.0, 1.5, -2.5),
           ('peak', 2700.0, 1.3, 2.0), ('highshelf', 6000.0, 0.7, -4.0))
OUT_CAL = 0.132      # one note at velocity 80 -> about -24 dBFS RMS (measured D2..D5)


def _note_num(m):
    if isinstance(m, str):
        from .sfz import note_number
        return float(note_number(m))
    return float(m)


def _tables(f0, rng, f_max, fc_dark=900.0, fc_bright=3500.0, slope=1.0):
    """Two single-cycle tables (dark, bright) with the same random harmonic phases."""
    kmax = max(1, int(f_max / f0))
    k = np.arange(1, kmax + 1)
    ph = rng.uniform(0, 2 * np.pi, kmax)
    out = []
    for fc in (fc_dark, fc_bright):
        a = k ** -slope / (1.0 + (k * f0 / fc) ** 4) ** 0.5
        a *= np.clip((f_max - k * f0) / (0.15 * f_max), 0.0, 1.0)      # soft top
        spec = np.zeros(TABLE // 2 + 1, dtype=np.complex128)
        spec[1:kmax + 1] = a * np.exp(1j * ph) * TABLE / 2
        w = np.fft.irfft(spec, TABLE)
        out.append(w / np.sqrt((w ** 2).mean()))
    return out


def _read(table, cyc):
    """Linear-interpolated table read; cyc = phase in cycles (float64)."""
    pos = np.mod(cyc, 1.0) * TABLE
    i = pos.astype(np.int64)
    f = pos - i
    t2 = np.append(table, table[0])
    return t2[i] * (1 - f) + t2[i + 1] * f


def pad(notes, *, voices: int = 5, detune=(3.0, 9.0), brightness: float = 0.45,
        motion: float = 0.5, motion_rate: float = 0.08, air: float = 0.06, attack: float = 1.2,
        release: float = 2.0, vibrato_cents: float = 4.0, vibrato_rate=(4.6, 5.6),
        vibrato_delay: float = 0.6, width: float = 0.8, body: bool = True, f_max: float = 9000.0,
        gain_db: float = 0.0, duration=None, seed: int = 0) -> np.ndarray:
    """Render a soft string-like pad. notes = [(start_s, dur_s, midi_or_name, velocity)].

    voices       : players per note (ensemble size).
    detune       : per-player detune magnitude range in cents (signs alternate).
    brightness   : 0 (dark, ~900 Hz roll-off) .. 1 (bright, ~3.5 kHz); velocity adds a little.
    motion       : depth of the slow brightness movement (0 = static).
    motion_rate  : speed of that movement (Hz, ~0.05-0.15 = one swell every 7-20 s).
    air          : bow / air noise level (0 = none).
    attack / release : seconds (each player staggers its attack by +-15 %).
    vibrato_cents: section vibrato depth (per player 0.6-1.4x); 0 = none.
    width        : stereo spread of the players (0 = mono).
    body         : fixed body-resonance EQ on/off.
    f_max        : highest harmonic frequency.
    Output is not normalised: one note at velocity 80 is about -24 dBFS RMS.
    """
    rng = np.random.default_rng(seed)
    ns = []
    for n in notes:
        st, du, m = float(n[0]), float(n[1]), _note_num(n[2])
        vel = float(n[3]) if len(n) > 3 else 80.0
        if du > 0 and vel > 0:
            ns.append((st, du, m, float(np.clip(vel, 1, 127))))
    end = max((st + du for st, du, _, _ in ns), default=0.0) + release * 1.2 + 0.2
    total = duration if duration is not None else end
    out = np.zeros((max(1, int(round(total * SR))), 2))
    nv = max(1, int(voices))
    for st, du, m, vel in ns:
        v = vel / 127.0
        f_note = float(midi_to_hz(m))
        n_len = int(round((du + release * 1.2) * SR)) + 1
        t = np.arange(n_len) / SR
        s0 = int(round(st * SR))
        # note-level slow brightness control shared by the section, plus small per-player offsets
        b_note = brightness + 0.15 * (v - 0.5) + motion * 0.35 * smooth_noise(n_len, motion_rate, rng)
        lvl = db2a(-20.0 * (1 - v) ** 1.3) / np.sqrt(nv)
        signs = np.where(np.arange(nv) % 2 == 0, 1.0, -1.0)
        rng.shuffle(signs)
        pans = np.linspace(-1, 1, nv) if nv > 1 else np.zeros(1)
        rng.shuffle(pans)
        acc = np.zeros((n_len, 2))
        for j in range(nv):
            prng = np.random.default_rng(rng.integers(1 << 62))
            cents = signs[j] * prng.uniform(*detune) if nv > 1 else 0.0
            cents = cents + 2.5 * smooth_noise(n_len, 0.15, prng)
            if vibrato_cents:
                vr = prng.uniform(*vibrato_rate)
                vd = vibrato_cents * prng.uniform(0.6, 1.4)
                build = np.clip((t - vibrato_delay * prng.uniform(0.8, 1.3)) / 0.8, 0, 1)
                build = build * build * (3 - 2 * build)
                cents = cents + vd * build * np.sin(2 * np.pi * vr * t + prng.uniform(0, 2 * np.pi))
            f0 = f_note * 2.0 ** (cents / 1200.0)
            cyc = prng.uniform(0, 1) + np.cumsum(f0) / SR
            dark, bright = _tables(f_note, prng, min(f_max, SR * 0.45))
            beta = np.clip(b_note + 0.08 * smooth_noise(n_len, motion_rate * 1.7, prng), 0.0, 1.0)
            y = _read(dark, cyc)
            y += beta * (_read(bright, cyc) - y)
            a_s = attack * prng.uniform(0.85, 1.15)
            env = np.clip(t / max(a_s, 1e-3), 0, 1)
            env = (env * env * (3 - 2 * env)) ** 1.5
            rel = np.clip((t - du) / max(release, 1e-3), 0, 1)
            env *= 0.5 + 0.5 * np.cos(np.pi * rel)
            env *= db2a(0.8 * smooth_noise(n_len, 0.3, prng))            # bow pressure
            y *= env * lvl * db2a(prng.normal(0, 0.7))
            p = float(np.clip(pans[j] * width + prng.uniform(-0.1, 0.1) * width, -1, 1))
            gl, gr = pan_gains(p)
            itd = int(abs(p) * 0.0003 * SR)
            if p >= 0:
                acc[itd:, 0] += y[:n_len - itd] * gl
                acc[:, 1] += y * gr
            else:
                acc[:, 0] += y * gl
                acc[itd:, 1] += y[:n_len - itd] * gr
        if air:
            a_env = np.clip(t / max(attack, 1e-3), 0, 1) ** 2
            a_env *= 0.5 + 0.5 * np.cos(np.pi * np.clip((t - du) / max(release * 0.7, 1e-3), 0, 1))
            a_env *= np.clip(1 + 0.35 * smooth_noise(n_len, 4.0, rng), 0, None)
            nz = sosfilt(butter('bandpass', (1500.0, 7000.0), 2), rng.standard_normal((n_len, 2)))
            acc += nz.astype(np.float64) * (a_env * air * 0.5 * db2a(-20.0 * (1 - v) ** 1.3))[:, None]
        a = max(0, s0)
        b = min(len(out), s0 + n_len)
        if b > a:
            out[a:b] += acc[a - s0:b - s0]
    y = out
    if body:
        for kind, f, q, g in BODY_EQ:
            y = sosfilt(biquad(kind, f, q, g), y).astype(np.float64)
    y = sosfilt(butter('highpass', 35.0, 2), y)
    y = y * np.float32(OUT_CAL * db2a(gain_db))
    return fade_edges(y.astype(np.float32), 0.005, 0.02)


def sub(note, dur: float, *, attack: float = 1.5, release: float = 2.5, gain_db: float = -12.0,
        h2_db=None, seed: int = 0) -> np.ndarray:
    """Very low sine sustain: note (MIDI or name) held for dur s, raised-cosine attack and
    release (the clip is dur + release long and starts at the first sample). Identical in
    both channels (low end stays mono). h2_db adds a 2nd harmonic at that level (dB re.
    the fundamental) so the note stays audible on small speakers; default None = pure sine.
    gain_db is the sustain peak level in dBFS.
    """
    f = float(midi_to_hz(_note_num(note)))
    n = int(round((dur + release) * SR))
    t = np.arange(n) / SR
    y = np.sin(2 * np.pi * f * t)
    if h2_db is not None:
        y = y + db2a(h2_db) * np.sin(2 * np.pi * 2 * f * t)
    env = np.ones(n)
    a = min(n, int(round(attack * SR)))
    if a > 1:
        env[:a] = cos_ramp(a)
    r0 = min(n, int(round(dur * SR)))
    r = n - r0
    if r > 1:
        env[r0:] *= cos_ramp(r, rising=False)
    y = y * env * db2a(gain_db)
    y[0] = 0.0
    y[-1] = 0.0
    return np.stack([y, y], axis=1).astype(np.float32)
