"""Instrument base class and shared helpers (pitch curves, ensemble deviations, voice allocation)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core import SR, add_into, midi_to_hz, pan_gains, smooth_noise, to_midi

# Expression keys every instrument accepts (handled by the base class / session).
COMMON_PARAMS = {
    "pan": (None, "per-note pan offset in [-1, 1] (added to the instrument's own placement)"),
    "gain_db": (0.0, "per-note gain in dB"),
    "tag": (None, "free text carried into events.json (e.g. 'motif', 'hit', 'answer')"),
    "voice": (None, "explicit voice id for legato phrasing (strings, choir)"),
}


@dataclass
class Note:
    """One scheduled note, already resolved to seconds by the session."""
    t0: float                 # onset (s)
    dur: float                # written duration (s)
    midi: float               # MIDI pitch (float, may be nan for unpitched sounds)
    vel: object               # 0..1, or (v0, v1) for a ramp across the note, or list of (frac, v)
    expr: dict = field(default_factory=dict)
    seed: int = 0
    index: int = 0
    beat_sec: float = 60.0 / 72.0
    pos: str = ""

    @property
    def hz(self):
        return float(midi_to_hz(self.midi))

    def vel_scalar(self) -> float:
        v = self.vel
        if isinstance(v, (tuple, list)):
            if len(v) and isinstance(v[0], (tuple, list)):
                return float(max(p[1] for p in v))
            return float(max(v))
        return float(v)


def vel_curve(vel, n: int):
    """Expand a velocity spec to a per-sample curve of length n.

    vel: float | (v0, v1) linear ramp | [(frac, v), ...] breakpoints with frac in 0..1 of the note.
    """
    if isinstance(vel, (int, float, np.floating)):
        return np.full(n, float(vel))
    vel = list(vel)
    if len(vel) and isinstance(vel[0], (tuple, list)):
        fr = np.array([p[0] for p in vel], dtype=float)
        vv = np.array([p[1] for p in vel], dtype=float)
    else:
        fr = np.linspace(0, 1, len(vel))
        vv = np.array(vel, dtype=float)
    x = np.linspace(0, 1, n)
    return np.interp(x, fr, vv)


def pitch_curve(n: int, midi0: float, rng: np.random.Generator, sr: int = SR, *,
                vib_cents: float = 0.0, vib_rate: float = 5.5, vib_delay: float = 0.3, vib_attack: float = 0.4,
                vib_rate_jitter: float = 0.06, vib_decay: float = 0.0,
                glide_from=None, glide_to=None, glide_start: float = 0.0, glide_time: float = 0.15,
                drift_cents: float = 0.0, drift_rate: float = 0.7, jitter_cents: float = 0.0,
                scoop_cents: float = 0.0, scoop_time: float = 0.08, detune_cents: float = 0.0):
    """Per-sample MIDI pitch curve with vibrato (delayed onset), glides, drift, jitter and scoop."""
    t = np.arange(n) / sr
    m = np.full(n, float(midi0) + detune_cents / 100.0)
    if glide_from is not None:
        g0 = to_midi(glide_from) - midi0
        k = np.clip(t / max(glide_time, 1e-3), 0, 1)
        sm = k * k * (3 - 2 * k)
        m += g0 * (1 - sm)
    if glide_to is not None:
        g1 = to_midi(glide_to) - midi0
        k = np.clip((t - glide_start) / max(glide_time, 1e-3), 0, 1)
        sm = k * k * (3 - 2 * k)
        m += g1 * sm
    if scoop_cents:
        m -= scoop_cents / 100.0 * np.exp(-t / max(scoop_time, 1e-3))
    if vib_cents > 0:
        rate = vib_rate * (1 + vib_rate_jitter * smooth_noise(n, 0.8, rng, sr))
        ph = 2 * np.pi * np.cumsum(rate) / sr + rng.uniform(0, 2 * np.pi)
        depth_env = np.clip((t - vib_delay) / max(vib_attack, 1e-3), 0, 1)
        depth_env = depth_env * depth_env * (3 - 2 * depth_env)
        if vib_decay > 0:
            depth_env *= np.exp(-np.maximum(t - vib_delay, 0) / vib_decay)
        depth = vib_cents * (1 + 0.15 * smooth_noise(n, 0.5, rng, sr))
        m += depth / 100.0 * depth_env * np.sin(ph)
    if drift_cents > 0:
        m += drift_cents / 100.0 * smooth_noise(n, drift_rate, rng, sr)
    if jitter_cents > 0:
        m += jitter_cents / 100.0 * smooth_noise(n, 30.0, rng, sr)
    return m


class Instrument:
    """Base class. Subclasses define PARAMS = {name: (default, doc)} and render(note, rng) -> (2, L).

    Construct with keyword overrides of any PARAMS default, e.g. Pluck(brightness=0.4).
    Per-note expression keywords override those again.
    """
    name = "instrument"
    PARAMS: dict = {}
    pitched = True
    # polyphony helpers
    legato_phrases = False   # if True, render_notes groups contiguous notes into phrases

    def __init__(self, **defaults):
        self.defaults = {k: v[0] for k, v in self.PARAMS.items()}
        for k, v in defaults.items():
            if k not in self.PARAMS:
                raise TypeError(f"{type(self).__name__}: unknown parameter {k!r}. Known: {sorted(self.PARAMS)}")
            self.defaults[k] = v

    # -- parameter access --------------------------------------------------
    def check_expr(self, expr: dict):
        for k in expr:
            if k not in self.PARAMS and k not in COMMON_PARAMS:
                raise TypeError(f"{type(self).__name__}: unknown expression {k!r}. "
                                f"Known: {sorted(set(self.PARAMS) | set(COMMON_PARAMS))}")

    def P(self, note: Note, key: str):
        return note.expr.get(key, self.defaults[key])

    @classmethod
    def describe(cls) -> str:
        lines = [f"{cls.__name__}: {cls.__doc__.strip().splitlines()[0] if cls.__doc__ else ''}"]
        for k, (d, doc) in cls.PARAMS.items():
            lines.append(f"  {k} = {d!r}: {doc}")
        return "\n".join(lines)

    # -- rendering ---------------------------------------------------------
    def render(self, note: Note, rng: np.random.Generator):  # pragma: no cover - abstract
        raise NotImplementedError

    def render_notes(self, notes, n_total: int, sr: int = SR):
        """Render all notes of a track into a (2, n_total) buffer. Default: independent notes."""
        buf = np.zeros((2, n_total))
        for note in notes:
            rng = np.random.default_rng(note.seed)
            y = self.render(note, rng)
            y = self._post(note, y)
            add_into(buf, y, int(round(note.t0 * sr)))
        return buf

    def _post(self, note: Note, y):
        g = 10 ** (float(note.expr.get("gain_db", 0.0) or 0.0) / 20)
        p = note.expr.get("pan")
        if p is not None and p != 0:
            th = (np.clip(p, -1, 1) + 1) * np.pi / 4
            gl, gr = np.cos(th) * np.sqrt(2), np.sin(th) * np.sqrt(2)
            y = np.vstack([y[0] * min(gl, 1.0), y[1] * min(gr, 1.0)])
        return y * g


def allocate_voices(notes, gap: float = 0.03):
    """Group notes into monophonic voices for legato instruments.

    Returns list of phrases; each phrase is a list of notes that follow each other (next onset within
    `gap` seconds of the previous note's end). Explicit expr['voice'] keeps notes in that voice.
    """
    notes = sorted(notes, key=lambda n: (n.t0, n.midi if np.isfinite(n.midi) else 0))
    voices = []  # list of dict(last_end, last_midi, notes, key)
    for nt in notes:
        key = nt.expr.get("voice")
        cand = None
        if key is not None:
            for v in voices:
                if v["key"] == key and v["end"] <= nt.t0 + gap:
                    cand = v
                    break
        else:
            best = None
            for v in voices:
                if v["key"] is None and v["end"] <= nt.t0 + gap:
                    d = abs(v["midi"] - nt.midi) + (0 if v["end"] >= nt.t0 - gap else 100)
                    if best is None or d < best[0]:
                        best = (d, v)
            cand = best[1] if best else None
        if cand is None:
            cand = {"key": key, "end": -1e9, "midi": nt.midi, "phrases": []}
            voices.append(cand)
        if cand["phrases"] and cand["end"] >= nt.t0 - gap:
            cand["phrases"][-1].append(nt)
        else:
            cand["phrases"].append([nt])
        cand["end"] = nt.t0 + nt.dur
        cand["midi"] = nt.midi
    out = []
    for v in voices:
        out.extend(v["phrases"])
    return out


def place_stereo(mono, pan: float):
    gl, gr = pan_gains(pan)
    return np.vstack([mono * gl, mono * gr])
