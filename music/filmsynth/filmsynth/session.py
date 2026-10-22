"""Composer-facing API: Session (tempo map, tracks, buses, markers, curves), parallel rendering of
stems, master chain, events.json export.

    s = Session(bpm=72, bars=40)
    hall = s.bus("hall", Reverb("cathedral", rt60=5.0))
    pl = s.track("pluck", Pluck(), gain_db=-4, sends={"hall": -10},
                 inserts=[persistence_delay(s.curve(PERSISTENCE_KEYS))])
    pl.note(5, 3, 1, "B3", 0.6)
    res = s.render("out", target_lufs=-16)   # writes score.wav, stems/*.wav, events.json

    s = Session.from_timeline("films/demo-light", tail=0.0)   # the grid (bpm, bars, fps) from a film's timeline.json

Positions are 1-based bar/beat: note(bar, beat, ...) where beat may be fractional (2.5 = the "and"
of beat 2). String positions "13.2", "13.2+0.5" (beat fraction) and (bar, beat) tuples are accepted
wherever a position is expected.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context

import numpy as np

from .core import SR, midi_to_hz, note_name, seed_of, to_midi
from .instruments.base import Instrument, Note


# ---------------------------------------------------------------------------
# Tempo
# ---------------------------------------------------------------------------
class TempoMap:
    """Beat-indexed tempo: constant BPM with optional linear ramps (ritardando/accelerando).

    Beat index 0 = bar 1 beat 1. Ramps are linear in BPM over beat index.
    """

    def __init__(self, bpm: float = 72.0, beats_per_bar: int = 4, max_beats: int = 400):
        self.bpm = float(bpm)
        self.bpb = int(beats_per_bar)
        self.max_beats = max_beats
        self.ramps = []  # (b0, b1, bpm0, bpm1)
        self._build()

    def _bpm_at_beats(self, b):
        b = np.asarray(b, dtype=np.float64)
        out = np.full(b.shape, self.bpm)
        cur = np.full(b.shape, self.bpm)
        # ramps applied in order; after a ramp the tempo holds its end value
        for b0, b1, v0, v1 in sorted(self.ramps):
            inside = (b >= b0) & (b < b1)
            after = b >= b1
            out = np.where(inside, v0 + (v1 - v0) * (b - b0) / (b1 - b0), out)
            out = np.where(after, v1, out)
        del cur
        return out

    def _build(self):
        step = 1.0 / 64
        self._grid = np.arange(0, self.max_beats + step, step)
        spb = 60.0 / self._bpm_at_beats(self._grid)
        # trapezoid integration of seconds per beat
        self._time = np.concatenate([[0.0], np.cumsum(0.5 * (spb[1:] + spb[:-1]) * step)])

    def ramp(self, b0: float, b1: float, end_bpm: float, start_bpm: float | None = None):
        start = float(self._bpm_at_beats(np.array([b0]))[0]) if start_bpm is None else start_bpm
        self.ramps.append((float(b0), float(b1), start, float(end_bpm)))
        self._build()

    def time_of_beat(self, b):
        return np.interp(b, self._grid, self._time)

    def beat_of_time(self, t):
        return np.interp(t, self._time, self._grid)

    def beat_seconds_at(self, t):
        b = self.beat_of_time(np.asarray(t, dtype=np.float64))
        return 60.0 / self._bpm_at_beats(b)


def parse_pos(pos, bpb: int = 4) -> float:
    """Position -> beat index (0-based). Accepts 'bar.beat', 'bar.beat+frac', (bar, beat), dict."""
    if isinstance(pos, (tuple, list)):
        bar, beat = pos[0], pos[1]
        frac = pos[2] if len(pos) > 2 else 0.0
        return (bar - 1) * bpb + (beat - 1) + frac
    if isinstance(pos, dict):
        return (pos["bar"] - 1) * bpb + (pos.get("beat", 1) - 1) + pos.get("frac", 0.0)
    if isinstance(pos, str):
        s = pos.strip()
        frac = 0.0
        if "+" in s:
            s, f = s.split("+", 1)
            frac = float(f)
        bar, beat = s.split(".") if "." in s else (s, "1")
        return (int(bar) - 1) * bpb + (float(beat) - 1) + frac
    raise TypeError(f"cannot parse position {pos!r}; use 'bar.beat' or (bar, beat)")


def pos_string(b: float, bpb: int = 4) -> str:
    bar = int(np.floor(b / bpb + 1e-9)) + 1
    beat = b - (bar - 1) * bpb + 1
    ib = int(np.floor(beat + 1e-9))
    fr = beat - ib
    return f"{bar}.{ib}" + (f"+{fr:.3f}".rstrip("0").rstrip(".") if fr > 1e-6 else "")


# ---------------------------------------------------------------------------
# Curves (picklable automation)
# ---------------------------------------------------------------------------
class Curve:
    """Piecewise curve over seconds. Call with a time array (s) -> values.

    interp: 'linear' | 'step' | 'smooth' (cosine between keys)."""

    def __init__(self, times, values, interp: str = "linear"):
        o = np.argsort(times, kind="stable")
        self.t = np.asarray(times, dtype=np.float64)[o]
        self.v = np.asarray(values, dtype=np.float64)[o]
        self.interp = interp

    def __call__(self, t):
        t = np.asarray(t, dtype=np.float64)
        if self.interp == "step":
            i = np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.v) - 1)
            return self.v[i]
        if self.interp == "smooth":
            i = np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.v) - 2)
            t0, t1 = self.t[i], self.t[i + 1]
            x = np.clip((t - t0) / np.maximum(t1 - t0, 1e-9), 0, 1)
            x = 0.5 - 0.5 * np.cos(np.pi * x)
            return np.where(t < self.t[0], self.v[0], np.where(t > self.t[-1], self.v[-1],
                                                               self.v[i] + (self.v[i + 1] - self.v[i]) * x))
        return np.interp(t, self.t, self.v)


# ---------------------------------------------------------------------------
# Tracks and buses
# ---------------------------------------------------------------------------
@dataclass
class Track:
    session: "Session"
    name: str
    instrument: Instrument
    gain_db: float = 0.0
    pan: float = 0.0
    width: float = 1.0
    sends: dict = field(default_factory=dict)
    inserts: list = field(default_factory=list)
    stem: str | None = None
    humanize_ms: float = 0.0
    humanize_vel: float = 0.0
    notes: list = field(default_factory=list)
    gain_curve: Curve | None = None
    pan_curve: Curve | None = None
    mute: bool = False
    duck_spec: dict | None = None

    def note(self, bar, beat=1, dur_beats=1.0, pitch=None, vel=0.7, **expr):
        """Schedule a note. pitch: MIDI number or name ('F#3'); None for unpitched sounds.
        vel: 0..1, (v0, v1) ramp, or [(frac, v), ...]. **expr: instrument expression keys."""
        s = self.session
        b0 = parse_pos((bar, beat), s.bpb)
        return self._add(b0, dur_beats, pitch, vel, expr)

    def at(self, pos, dur_beats=1.0, pitch=None, vel=0.7, **expr):
        """Like note() but with a position string/tuple: at('13.2+0.5', 0.5, 'B3')."""
        return self._add(parse_pos(pos, self.session.bpb), dur_beats, pitch, vel, expr)

    def hit(self, bar, beat=1, vel=0.7, dur_beats=1.0, pitch=None, **expr):
        """Unpitched convenience: hit(bar, beat, vel)."""
        return self.note(bar, beat, dur_beats, pitch, vel, **expr)

    def seq(self, bar, beat, items, **expr):
        """Write consecutive notes. items: list of (dur_beats, pitch[, vel[, expr dict]]); pitch None = rest.
        Returns the beat index after the last item."""
        b = parse_pos((bar, beat), self.session.bpb)
        vdef = expr.pop("vel", 0.7)
        for it in items:
            d, p = it[0], it[1]
            v = it[2] if len(it) > 2 else vdef
            ex = dict(expr)
            if len(it) > 3:
                ex.update(it[3])
            if p is not None:
                self._add(b, d, p, v, ex)
            b += d
        return b

    def _add(self, b0, dur_beats, pitch, vel, expr):
        s = self.session
        self.instrument.check_expr(expr)
        idx = len(self.notes)
        seed = seed_of(s.seed, self.name, idx)
        rng = np.random.default_rng(seed)
        t0 = float(s.tempo.time_of_beat(b0))
        t1 = float(s.tempo.time_of_beat(b0 + dur_beats))
        if self.humanize_ms > 0:
            t0 += float(np.clip(rng.normal(0, self.humanize_ms * 1e-3), -3 * self.humanize_ms * 1e-3,
                                3 * self.humanize_ms * 1e-3))
        v = vel
        if self.humanize_vel > 0 and isinstance(vel, (int, float)):
            v = float(np.clip(vel + rng.normal(0, self.humanize_vel), 0.01, 1.0))
        m = to_midi(pitch) if pitch is not None else float("nan")
        nt = Note(t0=max(0.0, t0), dur=max(1e-3, t1 - t0), midi=m, vel=v, expr=dict(expr), seed=seed, index=idx,
                  beat_sec=float(s.tempo.beat_seconds_at(t0)), pos=pos_string(b0, s.bpb))
        nt.expr_beat = b0  # keep the nominal grid position
        self.notes.append(nt)
        return nt

    def duck(self, source: str, depth_db: float = 6.0, attack_ms: float = 8.0, release_ms: float = 250.0,
             thresh_db: float = -30.0):
        """Side-chain: lower this track by up to depth_db while track `source` is loud (RMS detector)."""
        self.duck_spec = dict(source=source, depth_db=depth_db, attack_ms=attack_ms, release_ms=release_ms,
                              thresh_db=thresh_db)
        return self

    def automate(self, param: str, keys, interp: str = "linear"):
        """param 'gain_db' or 'pan'; keys [(pos, value), ...] (pos as 'bar.beat' or seconds float)."""
        c = self.session.curve(keys, interp)
        if param == "gain_db":
            self.gain_curve = c
        elif param == "pan":
            self.pan_curve = c
        else:
            raise ValueError("automate() supports 'gain_db' and 'pan'; use inserts for others")
        return c


@dataclass
class Bus:
    name: str
    effects: list
    gain_db: float = 0.0
    stem: str | None = None


@dataclass
class RenderCtx:
    tempo: TempoMap

    def beat_seconds_at(self, t):
        return self.tempo.beat_seconds_at(t)


def _choke_gate(n, end_i, fade_s):
    """1 until end_i, then a raised-cosine fall to 0 over fade_s (used to kill tails at a cut)."""
    g = np.ones(n)
    k = max(1, int(fade_s * SR))
    a = max(0, end_i - k)
    x = (np.arange(a, end_i) - (end_i - k) + 1) / k
    g[a:end_i] = 0.5 + 0.5 * np.cos(np.pi * np.clip(x, 0, 1))
    g[end_i:] = 0.0
    return g


def _segments(n, chokes):
    """Split [0, n) at choke points -> list of (start_i, end_i, fade_s); end_i None for the last."""
    pts = sorted((int(t * SR), f) for t, f in chokes)
    out, a = [], 0
    for b, f in pts:
        out.append((a, b, f))
        a = b
    out.append((a, None, 0.0))
    return out


def _render_track(payload):
    """Worker: render one track to a dry, post-fader, panned stereo buffer.

    With chokes (from Session.cut(..., choke=True)), notes are rendered in groups split at each cut and
    every group's output (including its inserts' echoes) is gated to zero at the next cut: nothing that
    began before a cut can ring on after it."""
    (name, inst, notes, n_total, inserts, gain_db, pan, width, gain_curve, pan_curve, tempo, chokes) = payload
    t = time.time()
    ctx = RenderCtx(tempo)
    if chokes:
        buf = np.zeros((2, n_total))
        for a, b, f in _segments(n_total, chokes):
            grp = [nt for nt in notes if a <= int(round(nt.t0 * SR)) < (b if b is not None else n_total + 1)]
            if not grp:
                continue
            y = inst.render_notes(grp, n_total)
            for fx in inserts:
                y = fx.process(y, ctx)
            if b is not None:
                y = y * _choke_gate(n_total, b, f)
            buf += y
    else:
        buf = inst.render_notes(notes, n_total)
        for fx in inserts:
            buf = fx.process(buf, ctx)
    if width != 1.0:
        m = 0.5 * (buf[0] + buf[1])
        sd = 0.5 * (buf[0] - buf[1]) * width
        buf = np.vstack([m + sd, m - sd])
    tt = None
    g = 10 ** (gain_db / 20)
    if gain_curve is not None:
        tt = np.arange(n_total) / SR
        g = g * 10 ** (gain_curve(tt) / 20)
    p = pan
    if pan_curve is not None:
        tt = np.arange(n_total) / SR if tt is None else tt
        p = pan + pan_curve(tt)
    if np.any(np.asarray(p) != 0):
        th = (np.clip(p, -1, 1) + 1) * np.pi / 4
        gl = np.minimum(np.cos(th) * np.sqrt(2), 1.0)
        gr = np.minimum(np.sin(th) * np.sqrt(2), 1.0)
        buf = np.vstack([buf[0] * gl, buf[1] * gr])
    buf = buf * g
    return name, buf, time.time() - t


def _render_bus(payload):
    name, effects, x, gain_db, tempo, chokes = payload
    t = time.time()
    ctx = RenderCtx(tempo)
    n = x.shape[1]
    segs = _segments(n, chokes) if chokes else [(0, None, 0.0)]
    out = np.zeros_like(x)
    for a, b, f in segs:
        w = np.zeros(n)
        w[a:(b if b is not None else n)] = 1.0
        y = x * w if len(segs) > 1 else x
        if not np.any(y):
            continue
        for fx in effects:
            y = fx.process(y, ctx)
        if b is not None:
            y = y * _choke_gate(n, b, f)
        out += y
    return name, out * 10 ** (gain_db / 20), time.time() - t


@dataclass
class RenderResult:
    mix: np.ndarray
    stems: dict
    stats: dict
    paths: dict


class Session:
    def __init__(self, bpm: float = 72.0, beats_per_bar: int = 4, bars: int = 40, seed: int = 0,
                 tail: float = 6.0, sr: int = SR, fps: int = 24):
        if sr != SR:
            raise ValueError("filmsynth renders at 48 kHz only")
        self.bpb = beats_per_bar
        self.bars = bars
        self.seed = seed
        self.tail = tail
        self.fps = fps
        self.tempo = TempoMap(bpm, beats_per_bar, max_beats=(bars + 8) * beats_per_bar)
        self.tracks: dict[str, Track] = {}
        self.buses: dict[str, Bus] = {}
        self.markers = []
        self.curves_export = {}
        self.master_curve: Curve | None = None
        self.cuts = []
        self.custom_events = []
        self.timeline = None      # the film's timeline dict when built from one (from_timeline / film.timeline_session)

    @classmethod
    def from_timeline(cls, timeline=None, **kw) -> "Session":
        """A Session on a film's grid. `timeline`: a film directory, a timeline.json path, a loaded dict, or None
        ($FILM_DIR, else the current directory and its parents; see filmsynth.film.find_timeline). bpm, beats per
        bar, bars and fps come from the timeline; other arguments (seed, tail) pass through. The timeline is kept
        as session.timeline."""
        from .film import timeline_session

        return timeline_session(timeline, **kw)

    # -- time ---------------------------------------------------------------
    def beat(self, pos) -> float:
        return parse_pos(pos, self.bpb)

    def t(self, pos) -> float:
        """Seconds of a position ('13.2', (13, 2), {'bar':..})."""
        if isinstance(pos, (int, float)) and not isinstance(pos, bool):
            raise TypeError("t() takes a position, not seconds")
        return float(self.tempo.time_of_beat(parse_pos(pos, self.bpb)))

    def pos(self, seconds: float) -> str:
        return pos_string(float(self.tempo.beat_of_time(seconds)), self.bpb)

    def ritardando(self, from_pos, to_pos, end_bpm: float):
        """Linear BPM ramp between two positions; tempo holds end_bpm afterwards."""
        for tr in self.tracks.values():
            if tr.notes:
                raise RuntimeError("set tempo changes before writing notes")
        self.tempo.ramp(self.beat(from_pos), self.beat(to_pos), end_bpm)

    @property
    def end_time(self) -> float:
        return float(self.tempo.time_of_beat(self.bars * self.bpb))

    def curve(self, keys, interp: str = "linear") -> Curve:
        """Automation curve from [(pos, value), ...]; pos may be 'bar.beat' strings or seconds (float)."""
        ts, vs = [], []
        for k, v in keys:
            ts.append(float(k) if isinstance(k, (int, float)) else self.t(k))
            vs.append(float(v))
        return Curve(ts, vs, interp)

    # -- structure -----------------------------------------------------------
    def track(self, name, instrument, gain_db=0.0, pan=0.0, width=1.0, sends=None, inserts=None, stem=None,
              humanize_ms=0.0, humanize_vel=0.0) -> Track:
        if name in self.tracks:
            raise ValueError(f"track {name!r} exists")
        for b in (sends or {}):
            if b not in self.buses:
                raise ValueError(f"send to unknown bus {b!r}; create it first with session.bus()")
        tr = Track(self, name, instrument, gain_db, pan, width, dict(sends or {}), list(inserts or []), stem or name,
                   humanize_ms, humanize_vel)
        self.tracks[name] = tr
        return tr

    def bus(self, name, *effects, gain_db=0.0, stem=None) -> Bus:
        b = Bus(name, list(effects), gain_db, stem or f"fx_{name}")
        self.buses[name] = b
        return b

    def marker(self, pos, name, **data):
        self.markers.append({"name": name, "pos": pos if isinstance(pos, str) else pos_string(self.beat(pos), self.bpb),
                             "t": round(self.t(pos), 6), **data})

    def export_curve(self, name, curve: Curve, hop: float = 0.05):
        """Sample a curve into events.json under curves[name]."""
        self.curves_export[name] = (curve, hop)

    def master_gain(self, keys, interp="linear"):
        """Master gain automation (dB) applied to every stem and the mix."""
        self.master_curve = self.curve(keys, interp)

    def cut(self, start, end, fade_out_ms: float = 4.0, fade_in_ms: float = 40.0, floor_db: float = -120.0,
            choke: bool = True, tails: bool = False):
        """Hard cut: silence everything (tracks, echoes, reverb tails) between two positions.

        The master gate falls over fade_out_ms (a raised cosine, so no click) and reopens over
        fade_in_ms at `end`. With choke=True (default) every sound that started before `start`,
        including delay echoes and reverb tails, is also killed for good, so nothing from before the
        cut leaks back in when the gate reopens. Notes written at or after `start` play normally.

        tails=True makes it a dry cut: every track (and its inserts' echoes) stops at `start` exactly as
        above, but the buses are neither gated nor choked, so whatever was sent to a reverb before the
        cut rings out naturally (the music stops on the frame, the room keeps sounding). A stem may not
        mix tracks and buses when a dry cut is present (the two parts are gated differently)."""
        self.cuts.append((self.t(start), self.t(end), fade_out_ms * 1e-3, fade_in_ms * 1e-3, floor_db, choke,
                          bool(tails)))

    def event(self, type: str, at, **data):
        """A custom event for the picture: lands in events.json `events` as {t, frame, type, pos, **data}.
        `at` is a position ('6.1+0.5', (6, 1, 0.5)) or a Note returned by track.note()/at() (its actual onset,
        humanisation included, is used)."""
        if isinstance(at, Note):
            t = float(at.t0)
            pos = at.pos
        else:
            t = self.t(at)
            pos = at if isinstance(at, str) else pos_string(self.beat(at), self.bpb)
        self.custom_events.append({"t": round(t, 6), "frame": round(t * self.fps, 3), "type": type, "pos": pos,
                                   **data})

    def _master_gain_array(self, n, buses: bool = False):
        """Master gain and cut gates. buses=True: the gain for bus (reverb) stems, which dry cuts leave open."""
        g = np.ones(n)
        tt = np.arange(n) / SR
        if self.master_curve is not None:
            g *= 10 ** (self.master_curve(tt) / 20)
        for a, b, fo, fi, floor, _, tails in self.cuts:
            if buses and tails:
                continue
            fl = 10 ** (floor / 20)
            x = np.ones(n)
            i0, i1 = int(a * SR), int(b * SR)
            no, ni = max(1, int(fo * SR)), max(1, int(fi * SR))
            x[i0:i1] = fl
            k = np.arange(no)
            x[max(0, i0 - no):i0] = fl + (1 - fl) * (0.5 + 0.5 * np.cos(np.pi * (k[: i0 - max(0, i0 - no)] + 1) / no))
            k = np.arange(ni)
            seg = fl + (1 - fl) * (0.5 - 0.5 * np.cos(np.pi * (k + 1) / ni))
            x[i1:i1 + ni] = seg[: max(0, min(ni, n - i1))]
            g *= x
        return g

    # -- render ----------------------------------------------------------------
    def render(self, out_dir, mix_name: str = "score.wav", workers: int | None = None, target_lufs: float | None = -16.0,
               ceiling_dbtp: float = -1.0, stems: bool = True, events: bool = True, master_inserts=None,
               length: float | None = None, write: bool = True, verbose: bool = True,
               end_fade: float = 0.5) -> RenderResult:
        """Render all tracks in parallel, run buses, sum, apply master gain/cuts, master inserts,
        loudness normalisation + true-peak limiting. Writes mix, stems and events.json."""
        import soundfile as sf

        from .dsp.dynamics import limit, lufs, true_peak_db

        t_all = time.time()
        os.makedirs(out_dir, exist_ok=True)
        dur = length if length is not None else self.end_time + self.tail
        n = int(dur * SR)
        payloads, local = [], []
        chokes = [(a, fo) for a, b, fo, fi, floor, ch, _ in self.cuts if ch]
        bus_chokes = [(a, fo) for a, b, fo, fi, floor, ch, tails in self.cuts if ch and not tails]
        dry_cut = any(c[6] for c in self.cuts)
        if dry_cut:
            mixed = {tr.stem for tr in self.tracks.values()} & {b.stem for b in self.buses.values()}
            if mixed:
                raise ValueError(f"stems {sorted(mixed)} mix tracks and buses; a dry cut (tails=True) needs them apart")
        for tr in self.tracks.values():
            if tr.mute or not tr.notes:
                continue
            p = (tr.name, tr.instrument, tr.notes, n, tr.inserts, tr.gain_db, tr.pan, tr.width, tr.gain_curve,
                 tr.pan_curve, self.tempo, chokes)
            try:
                pickle.dumps(p)
                payloads.append(p)
            except Exception:
                local.append(p)
        stats = {"tracks": {}, "buses": {}}
        stem_bufs: dict[str, np.ndarray] = {}
        bus_in = {b: np.zeros((2, n)) for b in self.buses}

        held = {}
        need = set()
        for tr in self.tracks.values():
            if tr.duck_spec and not tr.mute and tr.notes:
                if tr.duck_spec["source"] not in self.tracks:
                    raise ValueError(f"duck source {tr.duck_spec['source']!r} is not a track")
                need |= {tr.name, tr.duck_spec["source"]}

        def collect(name, buf, secs):
            stats["tracks"][name] = round(secs, 2)
            if name in need:
                held[name] = buf
                return
            _sum(name, buf)

        def _sum(name, buf):
            tr = self.tracks[name]
            stem_bufs.setdefault(tr.stem, np.zeros((2, n)))
            stem_bufs[tr.stem] += buf
            for bname, lvl in tr.sends.items():
                bus_in[bname] += buf * 10 ** (lvl / 20)

        workers = workers if workers is not None else max(1, min(len(payloads), (os.cpu_count() or 4) - 1))
        if workers > 1 and len(payloads) > 1:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
                for name, buf, secs in ex.map(_render_track, payloads):
                    collect(name, buf, secs)
                    if verbose:
                        print(f"  track {name:<18s} {secs:6.2f}s", flush=True)
        else:
            local = payloads + local
            payloads = []
        for p in local:
            name, buf, secs = _render_track(p)
            collect(name, buf, secs)
            if verbose:
                print(f"  track {name:<18s} {secs:6.2f}s (local)", flush=True)

        if held:
            from .dsp.dynamics import duck as _duck
            for name, buf in held.items():
                spec = self.tracks[name].duck_spec
                if spec:
                    src = held.get(spec["source"])
                    if src is not None:
                        buf = _duck(buf, src, spec["depth_db"], spec["attack_ms"], spec["release_ms"], spec["thresh_db"])
                _sum(name, buf)
            held.clear()
        bus_payloads = [(b.name, b.effects, bus_in[b.name], b.gain_db, self.tempo, bus_chokes)
                        for b in self.buses.values() if np.any(bus_in[b.name])]
        if workers > 1 and len(bus_payloads) > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(bus_payloads)), mp_context=get_context("spawn")) as ex:
                results = list(ex.map(_render_bus, bus_payloads))
        else:
            results = [_render_bus(p) for p in bus_payloads]
        for name, y, secs in results:
            stats["buses"][name] = round(secs, 2)
            st = self.buses[name].stem
            stem_bufs.setdefault(st, np.zeros((2, n)))
            stem_bufs[st] += y
            if verbose:
                print(f"  bus   {name:<18s} {secs:6.2f}s", flush=True)
        del bus_in

        from .core import raised_cos
        from .dsp.filters import dc_block

        g = self._master_gain_array(n)
        g_bus = self._master_gain_array(n, buses=True) if dry_cut else g
        bus_stems = {b.stem for b in self.buses.values()}
        mix = np.zeros((2, n))
        for k in stem_bufs:
            # master gain/cuts, DC / sub-sonic clean-up, 5 ms fade-in (ensembles may start before 0);
            # all linear, so the stems still sum to the pre-limiter mix
            gk = g_bus if k in bus_stems else g
            stem_bufs[k] = dc_block(stem_bufs[k], 6.0) * gk  # DC block first: the gate must be last
            stem_bufs[k][:, :240] *= raised_cos(240)
            if end_fade > 0:
                ke = int(end_fade * SR)
                stem_bufs[k][:, n - ke:] *= raised_cos(ke)[::-1]
            mix += stem_bufs[k]
        for fx in master_inserts or []:
            mix = fx.process(mix, RenderCtx(self.tempo))
        norm_db = 0.0
        if target_lufs is not None:
            cur = lufs(mix)
            norm_db = target_lufs - cur if cur > -100 else 0.0
            mix = mix * 10 ** (norm_db / 20)
            for _ in range(3):
                mix = limit(mix, ceiling_dbtp - 0.3)
                if true_peak_db(mix) <= ceiling_dbtp + 0.02:
                    break
            # fine re-trim after limiting
            cur2 = lufs(mix)
            if abs(cur2 - target_lufs) > 0.2 and cur2 > -100:
                adj = target_lufs - cur2
                mix = limit(mix * 10 ** (adj / 20), ceiling_dbtp - 0.3)
                norm_db += adj
        for k in stem_bufs:
            stem_bufs[k] *= 10 ** (norm_db / 20)
        mix = np.clip(mix, -1.0, 1.0)
        stats["normalise_db"] = round(norm_db, 2)
        stats["render_s"] = round(time.time() - t_all, 2)
        paths = {}
        if write:
            mp = os.path.join(out_dir, mix_name)
            sf.write(mp, mix.T, SR, subtype="PCM_24")
            paths["mix"] = mp
            if stems:
                sd = os.path.join(out_dir, "stems")
                os.makedirs(sd, exist_ok=True)
                for k, y in stem_bufs.items():
                    pk = np.max(np.abs(y))
                    sp = os.path.join(sd, f"{k}.wav")
                    sf.write(sp, y.T, SR, subtype="PCM_24" if pk < 0.999 else "FLOAT")
                    paths[f"stem:{k}"] = sp
            if events:
                ep = os.path.join(out_dir, "events.json")
                self.export_events(ep, duration=dur)
                paths["events"] = ep
        if verbose:
            print(f"  rendered {dur:.1f}s audio in {stats['render_s']:.1f}s", flush=True)
        return RenderResult(mix, stem_bufs, stats, paths)

    # -- events ------------------------------------------------------------------
    def events(self, duration=None):
        bpb = self.bpb
        total_beats = self.bars * bpb
        beats = []
        for b in range(total_beats + 1):
            t = float(self.tempo.time_of_beat(b))
            beats.append({"bar": b // bpb + 1, "beat": b % bpb + 1, "t": round(t, 6), "frame": round(t * self.fps, 3)})
        bars = [{"bar": x["bar"], "t": x["t"], "frame": x["frame"]} for x in beats if x["beat"] == 1]
        notes = []
        for tr in self.tracks.values():
            if tr.mute:
                continue
            for nt in tr.notes:
                d = {"track": tr.name, "stem": tr.stem, "pos": nt.pos,
                     "beat": round(float(getattr(nt, "expr_beat", 0.0)), 6),
                     "start": round(nt.t0, 6), "end": round(nt.t0 + nt.dur, 6),
                     "frame": round(nt.t0 * self.fps, 3),
                     "vel": nt.vel_scalar()}
                if np.isfinite(nt.midi):
                    d.update({"pitch": note_name(nt.midi), "midi": round(nt.midi, 3),
                              "hz": round(float(midi_to_hz(nt.midi)), 3)})
                if nt.expr.get("tag"):
                    d["tag"] = nt.expr["tag"]
                notes.append(d)
        notes.sort(key=lambda d: (d["start"], d["track"]))
        curves = {}
        for name, (c, hop) in self.curves_export.items():
            ts = np.arange(0, self.end_time + 1e-9, hop)
            curves[name] = {"hop": hop, "t0": 0.0, "values": [round(float(v), 5) for v in c(ts)]}
        cuts = [{"start": round(a, 6), "end": round(b, 6), "choke": bool(ch), "tails": bool(tl)}
                for a, b, fo, fi, floor, ch, tl in self.cuts]
        return {
            "version": 1, "sampleRate": SR, "fps": self.fps, "bpm": self.tempo.bpm, "beatsPerBar": bpb,
            "bars": self.bars, "tempoRamps": [list(r) for r in self.tempo.ramps],
            "end": round(self.end_time, 6), "duration": round(duration or self.end_time + self.tail, 6),
            "beats": [x["t"] for x in beats],                      # seconds, index 0 = 1.1 (the picture's schema)
            "barTimes": bars, "beatTimes": beats, "markers": self.markers, "cuts": cuts,
            "events": sorted(self.custom_events, key=lambda e: (e["t"], e["type"])),
            "notes": notes, "curves": curves,
        }

    def export_events(self, path, duration=None):
        with open(path, "w") as f:
            json.dump(self.events(duration), f, ensure_ascii=False, indent=1)
