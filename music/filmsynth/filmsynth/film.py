"""A film's timeline (read-only): where it is, its grid, its curves, its sync points and its loudness targets.

A film is a directory holding a timeline.json, the contract the picture and the music share (bpm, beats per bar,
frames per beat, bars, sections, shots, curves such as `energy`, the `sync` list, the loudness targets). Nothing
about a particular film is hard-coded here: every number comes from the timeline you point it at.

    from filmsynth.film import load_timeline, timeline_session, timeline_curve, mark_sync, music_out
    tl = load_timeline("films/demo-light")              # a film directory, a timeline file, or None (find_timeline)
    s = timeline_session(tl, tail=0.0, seed=1)          # bpm, bars, beats per bar and fps from the timeline
    energy = timeline_curve(s, tl, "energy")            # Curve: t (s) -> 0..1
    s.export_curve("energy", energy)
    mark_sync(s, tl)                                    # the timeline's sync points as markers in events.json
    s.render(music_out("films/demo-light"))             # <film>/out/music/: score.wav, stems/, events.json

`Session.from_timeline(where, **kw)` is the same as timeline_session(load_timeline(where), **kw).

The timeline uses 'bar.beat' strings, the same notation as Session positions, plus two forms Session does not
parse: a frame offset 'bar.beat+4f' (tl_pos() turns it into beats with framesPerBeat) and a range sync point
{"id": "T1-T6", "at": "1.3-2.4"} (sync_points() expands it into one point per beat when the ids and the beats
match up, as T1..T6 on 1.3..2.4 do; any other range, e.g. {"id": "P1", "at": "1.3-2.4"}, stays one point at its
first beat and carries the end as `to`).
"""
from __future__ import annotations

import json
import os
import re

from .session import parse_pos

ENV_FILM = "FILM_DIR"                                   # default film directory when none is given
_FRAMES = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*\+\s*(-?\d+(?:\.\d+)?)\s*f\s*$")
_RANGE_ID = re.compile(r"^([A-Za-z]+)(\d+)-\1?(\d+)$")


# ---------------------------------------------------------------------------
# Where the film is
# ---------------------------------------------------------------------------
def find_timeline(where=None) -> str:
    """Absolute path of a film's timeline.json.

    where: a timeline file, a film directory (holding timeline.json), or None: $FILM_DIR if it is set, else the
    current directory and its parents (so a score run from inside its film directory finds it)."""
    if where is None:
        where = os.environ.get(ENV_FILM) or None
    if where is None:
        d = os.path.abspath(os.getcwd())
        while True:
            p = os.path.join(d, "timeline.json")
            if os.path.isfile(p):
                return p
            up = os.path.dirname(d)
            if up == d:
                break
            d = up
        raise FileNotFoundError(f"no timeline.json in {os.getcwd()} or its parents; pass a film directory or set "
                                f"${ENV_FILM}")
    where = os.fspath(where)
    p = os.path.join(where, "timeline.json") if os.path.isdir(where) else where
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{p}: no such timeline")
    return os.path.abspath(p)


def film_dir(where=None) -> str:
    """The film directory (the one holding timeline.json)."""
    return os.path.dirname(find_timeline(where))


def music_out(where=None) -> str:
    """Where a film's score goes: <film>/out/music (the picture engine reads score.wav and events.json there)."""
    return os.path.join(film_dir(where), "out", "music")


def load_timeline(where=None) -> dict:
    """The timeline as a dict. `where` as in find_timeline(); a dict is returned unchanged."""
    if isinstance(where, dict):
        return where
    with open(find_timeline(where), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
def check_grid(timeline: dict):
    """The timeline's grid must agree with itself: framesPerBeat = fps * 60 / bpm (the picture counts frames, the
    music counts beats; if these disagree, nothing can be in sync). Raises ValueError."""
    fps, bpm = float(timeline["fps"]), float(timeline["bpm"])
    fpb = timeline.get("framesPerBeat")
    if fpb is not None and abs(float(fpb) - fps * 60.0 / bpm) > 1e-9:
        raise ValueError(f"timeline: framesPerBeat {fpb} != fps * 60 / bpm = {fps * 60.0 / bpm:.6g}")
    bpb, fpbar = timeline.get("beatsPerBar", 4), timeline.get("framesPerBar")
    if fpb is not None and fpbar is not None and abs(float(fpbar) - float(fpb) * bpb) > 1e-9:
        raise ValueError(f"timeline: framesPerBar {fpbar} != framesPerBeat * beatsPerBar = {float(fpb) * bpb:g}")


def timeline_session(timeline=None, **kw):
    """A Session on the timeline's grid (bpm, beatsPerBar, bars, fps); other Session arguments pass through.
    `timeline`: a dict, a film directory, a timeline file or None (find_timeline). The dict is kept as
    session.timeline (mark_sync() and the self-check tools use it)."""
    from .session import Session

    tl = load_timeline(timeline)
    check_grid(tl)
    s = Session(bpm=tl["bpm"], beats_per_bar=tl["beatsPerBar"], bars=tl["bars"], fps=tl["fps"], **kw)
    s.timeline = tl
    return s


def tl_pos(pos, timeline: dict):
    """A timeline position as a Session position: 'bar.beat+Nf' (N frames later) -> 'bar.beat+<N/framesPerBeat>';
    everything else passes through unchanged."""
    if isinstance(pos, str):
        m = _FRAMES.match(pos)
        if m:
            frac = float(m.group(2)) / float(timeline["framesPerBeat"])
            return f"{m.group(1)}+{frac:.10g}"
    return pos


def tl_beat(pos, timeline: dict) -> float:
    """0-based beat index of a timeline position ('5.1', '4.3+0.5', '5.1+3f')."""
    return parse_pos(tl_pos(pos, timeline), int(timeline.get("beatsPerBar", 4)))


def tl_seconds(pos, timeline: dict) -> float:
    """Seconds of a timeline position on the timeline's (fixed-tempo) grid: beat * 60 / bpm."""
    return tl_beat(pos, timeline) * 60.0 / float(timeline["bpm"])


def tl_frame(pos, timeline: dict) -> float:
    """Fractional frame of a timeline position: beat * framesPerBeat (the picture's own formula)."""
    fpb = timeline.get("framesPerBeat") or float(timeline["fps"]) * 60.0 / float(timeline["bpm"])
    return tl_beat(pos, timeline) * float(fpb)


def film_end(timeline: dict) -> float:
    """Seconds where the picture ends (`filmEnd`, e.g. a cut to black before a music tail); the last bar's end if the
    timeline does not say."""
    if timeline.get("filmEnd"):
        return tl_seconds(timeline["filmEnd"], timeline)
    return int(timeline["bars"]) * int(timeline.get("beatsPerBar", 4)) * 60.0 / float(timeline["bpm"])


def timeline_curve(session, timeline: dict, key: str, interp: str = "linear"):
    """Curve from timeline[key]['keys'] ([[pos, value], ...]); key: energy, loudness, or any curve of that shape.

    Positions past the session's last bar are clamped by the curve itself (it holds the last value)."""
    keys = timeline[key]["keys"]
    return session.curve([(tl_pos(k[0], timeline), k[1]) for k in keys], interp)


def loudness_target(timeline: dict) -> dict:
    """The timeline's loudness targets: {integrated (LUFS), true_peak (dBTP), keys ([[pos, LUFS], ...] or [])}.
    Reads `integratedLUFS` / `truePeakDbtp` (or the older `integrated` / `truePeakMax`); -16 LUFS and -1 dBTP when
    the timeline is silent about them."""
    ld = timeline.get("loudness") or {}
    integ = ld.get("integratedLUFS", ld.get("integrated", -16.0))
    tp = ld.get("truePeakDbtp", ld.get("truePeakMax", -1.0))
    return {"integrated": float(integ), "true_peak": float(tp), "keys": list(ld.get("keys", []))}


# ---------------------------------------------------------------------------
# Sync points
# ---------------------------------------------------------------------------
def _expand(p: dict, bpb: int):
    """A range point {"id": "T1-T6", "at": "1.3-2.4"} -> one (id, pos, None) per beat when the counts match, else
    (range id, the range's first beat, its last beat)."""
    at, pid = p["at"], p["id"]
    if not (isinstance(at, str) and "-" in at and "+" not in at):
        return [(pid, at, None)]
    a, b = at.split("-", 1)
    m = _RANGE_ID.match(pid)

    def beat(x):
        bar, bt = x.split(".") if "." in x else (x, "1")
        return (int(bar) - 1) * bpb + int(bt) - 1

    b0, b1 = beat(a), beat(b)
    if m and int(m.group(3)) - int(m.group(2)) == b1 - b0:
        pre, i0 = m.group(1), int(m.group(2))
        return [(f"{pre}{i0 + k}", f"{(b0 + k) // bpb + 1}.{(b0 + k) % bpb + 1}", None) for k in range(b1 - b0 + 1)]
    return [(pid, a, b)]


def sync_points(timeline: dict, hard: bool | None = None) -> list:
    """The timeline's sync points as dicts {id, at, hard, picture, music[, to]}, in timeline order (ranges expanded;
    an unexpanded range carries its last beat as `to`). hard=True / False keeps only the hard / soft ones."""
    out = []
    bpb = int(timeline.get("beatsPerBar", 4))
    for p in timeline.get("sync", []):
        if hard is not None and bool(p.get("hard", False)) != hard:
            continue
        for pid, at, to in _expand(p, bpb):
            d = {"id": pid, "at": tl_pos(at, timeline), "hard": bool(p.get("hard", False)),
                 "picture": p.get("picture", ""), "music": p.get("music", "")}
            if to is not None:
                d["to"] = tl_pos(to, timeline)
            out.append(d)
    return out


def hard_sync(timeline=None) -> list:
    """[(id, pos), ...] of the hard sync points."""
    return [(p["id"], p["at"]) for p in sync_points(load_timeline(timeline), hard=True)]


def mark_sync(session, timeline: dict | None = None, hard_only: bool = False):
    """Add the timeline's sync points as markers (they land in events.json): kind 'hard_sync' or 'soft_sync'.
    timeline: defaults to session.timeline (set by timeline_session / Session.from_timeline)."""
    tl = timeline if timeline is not None else getattr(session, "timeline", None)
    if tl is None:
        raise ValueError("mark_sync: no timeline (pass one, or build the session with timeline_session())")
    for p in sync_points(tl, hard=True if hard_only else None):
        extra = {"to": p["to"]} if "to" in p else {}
        session.marker(p["at"], p["id"], kind="hard_sync" if p["hard"] else "soft_sync", picture=p["picture"], **extra)


def mark_hard_sync(session, timeline: dict | None = None):
    """Add only the hard sync points as markers."""
    mark_sync(session, timeline, hard_only=True)
