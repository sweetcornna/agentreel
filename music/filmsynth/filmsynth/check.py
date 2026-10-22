"""Objective self-check of a rendered score, for any film. Nobody inside a render pipeline can listen, so every
claim about the music is measured. Everything here works from files: the film's timeline.json and what
Session.render() wrote (score.wav, stems/*.wav, events.json). A film's own check adds only what is particular to it.

    from filmsynth import check as C
    from filmsynth.film import load_timeline
    tl = load_timeline("films/demo-light")
    r = C.load("films/demo-light/out/music")          # Render(dir, mix, stems, events)
    C.levels(r.mix, r.stems, C.onsets(r.events))      # LUFS, true peak, sample peak, clipping, clicks, limiter, DC, L/R
    C.loudness(r.mix, tl)                             # integrated loudness and true peak against the timeline's targets
    C.ffmpeg_loudness(r.mix_path)                     # the same two numbers measured independently by ffmpeg (ebur128)
    C.events_check(r.events, tl, r.duration)          # beats / bars on the grid (<= 1 ms), frames, the events' schema
    C.sync_check(r, tl, spec)                         # where the music lands at each sync point (ms and frames)
    C.dry_cut_check(r.mix, r.stems, t)                # after a dry cut: dry stems silent, reverb returns ring, the tail
    C.harmony(r.events, stems=[...])                  # pitch classes on every beat (from the notes), changes, spacing
    C.chord_fit(r.events, chart, stems=[...])         # every note against a chord chart: the notes that do not fit
    C.energy_follow(r.mix, tl)                        # 1 s loudness against the timeline's energy curve
    C.stem_levels(r.stems, tl); C.audibility(r, pick) # balance per bar; events the picture follows over the rest
    C.figures(r, tl, out)                             # report/loudness.png, report/spectrogram.png
    C.compare(dir_a, dir_b)                           # determinism: two renders, sample for sample

    python -m filmsynth.check run --film films/demo-light     # the generic checks on <film>/out/music (exit 1 = fail)
    python -m filmsynth.check compare DIR_A DIR_B              # exit 0 = sample-for-sample identical

Conventions: bus (reverb) stems are recognised by name, `fx_<bus>` (Session's default) or `reverb_<...>`; pass
`wet=` to say otherwise. Times are seconds from 1.1; the grid is the timeline's (fixed tempo: beat * 60 / bpm).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from . import analysis as A
from .core import SR, to_midi
from .dsp.dynamics import lufs, peak_db, true_peak_db
from .session import parse_pos
from .film import film_end, find_timeline, load_timeline, loudness_target, music_out, sync_points, tl_seconds

WET_PREFIXES = ("fx_", "reverb_")
BANDS = {"sub": (20, 60), "low": (60, 250), "mid": (250, 4000), "high": (4000, 16000)}
_PCN = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ---------------------------------------------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Render:
    dir: str
    mix: np.ndarray            # (2, n) float64
    stems: dict                # name -> (2, n)
    events: dict               # events.json
    mix_path: str

    @property
    def duration(self) -> float:
        return self.mix.shape[1] / SR


def load(out_dir, mix_name: str = "score.wav", stems: bool = True) -> Render:
    """Read what Session.render() wrote into out_dir: the mix, stems/*.wav (if any) and events.json."""
    import soundfile as sf

    mp = os.path.join(out_dir, mix_name)
    x, sr = sf.read(mp, always_2d=True)
    if sr != SR:
        raise ValueError(f"{mp}: {sr} Hz, expected {SR}")
    st = {}
    sd = os.path.join(out_dir, "stems")
    if stems and os.path.isdir(sd):
        for f in sorted(os.listdir(sd)):
            if f.endswith(".wav"):
                st[f[:-4]] = sf.read(os.path.join(sd, f), always_2d=True)[0].T
    with open(os.path.join(out_dir, "events.json"), encoding="utf-8") as f:
        ev = json.load(f)
    return Render(out_dir, x.T, st, ev, mp)


def onsets(ev: dict) -> list:
    """Every note start and cut start in events.json (the click detector does not flag what starts there)."""
    return sorted({n["start"] for n in ev.get("notes", [])} | {c["start"] for c in ev.get("cuts", [])})


def is_wet(name: str, wet=None) -> bool:
    return name in wet if wet is not None else name.startswith(WET_PREFIXES)


# ---------------------------------------------------------------------------------------------------------------
# Envelopes and windows
# ---------------------------------------------------------------------------------------------------------------
def mono(x):
    return np.mean(x, axis=0)


def env_db(x, band=None, win_ms=2.0, kweight=False):
    """Zero-phase band-limited RMS envelope in dB, 1 ms hop (index = ms). band: (lo, hi) Hz, either may be None."""
    m = A.k_weight(x).sum(axis=0) if kweight else mono(x)
    if band is not None:
        lo, hi = band
        if lo and hi:
            sos = signal.butter(4, [lo, hi], "bandpass", fs=SR, output="sos")
        elif lo:
            sos = signal.butter(4, lo, "highpass", fs=SR, output="sos")
        else:
            sos = signal.butter(4, hi, "lowpass", fs=SR, output="sos")
        m = signal.sosfiltfilt(sos, m)
    p = np.maximum(uniform_filter1d(m ** 2, max(1, int(win_ms * 1e-3 * SR))), 0.0)   # the running sum can dip below 0
    return 10 * np.log10(p[:: SR // 1000] + 1e-20)


def onset(e, t0, before=0.08, after=0.15, rel_db=-20.0):
    """First ms in [t0-before, t0+after] where the envelope reaches (window peak + rel_db); also the peak time and
    level. For a layer that is silent before the event (its own stem)."""
    a, b = max(0, int((t0 - before) * 1000)), min(len(e), int((t0 + after) * 1000))
    w = e[a:b]
    k = int(np.argmax(w))
    on = int(np.nonzero(w >= w[k] + rel_db)[0][0])
    return (a + on) / 1000.0, (a + k) / 1000.0, float(w[k])


def largest_step(e, t0, span=0.3, w=0.05, rise=True):
    """Time (s) of the largest change of the mean level over `w` s (after minus before) within t0 +- span: the
    largest rise (rise=True) or drop. For a layer that keeps sounding through the event (the mix). Returns
    (time, step dB)."""
    i0, ww = int(round(t0 * 1000)), int(w * 1000)
    seg = np.arange(max(ww, i0 - int(span * 1000)), min(len(e) - ww, i0 + int(span * 1000)))
    c = np.concatenate([[0.0], np.cumsum(e)])
    d = (c[seg + ww] - c[seg]) / ww - (c[seg] - c[seg - ww]) / ww
    k = int(np.argmax(d if rise else -d))
    return seg[k] / 1000.0, float(d[k])


def band_steps(x, t0, span=0.3, w=0.05):
    """Per band (BANDS: sub / low / mid / high), the largest change of the 50 ms mean level, either direction, within
    t0 +- span: {band: (time s, step dB)}, and the principal band (the largest |step|). How the music changes across
    a sync point when nothing says which layer makes it."""
    out = {}
    for name, (lo, hi) in BANDS.items():
        e = uniform_filter1d(env_db(x, band=(lo, hi), win_ms=5.0), 10)
        tr, sr = largest_step(e, t0, span, w, rise=True)
        td, sd = largest_step(e, t0, span, w, rise=False)
        out[name] = (float(tr), sr) if abs(sr) >= abs(sd) else (float(td), sd)
    principal = max(out, key=lambda k: abs(out[k][1]))
    return out, principal


def window_loudness(x, t0, t1):
    """K-weighted mean-square loudness (LUFS-like, ungated) of x between t0 and t1 (s)."""
    seg = x[:, max(0, int(t0 * SR)):int(t1 * SR)]
    if seg.shape[1] < 16:
        return -200.0
    k = A.k_weight(seg)
    return float(-0.691 + 10 * np.log10(np.mean(np.sum(k ** 2, axis=0)) + 1e-20))


# ---------------------------------------------------------------------------------------------------------------
# Levels and loudness
# ---------------------------------------------------------------------------------------------------------------
def levels(mix, stems=None, onset_times=None):
    """Integrated loudness, true peak, sample peak, clipped samples, suspected clicks (outside onsets), DC, L/R
    correlation; with stems, the limiter's gain reduction (the stems sum to the mix before the limiter)."""
    clip = int(np.sum(np.abs(mix) >= (2 ** 23 - 1) / 2 ** 23))
    clicks = A.detect_clicks(mix, onset_times)
    r = dict(lufs_i=round(lufs(mix), 2), true_peak_dbtp=round(true_peak_db(mix), 2),
             sample_peak_dbfs=round(peak_db(mix), 2), clipped_samples=clip, clicks=len(clicks),
             click_times=clicks[:12], dc_db=round(A.dc_db(mix), 1), corr_lr=round(float(A.correlation(mix)), 3))
    if stems:
        pre = sum(stems.values())
        blk = int(0.005 * SR)
        nb = min(mix.shape[1], pre.shape[1]) // blk
        rm = np.sqrt(np.mean(mix[:, :nb * blk].reshape(2, nb, blk) ** 2, axis=(0, 2)) + 1e-20)
        rp = np.sqrt(np.mean(pre[:, :nb * blk].reshape(2, nb, blk) ** 2, axis=(0, 2)) + 1e-20)
        act = rp > 10 ** (-60 / 20)
        if np.any(act):
            gr = 20 * np.log10(rm[act] / rp[act])
            i = int(np.argmin(gr))
            r.update(limiter_max_gr_db=round(float(gr[i]), 2),
                     limiter_max_gr_at_s=round(float(np.nonzero(act)[0][i] * 0.005), 3),
                     limiter_blocks_over_0_5db=int(np.sum(gr < -0.5)))
    return r


def loudness(mix, tl: dict, tol_lu: float = 0.5, tol_keys_db: float = 3.0):
    """Integrated loudness and true peak against the timeline's targets (film.loudness_target); 1 s loudness at
    every `loudness.keys` point if the timeline has them (keys on hard sync points judged on the 1 s after them,
    keys at or below -60 judged as 'at most target + tol'); 1 s loudness per section."""
    tg = loudness_target(tl)
    integ, tp = lufs(mix), true_peak_db(mix)
    out = dict(target_lufs=tg["integrated"], lufs_i=round(integ, 2), lufs_ok=bool(abs(integ - tg["integrated"]) <= tol_lu),
               ceiling_dbtp=tg["true_peak"], true_peak_dbtp=round(tp, 2), true_peak_ok=bool(tp <= tg["true_peak"]))
    end = mix.shape[1] / SR
    hard = {p["at"] for p in sync_points(tl, hard=True)}
    keys = []
    for pos, target in tg["keys"]:
        tk = tl_seconds(pos, tl)
        if tk > end - 0.5:
            row = dict(pos=pos, t=round(tk, 3), target=target, centred=round(window_loudness(mix, end - 1.0, end), 1),
                       note="last 1 s of the file")
        else:
            row = dict(pos=pos, t=round(tk, 3), target=target,
                       centred=round(window_loudness(mix, max(0.0, tk - 0.5), tk + 0.5), 1))
        judged = row["centred"]
        if pos in hard and tk <= end - 1.0:             # a window across a hard sync point belongs to neither side
            judged = window_loudness(mix, tk, tk + 1.0)
            row["after"] = round(judged, 1)
        row["judged"] = round(max(judged, -99.0), 1)
        row["ok"] = bool(abs(judged - target) <= tol_keys_db) if target > -60 else bool(judged <= target + tol_keys_db)
        keys.append(row)
    out["keys"] = keys
    if keys:
        out["keys_ok"] = all(r["ok"] for r in keys)
    t, L = A.loudness_curve(mix, 1.0, 0.1)
    secs = []
    for sec in tl.get("sections", []):
        a, b = tl_seconds(sec["from"], tl), min(tl_seconds(sec["to"], tl), end)
        m = (t - 0.5 >= a - 1e-6) & (t + 0.5 <= b + 1e-6)
        Lm = L[m]
        Lv = Lm[Lm > -70]
        seg = mix[:, int(a * SR):int(b * SR)]
        secs.append(dict(id=sec.get("id", ""), zh=sec.get("zh", ""), span=f"{sec['from']}-{sec['to']}",
                         mean_1s=round(float(np.mean(Lv)), 1) if len(Lv) else None,
                         min_1s=round(float(np.min(Lm)), 1) if len(Lm) else None,
                         max_1s=round(float(np.max(Lm)), 1) if len(Lm) else None,
                         integrated=round(lufs(seg), 1) if seg.shape[1] > SR // 2 else None))
    out["sections"] = secs
    valid = L > -70
    if np.any(valid):
        out["span_1s_db"] = round(float(L[valid].max() - L[valid].min()), 1)
        out["loudest_1s"] = round(float(L.max()), 1)
        out["loudest_at_s"] = round(float(t[int(np.argmax(L))]), 2)
    return out


def ffmpeg_path():
    """ffmpeg to use: $FFMPEG, else `ffmpeg` on PATH, else the usual install locations; None if there is none."""
    env = os.environ.get("FFMPEG")
    if env:
        return env
    p = shutil.which("ffmpeg")
    if p:
        return p
    for c in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if os.path.isfile(c):
            return c
    return None


def ffmpeg_loudness(path, ffmpeg=None):
    """Integrated loudness (LUFS), loudness range (LU) and true peak (dBTP) of a file as ffmpeg's ebur128 filter
    measures them (`ffmpeg -i path -af ebur128=peak=true -f null -`): an implementation independent of pyloudnorm and
    of filmsynth's own true-peak meter. None if ffmpeg is not found."""
    exe = ffmpeg or ffmpeg_path()
    if not exe:
        return None
    p = subprocess.run([exe, "-nostats", "-hide_banner", "-i", os.fspath(path), "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    txt = p.stderr
    summ = txt[txt.rfind("Summary:"):] if "Summary:" in txt else ""

    def grab(label):
        m = re.findall(rf"{label}:\s+(-?inf|-?\d+(?:\.\d+)?)", summ)
        return float(m[-1]) if m else None

    return dict(ffmpeg=exe, returncode=p.returncode, lufs_i=grab("I"), lra_lu=grab("LRA"), true_peak_dbtp=grab("Peak"))


# ---------------------------------------------------------------------------------------------------------------
# events.json
# ---------------------------------------------------------------------------------------------------------------
def events_check(ev: dict, tl: dict, duration: float | None = None, tol_ms: float = 1.0):
    """events.json against the timeline: the same fps / bpm / beats per bar; every beat and bar on the grid
    (|t - beat * 60 / bpm| <= tol_ms); frames = t * fps; the events' schema ({t, frame, type, pos}, sorted, inside
    the file); markers and notes inside the file; every sync point of the timeline present as a marker (when the
    score marks them); the render as long as the timeline's totalFrames (when it says)."""
    spb = 60.0 / float(tl["bpm"])
    fps = float(tl["fps"])
    bpb = int(tl.get("beatsPerBar", 4))
    dur = float(duration if duration is not None else ev.get("duration", 0.0))
    problems = []
    for k_ev, k_tl in (("fps", "fps"), ("bpm", "bpm"), ("beatsPerBar", "beatsPerBar")):
        if k_tl in tl and float(ev.get(k_ev, -1)) != float(tl[k_tl]):
            problems.append(f"{k_ev} {ev.get(k_ev)} != timeline {tl[k_tl]}")
    if ev.get("tempoRamps"):
        problems.append("tempo ramps present: the timeline grid is fixed-tempo")
    beats = [b if isinstance(b, (int, float)) else b.get("t") for b in ev.get("beats", [])]
    if not beats or not all(isinstance(b, (int, float)) for b in beats):
        problems.append("beats missing or not numbers")
        beat_dev = None
    else:
        beat_dev = float(np.max(np.abs(np.array(beats) - np.arange(len(beats)) * spb)) * 1000)
        if beat_dev > tol_ms:
            problems.append(f"beats off the grid by {beat_dev:.3f} ms")
        want = int(ev.get("bars", tl.get("bars", 0))) * bpb + 1
        if len(beats) != want:
            problems.append(f"{len(beats)} beats, expected bars * beatsPerBar + 1 = {want}")
    bar_dev = 0.0
    for x in ev.get("barTimes", []):
        bar_dev = max(bar_dev, abs(x["t"] - (x["bar"] - 1) * bpb * spb) * 1000)
        if abs(x["frame"] - x["t"] * fps) > 0.002:
            problems.append(f"barTimes bar {x['bar']}: frame {x['frame']} != t * fps")
    if bar_dev > tol_ms:
        problems.append(f"barTimes off the grid by {bar_dev:.3f} ms")
    for x in ev.get("beatTimes", []):
        if abs(x["t"] - ((x["bar"] - 1) * bpb + x["beat"] - 1) * spb) * 1000 > tol_ms:
            problems.append(f"beatTimes {x['bar']}.{x['beat']} off the grid")
            break
    evs = ev.get("events", [])
    kinds, pos_dev = {}, {}
    last = -1.0
    for e in evs:
        if not (isinstance(e.get("t"), (int, float)) and isinstance(e.get("type"), str)):
            problems.append(f"event without numeric t / string type: {e}")
            continue
        kinds[e["type"]] = kinds.get(e["type"], 0) + 1
        if e["t"] < last - 1e-9:
            problems.append(f"events not sorted at t={e['t']}")
        last = e["t"]
        if not 0.0 <= e["t"] <= dur + 1e-6:
            problems.append(f"event {e['type']} at {e['t']} s outside the file (0..{dur:.3f})")
        if "frame" in e and abs(e["frame"] - e["t"] * fps) > 0.002:
            problems.append(f"event {e['type']} at {e['t']}: frame {e['frame']} != t * fps")
        if isinstance(e.get("pos"), str):
            try:
                d = abs(e["t"] - tl_seconds(e["pos"], tl)) * 1000
                pos_dev[e["type"]] = max(pos_dev.get(e["type"], 0.0), d)
            except (ValueError, KeyError):
                problems.append(f"event {e['type']}: cannot parse pos {e['pos']!r}")
    mk_dev = 0.0
    for m in ev.get("markers", []):
        if not 0.0 <= m.get("t", -1) <= dur + 1e-6:
            problems.append(f"marker {m.get('name')} outside the file")
        if isinstance(m.get("pos"), str):
            mk_dev = max(mk_dev, abs(m["t"] - tl_seconds(m["pos"], tl)) * 1000)
    if mk_dev > tol_ms:
        problems.append(f"a marker's t is {mk_dev:.3f} ms off its pos")
    sync_marks = {m["name"] for m in ev.get("markers", []) if str(m.get("kind", "")).endswith("_sync")}
    missing = [p["id"] for p in sync_points(tl) if p["id"] not in sync_marks]
    if sync_marks and missing:
        problems.append(f"sync points without a marker: {missing}")
    for c in ev.get("cuts", []):
        if not (0 <= c["start"] < c["end"]):
            problems.append(f"cut {c} malformed")
    for n in ev.get("notes", []):
        if n["start"] < 0 or n["end"] < n["start"] or n["start"] > dur + 1e-6:
            problems.append(f"note {n.get('track')} {n.get('pos')} outside the file")
            break
    out = dict(beats=len(beats), beats_max_dev_ms=None if beat_dev is None else round(beat_dev, 4),
               bars_max_dev_ms=round(bar_dev, 4), markers_max_dev_ms=round(mk_dev, 4), counts=kinds,
               events_max_dev_from_pos_ms={k: round(v, 3) for k, v in sorted(pos_dev.items())},
               sync_markers=sorted(sync_marks), duration_s=round(dur, 6))
    if "totalFrames" in tl:
        out["frames"] = round(dur * fps, 3)
        out["timeline_totalFrames"] = tl["totalFrames"]
        if abs(dur * fps - float(tl["totalFrames"])) > 0.5:
            problems.append(f"render is {dur * fps:.2f} frames long, the timeline {tl['totalFrames']}")
    out["problems"] = problems
    out["ok"] = not problems
    return out


def events_of(ev: dict, type_: str, **match) -> list:
    """events.json events of one type (and matching fields), in time order."""
    return [e for e in ev.get("events", []) if e.get("type") == type_ and all(e.get(k) == v for k, v in match.items())]


def on_grid(times, tl: dict, division: float = 1.0, tol_ms: float = 1.0):
    """Are these times (s) on the grid subdivided `division` times a beat (2 = eighths)? Max deviation in ms."""
    step = 60.0 / float(tl["bpm"]) / division
    t = np.asarray(list(times), dtype=float)
    if not len(t):
        return dict(n=0, max_dev_ms=None, ok=True)
    d = float(np.max(np.abs(t - np.round(t / step) * step)) * 1000)
    return dict(n=int(len(t)), max_dev_ms=round(d, 4), ok=bool(d <= tol_ms))


# ---------------------------------------------------------------------------------------------------------------
# Sync, cuts
# ---------------------------------------------------------------------------------------------------------------
def _row(sid, pos, what, target, measured, ok, fps, **extra):
    return dict(id=sid, pos=pos, what=what, target_s=round(target, 4), measured_s=round(measured, 4),
                dev_ms=round((measured - target) * 1000, 1), frames=round((measured - target) * fps, 2),
                ok=None if ok is None else bool(ok), **extra)


def _layer(r: Render, stem):
    if stem is None:
        return r.mix
    names = [stem] if isinstance(stem, str) else list(stem)
    return sum(r.stems[k] for k in names)


def sync_check(r: Render, tl: dict, spec: dict | None = None):
    """Where the music lands at the timeline's sync points (ranges expanded).

    spec: {sync id: dict(...)} or {sync id: [dict, ...]} with
      stem      None (the mix), a stem name or a list of names (summed)
      mode      'onset' (first ms reaching peak + rel_db: for a layer that is silent before, its own stem),
                'rise' / 'drop' (largest 50 ms step within +-span: for a layer that keeps sounding),
                'bands' (band_steps(): the largest step in the band that changes most)
      band      (lo, hi) Hz or None; kweight (bool); win_ms; before / after (onset window, s); span (step window)
      at        override the position (e.g. '5.1+0.75f' for a landing written late)
      tol_ms    (lo, hi) accepted deviation in ms, or frames=(lo, hi)
      what      label
    Points not in spec: measured on the mix with mode 'bands' (at a cut: 'drop'); hard ones are accepted within
    +-1 frame, soft ones are reported without a verdict. That default only says that the music changes on the
    frame; a film's own check should name the layer that makes each sync point."""
    spec = spec or {}
    fps = float(tl["fps"])
    cut_starts = [c["start"] for c in r.events.get("cuts", [])]
    rows = []
    for p in sync_points(tl):
        tg = tl_seconds(p["at"], tl)
        if tg >= r.duration - 0.05:
            continue
        items = spec.get(p["id"])
        if items is None:
            at_cut = any(abs(c - tg) < 0.002 for c in cut_starts)
            items = [dict(mode="drop" if at_cut else "bands", kweight=True, win_ms=10,
                          frames=(-1.0, 1.0) if p["hard"] else None, what="mix (default)")]
        elif isinstance(items, dict):
            items = [items]
        for it in items:
            t0 = tl_seconds(it["at"], tl) if it.get("at") else tg
            x = _layer(r, it.get("stem"))
            mode = it.get("mode", "onset")
            extra = {}
            if mode == "bands":
                steps, pb = band_steps(x, t0, it.get("span", 0.3), it.get("w", 0.05))
                m = steps[pb][0]
                extra.update(band=pb, step_db=round(steps[pb][1], 1),
                             bands={k: [round((v[0] - t0) * 1000, 1), round(v[1], 1)] for k, v in steps.items()})
            else:
                e = env_db(x, band=it.get("band"), win_ms=it.get("win_ms", 2.0), kweight=it.get("kweight", False))
                if mode == "onset":
                    m, pk, lv = onset(e, t0, it.get("before", 0.08), it.get("after", 0.15), it.get("rel_db", -20.0))
                    extra.update(peak_ms_after=round((pk - t0) * 1000, 1), peak_db=round(lv, 1))
                else:
                    m, step = largest_step(e, t0, it.get("span", 0.3), it.get("w", 0.05), rise=mode == "rise")
                    extra.update(step_db=round(step, 1))
            dev = (m - t0) * 1000
            if it.get("tol_ms") is not None:
                lo, hi = it["tol_ms"]
                ok = lo <= dev <= hi
            elif it.get("frames") is not None:
                lo, hi = it["frames"]
                ok = lo <= dev / 1000 * fps <= hi
            else:
                ok = None
            rows.append(_row(p["id"], p["at"], it.get("what", f"{it.get('stem') or 'mix'} {mode}"), t0, m, ok, fps,
                             hard=p["hard"], mode=mode, **extra))
    return rows


def dry_cut_check(mix, stems: dict, t: float, wet=None):
    """After a dry cut at t (Session.cut(..., tails=True)): the dry stems are silent from t on (max |x| in dBFS), the
    reverb returns ring on (their loudness in the 0.3 s after), how far the mix drops across t and how long the tail
    takes to fall below -60 / -70 LUFS (0.4 s windows). ok: every dry stem below -120 dBFS after the cut."""
    i = int(round(t * SR))
    dry = {k: v for k, v in stems.items() if not is_wet(k, wet)}
    wets = {k: v for k, v in stems.items() if is_wet(k, wet)}
    dry_after = max((float(np.max(np.abs(v[:, i:]))) for v in dry.values() if v.shape[1] > i), default=0.0)
    pre = window_loudness(mix, t - 0.4, t)
    post = window_loudness(mix, t + 0.004, t + 0.404)
    tt, L = A.loudness_curve(mix[:, i:], 0.4, 0.01)

    def first_below(th):
        k = np.nonzero(L < th)[0]
        return round(float(tt[k[0]]), 2) if len(k) else None

    return dict(t=round(t, 4), dry_stems=sorted(dry), wet_stems=sorted(wets),
                dry_after_max_dbfs=round(20 * np.log10(dry_after + 1e-30), 1),
                dry_before_lufs_0_3s={k: round(window_loudness(v, t - 0.3, t), 1) for k, v in dry.items()},
                wet_after_lufs_0_3s={k: round(window_loudness(v, t, t + 0.3), 1) for k, v in wets.items()},
                mix_0_4s_before=round(pre, 1), mix_0_4s_after=round(post, 1), drop_db=round(pre - post, 1),
                tail_to_minus60_lufs_s=first_below(-60), tail_to_minus70_lufs_s=first_below(-70),
                tail_left_in_file_s=round(mix.shape[1] / SR - t, 3),
                ok=bool(dry and 20 * np.log10(dry_after + 1e-30) < -120))


# ---------------------------------------------------------------------------------------------------------------
# Harmony, energy, balance
# ---------------------------------------------------------------------------------------------------------------
def harmony(ev: dict, stems=None, tracks=None, min_gap_beats: float | None = None):
    """Pitch classes sounding on every beat (at the middle of the beat, from events.json notes between their
    written start and end), restricted to some stems / tracks (the harmonic layers). Returns the changes, the
    beats between real changes (entering from silence is not a change) and, with min_gap_beats, whether the
    harmony ever moves faster than that."""
    spb = 60.0 / float(ev["bpm"])
    nb = int(ev["bars"]) * int(ev["beatsPerBar"])
    bpb = int(ev["beatsPerBar"])
    notes = [n for n in ev.get("notes", []) if "midi" in n and (stems is None or n["stem"] in stems)
             and (tracks is None or n["track"] in tracks)]
    per_beat = []
    for b in range(nb):
        t = (b + 0.5) * spb
        pcs = {int(round(n["midi"])) % 12 for n in notes if n["start"] <= t < n["end"]}
        per_beat.append(tuple(_PCN[k] for k in sorted(pcs)))
    lab = lambda b: f"{b // bpb + 1}.{b % bpb + 1}"  # noqa: E731
    changes = [dict(pos=lab(b), pcs=list(per_beat[b])) for b in range(nb)
               if per_beat[b] and (b == 0 or per_beat[b] != per_beat[b - 1])]
    real = [b for b in range(1, nb) if per_beat[b] and per_beat[b - 1] and per_beat[b] != per_beat[b - 1]]
    gaps = [b2 - b1 for b1, b2 in zip(real, real[1:])]
    out = dict(changes=changes, moves=[lab(b) for b in real], min_gap_beats=min(gaps) if gaps else None)
    if min_gap_beats is not None:
        out["ok"] = all(g >= min_gap_beats for g in gaps)
    return out


def _pcs(chord) -> set:
    """Pitch classes from 'D F# A E', ['D', 'F#'], or ints."""
    items = chord.split() if isinstance(chord, str) else chord
    out = set()
    for p in items:
        out.add(int(p) % 12 if isinstance(p, (int, np.integer)) else int(round(to_midi(f"{p}4"))) % 12)
    return out


def chord_fit(ev: dict, chart, stems=None, tracks=None):
    """Every pitched note (of these stems / tracks) against a chord chart [(pos, name, pitch classes), ...] (each
    chord holds until the next): at the middle of every beat the note covers between its written start and end,
    is its pitch class in the chord then? Returns {misfits: [...], notes, ok}. Ringing past the written end is not
    counted: write the duration you mean to sound."""
    spb = 60.0 / float(ev["bpm"])
    bpb = int(ev["beatsPerBar"])
    rows = sorted(((parse_pos(p, bpb), name, _pcs(pcs)) for p, name, pcs in chart), key=lambda x: x[0])
    notes = [n for n in ev.get("notes", []) if "midi" in n and (stems is None or n["stem"] in stems)
             and (tracks is None or n["track"] in tracks)]
    bad = []
    for n in notes:
        pc = int(round(n["midi"])) % 12
        b = int(np.floor(n["start"] / spb + 1e-6))
        while (b + 0.5) * spb < n["end"]:
            if (b + 0.5) * spb >= n["start"]:
                cur = [r for r in rows if r[0] <= b + 0.5]
                if cur and pc not in cur[-1][2]:
                    bad.append(dict(track=n["track"], pitch=n["pitch"], pos=n["pos"],
                                    beat=f"{b // bpb + 1}.{b % bpb + 1}", chord=cur[-1][1]))
                    break
            b += 1
    return dict(notes=len(notes), misfits=bad, ok=not bad)


def energy_follow(mix, tl: dict, key: str = "energy", floor: float = 0.05):
    """The 1 s loudness (0.1 s hop) against the timeline's intent curve (0..1) where the curve is above `floor` and
    the picture runs: Pearson correlation r, Spearman rank correlation, the straight-line fit L = a + b * energy and
    its residual. A score that follows the picture's energy has r near 1 and a slope of a few tens of dB per unit."""
    if key not in tl:
        return None
    t, L = A.loudness_curve(mix, 1.0, 0.1)
    ks = tl[key]["keys"]
    kt = np.array([tl_seconds(k[0], tl) for k in ks])
    kv = np.array([float(k[1]) for k in ks])
    o = np.argsort(kt, kind="stable")
    en = np.interp(t, kt[o], kv[o])
    m = (en > floor) & (t <= film_end(tl)) & (L > -70)
    if np.sum(m) < 5:
        return dict(windows=int(np.sum(m)), r=None)
    x, y = en[m], L[m]
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    rk = lambda v: np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float)  # noqa: E731
    return dict(windows=int(np.sum(m)), r=round(float(np.corrcoef(x, y)[0, 1]), 3),
                spearman=round(float(np.corrcoef(rk(x), rk(y))[0, 1]), 3), fit_a_db=round(float(a), 1),
                fit_b_db_per_unit=round(float(b), 1), resid_rms_db=round(float(np.sqrt(np.mean(res ** 2))), 2),
                resid_max_db=round(float(np.max(np.abs(res))), 2),
                worst_at_s=round(float(t[m][int(np.argmax(np.abs(res)))]), 2))


def stem_levels(stems: dict, tl: dict):
    """Integrated loudness of every stem in every bar ('-' = silent or below -70)."""
    bpb, spb = int(tl.get("beatsPerBar", 4)), 60.0 / float(tl["bpm"])
    rows = []
    for k, x in sorted(stems.items()):
        r = dict(stem=k)
        for bar in range(1, int(tl["bars"]) + 1):
            a, b = (bar - 1) * bpb * spb, min(bar * bpb * spb, x.shape[1] / SR)
            seg = x[:, int(a * SR):int(b * SR)]
            v = lufs(seg) if seg.shape[1] > 0 and np.any(seg) else -200.0
            r[f"bar{bar}"] = round(v, 1) if v > -70 else "-"
        rows.append(r)
    return rows


def audibility(r: Render, pick: dict, window: float = 0.25, wet=None):
    """For the events the picture follows: the event's own stem against the rest of the mix in the `window` s after
    its onset, broadband (K-weighted), in 300-3000 Hz, and against the dry rest only. pick: {event type: stem}.
    Positive = the event stands above everything else there."""
    total = sum(r.stems.values())
    drysum = sum(v for k, v in r.stems.items() if not is_wet(k, wet))
    sos = signal.butter(4, [300, 3000], "bandpass", fs=SR, output="sos")
    rows = []
    for e in r.events.get("events", []):
        st = pick.get(e["type"])
        if st is None or st not in r.stems:
            continue
        a, b = int(e["t"] * SR), int((e["t"] + window) * SR)
        own = r.stems[st][:, a:b]
        rest = total[:, a:b] - own
        dry = drysum[:, a:b] - own
        ko = window_loudness(own, 0, window)
        kr = window_loudness(rest, 0, window) if np.any(rest) else -200.0
        kd = window_loudness(dry, 0, window) if np.any(dry) else -200.0
        bo = 10 * np.log10(np.mean(signal.sosfilt(sos, own) ** 2) + 1e-20)
        br = 10 * np.log10(np.mean(signal.sosfilt(sos, rest) ** 2) + 1e-20)
        rows.append(dict(type=e["type"], pos=e.get("pos", ""), stem=st, own_lufs=round(ko, 1),
                         over_rest_db=round(ko - kr, 1), over_rest_300_3k_db=round(bo - br, 1),
                         over_dry_rest_db=round(ko - kd, 1)))
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------------------------------------------
def _sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def compare(dir_a, dir_b, mix_name: str = "score.wav"):
    """Two renders sample for sample: the mix and every stem (as 24-bit integers; float stems as floats), their
    sha256, and events.json (ignoring `measured`, which the self-check adds). Returns (identical, rows)."""
    import soundfile as sf

    rows, same = [], True
    names = [mix_name]
    sa, sb = os.path.join(dir_a, "stems"), os.path.join(dir_b, "stems")
    la = sorted(f for f in os.listdir(sa) if f.endswith(".wav")) if os.path.isdir(sa) else []
    lb = sorted(f for f in os.listdir(sb) if f.endswith(".wav")) if os.path.isdir(sb) else []
    if la != lb:
        same = False
        rows.append(dict(file="stems/", note=f"different stem sets: {sorted(set(la) ^ set(lb))}", same=False))
    names += [os.path.join("stems", f) for f in la if f in lb]
    for nm in names:
        a, b = os.path.join(dir_a, nm), os.path.join(dir_b, nm)
        if not (os.path.exists(a) and os.path.exists(b)):
            rows.append(dict(file=nm, note="missing", same=False))
            same = False
            continue
        info = sf.info(a)
        dt = "float64" if info.subtype in ("FLOAT", "DOUBLE") else "int32"
        x = sf.read(a, always_2d=True, dtype=dt)[0]
        y = sf.read(b, always_2d=True, dtype=dt)[0]
        d = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()) if x.shape == y.shape else None
        ha, hb = _sha(a), _sha(b)
        ok = d == 0 and ha == hb
        same &= ok
        rows.append(dict(file=nm, samples=int(x.shape[0]), max_abs_diff=d, sha256=ha[:16],
                         sha256_same=ha == hb, same=ok))
    ea, eb = os.path.join(dir_a, "events.json"), os.path.join(dir_b, "events.json")
    if os.path.exists(ea) and os.path.exists(eb):
        ja, jb = (json.load(open(p, encoding="utf-8")) for p in (ea, eb))
        ja.pop("measured", None)
        jb.pop("measured", None)
        ok = ja == jb
        same &= ok
        rows.append(dict(file="events.json", note="compared without 'measured'", same=ok))
    return bool(same), rows


# ---------------------------------------------------------------------------------------------------------------
# Figures and the report
# ---------------------------------------------------------------------------------------------------------------
def _sync_lines(ax, tl, labels=True):
    for p in sync_points(tl):
        x = tl_seconds(p["at"], tl)
        ax.axvline(x, color="#c0392b" if p["hard"] else "#888", lw=0.8 if p["hard"] else 0.5,
                   ls="-" if p["hard"] else ":")
        if labels:
            ax.text(x, 1.0, p["id"], transform=ax.get_xaxis_transform(), fontsize=7, ha="center", va="bottom")


def figures(r: Render, tl: dict, out_dir, title: str = "") -> list:
    """report/loudness.png (1 s loudness, the timeline's loudness keys or its energy curve, sync points) and
    report/spectrogram.png (mix, 0.4 s loudness above it). Returns the paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = os.path.join(out_dir, "report")
    os.makedirs(d, exist_ok=True)
    mix = r.mix
    t, L = A.loudness_curve(mix, 1.0, 0.1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, L, lw=1.1, color="#222", label="1 s K-weighted loudness (mix)")
    keys = loudness_target(tl)["keys"]
    if keys:
        ts, vs = [tl_seconds(k[0], tl) for k in keys], np.array([k[1] for k in keys])
        ax.plot(ts, vs, "o-", color="#c0392b", lw=0.8, ms=3, label="timeline loudness.keys")
        ax.fill_between(ts, vs - 3, vs + 3, color="#e67e22", alpha=0.18)
    if "energy" in tl:
        ax2 = ax.twinx()
        ks = tl["energy"]["keys"]
        ax2.plot([tl_seconds(k[0], tl) for k in ks], [k[1] for k in ks], "o-", color="#3a7bd5", lw=0.8, ms=3,
                 label="timeline energy")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("energy", color="#3a7bd5")
        ax2.legend(loc="lower center", fontsize=8)
    _sync_lines(ax, tl)
    ax.set_ylim(-75, 0)
    ax.set_xlim(0, mix.shape[1] / SR)
    ax.set_xlabel("s")
    ax.set_ylabel("LUFS (1 s)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title or "loudness")
    fig.tight_layout()
    p1 = os.path.join(d, "loudness.png")
    fig.savefig(p1, dpi=100)
    plt.close(fig)
    m = mono(mix)
    f, tt, S = signal.spectrogram(m, SR, nperseg=4096, noverlap=3072, window="hann", mode="psd")
    S = 10 * np.log10(S + 1e-20)
    S -= S.max()
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [1, 4]}, sharex=True)
    t4, L4 = A.loudness_curve(mix, 0.4, 0.02)
    axes[0].plot(t4, L4, lw=0.8, color="#333")
    axes[0].set_ylabel("LUFS 0.4 s")
    axes[0].set_ylim(-80, 0)
    axes[0].grid(alpha=0.3)
    _sync_lines(axes[0], tl)
    sel = (f >= 25) & (f <= 16000)
    axes[1].pcolormesh(tt, f[sel], S[sel], shading="auto", vmin=-100, vmax=0, cmap="magma")
    axes[1].set_yscale("log")
    axes[1].set_ylim(25, 16000)
    axes[1].set_ylabel("Hz")
    axes[1].set_xlabel("s")
    _sync_lines(axes[1], tl, labels=False)
    fig.suptitle((title + ": " if title else "") + "spectrogram (mix); hard sync points orange, soft grey")
    fig.tight_layout()
    p2 = os.path.join(d, "spectrogram.png")
    fig.savefig(p2, dpi=100)
    plt.close(fig)
    return [p1, p2]


def plain(o):
    """json.dump default for numpy scalars."""
    return o.item() if hasattr(o, "item") else str(o)


def md_table(rows, cols):
    """rows: list of dicts; cols: [(header, key), ...]."""
    head = "| " + " | ".join(c for c, _ in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(r.get(k, "")) for _, k in cols) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def md_json(title, obj):
    return [f"## {title}", "", "```", json.dumps(obj, ensure_ascii=False, indent=1, default=plain), "```", ""]


def run(film=None, out=None, use_ffmpeg: bool = True, write: bool = True, sync_spec: dict | None = None,
        harmony_stems=None, title: str = "", figs: bool = True):
    """The generic checks on a film's rendered score (<film>/out/music unless `out`). Writes selfcheck.json and
    CHECK.md (write=True) and report/*.png (figs=True) into the output directory. Returns (report, render,
    timeline)."""
    tl = load_timeline(film)
    out = out or music_out(film if not isinstance(film, dict) else None)
    r = load(out)
    lv = levels(r.mix, r.stems, onsets(r.events))
    ld = loudness(r.mix, tl)
    ff = ffmpeg_loudness(r.mix_path) if use_ffmpeg else None
    evc = events_check(r.events, tl, r.duration)
    sync = sync_check(r, tl, sync_spec)
    cuts = [dry_cut_check(r.mix, r.stems, c["start"]) for c in r.events.get("cuts", []) if c.get("tails")]
    harm = harmony(r.events, stems=harmony_stems)
    en = energy_follow(r.mix, tl)
    bal = stem_levels(r.stems, tl) if r.stems else []
    fig_paths = figures(r, tl, out, title) if figs else []
    summary = dict(
        lufs_i=ld["lufs_i"], target_lufs=ld["target_lufs"], lufs_ok=ld["lufs_ok"],
        true_peak_dbtp=ld["true_peak_dbtp"], ceiling_dbtp=ld["ceiling_dbtp"], true_peak_ok=ld["true_peak_ok"],
        ffmpeg=None if ff is None else dict(lufs_i=ff["lufs_i"], true_peak_dbtp=ff["true_peak_dbtp"]),
        clipping_ok=lv["clipped_samples"] == 0, clicks=lv["clicks"], events_ok=evc["ok"],
        beats_max_dev_ms=evc["beats_max_dev_ms"],
        sync_ok=all(x["ok"] for x in sync if x["ok"] is not None),
        dry_cuts_ok=all(c["ok"] for c in cuts) if cuts else None,
        energy_r=None if en is None else en.get("r"))
    if ff is not None and ff["lufs_i"] is not None:
        summary["ffmpeg_ok"] = bool(abs(ff["lufs_i"] - ld["target_lufs"]) <= 0.5 and ff["true_peak_dbtp"] is not None
                                    and ff["true_peak_dbtp"] <= ld["ceiling_dbtp"])
    summary["ok"] = bool(summary["lufs_ok"] and summary["true_peak_ok"] and summary["clipping_ok"]
                         and summary["events_ok"] and summary["sync_ok"] and summary["dry_cuts_ok"] is not False
                         and summary.get("ffmpeg_ok") is not False)
    report = dict(summary=summary, levels=lv, loudness=ld, ffmpeg=ff, events=evc, sync=sync, dry_cuts=cuts,
                  harmony=harm, energy=en, stem_levels=bal, figures=fig_paths)
    if write:
        write_report(out, report, tl, title=title)
    return report, r, tl


def write_report(out, report: dict, tl: dict, title: str = "", extra_md=None):
    """selfcheck.json and CHECK.md from a report dict (the generic keys run() makes, plus anything a film adds:
    extra_md is a list of markdown lines appended at the end)."""
    with open(os.path.join(out, "selfcheck.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=plain)
    md = [f"# {title or 'Score'} self-check (objective; not listened to)", ""]
    md += md_json("Summary", report["summary"])
    md += md_json("Levels", report["levels"])
    ld = report["loudness"]
    md += ["## Loudness", "", f"integrated {ld['lufs_i']} LUFS (target {ld['target_lufs']}), true peak "
           f"{ld['true_peak_dbtp']} dBTP (ceiling {ld['ceiling_dbtp']})", ""]
    if report.get("ffmpeg"):
        ff = report["ffmpeg"]
        md += [f"ffmpeg ebur128: I {ff['lufs_i']} LUFS, LRA {ff['lra_lu']} LU, true peak {ff['true_peak_dbtp']} dBTP", ""]
    if ld["keys"]:
        md += [md_table(ld["keys"], [("pos", "pos"), ("t", "t"), ("target", "target"), ("centred", "centred"),
                                     ("after", "after"), ("judged", "judged"), ("ok", "ok")]), ""]
    md += [md_table(ld["sections"], [("section", "id"), ("span", "span"), ("mean 1 s", "mean_1s"), ("min", "min_1s"),
                                     ("max", "max_1s"), ("integrated", "integrated")]), ""]
    md += ["## Sync", "", md_table(report["sync"], [("id", "id"), ("pos", "pos"), ("what", "what"), ("mode", "mode"),
                                                    ("target s", "target_s"), ("measured s", "measured_s"),
                                                    ("dev ms", "dev_ms"), ("frames", "frames"), ("ok", "ok")]), ""]
    for c in report.get("dry_cuts") or []:
        md += md_json(f"Dry cut at {c['t']} s", c)
    md += md_json("Harmony", report["harmony"])
    md += md_json("Energy follow", report["energy"])
    if report.get("stem_levels"):
        md += ["## Stem balance (integrated LUFS per bar; '-' = silent or below -70)", "",
               md_table(report["stem_levels"], [("stem", "stem")] + [(f"bar {i}", f"bar{i}")
                                                                    for i in range(1, int(tl["bars"]) + 1)]), ""]
    md += md_json("events.json", report["events"])
    md += list(extra_md or [])
    with open(os.path.join(out, "CHECK.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return md


# ---------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m filmsynth.check", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_run = sub.add_parser("run", help="the generic checks on a film's rendered score")
    a_run.add_argument("--film", default=None, help="film directory or timeline.json (default: $FILM_DIR, else the "
                                                     "current directory and its parents)")
    a_run.add_argument("--out", default=None, help="render directory (default <film>/out/music)")
    a_run.add_argument("--no-ffmpeg", action="store_true", help="skip the ffmpeg ebur128 cross-check")
    a_run.add_argument("--no-write", action="store_true", help="print only; do not write selfcheck.json / CHECK.md")
    a_cmp = sub.add_parser("compare", help="two renders sample for sample (exit 0 = identical)")
    a_cmp.add_argument("a")
    a_cmp.add_argument("b")
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        same, rows = compare(a.a, a.b)
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))
        print("identical" if same else "DIFFERENT")
        return 0 if same else 1
    find_timeline(a.film)                     # fail early with a clear message
    report, _, _ = run(a.film, a.out, use_ffmpeg=not a.no_ffmpeg, write=not a.no_write)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=1, default=plain))
    return 0 if report["summary"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
