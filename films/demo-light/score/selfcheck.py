"""Objective self-check of the demo-light score (《光与弦》). Nobody here can listen, so every claim in DESIGN.md is
measured: the generic checks of filmsynth.check (integrated loudness and true peak, the same measured again by
ffmpeg's ebur128, clipping, clicks, events.json on the grid, the dry cut, harmony, energy follow, stem balance), plus
what is particular to this film:

  * P1: twelve pulses on the eighth-note grid (<= 1 ms in events.json, sides alternating), each one's attack in the
    pulses stem at its event time (+-10 ms); every other pulse (S03, S04) and the S02 run on their own grid too
  * H1 3.1: harp, felt piano and bell onsets on the frame (+-10 ms); the reverse swell's crest 1-3 frames before
  * H2 5.1: the strings let go (their stem falls >= 10 dB within 0.2 s); the low hit's onset 0..+1 frame
  * H3 7.1: the taut string has reached its pitch (A3, within 10 cents) just before 7.1; the pulses thin out
  * CUT 9.1: dry stems silent from the frame on, reverb returns still ringing, the tail below -60 LUFS before 10.1;
    the file is exactly 576 frames (24.000 s)
  * figures: out/music/report/loudness.png, spectrogram.png, sync_zoom.png; numbers: selfcheck.json, CHECK.md

    python films/demo-light/score/selfcheck.py                  # check the files already in out/music
    python films/demo-light/score/selfcheck.py --determinism    # render twice (1 process / many) and compare
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np

from filmsynth import check as C
from filmsynth.core import SR, to_midi, midi_to_hz
from filmsynth.film import load_timeline, music_out, tl_seconds

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import score  # noqa: E402  (the chord chart; render() for --determinism)

HARMONIC = ["strings", "sub", "choir", "felt"]        # the stems that carry the harmony

# where each sync point is measured (filmsynth.check.sync_check spec)
SYNC = {
    "P1": dict(stem="pulses", mode="onset", before=0.03, after=0.08, tol_ms=(-10, 10), what="first pulse (pulses stem)"),
    "H1": [dict(stem="harp", mode="onset", tol_ms=(-10, 10), what="harp, falling chord (harp stem)"),
           dict(stem="felt", mode="onset", tol_ms=(-10, 10), what="felt piano chord (felt stem)"),
           dict(stem="bells", mode="onset", tol_ms=(-10, 10), what="glass bell (bells stem)"),
           dict(stem="swell", mode="peak", what="reverse swell crest (K-weighted 30 ms)")],
    "H2": [dict(stem="impact", mode="onset", band=(None, 200), before=0.05, after=0.3, frames=(-0.1, 1.0),
                what="low hit onset (impact stem, < 200 Hz)"),
           dict(stem="air", mode="peak", what="air rush crest (K-weighted 30 ms)")],
    "H3": [dict(stem="bells", mode="onset", tol_ms=(-10, 10), what="glass bell (bells stem)"),
           dict(stem="felt", mode="onset", before=0.03, tol_ms=(-10, 10), what="low felt chord (felt stem)")],
    "CUT": dict(stem=None, mode="drop", kweight=True, win_ms=10, span=0.2, frames=(-1.0, 1.0),
                what="the mix drops (largest 50 ms drop)"),
}


# the generic pass of filmsynth.check.run() gets the onset-type points only (run() replaces the table afterwards)
SYNC_GENERIC = {k: ([x for x in v if x.get("mode") != "peak"] if isinstance(v, list) else v) for k, v in SYNC.items()}


def peak_row(r, tl, sid, pos, stem, what, lo_frames=None, hi_frames=None, search=(-0.6, 0.2)):
    """Where a lead device crests (K-weighted 30 ms envelope of its own stem) relative to a sync point."""
    t0 = tl_seconds(pos, tl)
    e = C.env_db(r.stems[stem], win_ms=30, kweight=True)
    a, b = int((t0 + search[0]) * 1000), int((t0 + search[1]) * 1000)
    k = int(np.argmax(e[a:b])) + a
    fr = (k / 1000 - t0) * tl["fps"]
    ok = None if lo_frames is None else lo_frames <= fr <= hi_frames
    return dict(id=sid, pos=pos, what=what, mode="peak", target_s=round(t0, 4), measured_s=round(k / 1000, 4),
                dev_ms=round(k - t0 * 1000, 1), frames=round(fr, 2), ok=ok, hard=True)


def sync_table(r, tl):
    rows = C.sync_check(r, tl, SYNC_GENERIC)
    rows.append(peak_row(r, tl, "H1", "3.1", "swell", SYNC["H1"][3]["what"], -3.0, -1.0))
    rows.append(peak_row(r, tl, "H2", "5.1", "air", SYNC["H2"][1]["what"], 0.0, 4.0, search=(-0.6, 0.6)))
    return sorted(rows, key=lambda x: (x["target_s"], x["id"]))


def pulses_check(r, tl):
    """The pulses the picture follows: on their grid in events.json, sides alternating within each shot, and audible
    at their event time in the pulses stem (first ms within 20 dB of the local peak, +-10 ms)."""
    ev = r.events
    out = {}
    # attacks: the pluck's onset transient above 2 kHz, 1 ms envelope (a 16th still rings when the next one comes)
    e = C.env_db(r.stems["pulses"], band=(2000, None), win_ms=1.0)
    harp = C.env_db(r.stems["harp"], band=(2000, None), win_ms=1.0)
    for shot in ("S01", "S02", "S03", "S04"):
        ps = C.events_of(ev, "run" if shot == "S02" else "pulse", shot=shot)
        if not ps:
            continue
        divs = sorted({p["div"] for p in ps})
        grid = C.on_grid([p["t"] for p in ps], tl, division=max(divs) / 4.0)
        sides = [p.get("side", "C") for p in ps]
        alt = all(a != b for a, b in zip(sides, sides[1:])) if shot != "S02" else True
        devs = []
        for p in ps:      # the attack: the largest 3 ms rise of that envelope within +-15 ms
            on, _ = C.largest_step(harp if shot == "S02" else e, p["t"], span=0.015, w=0.003)
            devs.append(round((on - p["t"]) * 1000, 1))
        out[shot] = dict(n=len(ps), divs=divs, grid=grid, sides_alternate=alt, audio_onset_dev_ms_max=max(map(abs, devs)),
                         audio_onset_dev_ms=devs, ok=bool(grid["ok"] and alt and max(map(abs, devs)) <= 10.0))
    p1 = C.events_of(ev, "pulse", shot="S01")
    s1, s2 = tl_seconds("1.3", tl), tl_seconds("2.4+0.5", tl)
    out["P1"] = dict(n=len(p1), from_s=p1[0]["t"] if p1 else None, to_s=p1[-1]["t"] if p1 else None,
                     lights=sorted({p["light"] for p in p1}),
                     ok=bool(len(p1) == 12 and abs(p1[0]["t"] - s1) < 1e-3 and abs(p1[-1]["t"] - s2) < 1e-3))
    # the density arc: eighths in S01, sixteenths then eighths in S03, quarters then halves in S04
    out["density_arc"] = [(shot, sorted({p["div"] for p in C.events_of(ev, "pulse", shot=shot)}, reverse=True))
                          for shot in ("S01", "S02", "S03", "S04")]
    out["ok"] = all(v["ok"] for k, v in out.items() if isinstance(v, dict) and "ok" in v)
    return out


def strings_release(r, tl):
    """H2: the crescendo is at its top just before 5.1 and the strings let go on it: their stem (K-weighted 10 ms
    envelope) falls 10 dB under its level of the 100 ms before 5.1 within 3 frames, and 20 dB within 6."""
    t = tl_seconds("5.1", tl)
    x = r.stems["strings"]
    pre = C.window_loudness(x, t - 0.25, t)
    post = C.window_loudness(x, t + 0.1, t + 0.3)
    build = C.window_loudness(x, t - 1.25, t - 0.75)
    e = C.env_db(x, win_ms=10, kweight=True)
    i0 = int(round(t * 1000))
    ref = float(np.mean(e[i0 - 100:i0]))
    fall = {}
    for d in (10, 20):
        k = np.nonzero(e[i0:i0 + 600] < ref - d)[0]
        fall[d] = round(float(k[0]) / 1000 * tl["fps"], 2) if len(k) else None
    ok = pre - post >= 10.0 and pre > build and fall[10] is not None and fall[10] <= 3.0 and fall[20] is not None \
        and fall[20] <= 6.0
    return dict(before_0_25s=round(pre, 1), after_0_1_0_3s=round(post, 1), drop_db=round(pre - post, 1),
                crescendo_last_beat_db=round(pre - build, 1), frames_to_minus_10db=fall[10],
                frames_to_minus_20db=fall[20], ok=bool(ok))


def taut_pitch(r, tl, pitch="A3"):
    """H3: the taut string's pitch in the 0.4 s before 7.1 (FFT peak near the written pitch, parabolic), in cents."""
    t = tl_seconds("7.1", tl)
    m = np.mean(r.stems["taut"][:, int((t - 0.45) * SR):int((t - 0.05) * SR)], axis=0)
    n = 1 << 18
    X = np.abs(np.fft.rfft(m * np.hanning(len(m)), n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f0 = float(midi_to_hz(to_midi(pitch)))
    sel = np.nonzero((f > f0 * 0.94) & (f < f0 * 1.06))[0]
    k = sel[int(np.argmax(X[sel]))]
    a, b, c = np.log(X[k - 1] + 1e-20), np.log(X[k] + 1e-20), np.log(X[k + 1] + 1e-20)
    fk = f[k] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0])
    cents = 1200 * np.log2(fk / f0)
    # and at the start of the note, where it is slack
    t0 = tl_seconds("6.1", tl)
    m0 = np.mean(r.stems["taut"][:, int((t0 + 0.3) * SR):int((t0 + 0.9) * SR)], axis=0)
    X0 = np.abs(np.fft.rfft(m0 * np.hanning(len(m0)), n))
    sel0 = np.nonzero((f > f0 * 0.9) & (f < f0 * 1.04))[0]
    c0 = 1200 * np.log2(f[sel0[int(np.argmax(X0[sel0]))]] / f0)
    return dict(pitch=pitch, cents_before_7_1=round(float(cents), 1), cents_at_6_1=round(float(c0), 1),
                ok=bool(abs(cents) <= 10.0 and c0 < -15.0))


def sync_zoom(r, tl, out):
    """report/sync_zoom.png: H1, H2 and CUT, +-12 frames, the stems that make them, frame grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fps = tl["fps"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    plan = [("H1 3.1", "3.1", [("harp", "harp", {}), ("felt", "felt", {}), ("bells", "bells", {}),
                               ("swell", "swell", dict(win_ms=30, kweight=True))]),
            ("H2 5.1", "5.1", [("strings", "strings", dict(win_ms=10, kweight=True)),
                               ("impact < 200 Hz", "impact", dict(band=(None, 200), win_ms=5)),
                               ("air", "air", dict(win_ms=30, kweight=True))]),
            ("CUT 9.1", "9.1", [("dry (all but reverbs)", "dry", dict(win_ms=10)),
                                ("reverbs", "wet", dict(win_ms=10))])]
    dry = sum(v for k, v in r.stems.items() if not C.is_wet(k))
    wet = sum(v for k, v in r.stems.items() if C.is_wet(k))
    for ax, (title, pos, layers) in zip(axes, plan):
        t0 = tl_seconds(pos, tl)
        for label, st, kw in layers + [("mix", None, dict(win_ms=10, kweight=True))]:
            x = r.mix if st is None else dry if st == "dry" else wet if st == "wet" else r.stems[st]
            e = C.env_db(x, **kw)
            i0, i1 = int((t0 - 0.55) * 1000), int((t0 + 0.55) * 1000)
            ax.plot((np.arange(i0, i1) / 1000 - t0) * fps, e[i0:i1], lw=1.0, label=label)
        for fr in range(-12, 13):
            ax.axvline(fr, color="#ccc" if fr else "#c0392b", lw=0.5 if fr else 1.0, zorder=0)
        ax.set_xlim(-12, 12)
        ax.set_ylim(-110, -10)
        ax.set_title(title)
        ax.set_xlabel("frames (24 fps)")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("dB")
    fig.tight_layout()
    p = os.path.join(out, "report", "sync_zoom.png")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def run(out=None, res=None, use_ffmpeg=True):
    """All checks on the files in `out` (default <film>/out/music). `res` (a RenderResult) only adds render times."""
    tl = load_timeline(FILM)
    out = out or music_out(FILM)
    rep, r, _ = C.run(FILM, out, use_ffmpeg=use_ffmpeg, write=False, sync_spec=SYNC_GENERIC,
                      harmony_stems=HARMONIC, title="demo-light score")
    rep["sync"] = sync_table(r, tl)
    rep["pulses"] = pulses_check(r, tl)
    rep["h2_release"] = strings_release(r, tl)
    rep["h3_taut"] = taut_pitch(r, tl)
    cut = rep["dry_cuts"][0] if rep["dry_cuts"] else C.dry_cut_check(r.mix, r.stems, tl_seconds("9.1", tl))
    rep["cut_9_1"] = dict(cut, tail_gone_before_10_1=bool(cut["tail_to_minus60_lufs_s"] is not None
                                                           and cut["tail_to_minus60_lufs_s"] < cut["tail_left_in_file_s"]))
    rep["chord_fit"] = C.chord_fit(r.events, [(p, name, pcs) for p, name, pcs, _ in score.CHORDS], stems=HARMONIC)
    rep["chord_events"] = [(e["pos"], e["chord"]) for e in C.events_of(r.events, "chord")]
    rep["audibility"] = C.audibility(r, {"pulse": "pulses", "run": "harp", "chime": "bells", "motif": "felt",
                                         "impact": "impact"})
    aud = [a for a in rep["audibility"] if a["type"] == "pulse"]
    rep["pulse_audibility"] = dict(min_over_rest_300_3k_db=min(a["over_rest_300_3k_db"] for a in aud),
                                   median_over_rest_300_3k_db=float(np.median([a["over_rest_300_3k_db"] for a in aud])))
    rep["figures"].append(sync_zoom(r, tl, out))
    if res is not None:
        rep["render_stats"] = res.stats
    sm = rep["summary"]
    sm.update(sync_ok=all(x["ok"] for x in rep["sync"] if x["ok"] is not None), pulses_ok=rep["pulses"]["ok"],
              h2_release_ok=rep["h2_release"]["ok"], h3_taut_ok=rep["h3_taut"]["ok"],
              cut_tail_ok=rep["cut_9_1"]["tail_gone_before_10_1"] and rep["cut_9_1"]["ok"],
              chord_fit_ok=rep["chord_fit"]["ok"],
              energy_ok=bool(rep["energy"] and rep["energy"]["r"] is not None and rep["energy"]["r"] >= 0.8),
              length_ok=abs(r.duration * tl["fps"] - tl["totalFrames"]) < 1e-6, clicks_ok=sm["clicks"] == 0)
    verdicts = ("lufs_ok", "true_peak_ok", "clipping_ok", "events_ok", "sync_ok", "pulses_ok", "h2_release_ok",
                "h3_taut_ok", "cut_tail_ok", "chord_fit_ok", "energy_ok", "length_ok", "clicks_ok")
    sm["failed"] = [k for k in verdicts if not sm[k]] + (["ffmpeg_ok"] if sm.get("ffmpeg_ok") is False else [])
    sm["ok"] = not sm["failed"]
    r.events["measured"] = dict(note="written by films/demo-light/score/selfcheck.py from the rendered audio",
                                sync=rep["sync"], cut=dict(t=cut["t"], tail_to_minus60_lufs_s=cut["tail_to_minus60_lufs_s"]),
                                lufs_i=rep["levels"]["lufs_i"], true_peak_dbtp=rep["levels"]["true_peak_dbtp"])
    with open(os.path.join(out, "events.json"), "w", encoding="utf-8") as f:
        json.dump(r.events, f, ensure_ascii=False, indent=1, default=C.plain)
    extra = ["## Film-specific", ""]
    extra += C.md_json("Pulses (the picture follows them)", rep["pulses"])
    extra += C.md_json("H2: the strings let go", rep["h2_release"])
    extra += C.md_json("H3: the taut string's pitch", rep["h3_taut"])
    extra += C.md_json("CUT 9.1", rep["cut_9_1"])
    extra += C.md_json("Every harmonic note against the chord chart", dict(rep["chord_fit"], events=rep["chord_events"]))
    extra += ["## Audibility of the events the picture follows (250 ms after onset, own stem vs the rest)", "",
              C.md_table(rep["audibility"], [("type", "type"), ("pos", "pos"), ("stem", "stem"), ("own LUFS", "own_lufs"),
                                             ("over rest dB", "over_rest_db"), ("over rest 300-3k dB", "over_rest_300_3k_db"),
                                             ("over dry rest dB", "over_dry_rest_db")]), ""]
    C.write_report(out, rep, tl, title="demo-light score", extra_md=extra)
    print(json.dumps(sm, ensure_ascii=False, indent=1, default=C.plain))
    print(f"-> {os.path.join(out, 'CHECK.md')}")
    return rep


def determinism(workers_a=1, workers_b=None):
    """Render the score twice, with different numbers of processes, into temporary directories; compare them
    sample for sample (filmsynth.check.compare)."""
    with tempfile.TemporaryDirectory() as tmp:
        a, b = os.path.join(tmp, "a"), os.path.join(tmp, "b")
        score.render(a, workers=workers_a, verbose=False)
        score.render(b, workers=workers_b, verbose=False)
        same, rows = C.compare(a, b)
        ref = music_out(FILM)
        if os.path.exists(os.path.join(ref, "score.wav")):
            same_ref, rows_ref = C.compare(a, ref)
        else:
            same_ref, rows_ref = None, []
    for x in rows:
        print(json.dumps(x, ensure_ascii=False))
    print(f"workers={workers_a} vs workers={workers_b or 'default'}: " + ("identical" if same else "DIFFERENT"))
    if same_ref is not None:
        print(f"workers={workers_a} vs {ref}: " + ("identical" if same_ref else "DIFFERENT")
              + f" ({sum(x['same'] for x in rows_ref)}/{len(rows_ref)} files)")
    return same and same_ref is not False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="render directory (default <film>/out/music)")
    ap.add_argument("--no-ffmpeg", action="store_true")
    ap.add_argument("--determinism", action="store_true", help="render twice (1 process and the default) and compare")
    a = ap.parse_args(argv)
    if a.determinism:
        return 0 if determinism() else 1
    rep = run(a.out, use_ffmpeg=not a.no_ffmpeg)
    return 0 if rep["summary"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
