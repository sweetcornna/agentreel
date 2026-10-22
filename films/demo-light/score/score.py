"""《光与弦》配乐 (LIGHT AND STRING, the demo-light film): pulses in the dark -> the light gathers into a word (H1)
-> through the word (H2) -> the threads converge into one point (H3) -> a dry cut on 9.1, the room rings on to 10.1.

Every time and sync point comes from the film's timeline.json (90 BPM, 16 frames a beat, 9 bars = 576 frames =
24 s, bar 9 is the music's tail under black). Key: D major; the motif F#5 E5 D5 A4 is light falling. Design, the
events for the picture and the measured results: DESIGN.md (same directory).

    # from the repository root, with filmsynth installed (music/filmsynth/README.md)
    python films/demo-light/score/score.py              # render out/music/{score.wav, stems/, events.json}, self-check
    python films/demo-light/score/score.py --no-check   # render only
    python films/demo-light/score/score.py --workers 1  # one process: the same result, sample for sample

Layers (stem): ambience (room tone), pulses (muted plucks the picture follows), harp (the falling chord on H1 and
the run into H2), felt (felt piano: chords and the motif), bells (glass chimes), strings, choir, taut (the string of
light drawn taut into H3), sub, swell (reverse swell into H1), air (the rush through the word), impact (the low hit
on H2), sparkle (the word turning into light), and the three reverb returns (reverb_room, reverb_hall,
reverb_long), which alone ring on after the 9.1 cut.
"""
from __future__ import annotations

import argparse
import os
import sys

from filmsynth import (EQ, AirRush, Bell, Boom, Choir, Drone, FeltPiano, GainCurve, HeartDrum, Pluck, Reverb,
                       ReverseSwell, Sparkle, Strings, TautString, Wind, pos_string, to_midi)
from filmsynth.film import loudness_target, load_timeline, mark_sync, music_out, timeline_curve, timeline_session

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.dirname(HERE)                  # films/demo-light (holds timeline.json)

TAIL_S = 0.0                   # the timeline's 9 bars already hold the tail (9.1-10.1): the file is exactly 24 s
SEED = 1

# The reverse swell's loudest 30 ms (K-weighted) falls before its written end: the reversed reverb ends on its own
# pre-delay and the last 8 ms are a fade. Measured with the parameters below (plate, rt60 2.6; selfcheck.py
# re-measures it in the stem): peak = written end + SWELL_PEAK_REL_END frames.
SWELL_PEAK_REL_END = -2.16
SWELL_LEAD_FRAMES = 2.0        # the swell crests 2 frames before 3.1 (the light has gathered; then the word)
AIR_PEAK_LATE_FRAMES = 2.0     # the rush crests just after the lens is through the word (5.1)

# Track faders (dB). Every instrument is calibrated to about -24 LUFS (0.4 s) for one note at vel 0.7.
G = {
    "room": -4.0, "pulse": 2.0, "pulse_dark": -7.0, "harp": -10.0, "harp_run": 12.0, "felt_chord": -12.0, "felt_motif": 0.0,
    "felt_low": -13.0, "bell": -5.0, "str_dark": -5.0, "str_word": -10.0, "str_bed": -9.0, "sub": -7.0,
    "swell": -10.0, "air": -11.0, "boom": -8.0, "drum": -10.0, "sparkle": 2.0, "taut": -14.0, "choir": -16.0,
}

# Harmony: (position, name, pitch classes, sync id). A chord event for the picture on every change; H1 lands on the
# first major chord. selfcheck.py checks every note of the harmonic layers against this chart.
CHORDS = [("1.3", "B m", "B D F#", None), ("3.1", "D add9", "D F# A E", "H1"), ("4.1", "G add9", "G B D A", None),
          ("4.4", "A sus4", "A D E", None), ("5.1", "E m9", "E G B D F#", "H2"), ("6.1", "A sus4", "A D E", None),
          ("7.1", "D add9", "D F# A E", "H3")]

# Strings, 3.1-5.1 (one phrase, legato, voices move by step): D add9 | G add9 | A sus4, released on 5.1
WORD = {"cb": [["D2"], ["G1"], ["A1"]],
        "vc": [["A2", "F#3"], ["G2", "D3"], ["A2", "E3"]],
        "va": [["A3", "E4"], ["B3", "D4"], ["A3", "D4"]],
        "vn2": [["F#4", "A4"], ["G4", "A4"], ["E4", "A4"]],
        "vn1": [["E5"], ["D5"], ["D5"]]}
# Strings over the fingerboard, 5.1+0.5-9.1 (one phrase): E m9 | A sus4 | D add9, wide (the basses join on 7.1)
BED = {"vc": [["E3"], ["A2"], ["A2", "F#3"]],
       "va": [["B3", "D4"], ["A3", "D4"], ["A3", "E4"]],
       "vn2": [["F#4", "G4"], ["E4", "A4"], ["F#4", "A4"]],
       "vn1": [["B4"], ["D5"], ["E5", "F#5"]]}
PLAYERS = {"vn1": 10, "vn2": 8, "va": 8, "vc": 6, "cb": 4}

MOTIF = ["F#5", "E5", "D5", "A4"]                 # light falling

# Bus input rides (dB). The long space is for H1, H2 and the last chord's bloom; its input closes in bar 8, so that
# what rings on after the 9.1 cut is the hall alone and is gone before the file ends (10.1).
LONG_KEYS = [("1.1", 0.0), ("7.4", 0.0), ("8.3", -24.0)]
# The hall's input is ridden down into the cut, so that the tail left after 9.1 starts well under the music (the cut
# is heard as a cut) and rings out by itself.
HALL_KEYS = [("1.1", 0.0), ("8.2", 0.0), ("9.1", -8.0)]
# The master: the last section sits well under the middle (energy 0.35 -> 0.1), and goes down with the picture's fade.
MASTER_KEYS = [("1.1", 0.0), ("6.2", 0.0), ("7.1", -4.0), ("8.2", -5.0), ("9.1", -12.0)]


def build(tl: dict | None = None):
    tl = load_timeline(tl if tl is not None else FILM)
    s = timeline_session(tl, tail=TAIL_S, seed=SEED)
    beat = s.beat
    bpb = s.bpb
    frame = 1.0 / tl["framesPerBeat"]                                          # one frame in beats
    energy = timeline_curve(s, tl, "energy")
    s.export_curve("energy", energy)

    def at(b):
        """0-based beat index -> (bar, beat) for track.note()."""
        bar = int(b // bpb) + 1
        return bar, b - (bar - 1) * bpb + 1

    # ---- spaces ---------------------------------------------------------------------------------------------
    s.bus("room", Reverb("room", rt60=0.45, seed=3, lowcut=150.0), stem="reverb_room")
    hall_in = s.curve(HALL_KEYS, "smooth")
    s.bus("hall", GainCurve(hall_in), Reverb("hall", rt60=2.8, predelay=0.02, seed=5), stem="reverb_hall")
    long_in = s.curve(LONG_KEYS, "smooth")
    s.bus("long", GainCurve(long_in), Reverb("cathedral", rt60=4.5, seed=7), stem="reverb_long")

    # ---- 1.1-9.1 room tone: almost nothing, a little air above 60 Hz ------------------------------------------
    room = s.track("room", Wind(intensity=0.06, gustiness=0.15, whistle=0.0, brightness=0.12), gain_db=G["room"],
                   inserts=[EQ([("highpass", 60, 0.7, 0.0), ("lowpass", 3500, 0.7, 0.0)])], stem="ambience")
    room.at("1.1+0.1", beat("9.1") - 0.1, None, 0.5, fade_in=1.6, fade_out=0.05)

    # ---- the pulses the picture follows: muted plucks, left low / right high ----------------------------------
    # a 4 ms rolled onset keeps them soft and their peaks round; one fixed wooden body per side keeps them even
    mute = dict(let_ring=False, damp_time=0.12, body=0.55, decay=0.6, polar=0.2, spread=0.0, attack=0.004)
    pul = {"L": s.track("pulse_L", Pluck(wood=0.75, thump=0.3, pluck_pos=0.2, body_seed=5, **mute),
                        gain_db=G["pulse"], pan=-0.35, sends={"hall": -12, "room": -20}, stem="pulses"),
           "R": s.track("pulse_R", Pluck(wood=0.7, thump=0.25, pluck_pos=0.22, body_seed=2, **mute),
                        gain_db=G["pulse"], pan=0.35, sends={"hall": -12, "room": -20}, stem="pulses")}

    def pulse(b, side, pitch, vel, bright, div, shot, gain_db=0.0, **data):
        dur = {16: 0.25, 8: 0.45, 4: 0.9, 2: 1.8}[div]
        nt = pul[side].note(*at(b), dur, pitch, round(vel, 3), brightness=round(bright, 3), gain_db=gain_db)
        s.event("pulse", nt, shot=shot, side=side, div=div, vel=round(vel, 3), bright=round(bright, 3), pitch=pitch,
                **data)

    # P1 1.3-2.4: eighths, rocking low-high; the light index is the pitch's rank (the picture folds it onto its lights)
    p1 = [("L", "B3"), ("R", "F#4"), ("L", "D4"), ("R", "A4"), ("L", "B3"), ("R", "F#4"),
          ("L", "E4"), ("R", "A4"), ("L", "B3"), ("R", "F#4"), ("L", "D4"), ("R", "B4")]
    rank = {p: k for k, p in enumerate(sorted({p for _, p in p1}, key=to_midi))}
    b0 = beat("1.3")
    for i, (side, p) in enumerate(p1):
        b = b0 + 0.5 * i
        x = i / (len(p1) - 1)
        accent = 0.04 if (b % 2) == 0 else 0.0                                # beats 1 and 3 a touch firmer
        pulse(b, side, p, 0.34 + 0.18 * x + accent, 0.2 + 0.3 * x, 8, "S01", gain_db=G["pulse_dark"], light=rank[p])

    # ---- 2.1-3.1: muted strings over the fingerboard, B minor, one breath in --------------------------------
    for sec, p in (("vc", "B2"), ("va", "F#3"), ("vn2", "D4")):
        tr = s.track(f"str_dark_{sec}", Strings(section=sec, players=PLAYERS[sec] - 2, sordino=True, tasto=0.7,
                                                attack=1.2, release=0.6, brightness=0.3),
                     gain_db=G["str_dark"], sends={"hall": -12, "long": -18}, stem="strings")
        tr.at("2.1", 4, p, [(0.0, 0.05), (0.5, 0.2), (1.0, 0.44)])

    # ---- 2.4 -> H1 3.1: reverse swell cresting 2 frames early (the lights gather to the centre) -------------
    sw = s.track("swell", ReverseSwell(chord=(0, 7, 12, 16, 19, 24), space="plate", rt60=2.6, brightness=0.45),
                 gain_db=G["swell"], stem="swell")
    sw_end = beat("3.1") + (-SWELL_LEAD_FRAMES - SWELL_PEAK_REL_END) * frame
    sw.note(*at(sw_end - 2.0), 2.0, "D3", 0.7)
    s.marker(pos_string(beat("3.1") - SWELL_LEAD_FRAMES * frame), "swell_peak", kind="lead",
             frames=-SWELL_LEAD_FRAMES, of="H1")

    # ---- H1 3.1: a bright chord falls (harp, from the top, the first note on the frame), felt piano, a bell ---
    harp = s.track("harp", Pluck(brightness=0.62, decay=1.1, body=0.4, spread=0.5, thump=0.2),
                   gain_db=G["harp"], sends={"hall": -10, "long": -15}, stem="harp")
    for k, (p, v) in enumerate((("A5", 0.6), ("F#5", 0.56), ("E5", 0.54), ("D5", 0.52), ("A4", 0.5), ("D4", 0.48))):
        harp.note(*at(beat("3.1") + k / 32), 4 - k / 32, p, v, let_ring=False, damp_time=0.5)   # damped on 4.1
    fc = s.track("felt_chord", FeltPiano(felt=0.7, decay=1.2, body=0.5, noise=0.4), gain_db=G["felt_chord"],
                 sends={"hall": -9, "long": -13}, stem="felt")
    for p, v in (("D3", 0.5), ("A3", 0.46), ("E4", 0.44), ("F#4", 0.46)):
        fc.at("3.1", 4, p, v, let_ring=False, damp_time=0.6)                              # the pedal lifts on 4.1
    bell = s.track("bell", Bell(preset="glass", decay=1.3, brightness=0.45), gain_db=G["bell"],
                   sends={"hall": -6, "long": -8}, stem="bells")
    for pos, p, v, line in (("3.1", "D6", 0.5, 1), ("3.2", "E6", 0.48, 2)):
        nt = bell.at(pos, 3, p, v)
        s.event("chime", nt, line=line, pitch=p, **({"id": "H1"} if line == 1 else {}))

    # ---- 3.1-5.1: the strings open (no mute), D add9 -> G add9 -> A sus4, a crescendo released on 5.1 --------
    word_vel = [[(0.0, 0.46), (0.4, 0.4), (1.0, 0.34)],                     # 3.1-4.1: bloom, then make room
                [(0.0, 0.34), (0.6, 0.36), (1.0, 0.48)],                    # 4.1-4.4
                [(0.0, 0.5), (1.0, 0.82)]]                                  # 4.4-5.1: to the top
    spans = [("3.1", 4), ("4.1", 3), ("4.4", 1)]
    for sec in ("vn1", "vn2", "va", "vc", "cb"):
        tr = s.track(f"str_word_{sec}", Strings(section=sec, players=PLAYERS[sec], tasto=0.2, attack=0.35,
                                                release=0.1, brightness=0.45),
                     gain_db=G["str_word"] + (-3.0 if sec == "cb" else 0.0), sends={"hall": -10, "long": -15},
                     stem="strings")
        for (pos, n), chord, vel in zip(spans, WORD[sec], word_vel):
            for v, p in enumerate(chord):
                tr.at(pos, n, p, vel, voice=v)

    # ---- the sub: follows the roots from H1, cross-fades between them, ducks under the 5.1 hit ---------------
    sub = s.track("sub", Drone(chord=(0,), harmonics=2, brightness=0.1, beat_hz=0.15, attack=0.15, release=0.6,
                               motion=0.3),
                  gain_db=G["sub"], inserts=[EQ([("lowpass", 110, 0.7, 0.0), ("lowpass", 110, 0.7, 0.0)])],
                  stem="sub")
    for pos, n, p in (("3.1", 4, "D1"), ("4.1", 3, "G1"), ("4.4", 1, "A1"), ("5.1", 4, "E1"), ("6.1", 4, "A1"),
                      ("7.1", 8, "D1")):
        sub.at(pos, n, p, 0.62, **({} if pos == "3.1" else {"attack": 0.4}))
    sub.duck("boom", depth_db=6.0, release_ms=400.0)

    # ---- 3.3-4.3: the motif on the felt piano ------------------------------------------------------------------
    mo = s.track("felt_motif", FeltPiano(felt=0.8, decay=0.9, noise=0.5, let_ring=False, damp_time=0.45),
                 gain_db=G["felt_motif"], sends={"hall": -10, "long": -15}, stem="felt")
    for k, (pos, n, v) in enumerate((("3.3", 1.8, 0.4), ("3.4", 1.0, 0.36), ("4.1", 2.8, 0.42), ("4.3", 1.7, 0.36))):
        nt = mo.at(pos, n, MOTIF[k], v, center=0.15)
        s.event("motif", nt, n=k, pitch=MOTIF[k], vel=v)

    # ---- 4.4 -> H2 5.1: the harp runs up in sixteenths; the air rushes past with the lens -------------------
    for k, (p, v) in enumerate((("A4", 0.36), ("D5", 0.42), ("E5", 0.48), ("A5", 0.55))):
        b = beat("4.4") + 0.25 * k
        nt = harp.note(*at(b), 0.5, p, v, let_ring=False, damp_time=0.35, brightness=0.58 + 0.06 * k,
                       gain_db=G["harp_run"])                                # over the strings at their top
        s.event("run", nt, shot="S02", n=k, div=16, vel=v, pitch=p)
    air_peak = beat("5.1") + AIR_PEAK_LATE_FRAMES * frame
    air_start, air_dur = beat("4.4"), 1.5
    ar = s.track("air", AirRush(f_start=220, f_end=4200, peak=(air_peak - air_start) / air_dur, pan_from=-0.35,
                                pan_to=0.35, doppler=0.35, sub=0.0, tail=0.9, whistle=0.25),   # air, hardly a whistle
                 gain_db=G["air"], sends={"long": -10},
                 inserts=[EQ([("highpass", 80, 0.7, 0.0), ("highpass", 80, 0.7, 0.0)])], stem="air")
    ar.note(*at(air_start), air_dur, None, 0.7)
    s.marker(pos_string(air_peak), "air_crest", kind="lead", frames=AIR_PEAK_LATE_FRAMES, of="H2")

    # ---- H2 5.1: the strings let go; a soft low hit on the frame; the word turns into light -----------------
    s.event("release", "5.1", id="H2", stem="strings")
    bm = s.track("boom", Boom(sweep=12, sweep_time=0.05, decay=2.4, click=0.12, harmonics=0.45, attack=0.006),
                 gain_db=G["boom"], sends={"hall": -20}, stem="impact")
    nt = bm.at("5.1", 4, "E1", 0.72)
    s.event("impact", nt, id="H2")
    dr = s.track("drum", HeartDrum(dry=0.15, felt=0.85, body=0.55, position=0.15), gain_db=G["drum"],
                 sends={"hall": -14}, stem="impact")
    dr.at("5.1", 1, "E2", 0.6)
    sp = s.track("sparkle", Sparkle(density=40.0, span=24.0, width=0.9), gain_db=G["sparkle"],
                 sends={"hall": -8, "long": -12}, stem="sparkle")
    nt = sp.at("5.1", 2, "D5", 0.55)
    s.event("sparkle", nt, beats=2)

    # ---- S03 5.1-7.1: the pulses run down both sides of the corridor (sixteenths, then eighths) ---------------
    def ev_vel(b):
        e = float(energy(s.tempo.time_of_beat(b)))
        return 0.32 + 0.36 * e, 0.15 + 0.5 * e

    bar5 = ["E4", "B4", "G4", "D5", "E4", "B4", "G4", "F#5", "E4", "B4", "G4", "D5", "F#4", "B4", "G4", "E5"]
    for k in range(1, 16):                                  # from 5.1+0.25: the downbeat belongs to the hit
        b = beat("5.1") + 0.25 * k
        v, br = ev_vel(b)
        pulse(b, "L" if k % 2 == 0 else "R", bar5[k], v * 0.94, br, 16, "S03")
    bar6 = ["A3", "E4", "D4", "A4", "A3", "E4", "D4", "D5"]
    for k, p in enumerate(bar6):
        b = beat("6.1") + 0.5 * k
        v, br = ev_vel(b)
        pulse(b, "L" if k % 2 == 0 else "R", p, v * 0.88, br, 8, "S03")

    # ---- 5.1+0.5-9.1: a soft bed over the fingerboard, E m9 -> A sus4 -> D add9 (wide) -------------------------
    bed_vel = [[(0.0, 0.18), (0.4, 0.4), (1.0, 0.36)],                       # 5.1+0.5-6.1, entering after the hit
               [(0.0, 0.36), (1.0, 0.3)],                                   # 6.1-7.1, thinning
               [(0.0, 0.32), (0.25, 0.34), (0.6, 0.27), (1.0, 0.2)]]        # 7.1-9.1, the last chord
    bed_spans = [("5.1+0.5", 3.5), ("6.1", 4), ("7.1", 8)]
    for sec in ("vn1", "vn2", "va", "vc"):
        tr = s.track(f"str_bed_{sec}", Strings(section=sec, players=PLAYERS[sec] - 2, tasto=0.6, attack=1.0,
                                               release=0.5, brightness=0.35),
                     gain_db=G["str_bed"], sends={"hall": -10, "long": -14}, stem="strings")
        for (pos, n), chord, vel in zip(bed_spans, BED[sec], bed_vel):
            for v, p in enumerate(chord):
                tr.at(pos, n, p, vel, voice=v)
    cb = s.track("str_bed_cb", Strings(section="cb", players=4, tasto=0.4, attack=0.8, release=0.5, brightness=0.3),
                 gain_db=G["str_bed"] - 2.0, sends={"hall": -12, "long": -16}, stem="strings")
    cb.at("7.1", 8, "D2", bed_vel[2])

    # ---- 6.1 -> H3 7.1: a string of light drawn taut; a distant choir breathes in; it lands on 7.1 ------------
    ts = s.track("taut", TautString(slack_cents=45.0, tighten=0.92, beat_to=2.5, bright_to=0.6, attack=1.0),
                 gain_db=G["taut"], sends={"hall": -9, "long": -10}, stem="taut")
    nt = ts.at("6.1", 4, "A3", (0.46, 0.5))          # it tightens and brightens; it does not get louder
    s.event("taut", nt, to="7.1", pitch="A3")
    choir = {sec: s.track(f"choir_{sec}", Choir(section=sec, singers=8, distance=0.65, vowel="oo", attack=1.4,
                                                release=0.8),
                          gain_db=G["choir"], sends={"hall": -6, "long": -8}, stem="choir")
             for sec in ("soprano", "alto", "tenor", "bass")}
    for sec, p in (("soprano", "E4"), ("alto", "D4"), ("tenor", "A3")):              # A sus4, from nothing
        choir[sec].at("6.1", 4, p, [(0.0, 0.04), (1.0, 0.3)], vowel="oo")
    for sec, p in (("soprano", "F#4"), ("alto", "D4"), ("tenor", "A3"), ("bass", "D3")):  # E -> F#: it resolves
        choir[sec].at("7.1", 8, p, [(0.0, 0.32), (0.3, 0.34), (1.0, 0.2)], vowel=("ah", "oo"))

    # ---- H3 7.1: the wide chord; the light lands as one point; the rhythm thins ------------------------------
    fl = s.track("felt_low", FeltPiano(felt=0.75, decay=1.4, body=0.55, noise=0.35), gain_db=G["felt_low"],
                 sends={"hall": -9, "long": -10}, stem="felt")
    for p, v in (("D2", 0.44), ("A2", 0.4), ("F#3", 0.38)):
        fl.at("7.1", 8, p, v)
    nt = bell.at("7.1", 3, "D6", 0.42)
    s.event("chime", nt, line=0, pitch="D6", id="H3")
    s.event("land", "7.1", id="H3")
    for pos, side, p, div in (("7.1", "L", "D4", 4), ("7.2", "R", "A4", 4), ("7.3", "L", "F#4", 4),
                              ("7.4", "R", "A4", 4), ("8.1", "L", "D4", 2), ("8.3", "R", "A4", 2)):
        b = beat(pos)
        v, br = ev_vel(b)
        pulse(b, side, p, v * 0.9, br, div, "S04")
    for k, (pos, n, v) in enumerate((("7.3", 1.8, 0.38), ("7.4", 1.8, 0.35), ("8.1", 2.6, 0.37), ("8.3", 2.0, 0.32))):
        nt = mo.at(pos, n, MOTIF[k], v, center=0.15, let_ring=k == 3)
        s.event("motif", nt, n=k, pitch=MOTIF[k], vel=v)

    # ---- the end: down with the picture from 8.3; CUT 9.1, the dry music stops on the frame, the room rings on
    s.master_gain(MASTER_KEYS, "smooth")
    s.cut("9.1", "10.1", tails=True)
    s.event("cut", "9.1", id="CUT", tails=True)

    for pos, name, _, sid in CHORDS:
        s.event("chord", pos, chord=name, **({"id": sid} if sid else {}))
    mark_sync(s, tl)
    return s


def render(out=None, workers=None, verbose=True, tl=None):
    tl = load_timeline(tl if tl is not None else FILM)
    s = build(tl)
    tg = loudness_target(tl)
    res = s.render(out or music_out(FILM), mix_name="score.wav", workers=workers, target_lufs=tg["integrated"],
                   ceiling_dbtp=tg["true_peak"], verbose=verbose)
    return s, res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="output directory (default: <film>/out/music)")
    ap.add_argument("--workers", type=int, default=None, help="render processes (default: CPUs - 1)")
    ap.add_argument("--no-check", action="store_true", help="render only, skip the self-check")
    a = ap.parse_args(argv)
    out = a.out or music_out(FILM)
    s, res = render(out, a.workers)
    print(f"normalise {res.stats['normalise_db']:+.2f} dB, render {res.stats['render_s']:.1f} s -> {out}", flush=True)
    if not a.no_check:
        sys.path.insert(0, HERE)
        import selfcheck

        rep = selfcheck.run(out, res=res)
        return 0 if rep["summary"]["ok"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
