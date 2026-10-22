"""Isolated objective tests for every instrument (writes examples/out/calibrate.json; a few minutes):

  * level calibration: momentary (0.4 s) max loudness and peak of one vel-0.7 event
  * render speed (x real time)
  * edges: the buffer is exactly silent before the onset; the note's own tail ends below -90 dB
    re its peak (no truncation step); DC (mean / peak)
  * clicks: derivative-spike detector with only the onset guarded
  * aliasing: band-limited oscillators (bowed string, glottal source) measured raw at 1x vs the 2x
    oversampled path; complete instruments measured on steady test tones
"""
import json
import os
import time

import numpy as np
from _common import OUT
from scipy import signal

import filmsynth as F
from filmsynth import analysis as A
from filmsynth.core import midi_to_hz, to_midi
from filmsynth.instruments.base import Note
from filmsynth.instruments.choir import _glottal
from filmsynth.instruments.strings import _bow_osc

SR = 48000
CASES = [
    ("Pluck", F.Pluck(), "B3", 2, {}),
    ("Pluck B1 low", F.Pluck(), "B1", 2, {}),
    ("Pluck harmonic", F.Pluck(), "B3", 2, {"harmonic": 3}),
    ("Pluck slide+vib", F.Pluck(), "A3", 2, {"glide_to": "B3", "vib_cents": 20, "vib_rate": 2}),
    ("Bell glass", F.Bell(preset="glass"), "B5", 2, {}),
    ("Bell church", F.Bell(preset="church"), "B3", 2, {}),
    ("Bell bianzhong", F.Bell(preset="bianzhong"), "D5", 2, {}),
    ("Bell celesta", F.Bell(preset="celesta"), "B5", 2, {}),
    ("Bell bowed", F.Bell(preset="bowed"), "B5", 3, {}),
    ("Bell lamp", F.Bell(preset="lamp"), "D6", 2, {}),
    ("Organ 8' Gedackt", F.Organ(stops=("8g",)), "D4", 3, {}),
    ("Organ principal chorus", F.Organ(stops=("8p", "4p", "2p")), "D4", 3, {}),
    ("Organ pedal 32'+16'", F.Organ(stops=("32", "16")), "B2", 3, {}),
    ("Organ tutti + reeds", F.Organ(stops=("16p", "8p", "8g", "4p", "2.67", "2p", "mix", "16r", "8r")), "D4", 3, {}),
    ("Choir soprano (12)", F.Choir(section="soprano"), "B4", 3, {"vowel": "ah"}),
    ("Choir alto (12)", F.Choir(section="alto"), "D4", 3, {"vowel": "ah"}),
    ("Choir tenor (12)", F.Choir(section="tenor"), "F#3", 3, {"vowel": "oo"}),
    ("Choir bass (12)", F.Choir(section="bass"), "B2", 3, {"vowel": "mm"}),
    ("Strings vn1 (14)", F.Strings(section="vn1"), "B4", 3, {}),
    ("Strings va (10)", F.Strings(section="va"), "F#3", 3, {}),
    ("Strings vc (8)", F.Strings(section="vc"), "D3", 3, {}),
    ("Strings cb (8)", F.Strings(section="cb"), "B1", 3, {}),
    ("Strings tremolo", F.Strings(section="vn1"), "F#5", 3, {"articulation": "tremolo"}),
    ("Strings spiccato", F.Strings(section="va"), "D4", 0.25, {"articulation": "spiccato"}),
    ("Taiko nagado", F.Taiko(), None, 1, {}),
    ("Taiko odaiko", F.Taiko(size="odaiko"), None, 1, {}),
    ("Footsteps (24)", F.Footsteps(), None, 1, {}),
    ("Timpani", F.Timpani(), "B1", 1, {}),
    ("Timpani roll", F.Timpani(), "B1", 3, {"roll": True}),
    ("Boom", F.Boom(), "B0", 1, {}),
    ("Gong tam-tam", F.Gong(), "B1", 2, {}),
    ("Gong chau", F.Gong(preset="chau"), "B2", 2, {}),
    ("Cymbal swell", F.CymbalSwell(), None, 3, {}),
    ("LampBell (handbell)", F.LampBell(), "A6", 1, {}),
    ("Wind", F.Wind(), None, 4, {}),
    ("Blizzard", F.Blizzard(), None, 4, {}),
    ("Air rush", F.AirRush(), None, 3, {}),
    ("Shimmer", F.Shimmer(), "D6", 3, {}),
    ("Drone", F.Drone(), "D2", 4, {}),
    ("Riser", F.Riser(), "D6", 3, {}),
    ("Reverse swell", F.ReverseSwell(), "D4", 3, {}),
    ("Tick", F.Tick(), "B6", 0.2, {}),
    ("Wood key", F.WoodKey(), None, 0.2, {}),
    ("Sparkle", F.Sparkle(), "B5", 1.5, {}),
    ("Flame pop", F.FlamePop(), None, 0.5, {}),
    ("Puff", F.Puff(), None, 0.5, {}),
    ("Crunch (frost)", F.Crunch(), None, 0.3, {}),
    ("Hiss", F.Hiss(), None, 2, {}),
    ("Xiao", F.Xiao(), "D5", 2, {}),
    ("HeartDrum", F.HeartDrum(), "B1", 1, {}),
    ("Tap", F.Tap(), None, 0.2, {}),
    ("Breath (pulse)", F.Breath(), "D5", 0.5, {}),
    ("FeltTick", F.FeltTick(), "B5", 0.2, {}),
    ("FeltPiano", F.FeltPiano(), "C4", 2, {}),
    ("TautString", F.TautString(), "A3", 3, {}),
]


def instrument_tests():
    rows = []
    for name, inst, p, d, ex in CASES:
        m = to_midi(p) if p else float("nan")
        pre = 0.5
        nt = Note(pre, d, m, 0.7, ex, seed=11)
        n = int((pre + d + 18) * SR)
        t = time.time()
        y = inst.render_notes([nt], n)
        secs = time.time() - t
        # trim trailing silence for the speed figure
        a = np.max(np.abs(y), axis=0)
        last = int(np.nonzero(a > 1e-9)[0][-1]) + 1 if np.any(a > 1e-9) else n
        audio_s = last / SR - pre
        _, L = A.loudness_curve(y[:, :last + SR], 0.4, 0.05)
        pk = float(np.max(np.abs(y)))
        pre_ok = float(np.max(np.abs(y[:, : int((pre - 0.1) * SR)])))  # ensembles may start <= 100 ms early
        tail = float(np.max(np.abs(y[:, max(0, last - int(0.01 * SR)):last]))) if last > 0 else 0.0
        ycut = y[:, int((pre - 0.1) * SR): last + 1000]
        clicks = A.detect_clicks(ycut, [0.1], guard=(0.004, 0.06 + (d if ex.get("roll") or "Footsteps" in name or "Wood" in name
                                                                       or "Crunch" in name or "tremolo" in name.lower()
                                                                       or "Cymbal" in name else 0)))
        rows.append({
            "name": name, "loud_max_lufs": round(float(L.max()), 1), "peak_dbfs": round(20 * np.log10(pk + 1e-12), 1),
            "x_realtime": round(secs / max(audio_s, 1e-3), 3), "render_s": round(secs, 3), "audio_s": round(audio_s, 2),
            "pre_onset_abs": pre_ok, "tail_last10ms_db_re_peak": round(20 * np.log10(tail / pk + 1e-15), 1),
            "end_step_db_re_peak": round(20 * np.log10(float(np.max(np.abs(y[:, last - 1]))) / pk + 1e-15), 1),
            "dc_db_re_peak": round(20 * np.log10(np.max(np.abs(np.mean(y[:, :last], axis=1))) / pk + 1e-15), 1),
            "clicks": len(clicks), "corr_lr": round(float(A.correlation(y[:, :last])), 2),
        })
        print(rows[-1], flush=True)
    return rows


def alias_tests():
    out = []
    n = 2 * SR
    lp10 = signal.butter(8, 10000, fs=SR, output="sos")
    for f in (440.0, 1318.5, 2637.0):
        raw = _bow_osc(np.full(n, f), np.full(n, 0.11), 0.0, float(SR))
        os2 = signal.resample_poly(_bow_osc(np.full(2 * n, f), np.full(2 * n, 0.11), 0.0, float(2 * SR)), 1, 2)
        g1 = _glottal(np.full(n, f), np.ones(n), np.full(n, 0.6), np.ones(n), np.zeros(n), np.zeros(n), float(SR))
        g2 = signal.resample_poly(_glottal(np.full(2 * n, f), np.ones(2 * n), np.full(2 * n, 0.6), np.ones(2 * n),
                                           np.zeros(2 * n), np.zeros(2 * n), float(2 * SR)), 1, 2)
        t = np.arange(n) / SR
        naive = 2 * ((f * t) % 1) - 1
        r = {"f0": f}
        for k, x in (("naive_saw", naive), ("bow_1x", raw), ("bow_2x_used", os2), ("glottal_1x", g1), ("glottal_2x_used", g2)):
            r[k + "_full"] = round(A.alias_ratio(x, f, t0=0.5, dur=1.0, tol_hz=10), 1)
            r[k + "_below10k"] = round(A.alias_ratio(signal.sosfilt(lp10, x), f, t0=0.5, dur=1.0, tol_hz=10), 1)
        out.append(r)
        print(r, flush=True)
    # complete instruments on steady tones (no vibrato / ensemble spread)
    steady = []
    for name, inst, p, ex in (
        ("Pluck B5 (stiffness 0)", F.Pluck(stiffness=0.0, polar=0.0), "B5", {}),
        ("Organ tutti C6 (additive, harmonics < 0.46 fs)", F.Organ(stops=("8p", "4p", "2p", "mix", "8r"), detune=0.0, wind=0.0, chiff=0.0), "C6", {}),
        ("Strings vn1 1 player E7, no vibrato", F.Strings(section="vn1", players=1, vib_cents=0, detune=0, bow_noise=0, humanize=0), "E7", {}),
        ("Choir soprano 1 singer A5, no vibrato", F.Choir(section="soprano", singers=1, vib_cents=0, detune=0, breath=0, distance=0, humanize=0), "A5", {"vowel": "ah"}),
    ):
        m = to_midi(p)
        y = inst.render_notes([Note(0.2, 2.5, m, 0.8, ex, seed=5)], int(4 * SR))
        f0 = float(midi_to_hz(m))
        steady.append({"name": name, "f0": round(f0, 1),
                       "alias_full_db": round(A.alias_ratio(y, f0, t0=1.0, dur=1.0, tol_hz=max(10, 0.02 * f0)), 1),
                       "alias_below10k_db": round(A.alias_ratio(signal.sosfilt(lp10, y), f0, t0=1.0, dur=1.0, tol_hz=max(10, 0.02 * f0)), 1)})
        print(steady[-1], flush=True)
    return out, steady


if __name__ == "__main__":
    # python examples/calibrate.py
    os.makedirs(OUT, exist_ok=True)
    rows = instrument_tests()
    osc, steady = alias_tests()
    with open(os.path.join(OUT, "calibrate.json"), "w") as f:
        json.dump({"instruments": rows, "oscillator_alias": osc, "steady_alias": steady}, f, indent=1)
