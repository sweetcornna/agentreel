"""Pluck example: a question-and-answer motive in B yu, yin/nao vibrato, a slide, harmonics, a low B1, damping,
and the persistence-driven echo rising from 'forgotten' to 'remembered'.

    python examples/pluck.py          # -> examples/out/pluck.wav, pluck.png, pluck.qc.json
"""
from _common import finish

from filmsynth import Pluck, Reverb, Session, persistence_delay


def build():
    s = Session(bars=8, tail=5.0)
    s.bus("hall", Reverb("hall", rt60=2.4))
    pers = s.curve([("1.1", 0.05), ("3.1", 0.1), ("5.1", 0.9), ("7.1", 1.0)])
    pl = s.track("pluck", Pluck(), gain_db=0, sends={"hall": -9}, humanize_ms=6, humanize_vel=0.04,
                 inserts=[persistence_delay(pers, time_beats=0.75, time_r_beats=1.0, mix=0.45)])
    # phrase 1 (question, ends on E)
    pl.seq(1, 1, [(1, "B3", 0.6), (1, "D4", 0.55), (1, "E4", 0.62), (1, "F#4", 0.7),
                  (1, "E4", 0.6), (1, "D4", 0.5), (2, "E4", 0.55, {"vib_cents": 9, "vib_rate": 3.5})])
    # phrase 2 (answer, home to B) with a slide up into B and a nao vibrato
    pl.seq(3, 1, [(1, "F#4", 0.65), (1, "E4", 0.55), (1, "D4", 0.55), (1, "E4", 0.6),
                  (1, "D4", 0.55), (1, "A3", 0.5, {"glide_to": "B3", "glide_start": 0.3, "glide_time": 0.3}),
                  (2, "B3", 0.6, {"vib_cents": 22, "vib_rate": 2.0, "vib_delay": 0.35})])
    # low register and harmonics (fan yin)
    pl.note(5, 1, 4, "B1", 0.7, pluck_pos=0.22)
    pl.note(5, 3, 1, "B2", 0.5, harmonic=2)
    pl.note(5, 4, 1, "F#3", 0.5, harmonic=3)
    pl.note(6, 1, 1, "B2", 0.55, harmonic=4)
    pl.note(6, 2, 1, "E3", 0.5, harmonic=3)
    # damped (stopped) notes and a bright high run
    for i, p in enumerate(["B4", "D5", "E5", "F#5", "A5", "B5"]):
        pl.note(7, 1 + i * 0.5, 0.4, p, 0.55 + 0.05 * i, let_ring=False, brightness=0.7)
    pl.note(8, 1, 4, "B3", 0.5, vib_cents=12, vib_rate=3.2)
    pl.note(8, 1, 4, "F#4", 0.45)
    return s


if __name__ == "__main__":
    finish(build(), "pluck")
