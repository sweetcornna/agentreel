"""Choir example (formant synthesis): a distant SATB (12 singers per section) humming 'mm', opening to 'oo' and 'ah',
a legato line in the sopranos, ending on a D gong chord.

    python examples/choir.py          # -> examples/out/choir.wav, choir.png, choir.qc.json
"""
from _common import finish

from filmsynth import Choir, Reverb, Session


def build():
    s = Session(bars=8, tail=6.0)
    s.bus("cathedral", Reverb("cathedral", rt60=5.0))
    secs = {}
    for name, gain in (("soprano", -1), ("alto", 0), ("tenor", 0), ("bass", 1)):
        secs[name] = s.track(name, Choir(section=name, singers=12, distance=0.6), gain_db=gain,
                             sends={"cathedral": -3})
    S, A, T, B = (secs[k] for k in ("soprano", "alto", "tenor", "bass"))
    # bars 1-2: hum on B yu (B D F#)
    B.note(1, 1, 8, "B2", 0.45, vowel="mm"); T.note(1, 1, 8, "F#3", 0.4, vowel="mm")
    A.note(1, 1, 8, "D4", 0.4, vowel="mm"); S.note(1, 1, 8, "F#4", 0.35, vowel="mm")
    # bars 3-4: open to oo then ah, crescendo
    B.note(3, 1, 8, "E2", (0.45, 0.6), vowel=("oo", "ah")); T.note(3, 1, 8, "B3", (0.4, 0.6), vowel=("oo", "ah"))
    A.note(3, 1, 8, "E4", (0.4, 0.6), vowel=("oo", "ah"))
    S.seq(3, 1, [(2, "B4", (0.4, 0.5)), (2, "A4", 0.5), (2, "B4", 0.55), (2, "D5", (0.55, 0.65))], vowel="oo")
    # bars 5-6: A pedal, soprano line legato
    B.note(5, 1, 8, "A2", 0.6, vowel="ah"); T.note(5, 1, 8, "E3", 0.6, vowel="ah")
    A.note(5, 1, 8, "A3", 0.55, vowel="ah")
    S.seq(5, 1, [(1, "E5", 0.62), (1, "D5", 0.6), (2, "B4", 0.6), (2, "C#5", 0.62), (2, "E5", 0.66)], vowel="ah")
    # bars 7-8: D gong arrival
    B.note(7, 1, 8, "D2", (0.7, 0.4), vowel="ah"); T.note(7, 1, 8, "A3", (0.7, 0.4), vowel="ah")
    A.note(7, 1, 8, "F#4", (0.7, 0.4), vowel=("ah", "oh")); S.note(7, 1, 8, "D5", (0.72, 0.4), vowel=("ah", "oo"))
    return s


if __name__ == "__main__":
    finish(build(), "choir")
