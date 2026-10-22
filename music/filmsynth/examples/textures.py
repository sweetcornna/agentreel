"""Texture example (the transition toolbox): night wind with whistles tuned to B, rising into a blizzard that is cut
dead, silence, a reverse-reverb inhale into an air rush (cloud punch-through), a shimmer on D gong, a beating drone
underneath, and a riser.

    python examples/textures.py       # -> examples/out/textures.wav, textures.png, textures.qc.json
"""
from _common import finish

from filmsynth import AirRush, Blizzard, Drone, Reverb, ReverseSwell, Riser, Session, Shimmer, Wind


def build():
    s = Session(bars=10, tail=4.0)
    s.bus("plate", Reverb("plate", rt60=2.5))
    w = s.track("wind", Wind(), gain_db=0)
    w.note(1, 1, 12, None, 0.6, intensity=0.35, intensity_end=0.8, whistle=0.5, whistle_pitch="B5", fade_in=2.0,
           fade_out=0.6)
    bz = s.track("blizzard", Blizzard(), gain_db=-2)
    bz.note(3, 3, 6, None, 0.8, intensity=0.4, intensity_end=1.0, fade_in=1.5)
    s.cut("5.1", "5.3", fade_out_ms=4, fade_in_ms=30)     # a hard cut to silence (everything, tails included)
    dr = s.track("drone", Drone(chord=(0, 7, 12, 19)), sends={"plate": -12})
    dr.note(5, 3, 14, "D2", 0.55)
    rs = s.track("rev_swell", ReverseSwell(chord=(0, 7, 12, 16, 19)))
    rs.note(6, 1, 4, "D4", 0.75)                           # crests just before 7.1 (its written end)
    ar = s.track("air_rush", AirRush(), gain_db=-1)
    ar.note(7, 1, 4, None, 0.9)
    sh = s.track("shimmer", Shimmer(chord=(0, 2, 4, 7, 9, 12)), sends={"plate": -6})
    sh.note(8, 1, 10, "D6", 0.6, density=24)
    ri = s.track("riser", Riser(), gain_db=-6)
    ri.note(9, 1, 4, "D6", 0.7)
    return s


if __name__ == "__main__":
    finish(build(), "textures")
