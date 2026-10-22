"""filmsynth: code-only instruments and DSP for film scores (48 kHz, float64, stereo).

Every sound is synthesised by code in this package: no recordings, samples, sound fonts,
external impulse responses or generative models.
"""
from .core import SR, PENTA_PCS, midi_to_hz, hz_to_midi, note_name, penta, snap_penta, to_midi
from .session import Curve, Session, TempoMap, parse_pos, pos_string
from .instruments import (AirRush, Bell, Blizzard, Boom, Breath, Choir, Crunch, CymbalSwell, Drone, FeltPiano, FeltTick,
                          FlamePop, Footsteps, Gong, HeartDrum, Hiss, LampBell, Organ, Pluck, Puff, ReverseSwell, Riser,
                          Shimmer, Sparkle, Strings, Taiko, Tap, TautString, Tick, Timpani, Wind, WoodKey, Xiao)
from .dsp import (EQ, Compressor, DCBlock, FeedbackDelay, FilterSweep, Gain, GainCurve, Haas, Limiter, Reverb,
                  Saturate, Swell, Width, persistence_delay, persistence_feedback, persistence_lowpass)

__version__ = "0.2.0"
