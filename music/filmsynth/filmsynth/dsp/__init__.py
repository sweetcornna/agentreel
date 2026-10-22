from .automation import FilterSweep, GainCurve, Swell
from .delay import FeedbackDelay, persistence_delay, persistence_feedback, persistence_lowpass, vardelay
from .dynamics import (Compressor, compress, duck, limit, lufs, master_chain, normalize_lufs, peak_db, saturate,
                       true_peak_db)
from .filters import (biquad, bandpass, butter, dc_block, eq, highpass, highshelf, lowpass, lowshelf, modal_bank,
                      onepole_lp, peak, rbj, svf)
from .reverb import Reverb, convolve, make_ir, reverb
from .stereo import correlation, haas, pan, width
from .inserts import DCBlock, EQ, Gain, Haas, Limiter, Saturate, Width
