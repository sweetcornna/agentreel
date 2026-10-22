"""Shared example harness: render a session, write WAV + spectrogram + QC json into examples/out/ (or $FILMSYNTH_OUT).

Ported from the instrument demos of the original library; the examples are not part of any film."""
from __future__ import annotations

import json
import os
import time

from filmsynth import analysis as A

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("FILMSYNTH_OUT") or os.path.join(HERE, "out")


def finish(session, name, target_lufs=-18.0, workers=None, extra=None, stems=False):
    import soundfile as sf

    os.makedirs(OUT, exist_ok=True)
    t = time.time()
    tmp = os.path.join(OUT, f"_{name}_tmp")
    res = session.render(tmp, mix_name="mix.wav", workers=workers, target_lufs=target_lufs, stems=stems, events=False,
                         write=False, verbose=False)
    secs = time.time() - t
    wav = os.path.join(OUT, f"{name}.wav")
    sf.write(wav, res.mix.T, 48000, subtype="PCM_24")
    onsets = sorted(nt.t0 for tr in session.tracks.values() for nt in tr.notes)
    q = A.qc(res.mix, onsets)
    q["render_s"] = round(secs, 2)
    q["render_x_realtime"] = round(secs / q["duration_s"], 3)
    q["track_render_s"] = res.stats["tracks"]
    q["bus_render_s"] = res.stats["buses"]
    if extra:
        q.update(extra)
    A.spectrogram_png(res.mix, os.path.join(OUT, f"{name}.png"),
                      title=f"{name}  LUFS {q['lufs_i']}  TP {q['true_peak_dbtp']}")
    with open(os.path.join(OUT, f"{name}.qc.json"), "w") as f:
        json.dump(q, f, indent=1)
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    print(json.dumps({k: v for k, v in q.items() if k not in ("track_render_s", "bus_render_s")}, indent=None))
    print(f"-> {wav}")
    return res, q
