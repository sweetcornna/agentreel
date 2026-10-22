"""Convert the Wikimedia Commons field recordings into the library format (sampler.nature, sampler.birds).

    process(name, original_path, meta)    one downloaded file -> <samples>/field/<name>.flac + manifest entry

For every file: decode with ffmpeg (some Commons .ogg files carry an extra skeleton stream libsndfile cannot
open), keep the original bytes in <samples>/field/original/, trim to the usable range, fold to mono when the
two channels are identical (L/R correlation > 0.99), resample to 48 kHz (scipy resample_poly, Kaiser windowed
sinc), remove DC (20 Hz high-pass) and write FLAC 24-bit. Metadata (source page, author, licence, processing)
goes to <samples>/field/manifest.json. No level normalisation is baked in (nature.py normalises at render
time). Ported from the promo film's tools/prepare_field.py; the processing is the same.

ffmpeg: $FFMPEG, else `ffmpeg` on PATH, else the usual install locations (see find_ffmpeg).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .._dsp import resample_sr
from .._paths import SAMPLES

FIELD = SAMPLES / 'field'

# library name -> (trim (start_s, end_s) or None, note, extra high-pass Hz, de-click, de-clip)
SELECT = {
    'stream_molln_bachlauf': (None, 'babbling brook, Neumann TLM 102, 48 kHz', None, False, False),
    'rivulet_molln_rinnsal': (None, 'small rivulet, Neumann TLM 102, 48 kHz', None, False, False),
    'creek_forest_lawn': (None, '7.5 min creek (shotgun mic), MP3-era bandwidth ~16 kHz', None, False, False),
    'river_shallow_stony': (None, 'shallow stony river, stereo, ~16 kHz bandwidth', None, False, False),
    'forest_ambience_nille': (None, 'soft wind in trees, some distant birds', None, False, False),
    'pine_forest_dobroide': (None, 'summer pine forest: wind in pines, birds, cicadas', None, False, False),
    'birds_fontainebleau': (None, 'forest bird chorus (wrens), ~11 kHz bandwidth', None, False, False),
    'birds_garden_sparse': (None, 'sparse garden bird phrases (blackbird-like), mono', None, False, False),
    'dawn_chorus_dorset': ((600.0, 1020.0),
                           '7-min excerpt (10:00-17:00, the densest part) of a 50-min dawn chorus; ~6 kHz '
                           'bandwidth; 300 Hz high-pass removes low-frequency wind/handling thumps', 300.0, True, False),
    'birds_evening_waidachswald': (None, 'summer evening forest birdsong, true stereo', None, False, False),
    'cricket_gryllus_campestris': ((44.5, 192.0),
                                   'field cricket song, 15 cm; first 44 s (handling/tape noise) removed', None,
                                   False, False),
    'rooster_chardonnet': (None, 'single rooster crow, clean', None, False, False),
    'rooster_alys_medium': (None, 'rooster crow with farm background', None, False, False),
    'rooster_alys_small': (None, 'small rooster crow with background', None, False, False),
    'dog_bark_edo': (None, 'three dog barks; source clipped in 38 samples -> cubic de-clip', None, False, True),
    'dog_bark_george': (None, 'single dog bark (labrador "George", ca. 1996)', None, False, False),
}
PROCESSING = ('ffmpeg decode -> trim -> mono fold if L==R [-> cubic de-clip] -> resample_poly to 48 kHz '
              '-> 20 Hz high-pass (zero-phase) [-> extra high-pass] [-> declick patching] '
              '-> 5 ms edge fades -> scale so peak <= -0.2 dBFS -> FLAC 24-bit')


def find_ffmpeg():
    """$FFMPEG, then PATH, then the usual locations on macOS (Homebrew), Linux and Windows."""
    env = os.environ.get('FFMPEG')
    if env:
        return env
    hit = shutil.which('ffmpeg')
    if hit:
        return hit
    cands = ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg', '/usr/bin/ffmpeg']
    if sys.platform.startswith('win'):
        cands = [r'C:\ffmpeg\bin\ffmpeg.exe', r'C:\Program Files\ffmpeg\bin\ffmpeg.exe']
    for c in cands:
        if Path(c).exists():
            return c
    raise FileNotFoundError('ffmpeg not found: install it or set FFMPEG=/path/to/ffmpeg')


def declip(x, thr=0.999, ctx=4):
    from scipy.interpolate import CubicSpline
    x = x.astype(np.float64).copy()
    for ch in range(x.shape[1]):
        c = np.abs(x[:, ch]) >= thr
        if not c.any():
            continue
        good = np.flatnonzero(~c)
        idx = np.flatnonzero(c)
        # runs of clipped samples
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for r in runs:
            a, b = r[0], r[-1]
            left = good[(good < a)][-ctx:]
            right = good[(good > b)][:ctx]
            if len(left) < 2 or len(right) < 2:
                continue
            k = np.concatenate([left, right])
            cs = CubicSpline(k, x[k, ch])
            x[a:b + 1, ch] = cs(np.arange(a, b + 1))
    return x


# Rejected after checking the file pages / spectrograms (kept out of the library):
#   'Dog.ogg' (PD)                  -> tail thumping, licking, yawning; no barking
#   'Bark (avk).wav' (CC0)          -> a person pronouncing the Kotava word 'bark'
#   'Dog in Antefasy.wav'           -> pronunciation of the word 'dog' in a Malagasy dialect
#   PLOS ONE growls (CC BY)         -> resynthesised growl stimuli, not barks
#   huamei (CC0)                    -> strong steady tonal noise bands 4-10 kHz
#   reinsamba (CC BY)               -> 2068 clipped samples; busy evening chorus, not insects
#   country night (PD)              -> 7 kHz MP3 bandwidth, car-like event at 5 s


def declick(x, times, half_ms=4.0, xf_ms=1.0):
    """Patch short broadband bumps: replace +-half_ms around each detected event with
    the audio just before it (noise-like material), raised-cosine crossfaded."""
    x = x.copy()
    h = int(half_ms / 1000 * 48000)
    f = int(xf_ms / 1000 * 48000)
    w = 0.5 - 0.5 * np.cos(np.pi * (np.arange(f) + 0.5) / f)
    for t in times:
        i = int(t * 48000)
        a, b = i - h, i + h
        src_a = a - 2 * h - 2 * f
        if src_a < 0 or b + f > len(x):
            continue
        patch = x[src_a:src_a + (b - a) + 2 * f].copy()
        seg = x[a - f:b + f].copy()
        mix = np.ones(len(seg))
        mix[:f] = w
        mix[-f:] = w[::-1]
        x[a - f:b + f] = seg * (1 - mix[:, None]) + patch * mix[:, None]
    return x


def decode(path: Path):
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / 'x.wav'
        subprocess.run([find_ffmpeg(), '-v', 'error', '-y', '-i', str(path), '-map', '0:a:0',
                        '-c:a', 'pcm_f32le', str(out)], check=True)
        x, sr = sf.read(out, dtype='float32', always_2d=True)
    return x, sr


def process(name, src: Path, m: dict):
    """m: Commons metadata (title, page, url, license, license_url, artist, credit, desc)."""
    trim, note, hp, do_declick, do_declip = SELECT[name]
    (FIELD / 'original').mkdir(parents=True, exist_ok=True)
    man_path = FIELD / 'manifest.json'
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    orig_name = m['title'].split(':', 1)[1].replace('/', '_')
    if Path(src).resolve() != (FIELD / 'original' / orig_name).resolve():
        shutil.copyfile(src, FIELD / 'original' / orig_name)
    x, sr = decode(src)
    if trim:
        x = x[int(trim[0] * sr):int(trim[1] * sr)]
    ch = x.shape[1]
    if ch == 2 and np.corrcoef(x[:, 0], x[:, 1])[0, 1] > 0.99:
        x = x.mean(axis=1, keepdims=True)
    n_clip = int((np.abs(x) >= 0.999).sum())
    if do_declip:
        x = declip(x)
    x = resample_sr(x, sr, 48000)
    x = signal.sosfiltfilt(signal.butter(2, 20 / 24000, 'highpass', output='sos'), x, axis=0)
    if hp:
        x = signal.sosfiltfilt(signal.butter(4, hp / 24000, 'highpass', output='sos'), x, axis=0)
    f = int(0.005 * 48000)
    ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(f) + 0.5) / f)
    x[:f] *= ramp[:, None]
    x[-f:] *= ramp[::-1, None]
    n_clicks = None
    if do_declick:
        from ..analysis import detect_clicks
        ev = detect_clicks(np.repeat(x, 2, axis=1) if x.shape[1] == 1 else x)
        x = declick(x, ev)
        n_clicks = [len(ev), len(detect_clicks(np.repeat(x, 2, axis=1) if x.shape[1] == 1 else x))]
    peak = float(np.abs(x).max())
    g = min(1.0, 0.98 / peak)          # resampling can create inter-sample overs
    x = x * g
    sf.write(FIELD / f'{name}.flac', x.astype(np.float32), 48000, subtype='PCM_24')
    manifest[name] = {
        'file': f'{name}.flac', 'original_file': f'original/{orig_name}',
        'title': m['title'], 'page': m['page'], 'download_url': m['url'].split('?')[0],
        'license': m['license'], 'license_url': m['license_url'], 'author': m['artist'],
        'credit_line': m['credit'], 'description': m['desc'], 'note': note,
        'source_sr': sr, 'source_channels': ch, 'channels': int(x.shape[1]),
        'duration_s': round(len(x) / 48000, 2), 'peak_dbfs': round(20 * np.log10(peak + 1e-12), 1),
        'trim_s': trim, 'extra_highpass_hz': hp, 'source_clipped_samples': n_clip,
        'declipped': do_declip, 'declick_before_after': n_clicks, 'gain_applied_db': round(20 * np.log10(g), 2),
        'processing': PROCESSING,
    }
    print(f'{name:30} {m["license"]:14} {len(x) / 48000:7.1f}s ch={x.shape[1]} peak {20 * np.log10(peak):6.1f} dBFS',
          flush=True)
    man_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return manifest[name]
