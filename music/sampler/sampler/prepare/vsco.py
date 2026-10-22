"""Import VSCO 2 Community Edition (Versilian Studios: Sam Gossner, Simon Dalzell; github.com/sgossner/VSCO-2-CE,
CC0 1.0) for sampler.strings.

    import_wavs(staging_dir)   staging_dir/<repository path>.wav -> <samples>/vsco2/<section>/<art>/<name>.flac
    build_index()              (re)write <samples>/vsco2/index.json from the FLACs present

The index holds the MEASURED pitch of every sample (the file names use two different octave conventions), its
onset, level and dynamic layer. Rows come in the order the promo film's three import passes wrote them: the
strings (FOLDERS), their sustained tuning re-measured on the steady part (0.8-3.0 s after the onset: e.g. the
solo bass E1 v1 slides in over the first 1.5 s), then the woodwinds and horn (WINDS, same shape, sections
fl ob cl bn hn, 'sus' / 'spic' = staccato) and the timpani and suspended cymbal (PERC, sec 'timp' / 'cym',
art 'hit' / 'roll' / 'fx', the timpani with their measured pitch). Ported from tools/prepare_vsco.py
(+ --remeasure) and tools/prepare_vsco_winds.py; the measurements are the same.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

from .._paths import SAMPLES

warnings.filterwarnings('ignore')
OUT = SAMPLES / 'vsco2'
REPO = 'sgossner/VSCO-2-CE'
COMMIT = '440300901dfe9275fd84e0b7763af1f8443ae62e'
# (source folder, section id, articulation id)
FOLDERS = [('Strings/Violin Section/susVib', 'vn', 'sus'), ('Strings/Violin Section/Pizz', 'vn', 'pizz'),
           ('Strings/Violin Section/Spic', 'vn', 'spic'),
           ('Strings/Viola Section/susvib', 'va', 'sus'), ('Strings/Viola Section/pizz', 'va', 'pizz'),
           ('Strings/Viola Section/spic', 'va', 'spic'),
           ('Strings/Cello Section/susvib', 'vc', 'sus'), ('Strings/Cello Section/pizzT', 'vc', 'pizz'),
           ('Strings/Cello Section/spic', 'vc', 'spic'),
           ('Strings/Solo Contrabass/SusVib', 'cb', 'sus'), ('Strings/Solo Contrabass/SusNV', 'cb', 'sus_nv'),
           ('Strings/Solo Contrabass/Pizz', 'cb', 'pizz'), ('Strings/Solo Contrabass/Spic', 'cb', 'spic'),
           ('Strings/Solo Violin/Arco Vib', 'vn_solo', 'sus'), ('Strings/Solo Violin/Pizz', 'vn_solo', 'pizz')]
WINDS = [('Woodwinds/Flute/susvib', 'fl', 'sus'), ('Woodwinds/Flute/stac', 'fl', 'spic'),
         ('Woodwinds/Oboe/Vib', 'ob', 'sus'), ('Woodwinds/Oboe/Stacc', 'ob', 'spic'),
         ('Woodwinds/Clarinet/susLong', 'cl', 'sus'), ('Woodwinds/Clarinet/stac', 'cl', 'spic'),
         ('Brass/F Horn/sus', 'hn', 'sus'), ('Woodwinds/Bassoon/stac', 'bn', 'spic')]
PERC = [('Percussion/Timpani', 'timp', 'hit'), ('Percussion/Timpani/Rolls', 'timp', 'roll'),
        ('VSCO 1 Percussion/varMetal/Cymbals/susp', 'cym', None)]
# sounding pitch = file-name pitch + this (measured; used by fetch_samples.py --range only)
NAME_OFFSET = {'vn': 12, 'va': 12, 'vc': 12, 'cb': 0, 'vn_solo': 0, 'fl': 12, 'ob': 12, 'cl': 12, 'hn': 12, 'bn': 12}
_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def name_midi(name):
    m = re.search(r'_([A-G])(#?)(-?\d)_', name + '_')
    if not m:
        return None
    return _PC[m.group(1)] + (1 if m.group(2) else 0) + 12 * (int(m.group(3)) + 1)


def dyn_layer(name):
    m = re.search(r'_v(\d)', name)
    if m:
        return int(m.group(1))
    m = re.search(r'_(p|f)(?:_|\.|$)', name + '.')
    return {'p': 1, 'f': 3}[m.group(1)] if m else 2


def rr(name):
    m = re.search(r'_(?:RR|rr)(\d)', name)
    return int(m.group(1)) if m else 1


def measure(x, sr, nominal):
    """Onset (s), measured MIDI (float), RMS level of 0.25-1.25 s after onset (dB)."""
    import pyworld as pw
    mono = x.mean(axis=1) if x.ndim == 2 else x
    env = np.abs(mono)
    pk = env.max()
    on = int(np.argmax(env > pk * 10 ** (-40 / 20)))
    lo = max(25.0, 440 * 2 ** ((nominal - 69 - 14) / 12)) if nominal else 25.0
    hi = min(2200.0, 440 * 2 ** ((nominal - 69 + 14) / 12)) if nominal else 2200.0
    a = on + int(0.08 * sr)
    b = min(len(mono), on + int(2.5 * sr))
    seg = mono[a:b].astype(np.float64)
    midi = None
    if len(seg) > int(0.1 * sr):
        f0, _ = pw.harvest(seg, sr, f0_floor=lo, f0_ceil=hi, frame_period=5)
        f0 = f0[f0 > 0]
        if len(f0) > 5:
            midi = float(np.median(69 + 12 * np.log2(f0 / 440)))
    r0, r1 = on + int(0.25 * sr), on + int(1.25 * sr)
    lvl = float(10 * np.log10(np.mean(mono[r0:r1] ** 2) + 1e-14)) if r1 <= len(mono) else \
        float(10 * np.log10(np.mean(mono[on:] ** 2) + 1e-14))
    return on / sr, midi, lvl


def timp_pitch(x, sr):
    """strongest spectral peak 60-250 ms after the onset between 60 and 300 Hz (a drum's pitch)"""
    m = x.mean(axis=1)
    on = int(np.argmax(np.abs(m) > np.abs(m).max() * 0.1))
    seg = m[on + int(0.06 * sr): on + int(0.6 * sr)]
    seg = seg * np.hanning(len(seg))
    n = 1 << 18
    s = np.abs(np.fft.rfft(seg, n))
    f = np.fft.rfftfreq(n, 1 / sr)
    band = (f > 60) & (f < 300)
    k = np.flatnonzero(band)[np.argmax(s[band])]
    return float(69 + 12 * np.log2(f[k] / 440))


def local_path(source):
    """repository path of a .wav -> its path under <samples>/vsco2 (None if not a sampled folder)"""
    folder, name = source.rsplit('/', 1)
    stem = name[:-4]
    for f, sec, art in FOLDERS + WINDS:
        if folder == f:
            return f'{sec}/{art}/{stem}.flac'
    for f, sec, _ in PERC:
        if folder == f:
            return f'perc/{sec}/{stem}.flac'
    return None


def import_wavs(staging: Path, keep=False):
    """staging/<repository path>.wav -> lossless FLAC (same bit depth) under <samples>/vsco2."""
    n = 0
    for w in sorted(Path(staging).rglob('*.wav')):
        src = w.relative_to(staging).as_posix()
        rel = local_path(src)
        if rel is None:
            continue
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            x, sr = sf.read(w, dtype='float32', always_2d=True)
            info = sf.info(w)
            sf.write(dst, x, sr, subtype='PCM_24' if info.subtype == 'PCM_24' else 'PCM_16', format='FLAC')
        if not keep:
            w.unlink()
        n += 1
    return n


def _rows(folders, verbose):
    rows = []
    for folder, sec, art in folders:
        for f in sorted((OUT / sec / art).glob('*.flac')):
            x, sr = sf.read(f, dtype='float32', always_2d=True)
            nom = name_midi(f.stem)
            on, midi, lvl = measure(x, sr, nom)
            rows.append(dict(file=f'{sec}/{art}/{f.name}', source=f'{folder}/{f.stem}.wav', sec=sec, art=art,
                             name_midi=nom, midi=round(midi, 3) if midi is not None else None,
                             onset_s=round(on, 4), level_db=round(lvl, 2), dyn=dyn_layer(f.stem), rr=rr(f.stem),
                             dur=round(len(x) / sr, 3), sr=sr))
            if verbose:
                r = rows[-1]
                print(f"{sec:8} {art:6} {f.stem:34} name {nom} measured {r['midi']} dyn {r['dyn']} rr {r['rr']}",
                      flush=True)
    # per folder the octave offset between file name and measured pitch (median of rounded offsets)
    for folder, sec, art in folders:
        rs = [r for r in rows if r['sec'] == sec and r['art'] == art]
        offs = [round((r['midi'] - r['name_midi']) / 12) * 12 for r in rs
                if r['midi'] is not None and r['name_midi'] is not None]
        oct_off = int(np.median(offs)) if offs else 0
        for r in rs:
            r['name_octave_offset'] = oct_off
            r['pitch'] = (r['name_midi'] + oct_off) if r['name_midi'] is not None else None
            # fine tuning from the measurement when it agrees with the name (within 60 cents)
            if r['midi'] is not None and r['pitch'] is not None and abs(r['midi'] - r['pitch']) < 0.6:
                r['tune_cents'] = round((r['midi'] - r['pitch']) * 100, 1)
            else:
                r['tune_cents'] = 0.0
                r['pitch_suspect'] = r['midi'] is not None
    return rows


def _remeasure(rows):
    """tune_cents of the SUSTAINED strings from their steady part (short articulations keep the first
    measurement: their pitch moves after the pluck; below ~40 Hz harvest is unreliable)."""
    import pyworld as pw
    for r in rows:
        x, sr = sf.read(OUT / r['file'], dtype='float64', always_2d=True)
        mono = x.mean(axis=1)
        on = int(r['onset_s'] * sr)
        if r['art'] not in ('sus', 'sus_nv'):
            continue
        a, b = on + int(0.8 * sr), min(len(mono), on + int(3.0 * sr))
        if b - a < int(0.4 * sr):
            a, b = on + int(0.3 * sr), len(mono)
        nom = r['pitch']
        if nom is None or b - a < int(0.1 * sr) or nom < 28:
            continue
        lo = 440 * 2 ** ((nom - 69 - 3) / 12)
        hi = 440 * 2 ** ((nom - 69 + 3) / 12)
        f0, _ = pw.harvest(np.ascontiguousarray(mono[a:b]), sr, f0_floor=max(25.0, lo), f0_ceil=hi, frame_period=5)
        f0 = f0[f0 > 0]
        if len(f0) < 5:
            continue
        mid = float(np.median(69 + 12 * np.log2(f0 / 440)))
        r['midi_steady'] = round(mid, 3)
        if abs(mid - nom) < 0.6:
            r['tune_cents'] = round((mid - nom) * 100, 1)
            r.pop('pitch_suspect', None)


def _perc_rows(verbose):
    rows = []
    for folder, sec, art in PERC:
        files = sorted((OUT / 'perc' / sec).glob('*.flac'))
        # the timpani hits and rolls share perc/timp: a roll folder's files are the ones named *roll*
        for f in files:
            is_roll = 'roll' in f.stem.lower()
            if sec == 'timp' and (art == 'roll') != is_roll:
                continue
            x, sr = sf.read(f, dtype='float32', always_2d=True)
            a = art or ('roll' if 'roll' in f.stem else 'fx' if 'FX' in f.stem else 'hit')
            r = dict(file=f'perc/{sec}/{f.name}', source=f'{folder}/{f.stem}.wav', sec=sec, art=a, name=f.stem,
                     dyn=dyn_layer(f.stem), rr=rr(f.stem), dur=round(len(x) / sr, 3), sr=sr,
                     peak_db=round(float(20 * np.log10(np.abs(x).max() + 1e-12)), 2))
            if sec == 'timp':
                r['midi'] = round(timp_pitch(x, sr), 2)
                r['drum'] = int(f.stem[len('Timpani')])
            rows.append(r)
            if verbose:
                print(f"{sec:4} {a:5} {f.stem:34} {r.get('midi')}", flush=True)
    return rows


def build_index(verbose=True):
    strings = _rows(FOLDERS, verbose)
    _remeasure(strings)
    winds = _rows(WINDS, verbose)
    perc = _perc_rows(verbose)
    idx = dict(source=f'VSCO 2 Community Edition, github.com/{REPO} @ {COMMIT}, CC0 1.0',
               samples=strings + winds + perc,
               remeasured='tune_cents of the sustained strings from the steady part (sampler/prepare/vsco.py)',
               winds='woodwinds, horn, timpani, suspended cymbal (same repository and commit)')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'index.json').write_text(json.dumps(idx, ensure_ascii=False, indent=0))
    if verbose:
        print(len(idx['samples']), 'samples ->', OUT / 'index.json')
    return idx
