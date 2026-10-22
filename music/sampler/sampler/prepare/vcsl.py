"""Import VCSL (Versilian Community Sample Library, github.com/sgossner/VCSL, CC0 1.0) for sampler.folk and
sampler.nature.

Two kinds of sets (SETS below):
  * 'wav' sets, used by sampler.nature.sfx / sampler.fx.impact (woodblock, hand chimes, tubular bells, gong,
    slit drum): the WAV files are kept as they are in <samples>/vcsl/<Folder>/;
  * 'folk' sets, played by sampler.folk (plucked strings, bars, the grand piano, hand percussion): converted
    losslessly to FLAC (same sample rate and bit depth) in <samples>/vcsl/<Folder>/<name>.flac and indexed in
    <samples>/vcsl/index_folk.json: per sample the set, the articulation / hit kind, the dynamic layer,
    round-robin, onset, level and, for the pitched sets, the MEASURED pitch (the name's pitch, the octave
    convention checked per set, the fine tuning from the measurement).

    import_wavs(staging_dir)   staging_dir/<set id>/*.wav -> the library (fetch_samples.py calls this)
    build_index()              (re)write index_folk.json from the FLACs present

The index also carries 'ref': the layer medians and hit peaks of each COMPLETE set (sampler/data/
vcsl_ref_levels.json), so that sampler.folk levels a partial download exactly as the whole set. Ported from
the promo film's tools/prepare_vcsl_folk.py; the measurements are the same.
"""
from __future__ import annotations

import json
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

from .._paths import SAMPLES

warnings.filterwarnings('ignore')
OUT = SAMPLES / 'vcsl'
REF_FILE = Path(__file__).resolve().parents[1] / 'data' / 'vcsl_ref_levels.json'
REPO = 'sgossner/VCSL'
COMMIT = 'c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e'
# set id -> (folder under <samples>/vcsl, VCSL source folder, kind, pitched, sounding = name + offset or None)
SETS = {
    # used by sampler.nature / sampler.fx, kept as WAV
    'woodblock': ('Woodblock', 'Idiophones/Struck Idiophones/Woodblock', 'wav', False, None),
    'hand_chimes': ('Hand Chimes', 'Idiophones/Struck Idiophones/Hand Chimes', 'wav', True, 12),
    'tubular_bells': ('Tubular Bells 1', 'Idiophones/Struck Idiophones/Tubular Bells 1', 'wav', True, 24),
    'gong': ('Gong 1', 'Idiophones/Struck Idiophones/Gong 1', 'wav', False, None),
    'slit_drum': ('Slit Drum', 'Idiophones/Struck Idiophones/Slit Drum', 'wav', False, None),
    # sampler.folk, FLAC + index
    'strumstick': ('Strumstick', 'Chordophones/Composite Chordophones/Strumstick/Finger', 'folk', True, 12),
    'folk_harp': ('Folk Harp', 'Chordophones/Composite Chordophones/Folk Harp', 'folk', True, 12),
    'marimba': ('Marimba', 'Idiophones/Struck Idiophones/Marimba', 'folk', True, 12),
    'balafon': ('Balafon Soft Mallet', 'Idiophones/Struck Idiophones/Balafon/Soft Mallet', 'folk', True, 12),
    'dan_tranh': ('Dan Tranh', 'Chordophones/Zithers/Dan Tranh/Normal', 'folk', True, 12),
    'dan_tranh_gliss': ('Dan Tranh Gliss', 'Chordophones/Zithers/Dan Tranh/Gliss', 'folk', False, None),
    'grand_piano': ('Grand Piano', 'Chordophones/Zithers/Grand Piano, Steinway B/Sus', 'folk', True, 0),
    'cajon': ('Cajon', 'Idiophones/Struck Idiophones/Cajon', 'folk', False, None),
    'claps': ('Claps', 'Idiophones/Struck Idiophones/Claps', 'folk', False, None),
    'shaker_small': ('Shaker, Small', 'Idiophones/Struck Idiophones/Shaker, Small', 'folk', False, None),
    'shaker_large': ('Shaker, Large', 'Idiophones/Struck Idiophones/Shaker, Large', 'folk', False, None),
    'frame_drum': ('Frame Drum', 'Membranophones/Struck Membranophones/Frame Drum', 'folk', False, None),
    'cabasa': ('Cabasa', 'Idiophones/Struck Idiophones/Cabasa', 'folk', False, None),
    'bongos': ('Bongos', 'Membranophones/Struck Membranophones/Bongos', 'folk', False, None),
    'conga': ('Conga', 'Membranophones/Struck Membranophones/Conga', 'folk', False, None),
    'tambourine': ('Tambourine 1', 'Idiophones/Struck Idiophones/Tambourine 1', 'folk', False, None),
    'hihat': ('Hi-Hat Cymbal', 'Idiophones/Struck Idiophones/Hi-Hat Cymbal', 'folk', False, None),
}
FOLK = [k for k, v in SETS.items() if v[2] == 'folk']
_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
_DYN_WORDS = {'ppp': 1, 'pp': 2, 'p': 3, 'mp': 4, 'mf': 5, 'f': 6, 'ff': 7, 'fff': 8,
              'soft': 3, 'med': 5, 'loud': 7}
_DROP = {'Mid', 'Sum', 'Main', 'Finger', 'Normal', 'EWHarp', 'Strumstick', 'Marimba', 'hit', 'Outrigger',
         'EthnicXylo', 'softM'}
_DROP_PIANO = {'JHPiano', 'Sus', 'Close'}       # Steinway B：JHPiano_Sus_Close_<音>_vl<n>_rr1（只在这一组里丢）


def parse(stem):
    """(name_midi, kind, dyn, rr) from a VCSL file name."""
    toks = stem.split('_')
    drop = _DROP | _DROP_PIANO if toks[0] == 'JHPiano' else _DROP
    midi, dyn, rr, kind = None, None, 1, []
    for i, t in enumerate(toks):
        m = re.fullmatch(r'([A-Ga-g])(#?)(-?\d)', t)
        if m and midi is None:
            midi = _PC[m.group(1).upper()] + (1 if m.group(2) else 0) + 12 * (int(m.group(3)) + 1)
            continue
        m = re.fullmatch(r'(?:vl|v)(\d)', t)
        if m:
            dyn = int(m.group(1))
            continue
        if t in _DYN_WORDS:
            dyn = _DYN_WORDS[t]
            continue
        m = re.fullmatch(r'(?:rr|RR)(\d+)', t)
        if m:
            rr = int(m.group(1))
            continue
        if re.fullmatch(r'\d+', t) and i == len(toks) - 1:
            rr = int(t)
            continue
        if t not in drop:
            kind.append(t)
    return midi, '_'.join(kind) or 'hit', dyn if dyn is not None else 1, rr


def name_midi(filename):
    """MIDI number of the note name in a VCSL file name (None if there is none)."""
    m = re.search(r'(?:^|_)([A-G])(#?)(-?\d)(?=_|\.|$)', Path(filename).stem)
    if not m:
        return None
    return _PC[m.group(1)] + (1 if m.group(2) else 0) + 12 * (int(m.group(3)) + 1)


def onset_level(mono, sr):
    env = np.abs(mono)
    pk = float(env.max())
    on = int(np.argmax(env > pk * 10 ** (-40 / 20)))
    a, b = on, min(len(mono), on + int(0.3 * sr))
    lvl = float(10 * np.log10(np.mean(mono[a:b] ** 2) + 1e-14))
    return on, lvl, 20 * np.log10(pk + 1e-12)


def measure_pitch(mono, sr, on, nominal):
    """Median f0 (MIDI) over 40-700 ms after the onset (a pluck or a bar settles after its strike),
    searched within +-14 semitones of the name so that an octave convention shows up."""
    import pyworld as pw
    lo = max(25.0, 440 * 2 ** ((nominal - 69 - 14) / 12))
    hi = min(3000.0, 440 * 2 ** ((nominal - 69 + 14) / 12))
    a, b = on + int(0.04 * sr), min(len(mono), on + int(0.7 * sr))
    seg = np.ascontiguousarray(mono[a:b].astype(np.float64))
    if len(seg) < int(0.1 * sr):
        return None
    f0, _ = pw.harvest(seg, sr, f0_floor=lo, f0_ceil=hi, frame_period=5)
    f0 = f0[f0 > 0]
    if len(f0) < 5:
        return None
    return float(np.median(69 + 12 * np.log2(f0 / 440)))


def measure_bar(mono, sr, on, nominal):
    """Struck bars (marimba, balafon): the strongest spectral peak of 10-510 ms after the strike
    within +-14 semitones of the name (harvest often fails on the bars' inharmonic spectra)."""
    a = on + int(0.01 * sr)
    seg = mono[a:a + int(0.5 * sr)].astype(np.float64)
    seg = seg * np.hanning(len(seg))
    n = 1 << 19
    spec = np.abs(np.fft.rfft(seg, n))
    f = np.fft.rfftfreq(n, 1 / sr)
    lo = 440 * 2 ** ((nominal - 69 - 14) / 12)
    hi = 440 * 2 ** ((nominal - 69 + 14) / 12)
    band = (f > lo) & (f < hi)
    k = int(np.flatnonzero(band)[np.argmax(spec[band])])
    return float(69 + 12 * np.log2(f[k] / 440))


BARS = {'marimba', 'balafon'}


def import_wavs(staging: Path, sets=None, keep=False):
    """Move / convert the downloaded WAVs staging/<set id>/*.wav into the library. Returns the folk sets touched."""
    touched = []
    for sid in (sets or SETS):
        folder, _, kind, _, _ = SETS[sid]
        src = Path(staging) / sid
        if not src.is_dir():
            continue
        dst_dir = OUT / folder
        dst_dir.mkdir(parents=True, exist_ok=True)
        for w in sorted(src.glob('*.wav')):
            if kind == 'wav':
                if not (dst_dir / w.name).exists():
                    shutil.copyfile(w, dst_dir / w.name)
            else:
                dst = dst_dir / (w.stem + '.flac')
                if not dst.exists():
                    x, sr = sf.read(w, dtype='float32', always_2d=True)
                    info = sf.info(w)
                    sf.write(dst, x, sr, subtype='PCM_24' if info.subtype == 'PCM_24' else 'PCM_16', format='FLAC')
            if not keep:
                w.unlink()
        if kind == 'folk':
            touched.append(sid)
    return touched


def _ref_levels():
    return json.loads(REF_FILE.read_text())['sets'] if REF_FILE.exists() else {}


def build_index(verbose=True):
    """Measure every FLAC of the folk sets present and write <samples>/vcsl/index_folk.json."""
    rows = []
    for sid in FOLK:
        folder, vfolder, _, pitched, _ = SETS[sid]
        for f in sorted((OUT / folder).glob('*.flac')):
            x, sr = sf.read(f, dtype='float32', always_2d=True)
            rel = Path(folder) / f.name
            nom, kind, dyn, rr = parse(f.stem)
            mono = x.mean(axis=1)
            on, lvl, pk = onset_level(mono, sr)
            r = dict(file=str(rel), source=f'{vfolder}/{f.stem}.wav', set=sid, kind=kind, dyn=dyn, rr=rr,
                     onset_s=round(on / sr, 4), level_db=round(lvl, 2), peak_db=round(pk, 2),
                     dur=round(len(x) / sr, 3), sr=sr, channels=int(x.shape[1]))
            if pitched and nom is not None:
                r['name_midi'] = nom
                mid = (measure_bar if sid in BARS else measure_pitch)(mono, sr, on, nom)
                r['midi'] = round(mid, 3) if mid is not None else None
            rows.append(r)
            if verbose:
                print(f"{sid:14} {f.stem:44} {kind:16} dyn {dyn} rr {rr} name {r.get('name_midi')} "
                      f"measured {r.get('midi')} {lvl:6.1f} dB", flush=True)
    # per pitched set: the octave between the names and the measured pitches, then the fine tuning
    for sid in FOLK:
        if not SETS[sid][3]:
            continue
        rs = [r for r in rows if r['set'] == sid and r.get('name_midi') is not None]
        offs = [round((r['midi'] - r['name_midi']) / 12) * 12 for r in rs if r.get('midi') is not None]
        oct_off = int(np.median(offs)) if offs else (SETS[sid][4] or 0)
        for r in rs:
            r['name_octave_offset'] = oct_off
            r['pitch'] = r['name_midi'] + oct_off
            if r.get('midi') is not None and abs(r['midi'] - r['pitch']) < 0.6:
                r['tune_cents'] = round((r['midi'] - r['pitch']) * 100, 1)
            else:
                r['tune_cents'] = 0.0
                r['pitch_suspect'] = r.get('midi') is not None
    present = {r['set'] for r in rows}
    ref = {k: v for k, v in _ref_levels().items() if k in present}
    for sid, v in ref.items():           # a complete set must reproduce its reference exactly
        rs = [r for r in rows if r['set'] == sid]
        if len(rs) == v.get('files'):
            for d, med in v.get('layer_median_db', {}).items():
                got = float(np.median([r['level_db'] for r in rs if r['dyn'] == int(d)]))
                if got != med:
                    print(f'warning: {sid} layer {d} median {got} != reference {med}')
    idx = dict(source=f'VCSL, https://github.com/{REPO} @ {COMMIT}, CC0 1.0', samples=rows, ref=ref)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'index_folk.json').write_text(json.dumps(idx, ensure_ascii=False, indent=0))
    if verbose:
        print(len(rows), 'samples ->', OUT / 'index_folk.json')
    return idx


def measure_ref(index_path, sets=None):
    """Reference statistics of complete sets from an index_folk.json (maintainers: run on a full download)."""
    from .. import folk
    idx = json.loads(Path(index_path).read_text())
    rows = idx['samples']
    out = {}
    for sid in sets or sorted({r['set'] for r in rows}):
        rs = [r for r in rows if r['set'] == sid]
        dyns = sorted({r['dyn'] for r in rs})
        top = max(dyns)
        v = dict(files=len(rs),
                 layer_median_db={str(d): float(np.median([r['level_db'] for r in rs if r['dyn'] == d])) for d in dyns},
                 top_median_db=float(np.median([r['level_db'] for r in rs if r['dyn'] == top])))
        if not SETS[sid][3]:                # percussion: median peak of the loudest layer per hit kind
            kp = {}
            for kind in sorted({r['kind'] for r in rs}):
                kr = [r for r in rs if r['kind'] == kind]
                kt = max(r['dyn'] for r in kr)
                kp[kind] = float(np.median([folk._hit_raw(r['file'])[2] for r in kr if r['dyn'] == kt]))
            v['kind_peak'] = kp
        out[sid] = v
    return out
