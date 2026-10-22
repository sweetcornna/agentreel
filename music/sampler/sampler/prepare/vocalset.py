"""Import the VocalSet subset used by sampler.voice and build its note index.

VocalSet 1.1 (Wilkins, Seetharaman, Wahl & Pardo, ISMIR 2018), Zenodo record 1442513,
CC BY 4.0. The zip (2.08 GB) is not downloaded whole: fetch_samples.py reads the members
listed in SUBSET with HTTP range requests (zipfile checks each member's CRC-32).

    build(<dir with FULL/...wav>, jobs=12)      (fetch_samples.py --only vocalset calls this)

runs the three passes of the promo film's tools/prepare_vocalset.py in order (units, then
refine_breaths, then fix_edges) and writes
  <samples>/vocalset/<singer>/<context>/<tech>/<name>.flac   lossless copy (44.1 kHz, 16 bit)
  <samples>/vocalset/f0.npz      harvest f0 per file, 5 ms hop, float32 Hz (0 = unvoiced)
  <samples>/vocalset/index.json  singers, files, note units, legato transitions, inhalations

Note units are found from the pitch track only (no score alignment): the f0 contour is
smoothed over 180 ms (one vibrato cycle), quantised to the file's own tuning, and runs of
one semitone that last >= 120 ms become notes. Everything here is classic DSP (WORLD
harvest for f0); no model is trained or used.
"""
from __future__ import annotations

import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from .._paths import SAMPLES

warnings.filterwarnings('ignore')
OUT = SAMPLES / 'vocalset'
HOP = 0.005
SUBSET = {('long_tones', 'straight'), ('long_tones', 'pp'), ('long_tones', 'forte'), ('long_tones', 'messa'),
          ('scales', 'straight'), ('scales', 'breathy'), ('scales', 'vibrato'), ('scales', 'slow_piano'),
          ('arpeggios', 'straight'), ('arpeggios', 'breathy'), ('arpeggios', 'vibrato'),
          ('arpeggios', 'slow_piano'), ('excerpts', 'straight'), ('excerpts', 'vibrato')}
VOICE_TYPE = {'f1': 'soprano', 'f2': 'soprano', 'f3': 'soprano', 'f4': 'soprano', 'f5': 'mezzo-soprano',
              'f6': 'soprano', 'f7': 'soprano', 'f8': 'mezzo-soprano', 'f9': 'soprano',
              'm1': 'baritone', 'm2': 'tenor', 'm3': 'tenor', 'm4': 'bass(?)', 'm5': 'baritone',
              'm6': 'baritone', 'm7': 'tenor(?)', 'm8': 'bass', 'm9': 'countertenor',
              'm10': 'bass-baritone', 'm11': 'tenor'}   # readme-anon.txt of the dataset


def _frame_db(x, sr, hop_s=HOP, win_s=0.02, n_frames=None):
    hop, win = int(round(hop_s * sr)), int(round(win_s * sr))
    n = n_frames or 1 + len(x) // hop
    pad = np.pad(x, (win // 2, win + hop * n))
    idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    return 10 * np.log10(np.mean(pad[idx] ** 2, axis=1) + 1e-14)


def _band_db(x, sr, lo, hi, n_frames):
    from scipy import signal
    sos = signal.butter(4, [lo / (sr / 2), hi / (sr / 2)], btype='bandpass', output='sos')
    return _frame_db(signal.sosfilt(sos, x), sr, n_frames=n_frames)


def _runs(mask):
    """[(i0, i1_exclusive)] of True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def segment(f0, db):
    """Note units, legato links and phrase edges from an f0 track (Hz, 0 = unvoiced)."""
    from scipy import signal
    n = len(f0)
    v = f0 > 0
    midi = np.where(v, 69 + 12 * np.log2(np.where(v, f0, 1.0) / 440.0), np.nan)
    sm = np.full(n, np.nan)
    sos = signal.butter(4, 3.5 / (0.5 / HOP), output='sos')      # zero-phase 3.5 Hz: removes vibrato
    for a, b in _runs(v):
        seg = midi[a:b]
        if b - a >= 40:
            sm[a:b] = signal.sosfiltfilt(sos, seg, padlen=min(3 * 12, b - a - 1))
        else:
            sm[a:b] = np.median(seg)
    ok = np.isfinite(sm)
    if ok.sum() < 20:
        return [], 0.0, -120.0
    ref = float(np.percentile(db[v], 95)) if v.any() else float(db.max())
    loud = ok & (db > ref - 30)
    frac = sm[loud] - np.round(sm[loud]) if loud.any() else sm[ok] - np.round(sm[ok])
    off = float(np.angle(np.exp(2j * np.pi * frac).mean()) / (2 * np.pi))   # file tuning (semitones)
    q = np.where(ok, np.round(sm - off), -999).astype(int)
    cand = []
    i = 0
    while i < n:
        if q[i] == -999:
            i += 1
            continue
        j = i
        while j + 1 < n and q[j + 1] == q[i]:
            j += 1
        if (j - i + 1) >= 30:                  # >= 150 ms at one semitone
            cand.append([i, j, int(q[i])])
        i = j + 1
    merged = []                                # same semitone split by a short glitch
    for c in cand:
        if merged and c[2] == merged[-1][2] and c[0] - merged[-1][1] < 12:
            merged[-1][1] = c[1]
        else:
            merged.append(c)
    notes = []
    for i, j, qq in merged:
        a, b = i + 4, j - 3                     # trim 20 ms at each edge
        seg = midi[a:b + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) < 16:
            continue
        lvl = float(np.median(db[a:b + 1]))
        if lvl < ref - 30:                      # background / tails picked up as voicing
            continue
        vib = np.ptp(sm[a:b + 1]) < 0.5 and np.std(seg) > 0.2
        med = float(np.mean(seg)) if vib else float(np.median(seg))
        if abs(float(np.mean(sm[a:b + 1])) - (qq + off)) > 0.4:
            continue                            # a glide, not a note
        det = midi[a:b + 1] - sm[a:b + 1]
        det = det[np.isfinite(det)]
        ext = float(np.percentile(np.abs(det), 95) * 100) if len(det) else 0.0
        rate = 0.0
        if len(det) > 60:
            spec = np.abs(np.fft.rfft(det * np.hanning(len(det)), 4096))
            fr = np.fft.rfftfreq(4096, HOP)
            band = (fr > 3) & (fr < 9)
            rate = float(fr[band][np.argmax(spec[band])])
        notes.append(dict(i0=int(i), i1=int(j), a=int(a), b=int(b), midi=round(med, 3), q=qq,
                          vib_cents=round(ext, 1), vib_hz=round(rate, 2), level_db=round(lvl, 2)))
    floor = float(np.percentile(db, 5))
    for k, nt in enumerate(notes):
        prev = notes[k - 1] if k else None
        nxt = notes[k + 1] if k + 1 < len(notes) else None
        nt['prev'] = k - 1 if (prev is not None and nt['i0'] - prev['i1'] < 50
                               and v[prev['i1']:nt['i0'] + 1].mean() > 0.7) else None
        nt['next'] = k + 1 if (nxt is not None and nxt['i0'] - nt['i1'] < 50
                               and v[nt['i1']:nxt['i0'] + 1].mean() > 0.7) else None
    for k, nt in enumerate(notes):
        lo = notes[k - 1]['i1'] + 1 if k else 0
        hi = notes[k + 1]['i0'] - 1 if k + 1 < len(notes) else n - 1
        s = nt['i0']
        if nt['prev'] is None:                  # walk back through the attack (max 0.5 s)
            while s > lo and (v[s - 1] or db[s - 1] > floor + 18) and nt['i0'] - s < 100:
                s -= 1
        nt['on'] = int(s)
        nt['onset_natural'] = bool(nt['prev'] is None and s > lo and not v[s - 1] and db[s - 1] <= floor + 18)
        e = nt['i1']
        if nt['next'] is None:
            while e < hi and (v[e + 1] or db[e + 1] > floor + 18) and e - nt['i1'] < 100:
                e += 1
        nt['off'] = int(e)
        nt['release_natural'] = bool(nt['next'] is None and e < hi and not v[e + 1] and db[e + 1] <= floor + 18)
    return notes, off, ref


def find_breaths(x, sr, f0, db):
    """Inhalations: unvoiced, broadband 0.4-6 kHz noise bursts right before a sung onset."""
    n = len(f0)
    v = f0 > 0
    band = _band_db(x, sr, 400.0, 6000.0, n)
    lowb = _band_db(x, sr, 60.0, 300.0, n)
    uv = ~v
    floor = float(np.percentile(band[uv], 10)) if uv.any() else float(np.percentile(band, 5))
    out = []
    for a, b in _runs(uv):
        if b - a < 30 or b >= n:                 # gap must end in a voiced onset
            continue
        hot = band[a:b] > floor + 9
        # take the last hot run before the onset
        runs = [(a + p, a + q) for p, q in _runs(hot) if q - p >= 20]
        if not runs:
            continue
        p, q = runs[-1]
        if b - q > 100 or q - p > 240:           # must end < 0.5 s before singing, last < 1.2 s
            continue
        seg = x[int(p * HOP * sr):int(q * HOP * sr)]
        if len(seg) < 256:
            continue
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        fr = np.fft.rfftfreq(len(seg), 1 / sr)
        cen = float((spec * fr).sum() / (spec.sum() + 1e-20))
        low_ratio = float(np.median(lowb[p:q]) - np.median(band[p:q]))
        if not (700 < cen < 6000) or low_ratio > 3:    # not a thump / handling noise
            continue
        out.append(dict(t0=round(p * HOP, 3), t1=round(q * HOP, 3), peak_db=round(float(band[p:q].max()), 1),
                        floor_db=round(floor, 1), centroid_hz=round(cen), before_onset_s=round((b - q) * HOP, 3)))
    return out


def process(args):
    src, rel = args
    import pyworld as pw
    from .._dsp import resample_sr
    x, sr = sf.read(src, dtype='float64')
    dst = OUT / rel.with_suffix('.flac')
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        sf.write(dst, x, sr, subtype='PCM_16', format='FLAC')
    x48 = resample_sr(x, sr, 48000).astype(np.float64)
    f0, _ = pw.harvest(x48, 48000, f0_floor=65.0, f0_ceil=1300.0, frame_period=HOP * 1000)
    db = _frame_db(x48, 48000, n_frames=len(f0))
    notes, off, ref = segment(f0, db)
    br = find_breaths(x48, 48000, f0, db)
    for b in br:
        b['rel_db'] = round(b['peak_db'] - ref, 1)
    return str(rel.with_suffix('.flac')), f0.astype(np.float32), notes, (off, ref), br, len(x) / sr


def parse(rel: Path):
    p = rel.parts          # singer_dir, ctx, tech, name
    singer = p[0].replace('female', 'f').replace('male', 'm')
    name = rel.stem
    vowel = name.rsplit('_', 1)[1] if p[1] != 'excerpts' else None
    key = None
    if '_c_' in name:
        key = 'c'
    elif '_f_' in name:
        key = 'f'
    return dict(singer=singer, ctx={'long_tones': 'long', 'scales': 'scale', 'arpeggios': 'arp',
                                    'excerpts': 'excerpt'}[p[1]], tech=p[2], vowel=vowel, key=key)


def main(raw_dir, jobs=12):
    raw = Path(raw_dir) / 'FULL'
    todo = []
    for w in sorted(raw.rglob('*.wav')):
        rel = w.relative_to(raw)
        if (rel.parts[1], rel.parts[2]) in SUBSET:
            todo.append((str(w), rel))
    print(len(todo), 'files', flush=True)
    files, f0s, units, breaths = {}, {}, [], []
    with ProcessPoolExecutor(jobs) as ex:
        for k, (rel, f0, notes, off, br, dur) in enumerate(ex.map(process, todo, chunksize=4)):
            meta = parse(Path(rel))
            files[rel] = dict(meta, dur=round(dur, 3), tuning_semitones=round(off[0], 3), ref_db=round(off[1], 1))
            f0s[rel] = f0
            base = len(units)
            for nt in notes:
                u = dict(file=rel, **{k2: meta[k2] for k2 in ('singer', 'ctx', 'tech', 'vowel')},
                         t_on=round(nt['on'] * HOP, 3), t0=round(nt['a'] * HOP, 3), t1=round(nt['b'] * HOP, 3),
                         t_off=round(nt['off'] * HOP, 3), midi=nt['midi'], vib_cents=nt['vib_cents'],
                         vib_hz=nt['vib_hz'], level_db=nt['level_db'], onset_natural=nt['onset_natural'],
                         release_natural=nt['release_natural'],
                         prev=None if nt['prev'] is None else base + nt['prev'],
                         next=None if nt['next'] is None else base + nt['next'])
                units.append(u)
            for b in br:
                breaths.append(dict(file=rel, singer=meta['singer'], **b))
            if k % 100 == 0:
                print(k, rel, len(notes), 'notes', len(br), 'breaths', flush=True)
    singers = {s: dict(gender='female' if s[0] == 'f' else 'male', voice_type=VOICE_TYPE[s]) for s in VOICE_TYPE}
    for s in singers:
        ms = [u['midi'] for u in units if u['singer'] == s and u['ctx'] != 'excerpt']
        if ms:
            singers[s]['range_midi'] = [round(float(np.percentile(ms, 2)), 1), round(float(np.percentile(ms, 98)), 1)]
    idx = dict(source='VocalSet 1.1, Zenodo 10.5281/zenodo.1442513 (record 1442513), CC BY 4.0',
               hop_s=HOP, singers=singers, files=files, units=units, breaths=breaths)
    (OUT / 'index.json').write_text(json.dumps(idx, ensure_ascii=False, separators=(',', ':')))
    np.savez_compressed(OUT / 'f0.npz', **{k.replace('/', '|'): v for k, v in f0s.items()})
    print('units', len(units), 'breaths', len(breaths), 'files', len(files))


def refine_breaths():
    """Second pass: keep only inhalations that are really unvoiced noise.

    The first pass finds loud unvoiced stretches before onsets; some of them are breathy
    singing that harvest did not mark as voiced. Here every candidate must have a low
    periodicity (McLeod NSDF clarity, median < 0.45 over 20 ms frames) and a flat-ish
    spectrum (spectral flatness 0.5-8 kHz >= 0.08); the kept region is tightened to the
    frames that pass, and the breath's own RMS is stored for level matching.
    """
    from ..analysis import pitch_track
    from .._dsp import resample_sr
    idx = json.loads((OUT / 'index.json').read_text())
    kept = []
    for b in idx['breaths']:
        x, sr = sf.read(OUT / b['file'], dtype='float64')
        a, e = int(b['t0'] * sr), int(b['t1'] * sr)
        seg = resample_sr(x[a:e], sr, 48000).astype(np.float64)
        if len(seg) < int(0.12 * 48000):
            continue
        _, f0, clar = pitch_track(seg, fmin=70, fmax=1000, hop_s=0.01, win_s=0.03)
        n = int(0.02 * 48000)
        fr = [seg[i:i + n] for i in range(0, len(seg) - n, n // 2)]
        flat = []
        for w in fr:
            S = np.abs(np.fft.rfft(w * np.hanning(n))) ** 2
            f = np.fft.rfftfreq(n, 1 / 48000)
            band = S[(f > 500) & (f < 8000)] + 1e-20
            flat.append(float(np.exp(np.mean(np.log(band))) / np.mean(band)))
        med_clar = float(np.median(clar)) if len(clar) else 1.0
        med_flat = float(np.median(flat)) if flat else 0.0
        if med_clar < 0.45 and med_flat >= 0.08:
            kept.append(dict(b, clarity=round(med_clar, 3), flatness=round(med_flat, 3),
                             rms_db=round(float(10 * np.log10(np.mean(seg ** 2) + 1e-20)), 1)))
    idx['breaths_first_pass'] = len(idx['breaths'])
    idx['breaths'] = kept
    (OUT / 'index.json').write_text(json.dumps(idx, ensure_ascii=False, separators=(',', ':')))
    print('breaths kept', len(kept), 'of', idx['breaths_first_pass'])


def fix_edges():
    """Third pass: move each unit's stable start / end to where the level is within 6 dB of the
    note's median level (the pitch run alone put some attacks inside the 'stable' part)."""
    from .._dsp import resample_sr
    idx = json.loads((OUT / 'index.json').read_text())
    by_file = {}
    for i, u in enumerate(idx['units']):
        by_file.setdefault(u['file'], []).append(i)
    moved = 0
    for f, ids in by_file.items():
        x, sr = sf.read(OUT / f, dtype='float64')
        x48 = resample_sr(x, sr, 48000).astype(np.float64)
        db = _frame_db(x48, 48000)
        for i in ids:
            u = idx['units'][i]
            a, b = int(round(u['t0'] / HOP)), int(round(u['t1'] / HOP))
            seg = db[a:b + 1]
            if len(seg) < 12:
                continue
            med = float(np.median(seg))
            ok = np.flatnonzero(seg >= med - 6)
            if not len(ok):
                continue
            na, nb = a + int(ok[0]), a + int(ok[-1])
            if nb - na < 12:
                continue
            if u['prev'] is None and na > a:
                u['t0'] = round(na * HOP, 3)
                moved += 1
            if u['next'] is None and nb < b:
                u['t1'] = round(nb * HOP, 3)
                moved += 1
            u['level_db'] = round(float(np.median(db[na:nb + 1])), 2)
    idx['edges_fixed'] = True
    (OUT / 'index.json').write_text(json.dumps(idx, ensure_ascii=False, separators=(',', ':')))
    print('edges moved', moved)



def build(raw_dir, jobs=12):
    """The three passes in the order they were run for the promo film."""
    OUT.mkdir(parents=True, exist_ok=True)
    main(raw_dir, jobs)
    refine_breaths()
    fix_edges()
