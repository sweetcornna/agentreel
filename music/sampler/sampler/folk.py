"""Folk instruments, a grand piano and hand percussion from VCSL (CC0).

    render(set_id, notes, ...)      -> (n, 2) float32   pitched: strumstick, folk_harp, marimba,
                                                        balafon, dan_tranh, grand_piano
    hit(set_id, kind, vel, seed)    -> (n, 2) float32   one percussion hit, its perceived onset at
                                                        index `lead(set_id, kind)` (see below)
    kinds(set_id)                   -> the hit kinds of a percussion set

notes : [(start_s, dur_s, midi, velocity[, {kw}])]  velocity 1..127; kw: pan, ring_s, choke
Sampler (classic): the nearest MEASURED pitch (<samples>/vcsl/index_folk.json, written by
sampler/prepare/vcsl.py; VCSL's file names for the folk sets are an octave low, the grand piano's are
not - measured), varispeed to the
exact pitch (band-limited polyphase, 44.1 -> 48 kHz in the same step), dynamic layers crossfaded
by velocity, round-robin where recorded. Levels are evened out per layer across the range (60 %
of the difference to the layer median, at most +-6 dB), so the recorded dynamics survive but one
loud or quiet sample does not stick out. The layer medians are those of the COMPLETE set
(sampler/data/vcsl_ref_levels.json, copied into the index by the import), so a score that fetched only
the files it plays sounds exactly as with the whole set; without reference values they are computed
from the samples present.

Plucked strings ring until the next note on the same string (strumstick: its three strings; harp,
dan tranh and grand piano: the same pitch) chokes them with a 25 ms damp, or until `ring_s` after the
written end (default: let ring, i.e. the piano's sustain pedal is down; a small ring_s such as 0.3 is
the damper). Bars (marimba, balafon) always ring out.

Percussion hits are aligned on their perceived onset (the first moment the 2 ms envelope is
within 12 dB of its peak), not on the first sound: a shaker's swing, a hand's brush before the
slap stay in front of the beat. `hit()` returns the clip with that pre-roll; place it with
mix.place(track, clip, t - lead) or use `place_hit()`.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d

from ._dsp import SR, add_at, cos_ramp, db2a, fade_edges, pan_gains, resample_by
from ._paths import SAMPLES, log_access, need

VCSL = SAMPLES / 'vcsl'
PITCHED = ('strumstick', 'folk_harp', 'marimba', 'balafon', 'dan_tranh', 'grand_piano')
BARS = ('marimba', 'balafon')
# per set: the level (dB) a velocity-100 note is normalised to, and the extra velocity trim (dB at vel 1)
LEVEL = {'strumstick': -20.0, 'folk_harp': -20.0, 'marimba': -19.0, 'balafon': -20.0, 'dan_tranh': -21.0,
         'grand_piano': -20.0}
VEL_TRIM_DB = 14.0


@lru_cache(maxsize=1)
def _index_doc():
    return json.loads(need(VCSL / 'index_folk.json', 'VCSL 民乐 / 钢琴索引').read_text())


def _index():
    return _index_doc()['samples']


def _ref(set_id):
    """reference statistics of the complete set ({} when the index carries none)"""
    return (_index_doc().get('ref') or {}).get(set_id) or {}


def _rows(set_id):
    rows = [r for r in _index() if r['set'] == set_id]
    if not rows:
        raise ValueError(f'no VCSL samples for {set_id!r}')
    return rows


@lru_cache(maxsize=512)
def _raw(file):
    log_access(VCSL / file)
    x, sr = sf.read(need(VCSL / file), dtype='float32', always_2d=True)
    x = x[:, :2] if x.shape[1] >= 2 else np.repeat(x, 2, axis=1)
    return x, sr


@lru_cache(maxsize=1024)
def _playable(file, ratio_q):
    x, sr = _raw(file)
    return resample_by(x, ratio_q / 1e6 * sr / SR)


@lru_cache(maxsize=16)
def _layer_gain(set_id):
    """file -> gain (dB) that evens out a sample against the median of its dynamic layer."""
    rows = _rows(set_id)
    ref = _ref(set_id).get('layer_median_db', {})
    out = {}
    for d in {r['dyn'] for r in rows}:
        lv = [r['level_db'] for r in rows if r['dyn'] == d]
        med = float(ref[str(d)]) if str(d) in ref else float(np.median(lv))
        for r in rows:
            if r['dyn'] == d:
                out[r['file']] = float(np.clip(0.6 * (med - r['level_db']), -6.0, 6.0))
    return out


@lru_cache(maxsize=16)
def _set_ref(set_id):
    """gain (dB) that brings the loudest layer's median level to LEVEL[set_id]"""
    ref = _ref(set_id)
    if 'top_median_db' in ref:
        return LEVEL.get(set_id, -20.0) - float(ref['top_median_db'])
    rows = _rows(set_id)
    top = max(r['dyn'] for r in rows)
    med = float(np.median([r['level_db'] for r in rows if r['dyn'] == top]))
    return LEVEL.get(set_id, -20.0) - med


def _pick(set_id, m, vel, rr_state):
    rows = [r for r in _rows(set_id) if r.get('pitch') is not None]
    pitches = np.array(sorted({r['pitch'] + r.get('tune_cents', 0.0) / 100 for r in rows}))
    # nearest, preferring to transpose DOWN a little (brighter source) when two are as near
    d = pitches - m
    k = int(np.argmin(np.abs(d) - 0.01 * (d > 0)))
    p_true = pitches[k]
    at = [r for r in rows if abs(r['pitch'] + r.get('tune_cents', 0.0) / 100 - p_true) < 1e-6]
    dyns = sorted({r['dyn'] for r in at})
    x = np.clip((vel - 25) / 95.0, 0, 1) * (len(dyns) - 1)
    lo, hi = int(np.floor(x)), int(np.ceil(x))
    w_hi = x - lo
    out = []
    for dd, w in ((dyns[lo], 1 - w_hi), (dyns[hi], w_hi)):
        if w <= 1e-3:
            continue
        cands = sorted([r for r in at if r['dyn'] == dd], key=lambda r: r['rr'])
        key = (set_id, round(p_true, 2), dd)
        i = rr_state.get(key, rr_state['_seed'])
        rr_state[key] = i + 1
        out.append((cands[i % len(cands)], np.sqrt(w) if lo != hi else 1.0))
        if lo == hi:
            break
    return float(p_true), out


def _note(set_id, r, m, p_true, pre_s=0.004):
    """sample r at pitch m, starting pre_s before its onset, at the set's reference level"""
    ratio = 2.0 ** ((m - p_true) / 12.0)
    x = _playable(r['file'], int(round(ratio * 1e6)))
    on = int(round(r['onset_s'] * SR / ratio))      # the onset in the varispeeded 48 kHz clip
    a = max(0, on - int(pre_s * SR))
    y = x[a:]
    g = db2a(_set_ref(set_id) + _layer_gain(set_id).get(r['file'], 0.0))
    return (y * g).astype(np.float32), on - a


def _choke_group(set_id, r, m):
    if set_id == 'strumstick':
        return r['kind']                 # Str1 / Str2 / Str3
    if set_id in ('folk_harp', 'dan_tranh', 'grand_piano'):
        return int(round(m))
    return None


def render(set_id, notes, *, duration=None, ring_s=None, damp_s=0.025, pan=0.0, width=1.0, seed=0,
           gain_db=0.0):
    """Render pitched notes of one VCSL set (see module doc). Dry: add sampler.mix.reverb."""
    if set_id not in PITCHED:
        raise ValueError(f'{set_id!r} is not pitched: {PITCHED}')
    rr_state = {'_seed': int(seed)}
    evs = []
    for n in sorted(notes, key=lambda n: n[0]):
        st, du, m = float(n[0]), float(n[1]), float(n[2])
        vel = float(n[3]) if len(n) > 3 else 80.0
        kw = n[4] if len(n) > 4 and isinstance(n[4], dict) else {}
        if vel <= 0:
            continue
        evs.append((st, du, m, float(np.clip(vel, 1, 127)), kw))
    if not evs:
        return np.zeros((int(round((duration or 0.1) * SR)), 2), np.float32)
    total = max(e[0] + e[1] for e in evs) + 8.0
    n_out = int(round((duration if duration is not None else total) * SR))
    out = np.zeros((n_out, 2), np.float32)
    # plan: each note's clip, its start, and when its string is taken by the next note
    plan = []
    for st, du, m, vel, kw in evs:
        p_true, layers = _pick(set_id, m, vel, rr_state)
        clip, lead = None, 0
        for r, w in layers:
            y, ld = _note(set_id, r, m, p_true)
            if clip is None:
                clip, lead = y * w, ld
            else:
                # align the second layer on the same onset
                sh = ld - lead
                y = y[sh:] if sh > 0 else np.concatenate([np.zeros((-sh, 2), np.float32), y])
                k = min(len(clip), len(y))
                clip = clip[:k] + y[:k] * w
        trim = db2a(gain_db - VEL_TRIM_DB * (1 - vel / 127.0) ** 1.5)
        grp = kw.get('choke', _choke_group(set_id, layers[0][0], m))
        plan.append(dict(st=st, du=du, clip=clip * trim, lead=lead, grp=grp,
                         ring=kw.get('ring_s', ring_s), pan=kw.get('pan', pan)))
    for i, e in enumerate(plan):
        y = e['clip']
        end = None
        if e['grp'] is not None and set_id not in BARS:
            nxt = next((f for f in plan[i + 1:] if f['grp'] == e['grp']), None)
            if nxt is not None:
                end = nxt['st'] - e['st']
        if e['ring'] is not None and set_id not in BARS:
            end = min(end, e['du'] + e['ring']) if end is not None else e['du'] + e['ring']
        if end is not None:
            k = e['lead'] + int(round(max(0.0, end) * SR))
            f = int(round((damp_s if end < e['du'] + (e['ring'] or 0) + 1e-6 else 0.12) * SR))
            if k < len(y):
                y = y[:k + f].copy()
                y[k:k + f] *= cos_ramp(min(f, len(y) - k), rising=False)[:, None]
        y = fade_edges(y, 0.0005, 0.02)
        gl, gr = pan_gains(float(np.clip(e['pan'], -1, 1)))
        if width != 1.0:
            mid = y.mean(axis=1, keepdims=True)
            y = mid + (y - mid) * width
        y = y * np.array([gl, gr], np.float32) * np.sqrt(2)
        add_at(out, y.astype(np.float32), int(round(e['st'] * SR)) - e['lead'])
    return out


# ----------------------------------------------------------------------------- percussion
PERC_LEVEL = -18.0      # peak (dBFS) of a velocity-127 hit of the loudest layer


def kinds(set_id):
    return sorted({r['kind'] for r in _rows(set_id)})


@lru_cache(maxsize=512)
def _hit_raw(file):
    x, sr = _raw(file)
    if sr != SR:
        x = resample_by(x, sr / SR)
    env = maximum_filter1d(np.abs(x).max(axis=1), max(1, int(0.002 * SR)))
    pk = float(env.max())
    k = int(np.argmax(env >= pk * db2a(-12.0)))
    first = int(np.argmax(env >= pk * db2a(-45.0)))
    a = max(0, first - int(0.002 * SR))
    # tail: to -70 dB re peak
    above = np.flatnonzero(env > pk * db2a(-70.0))
    b = min(len(x), int(above[-1]) + int(0.02 * SR)) if len(above) else len(x)
    # a recording that starts right on its attack gets a 2 ms fade in, not a step
    y = fade_edges(x[a:b], 0.002 if k - a < int(0.002 * SR) else 0.001, 0.015)
    return y, k - a, pk


@lru_cache(maxsize=64)
def _kind_peak(set_id, kind):
    ref = _ref(set_id).get('kind_peak', {})
    if kind in ref:
        return float(ref[kind])
    rows = [r for r in _rows(set_id) if r['kind'] == kind]
    top = max(r['dyn'] for r in rows)
    return float(np.median([_hit_raw(r['file'])[2] for r in rows if r['dyn'] == top]))


def _pick_hit(set_id, kind, vel, rr):
    rows = [r for r in _rows(set_id) if r['kind'] == kind]
    if not rows:
        raise ValueError(f'{set_id}: no kind {kind!r}; kinds: {kinds(set_id)}')
    dyns = sorted({r['dyn'] for r in rows})
    x = np.clip((vel - 20) / 100.0, 0, 1) * (len(dyns) - 1)
    d = dyns[int(round(x))]
    cands = sorted([r for r in rows if r['dyn'] == d], key=lambda r: (r['rr'], r['file']))
    return cands[rr % len(cands)]


def hit(set_id, kind, vel=100, seed=0, *, gain_db=0.0, pitch=0.0):
    """One hit: (clip, lead) where lead is the index of the perceived onset in the clip.
    vel 1..127 picks the layer (and a velocity trim); seed picks the round-robin; pitch in
    semitones varispeeds the hit (a hand lands a little differently each time)."""
    r = _pick_hit(set_id, kind, vel, seed)
    y, lead, pk = _hit_raw(r['file'])
    ref = _kind_peak(set_id, kind)
    g = db2a(PERC_LEVEL - 20 * np.log10(ref + 1e-12) + gain_db - 18.0 * (1 - vel / 127.0) ** 1.6)
    y = y * g
    if pitch:
        ratio = 2.0 ** (pitch / 12.0)
        y = resample_by(y, ratio)
        lead = int(round(lead / ratio))
    return y.astype(np.float32), lead


def place_hit(track, set_id, kind, at, vel=100, seed=0, *, gain_db=0.0, pan=0.0, pitch=0.0, length=None):
    """Mix one hit into a track so that its perceived onset lands on `at` (s). length (s): stop a roll or a
    long ring that long after the onset, with an 80 ms fade."""
    y, lead = hit(set_id, kind, vel, seed, gain_db=gain_db, pitch=pitch)
    if length is not None:
        k = lead + int(round(length * SR))
        if k < len(y):
            f = min(int(0.08 * SR), len(y) - k)
            y = y[:k + f].copy()
            y[k:] *= cos_ramp(f, rising=False)[:, None]
    gl, gr = pan_gains(float(np.clip(pan, -1, 1)))
    y = y * np.array([gl, gr], np.float32) * np.sqrt(2)
    start = int(round(at * SR)) - lead
    need = start + len(y)
    if need > len(track):
        track = np.concatenate([track, np.zeros((need - len(track), 2), np.float32)])
    add_at(track, y, start)
    return track
