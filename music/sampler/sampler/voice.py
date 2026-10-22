"""Real sung voice: a concatenative sampler over VocalSet recordings, resynthesised with WORLD.

    render_voice(notes, voices=1, ...)          -> (n, 2) float32
    tintinnabuli(melody, triad='D', position=1)  -> T-voice notes (Arvo Pärt's rule)
    tintinnabuli_pair(melody, ...)               -> M + T notes, ready for render_voice
    SINGERS                                      -> {id: {gender, voice_type, range_midi}}

Source: VocalSet 1.1 (Wilkins, Seetharaman, Wahl & Pardo, ISMIR 2018; Zenodo 1442513;
CC BY 4.0): 20 professional singers (9 female, 11 male) singing long tones, scales and
arpeggios on a e i o u, straight / breathy / vibrato / soft legato. sampler/prepare/vocalset.py
cut every recording into note units (stable part, natural onset, natural release, and the
real legato transition to the next note) from its WORLD-harvest pitch track.

How a line is rendered (all classic DSP; no neural or generative model anywhere):
1. Unit selection per phrase (Viterbi): for each target note the candidates are that singer's
   units with the same vowel and a matching technique. Cost = pitch shift (semitones),
   too-short stable part, technique / vibrato mismatch, velocity-vs-dynamics mismatch, a bonus
   for a natural onset on the first note and a natural release on the last, and a large bonus
   when two consecutive target notes can be taken from two consecutive notes of the same
   recording with the same interval (then the singer's REAL transition is used).
2. The recording is analysed with WORLD (CheapTrick spectral envelope, D4C aperiodicity,
   5 ms frames) on the singer's own harvest f0. The phrase is rebuilt in the parameter domain:
   - pitch: the target pitch plus the singer's own deviation from that note's centre (drift,
     jitter, vibrato: kept, not modelled);
   - formants: the spectral envelope is not moved, so pitch changes keep the singer's formants;
   - longer notes than the recording: the stable part is read forwards, then back and forth
     at the recorded speed (no speed-up / slow-down of the vibrato);
   - legato: either the real transition (step 1) or a parameter crossfade whose pitch glide has
     the average shape of the dataset's own transitions (median of ~13,000, per direction);
   - onset / release: natural ones from the recording when available, otherwise a soft
     attack with a small scoop and a breathy decay;
   - optional phrase-end fall, re-articulation dips on repeated notes;
   - real inhalations from the same singer between phrases (``breaths='auto'``).
3. WORLD synthesis at 48 kHz; the envelope below 0.85 f0 is tapered (WORLD otherwise leaks
   noise below the fundamental).

A choir (``voices > 1``) gives every line ``voices`` DIFFERENT singers (no sample is copied),
each with their own unit choices, timing and pan.
"""
from __future__ import annotations

import json
import warnings
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import soundfile as sf

from ._dsp import SR, db2a, fade_edges, pan_gains, resample_sr
from ._paths import SAMPLES, log_access, need

warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

VS = SAMPLES / 'vocalset'
HOP = 0.005                     # WORLD frame period (s), same as the index
FFT = 2048
NB = FFT // 2 + 1
VEL_REF_DB = -13.9              # unit RMS at vel 127; vel 100 -> ~ -18 dBFS (as sampler.choir)
VOWEL_TRIM_DB = {'a': 0.0, 'o': -1.0, 'e': -1.0, 'i': -1.5, 'u': -2.5, 'm': -4.0}
ALIASES = {'ah': 'a', 'aa': 'a', 'oh': 'o', 'oo': 'u', 'ee': 'i', 'eh': 'e', 'mm': 'm', 'hum': 'm',
           'n': 'm', '嗯': 'm', '啊': 'a', '哦': 'o', '噢': 'o', '呜': 'u', '乌': 'u'}
# technique -> [(context, VocalSet technique, cost)]; context long = long tones, scale, arp
TECH_SOURCES = {
    'straight': [('long', 'straight', 0.0), ('scale', 'straight', 0.1), ('arp', 'straight', 0.1),
                 ('long', 'pp', 0.45), ('scale', 'slow_piano', 0.7), ('arp', 'slow_piano', 0.7)],
    'breathy': [('scale', 'breathy', 0.0), ('arp', 'breathy', 0.0), ('long', 'pp', 1.2)],
    'vibrato': [('long', 'forte', 0.0), ('long', 'messa', 0.6), ('scale', 'slow_piano', 0.15),
                ('arp', 'slow_piano', 0.15), ('scale', 'vibrato', 0.5), ('arp', 'vibrato', 0.5)],
}
TECHS = tuple(TECH_SOURCES)
# vibrato extent (95th pct |deviation|, cents) each technique expects
VIB_RANGE = {'straight': (0, 35), 'breathy': (0, 45), 'vibrato': (40, 90)}
# the stable part's own pitch deviation is scaled down to at most this extent (cents, 95th pct):
# VocalSet's 'molto vibrato' and some legato scales swing +-130 cents; the score wants a light one
VIB_CAP = {'straight': 35.0, 'breathy': 40.0, 'vibrato': 55.0}


# ----------------------------------------------------------------------------- index
class _Index:
    def __init__(self):
        d = json.loads((VS / 'index.json').read_text())
        self.singers = d['singers']
        self.files = list(d['files'])
        self.file_meta = d['files']
        fid = {f: i for i, f in enumerate(self.files)}
        U = d['units']
        self.U = U
        self.n = len(U)
        g = lambda k: np.array([u[k] for u in U])                      # noqa: E731
        self.singer = g('singer')
        self.ctx = g('ctx')
        self.tech = g('tech')
        self.vowel = np.array([u['vowel'] or '' for u in U])
        self.midi = g('midi').astype(float)
        self.t_on, self.t0, self.t1, self.t_off = (g(k).astype(float) for k in ('t_on', 't0', 't1', 't_off'))
        self.onset_nat = g('onset_natural').astype(bool)
        self.release_nat = g('release_natural').astype(bool)
        self.next = np.array([-1 if u['next'] is None else u['next'] for u in U])
        self.prev = np.array([-1 if u['prev'] is None else u['prev'] for u in U])
        self.level = g('level_db').astype(float)
        self.vib = g('vib_cents').astype(float)
        self.file = np.array([fid[u['file']] for u in U])
        self.breaths = d['breaths']
        self._f0 = None

    def f0(self, file_id):
        if self._f0 is None:
            self._f0 = np.load(VS / 'f0.npz')
        return self._f0[self.files[file_id].replace('/', '|')].astype(np.float64)


@lru_cache(maxsize=1)
def _index() -> _Index:
    need(VS / 'index.json', 'VocalSet 音符单元索引（fetch_samples.py --only vocalset 会生成）')
    return _Index()


def _singers():
    return _index().singers


class _LazySingers(dict):
    def _fill(self):
        if not dict.__len__(self):
            self.update(_singers())

    def __getitem__(self, k):
        self._fill()
        return dict.__getitem__(self, k)

    def __iter__(self):
        self._fill()
        return dict.__iter__(self)

    def __len__(self):
        self._fill()
        return dict.__len__(self)

    def items(self):
        self._fill()
        return dict.items(self)

    def keys(self):
        self._fill()
        return dict.keys(self)

    def __repr__(self):
        self._fill()
        return dict.__repr__(self)


SINGERS = _LazySingers()


# ----------------------------------------------------------------------------- audio + WORLD
@lru_cache(maxsize=96)
def _audio(file_id: int) -> np.ndarray:
    idx = _index()
    log_access(VS / idx.files[file_id])
    x, sr = sf.read(need(VS / idx.files[file_id]), dtype='float64')
    if x.ndim == 2:
        x = x.mean(axis=1)
    return resample_sr(x, sr, SR).astype(np.float64)


_CHUNK = 200                    # frames per analysis chunk (1 s)
_CACHE: 'OrderedDict[tuple, tuple]' = OrderedDict()
_CACHE_MAX = 700                # chunks (~1.1 GB of float32 sp + ap)


def _chunk(file_id: int, c: int):
    key = (file_id, c)
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit
    import pyworld as pw
    idx = _index()
    f0 = idx.f0(file_id)
    x = _audio(file_id)
    lo = c * _CHUNK
    hi = min(len(f0), lo + _CHUNK)
    if hi <= lo:
        z = (np.full((1, NB), 1e-12, np.float32), np.ones((1, NB), np.float32))
        return z
    t = np.arange(lo, hi) * HOP
    ff = np.ascontiguousarray(f0[lo:hi])
    tt = np.ascontiguousarray(t)
    sp = pw.cheaptrick(x, ff, tt, SR, f0_floor=71.0, fft_size=FFT).astype(np.float32)
    ap = pw.d4c(x, ff, tt, SR, fft_size=FFT).astype(np.float32)
    _CACHE[key] = (sp, ap)
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return sp, ap


def _frames(file_id: int, pos: np.ndarray):
    """sp, ap (float64) at fractional frame positions (log-linear interpolation for sp)."""
    idx = _index()
    nf = len(idx.f0(file_id))
    pos = np.clip(np.asarray(pos, dtype=np.float64), 0, nf - 1.001)
    i0 = np.floor(pos).astype(int)
    fr = pos - i0
    need = np.unique(np.concatenate([i0, i0 + 1]))
    need = need[need < nf]
    table_sp = {}
    table_ap = {}
    for c in np.unique(need // _CHUNK):
        sp, ap = _chunk(file_id, int(c))
        table_sp[c], table_ap[c] = sp, ap

    def get(ix):
        ix = np.minimum(ix, nf - 1)
        c = ix // _CHUNK
        o = ix - c * _CHUNK
        sp = np.empty((len(ix), NB))
        ap = np.empty((len(ix), NB))
        for cc in np.unique(c):
            m = c == cc
            tsp, tap = table_sp.get(cc), table_ap.get(cc)
            if tsp is None:
                tsp, tap = _chunk(file_id, int(cc))
            oo = np.minimum(o[m], len(tsp) - 1)
            sp[m] = tsp[oo]
            ap[m] = tap[oo]
        return sp, ap
    sa, aa = get(i0)
    sb, ab = get(i0 + 1)
    w = fr[:, None]
    sp = np.exp((1 - w) * np.log(sa + 1e-20) + w * np.log(sb + 1e-20))
    ap = (1 - w) * aa + w * ab
    return sp, ap


def _src_cents(file_id: int, pos: np.ndarray):
    """Source pitch in cents (MIDI*100) at fractional positions; NaN where unvoiced."""
    f0 = _index().f0(file_id)
    pos = np.clip(pos, 0, len(f0) - 1.001)
    i0 = np.floor(pos).astype(int)
    fr = pos - i0
    a, b = f0[i0], f0[np.minimum(i0 + 1, len(f0) - 1)]
    ok = (a > 0) & (b > 0)
    c = np.full(len(pos), np.nan)
    c[ok] = 100 * (69 + 12 * np.log2(((1 - fr[ok]) * a[ok] + fr[ok] * b[ok]) / 440.0))
    ok1 = (a > 0) & ~ok
    c[ok1] = 100 * (69 + 12 * np.log2(a[ok1] / 440.0))
    ok2 = (b > 0) & ~ok & ~ok1
    c[ok2] = 100 * (69 + 12 * np.log2(b[ok2] / 440.0))
    return c


# ----------------------------------------------------------------------------- transition shape
@lru_cache(maxsize=1)
def glide_templates():
    """Median normalised pitch path of the dataset's real legato transitions.

    For every pair of consecutive note units (interval 1-5 semitones) the smoothed pitch across
    [end of stable A - 40 ms, start of stable B + 40 ms] is normalised to 0 (note A) .. 1
    (note B) and resampled to 64 points; the median over all pairs, separately for rising and
    falling steps, is the template. Also returns the median transition duration.
    """
    from scipy.ndimage import uniform_filter1d
    idx = _index()
    out = {}
    durs = {}
    for sign in (1, -1):
        curves, dd = [], []
        for u in range(idx.n):
            v = idx.next[u]
            if v < 0 or idx.ctx[u] == 'excerpt':
                continue
            iv = idx.midi[v] - idx.midi[u]
            if not (1 <= sign * iv <= 5.5):
                continue
            f0 = idx.f0(idx.file[u])
            a = int(round(idx.t1[u] / HOP)) - 8
            b = int(round(idx.t0[v] / HOP)) + 8
            if b - a < 16 or a < 0 or b > len(f0):
                continue
            seg = f0[a:b]
            if (seg <= 0).mean() > 0.2:
                continue
            c = np.where(seg > 0, 69 + 12 * np.log2(np.maximum(seg, 1) / 440.0), np.nan)
            m = np.isfinite(c)
            c = np.interp(np.arange(len(c)), np.flatnonzero(m), c[m])
            c = uniform_filter1d(c, 5, mode='nearest')
            y = (c - idx.midi[u]) / iv
            curves.append(np.interp(np.linspace(0, 1, 64), np.linspace(0, 1, len(y)), y))
            dd.append((b - a - 16) * HOP)
            if len(curves) >= 4000:
                break
        cur = np.median(np.array(curves), axis=0)
        cur = (cur - cur[0]) / max(cur[-1] - cur[0], 1e-6)       # exact 0 -> 1 end points
        out[sign] = cur
        durs[sign] = float(np.median(dd))
    return out, durs


def _glide(u, sign):
    """Pitch weight 0..1 (with the template's own over/undershoot) at normalised time u."""
    tpl, _ = glide_templates()
    cur = tpl[1 if sign >= 0 else -1]
    return np.interp(np.clip(u, 0, 1), np.linspace(0, 1, len(cur)), cur)


# ----------------------------------------------------------------------------- tintinnabuli
_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def _pc(name):
    p = _PC[name[0].upper()]
    for ch in name[1:]:
        if ch == '#':
            p += 1
        elif ch == 'b':
            p -= 1
    return p % 12


def triad_pcs(triad):
    """'D' -> {2, 6, 9}; 'Bm' / 'F#m' minor; or an iterable of pitch classes / MIDI numbers."""
    if isinstance(triad, str):
        minor = triad.endswith('m') and len(triad) > 1 and triad[-2] != '#' or triad.endswith('#m')
        root = _pc(triad[:-1] if triad.endswith('m') else triad)
        return {root, (root + (3 if minor else 4)) % 12, (root + 7) % 12}
    return {int(p) % 12 for p in triad}


def t_pitch(m, triad='D', position=1):
    """Pärt's T-voice for one melody pitch: the |position|-th triad tone above (position > 0,
    'superior') or below (< 0, 'inferior') the M pitch, never the M pitch itself."""
    pcs = triad_pcs(triad)
    step = 1 if position > 0 else -1
    k, q = 0, int(round(m))
    while True:
        q += step
        if q % 12 in pcs:
            k += 1
            if k == abs(position):
                return q


def _note_parts(n):
    st, du, m = float(n[0]), float(n[1]), n[2]
    vel = float(n[3]) if len(n) > 3 else 80.0
    kw = {}
    if len(n) > 4:
        if isinstance(n[4], dict):
            kw = dict(n[4])
        elif n[4] is not None:            # old sampler.choir tuple: (.., vowel[, part])
            kw['vowel'] = n[4]
            if len(n) > 5:
                kw['part'] = n[5]
    return st, du, m, vel, kw


def tintinnabuli(melody, triad='D', position=1, mode=None, *, octave=0, vel_offset=-4, part='T'):
    """T-voice notes for an M-voice melody (Arvo Pärt's tintinnabuli).

    melody   : notes (start, dur, midi, vel[, {kw}]) - the M-voice.
    triad    : 'D', 'Bm', 'G', 'F#m' ... or pitch classes.
    position : 1 = first position (nearest triad tone), 2 = second position (skip one).
               A negative value means inferior (below the melody).
    mode     : None (from the sign of position), 'superior', 'inferior' or 'alternating'
               (superior / inferior alternate note by note).
    octave   : extra octave displacement of the T-voice (e.g. -1).
    The T note copies timing and keywords of its M note, with part=`part` and vel + vel_offset.
    """
    pos = abs(int(position)) or 1
    if mode is None:
        mode = 'inferior' if position < 0 else 'superior'
    out = []
    for i, n in enumerate(melody):
        st, du, m, vel, kw = _note_parts(n)
        if m is None:
            continue
        sgn = {'superior': 1, 'inferior': -1}.get(mode, 1 if i % 2 == 0 else -1)
        q = t_pitch(m, triad, sgn * pos) + 12 * octave
        kw2 = dict(kw, part=part)
        out.append((st, du, q, max(1.0, vel + vel_offset), kw2))
    return out


def tintinnabuli_pair(melody, triad='D', position=1, mode=None, *, octave=0, vel_offset=-4,
                      m_part='M', t_part='T'):
    """M-voice (labelled m_part) + its T-voice (t_part), for render_voice(..., voices=n)."""
    mv = []
    for n in melody:
        st, du, m, vel, kw = _note_parts(n)
        mv.append((st, du, m, vel, dict(kw, part=m_part)))
    return mv + tintinnabuli(melody, triad, position, mode, octave=octave, vel_offset=vel_offset,
                             part=t_part)


# ----------------------------------------------------------------------------- notes -> lines
def _norm_vowel(v):
    v = (v or 'a')
    parts = [ALIASES.get(p.strip(), p.strip()) for p in str(v).split('>')]
    for p in parts:
        if p not in VOWEL_TRIM_DB:
            raise ValueError(f'vowel {v!r}: use a e i o u m (hum) or a morph like "m>a"')
    return parts


def _prep(notes, tech_default, vowel_default):
    out = []
    for n in notes:
        st, du, m, vel, kw = _note_parts(n)
        if du <= 0 or vel <= 0 or m is None:
            continue
        tech = kw.get('tech', tech_default) or 'straight'
        if tech not in TECH_SOURCES:
            raise ValueError(f'tech {tech!r}: one of {TECHS}')
        out.append(dict(st=st, du=du, m=float(m), vel=float(np.clip(vel, 1, 127)),
                        vowels=_norm_vowel(kw.get('vowel', vowel_default)), tech=tech,
                        part=kw.get('part'), singer=kw.get('singer'), fall=kw.get('fall'),
                        breath=kw.get('breath'), pan=kw.get('pan'), kw=kw))
    out.sort(key=lambda x: (x['st'], -x['m']))
    return out


def _lines(notes):
    """Split into monophonic lines: per part, a note joins the line whose last note ended
    (allowing a legato overlap of 30 % of the shorter note) and is closest in pitch."""
    parts = OrderedDict()
    for n in notes:
        parts.setdefault(n['part'], []).append(n)
    lines = []
    for part, ns in parts.items():
        cur = []
        for n in ns:
            best, bd = None, 1e9
            for L in cur:
                last = L[-1]
                ov = last['st'] + last['du'] - n['st']
                if n['st'] - last['st'] < 0.05 or ov > 0.3 * min(last['du'], n['du']):
                    continue
                d = abs(last['m'] - n['m'])
                if d < bd:
                    best, bd = L, d
            if best is None:
                best = []
                cur.append(best)
            best.append(n)
        cur.sort(key=lambda L: -np.mean([x['m'] for x in L]))
        for L in cur:
            lines.append((part, L))
    return lines


def _phrases(line, legato_gap):
    ph = [[line[0]]]
    for n in line[1:]:
        last = ph[-1][-1]
        if n['st'] - (last['st'] + last['du']) <= legato_gap:
            ph[-1].append(n)
        else:
            ph.append([n])
    return ph


# ----------------------------------------------------------------------------- selection
def _cand_mask(idx, singer, vowel, tech):
    v = 'u' if vowel == 'm' else vowel
    m = (idx.singer == singer) & (idx.vowel == v)
    tm = np.zeros(idx.n, bool)
    tc = np.full(idx.n, np.inf)
    for ctx, tq, c in TECH_SOURCES[tech]:
        k = m & (idx.ctx == ctx) & (idx.tech == tq)
        tm |= k
        tc[k] = np.minimum(tc[k], c)
    return tm, tc


def _unary(idx, cand, tc, n, first, last, K=14):
    stable = idx.t1[cand] - idx.t0[cand]
    shift = np.abs(n['m'] - idx.midi[cand])
    cost = shift + 2.0 * np.maximum(shift - 3.0, 0)
    need = max(0.15, n['du'] - 0.1)
    cost += 0.55 * np.log2(np.maximum(need / np.maximum(stable, 0.08), 1.0))
    cost += tc[cand]
    lo, hi = VIB_RANGE[n['tech']]
    vib = idx.vib[cand]
    cost += 0.5 * np.maximum(vib - hi, 0) / 25 + 0.4 * np.maximum(lo - vib, 0) / 25
    if n['vel'] < 50:
        cost += 0.25 * np.isin(idx.tech[cand], ['forte'])
    elif n['vel'] > 95:
        cost += 0.25 * np.isin(idx.tech[cand], ['pp', 'slow_piano'])
    if first:
        cost -= 0.45 * idx.onset_nat[cand]
    if last:
        cost -= 0.35 * idx.release_nat[cand]
    order = np.argsort(cost)[:K]
    return cand[order], cost[order]


def _select(idx, phrase, singer, rng):
    """Viterbi over candidate units; returns [unit id] and per-note 'contiguous with next'."""
    C, UC = [], []
    for k, n in enumerate(phrase):
        tm, tc = _cand_mask(idx, singer, n['vowels'][0], n['tech'])
        cand = np.flatnonzero(tm)
        if not len(cand):                  # vowel/technique missing for this singer: any tech
            tm = (idx.singer == singer) & (idx.vowel == ('u' if n['vowels'][0] == 'm' else n['vowels'][0]))
            tm &= idx.ctx != 'excerpt'
            cand = np.flatnonzero(tm)
            tc = np.full(idx.n, 1.5)
        if not len(cand):
            raise ValueError(f'singer {singer} has no units for vowel {n["vowels"][0]}')
        c, u = _unary(idx, cand, tc, n, k == 0, k == len(phrase) - 1)
        u = u + rng.uniform(0, 0.05, len(u))          # seed-dependent tie breaking
        C.append(c)
        UC.append(u)
    # pairwise: real transition bonus, same-file bonus, re-use penalty
    D = [UC[0]]
    back = []
    for k in range(1, len(phrase)):
        prev_c, cur_c = C[k - 1], C[k]
        iv = phrase[k]['m'] - phrase[k - 1]['m']
        P = np.full((len(prev_c), len(cur_c)), 0.35)
        for a, ua in enumerate(prev_c):
            for b, ub in enumerate(cur_c):
                if idx.next[ua] == ub and abs((idx.midi[ub] - idx.midi[ua]) - iv) < 0.6 \
                        and len(phrase[k - 1]['vowels']) == 1 and len(phrase[k]['vowels']) == 1:
                    P[a, b] = -0.9
                elif idx.file[ua] == idx.file[ub]:
                    P[a, b] = 0.15 if ua != ub else 0.5
        tot = D[-1][:, None] + P + UC[k][None, :]
        bi = np.argmin(tot, axis=0)
        back.append(bi)
        D.append(tot[bi, np.arange(len(cur_c))])
    j = int(np.argmin(D[-1]))
    path = [j]
    for bi in reversed(back):
        j = int(bi[j])
        path.append(j)
    path = path[::-1]
    units = [int(C[k][path[k]]) for k in range(len(phrase))]
    contig = [bool(k + 1 < len(units) and idx.next[units[k]] == units[k + 1]
                   and abs((idx.midi[units[k + 1]] - idx.midi[units[k]]) - (phrase[k + 1]['m'] - phrase[k]['m'])) < 0.6)
              for k in range(len(units))]
    return units, contig


# ----------------------------------------------------------------------------- stable-part reading
def _fill(a, b, N, rng, start_at_a=True, end_at_b=True):
    """N source positions inside [a, b] (frames) read at the recorded speed.

    Longer than the recording: forward to a point m, back and forth around m (bounces of equal
    size), then forward to b. Shorter: crop (or compress when both ends are pinned)."""
    L = max(b - a, 1.0)
    if N <= 0:
        return np.zeros(0)
    if N <= L:
        if start_at_a and end_at_b:
            return a + (np.arange(N) + 0.5) * (L / N)
        if start_at_a:
            return a + np.arange(N, dtype=float)
        if end_at_b:
            return b - N + np.arange(N, dtype=float) + 1
        off = rng.uniform(0, L - N)
        return a + off + np.arange(N, dtype=float)
    if N < 1.25 * L:                      # slightly longer: read a little slower
        return a + (np.arange(N) + 0.5) * (L / N)
    extra = N - L
    k = int(np.ceil(extra / (1.6 * L)))   # number of bounces, each <= 0.8 L deep
    A = extra / (2 * k)
    m = a + min(L, A + 0.25 * (L - A) + rng.uniform(0, 0.5) * (L - A))
    way = [a, m]
    for _ in range(k):
        way += [m - A, m]
    way += [b]
    seg = np.abs(np.diff(way))
    cum = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, cum[-1], N)
    return np.interp(u, cum, way)


# ----------------------------------------------------------------------------- phrase synthesis
def _smooth01(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _hum_gain(f):
    """|H(f)|^2 closed-mouth hum from an 'u': lip low-pass ~ 900 Hz, nasal murmur boost ~ 250 Hz."""
    lp = 1.0 / (1.0 + (f / 900.0) ** 4)
    murmur = 1.0 + 0.6 * np.exp(-0.5 * ((f - 250.0) / 90.0) ** 2)
    return (lp * murmur) ** 2 + 10 ** (-45 / 10)


class _Layer:
    """One unit's contribution to a phrase: output frames [f0, f0+len), source positions,
    sp weight, pitch weight (for crossfades), gain (dB) and pitch correction (cents)."""

    def __init__(self, unit, fstart, pos, target_cents):
        self.u = unit
        self.f = int(fstart)
        self.pos = np.asarray(pos, float)
        n = len(self.pos)
        self.ws = np.ones(n)
        self.corr = np.full(n, float(target_cents))       # absolute target per frame (cents)
        self.voiced_only = np.zeros(n, bool)              # True: keep source voicing (onset/release)
        self.vib = None                                    # per-frame scale of the source deviation

    @property
    def end(self):
        return self.f + len(self.pos)


def _borrow(idx, singer, vowel, tech, m, kind, exclude=-1):
    """A unit of the same singer / vowel with a natural onset (kind='on') or release ('off'),
    nearest in pitch (<= 7 semitones), preferring the requested technique."""
    tm, tc = _cand_mask(idx, singer, vowel, tech)
    flag = idx.onset_nat if kind == 'on' else idx.release_nat
    cand = np.flatnonzero(tm & flag)
    if not len(cand):
        v = 'u' if vowel == 'm' else vowel
        cand = np.flatnonzero((idx.singer == singer) & (idx.vowel == v) & flag & (idx.ctx != 'excerpt'))
        tc = np.full(idx.n, 1.0)
    cand = cand[cand != exclude]
    if not len(cand):
        return None
    d = np.abs(idx.midi[cand] - m)
    cost = d + tc[cand] + 0.3 * (idx.t0[cand] - idx.t_on[cand] > 0.45)
    j = int(np.argmin(cost))
    return int(cand[j]) if d[j] <= 7 else None


def _render_phrase(idx, phrase, units, contig, singer, rng, opts):
    """Synthesize one phrase for one singer. Returns (start_sample, mono float64)."""
    import pyworld as pw
    n_notes = len(phrase)
    S = [int(round(n['st'] / HOP)) for n in phrase]
    E = [int(round((n['st'] + n['du']) / HOP)) for n in phrase]
    for k in range(n_notes - 1):                  # legato: the next note starts where this ends
        E[k] = S[k + 1]
    for k in range(n_notes):
        E[k] = max(E[k], S[k] + 6)
    tr_in = [0] * n_notes     # frames of note k consumed by the transition from k-1
    tr_out = [0] * n_notes    # frames of note k consumed by the transition to k+1
    trans = [None] * n_notes  # transition k -> k+1: dict
    for k in range(n_notes - 1):
        ua, ub = units[k], units[k + 1]
        iv = phrase[k + 1]['m'] - phrase[k]['m']
        len_a, len_b = E[k] - S[k], E[k + 1] - S[k + 1]
        if contig[k]:
            a_end = int(round(idx.t1[ua] / HOP))
            b_start = int(round(idx.t0[ub] / HOP))
            L = max(b_start - a_end, 2)
            # align the source's pitch midpoint with the boundary
            c = _src_cents(idx.file[ua], np.arange(a_end, b_start + 1, dtype=float))
            mid_c = 50 * (idx.midi[ua] + idx.midi[ub])
            ok = np.isfinite(c)
            if ok.sum() > 2:
                cc = np.interp(np.arange(len(c)), np.flatnonzero(ok), c[ok])
                cross = np.flatnonzero(np.sign(cc - mid_c) != np.sign(cc[0] - mid_c))
                mid = int(cross[0]) if len(cross) else L // 2
            else:
                mid = L // 2
            pre, post = mid, L - mid
            scale = min(1.0, 0.45 * len_a / max(pre, 1), 0.45 * len_b / max(post, 1))
            pre_o, post_o = max(1, int(pre * scale)), max(1, int(post * scale))
            trans[k] = dict(kind='real', src=(a_end, b_start), pre=pre_o, post=post_o, iv=iv)
        else:
            _, durs = glide_templates()
            T = (0.05 + 0.012 * abs(iv) + 0.4 * durs[1 if iv >= 0 else -1] * (abs(iv) > 0)) / HOP
            if abs(iv) < 0.01:
                T = 0.06 / HOP
            T = int(min(T, 0.8 * len_a, 0.8 * len_b))
            T = max(T, 4)
            trans[k] = dict(kind='xfade', pre=T // 2, post=T - T // 2, iv=iv)
        tr_out[k] = trans[k]['pre']
        tr_in[k + 1] = trans[k]['post']
    layers = []
    corr_c = [phrase[k]['m'] * 100 for k in range(n_notes)]          # target cents
    gain_db = [(-26.0 * (1 - phrase[k]['vel'] / 127) ** 1.2) + VEL_REF_DB
               + VOWEL_TRIM_DB[phrase[k]['vowels'][-1]] for k in range(n_notes)]
    # onset: the unit's own natural onset, else one borrowed from another note of the same
    # singer / vowel (shifted to this pitch and crossfaded in), else a synthetic soft attack
    XF = 10                                            # 50 ms parameter crossfade
    u0 = units[0]
    onset_unit = u0 if idx.onset_nat[u0] else (
        _borrow(idx, singer, phrase[0]['vowels'][0], phrase[0]['tech'], phrase[0]['m'], 'on', u0)
        if opts['borrow'] else None)
    natural_on = onset_unit is not None
    att_f = 0
    on_pre = on_post = None
    if natural_on:
        a0 = int(round(idx.t0[onset_unit] / HOP))
        s0 = int(round(idx.t_on[onset_unit] / HOP))
        s0 = max(s0, a0 - int(0.40 / HOP))
        on_frames = max(a0 - s0, 1)
        # perceptual onset: first frame within 15 dB of the stable level
        sp_on, _ = _frames(idx.file[onset_unit], np.arange(s0, a0 + 1, dtype=float))
        lev = 10 * np.log10(sp_on.sum(axis=1) + 1e-30)
        ref = np.median(lev[-5:])
        hit = np.flatnonzero(lev > ref - 15)
        p_on = int(hit[0]) if len(hit) else on_frames // 2
        p_on = min(p_on, on_frames - 1)
        post = on_frames - p_on                     # rise after the perceptual onset
        maxpost = max(3, int(0.4 * (E[0] - S[0])))
        post_o = min(post, maxpost)
        on_pre = np.arange(s0, s0 + p_on, dtype=float)
        on_post = np.linspace(s0 + p_on, a0, post_o, endpoint=False)
    else:
        att = (0.05 + 0.10 * (1 - phrase[0]['vel'] / 127)) * (1.5 if phrase[0]['tech'] == 'breathy' else 1.0)
        att_f = max(2, int(att / HOP))
    u_end = units[-1]
    rel_unit = None
    if opts['natural_release']:
        rel_unit = u_end if idx.release_nat[u_end] else (
            _borrow(idx, singer, phrase[-1]['vowels'][-1], phrase[-1]['tech'], phrase[-1]['m'], 'off', u_end)
            if opts['borrow'] else None)
    rel_f = 0
    if rel_unit is None:
        rel = opts['release_s'] * (1.4 if phrase[-1]['tech'] == 'breathy' else 1.0)
        rel_f = max(2, int(rel / HOP))
    borrowed_on = natural_on and onset_unit != u0
    borrowed_off = rel_unit is not None and rel_unit != u_end
    extra_layers = []
    for k in range(n_notes):
        u = units[k]
        a = idx.t0[u] / HOP
        b = idx.t1[u] / HOP
        pin_start = (k == 0 and natural_on and not borrowed_on) or (k > 0 and contig[k - 1])
        pin_end = contig[k] or (k == n_notes - 1 and rel_unit == u)
        # output frames of this note's stable part (crossfade overlaps included)
        if k == 0:
            if natural_on:
                st = S[0] + len(on_post) - (XF if borrowed_on else 0)
            else:
                st = S[0] - att_f // 3 - 1
        elif contig[k - 1]:
            st = S[k] + trans[k - 1]['post']
        else:
            st = S[k] - trans[k - 1]['pre']
        if k == n_notes - 1:
            en = E[k] + (rel_f + 2 if rel_f else 0)
        elif contig[k]:
            en = S[k + 1] - trans[k]['pre']
        else:
            en = S[k + 1] + trans[k]['post']
        N = max(en - st, 2)
        pos = _fill(a, b, N, rng, start_at_a=pin_start, end_at_b=pin_end)
        L = _Layer(u, st, pos, corr_c[k])
        L.gain = np.full(N, gain_db[k])
        L.k = k
        L.vib_stable = min(1.0, VIB_CAP[phrase[k]['tech']] / max(idx.vib[u], 1.0)) * opts['vibrato']
        L.vib = np.full(N, min(L.vib_stable, 0.998))
        if k == 0 and natural_on and not borrowed_on:      # own natural onset in front
            pre_pos = np.concatenate([on_pre, on_post])
            L.f = st - len(pre_pos)
            L.pos = np.concatenate([pre_pos, L.pos])
            L.ws = np.concatenate([np.ones(len(pre_pos)), L.ws])
            L.corr = np.concatenate([np.full(len(pre_pos), corr_c[k]), L.corr])
            L.gain = np.concatenate([np.full(len(pre_pos), gain_db[k]), L.gain])
            L.voiced_only = np.concatenate([np.ones(len(pre_pos), bool), L.voiced_only])
            L.vib = np.concatenate([np.ones(len(pre_pos)), L.vib])
        if k == 0 and borrowed_on:                        # borrowed onset layer, crossfaded
            ou = onset_unit
            a_o = idx.t0[ou] / HOP
            pre_pos = np.concatenate([on_pre, on_post, a_o + np.arange(XF, dtype=float)])
            O = _Layer(ou, S[0] - len(on_pre), pre_pos, corr_c[0])
            O.k = 0
            O.gain = np.full(len(pre_pos), gain_db[0])
            O.voiced_only = np.concatenate([np.ones(len(on_pre) + len(on_post), bool), np.zeros(XF, bool)])
            O.vib = np.ones(len(pre_pos))
            O.vib_stable = min(1.0, VIB_CAP[phrase[0]['tech']] / max(idx.vib[ou], 1.0)) * opts['vibrato']
            O.corr_ref = np.full(len(pre_pos), idx.midi[ou] * 100)
            O.ws[-XF:] = 1 - _smooth01(np.linspace(0, 1, XF))
            O.pitch_w = np.ones(len(pre_pos))
            O.vowel = phrase[0]['vowels'][0]
            L.ws[:XF] = _smooth01(np.linspace(0, 1, XF))
            extra_layers.append(O)
        # real transition to the next note: continue reading through the recording
        if k < n_notes - 1 and contig[k]:
            tr = trans[k]
            src_a, src_b = tr['src']
            n_o = tr['pre'] + tr['post']
            tpos = np.linspace(src_a, src_b, n_o + 1)[1:]
            L.pos = np.concatenate([L.pos, tpos])
            L.ws = np.concatenate([L.ws, np.ones(n_o)])
            w = _smooth01(np.linspace(0, 1, n_o))
            L.corr = np.concatenate([L.corr, (1 - w) * corr_c[k] + w * corr_c[k + 1]])
            # the correction is applied relative to each unit's own centre (below)
            L.corr_ref = np.concatenate([np.full(len(L.pos) - n_o, idx.midi[u] * 100),
                                         (1 - w) * idx.midi[u] * 100 + w * idx.midi[units[k + 1]] * 100])
            L.gain = np.concatenate([L.gain, (1 - w) * gain_db[k] + w * gain_db[k + 1]])
            L.voiced_only = np.concatenate([L.voiced_only, np.zeros(n_o, bool)])
            L.vib = np.concatenate([L.vib, np.ones(n_o)])
        if not hasattr(L, 'corr_ref'):
            L.corr_ref = np.full(len(L.pos), idx.midi[u] * 100)
        # natural release after the note end
        if k == n_notes - 1 and pin_end:
            b_i = int(round(b))
            off = int(round(idx.t_off[u] / HOP))
            off = min(off, b_i + int(opts['release_max_s'] / HOP))
            rpos = np.arange(b_i + 1, off + 1, dtype=float)
            if len(rpos):
                L.pos = np.concatenate([L.pos, rpos])
                L.ws = np.concatenate([L.ws, np.ones(len(rpos))])
                L.corr = np.concatenate([L.corr, np.full(len(rpos), corr_c[k])])
                L.corr_ref = np.concatenate([L.corr_ref, np.full(len(rpos), idx.midi[u] * 100)])
                L.gain = np.concatenate([L.gain, np.full(len(rpos), gain_db[k])])
                L.voiced_only = np.concatenate([L.voiced_only, np.ones(len(rpos), bool)])
                L.vib = np.concatenate([L.vib, np.ones(len(rpos))])
        if k == n_notes - 1 and borrowed_off:             # borrowed release layer, crossfaded
            ru = rel_unit
            b_r = int(round(idx.t1[ru] / HOP))
            off = min(int(round(idx.t_off[ru] / HOP)), b_r + int(opts['release_max_s'] / HOP))
            rpos = np.arange(b_r - XF + 1, off + 1, dtype=float)
            R = _Layer(ru, E[k] - XF, rpos, corr_c[k])
            R.k = k
            R.gain = np.full(len(rpos), gain_db[k])
            R.voiced_only = np.concatenate([np.zeros(XF, bool), np.ones(len(rpos) - XF, bool)])
            R.vib = np.ones(len(rpos))
            R.vib_stable = min(1.0, VIB_CAP[phrase[k]['tech']] / max(idx.vib[ru], 1.0)) * opts['vibrato']
            R.corr_ref = np.full(len(rpos), idx.midi[ru] * 100)
            R.ws[:XF] = _smooth01(np.linspace(0, 1, XF))
            R.pitch_w = np.ones(len(rpos))
            R.vowel = phrase[k]['vowels'][-1]
            nL = len(L.pos)
            L.ws[max(0, nL - XF):] *= 1 - _smooth01(np.linspace(0, 1, min(XF, nL)))
            extra_layers.append(R)
        L.pitch_w = np.ones(len(L.pos))
        # vowel morph inside the note: a second layer for the second vowel
        layers.append(L)
        if len(phrase[k]['vowels']) > 1:
            vb = phrase[k]['vowels'][1]
            tm, tc = _cand_mask(idx, singer, vb, phrase[k]['tech'])
            cand = np.flatnonzero(tm)
            if len(cand):
                c, _ = _unary(idx, cand, tc, phrase[k], False, False, K=1)
                ub = int(c[0])
                pos_b = _fill(idx.t0[ub] / HOP, idx.t1[ub] / HOP, len(L.pos), rng, False, False)
                L2 = _Layer(ub, L.f, pos_b, corr_c[k])
                L2.corr = L.corr.copy()
                L2.corr_ref = np.full(len(pos_b), idx.midi[ub] * 100)
                L2.gain = L.gain.copy() + (VOWEL_TRIM_DB[vb] - VOWEL_TRIM_DB[phrase[k]['vowels'][0]])
                L2.voiced_only = np.zeros(len(pos_b), bool)
                L2.vib_stable = min(1.0, VIB_CAP[phrase[k]['tech']] / max(idx.vib[ub], 1.0)) * opts['vibrato']
                L2.vib = np.full(len(pos_b), min(L2.vib_stable, 0.998))
                L2.pitch_w = np.ones(len(pos_b))
                L2.k = k
                L2.vowel = vb
                t_mid = (S[k] + E[k]) / 2 - L.f
                vt = opts['vowel_time'] / HOP
                w = _smooth01((np.arange(len(pos_b)) - (t_mid - vt / 2)) / max(vt, 1))
                L.ws = L.ws * (1 - w)
                L2.ws = w
                L.vowel = phrase[k]['vowels'][0]
                layers.append(L2)
        if not hasattr(L, 'vowel'):
            L.vowel = phrase[k]['vowels'][0]
    layers += extra_layers
    # crossfade weights for non-real transitions
    by_note = {}
    for L in layers:
        by_note.setdefault(L.k, []).append(L)
    for k in range(n_notes - 1):
        if contig[k]:
            continue
        tr = trans[k]
        B = S[k + 1]
        f_a, f_b = B - tr['pre'], B + tr['post']
        sign = 1 if tr['iv'] >= 0 else -1
        for L in by_note[k]:
            idxs = np.arange(len(L.pos)) + L.f
            u = (idxs - f_a) / max(f_b - f_a, 1)
            m = idxs >= f_a
            L.ws[m] *= 1 - _smooth01(u[m])
            L.pitch_w[m] = 1 - _glide(u[m], sign)
        for L in by_note[k + 1]:
            idxs = np.arange(len(L.pos)) + L.f
            u = (idxs - f_a) / max(f_b - f_a, 1)
            m = idxs < f_b
            L.ws[m] *= _smooth01(u[m])
            L.pitch_w[m] = _glide(u[m], sign)
    # assemble frames
    f_lo = min(L.f for L in layers)
    f_hi = max(L.end for L in layers)
    nF = f_hi - f_lo + 1
    log_sp = np.zeros((nF, NB))
    ap = np.zeros((nF, NB))
    wsum = np.zeros(nF)
    pitch = np.zeros(nF)
    pwsum = np.zeros(nF)
    voiced_src = np.zeros(nF)
    keep_src_voicing = np.zeros(nF, bool)
    gain = np.zeros(nF)
    fr = np.fft.rfftfreq(FFT, 1 / SR)
    hum = _hum_gain(fr)
    for L in layers:
        fid = idx.file[L.u]
        sp_l, ap_l = _frames(fid, L.pos)
        if L.vowel == 'm':
            sp_l = sp_l * hum[None, :]
        # level normalisation of this unit + note gain (dB -> power)
        g = L.gain - idx.level[L.u]
        sl = slice(L.f - f_lo, L.f - f_lo + len(L.pos))
        w = L.ws
        log_sp[sl] += w[:, None] * (np.log(sp_l + 1e-30) + (g * np.log(10) / 10)[:, None])
        ap[sl] += w[:, None] * ap_l
        wsum[sl] += w
        gain[sl] += w * g
        src = _src_cents(fid, L.pos)
        dev = np.where(np.isfinite(src), src - L.corr_ref, 0.0)
        # onset / release / real-transition frames: keep the slow shape (scoop, fall, glide), scale
        # only the vibrato-rate part like the stable part; stable frames: scale everything
        from scipy.ndimage import uniform_filter1d
        trend = uniform_filter1d(dev, 33, mode='nearest')
        vs_ = getattr(L, 'vib_stable', float(np.median(L.vib)) if len(L.vib) else 1.0)
        shaped = trend + vs_ * (dev - trend)
        natural = L.vib >= 0.999
        dev = np.where(natural, shaped, L.vib * dev)
        dev = np.where(L.voiced_only, np.clip(dev, -250.0, 35.0), dev)   # scoops from below, falls
        p = L.corr + dev
        pw_ = L.pitch_w
        pitch[sl] += pw_ * p
        pwsum[sl] += pw_
        voiced_src[sl] += w * np.isfinite(src)
        keep_src_voicing[sl] |= L.voiced_only
    act = wsum > 1e-6
    log_sp[act] /= wsum[act, None]
    ap[act] /= wsum[act, None]
    ap = np.clip(ap, 0.0, 1.0)
    cents = np.where(np.abs(pwsum) > 1e-6, pitch / np.where(np.abs(pwsum) > 1e-6, pwsum, 1), np.nan)
    voiced = act.copy()
    vs = np.where(act, voiced_src / np.maximum(wsum, 1e-9), 0)
    voiced[keep_src_voicing & (vs < 0.5)] = False
    # envelope: synthetic attack, synthetic release, re-articulation, fall
    env_db = np.zeros(nF)
    t_rel = (np.arange(nF) + f_lo)
    if not natural_on:
        u = (t_rel - (S[0] - att_f // 3)) / att_f
        env_db += np.where(u < 1, 20 * np.log10(np.maximum(_smooth01(u), 1e-4)), 0)
        scoop = -opts['scoop_cents'] * np.exp(-np.maximum(t_rel - S[0], 0) * HOP / 0.04)
        cents = cents + np.where(t_rel >= S[0] - att_f, scoop, 0)
        breathy_on = np.clip(1 - (t_rel - (S[0] - att_f // 3)) / att_f, 0, 1)
        ap = np.clip(ap + 0.35 * breathy_on[:, None] * (1 - ap), 0, 1)
    last = phrase[-1]
    u_last = units[-1]
    if rel_f:
        u = (t_rel - E[-1]) / rel_f
        env_db += np.where(u > 0, -60 * _smooth01(u) ** 1.3, 0)
        cents = cents + np.where(u > 0, -25 * _smooth01(u), 0)
        ap = np.clip(ap + 0.4 * _smooth01(u)[:, None] * (1 - ap), 0, 1)
        voiced &= u < 1.0
    fall = last['fall'] if last['fall'] is not None else opts['fall_cents']
    if fall:
        fs_f = max(4, int(opts['fall_s'] / HOP))
        u = (t_rel - (E[-1] - int(0.6 * fs_f))) / fs_f
        cents = cents + np.where(u > 0, fall * _smooth01(u) ** 1.6, 0)
        env_db += np.where(u > 0, -30 * _smooth01(u) ** 2.2, 0)
        ap = np.clip(ap + 0.3 * _smooth01(u)[:, None] * (1 - ap), 0, 1)
    for k in range(1, n_notes):          # re-articulated repeated pitch: a short dip
        if abs(phrase[k]['m'] - phrase[k - 1]['m']) < 0.01:
            d = (t_rel - S[k]) * HOP
            env_db += -opts['rearticulate_db'] * np.exp(-0.5 * (d / 0.03) ** 2)
    log_sp += (env_db * np.log(10) / 10)[:, None]
    if opts.get('presence_db'):
        bell = np.exp(-0.5 * (np.log2(np.maximum(fr, 1.0) / 3200.0) / 0.3) ** 2)
        log_sp += (opts['presence_db'] * np.log(10) / 10) * bell[None, :]
    # WORLD leaks aperiodic noise below f0: taper the envelope under 0.85 f0
    f0_hz = np.where(voiced & np.isfinite(cents), 440 * 2 ** ((np.nan_to_num(cents) / 100 - 69) / 12), 0.0)
    vi = np.flatnonzero(f0_hz > 0)
    if len(vi):
        r = fr[None, :] / f0_hz[vi, None]
        tap = np.clip((r - 0.45) / 0.40, 0, 1) ** 2
        log_sp[vi] += np.log(np.maximum(tap, 1e-4))
    sp = np.exp(log_sp)
    sp[~act] = 1e-16
    ap[~act] = 1.0
    f0_hz[~act] = 0
    # pad one silent frame at both ends
    f0_hz = np.concatenate([[0], f0_hz, [0]])
    sp = np.concatenate([np.full((1, NB), 1e-16), sp, np.full((1, NB), 1e-16)])
    ap = np.concatenate([np.ones((1, NB)), ap, np.ones((1, NB))])
    y = pw.synthesize(np.ascontiguousarray(f0_hz), np.ascontiguousarray(sp), np.ascontiguousarray(ap),
                      SR, frame_period=HOP * 1000)
    start_sample = int(round((f_lo - 1) * HOP * SR))
    info = dict(units=[idx.U[u] | {'id': u} for u in units], contiguous=contig,
                onset='own' if natural_on and not borrowed_on else ('borrowed' if borrowed_on else 'synthetic'),
                release='own' if rel_unit == u_end else ('borrowed' if borrowed_off else 'synthetic'),
                shifts_semitones=[round(phrase[k]['m'] - idx.midi[units[k]], 2) for k in range(n_notes)])
    return start_sample, y, info


# ----------------------------------------------------------------------------- breaths
def _breath_clip(idx, singer, gender, rng, max_len, exclude=()):
    B = [b for b in idx.breaths if 800 <= b['centroid_hz'] <= 3200 and 0.18 <= b['t1'] - b['t0'] <= 0.8
         and b['rel_db'] > -48 and b.get('clarity', 0) < 0.4]
    own = [b for b in B if b['singer'] == singer]
    pool = own if len(own) >= 2 else [b for b in B if idx.singers[b['singer']]['gender'] == gender]
    pool = [b for b in pool if (b['file'], b['t0']) not in exclude] or pool
    b = pool[int(rng.integers(len(pool)))]
    log_access(VS / b['file'])
    x, sr = sf.read(need(VS / b['file']), dtype='float64')
    pre = 0.04
    a = max(0, int((b['t0'] - pre) * sr))
    e = int(b['t1'] * sr)
    seg = resample_sr(x[a:e], sr, SR).astype(np.float64)
    seg = seg[-int(max_len * SR):] if len(seg) > max_len * SR else seg
    from ._dsp import butter, sosfilt
    seg = sosfilt(butter('highpass', 120.0, 2), seg).astype(np.float64)
    seg = fade_edges(seg.astype(np.float32), 0.03, 0.02).astype(np.float64)
    rms = np.sqrt(np.mean(seg ** 2)) + 1e-12
    return seg / rms, (b['file'], b['t0'])


# ----------------------------------------------------------------------------- singer choice
def _range_cost(idx, singer, line):
    c = 0.0
    for n in line:
        v = 'u' if n['vowels'][0] == 'm' else n['vowels'][0]
        tm, _ = _cand_mask(idx, singer, n['vowels'][0], n['tech'])
        ms = idx.midi[tm] if tm.any() else idx.midi[(idx.singer == singer) & (idx.vowel == v)]
        if not len(ms):
            return 1e9
        d = np.min(np.abs(ms - n['m']))
        c += d + 2 * max(d - 2.5, 0)
    return c / len(line)


def _choose_singers(idx, line, n, used, gender, explicit, rng, blocked=(), prefer=()):
    """n singers for a line: best range cover, not already singing at the same time (blocked),
    fresh voices preferred (used), the singers of this line's previous section kept if they fit."""
    if explicit:
        pool = [s for s in explicit if s in idx.singers]
    else:
        pool = [s for s in idx.singers if gender in (None, 'auto', 'any') or idx.singers[s]['gender'] == gender]
    free = [s for s in pool if s not in blocked]
    if len(free) >= n:
        pool = free
    costs = [(_range_cost(idx, s, line) + (0.8 if s in used and s not in prefer else 0.0)
              - (0.6 if s in prefer else 0.0) + rng.uniform(0, 0.3), s) for s in pool]
    costs.sort()
    if explicit:                          # keep the given order, skip impossible ones
        costs = [(c, s) for c, s in [(_range_cost(idx, s, line), s) for s in pool] if c < 1e8]
    elif gender in (None, 'auto', 'any') and n > 1:
        # one voice type per line: take the gender whose best n singers fit better
        best = None
        for g in ('female', 'male'):
            cg = [(c, s) for c, s in costs if idx.singers[s]['gender'] == g]
            if len(cg) >= n:
                tot = sum(c for c, _ in cg[:n])
                if best is None or tot < best[0]:
                    best = (tot, cg)
        if best is not None:
            costs = best[1]
    return [s for _, s in costs[:n]]


# ----------------------------------------------------------------------------- API
def render_voice(notes, *, voices=1, singers=None, gender='auto', tech='straight', vowel='a',
                 legato_gap=0.12, breaths='auto', breath_db=-22.0, breath_min_gap=0.35,
                 fall_cents=0.0, fall_s=0.35, vibrato=1.0, scoop_cents=45.0, release_s=0.18,
                 natural_release=True, release_max_s=0.45, vowel_time=0.25, rearticulate_db=9.0,
                 borrow=True, humanize_ms=None, presence_db=0.0, section_gap=4.0, width=0.7, pan=0.0,
                 duration=None, gain_db=0.0, seed=0, return_info=False):
    """Sing `notes` with real VocalSet singers. Returns (n, 2) float32 at 48 kHz (dry).

    notes : [(start_s, dur_s, midi, velocity[, {vowel, tech, part, singer, fall, breath, pan}])]
            - vowel: a e i o u, 'm' (hum: an 'u' through a closed-mouth filter) or a morph 'u>a';
            - tech: 'straight' | 'breathy' | 'vibrato' (natural classical vibrato);
            - part: line label; notes of one part are split into monophonic lines automatically;
            - singer: force a VocalSet singer id for that note's line ('f1'..'f9', 'm1'..'m11');
            - fall: phrase-end fall in cents for the last note of its phrase (e.g. -150);
            - breath: False to suppress / True to force an inhalation before this phrase.
            The old sampler.choir tuple (start, dur, midi, vel, vowel[, part]) is accepted too.
    voices : singers per line (1 = solo). Every line gets DIFFERENT singers, chosen by how well
             their recorded range covers the line (and `gender`, `singers`).
    singers : preferred singer ids (list), used in order.
    section_gap : a line that rests at least this long (s) starts a new section that chooses its
             singers again (the previous ones are kept if their range fits); None = one choice
             per line for the whole piece. A singer never sings two sections at once.
    gender : 'female' | 'male' | 'auto' (by range: VocalSet women sang ~C4-G5, men ~A#2-F4).
    tech, vowel : defaults for notes without keywords.
    legato_gap : notes closer than this (s) are joined into one legato phrase.
    breaths : 'auto' (real inhalation of the same singer before phrases that follow a gap of at
              least breath_min_gap), False, or True (also before the first phrase).
    breath_db : breath level relative to the following phrase's first note (RMS, dB).
    fall_cents / fall_s : default phrase-end fall (山歌 style), e.g. -120 over 0.35 s.
    vibrato : scale of the singer's own pitch deviation (0 = dead straight, 1 = as recorded).
    humanize_ms : per-singer onset scatter for voices > 1 (default 22 ms; 0 for solo).
    presence_db : gain of a broad bell at 3.2 kHz (about 2.5-4 kHz) applied to the spectral
             envelope before synthesis (0 = the singer as recorded). VocalSet singers differ by
             ~19 dB in this band (measured: f1 brightest, f4 darkest); -4 tames the bright ones.
    width / pan : stereo spread of the singers / centre.
    return_info : also return a dict with the singers, units, shifts and real transitions used.
    """
    idx = _index()
    rng = np.random.default_rng(seed)
    ns = _prep(notes, tech, vowel)
    if not ns:
        out = np.zeros((int((duration or 0.1) * SR), 2), np.float32)
        return (out, {}) if return_info else out
    opts = dict(borrow=bool(borrow), fall_cents=float(fall_cents or 0.0), fall_s=float(fall_s), vibrato=float(vibrato),
                scoop_cents=float(scoop_cents), release_s=float(release_s), natural_release=bool(natural_release),
                release_max_s=float(release_max_s), vowel_time=float(vowel_time),
                rearticulate_db=float(rearticulate_db), presence_db=float(presence_db))
    if humanize_ms is None:
        humanize_ms = 22.0 if voices > 1 else 0.0
    lines = _lines(ns)
    # a line with long rests is cut into sections that choose their singers separately (a part
    # that sings low in one passage and high in another gets singers who have those notes);
    # nobody sings two sections at the same time
    sections = []
    for li, (part, line) in enumerate(lines):
        cur = [line[0]]
        for n in line[1:]:
            if section_gap and n['st'] - (cur[-1]['st'] + cur[-1]['du']) >= section_gap:
                sections.append((li, part, cur))
                cur = [n]
            else:
                cur.append(n)
        sections.append((li, part, cur))
    used = set()
    busy = {}
    prev_of_line = {}
    jobs = []
    for li, part, line in sections:
        a, b = line[0]['st'] - 0.5, max(n['st'] + n['du'] for n in line) + 0.5
        blocked = {s_ for s_, iv in busy.items() if any(x < b and a < y for x, y in iv)}
        forced = [n['singer'] for n in line if n['singer']]
        if forced:
            chosen = [forced[0]] + _choose_singers(idx, line, voices - 1, used | {forced[0]}, gender, singers,
                                                   rng, blocked | {forced[0]}, prev_of_line.get(li, ()))
        else:
            chosen = _choose_singers(idx, line, voices, used, gender, singers, rng, blocked,
                                     prev_of_line.get(li, ()))
        chosen = chosen[:voices]
        used.update(chosen)
        for s_ in chosen:
            busy.setdefault(s_, []).append((a, b))
        prev_of_line[li] = set(chosen)
        jobs.append((li, part, line, chosen))
    # pan: lines spread across the width, singers of a line spread around it
    n_lines = len(lines)
    total = 0
    renders = []
    info = dict(lines=[])
    sec_no = {}
    for li, part, line, chosen in jobs:
        k_sec = sec_no.get(li, 0)
        sec_no[li] = k_sec + 1
        line_pan = pan + (0.0 if n_lines == 1 else width * 0.6 * (li / (n_lines - 1) - 0.5) * 2)
        linfo = dict(part=part, section=k_sec, start=round(line[0]['st'], 3), notes=len(line), singers=chosen,
                     phrases=[])
        for si, singer in enumerate(chosen):
            srng = np.random.default_rng([seed, li, si] if k_sec == 0 else [seed, li, si, k_sec])
            dt = srng.normal(0, humanize_ms / 1000.0) if humanize_ms else 0.0
            dv = srng.uniform(-3, 3) if voices > 1 else 0.0
            sp = line_pan + (0.0 if len(chosen) == 1 else width * 0.35 * (si / (len(chosen) - 1) - 0.5) * 2)
            gl = pan_gains(float(np.clip(sp if line[0]['pan'] is None else line[0]['pan'], -1, 1)))
            shifted = [dict(n, st=n['st'] + dt, vel=float(np.clip(n['vel'] + dv, 1, 127))) for n in line]
            phrases = _phrases(shifted, legato_gap)
            prev_end = None
            used_breaths = set()
            for pi, ph in enumerate(phrases):
                units, contig = _select(idx, ph, singer, srng)
                s0, y, pinfo = _render_phrase(idx, ph, units, contig, singer, srng, opts)
                renders.append((s0, y, gl))
                total = max(total, s0 + len(y))
                pinfo['singer'] = singer
                linfo['phrases'].append(pinfo)
                # inhalation before this phrase
                want = ph[0]['breath']
                gap = ph[0]['st'] - (prev_end if prev_end is not None else -1e9)
                if want is None:
                    want = breaths is True or (breaths == 'auto' and prev_end is not None and gap >= breath_min_gap)
                if want and breaths is not False:
                    maxlen = min(0.7, (gap - 0.05) if prev_end is not None else 0.7)
                    if maxlen > 0.15:
                        br, key = _breath_clip(idx, singer, idx.singers[singer]['gender'], srng, maxlen, used_breaths)
                        used_breaths.add(key)
                        lvl = np.sqrt(np.mean(y[len(y) // 4: len(y) // 2] ** 2) + 1e-20) if len(y) > 400 else 1e-3
                        br = br * lvl * db2a(breath_db)
                        onset = int(round(ph[0]['st'] * SR))
                        b0 = onset - int(srng.uniform(0.03, 0.08) * SR) - len(br)
                        renders.append((b0, br, gl))
                        total = max(total, b0 + len(br))
                prev_end = ph[-1]['st'] + ph[-1]['du']
        info['lines'].append(linfo)
    n_out = int(round(duration * SR)) if duration is not None else total + int(0.3 * SR)
    out = np.zeros((max(n_out, 1), 2), np.float64)
    norm = 1.0 / np.sqrt(max(voices, 1))
    for s0, y, (gl_, gr_) in renders:
        a = max(0, s0)
        o = a - s0
        n = min(len(y) - o, len(out) - a)
        if n > 0:
            out[a:a + n, 0] += gl_ * norm * y[o:o + n]
            out[a:a + n, 1] += gr_ * norm * y[o:o + n]
    out *= db2a(gain_db)
    out = fade_edges(out.astype(np.float32), 0.002, 0.02)
    return (out, info) if return_info else out
