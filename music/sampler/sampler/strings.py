"""Real strings from VSCO 2 Community Edition (CC0): sustained lines, pizzicato, spiccato,
and a harmonic colour derived from the sustained samples.

    render_strings(notes, articulation='sus', ...) -> (n, 2) float32
    pluck(note, vel=70, sec='vc', seed=0)           -> (n, 2) float32   one pizzicato note

notes : [(start_s, dur_s, midi, velocity[, {sec, art}])]
    sec : 'vn' violins (section), 'va' violas (section), 'vc' cellos (section),
          'cb' double bass (VSCO has only a SOLO bass), 'vn_solo' solo violin;
          winds: 'fl' flute, 'ob' oboe, 'cl' clarinet, 'hn' horn ('sus', 'spic' = staccato), 'bn' bassoon ('spic')
    art : 'sus' (bowed, with the recorded vibrato), 'harm' (harmonic, see below), 'pizz',
          'spic' (spiccato), 'sus_nv' (double bass without vibrato only)

Sampler (classic): nearest recorded pitch (VSCO samples every 3-4 semitones; the index holds
the MEASURED pitch, because the section files are named an octave low), varispeed to the exact
pitch (band-limited polyphase), dynamic layers crossfaded by velocity, round-robin for the
short articulations. Levels are normalised per sample and then follow the velocity curve of
sampler.choir / sampler.voice (vel 100 ~ -18 dBFS RMS for one section).

Legato ('sus', 'harm'): notes of one section closer than `legato_gap` form one bow line. The
next note does not re-attack: it enters `legato_offset` into its own sample (bow already moving,
past the recorded swell) with an equal-power crossfade of `xfade_s`, optionally with a short
portamento; it enters at the level the previous note had just before the change and eases to
its own level over 0.6 s (at most +-6 dB), as the bow carries the dynamic across. Notes longer than the recording are extended by crossfading random stretches of
the sustain (no fixed loop).

'harm': VSCO 2 CE has no string harmonics. A harmonic is approximated from the nearest
sustained sample of the same section with the WORLD vocoder: the pitch is set to the target
and held without vibrato, the spectral envelope is low-passed around the fundamental (the
pure, flute-like harmonic tone) and the aperiodic part is raised a little (bow air).
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import soundfile as sf

from ._dsp import SR, db2a, fade_edges, pan_gains, resample_by
from ._paths import SAMPLES, log_access, need

VSCO = SAMPLES / 'vsco2'
SECTIONS = {'vn': 'violins (section, 1st+2nd)', 'va': 'violas (section)', 'vc': 'cellos (section)',
            'cb': 'double bass (VSCO 2 CE has a solo bass only)', 'vn_solo': 'solo violin',
            # VSCO 2 CE winds and horn (sampler/prepare/vsco.py, pack vsco2:winds); 'sus' and 'spic' (staccato)
            'fl': 'flute', 'ob': 'oboe', 'cl': 'clarinet', 'bn': 'bassoon (staccato only)', 'hn': 'French horn'}
# stage seating (balance applied to the stereo samples)
SEAT = {'vn': -0.35, 'vn_solo': -0.2, 'va': 0.1, 'vc': 0.3, 'cb': 0.45,
        'fl': -0.15, 'ob': 0.05, 'cl': -0.25, 'bn': 0.2, 'hn': 0.35}
VEL_REF_DB = -13.9
_ART_ALIAS = {'sus': 'sus', 'arco': 'sus', 'legato': 'sus', 'harm': 'harm', 'harmonic': 'harm',
              'pizz': 'pizz', 'pizzicato': 'pizz', 'spic': 'spic', 'spiccato': 'spic', 'sus_nv': 'sus_nv'}


@lru_cache(maxsize=1)
def _index():
    d = json.loads(need(VSCO / 'index.json', 'VSCO 2 索引').read_text())
    return d['samples']


def _pool(sec, art):
    if art == 'harm':
        art = 'sus'
    rows = [r for r in _index() if r['sec'] == sec and r['art'] == art and r['pitch'] is not None]
    if not rows and art == 'sus' and sec == 'cb':
        rows = [r for r in _index() if r['sec'] == 'cb' and r['art'] == 'sus_nv']
    if not rows:
        raise ValueError(f'no VSCO samples for section {sec!r} articulation {art!r}')
    return rows


@lru_cache(maxsize=160)
def _raw(file):
    log_access(VSCO / file)
    x, sr = sf.read(need(VSCO / file), dtype='float32', always_2d=True)
    return x[:, :2] if x.shape[1] >= 2 else np.repeat(x, 2, axis=1), sr


@lru_cache(maxsize=256)
def _playable(file, ratio_q):
    """Sample varispeeded to pitch ratio (quantised key) and converted to 48 kHz."""
    x, sr = _raw(file)
    ratio = ratio_q / 1e5
    return resample_by(x, ratio * sr / SR)


@lru_cache(maxsize=512)
@lru_cache(maxsize=512)
def _attack_s(file):
    """Seconds from the indexed onset (first sound, -40 dB) to the attack proper: the first
    moment the 2 ms envelope is within 15 dB of its peak (VSCO pizzicati often begin with
    10-40 ms of finger noise before the pluck)."""
    from scipy.ndimage import maximum_filter1d
    x, sr = _raw(file)
    env = maximum_filter1d(np.abs(x).max(axis=1), max(1, int(0.002 * sr)))
    k = int(np.argmax(env >= env.max() * db2a(-15.0)))
    on = next((r['onset_s'] for r in _index() if r['file'] == file), 0.0)
    return max(0.0, k / sr - on)


def _sustain_info(file):
    """(steady_start_s, sustain_end_s, steady_rms_db) of a sustained sample, from its 50 ms RMS."""
    x, sr = _raw(file)
    m = x.mean(axis=1).astype(np.float64)
    hop = int(0.05 * sr)
    n = len(m) // hop
    e = 10 * np.log10((m[:n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-14)
    loud = e > e.max() - 30
    last = int(np.flatnonzero(loud)[-1]) if loud.any() else n - 1
    end = max(4, last - 6)                              # stop ~0.3 s before the recorded release
    med = float(np.median(e[: end]))
    ok = np.flatnonzero(e[:end] > med - 4)
    steady = int(ok[0]) if len(ok) else 0
    return steady * hop / sr, end * hop / sr, med


def _vel_gain_db(vel):
    return -26.0 * (1 - vel / 127.0) ** 1.2 + VEL_REF_DB


def _pick(sec, art, m, vel, rr_state):
    """[(row, weight)] dynamic layers for pitch m / velocity vel, nearest sampled pitch."""
    rows = _pool(sec, art)
    pitches = np.array(sorted({r['pitch'] for r in rows}))
    p = int(pitches[np.argmin(np.abs(pitches - m))])
    at = [r for r in rows if r['pitch'] == p]
    dyns = sorted({r['dyn'] for r in at})
    # velocity -> layer position: 0 at vel 30, 1 at vel 110
    x = np.clip((vel - 30) / 80.0, 0, 1) * (len(dyns) - 1)
    lo, hi = int(np.floor(x)), int(np.ceil(x))
    w_hi = x - lo
    out = []
    for d, w in ((dyns[lo], 1 - w_hi), (dyns[hi], w_hi)):
        if w <= 1e-3:
            continue
        cands = [r for r in at if r['dyn'] == d]
        k = rr_state.get((sec, art, p, d), rr_state.get('_seed', 0))   # seed picks the first round-robin
        rr_state[(sec, art, p, d)] = k + 1
        r = cands[k % len(cands)]
        out.append((r, w if lo != hi else 1.0))
        if lo == hi:
            break
    return p, out


def _note_audio(r, m, start_off_s, length_s, rng):
    """Stereo audio of sample r transposed to pitch m, starting start_off_s after its onset,
    length_s long; sustained samples are extended by random crossfaded stretches if needed."""
    tune = r.get('tune_cents', 0.0) / 100.0
    ratio = 2.0 ** ((m - r['pitch'] - tune) / 12.0)
    y = _playable(r['file'], int(round(ratio * 1e5)))
    on = r['onset_s'] / ratio
    a = int(round((on + start_off_s) * SR))
    n = int(round(length_s * SR))
    if a < 0:                                          # pre-roll before the file start: silence
        y = np.concatenate([np.zeros((-a, 2), np.float32), y])
        on -= a / SR
        a = 0
    if r['art'] not in ('sus', 'sus_nv'):
        seg = y[a:a + n]
        if len(seg) < n:
            seg = np.concatenate([seg, np.zeros((n - len(seg), 2), np.float32)])
        return seg
    steady, end, _ = _sustain_info(r['file'])
    pad = int(round((on - r['onset_s'] / ratio) * SR))   # samples added in front (pre-roll)
    end_i = int(round(end / ratio * SR)) + pad         # sample time t lands at t / ratio
    loop_lo = int(round(max(steady, 0.8) / ratio * SR)) + pad
    out = y[a:min(end_i, a + n)].copy()
    xf = int(0.35 * SR)
    if len(out) < 2 * xf and len(out) < n:             # offset past the sustain: start in the loop
        st = int(rng.integers(loop_lo, max(loop_lo + 1, end_i - 2 * xf)))
        out = y[st:min(end_i, st + n)].copy()
    while len(out) < n:
        seglen = int(rng.uniform(1.5, 3.0) * SR)
        lo = loop_lo
        hi = max(lo + 1, end_i - seglen - xf)
        st = int(rng.integers(lo, hi)) if hi > lo else lo
        piece = y[st:st + seglen + xf].copy()
        if len(piece) < xf * 2:
            break
        w = np.sin(0.5 * np.pi * (np.arange(xf) + 0.5) / xf)[:, None].astype(np.float32)
        tail = out[-xf:] * w[::-1]
        piece[:xf] = piece[:xf] * w + tail
        out = np.concatenate([out[:-xf], piece])
        if len(out) >= n + xf:
            break
    if len(out) < n:
        out = np.concatenate([out, np.zeros((n - len(out), 2), np.float32)])
    return out[:n]


def _harmonic(r, m, length_s, vel, rng):
    """WORLD-derived harmonic tone from sustained sample r (see module doc)."""
    import pyworld as pw
    tune = r.get('tune_cents', 0.0) / 100.0
    y = _playable(r['file'], int(round(2.0 ** ((m - r['pitch'] - tune) / 12.0) * 1e5)))
    steady, end, _ = _sustain_info(r['file'])
    ratio = 2.0 ** ((m - r['pitch'] - tune) / 12.0)
    a = int(round(max(steady, 0.6) / ratio * SR))
    e = int(round(end / ratio * SR))
    need = int((length_s + 0.1) * SR)
    seg = y[a:e].mean(axis=1).astype(np.float64)
    if len(seg) < need:                                   # ping-pong the source to length
        reps = int(np.ceil(need / max(len(seg), 1))) + 1
        seg = np.concatenate([seg if i % 2 == 0 else seg[::-1] for i in range(reps)])
    seg = seg[:need]
    f0_src, t = pw.harvest(seg, SR, f0_floor=55.0, f0_ceil=2200.0, frame_period=5.0)
    f_tgt = 440.0 * 2 ** ((m - 69) / 12.0)
    f0_src = np.where(f0_src > 0, f0_src, f_tgt)
    sp = pw.cheaptrick(seg, f0_src, t, SR, fft_size=2048)
    ap = pw.d4c(seg, f0_src, t, SR, fft_size=2048)
    fr = np.fft.rfftfreq(2048, 1 / SR)
    g = 1.0 / (1.0 + (fr / (1.25 * f_tgt)) ** 6)          # keep the fundamental, thin the rest
    g += 0.06 / (1.0 + ((fr - 2 * f_tgt) / (0.15 * f_tgt)) ** 2)
    g *= np.clip((fr / f_tgt - 0.5) / 0.3, 0, 1) ** 2      # nothing below the fundamental
    sp = sp * (g ** 2)[None, :] + 1e-16
    ap = np.clip(ap + 0.12, 0, 1)
    drift = 1 + 0.0006 * np.cumsum(rng.standard_normal(len(t))) / np.sqrt(len(t))
    f0 = f_tgt * drift
    z = pw.synthesize(f0, sp, ap, SR, frame_period=5.0)
    # airy bow noise from the same sample, high band, very low
    from ._dsp import butter, sosfilt
    air = sosfilt(butter('bandpass', (3000.0, 9000.0), 2), seg).astype(np.float64)
    n = min(len(z), len(air))
    z = z[:n] + 0.04 * air[:n] * np.sqrt(np.mean(z[:n] ** 2) / (np.mean(air[:n] ** 2) + 1e-20))
    return np.stack([z, z], axis=1).astype(np.float32)


def _lines(notes):
    lines = []
    for n in notes:
        best, bd = None, 1e9
        for L in lines:
            last = L[-1]
            ov = last['st'] + last['du'] - n['st']
            if n['st'] - last['st'] < 0.05 or ov > 0.35 * min(last['du'], n['du']):
                continue
            d = abs(last['m'] - n['m'])
            if d < bd:
                best, bd = L, d
        if best is None:
            best = []
            lines.append(best)
        best.append(n)
    return lines


def render_strings(notes, articulation='sus', *, section=None, legato=True, legato_gap=0.08,
                   legato_offset=None, xfade_s=0.14, portamento_s=0.0, attack_s=None, release_s=0.45,
                   pizz_ring_s=0.6, width=1.0, pan=0.0, duration=None, gain_db=0.0, seed=0):
    """Render string notes (see module doc). Returns (n, 2) float32 at 48 kHz, dry
    (the VSCO samples carry their own close room; add sampler.mix.reverb for the hall).

    articulation / section : defaults for notes without the {art, sec} keywords.
    legato, legato_gap : join close notes of one section into bow lines (sus / harm).
    legato_offset : where a legato note enters its own sample (s after the onset);
                    None = the sample's measured steady point (typ. 0.3-1.5 s).
    xfade_s : legato crossfade length; portamento_s : optional slide into the new pitch.
    attack_s : extra fade-in for bow starts (None = the recorded attack only).
    release_s : bow release (sus / harm) fade after the note end.
    pizz_ring_s : pizzicato / spiccato keep ringing this long after the written duration.
    width : 0 = every section in the centre, 1 = orchestral seating.
    """
    rng = np.random.default_rng(seed)
    rr_state = {'_seed': int(seed)}
    evs = []
    for n in notes:
        st, du, m = float(n[0]), float(n[1]), float(n[2])
        vel = float(n[3]) if len(n) > 3 else 80.0
        kw = n[4] if len(n) > 4 and isinstance(n[4], dict) else {}
        if du <= 0 or vel <= 0:
            continue
        art = _ART_ALIAS.get(kw.get('art', articulation), None)
        if art is None:
            raise ValueError(f"articulation {kw.get('art', articulation)!r}: sus harm pizz spic sus_nv")
        sec = kw.get('sec', section) or _default_section(m)
        if sec not in SECTIONS:
            raise ValueError(f'section {sec!r}: one of {list(SECTIONS)}')
        evs.append(dict(st=st, du=du, m=m, vel=float(np.clip(vel, 1, 127)), art=art, sec=sec))
    evs.sort(key=lambda e: e['st'])
    total = max([e['st'] + e['du'] for e in evs] + [0.0]) + max(release_s, pizz_ring_s) + 1.0
    n_out = int(round((duration if duration is not None else total) * SR))
    out = np.zeros((max(n_out, 1), 2), np.float32)

    def put(x, t, sec):
        g_l, g_r = pan_gains(float(np.clip(pan + width * SEAT.get(sec, 0.0), -1, 1)))
        a = int(round(t * SR))
        s0 = max(0, -a)
        b = min(len(out), a + len(x))
        if b > max(a, 0):
            seg = x[s0:s0 + b - max(a, 0)]
            out[max(a, 0):b, 0] += seg[:, 0] * g_l * np.sqrt(2)
            out[max(a, 0):b, 1] += seg[:, 1] * g_r * np.sqrt(2)

    # short articulations: one shot each
    for e in [e for e in evs if e['art'] in ('pizz', 'spic')]:
        p, layers = _pick(e['sec'], e['art'], e['m'], e['vel'], rr_state)
        ring = e['du'] + pizz_ring_s
        acc = None
        for r, w in layers:
            # align the pluck itself (first moment within 15 dB of the peak), not the finger
            # noise before it, on the note's start; 8 ms of what precedes it are kept
            ratio = 2.0 ** ((e['m'] - r['pitch'] - r.get('tune_cents', 0.0) / 100.0) / 12.0)
            x = _note_audio(r, e['m'], _attack_s(r['file']) / ratio - 0.008, ring + 0.3, rng)
            g = w * db2a(_vel_gain_db(e['vel']) - _oneshot_level(r['file']))
            acc = x * g if acc is None else acc + x * g
        env = np.ones(len(acc), np.float32)
        k = int(ring * SR)
        if k < len(acc):
            tail = len(acc) - k
            env[k:] = np.exp(-np.arange(tail) / (0.08 * SR)).astype(np.float32)
        acc = fade_edges(acc * env[:, None], 0.002, 0.01)
        put(acc, e['st'] - 0.008, e['sec'])
    # bowed lines
    for sec in SECTIONS:
        for art in ('sus', 'sus_nv', 'harm'):
            group = [e for e in evs if e['sec'] == sec and e['art'] == art]
            if not group:
                continue
            for line in _lines(group):
                phrases = [[line[0]]]
                for e in line[1:]:
                    last = phrases[-1][-1]
                    if legato and e['st'] - (last['st'] + last['du']) <= legato_gap:
                        phrases[-1].append(e)
                    else:
                        phrases.append([e])
                for ph in phrases:
                    y, t0 = _render_bow_phrase(ph, art, legato_offset, xfade_s, portamento_s, attack_s,
                                               release_s, rr_state, rng)
                    put(y, t0, sec)
    out *= np.float32(db2a(gain_db))
    return fade_edges(out, 0.002, 0.02)


def _default_section(m):
    return 'cb' if m < 43 else 'vc' if m < 55 else 'va' if m < 67 else 'vn'


@lru_cache(maxsize=512)
def _oneshot_level(file):
    """Normalising level of a one-shot sample: RMS of its first 250 ms after the onset (dB)."""
    x, sr = _raw(file)
    m = x.mean(axis=1).astype(np.float64)
    env = np.abs(m)
    on = int(np.argmax(env > env.max() * 0.01))
    seg = m[on:on + int(0.25 * sr)]
    return float(10 * np.log10(np.mean(seg ** 2) + 1e-14))


def _render_bow_phrase(ph, art, legato_offset, xfade_s, portamento_s, attack_s, release_s, rr_state, rng):
    """One legato bow line -> (stereo, start_s)."""
    t0 = ph[0]['st'] - 0.01
    end = ph[-1]['st'] + ph[-1]['du'] + release_s + 0.05
    n = int(round((end - t0) * SR)) + 1
    acc = np.zeros((n, 2), np.float32)
    xf = int(xfade_s * SR)
    prev_tail_db = None
    for i, e in enumerate(ph):
        first, last = i == 0, i == len(ph) - 1
        p, layers = _pick(e['sec'], 'sus' if art == 'harm' else art, e['m'], e['vel'], rr_state)
        seg_start = e['st'] - (0 if first else xfade_s / 2)
        seg_end = (ph[i + 1]['st'] + xfade_s / 2) if not last else e['st'] + e['du'] + release_s
        length = seg_end - seg_start
        note = None
        for r, w in layers:
            steady, _, med = _sustain_info(r['file'])
            off = 0.0 if first else (legato_offset if legato_offset is not None else max(steady, 0.35))
            if art == 'harm':
                x = _harmonic(r, e['m'], length + 0.05, e['vel'], rng)[:int(length * SR)]
                ref = 10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-14)
            else:
                x = _note_audio(r, e['m'], off - (0.01 if first else 0.0), length, rng)
                ref = med
            g = w * db2a(_vel_gain_db(e['vel']) - ref)
            note = x * g if note is None else note[:len(x)] + (x * g)[:len(note)]
        L = len(note)

        def _rms_db(z):
            return 10 * np.log10(np.mean(z.astype(np.float64) ** 2) + 1e-14) if len(z) else None
        # legato level match: the bow carries the dynamic across the change, so the new note
        # enters at the level the previous one had, then eases to its own over 0.6 s (<= +-6 dB)
        if not first and prev_tail_db is not None and art != 'harm':
            head = _rms_db(note[xf:xf + int(0.15 * SR)])
            if head is not None:
                d_db = float(np.clip(prev_tail_db - head, -6.0, 6.0))
                k = min(L, int(0.6 * SR))
                u = np.arange(k) / max(k, 1)
                ramp = np.zeros(L)
                ramp[:k] = d_db * (1 - u * u * (3 - 2 * u))
                note = note * db2a(ramp)[:, None].astype(np.float32)
        prev_tail_db = _rms_db(note[max(0, L - xf - int(0.12 * SR)):max(0, L - xf)]) if not last else None
        env = np.ones(L, np.float32)
        if first:
            k5 = min(int(0.005 * SR), L)
            env[:k5] *= np.linspace(0, 1, k5, dtype=np.float32)
            if art == 'harm':
                a = int(0.18 * SR)
                env[:a] *= (np.sin(0.5 * np.pi * np.arange(a) / a) ** 2).astype(np.float32)
            elif attack_s:
                a = int(attack_s * SR)
                env[:a] *= (np.sin(0.5 * np.pi * np.arange(a) / a) ** 2).astype(np.float32)
        else:
            k = min(xf, L)
            env[:k] *= np.sin(0.5 * np.pi * (np.arange(k) + 0.5) / k).astype(np.float32)
        if not last:
            k = min(xf, L)
            env[L - k:] *= np.cos(0.5 * np.pi * (np.arange(k) + 0.5) / k).astype(np.float32)
        else:
            r_n = int(release_s * SR)
            k0 = max(0, L - r_n)
            u = np.arange(L - k0) / max(r_n, 1)
            env[k0:] *= np.exp(-4.5 * u).astype(np.float32) * (1 - u).astype(np.float32)
        note = note * env[:, None]
        if portamento_s > 0 and not first:
            note = _slide_in(note, ph[i - 1]['m'] - e['m'], portamento_s)
        a = int(round((seg_start - t0) * SR))
        b = min(n, a + len(note))
        acc[a:b] += note[:b - a]
    return acc, t0


def _slide_in(x, semis, dur_s):
    """Time-varying resample of the first dur_s: pitch glides from +semis to 0."""
    n = int(dur_s * SR)
    if n < 8 or abs(semis) < 0.01:
        return x
    u = np.arange(len(x), dtype=np.float64)
    c = np.ones(len(x))
    ramp = np.clip(u / n, 0, 1)
    c = 2.0 ** (semis * (1 - ramp) ** 2 / 12.0)          # instantaneous rate
    pos = np.cumsum(c) - c[0]
    pos = np.minimum(pos, len(x) - 1.001)
    i0 = np.floor(pos).astype(int)
    fr = (pos - i0)[:, None]
    y = x[i0] * (1 - fr) + x[np.minimum(i0 + 1, len(x) - 1)] * fr
    return y.astype(np.float32)


def pluck(note, vel=70, sec='vc', seed=0, ring_s=0.8):
    """One real pizzicato note (VSCO 2 CE), peak-normalised to -6 dBFS. note: MIDI or name."""
    from .nature import _note_num
    m = _note_num(note) if isinstance(note, str) else float(note)
    y = render_strings([(0.0, 0.25, m, vel, {'sec': sec, 'art': 'pizz'})], pizz_ring_s=ring_s, width=0.0,
                       seed=seed)
    y = y[: int(np.flatnonzero(np.abs(y).max(axis=1) > 1e-5)[-1]) + 1] if np.abs(y).max() > 0 else y
    pk = float(np.abs(y).max())
    return fade_edges((y * db2a(-6.0) / pk).astype(np.float32) if pk > 0 else y, 0.0, 0.01)
