"""Found sounds as instruments: real recordings of the things on screen (pencil, paper, keys, snaps,
knocks, water drops, creaks), CC0 from Freesound (<samples>/found/manifest.json), cut into single events.

    events(name)                                 -> [{a, b, onset, peak_db, centroid, dur}]  (samples at 48 kHz)
    hit(name, pick, vel, seed, pitch)            -> (clip, lead)   one event, its onset at index `lead`
    place(track, name, at, vel, seed, pick, ...) -> track          the event's onset lands on `at` (s)
    bed(name, dur, seed)                         -> (n, 2)         a stretch of the recording as a texture
    stroke(name, dur, vel, seed)                 -> (n, 2)         one continuous gesture (a pencil line) of dur s

pick chooses among a recording's events: 'any' (round-robin by seed), 'bright' / 'dark' (upper / lower
third by spectral centroid), 'short' / 'long' (by length), 'loud' (upper third by peak) or an int index.
Events are found from the 3 ms envelope: an onset is a rise of `rise_db` above the level 30 ms before
it, the event ends at the next onset or when the envelope falls to the recording's floor + 6 dB, at
most `max_s`. Events that click on their own are left out of every pick (an int pick indexes the rest). Every event is peak-normalised, so the velocity sets its level (vel 127 -> -12 dBFS peak).
Classic DSP only.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from ._dsp import SR, add_at, cos_ramp, db2a, fade_edges, pan_gains, resample_by, resample_sr
from ._paths import SAMPLES, log_access, need

FOUND = SAMPLES / 'found'
# per recording: (rise_db, max_s, min_gap_s, high-pass Hz)
CUT = {
    'pencil_writing_rivernile7': (7, 0.35, 0.06, 150), 'pencil_writing_fthgurdy': (7, 0.35, 0.06, 150),
    'scribble_long_waldram': (7, 0.5, 0.08, 150), 'paper_rustle_keweldog': (8, 0.6, 0.1, 80),
    'sketchbook_flips_owlstorm': (9, 0.9, 0.2, 60), 'page_turn_davidbain': (10, 1.5, 0.3, 60),
    'keyboard_typing_trollarch2': (8, 0.18, 0.05, 120), 'keyboard_typing_florianreichelt': (8, 0.18, 0.05, 120),
    'finger_snaps_7146007': (12, 0.35, 0.15, 200), 'finger_snaps_johnsonbrand': (12, 0.35, 0.15, 200),
    'wood_knock_mrguff': (12, 0.5, 0.1, 60), 'knuckle_hits_batchku': (10, 0.4, 0.08, 60),
    'water_drops_reiyamanor': (10, 0.6, 0.08, 100), 'water_drop_paespedro': (10, 0.5, 0.1, 100),
    'creak_slow_harveyism': (6, 2.5, 0.4, 60), 'rope_creaks_6polnic': (6, 2.0, 0.4, 60),
}
PEAK_DB = -12.0


@lru_cache(maxsize=32)
def _load(name):
    log_access(FOUND / f'{name}.ogg')
    x, sr = sf.read(need(FOUND / f'{name}.ogg', f'found 录音 {name}'), dtype='float32', always_2d=True)
    x = x[:, :2] if x.shape[1] >= 2 else np.repeat(x, 2, axis=1)
    x = resample_sr(x, sr, SR)
    hp = CUT.get(name, (8, 0.5, 0.08, 80))[3]
    from scipy import signal
    sos = signal.butter(2, hp, 'highpass', fs=SR, output='sos')
    return signal.sosfilt(sos, x, axis=0).astype(np.float32)


@lru_cache(maxsize=32)
def _events(name):
    rise_db, max_s, gap_s, _ = CUT.get(name, (8, 0.5, 0.08, 80))
    x = _load(name)
    m = np.abs(x).max(axis=1)
    env = maximum_filter1d(m, int(0.003 * SR))
    env_db = 20 * np.log10(uniform_filter1d(env, int(0.003 * SR)) + 1e-9)
    floor = float(np.percentile(env_db, 15))
    lag = int(0.03 * SR)
    prev = np.concatenate([np.full(lag, env_db[0]), env_db[:-lag]])
    cand = np.flatnonzero((env_db - prev > rise_db) & (env_db > floor + 12))
    ons = []
    for c in cand:
        if not ons or c - ons[-1] > gap_s * SR:
            ons.append(int(c))
    evs = []
    for i, o in enumerate(ons):
        a = max(0, o - int(0.004 * SR))
        nxt = ons[i + 1] - int(0.004 * SR) if i + 1 < len(ons) else len(x)
        lim = min(nxt, o + int(max_s * SR), len(x))
        pk_i = o + int(np.argmax(env_db[o:min(lim, o + int(0.05 * SR))]))
        tail = np.flatnonzero(env_db[pk_i:lim] < floor + 6)
        b = pk_i + (int(tail[0]) if len(tail) else lim - pk_i)
        b = max(b, pk_i + int(0.02 * SR))
        seg = x[a:b].mean(axis=1)
        if len(seg) < int(0.015 * SR):
            continue
        sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        evs.append(dict(a=a, b=b, onset=o - a, peak_db=float(env_db[pk_i]),
                        centroid=float((sp * f).sum() / (sp.sum() + 1e-12)), dur=(b - a) / SR))
    return evs


def events(name):
    return list(_events(name))


@lru_cache(maxsize=32)
def _clean(name):
    """the events that do not click on their own (sampler.analysis.detect_clicks on the isolated hit): a pencil
    tapping the paper can start with a step that reads as a click in a mix; those are never picked"""
    from .analysis import detect_clicks
    pad = np.zeros((4800, 2), np.float32)
    ok = []
    for i in range(len(_events(name))):
        y, _ = _cut(name, _events(name)[i], 100, None)
        if not len(detect_clicks(np.concatenate([pad, y, pad]))):
            ok.append(i)
    return tuple(ok) or tuple(range(len(_events(name))))


def _choose(name, pick, seed):
    evs = _events(name)
    if not evs:
        raise ValueError(f'{name}: no events')
    clean = _clean(name)
    if isinstance(pick, int):
        return evs[clean[pick % len(clean)]]
    idx = list(clean)
    if pick in ('bright', 'dark'):
        idx.sort(key=lambda i: evs[i]['centroid'], reverse=pick == 'bright')
        idx = idx[:max(1, len(idx) // 3)]
    elif pick in ('short', 'long'):
        idx.sort(key=lambda i: evs[i]['dur'], reverse=pick == 'long')
        idx = idx[:max(1, len(idx) // 3)]
    elif pick == 'loud':
        idx.sort(key=lambda i: evs[i]['peak_db'], reverse=True)
        idx = idx[:max(1, len(idx) // 3)]
    return evs[idx[seed % len(idx)]]


def _cut(name, e, vel, length):
    y = _load(name)[e['a']:e['b']].copy()
    if length is not None:
        y = y[:e['onset'] + int(length * SR)]
    y = fade_edges(y, 0.002, min(0.04, 0.3 * len(y) / SR))
    pk = float(np.abs(y).max()) + 1e-9
    return y * (db2a(PEAK_DB - 16.0 * (1 - vel / 127.0) ** 1.4) / pk), e['onset']


def hit(name, pick='any', vel=100, seed=0, *, pitch=0.0, length=None):
    y, lead = _cut(name, _choose(name, pick, seed), vel, length)
    if pitch:
        r = 2 ** (pitch / 12)
        y = resample_by(y, r)
        lead = int(round(lead / r))
    return y.astype(np.float32), lead


def place(track, name, at, vel=100, seed=0, *, pick='any', pan=0.0, pitch=0.0, gain_db=0.0, length=None):
    y, lead = hit(name, pick, vel, seed, pitch=pitch, length=length)
    gl, gr = pan_gains(float(np.clip(pan, -1, 1)))
    y = y * np.array([gl, gr], np.float32) * np.sqrt(2) * db2a(gain_db)
    start = int(round(at * SR)) - lead
    need = start + len(y)
    if need > len(track):
        track = np.concatenate([track, np.zeros((need - len(track), 2), np.float32)])
    add_at(track, y.astype(np.float32), start)
    return track


def bed(name, dur, seed=0, *, fade_s=0.3):
    """a stretch of the recording (looped with crossfades if it is shorter), level as recorded"""
    x = _load(name)
    n = int(dur * SR)
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 2), np.float32)
    pos, xf = 0, int(0.25 * SR)
    while pos < n:
        a = int(rng.integers(0, max(1, len(x) - xf * 2)))
        seg = x[a:a + min(len(x) - a, n - pos + xf)].copy()
        seg[:xf] *= cos_ramp(min(xf, len(seg)))[:, None][:len(seg[:xf])]
        seg[-xf:] *= cos_ramp(min(xf, len(seg)), rising=False)[:, None][-len(seg[-xf:]):]
        add_at(out, seg, max(0, pos - (xf if pos else 0)))
        pos += len(seg) - xf
    return fade_edges(out, fade_s, fade_s)


@lru_cache(maxsize=32)
def _steady(name, dur, hop_s=0.02, n_best=4):
    """录音里连续发声、电平最平的几段（长 dur 秒）的起点样本：20 ms RMS 在整段里都不低于中位数 - 3 dB，
    平均电平不低于 75 % 分位 - 4 dB，按电平的标准差从小到大，彼此不重叠（找不到就放宽成平均电平最高的几段）；
    起点再对到段里的起音上。"""
    x = _load(name).astype(np.float64).mean(axis=1)
    h = int(hop_s * SR)
    k = len(x) // h
    db = 10 * np.log10((x[:k * h] ** 2).reshape(k, h).mean(axis=1) + 1e-12)
    w = max(1, int(round(dur / hop_s)))
    if k <= w:
        return (0,)
    p50, p75 = np.percentile(db, [50, 75])
    c = np.concatenate([[0.0], np.cumsum(db)])
    c2 = np.concatenate([[0.0], np.cumsum(db * db)])
    mean = (c[w:] - c[:-w]) / w
    std = np.sqrt(np.maximum(0.0, (c2[w:] - c2[:-w]) / w - mean ** 2))
    low = np.array([db[i:i + w].min() for i in range(len(mean))])
    ok = np.flatnonzero((low > p50 - 3) & (mean > p75 - 4))
    order = ok[np.argsort(std[ok], kind='stable')] if len(ok) else np.argsort(-mean, kind='stable')
    picked = []
    for i in order:
        if all(abs(int(i) - j) >= w for j in picked):
            picked.append(int(i))
        if len(picked) == n_best:
            break
    # 起点挪到这一段里第一次「听得到」的地方（2 ms 包络到前 0.3 s 峰值的 -12 dB 以内，与 check.measure_onset
    # 同一定义），往前留 4 ms：放在画面这一笔开始的那一刻，听到的起音也在那一刻
    env = maximum_filter1d(np.abs(x), int(0.002 * SR))
    out = []
    for i in picked:
        a = i * h
        seg = env[a:a + int(0.3 * SR)]
        k = int(np.argmax(seg >= seg.max() * db2a(-12.0)))
        out.append(max(0, a + k - int(0.004 * SR)))
    return tuple(out)


def stroke(name, dur, vel=100, seed=0, *, attack_s=0.004, release_s=0.06, pitch=0.0):
    """一个连续的动作（铅笔画一条线、一次擦）：录音里一段 dur 秒的连续发声（_steady 选出的几段之一，按 seed 轮换），
    首尾 attack_s / release_s 余弦淡入淡出，峰值按力度归一（与 hit() 同一标尺：vel 127 -> -12 dBFS）。
    起点就是第 0 个样本：放在画面这一笔开始的那一刻。"""
    starts = _steady(name, round(float(dur), 4))
    a = starts[seed % len(starts)]
    y = _load(name)[a:a + int(round(dur * SR))].copy()
    if pitch:
        y = resample_by(y, 2 ** (pitch / 12))
    na, nr = min(len(y), int(attack_s * SR)), min(len(y), int(release_s * SR))
    y[:na] *= cos_ramp(na)[:, None]
    y[len(y) - nr:] *= cos_ramp(nr, rising=False)[:, None]
    pk = float(np.abs(y).max()) + 1e-9
    return (y * (db2a(PEAK_DB - 16.0 * (1 - vel / 127.0) ** 1.4) / pk)).astype(np.float32)
