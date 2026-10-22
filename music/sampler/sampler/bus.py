"""分轨与收尾：一部片的渲染脚本反复用到的那几步（从一部示例片的各版渲染脚本里抽出来）。

结构：每个分轨（stem）一个函数，返回整片长度的 (n, 2)；`to_lufs` 把分轨拉到它的目标响度（分轨之间的相对平衡），
`automate` / `gate` 在干声上做增益曲线和静音窗口，混响放在最后（静音时房间的尾巴还在）；所有分轨相加后
`finish` 做 淡入淡出 → 母带（sampler.mix.master：-16 LUFS、真峰值限幅）→ 首尾 5/20 ms 淡出；`write_outputs`
写 score.wav、stems/*.wav 和带 SHA-256 的 manifest.json（源文件与产物）。

画面重音按「实测起音」对齐：`attack_index` 找片段里最强的 3 ms 能量上升（钢琴是击槌那一下，不是文件开头），
`place_on_attack` 把它放在指定时刻上（误差 0 样本），返回对齐记录供 events.json / 自检用。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import signal

from . import analysis, mix
from ._dsp import SR, add_at, cos_ramp, db2a, pan_gains, stereo


def n_samples(seconds) -> int:
    return int(round(seconds * SR))


def fit(x, n):
    """裁到 / 补零到 n 个样本，float32 立体声。"""
    x = stereo(x)
    if len(x) >= n:
        return x[:n].astype(np.float32)
    return np.concatenate([x, np.zeros((n - len(x), 2), np.float32)]).astype(np.float32)


def lufs_gain_db(x, target):
    cur = analysis.lufs(x)
    return float(target - cur) if np.isfinite(cur) else 0.0


def to_lufs(x, target):
    """整轨增益到目标整合响度（BS.1770）。分轨之间的平衡这样定，不在分轨上压限。"""
    return (stereo(x) * np.float32(db2a(lufs_gain_db(x, target)))).astype(np.float32)


def automate(x, *curves):
    """增益曲线（dB）：每条曲线 [(t, dB), ...] 在两端点之间线性插值、曲线外为 0 dB，多条相加。"""
    x = stereo(x)
    tt = np.arange(len(x)) / SR
    db = np.zeros(len(x))
    for pts in curves:
        if not pts:
            continue
        ts = np.array([a for a, _ in pts], float)
        ds = np.array([d for _, d in pts], float)
        inside = (tt >= ts[0]) & (tt <= ts[-1])
        db[inside] += np.interp(tt[inside], ts, ds)
    return (x * db2a(db)[:, None]).astype(np.float32)


def gate(x, windows, fade=0.12, ret=0.12):
    """静音窗口 [(a, b), ...]：a 处 fade 秒余弦收掉，b 之后 ret 秒慢慢回来（窗口外响着的东西不会突然冒出来）。"""
    x = np.array(stereo(x), copy=True)
    f = int(fade * SR)
    for a, b in windows:
        i0, i1 = int(a * SR), int(b * SR)
        if i0 >= len(x):
            continue
        k = min(f, i1 - i0, len(x) - i0)
        x[i0:i0 + k] *= cos_ramp(k, rising=False)[:, None]
        x[i0 + k:i1] = 0
        if i1 < len(x):
            g = min(int(ret * SR), len(x) - i1)
            x[i1:i1 + g] *= cos_ramp(g)[:, None]
    return x


def attack_index(clip, start=0.0, end=None, hp_hz=180.0):
    """片段里最强的正向 3 ms 能量上升所在样本（在 [start, end) 秒内找）。"""
    x = stereo(clip)
    y = signal.sosfilt(signal.butter(2, hp_hz, 'highpass', fs=SR, output='sos'), x.mean(axis=1))
    energy = np.convolve(y * y, np.ones(144) / 144, mode='same')
    flux = energy - np.roll(energy, 96)
    a = int(start * SR)
    b = len(x) if end is None else min(len(x), int(end * SR))
    return a + int(np.argmax(flux[a:b]))


def onset_index(clip, thr_db=-12.0, win_ms=2.0):
    """片段里第一次到达峰值 thr_db 以内的样本（2 ms 包络）：打击、拨弦类的「听到的起点」。"""
    from scipy.ndimage import maximum_filter1d
    x = stereo(clip)
    env = maximum_filter1d(np.abs(x).max(axis=1), max(1, int(win_ms / 1000 * SR)))
    return int(np.argmax(env >= env.max() * db2a(thr_db)))


def place_on_attack(track, clip, at, *, gain_db=0.0, pan=0.0, attack=None, how='attack'):
    """把 clip 放进 track，使它的起音（attack 样本，默认按 how 现量：'attack' 或 'onset'）正好落在 at 秒。
    返回 (track, 记录)。track 不够长时补长。"""
    clip = stereo(clip)
    if attack is None:
        attack = attack_index(clip) if how == 'attack' else onset_index(clip)
    if pan:
        gl, gr = pan_gains(pan)
        clip = clip * np.array([gl, gr], np.float32) * np.float32(np.sqrt(2))
    clip = clip * np.float32(db2a(gain_db))
    start = int(round(at * SR)) - int(attack)
    if start < 0:
        clip, attack, start = clip[-start:], attack + start, 0
    need = start + len(clip)
    track = stereo(track)
    if need > len(track):
        track = np.concatenate([track, np.zeros((need - len(track), 2), np.float32)])
    elif not track.flags.writeable:
        track = track.copy()
    add_at(track, clip.astype(np.float32), start)
    return track, dict(target_s=round(at, 6), attack_sample=int(attack), placed_start_sample=start,
                       error_samples=start + int(attack) - int(round(at * SR)))


def room(x, wet=0.17, seed=0, kind='room', predelay_s=0.018, lowcut_hz=200.0, highcut_hz=6500.0):
    """分轨最后加的空间（默认小房间）。"""
    return mix.reverb(x, kind, mix=wet, seed=seed, predelay_s=predelay_s, lowcut_hz=lowcut_hz,
                      highcut_hz=highcut_hz)


def finish(x, n, *, target_lufs=-16.0, true_peak_db=-1.2, fade_in=0.025, fade_out=1.0, edge_in=0.005,
           edge_out=0.02, **master_kw):
    """总线收尾：淡入淡出 → 母带 → 首尾再淡一下（master 的零相位滤波不保证首尾为 0）。返回 (y, info)。"""
    x = mix.fade(fit(x, n), fade_in, fade_out)
    y, info = mix.master(x, target_lufs=target_lufs, true_peak_db=true_peak_db, report=True, **master_kw)
    y = mix.fade(y, edge_in, edge_out)
    return fit(y, n), info


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_outputs(out_dir, master, stems=None, *, name='score', sources=(), extra=None):
    """写 <out>/<name>.wav（24-bit 48 kHz）、<out>/stems/*.wav、<out>/manifest.json。返回 manifest。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wav = out / f'{name}.wav'
    mix.write(str(wav), master)
    files = {wav.name: sha256(wav)}
    if stems:
        (out / 'stems').mkdir(exist_ok=True)
        for k, x in stems.items():
            p = out / 'stems' / f'{k}.wav'
            mix.write(str(p), x)
            files[f'stems/{p.name}'] = sha256(p)
    man = dict(sample_rate=SR, samples=int(len(master)), duration_s=round(len(master) / SR, 6),
               outputs=files, sources={str(Path(p).name): sha256(p) for p in sources})
    if extra:
        man.update(extra)
    (out / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str) + '\n',
                                       encoding='utf-8')
    return man
