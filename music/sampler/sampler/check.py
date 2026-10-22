"""乐谱自检：渲染出来以后量，不靠「应该」。我们听不到，所以每一条结论都要有数字。

    measure_onset(x, t, ...)        分轨里 t 附近的实测起音（2 ms 包络第一次到峰值 -12 dB 以内）与偏差
    sync_table(stems, expect)       [(id, 分轨名, 目标秒, 说明)] -> 每条的实测起音、偏差毫秒、偏差帧
    grid_alignment(events, grid)    events.json 里写在拍上的事件与网格的差（应 <= 1 ms）
    tail(x, t_cut)                  切点之后余音降到 -60 / -70 dBFS 的时间、末 10 ms 峰值
    jumps(x)                        电平跳变（0.6 s 前后比较，>= 6 dB）
    section_loudness(x, sections)   每段的整合响度（BS.1770）
    ebur128(path)                   用 ffmpeg 的 ebur128 滤镜复核整合响度和真峰值（$FFMPEG 或 PATH）
    summary(master, ...)            sampler.analysis.stats 加上面几项，写成一个 dict

jumps 与分段响度移植自一部示例片的 tools/score_check.py（原来写死了那部片的段落表，这里改成读时间线）。
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d

from . import analysis
from ._dsp import SR, a2db, stereo


def measure_onset(x, t, *, before=0.05, after=0.25, thr_db=-12.0, win_ms=2.0, floor_db=-70.0):
    """[t - before, t + after] 里第一次到达该窗口峰值 thr_db 以内的时刻（秒）；窗口里没声音时返回 None。"""
    x = stereo(x)
    a, b = max(0, int((t - before) * SR)), min(len(x), int((t + after) * SR))
    if b <= a:
        return None
    env = maximum_filter1d(np.abs(x[a:b]).max(axis=1), max(1, int(win_ms / 1000 * SR)))
    pk = float(env.max())
    if pk <= 10 ** (floor_db / 20):
        return None
    k = int(np.argmax(env >= pk * 10 ** (thr_db / 20)))
    return (a + k) / SR


def sync_table(stems, expect, fps, **kw):
    rows = []
    for item in expect:
        cid, stem, target = item[:3]
        note = item[3] if len(item) > 3 else ''
        on = measure_onset(stems[stem], target, **kw)
        dev = None if on is None else (on - target) * 1000
        rows.append(dict(id=cid, stem=stem, what=note, target_s=round(target, 4),
                         onset_s=None if on is None else round(on, 4),
                         dev_ms=None if dev is None else round(dev, 1),
                         frames=None if dev is None else round(dev / 1000 * fps, 2)))
    return rows


def grid_alignment(events, grid):
    """events.json 的 events：pos 写成 'bar.beat'（没有 +偏移）的事件，t 与网格的差（毫秒）。"""
    worst, rows = 0.0, []
    for e in events:
        pos = e.get('pos', '')
        if '+' in pos or '-' in pos:
            continue
        d = (e['t'] - grid.pos(pos)) * 1000
        worst = max(worst, abs(d))
        rows.append((e.get('type'), pos, round(d, 4)))
    return dict(max_abs_ms=round(worst, 4), checked=len(rows))


def tail(x, t_cut, *, win_s=0.05):
    """切点后的余音：50 ms RMS 第一次低于 -60 / -70 dBFS 的时间（相对切点），以及最后 10 ms 的峰值。"""
    x = stereo(x).astype(np.float64)
    a = int(t_cut * SR)
    seg = x[a:]
    hop = int(win_s * SR)
    n = len(seg) // hop
    if n == 0:
        return {}
    rms = np.sqrt((seg[:n * hop] ** 2).reshape(n, hop, 2).mean(axis=(1, 2)))
    db = a2db(rms)

    def first_below(thr):
        k = np.flatnonzero(db < thr)
        # 之后不再回升到阈值以上
        for i in k:
            if (db[i:] < thr).all():
                return round(i * win_s, 3)
        return None
    return dict(cut_s=round(t_cut, 4), tail_s=round(len(seg) / SR, 4),
                to_minus60_dbfs_s=first_below(-60), to_minus70_dbfs_s=first_below(-70),
                last_10ms_peak_dbfs=round(float(a2db(np.abs(x[-int(0.01 * SR):]).max())), 1),
                last_sample=float(np.abs(x[-1]).max()))


def jumps(x, thr=6.0):
    """电平跳变：0.3 s 窗、50 ms 步，比较前后各 0.6 s 的平均电平，>= thr dB 的局部最大。返回 [(dB, 秒)]。"""
    x = stereo(x)
    mm = (x.astype(np.float64) ** 2).mean(1)
    hop, win = int(0.05 * SR), int(0.3 * SR)
    c = np.concatenate([[0], np.cumsum(mm)])
    t = np.arange(0, len(mm) - win, hop)
    lv = 10 * np.log10((c[t + win] - c[t]) / win + 1e-10)
    tc = (t + win / 2) / SR
    k = int(0.6 / 0.05)
    d = np.array([lv[i:i + k].mean() - lv[i - k:i].mean() if k <= i < len(lv) - k else 0.0 for i in range(len(lv))])
    out = []
    for i in range(1, len(d) - 1):
        if abs(d[i]) >= thr and abs(d[i]) >= abs(d[i - 1]) and abs(d[i]) >= abs(d[i + 1]):
            if not out or tc[i] - out[-1][1] > 0.8:
                out.append((round(float(d[i]), 1), round(float(tc[i]), 2)))
            elif abs(d[i]) > abs(out[-1][0]):
                out[-1] = (round(float(d[i]), 1), round(float(tc[i]), 2))
    return out


def section_loudness(x, sections):
    """sections: [(名字, 起秒, 止秒)] -> {名字: LUFS}（短于 0.4 s 的段落补零后测）。"""
    x = stereo(x)
    out = {}
    for name, a, b in sections:
        seg = x[int(a * SR):int(b * SR)]
        out[name] = round(analysis.lufs(seg), 2) if len(seg) else None
    return out


def find_ffmpeg():
    from .prepare.field import find_ffmpeg as f
    return f()


def ebur128(path):
    """ffmpeg -af ebur128=peak=true 的汇总：{'I': LUFS, 'LRA': LU, 'TP': dBTP}。"""
    cmd = [find_ffmpeg(), '-hide_banner', '-nostats', '-i', str(path), '-af', 'ebur128=peak=true', '-f', 'null', '-']
    r = subprocess.run(cmd, capture_output=True, text=True)
    txt = r.stderr
    summ = txt[txt.rfind('Summary:'):]
    def grab(pat):
        mt = re.search(pat, summ)
        return float(mt.group(1)) if mt else None
    return dict(I=grab(r'I:\s+(-?[\d.]+) LUFS'), LRA=grab(r'LRA:\s+(-?[\d.]+) LU'),
                TP=grab(r'Peak:\s+(-?[\d.]+) dBFS'), ffmpeg=shutil.which(cmd[0]) or cmd[0])


def summary(master, *, grid=None, cut_s=None, sections=None):
    s = analysis.stats(master)
    out = dict(lufs=s['lufs'], true_peak_dbtp=s['true_peak_dbtp'], peak_dbfs=s['peak_dbfs'],
               clipped_samples=s['clipped_samples'], click_candidates=s['click_candidates'],
               click_times=s['click_times'], lr_corr=s['lr_corr'], first_sample=s['first_sample'],
               last_sample=s['last_sample'], edge_10ms_dbfs=s['edge_10ms_dbfs'], dur_s=s['dur_s'],
               harsh_2k5_4k_rel_total_db=s['harsh_2k5_4k_rel_total_db'], jumps=jumps(master))
    if cut_s is not None:
        out['tail'] = tail(master, cut_s)
    if sections:
        out['sections_lufs'] = section_loudness(master, sections)
    return out


if __name__ == '__main__':          # python -m sampler.check some.wav  -> 一行汇总 + ffmpeg 复核
    import json
    import soundfile as sf
    p = Path(sys.argv[1])
    y, sr = sf.read(p, dtype='float32', always_2d=True)
    assert sr == SR, sr
    print(json.dumps(dict(summary(y), ffmpeg_ebur128=ebur128(p)), ensure_ascii=False, indent=1))
