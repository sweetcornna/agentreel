"""示例片《纸上一线》的配乐：照 films/demo-paper/timeline.json 的拍网格写，渲染到 out/music/。

    python films/demo-paper/score/score.py                 # -> out/music/score.wav、events.json、stems/、manifest.json
    python films/demo-paper/score/score.py --check         # 另外打印自检（响度、真峰值、同步表、余音）和 ffmpeg 复核
    python films/demo-paper/score/score.py --out <目录> --no-stems

采样用 music/sampler/fetch_samples.py 取（只取这部乐谱用到的文件：--files films/demo-paper/score/samples.txt），
采样根是 $AGENTREEL_SAMPLES（默认 music/sampler/samples/）。全部声音都是 CC0 或合成的，公开时不要求署名（见 README.md）。

D 大调，72 BPM，4/4（一拍 25 帧）。画面 6 小节 + 1 小节余音，共 700 帧 = 23.333 s：

    1.1-3.1  成纸   pad 从 D 的空五度慢慢起来；1.3 1.4 2.1 2.2 铅笔四笔（每笔一段连续的笔声，声像跟着笔走：
                    上边从左到右、右边在右、下边从右到左、左边在左）；2.3 成纸：速写本翻落的一声 + 钢琴 Dadd9
                    分解和弦，2.3+2f / +4f 两个对角的铅笔短线各一下
    3.1-5.1  一线   3.1 红线进场：竖琴从 G3 上行到 D5（Gmaj7），之后每拍后半拍一个竖琴音跟着线绕卡片；
                    4.1 线停在「帧号」下：手铃 F#5（Bm7）；4.3 第二句：钢琴 F#4 A4 B4 C#5 八分音符（A 上），
                    落到 5.1 的 D5
    5.1-7.1  片尾   5.1 线绷直：弦乐 D2 A2 F#3 E4 A4 宽和弦 + 钢琴低音 D；5.2+20f 绷直回颤：竖琴 D5 + A5；
                    5.3 名字落下：小提琴 F#5 进来、钢琴 E5，5.3+18f 仓库地址：钢琴 F#5；6.4 起随画面渐弱
    7.1-8.1  余音   7.1 所有干声 0.25 s 收住（钢琴制音、弓离弦），只剩混响尾巴，8.1 前淡到 0

画面上的重音按实测起音对齐（sampler.bus.place_on_attack / folk 与 found 自带的起音对齐），误差记在
events.json 的 measured.sync 里。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

FILM = Path(__file__).resolve().parents[1]
REPO = FILM.parents[1]
if importlib.util.find_spec('sampler') is None:     # 没装包时直接从仓库里用
    sys.path.insert(0, str(REPO / 'music' / 'sampler'))

from sampler import _paths, bus, check, folk, found, mix, nature, pads, strings  # noqa: E402
from sampler._dsp import add_at  # noqa: E402
from sampler.grid import EventLog, Grid, Human, m  # noqa: E402

G = Grid(FILM / 'timeline.json')
N = bus.n_samples(G.duration)                        # 700 帧 -> 1 120 000 个样本
CUT = G.cue('cut')['t']                              # 7.1 = 20.0 s
FPS = G.fps

# 画面里的两个量（films/demo-paper/src/timing.ts）：铅笔一笔画 19 帧；卡片左右边的 x（画面宽 1920）
STROKE_FRAMES = 19
CARD_X = (390.0, 1530.0)
SIDES = ('top', 'right', 'bottom', 'left')

# 分轨之间的平衡：每个分轨的整合响度（LUFS，干声、加混响之前）。母带最后再把总和拉到 timeline 的 -16 LUFS
LEVEL = dict(strings=-21.0, piano=-21.5, pad=-30.0, harp=-24.0, chime=-25.0, pencil=-25.0, paper=-31.0)
# 分轨的空间：(混响, 湿声比例)
SPACE = dict(strings=('hall', 0.30), piano=('hall', 0.26), pad=('hall', 0.22), harp=('hall', 0.32),
             chime=('hall', 0.36), pencil=('room', 0.14), paper=('room', 0.20))


def x_pan(x):
    """画面横坐标 -> 声像（-1 左 .. +1 右），收窄到 0.8：边上的东西不贴死一侧。"""
    return 0.8 * (x / 960.0 - 1.0)


def pan_sweep(clip, p0, p1):
    """声像随时间从 p0 线性走到 p1（等功率，和 folk / found 的 pan 同一定标：居中时两边各 0 dB）。"""
    clip = np.asarray(clip, np.float32)
    p = np.linspace(p0, p1, len(clip))
    th = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return (clip * np.stack([np.cos(th), np.sin(th)], axis=1) * np.sqrt(2)).astype(np.float32)


# ----------------------------------------------------------------------------- 分轨
def pencil(log, sync):
    """1.3-2.3 每拍一笔：录音里一段连续的笔声，长度等于画面的一笔；2.3+2f / +4f 两个角上的短线。"""
    tr = np.zeros((N, 2), np.float32)
    cue = G.cue('pencil')
    n = int(round((cue['end'] - cue['t']) / G.spb))
    x0, x1 = CARD_X
    pans = {'top': (x_pan(x0), x_pan(x1)), 'right': (x_pan(x1),) * 2, 'bottom': (x_pan(x1), x_pan(x0)),
            'left': (x_pan(x0),) * 2}
    dur = G.at_frame(STROKE_FRAMES)
    for i in range(n):
        side = SIDES[i % 4]
        at = cue['t'] + i * G.spb
        clip = found.stroke('scribble_long_waldram', dur, vel=96 if side in ('top', 'bottom') else 90, seed=i)
        clip = pan_sweep(clip, *pans[side])
        tr, rec = bus.place_on_attack(tr, clip, at, attack=0)
        log.event('pencil', at, side=side, frames=STROKE_FRAMES, pan=[round(p, 3) for p in pans[side]], **rec)
        sync.append((f'pencil.{side}', 'pencil', at, f'铅笔第 {i + 1} 笔（{side}）'))
    paper_t = G.cue('paper')['t']
    for k, (off, x) in enumerate(((2, x0), (4, x1))):          # Burst：左上角 +2f、右下角 +4f
        at = paper_t + G.at_frame(off)
        clip, lead = found.hit('pencil_writing_rivernile7', 'short', 70, seed=3 + k)
        tr, _ = bus.place_on_attack(tr, clip, at, attack=lead, pan=x_pan(x))
        log.event('pencil_tick', at, pos=f'2.3+{off}f', corner='top_left' if k == 0 else 'bottom_right')
    return bus.fit(tr, N)


def paper(log, sync):
    """2.3 成纸：速写本翻落的一声（偏暗的那几下），起音对在 2.3。"""
    tr = np.zeros((N, 2), np.float32)
    at = G.cue('paper')['t']
    clip, lead = found.hit('sketchbook_flips_owlstorm', 'dark', 100, seed=1)
    tr, rec = bus.place_on_attack(tr, clip, at, attack=lead, pan=0.05)
    log.event('paper', at, source='sketchbook_flips_owlstorm', **rec)
    sync.append(('paper', 'paper', at, '成纸：纸落下'))
    return bus.fit(tr, N)


def piano_notes(h):
    """VCSL Steinway B（CC0）。(start, dur, midi, vel, kw)；ring_s 小 = 抬踏板 / 制音。"""
    P = G.pos
    out = []

    def add(t, end, name, vel, ring=0.3, human=True):
        st = h.at(t, -3, 6) if human else t
        out.append((st, end - st, m(name), h.vel(vel, 1.5), {'ring_s': ring}))

    # 2.3 成纸：Dadd9 自下而上分解（每音 22 ms），第一音正好在拍上
    for j, (name, vel) in enumerate((('D3', 46), ('A3', 42), ('E4', 44), ('F#4', 48))):
        add(P('2.3') + 0.022 * j, P('3.1'), name, vel, ring=0.35, human=False)
    add(P('3.1'), P('4.1'), 'G2', 40, ring=0.3)                       # 竖琴上行下面的低音
    add(P('4.1'), P('4.3'), 'B1', 36, ring=0.25)                      # 「帧号」：Bm 的根音，很轻
    add(P('4.1') + 0.02, P('4.3'), 'F#2', 32, ring=0.25)
    # 4.3 第二句：A 上的四个八分音符，踏板一直踩到 5.1
    add(P('4.3'), P('5.1'), 'A1', 38, ring=0.12)
    for k, (name, vel) in enumerate((('F#4', 50), ('A4', 47), ('B4', 52), ('C#5', 49))):
        add(P('4.3') + 0.5 * k * G.spb, P('5.1'), name, vel, ring=0.12, human=k > 0)
    # 5.1 线绷直：低音 D 与高音 D5（乐句落在这里），一直响到 7.1 制音
    for name, vel in (('D2', 52), ('D3', 44), ('A3', 40), ('D5', 56)):
        add(P('5.1'), CUT, name, vel, ring=0.0, human=name != 'D2')
    add(P('5.3'), CUT, 'E5', 40, ring=0.0)                            # 名字落下
    add(P('5.3+18f'), CUT, 'F#5', 38, ring=0.0)                       # 仓库地址打出来
    return out


def harp_notes(h):
    P = G.pos
    out = []
    # 3.1 红线从左下甩进来：G3 B3 D4 F#4 A4 B4 D5，70 ms 一个，越来越响，声像从左往中间走
    run = ('G3', 'B3', 'D4', 'F#4', 'A4', 'B4', 'D5')
    for j, name in enumerate(run):
        st = P('3.1') + 0.070 * j
        out.append((st, 0.4, m(name), 46 + 3.4 * j, {'pan': -0.5 + 0.1 * j, 'ring_s': 1.8}))
    # 线绕卡片：每拍的后半拍一个音（3.2 3.3 3.4 之后），声像跟着线绕：右、下、左
    for pos, name, pan in (('3.2+0.5', 'B4', 0.35), ('3.3+0.5', 'A4', 0.1), ('3.4+0.5', 'F#4', -0.3)):
        out.append((h.at(P(pos), -3, 5), 0.5, m(name), h.vel(44, 2), {'pan': pan, 'ring_s': 1.5}))
    # 5.2+20f 线绷直回颤：D5 + A5（6 ms），D3 托底
    t = P('5.2+20f')
    out.append((t, 0.5, m('D5'), 70, {'pan': 0.0, 'ring_s': 2.4}))
    out.append((t + 0.006, 0.5, m('A5'), 62, {'pan': 0.08, 'ring_s': 2.4}))
    out.append((t + 0.012, 0.5, m('D3'), 50, {'pan': -0.05, 'ring_s': 2.4}))
    return out


def string_notes():
    """VSCO 2 CE 弦乐（CC0），弓上长音。5.1 宽和弦，5.3 小提琴 F#5 进来，7.1 一起收。"""
    P = G.pos
    t, e = P('5.1'), CUT
    return [(t, e - t, m('D2'), 54, {'sec': 'cb'}), (t, e - t, m('A2'), 50, {'sec': 'vc'}),
            (t, e - t, m('F#3'), 46, {'sec': 'va'}), (t, e - t, m('E4'), 44, {'sec': 'vn'}),
            (t, e - t, m('A4'), 42, {'sec': 'vn'}), (P('5.3'), e - P('5.3'), m('F#5'), 40, {'sec': 'vn'})]


def pad_notes():
    """合成 pad：和声的底色，和弦跟着段落换（彼此重叠 0.15 s）。"""
    P = G.pos
    chords = [(0.25, P('2.3'), ('D3', 'A3'), 58),
              (P('2.3'), P('3.1'), ('D3', 'A3', 'E4'), 56),
              (P('3.1'), P('4.1'), ('G2', 'B3', 'D4', 'F#4'), 54),
              (P('4.1'), P('4.3'), ('B2', 'D4', 'F#4', 'A4'), 52),
              (P('4.3'), P('4.4'), ('A2', 'D4', 'E4'), 52),
              (P('4.4'), P('5.1'), ('A2', 'C#4', 'E4'), 52),
              (P('5.1'), CUT, ('D3', 'A3', 'F#4'), 50)]
    return [(a, b - a + 0.15, m(n), v) for a, b, ns, v in chords for n in ns], chords


# 给画面读的音乐事件（拍上的那些；每个音的实际起点另见 events.json 的 notes）
MUSIC_EVENTS = [
    ('chord', '2.3', dict(stem='piano', pitches=['D3', 'A3', 'E4', 'F#4'], what='成纸：Dadd9 分解和弦')),
    ('run', '3.1', dict(stem='harp', pitches=['G3', 'B3', 'D4', 'F#4', 'A4', 'B4', 'D5'], step_s=0.07,
                        what='红线进场：竖琴上行')),
    ('thread', '3.2+0.5', dict(stem='harp', pitch='B4')), ('thread', '3.3+0.5', dict(stem='harp', pitch='A4')),
    ('thread', '3.4+0.5', dict(stem='harp', pitch='F#4')),
    ('phrase', '4.3', dict(stem='piano', pitch='F#4')), ('phrase', '4.3+0.5', dict(stem='piano', pitch='A4')),
    ('phrase', '4.4', dict(stem='piano', pitch='B4')), ('phrase', '4.4+0.5', dict(stem='piano', pitch='C#5')),
    ('chord', '5.1', dict(stem='strings+piano', pitches=['D2', 'A2', 'F#3', 'E4', 'A4', 'D5'], what='线绷直：宽和弦')),
    ('taut', '5.2+20f', dict(stem='harp', pitches=['D5', 'A5', 'D3'], what='绷直回颤')),
    ('name', '5.3', dict(stem='piano+strings', pitches=['E5', 'F#5'], what='名字落下')),
    ('repo', '5.3+18f', dict(stem='piano', pitch='F#5', what='仓库地址')),
    ('cut', '7.1', dict(what='干声收住，余音到 8.1')),
]


def render(log, sync, curves):
    h = Human(2026)
    for type_, pos, kw in MUSIC_EVENTS:
        log.event(type_, G.pos(pos), pos=pos, **kw)
    stems = {}
    stems['pencil'] = pencil(log, sync)
    stems['paper'] = paper(log, sync)

    pn = piano_notes(h)
    stems['piano'] = folk.render('grand_piano', pn, duration=G.duration, damp_s=0.08, seed=11)
    for st, du, mi, vel, kw in pn:
        log.note('piano', 'piano', st, du, mi, vel, ring_s=kw['ring_s'])
    sync += [('paper.piano', 'piano', G.pos('2.3'), '成纸：钢琴 Dadd9 第一音'),
             ('word2', 'piano', G.pos('4.3'), '第二句：F#4'),
             ('end.piano', 'piano', G.pos('5.1'), '线绷直：钢琴低音 D')]

    hn = harp_notes(h)
    stems['harp'] = folk.render('folk_harp', hn, duration=G.duration, seed=12)
    for st, du, mi, vel, kw in hn:
        log.note('harp', 'harp', st, du, mi, vel, pan=kw['pan'], ring_s=kw['ring_s'])
    sync += [('line', 'harp', G.pos('3.1'), '红线进场：竖琴上行第一音'),
             ('taut', 'harp', G.pos('5.2+20f'), '线绷直回颤：竖琴 D5')]

    tr = np.zeros((N, 2), np.float32)
    at = G.cue('word1')['t']
    tr, rec = bus.place_on_attack(tr, nature.sfx('hand_chime', note='F#5', seed=0), at, how='onset', pan=0.1)
    stems['chime'] = bus.fit(tr, N)
    log.event('chime', at, note='F#5', **rec)
    log.note('chime', 'chime', at, 2.0, m('F#5'), 80)
    sync.append(('word1', 'chime', at, '「帧号」：手铃 F#5'))

    sn = string_notes()
    s = strings.render_strings(sn, duration=G.duration, attack_s=0.35, release_s=0.5, seed=13)
    swell = [(G.pos('5.1'), -7.0), (G.pos('5.3'), 0.0), (G.pos('6.4'), 0.0), (CUT, -9.0)]
    stems['strings'] = bus.automate(s, swell)
    curves['strings_gain_db'] = swell
    for st, du, mi, vel, kw in sn:
        log.note('strings', 'strings', st, du, mi, vel, sec=kw['sec'])
    sync.append(('end', 'strings', G.pos('5.1'), '线绷直：弦乐（弓上慢起，起音偏晚是正常的）'))

    notes, chords = pad_notes()
    p = pads.pad(notes, attack=1.0, release=0.7, brightness=0.35, motion=0.3, air=0.03, width=0.9,
                 duration=G.duration, seed=14)
    lift = [(0.25, -6.0), (G.pos('2.3'), 0.0), (G.pos('5.1'), 0.0), (G.pos('5.3'), -4.0), (CUT, -6.0)]
    stems['pad'] = bus.automate(p, lift)
    curves['pad_gain_db'] = lift
    for a, b, ns, v in chords:
        for n in ns:
            log.note('pad', 'pad', a, b - a, m(n), v)

    # 平衡（干声）→ 7.1 收住（0.25 s）→ 空间。gate 在混响前：切点之后房间的尾巴还在
    dry = {}
    for k in stems:
        y = bus.to_lufs(bus.fit(stems[k], N), LEVEL[k])
        y = bus.gate(y, [(CUT, G.duration)], fade=0.25)
        dry[k] = y
        kind, wet = SPACE[k]
        stems[k] = bus.fit(mix.reverb(y, kind, mix=wet, seed=len(k), predelay_s=0.015,
                                      lowcut_hz=160.0, highcut_hz=7000.0), N)
    return stems, dry


def main(argv=None):
    ap = argparse.ArgumentParser(description='《纸上一线》配乐')
    ap.add_argument('--out', default=str(FILM / 'out' / 'music'))
    ap.add_argument('--no-stems', action='store_true')
    ap.add_argument('--check', action='store_true', help='打印自检和 ffmpeg ebur128 复核')
    a = ap.parse_args(argv)
    out = Path(a.out)

    log, sync, curves = EventLog(G), [], {}
    for c in G.cues:                # 重音类提示是硬同步（起音对在点上），铅笔一串与乐句是软同步
        log.marker(c['id'], c['at'], kind='hard_sync' if c['kind'] in ('hit', 'accent', 'cut') else 'soft_sync',
                   cue_kind=c['kind'], note=c.get('note', ''))
    try:
        stems, dry = render(log, sync, curves)
    except _paths.MissingSamples as e:
        print(e, file=sys.stderr)
        print('这部乐谱用到的采样：music/sampler/.venv/bin/python music/sampler/fetch_samples.py --files '
              'films/demo-paper/score/samples.txt',
              file=sys.stderr)
        return 3
    log.cut(CUT, G.duration, choke=True, tails=True, note='7.1 干声 0.25 s 收住，混响尾巴继续响到 8.1')
    for k, pts in curves.items():
        log.curve(k, pts)

    total = np.zeros((N, 2), np.float32)
    for y in stems.values():
        add_at(total, y, 0)
    tgt = G.loudness.get('integratedLUFS', -16.0)
    tp = G.loudness.get('truePeakDbtp', -1.0) - 0.2               # 留 0.2 dB 给 24 bit 量化与其他测量器
    master, info = bus.finish(total, N, target_lufs=tgt, true_peak_db=tp, fade_in=0.02, fade_out=1.0)

    sources = [_paths.SAMPLES / r for r in _paths.accessed()]
    sync_rows = check.sync_table(dry, sync, FPS)
    summ = check.summary(master, cut_s=CUT, sections=[(s['id'], s['t'], s['end']) for s in G.sections])
    measured = dict(master=info, sync=sync_rows, summary=summ, samples=_paths.accessed())
    man = bus.write_outputs(out, master, None if a.no_stems else stems, name='score', sources=sources,
                            extra=dict(film='demo-paper', target_lufs=tgt, true_peak_ceiling_dbtp=tp,
                                       levels=LEVEL, space=SPACE))
    log.write(out / 'events.json', measured=dict(measured, outputs=man['outputs']))
    print(f"score.wav  {man['outputs']['score.wav']}  {man['duration_s']} s  "
          f"{summ['lufs']} LUFS  {summ['true_peak_dbtp']} dBTP")
    if a.check:
        worst = max(abs(r['dev_ms']) for r in sync_rows if r['dev_ms'] is not None and r['stem'] != 'strings')
        print(json.dumps(dict(summary=summ, master=info), ensure_ascii=False, indent=1))
        for r in sync_rows:
            print(f"  {r['id']:14} {r['stem']:8} {r['target_s']:8.3f}  {r['onset_s']}  {r['dev_ms']} ms  {r['what']}")
        print('worst sync (除弦乐) ms', worst)
        print('grid', check.grid_alignment(log.doc()['events'], G))
        print('ffmpeg ebur128', check.ebur128(out / 'score.wav'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
