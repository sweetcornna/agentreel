"""拍网格与事件表：配乐和画面对着同一张拍子表。

    g = Grid('films/<片>/timeline.json')
    g.pos('2.3')          -> 秒（「小节.拍」从 1 起；'2.3+0.5' 晚半拍，'2.3+4f' 晚 4 帧，'2.3-2f' 早 2 帧）
    g.t(2, 2.0)           -> 秒（小节从 1 起、拍是小节内从 0 起的偏移，和原来各版 score*.py 的 t() 相同）
    g.frame(秒) / g.at_frame(帧) / g.label(秒) -> 'bar.beat+frac'
    g.span('1.3-2.3')     -> (起, 止) 秒；g.cue('pencil') -> 时间线 cues 里那一条，带 t / end
    m('F#4') -> 66        音名（# 与 b，C4 = 60）
    h = Human(seed); h.at(t, -5, 8); h.vel(60, 2)      固定种子的人性化（毫秒、力度）

    log = EventLog(g)
    log.event('pencil', g.pos('1.3'), side='top')     画面可以读的事件
    log.note('piano', 'piano', start, dur, midi, vel)  每个音
    log.marker('paper', '2.3', kind='hit')             时间线上的同步点（区间 '1.3-2.3' 也行）
    log.cut(g.pos('7.1'), g.duration)                  切点（之后只留余音）
    log.write('films/<片>/out/music/events.json', measured=...)

events.json 的顶层字段与 filmsynth 相同：version sampleRate fps bpm beatsPerBar bars tempoRamps end duration beats
barTimes beatTimes markers cuts events notes curves measured；时间一律是秒，frame 是浮点帧号（画面 fps）。细节不同：markers
多 frame（区间另带 end / end_frame），curves 是 [{t, frame, v}] 折点（filmsynth 是 {hop, t0, values}），notes 没有
beat、hz。逐字段的对照见 docs/TIMELINE.md 第 10 节。
时间线里的 framesPerBeat 必须等于 fps × 60 / bpm（不等就报错：网格和帧对不上）。
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
_NAMES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
_POS = re.compile(r'^\s*(\d+)(?:\.(\d+(?:\.\d+)?))?\s*(?:([+-])\s*(\d+(?:\.\d+)?)\s*(f?))?\s*$')


def m(name):
    """音名 -> MIDI（'C4' = 60，'Bb1'、'F#4'）；数字原样返回。"""
    if not isinstance(name, str):
        return name
    s = name.strip()
    pc = _PC[s[0].upper()]
    i = 1
    while i < len(s) and s[i] in '#b':
        pc += 1 if s[i] == '#' else -1
        i += 1
    return pc + 12 * (int(s[i:]) + 1)


def note_name(midi) -> str:
    k = int(round(midi))
    return f'{_NAMES[k % 12]}{k // 12 - 1}'


class Human:
    """固定种子的人性化。和原来的乐谱一样用 random.Random：同一种子、同样的调用顺序，结果逐位相同。"""

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def at(self, t, lo_ms=-5.0, hi_ms=8.0):
        return max(0.0, t + self.rng.uniform(lo_ms, hi_ms) / 1000.0)

    def vel(self, v, spread=2.0):
        return v + self.rng.uniform(-spread, spread)

    def uniform(self, a, b):
        return self.rng.uniform(a, b)


class Grid:
    def __init__(self, timeline):
        if isinstance(timeline, (str, Path)):
            self.path = Path(timeline)
            d = json.loads(self.path.read_text(encoding='utf-8'))
        else:
            self.path, d = None, dict(timeline)
        self.doc = d
        self.fps = float(d['fps'])
        self.bpm = float(d['bpm'])
        self.bpb = int(d.get('beatsPerBar', 4))
        self.spb = 60.0 / self.bpm
        self.fpb = self.fps * self.spb
        if 'framesPerBeat' in d and abs(float(d['framesPerBeat']) - self.fpb) > 1e-6:
            raise ValueError(f"framesPerBeat {d['framesPerBeat']} != fps*60/bpm = {self.fpb}")
        self.bars = int(d.get('bars', 0))
        self.total_frames = int(d['totalFrames']) if 'totalFrames' in d else None
        self.duration = self.total_frames / self.fps if self.total_frames is not None else self.bars * self.bpb * self.spb
        self.end = self.pos(d['filmEnd']) if 'filmEnd' in d else self.duration
        self.loudness = d.get('loudness', {})

    # ------------------------------------------------------------------ 时间
    def t(self, bar, beat=0.0):
        """小节从 1 起，beat 是小节内从 0 起的拍数（原 score*.py 的写法）。"""
        return ((bar - 1) * self.bpb + beat) * self.spb

    def pos(self, s) -> float:
        """'小节.拍[+偏移]' -> 秒。拍从 1 起，可带小数（'2.3.5' 不合法，用 '2.3+0.5'）；偏移以拍计，带 f 以帧计。"""
        if isinstance(s, (int, float)):
            return float(s)
        mt = _POS.match(str(s))
        if not mt or int(mt.group(1)) < 1:
            raise ValueError(f'bad position {s!r}; write "bar.beat", "bar.beat+0.5" or "bar.beat+4f"')
        bar, beat = int(mt.group(1)), float(mt.group(2) or 1)
        t = self.t(bar, beat - 1)
        if mt.group(3):
            off = float(mt.group(4))
            off = off / self.fps if mt.group(5) else off * self.spb
            t += off if mt.group(3) == '+' else -off
        return t

    def span(self, s):
        """'1.3-2.3' -> (起, 止)；单个位置 -> (t, t)。"""
        s = str(s)
        for k in [i for i, ch in enumerate(s) if ch == '-']:     # 'a-b'：两边都是合法位置且 b >= a 才算区间
            try:
                a, b = self.pos(s[:k]), self.pos(s[k + 1:])
            except ValueError:
                continue
            if b >= a and '.' in s[k + 1:]:
                return a, b
        t = self.pos(s)
        return t, t

    def frame(self, t):
        return t * self.fps

    def at_frame(self, f):
        return f / self.fps

    def label(self, t) -> str:
        """秒 -> 'bar.beat'（落在拍上）或 'bar.beat+0.25'（拍后多少拍，保留 3 位）。"""
        k = t / self.spb
        b = math.floor(k + 1e-6)
        frac = max(0.0, k - b)
        bar, beat = b // self.bpb + 1, b % self.bpb + 1
        return f'{bar}.{beat}' if frac < 1e-6 else f'{bar}.{beat}+' + f'{frac:.3f}'.rstrip('0').rstrip('.')

    def beats(self, until=None):
        """每拍起点（秒），含 until（默认整片时长）那一拍。"""
        until = self.duration if until is None else until
        n = int(round(until / self.spb))
        return [i * self.spb for i in range(n + 1)]

    # ------------------------------------------------------------------ 时间线条目
    def _items(self, key):
        out = []
        for c in self.doc.get(key, []):
            c = dict(c)
            a = c.get('at', c.get('from'))
            b = c.get('to')
            t0, t1 = self.span(a) if a is not None else (None, None)
            if b is not None:
                t1 = self.pos(b)
            c['t'], c['end'] = t0, t1
            out.append(c)
        return out

    @property
    def cues(self):
        return self._items('cues')

    @property
    def shots(self):
        return self._items('shots')

    @property
    def sections(self):
        return self._items('sections')

    def cue(self, cid):
        for c in self.cues:
            if c['id'] == cid:
                return c
        raise KeyError(f'no cue {cid!r} in the timeline')


class EventLog:
    def __init__(self, grid: Grid, sample_rate=48000):
        self.g = grid
        self.sr = sample_rate
        self.events, self.notes, self.markers, self.cuts, self.curves = [], [], [], [], {}

    def _stamp(self, t, pos=None):
        t = float(t)
        return dict(t=round(t, 6), frame=round(self.g.frame(t), 3), pos=pos or self.g.label(t))

    def event(self, type_, t, pos=None, **kw):
        e = dict(self._stamp(t, pos), type=type_, **kw)
        self.events.append(e)
        return e

    def note(self, track, stem, start, dur, midi, vel, pos=None, **kw):
        e = dict(track=track, stem=stem, **self._stamp(start, pos), end=round(float(start + dur), 6),
                 midi=round(float(midi), 3), pitch=note_name(midi), vel=round(float(vel), 2), **kw)
        e['start'] = e.pop('t')
        self.notes.append(e)
        return e

    def marker(self, name, pos, kind='sync', **kw):
        """pos 可以是位置（'2.3'）或区间（'1.3-2.3'，另记 end）。"""
        t, end = self.g.span(pos)
        e = dict(name=name, pos=str(pos), t=round(t, 6), frame=round(self.g.frame(t), 3), kind=kind, **kw)
        if end != t:
            e.update(end=round(end, 6), end_frame=round(self.g.frame(end), 3))
        self.markers.append(e)
        return e

    def cut(self, start, end, **kw):
        self.cuts.append(dict(start=round(float(start), 6), end=round(float(end), 6), **kw))

    def curve(self, name, points):
        """[(t, value)] -> curves[name] = [{t, frame, v}]"""
        self.curves[name] = [dict(t=round(float(t), 6), frame=round(self.g.frame(t), 3), v=round(float(v), 4))
                             for t, v in points]

    def doc(self, measured=None):
        g = self.g
        beats = g.beats()
        bar_times, beat_times = [], []
        for i, t in enumerate(beats):
            bar, beat = i // g.bpb + 1, i % g.bpb + 1
            beat_times.append(dict(bar=bar, beat=beat, t=round(t, 6), frame=round(g.frame(t), 3)))
            if beat == 1:
                bar_times.append(dict(bar=bar, t=round(t, 6), frame=round(g.frame(t), 3)))
        key = lambda e: (e.get('t', e.get('start', 0.0)), e.get('type', e.get('track', '')))  # noqa: E731
        return dict(version=1, sampleRate=self.sr, fps=g.fps, bpm=g.bpm, beatsPerBar=g.bpb, bars=g.bars,
                    tempoRamps=[], end=round(g.end, 6), duration=round(g.duration, 6),
                    beats=[round(t, 6) for t in beats], barTimes=bar_times, beatTimes=beat_times,
                    markers=sorted(self.markers, key=lambda e: e['t']), cuts=self.cuts,
                    events=sorted(self.events, key=key), notes=sorted(self.notes, key=key), curves=self.curves,
                    measured=measured or {})

    def write(self, path, measured=None):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.doc(measured), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        return p
