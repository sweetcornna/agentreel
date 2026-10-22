"""Tone check for a grade: are the film's darks its ink, and is its dim and its bright light the film's hue?

  python3 scripts/qa_tone.py ../../films/demo-light/out/demo-light-1080p.mp4             # whole film, per shot, every 4th frame
  python3 scripts/qa_tone.py out/seg.mp4 --film ../../films/demo-light --from 5.1         # a render.ts --from 5.1 segment
  python3 scripts/qa_tone.py ../../films/demo-light/out/stills/*.png                     # per image (name carries the position, 5.1+3f_1080p.png)

Needs Python 3.10+ with numpy, and ffmpeg (FFMPEG or on the PATH). Without a numpy install:
  uv run --with numpy python scripts/qa_tone.py <...>

Targets come from <film>/tone.json (all optional; command-line flags override):
  { "ink": [13, 12, 16], "inkTol": 5, "satMin": 0.3, "dimHue": [12, 38], "lightHue": [14, 40], "exclude": ["S05"] }

Three buckets by display luma (Rec.709 weights on 8-bit sRGB):
  deep  L in [4, 16):   unlit space. Must read as the ink: every channel of the bucket's mean within inkTol
                        code values (judged when the bucket holds >= 2 % of the pixels).
  dim   L in [16, 40):  haze and surfaces in falling light. When its mean is saturated (> satMin), its hue must
                        lie in dimHue (degrees), e.g. not magenta where a dim orange falls through the display toe.
  light L in [40, 170): the light itself. When saturated (> satMin), its hue must lie in lightHue.
                        Judged when it holds >= 0.2 % of the pixels. Warm white is unsaturated and passes.
Letterboxed frames are cropped to the picture (the bars are black, not ink), from the timeline's aspect
curve. Shots listed in `exclude` (real recordings, the app's own pixels) are reported, not judged.
"""

import colorsys
import json
import pathlib
import re
import shutil
import subprocess
import sys
import os

import numpy as np

W, H = 480, 270
ENGINE = pathlib.Path(__file__).resolve().parents[1]
POS = re.compile(r'(?<![\d.])(\d+)\.(\d+)(?:\+(\d+(?:\.\d+)?)(f?))?')


def ffmpeg():
    exe = os.environ.get('FFMPEG') or shutil.which('ffmpeg')
    if not exe:
        sys.exit('ffmpeg not found: set FFMPEG or put it on the PATH')
    return exe


def take(args, flag, default=None):
    if flag in args:
        i = args.index(flag)
        v = args[i + 1]
        del args[i:i + 2]
        return v
    return default


ARGS = sys.argv[1:]
FILM = pathlib.Path(take(ARGS, '--film', str(ENGINE.parent.parent / 'films' / 'demo-light'))).resolve()
TIMELINE = json.loads((FILM / 'timeline.json').read_text())
TONE = json.loads((FILM / 'tone.json').read_text()) if (FILM / 'tone.json').exists() else {}


def triple(s):
    return [float(x) for x in s.split(',')]


def pair(s):
    a, b = s.split('-')
    return float(a), float(b)


INK = np.array(triple(take(ARGS, '--ink')) if '--ink' in ARGS else TONE.get('ink', [15, 14, 19]), np.float32)
INK_TOL = float(take(ARGS, '--ink-tol', TONE.get('inkTol', 5.0)))
SAT_MIN = float(take(ARGS, '--sat-min', TONE.get('satMin', 0.30)))
HUE_LO, HUE_HI = pair(take(ARGS, '--dim-hue')) if '--dim-hue' in ARGS else TONE.get('dimHue', [10, 30])
LIGHT_LO, LIGHT_HI = pair(take(ARGS, '--light-hue')) if '--light-hue' in ARGS else TONE.get('lightHue', [12, 26])
EXCLUDE = set((take(ARGS, '--exclude') or ','.join(TONE.get('exclude', []))).split(',')) - {''}


def luma(px):
    return 0.2126 * px[:, 0] + 0.7152 * px[:, 1] + 0.0722 * px[:, 2]


def judge(px):
    """px: (n, 3) float32. Returns (columns, failures)."""
    l = luma(px)
    cols, fails = {}, []
    cols['L p10/50/90'] = '/'.join(f'{x:.0f}' for x in np.percentile(l, [10, 50, 90]))
    deep = (l >= 4) & (l < 16)
    if deep.mean() >= 0.02:
        m = px[deep].mean(0)
        dev = float(np.abs(m - INK).max())
        cols['deep mean'] = f'({m[0]:.0f},{m[1]:.0f},{m[2]:.0f}) {deep.mean() * 100:.0f}%'
        cols['ink dev'] = f'{dev:.1f}'
        if dev > INK_TOL:
            fails.append(f'deep {dev:.1f} off ink')
    else:
        cols['deep mean'] = f'- {deep.mean() * 100:.0f}%'
        cols['ink dev'] = '-'
    dim = (l >= 16) & (l < 40)
    if dim.mean() >= 0.02:
        m = px[dim].mean(0)
        h, s, _ = colorsys.rgb_to_hsv(*(m / 255))
        h *= 360
        cols['dim mean'] = f'({m[0]:.0f},{m[1]:.0f},{m[2]:.0f}) {dim.mean() * 100:.0f}%'
        cols['dim hue/sat'] = f'{h:3.0f} {s:.2f}'
        if s > SAT_MIN and not (HUE_LO <= h <= HUE_HI):
            fails.append(f'dim hue {h:.0f} (sat {s:.2f})')
    else:
        cols['dim mean'] = f'- {dim.mean() * 100:.0f}%'
        cols['dim hue/sat'] = '-'
    lit = (l >= 40) & (l < 170)
    if lit.mean() >= 0.002:
        m = px[lit].mean(0)
        h, s, _ = colorsys.rgb_to_hsv(*(m / 255))
        h *= 360
        cols['light mean'] = f'({m[0]:.0f},{m[1]:.0f},{m[2]:.0f}) {lit.mean() * 100:.1f}%'
        cols['light hue/sat'] = f'{h:3.0f} {s:.2f}'
        if s > SAT_MIN and not (LIGHT_LO <= h <= LIGHT_HI):
            fails.append(f'light hue {h:.0f} (sat {s:.2f})')
    else:
        cols['light mean'] = f'- {lit.mean() * 100:.1f}%'
        cols['light hue/sat'] = '-'
    return cols, fails


def frame_of(pos):
    """Film frame of a position written as in the engine: 7.1, 4.3+0.5 (beats), 5.1+3f (frames)."""
    m = POS.fullmatch(pos)
    if not m:
        sys.exit(f'bad position {pos!r}')
    bar, beat, extra, unit = m.groups()
    f = (int(bar) - 1) * TIMELINE['framesPerBar'] + (int(beat) - 1) * TIMELINE['framesPerBeat']
    if extra:
        f += float(extra) if unit else float(extra) * TIMELINE['framesPerBeat']
    return round(f)


ASPECT = sorted(((frame_of(p), v) for p, v in TIMELINE['aspect']['keys']), key=lambda k: k[0])


def aspect_at(frame):
    """Aspect in effect at a frame (the later key wins on a snap; held between keys: good enough for a crop)."""
    a = ASPECT[0][1]
    for f, v in ASPECT:
        if frame >= f:
            a = v
    return a


def bars_rows(height, aspect):
    """Rows of the letterbox at this height and picture aspect."""
    vis = min(height, height * (16 / 9) / aspect)
    return max(0, round((height - vis) / 2) + (1 if vis < height else 0))


def shot_at(frame):
    for s in TIMELINE['shots']:
        if frame_of(s['from']) <= frame < frame_of(s['to']):
            return s['id']
    return '-'


def crop(img, frame):
    k = bars_rows(H, aspect_at(frame))
    return img[k:H - k] if k else img


def film(path, start):
    p = subprocess.run([ffmpeg(), '-v', 'error', '-i', str(path), '-vf', f"select='not(mod(n\\,4))',scale={W}:{H}",
                        '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True)
    if p.returncode:
        sys.exit(p.stderr.decode())
    fr = np.frombuffer(p.stdout, np.uint8).reshape(-1, H, W, 3)
    rows = []
    for s in TIMELINE['shots']:
        a, b = frame_of(s['from']), frame_of(s['to'])
        idx = [i for i in range(len(fr)) if a <= start + 4 * i < b]
        if not idx:
            continue
        sel = np.concatenate([crop(fr[i].astype(np.float32), start + 4 * i).reshape(-1, 3) for i in idx])
        rows.append((s['id'], sel, s['id'] in EXCLUDE))
    return rows


def images(paths):
    rows = []
    for path in paths:
        p = subprocess.run([ffmpeg(), '-v', 'error', '-i', str(path), '-vf', f'scale={W}:{H}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                           capture_output=True)
        if p.returncode:
            sys.exit(p.stderr.decode())
        img = np.frombuffer(p.stdout, np.uint8).reshape(H, W, 3).astype(np.float32)
        name = pathlib.Path(path).name
        m = POS.search(name)
        if not m:
            sys.exit(f'no position in {name!r}')
        frame = frame_of(m.group(0))
        tag = re.search(r'S\d\d', name)
        shot = tag.group(0) if tag else shot_at(frame)
        rows.append((name, crop(img, frame).reshape(-1, 3), shot in EXCLUDE))
    return rows


def main():
    args = ARGS
    start = 0
    v = take(args, '--from')
    if v:
        start = frame_of(v)
    if not args:
        sys.exit(__doc__)
    rows = film(args[0], start) if len(args) == 1 and args[0].endswith('.mp4') else images(args)
    if not rows:
        sys.exit('no frames')
    head = ['L p10/50/90', 'deep mean', 'ink dev', 'dim mean', 'dim hue/sat', 'light mean', 'light hue/sat']
    width = max(len(r[0]) for r in rows)
    print(f'{"":{width}}  ' + '  '.join(f'{h:>20}' for h in head) + '  verdict')
    bad = 0
    for name, px, excluded in rows:
        cols, fails = judge(px)
        verdict = 'excluded (not judged)' if excluded else ('FAIL: ' + '; '.join(fails) if fails else 'ok')
        bad += bool(fails) and not excluded
        print(f'{name:{width}}  ' + '  '.join(f'{cols[h]:>20}' for h in head) + '  ' + verdict)
    print(f'\nink {tuple(int(x) for x in INK)} ±{INK_TOL:.0f}; when sat > {SAT_MIN}: dim hue {HUE_LO:g}–{HUE_HI:g}°, light hue {LIGHT_LO:g}–{LIGHT_HI:g}°. {bad} failing.')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
