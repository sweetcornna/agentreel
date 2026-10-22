// Keyframe curves addressed in musical positions, evaluated per (fractional) frame.
// The film-wide intention curves (aspect, and when the timeline has them, energy and persistence)
// come straight from the film's timeline.json; shots can build their own with the same class.

import { CLOCK, TIMELINE, type CurveKeys, type Pos } from './time';
import { EASINGS, type EasingName, clamp } from './easing';

export type CurveInterp = EasingName | 'monotone';

export interface CurveKey {
  pos: Pos;
  value: number;
  /** Easing of the segment that *starts* at this key. Defaults to the curve's interp. */
  ease?: EasingName;
}

export class Curve {
  private readonly frames: number[];
  private readonly values: number[];
  private readonly eases: (EasingName | undefined)[];
  private readonly tangents: number[] | null;

  constructor(keys: CurveKey[], readonly interp: CurveInterp = 'linear') {
    if (keys.length === 0) throw new Error('Curve needs at least one key');
    const sorted = keys.map((k) => ({ ...k, f: CLOCK.frameAt(k.pos) })).sort((a, b) => a.f - b.f);
    this.frames = sorted.map((k) => k.f);
    this.values = sorted.map((k) => k.value);
    this.eases = sorted.map((k) => k.ease);
    this.tangents = interp === 'monotone' ? monotoneTangents(this.frames, this.values) : null;
  }

  static fromSpec(keys: CurveKeys, interp: CurveInterp = 'linear', eases: Record<string, EasingName> = {}): Curve {
    return new Curve(
      keys.map(([pos, value]) => ({ pos, value, ease: eases[pos] })),
      interp,
    );
  }

  at(frame: number): number {
    const fs = this.frames;
    const vs = this.values;
    const n = fs.length;
    if (frame <= fs[0]) return vs[0];
    if (frame >= fs[n - 1]) return vs[n - 1];
    let i = 0;
    while (i < n - 2 && frame >= fs[i + 1]) i++;
    const f0 = fs[i];
    const f1 = fs[i + 1];
    const t = clamp((frame - f0) / (f1 - f0));
    if (this.tangents && !this.eases[i]) {
      // Fritsch-Carlson monotone cubic Hermite: smooth without overshoot.
      const h = f1 - f0;
      const t2 = t * t;
      const t3 = t2 * t;
      return (
        (2 * t3 - 3 * t2 + 1) * vs[i] +
        (t3 - 2 * t2 + t) * h * this.tangents[i] +
        (-2 * t3 + 3 * t2) * vs[i + 1] +
        (t3 - t2) * h * this.tangents[i + 1]
      );
    }
    const easeName = this.eases[i] ?? (this.interp === 'monotone' ? 'linear' : this.interp);
    return vs[i] + (vs[i + 1] - vs[i]) * EASINGS[easeName](t);
  }
}

function monotoneTangents(x: number[], y: number[]): number[] {
  const n = x.length;
  const d: number[] = [];
  for (let i = 0; i < n - 1; i++) d.push((y[i + 1] - y[i]) / (x[i + 1] - x[i]));
  const m: number[] = new Array(n).fill(0);
  m[0] = d[0] ?? 0;
  m[n - 1] = d[n - 2] ?? 0;
  for (let i = 1; i < n - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (d[i] === 0) {
      m[i] = 0;
      m[i + 1] = 0;
      continue;
    }
    const a = m[i] / d[i];
    const b = m[i + 1] / d[i];
    const s = a * a + b * b;
    if (s > 9) {
      const tau = 3 / Math.sqrt(s);
      m[i] = tau * a * d[i];
      m[i + 1] = tau * b * d[i];
    }
  }
  return m;
}

/**
 * Aspect of the visible picture. The timeline's `aspect.interp` names the easing between keys
 * ("constant" holds each value until the next key; default easeInOutCubic); `aspect.eases` can set
 * it per key. Two keys less than a frame apart are a snap, taken on the first key's frame: keys
 * [5.1, 2.39], [5.1, 16:9] open the frame on 5.1 itself, the first frame of the cut, not one frame
 * into it.
 */
class AspectCurve {
  private readonly curve: Curve;
  private readonly snaps: { at: number; value: number }[] = [];

  constructor(keys: CurveKeys, interp = 'easeInOutCubic', eases: Record<string, string> = {}) {
    const word = (interp.match(/^[A-Za-z]+/)?.[0] ?? 'easeInOutCubic') as string;
    const name = (word === 'constant' ? 'step' : word in EASINGS || word === 'monotone' ? word : 'easeInOutCubic') as CurveInterp;
    const perKey = Object.fromEntries(Object.entries(eases).filter(([, e]) => e in EASINGS)) as Record<string, EasingName>;
    this.curve = Curve.fromSpec(keys, name, perKey);
    const fs = keys.map(([p, v]) => ({ f: CLOCK.frameAt(p), v })).sort((a, b) => a.f - b.f);
    for (let i = 0; i + 1 < fs.length; i++) if (fs[i + 1].f - fs[i].f <= 1 + 1e-9 && fs[i + 1].v !== fs[i].v) this.snaps.push({ at: fs[i].f, value: fs[i + 1].v });
  }

  at(frame: number): number {
    for (const s of this.snaps) if (frame >= s.at && frame < s.at + 1) return s.value;
    return this.curve.at(frame);
  }
}

export const ASPECT = new AspectCurve(TIMELINE.aspect.keys, TIMELINE.aspect.interp, TIMELINE.aspect.eases);
/** Shared intention curve for picture density / music loudness, 0..1 (0 throughout if the timeline has none). */
export const ENERGY = Curve.fromSpec(TIMELINE.energy?.keys ?? [['1.1', 0]], 'monotone');
/** 0 = trails forgotten at once, 1 = permanent (lib/persistence.ts); 0 throughout if the timeline has none. */
export const PERSISTENCE = Curve.fromSpec(TIMELINE.persistence?.keys ?? [['1.1', 0]], 'linear');
