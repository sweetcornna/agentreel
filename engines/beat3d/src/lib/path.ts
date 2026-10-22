// Keyed motion (camera rigs, anything that travels): positions as cubic Hermite splines in time, with
// velocities that are either given at a key (a stop, a speed to match) or the mean of the slopes on
// either side. Keys sit on musical positions; outside the keys the path carries on at its end velocity
// (so motion-blur sub-samples a hair before the first key still move the right way).
// Watch the tangents: a key just after a hand-over whose next segment has a large velocity makes the
// curve back up first (velocity x interval > 3 x the change of position).

import * as THREE from 'three';
import { CLOCK, type Pos } from './time';

export interface Key3 {
  pos: Pos;
  p: THREE.Vector3 | [number, number, number];
  /** Velocity at the key, units per frame. */
  v?: THREE.Vector3 | [number, number, number];
}

const vec = (v: THREE.Vector3 | [number, number, number]): THREE.Vector3 => (Array.isArray(v) ? new THREE.Vector3(...v) : v.clone());

export class Path3 {
  readonly f: number[];
  private readonly p: THREE.Vector3[];
  private readonly v: THREE.Vector3[];

  constructor(keys: Key3[]) {
    if (keys.length < 2) throw new Error('Path3 needs two keys');
    const k = keys.map((x) => ({ f: CLOCK.frameAt(x.pos), p: vec(x.p), v: x.v ? vec(x.v) : null })).sort((a, b) => a.f - b.f);
    this.f = k.map((x) => x.f);
    this.p = k.map((x) => x.p);
    const n = k.length;
    this.v = k.map((x, i) => {
      if (x.v) return x.v;
      const slope = (a: number, b: number) => k[b].p.clone().sub(k[a].p).divideScalar(k[b].f - k[a].f);
      if (i === 0) return slope(0, 1);
      if (i === n - 1) return slope(n - 2, n - 1);
      return slope(i - 1, i).add(slope(i, i + 1)).multiplyScalar(0.5);
    });
  }

  at(frame: number, out = new THREE.Vector3()): THREE.Vector3 {
    const fs = this.f;
    const n = fs.length;
    if (frame <= fs[0]) return out.copy(this.p[0]).addScaledVector(this.v[0], frame - fs[0]);
    if (frame >= fs[n - 1]) return out.copy(this.p[n - 1]).addScaledVector(this.v[n - 1], frame - fs[n - 1]);
    let i = 0;
    while (i < n - 2 && frame >= fs[i + 1]) i++;
    const h = fs[i + 1] - fs[i];
    const t = (frame - fs[i]) / h;
    const t2 = t * t;
    const t3 = t2 * t;
    return out
      .copy(this.p[i])
      .multiplyScalar(2 * t3 - 3 * t2 + 1)
      .addScaledVector(this.v[i], (t3 - 2 * t2 + t) * h)
      .addScaledVector(this.p[i + 1], -2 * t3 + 3 * t2)
      .addScaledVector(this.v[i + 1], (t3 - t2) * h);
  }

  /** Velocity (units per frame), central difference. */
  velocity(frame: number, out = new THREE.Vector3()): THREE.Vector3 {
    const a = this.at(frame - 0.05);
    return this.at(frame + 0.05, out).sub(a).divideScalar(0.1);
  }
}

/** A scalar keyed the same way (value and optional slope per frame). */
export class Path1 {
  private readonly path: Path3;
  constructor(keys: { pos: Pos; value: number; slope?: number }[]) {
    this.path = new Path3(keys.map((k) => ({ pos: k.pos, p: [k.value, 0, 0] as [number, number, number], v: k.slope === undefined ? undefined : ([k.slope, 0, 0] as [number, number, number]) })));
  }
  at(frame: number): number {
    return this.path.at(frame).x;
  }
}
