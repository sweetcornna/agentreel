// Motes of dust hanging in the haze: tiny, drifting slowly, visible only where a light catches them
// (inverse square, strongly forward-scattering, so motes between the lens and a light shine most),
// each drawn through the lens (world/bokeh.ts) as a spark in focus or a dim disc of the aperture's shape
// out of focus. A few are flakes that glint as they turn. Positions are analytic in time (base + drift
// + slow noise wander).

import * as THREE from 'three';
import { Rng } from '../lib/rng';
import { snoise3 } from '../lib/noise';
import type { BokehPoints } from './bokeh';
import type { HazeLight, HazeRing } from './haze';
import { sdRoundRect } from '../lib/leaf';
import type { Plane } from '../post/LensBlur';

interface Mote {
  p: THREE.Vector3;
  v: THREE.Vector3;
  albedo: number;
  glint: number;
  phase: number;
  seed: number;
}

export interface DustLight extends HazeLight {}

export interface DustLook {
  /** Seconds (the drift's clock; usually frame / fps). */
  time: number;
  camera: THREE.Vector3;
  lights: DustLight[];
  ring?: HazeRing | null;
  /** Motes on the camera side of every plane go to `front`, the rest to `back`. */
  planes: Plane[];
  gain: number;
  /** Haze extinction (1/m): a mote's light dims with its distance to the lens. */
  extinction: number;
  /** Brightest a mote may get (flux, as BokehPoints.set): a mote grazing a light stays a mote. */
  maxFlux?: number;
}

const hgPhase = (c: number, g: number): number => (1 - g * g) / (4 * Math.PI * Math.pow(Math.max(1 + g * g - 2 * g * c, 1e-4), 1.5));

export class Dust {
  private readonly motes: Mote[] = [];

  /** `count` motes in the box [min, max] of `frame` (a rigid transform to world; default identity). */
  constructor(key: string, count: number, min: THREE.Vector3, max: THREE.Vector3, frame = new THREE.Matrix4()) {
    const rng = new Rng(`${key}/dust`);
    const rot = new THREE.Matrix3().setFromMatrix4(frame);
    for (let i = 0; i < count; i++) {
      const p = new THREE.Vector3(rng.range(min.x, max.x), rng.range(min.y, max.y), rng.range(min.z, max.z)).applyMatrix4(frame);
      // Slow drift: a centimetre or two per second, mostly sideways and a little down.
      const v = new THREE.Vector3(rng.normal(0.004, 0.009), rng.normal(-0.003, 0.005), rng.normal(0, 0.006)).applyMatrix3(rot);
      // Most motes are faint; a long tail of brighter ones (log-normal albedo).
      const albedo = Math.exp(rng.normal(-0.4, 0.75));
      this.motes.push({ p, v, albedo, glint: rng.chance(0.12) ? rng.range(0.6, 2.5) : 0, phase: rng.range(0, 6.283), seed: rng.range(0, 100) });
    }
  }

  get count(): number {
    return this.motes.length;
  }

  /** Light every mote for this instant and hand it to the back or front bokeh set. Returns counts. */
  fill(look: DustLook, back: BokehPoints, front: BokehPoints | null): { back: number; front: number } {
    let nb = 0;
    let nf = 0;
    const p = new THREE.Vector3();
    const toEye = new THREE.Vector3();
    const fromL = new THREE.Vector3();
    const t = look.time;
    for (const m of this.motes) {
      p.copy(m.p).addScaledVector(m.v, t);
      const w = 0.05;
      p.x += w * snoise3(m.seed, t * 0.09, 0.3);
      p.y += w * snoise3(m.seed + 7.7, t * 0.09, 1.1);
      p.z += w * snoise3(m.seed + 3.1, t * 0.09, 2.7);
      toEye.copy(look.camera).sub(p);
      const de = toEye.length();
      if (de < 0.03) continue;
      toEye.divideScalar(de);
      let r = 0;
      let g = 0;
      let b = 0;
      for (const l of look.lights) {
        fromL.copy(p).sub(l.position);
        const d2 = fromL.lengthSq() + l.radius * l.radius;
        const c = fromL.normalize().dot(toEye);
        const k = (hgPhase(c, 0.72) * 0.8 + 0.2 / (4 * Math.PI)) / d2;
        r += l.color[0] * k;
        g += l.color[1] * k;
        b += l.color[2] * k;
      }
      const ring = look.ring;
      if (ring) {
        fromL.copy(p).sub(ring.centre);
        const qx = fromL.dot(ring.ax);
        const qy = fromL.dot(ring.ay);
        const qz = fromL.dot(new THREE.Vector3().crossVectors(ring.ax, ring.ay));
        const sd = sdRoundRect(qx, qy, ring.half[0], ring.half[1], { tl: ring.radii[0], tr: ring.radii[1], br: ring.radii[2], bl: ring.radii[3] });
        const band = Math.exp(-(sd * sd) / (ring.width * ring.width) - (qz * qz) / 0.004);
        r += ring.color[0] * band * 0.6;
        g += ring.color[1] * band * 0.6;
        b += ring.color[2] * band * 0.6;
      }
      let a = m.albedo * look.gain * Math.exp(-look.extinction * de);
      if (m.glint > 0) a *= 1 + m.glint * Math.pow(Math.max(0, Math.sin(t * 1.7 + m.phase)), 12);
      if (r * a < 1e-4 && g * a < 1e-4) continue;
      const peak = Math.max(r, g, b) * a;
      if (look.maxFlux !== undefined && peak > look.maxFlux) a *= look.maxFlux / peak;
      const inside = look.planes.every((pl) => fromL.copy(p).sub(pl.point).dot(pl.normal) >= 0);
      if (inside && front) front.set(nf++, p.x, p.y, p.z, r * a, g * a, b * a, 0.00025);
      else back.set(nb++, p.x, p.y, p.z, r * a, g * a, b * a, 0.00025);
    }
    return { back: nb, front: nf };
  }
}
