// Structural members built from beams (world/beams.ts): box trusses (corner chords, +-45 deg lacing on
// the faces, tied through at the nodes), lattice columns, and moving them as rigid groups.

import * as THREE from 'three';
import { box, lattice, type Beam, type Frame } from './beams';

export const UP = new THREE.Vector3(0, 1, 0);

export interface TrussSpec {
  pitch: number;
  member: number;
  chord: number;
  phase?: number;
  /** Lace all four long faces (default: only the two faces across `up`). */
  allFaces?: boolean;
}

/**
 * A box truss from the centre of its section at `start` along `d` (unit), `len` long, its section `w`
 * across (along d x up) by `h` along `up`.
 */
export function boxTruss(start: THREE.Vector3, d: THREE.Vector3, up: THREE.Vector3, len: number, w: number, h: number, o: TrussSpec): Beam[] {
  const side = new THREE.Vector3().crossVectors(d, up).normalize();
  const hw = w / 2;
  const hh = h / 2;
  const ch = o.chord / 2;
  const out: Beam[] = [];
  const fr: Frame = { origin: start.clone(), u: d.clone(), v: up.clone(), n: side };
  out.push(
    ...lattice(fr, {
      region: { u0: 0.05, u1: len - 0.05, v0: -hh + o.chord * 0.8, v1: hh - o.chord * 0.8 },
      pitch: o.pitch,
      width: o.member,
      depth: o.member * 1.2,
      layers: [hw - ch, -hw + ch],
      strut: o.member,
      phase: o.phase ?? 0,
    }),
  );
  if (o.allFaces) {
    const fr2: Frame = { origin: start.clone(), u: d.clone(), v: side.clone(), n: up.clone().negate() };
    out.push(...lattice(fr2, { region: { u0: 0.05, u1: len - 0.05, v0: -hw + o.chord * 0.8, v1: hw - o.chord * 0.8 }, pitch: o.pitch, width: o.member, depth: o.member * 1.2, layers: [hh - ch, -hh + ch], phase: (o.phase ?? 0) + 0.13 }));
  }
  for (const v of [hh - ch, -hh + ch]) for (const n of [hw - ch, -hw + ch]) out.push(box(fr, len / 2, v, n, len / 2, ch, ch));
  return out;
}

/** A square lattice column from `top` down `height`, its faces square to the horizontal `facing`. */
export function columnBeams(top: THREE.Vector3, facing: THREE.Vector3, height: number, size: number, o: Partial<TrussSpec> = {}): Beam[] {
  return boxTruss(top.clone(), UP.clone().negate(), facing.clone().setY(0).normalize(), height, size, size, { pitch: o.pitch ?? 0.34, member: o.member ?? 0.03, chord: o.chord ?? 0.055, phase: o.phase, allFaces: true });
}

/** Beams moved by a rigid transform. */
export function transformBeams(beams: Beam[], m: THREE.Matrix4): Beam[] {
  return beams.map((b) => ({ centre: b.centre.clone().applyMatrix4(m), ax: b.ax.clone().transformDirection(m), ay: b.ay.clone().transformDirection(m), half: b.half.clone() }));
}
