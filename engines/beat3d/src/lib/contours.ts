// Contour lines of a height field (marching squares), chained into polylines: terraces, hillsides and
// ridges drawn in light (world/brush.ts) are drawn as their contours.

import * as THREE from 'three';

export interface ContourGrid {
  /** Height at a ground point (metres). */
  heightAt: (x: number, z: number) => number;
  x0: number;
  x1: number;
  z0: number;
  z1: number;
  /** Cell size (m). */
  cell: number;
}

/**
 * Polylines at each level, lying on the surface (y = level). Each polyline is a list of points;
 * closed loops repeat their first point at the end. Segments shorter than `minLength` are dropped.
 */
export function contourLines(grid: ContourGrid, levels: number[], minLength = 0): { level: number; points: THREE.Vector3[] }[] {
  const nx = Math.max(2, Math.ceil((grid.x1 - grid.x0) / grid.cell) + 1);
  const nz = Math.max(2, Math.ceil((grid.z1 - grid.z0) / grid.cell) + 1);
  const dx = (grid.x1 - grid.x0) / (nx - 1);
  const dz = (grid.z1 - grid.z0) / (nz - 1);
  const h = new Float32Array(nx * nz);
  for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) h[j * nx + i] = grid.heightAt(grid.x0 + i * dx, grid.z0 + j * dz);

  const out: { level: number; points: THREE.Vector3[] }[] = [];
  for (const level of levels) {
    // Edge points keyed by edge id; segments link two edge ids.
    const edgePoint = new Map<number, THREE.Vector3>();
    const adj = new Map<number, number[]>();
    const edgeId = (i: number, j: number, horizontal: boolean): number => (j * nx + i) * 2 + (horizontal ? 0 : 1);
    const pointOn = (i0: number, j0: number, i1: number, j1: number, id: number): number => {
      if (!edgePoint.has(id)) {
        const a = h[j0 * nx + i0] - level;
        const b = h[j1 * nx + i1] - level;
        const t = a / (a - b);
        edgePoint.set(id, new THREE.Vector3(grid.x0 + (i0 + (i1 - i0) * t) * dx, level, grid.z0 + (j0 + (j1 - j0) * t) * dz));
      }
      return id;
    };
    const link = (a: number, b: number): void => {
      (adj.get(a) ?? adj.set(a, []).get(a)!).push(b);
      (adj.get(b) ?? adj.set(b, []).get(b)!).push(a);
    };
    for (let j = 0; j < nz - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const v0 = h[j * nx + i] - level;
        const v1 = h[j * nx + i + 1] - level;
        const v2 = h[(j + 1) * nx + i + 1] - level;
        const v3 = h[(j + 1) * nx + i] - level;
        const code = (v0 > 0 ? 1 : 0) | (v1 > 0 ? 2 : 0) | (v2 > 0 ? 4 : 0) | (v3 > 0 ? 8 : 0);
        if (code === 0 || code === 15) continue;
        // Edges: bottom (i,j)-(i+1,j), right (i+1,j)-(i+1,j+1), top (i,j+1)-(i+1,j+1), left (i,j)-(i,j+1).
        const B = () => pointOn(i, j, i + 1, j, edgeId(i, j, true));
        const R = () => pointOn(i + 1, j, i + 1, j + 1, edgeId(i + 1, j, false));
        const T = () => pointOn(i, j + 1, i + 1, j + 1, edgeId(i, j + 1, true));
        const L = () => pointOn(i, j, i, j + 1, edgeId(i, j, false));
        const centre = (v0 + v1 + v2 + v3) / 4 > 0;
        switch (code) {
          case 1: case 14: link(L(), B()); break;
          case 2: case 13: link(B(), R()); break;
          case 3: case 12: link(L(), R()); break;
          case 4: case 11: link(R(), T()); break;
          case 6: case 9: link(B(), T()); break;
          case 7: case 8: link(L(), T()); break;
          case 5: if (centre) { link(L(), T()); link(B(), R()); } else { link(L(), B()); link(R(), T()); } break;
          case 10: if (centre) { link(L(), B()); link(R(), T()); } else { link(L(), T()); link(B(), R()); } break;
        }
      }
    }
    // Chain: start from ends (degree 1) first, then loops.
    const used = new Set<string>();
    const key = (a: number, b: number) => (a < b ? `${a}_${b}` : `${b}_${a}`);
    const walk = (start: number): THREE.Vector3[] => {
      const pts = [edgePoint.get(start)!];
      let prev = -1;
      let cur = start;
      for (;;) {
        const nb = (adj.get(cur) ?? []).find((n) => n !== prev && !used.has(key(cur, n)));
        if (nb === undefined) break;
        used.add(key(cur, nb));
        pts.push(edgePoint.get(nb)!);
        prev = cur;
        cur = nb;
        if (cur === start) break;
      }
      return pts;
    };
    const ids = [...adj.keys()].sort((a, b) => a - b);
    for (const id of ids) if ((adj.get(id) ?? []).length === 1 && (adj.get(id) ?? []).some((n) => !used.has(key(id, n)))) push(walk(id));
    for (const id of ids) if ((adj.get(id) ?? []).some((n) => !used.has(key(id, n)))) push(walk(id));

    function push(pts: THREE.Vector3[]): void {
      if (pts.length < 2) return;
      let len = 0;
      for (let k = 1; k < pts.length; k++) len += pts[k].distanceTo(pts[k - 1]);
      if (len >= minLength) out.push({ level, points: pts });
    }
  }
  return out;
}

/**
 * The skyline of a height field seen from `eye`: for each azimuth, the terrain point with the highest
 * elevation angle along the ground ray, out to `maxDist`. Azimuths follow the camera yaw convention
 * (yaw a looks along (-sin a, 0, -cos a)); pass yawLeft > yawRight to get the points left to right.
 * `keep(x, z)` limits the march to the modelled area. Draw the result as one stroke: the outline of the
 * land against the sky, with hidden lines removed by construction.
 */
export function skyline(
  heightAt: (x: number, z: number) => number,
  eye: THREE.Vector3,
  yawLeft: number,
  yawRight: number,
  samples: number,
  maxDist: number,
  keep: (x: number, z: number) => boolean = () => true,
  smoothing = 2,
): THREE.Vector3[] {
  const out: THREE.Vector3[] = [];
  for (let k = 0; k < samples; k++) {
    const a = yawLeft + ((yawRight - yawLeft) * k) / Math.max(samples - 1, 1);
    const sx = -Math.sin(a);
    const sz = -Math.cos(a);
    let best = -Infinity;
    let bp: THREE.Vector3 | null = null;
    for (let d = 4; d <= maxDist; d += Math.max(2, d * 0.004)) {
      const x = eye.x + sx * d;
      const z = eye.z + sz * d;
      if (!keep(x, z)) continue;
      const h = heightAt(x, z);
      const e = (h - eye.y) / d;
      if (e > best) {
        best = e;
        bp = new THREE.Vector3(x, h, z);
      }
    }
    if (bp) out.push(bp);
  }
  // The march quantises the summit (angular and distance steps): a short moving average removes the
  // stair steps without rounding the shape.
  return out.map((_, k) => {
    const a = Math.max(0, k - smoothing);
    const b = Math.min(out.length - 1, k + smoothing);
    const p = new THREE.Vector3();
    for (let i = a; i <= b; i++) p.add(out[i]);
    return p.divideScalar(b - a + 1);
  });
}

/** Resample a polyline to roughly even spacing (m), keeping its ends. */
export function resample(points: THREE.Vector3[], spacing: number): THREE.Vector3[] {
  if (points.length < 2) return points.slice();
  const out = [points[0].clone()];
  let carry = 0;
  for (let k = 1; k < points.length; k++) {
    const a = points[k - 1];
    const b = points[k];
    const seg = a.distanceTo(b);
    let t = spacing - carry;
    while (t <= seg) {
      out.push(a.clone().lerp(b, t / seg));
      t += spacing;
    }
    carry = seg - (t - spacing);
  }
  if (out[out.length - 1].distanceTo(points[points.length - 1]) > 1e-6) out.push(points[points.length - 1].clone());
  return out;
}
