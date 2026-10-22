// Beams as real structure: boxes with softly rounded edges (a rounded-box normal, so light catches as
// thin lines along their bevels), dark graphite, lit only by the take's few lights (GGX + Lambert, and
// the grazing rim a light behind a beam puts on its edges), dimmed by the haze with distance. Walls,
// corridors, gates and trusses (world/truss.ts) are built from these. All beams are one mesh.
//
// Builders work in a frame (origin, u, v, n): `lattice()` lays two families of +-45 deg lines across a
// rectangle of the (u, v) plane, `box()` adds a single member.

import * as THREE from 'three';
import { MAX_LIGHTS, type HazeLight } from './haze';
import type { RGB } from '../lib/color';

export interface Beam {
  centre: THREE.Vector3;
  /** Beam frame (unit): x along the beam, y across it; z = x cross y. */
  ax: THREE.Vector3;
  ay: THREE.Vector3;
  /** Half extents along ax, ay, z. */
  half: THREE.Vector3;
}

export interface Frame {
  origin: THREE.Vector3;
  u: THREE.Vector3;
  v: THREE.Vector3;
  /** Normal of the (u, v) plane. */
  n: THREE.Vector3;
}

export interface Rect {
  u0: number;
  u1: number;
  v0: number;
  v1: number;
}

export interface LatticeSpec {
  region: Rect;
  /** Perpendicular spacing of the diagonal lines (m). */
  pitch: number;
  /** Beam section: width in the plane, depth along n. */
  width: number;
  depth: number;
  /** Offsets along n of each layer's mid-plane, and the strut section tying them at the nodes. */
  layers?: number[];
  strut?: number;
  /** Rectangles the lattice keeps clear. */
  holes?: Rect[];
  /** Phase of the lines (m along their normal), so neighbouring lattices need not line up. */
  phase?: number;
}

/** Parameter interval of the line c + d t inside the rectangle, or null. */
function clipRect(cx: number, cy: number, dx: number, dy: number, r: Rect): [number, number] | null {
  let t0 = -1e9;
  let t1 = 1e9;
  for (const [c, d, lo, hi] of [
    [cx, dx, r.u0, r.u1],
    [cy, dy, r.v0, r.v1],
  ]) {
    if (Math.abs(d) < 1e-9) {
      if (c < lo || c > hi) return null;
      continue;
    }
    const a = (lo - c) / d;
    const b = (hi - c) / d;
    t0 = Math.max(t0, Math.min(a, b));
    t1 = Math.min(t1, Math.max(a, b));
  }
  return t1 > t0 ? [t0, t1] : null;
}

const at = (fr: Frame, u: number, v: number, w: number): THREE.Vector3 =>
  fr.origin.clone().addScaledVector(fr.u, u).addScaledVector(fr.v, v).addScaledVector(fr.n, w);

/** A member from (u0, v0) to (u1, v1) at offset w along n, of the given width (in plane) and depth (along n). */
export function member(fr: Frame, u0: number, v0: number, u1: number, v1: number, w: number, width: number, depth: number): Beam {
  const a = at(fr, u0, v0, w);
  const b = at(fr, u1, v1, w);
  const ax = b.clone().sub(a);
  const len = ax.length();
  ax.divideScalar(len);
  const ay = new THREE.Vector3().crossVectors(fr.n, ax).normalize();
  return { centre: a.add(b).multiplyScalar(0.5), ax, ay, half: new THREE.Vector3(len / 2, width / 2, depth / 2) };
}

/** An axis-aligned (in the frame) box: centre (u, v, w) and half sizes. */
export function box(fr: Frame, u: number, v: number, w: number, hu: number, hv: number, hw: number): Beam {
  return { centre: at(fr, u, v, w), ax: fr.u.clone(), ay: fr.v.clone(), half: new THREE.Vector3(hu, hv, hw) };
}

/** The two families of +-45 deg beams across a rectangle of the frame's plane, layers tied by struts. */
export function lattice(fr: Frame, spec: LatticeSpec): Beam[] {
  const out: Beam[] = [];
  const r = spec.region;
  const holes = spec.holes ?? [];
  const layers = spec.layers ?? [0];
  const pieces = (cx: number, cy: number, dx: number, dy: number, t: [number, number]): [number, number][] => {
    let segs: [number, number][] = [t];
    for (const h of holes) {
      const cut = clipRect(cx, cy, dx, dy, h);
      if (!cut) continue;
      segs = segs.flatMap(([a, b]) => {
        const keep: [number, number][] = [];
        if (cut[0] > a) keep.push([a, Math.min(b, cut[0])]);
        if (cut[1] < b) keep.push([Math.max(a, cut[1]), b]);
        return keep.filter(([p, q]) => q - p > 0.05);
      });
    }
    return segs;
  };
  const s = Math.SQRT1_2;
  const cu = (r.u0 + r.u1) / 2;
  const cv = (r.v0 + r.v1) / 2;
  const span = Math.hypot(r.u1 - r.u0, r.v1 - r.v0);
  const lines = Math.ceil(span / spec.pitch) + 1;
  const phase = spec.phase ?? 0;
  [new THREE.Vector2(s, s), new THREE.Vector2(s, -s)].forEach((d, fi) => {
    const nrm = new THREE.Vector2(-d.y, d.x);
    for (let k = -lines; k <= lines; k++) {
      const c = nrm.clone().multiplyScalar(k * spec.pitch + phase).add(new THREE.Vector2(cu, cv));
      const t = clipRect(c.x, c.y, d.x, d.y, r);
      if (!t) continue;
      for (const [a, b] of pieces(c.x, c.y, d.x, d.y, t)) {
        // The second family sits 3 mm back, so the two never share a face where they cross.
        for (const w of layers) out.push(member(fr, c.x + d.x * a, c.y + d.y * a, c.x + d.x * b, c.y + d.y * b, w - 0.003 * fi, spec.width, spec.depth));
      }
    }
  });
  if (layers.length > 1 && spec.strut) {
    // Struts in depth where the families cross, outside the holes.
    const w0 = Math.min(...layers);
    const w1 = Math.max(...layers);
    for (let i = -lines; i <= lines; i++) {
      for (let j = -lines; j <= lines; j++) {
        const u = cu + (j - i) * s * spec.pitch;
        const v = cv + (j + i) * s * spec.pitch + phase * Math.SQRT2;
        if (u < r.u0 || u > r.u1 || v < r.v0 || v > r.v1) continue;
        if (holes.some((h) => u > h.u0 - 0.03 && u < h.u1 + 0.03 && v > h.v0 - 0.03 && v < h.v1 + 0.03)) continue;
        out.push({ centre: at(fr, u, v, (w0 + w1) / 2), ax: fr.n.clone(), ay: fr.u.clone(), half: new THREE.Vector3((w1 - w0) / 2 + spec.depth / 2, spec.strut / 2, spec.strut / 2) });
      }
    }
  }
  return out;
}

const VERT = /* glsl */ `
attribute vec3 aBox;      // position in the beam's own frame (m, from its centre)
attribute vec3 aHalf;     // half extents
attribute vec3 aAx;       // beam frame: x and y axes (world, unit); z = cross
attribute vec3 aAy;
varying vec3 vBox;
varying vec3 vHalf;
varying vec3 vAx;
varying vec3 vAy;
varying vec3 vW;
void main() {
  vBox = aBox;
  vHalf = aHalf;
  vAx = normalize(mat3(modelMatrix) * aAx);
  vAy = normalize(mat3(modelMatrix) * aAy);
  vec4 w = modelMatrix * vec4(position, 1.0);
  vW = w.xyz;
  gl_Position = projectionMatrix * viewMatrix * w;
}
`;

const BEVEL = 0.012;

const FRAG = /* glsl */ `
#define MAX_LIGHTS ${MAX_LIGHTS}
uniform vec3 uCam;
uniform int uNL;
uniform vec3 uLP[MAX_LIGHTS];
uniform vec3 uLC[MAX_LIGHTS];
uniform float uLR[MAX_LIGHTS];
uniform vec3 uAlbedo;
uniform vec3 uAmbient;
uniform float uRough;
uniform float uRim;
uniform vec3 uBackDir;
uniform vec3 uBackCol;
uniform vec3 uBackAt;
uniform float uBackFall;
uniform float uExt;
uniform float uOpacity;
varying vec3 vBox;
varying vec3 vHalf;
varying vec3 vAx;
varying vec3 vAy;
varying vec3 vW;
float ggx(float nh, float a) {
  float a2 = a * a;
  float d = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / (3.14159265 * d * d);
}
void main() {
  // Rounded-box normal: flat on the faces, rolling over a ${BEVEL * 1000} mm radius at every edge.
  vec3 q = abs(vBox) - (vHalf - ${BEVEL.toFixed(4)});
  vec3 nl = sign(vBox) * max(q, 0.0);
  // On the rounded edges (1) or a flat face (0): only the edges catch a rim.
  float onEdge = smoothstep(0.55, 0.95, length(max(q, 0.0)) / ${BEVEL.toFixed(4)});
  if (dot(nl, nl) < 1e-12) {
    vec3 r = abs(vBox) / vHalf;
    nl = r.x > r.y && r.x > r.z ? vec3(sign(vBox.x), 0.0, 0.0) : (r.y > r.z ? vec3(0.0, sign(vBox.y), 0.0) : vec3(0.0, 0.0, sign(vBox.z)));
  }
  nl = normalize(nl);
  vec3 az = cross(vAx, vAy);
  vec3 n = normalize(vAx * nl.x + vAy * nl.y + az * nl.z);
  vec3 toCam = uCam - vW;
  float dist = length(toCam);
  vec3 v = toCam / dist;
  float nv = max(dot(n, v), 1e-3);
  vec3 col = uAmbient * uAlbedo * (0.6 + 0.4 * n.y);
  float a = uRough * uRough;
  float graze = pow(1.0 - nv, 6.0);
  for (int i = 0; i < MAX_LIGHTS; i++) {
    if (i >= uNL) break;
    vec3 L = uLP[i] - vW;
    float d2 = dot(L, L) + uLR[i] * uLR[i];
    vec3 l = L * inversesqrt(dot(L, L) + 1e-8);
    float nl2 = max(dot(n, l), 0.0);
    vec3 h = normalize(l + v);
    float nh = max(dot(n, h), 0.0);
    float F = 0.045 + 0.955 * pow(1.0 - max(dot(h, v), 0.0), 5.0);
    float spec = ggx(nh, a) * F / (4.0 * nv * max(nl2, 1e-3) + 1e-3) * nl2;
    // Rim: a light behind the beam grazes its edges toward the lens.
    float rim = uRim * onEdge * graze * smoothstep(0.2, 0.9, dot(l, -v)) * smoothstep(0.05, 0.5, dot(n, l));
    col += uLC[i] / d2 * (uAlbedo / 3.14159265 * nl2 + spec + rim);
  }
  // A far backlight (direction toward it, from a plane at uBackAt): rims every edge it grazes toward
  // the lens, a little fill; it falls off over uBackFall metres back from that plane.
  {
    vec3 l = uBackDir;
    float nl2 = max(dot(n, l), 0.0);
    float rim = onEdge * graze * smoothstep(0.2, 0.9, dot(l, -v)) * smoothstep(0.05, 0.5, dot(n, l));
    float fall = exp(-max(dot(uBackAt - vW, l), 0.0) / uBackFall);
    col += uBackCol * fall * (uAlbedo / 3.14159265 * nl2 + rim);
  }
  gl_FragColor = vec4(col * exp(-uExt * dist), 1.0) * uOpacity;
}
`;

function geometry(beams: Beam[]): THREE.BufferGeometry {
  const pos: number[] = [];
  const bx: number[] = [];
  const half: number[] = [];
  const axx: number[] = [];
  const axy: number[] = [];
  const idx: number[] = [];
  // Unit cube faces: normal axis, and the two in-face axes.
  const faces: [number, number, number][] = [
    [0, 1, 2],
    [1, 2, 0],
    [2, 0, 1],
  ];
  for (const b of beams) {
    const az = new THREE.Vector3().crossVectors(b.ax, b.ay);
    const axes = [b.ax, b.ay, az];
    const h = [b.half.x, b.half.y, b.half.z];
    for (const [na, ua, va] of faces) {
      for (const sgn of [1, -1]) {
        const base = pos.length / 3;
        for (const [su, sv] of [
          [-1, -1],
          [1, -1],
          [1, 1],
          [-1, 1],
        ]) {
          const l = [0, 0, 0];
          l[na] = sgn * h[na];
          l[ua] = su * h[ua];
          l[va] = sv * h[va];
          const p = b.centre.clone().addScaledVector(axes[0], l[0]).addScaledVector(axes[1], l[1]).addScaledVector(axes[2], l[2]);
          pos.push(p.x, p.y, p.z);
          bx.push(l[0], l[1], l[2]);
          half.push(h[0], h[1], h[2]);
          axx.push(b.ax.x, b.ax.y, b.ax.z);
          axy.push(b.ay.x, b.ay.y, b.ay.z);
        }
        // Winding so the face points along +sgn on its axis (u x v = n for the cyclic triple).
        if (sgn > 0) idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
        else idx.push(base, base + 2, base + 1, base, base + 3, base + 2);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('aBox', new THREE.Float32BufferAttribute(bx, 3));
  g.setAttribute('aHalf', new THREE.Float32BufferAttribute(half, 3));
  g.setAttribute('aAx', new THREE.Float32BufferAttribute(axx, 3));
  g.setAttribute('aAy', new THREE.Float32BufferAttribute(axy, 3));
  g.setIndex(idx);
  return g;
}

export interface BeamLook {
  camera: THREE.Vector3;
  lights: HazeLight[];
  albedo: RGB;
  ambient: RGB;
  /** Haze extinction (1/m). */
  extinction: number;
  /** 0..1, premultiplied (0 = not there). */
  opacity: number;
  /** Strength of the grazing rim from lights behind (default 1). */
  rim?: number;
  /** A far backlight: unit direction toward it, its colour (irradiance) at the plane `at` (normal
   *  `dir`), falling off exponentially over `fall` metres back from it. */
  back?: { dir: THREE.Vector3; color: RGB; at: THREE.Vector3; fall: number };
}

export class Beams {
  readonly mesh: THREE.Mesh;
  private readonly mat: THREE.ShaderMaterial;

  constructor(beams: Beam[]) {
    this.mat = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: {
        uCam: { value: new THREE.Vector3() },
        uNL: { value: 0 },
        uLP: { value: Array.from({ length: MAX_LIGHTS }, () => new THREE.Vector3()) },
        uLC: { value: Array.from({ length: MAX_LIGHTS }, () => new THREE.Vector3()) },
        uLR: { value: new Array(MAX_LIGHTS).fill(0.01) },
        uAlbedo: { value: new THREE.Vector3() },
        uAmbient: { value: new THREE.Vector3() },
        uRough: { value: 0.34 },
        uRim: { value: 1 },
        uBackDir: { value: new THREE.Vector3(0, 0, 1) },
        uBackCol: { value: new THREE.Vector3() },
        uBackAt: { value: new THREE.Vector3() },
        uBackFall: { value: 10 },
        uExt: { value: 0 },
        uOpacity: { value: 1 },
      },
      side: THREE.FrontSide,
      toneMapped: false,
    });
    this.mesh = new THREE.Mesh(geometry(beams), this.mat);
    this.mesh.frustumCulled = false;
  }

  update(look: BeamLook): void {
    const u = this.mat.uniforms;
    u.uCam.value.copy(look.camera);
    const n = Math.min(look.lights.length, MAX_LIGHTS);
    u.uNL.value = n;
    for (let i = 0; i < n; i++) {
      u.uLP.value[i].copy(look.lights[i].position);
      u.uLC.value[i].set(...look.lights[i].color);
      u.uLR.value[i] = look.lights[i].radius;
    }
    u.uAlbedo.value.set(...look.albedo);
    u.uAmbient.value.set(...look.ambient);
    u.uExt.value = look.extinction;
    u.uOpacity.value = look.opacity;
    u.uRim.value = look.rim ?? 1;
    if (look.back) {
      u.uBackDir.value.copy(look.back.dir);
      u.uBackCol.value.set(...look.back.color);
      u.uBackAt.value.copy(look.back.at);
      u.uBackFall.value = look.back.fall;
    } else u.uBackCol.value.set(0, 0, 0);
    this.mesh.visible = look.opacity > 0;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.mat.dispose();
  }
}
