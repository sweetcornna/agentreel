// 霜草: real blades of dry winter grass for close shots (85 mm at 20 cm, 50 mm at 40 cm, a 100 mm macro
// at 5 cm). Beyond a few metres the terrain's parallax tussocks take over.
//
// Each blade is an instanced curved ribbon (quadratic Bezier centre line, tapering to a point),
// turned towards the eye about its own axis so it never collapses to an edge. Shading: straw albedo
// darkening to the base, hoar frost on the edges and tip (whitish, a little blue), sky light from
// above, and up to 4 character lights plus up to 12 ground lights (footprints) with inverse-square
// falloff, two-sided wrap and translucency (dry grass glows when lit from behind). Blades write
// depth, so the post DOF blurs the near ones into soft shapes.
// Placement is seeded, so the field is identical in every frame and every render.

import * as THREE from 'three';
import { Rng } from '../lib/rng';
import type { PointLightSpec } from './terrain';

export interface GrassWedge {
  /** Eye position (xz) and view yaw (radians, 0 = looking down -Z, positive turns right). */
  x: number;
  z: number;
  yaw: number;
  /** Half-angle of the wedge (radians), and the distance range (metres). */
  halfAngle: number;
  near: number;
  far: number;
}

export interface GrassSpec {
  key: string;
  count: number;
  wedge: GrassWedge;
  heightAt: (x: number, z: number) => number;
  /** Blade height and base width ranges (metres). */
  height?: [number, number];
  width?: [number, number];
  /** Fraction of tall dry stems and their height range. */
  tallFraction?: number;
  tallHeight?: [number, number];
  /** Base width range of the tall stems (default: the blade width range x 0.7). */
  tallWidth?: [number, number];
  /** Max lean from vertical (radians). */
  lean?: number;
  /** Segments along a blade. */
  segments?: number;
  /** Clumping: blades gather around this many clump centres per square metre (0 = uniform). */
  clumpsPerM2?: number;
  /** Optional height multiplier by position (e.g. keep blades near a low lens below the eye line). */
  heightScale?: (x: number, z: number, tall: boolean) => number;
}

export interface GrassLook {
  skyAmbient: [number, number, number];
  straw: [number, number, number];
  strawDark: [number, number, number];
  frost: [number, number, number];
  /** Frost coverage 0..1 (edges and tips). */
  frostAmount: number;
  translucency: number;
  /** Brightness of the tall stems relative to the rest (default 1). */
  tallGain?: number;
}

export const FROST_GRASS_LOOK: GrassLook = {
  skyAmbient: [0.0105, 0.0138, 0.0225],
  straw: [0.25, 0.19, 0.12],
  strawDark: [0.08, 0.06, 0.045],
  frost: [0.62, 0.68, 0.76],
  frostAmount: 0.75,
  translucency: 0.55,
};

const MAX_GROUND = 12;

const VERT = /* glsl */ `
attribute vec2 aUV;          // x: -1..1 across, y: 0..1 along
attribute vec4 iBase;        // xyz base, w height
attribute vec4 iShape;       // width, lean azimuth, lean angle, curl
attribute vec4 iLook;        // tone, frost, seed, tall
varying vec3 vPos;
varying vec3 vN;
varying vec2 vUV;
varying vec4 vLook;
void main() {
  float v = aUV.y;
  float h = iBase.w;
  vec3 up = vec3(0.0, 1.0, 0.0);
  vec3 ld = vec3(cos(iShape.y), 0.0, sin(iShape.y));
  float a = iShape.z;
  // Quadratic Bezier: rises, then droops towards the lean direction (curl adds a late bend).
  vec3 p0 = iBase.xyz;
  vec3 p1 = p0 + up * h * 0.62 + ld * h * sin(a) * 0.15;
  vec3 p2 = p0 + up * h * cos(a + iShape.w) + ld * h * sin(a + iShape.w);
  vec3 c = (1.0 - v) * (1.0 - v) * p0 + 2.0 * (1.0 - v) * v * p1 + v * v * p2;
  vec3 tr = 2.0 * (1.0 - v) * (p1 - p0) + 2.0 * v * (p2 - p1);
  vec3 t = dot(tr, tr) > 1e-12 ? normalize(tr) : vec3(0.0, 1.0, 0.0);
  vec3 toEye = normalize(cameraPosition - c);
  vec3 w = cross(t, toEye);
  float wl = length(w);
  w = wl > 1e-4 ? w / wl : vec3(1.0, 0.0, 0.0);
  float width = iShape.x * pow(max(1.0 - v, 0.0), 0.75) * (0.85 + 0.15 * sin(v * 9.0 + iLook.z * 30.0));
  vec3 pos = c + w * aUV.x * width * 0.5;
  vPos = pos;
  vN = normalize(cross(w, t));
  vUV = aUV;
  vLook = iLook;
  gl_Position = projectionMatrix * viewMatrix * vec4(pos, 1.0);
}
`;

const FRAG = /* glsl */ `
uniform vec3 uSkyAmb;
uniform vec3 uStraw;
uniform vec3 uStrawDark;
uniform vec3 uFrostCol;
uniform float uFrostAmt;
uniform float uTrans;
uniform vec4 uLightPos[4];
uniform vec3 uLightCol[4];
uniform float uLightRange[4];
uniform int uLightCount;
uniform vec4 uGroundPos[${MAX_GROUND}];   // xyz, w = radius of the glowing patch (m)
uniform vec3 uGroundCol[${MAX_GROUND}];
uniform int uGroundCount;
uniform float uGroundRange;
uniform float uGain;
uniform float uTallGain;
varying vec3 vPos;
varying vec3 vN;
varying vec2 vUV;
varying vec4 vLook;

vec3 shadeLight(vec3 P, vec3 N, vec3 V, vec3 alb, float frost, vec3 Lp, vec3 col, float r0, float range) {
  vec3 lv = Lp - P;
  float d2 = dot(lv, lv);
  vec3 l = lv * inversesqrt(d2);
  float rq = d2 / (range * range);
  vec3 E = col / (d2 + r0 * r0) / (1.0 + rq * rq);
  float ndl = abs(dot(N, l));
  vec3 o = alb * E * (0.25 + 0.75 * ndl);
  // Back light through the blade (dry grass is thin): stronger when the light is behind it.
  float back = pow(max(dot(-V, l), 0.0), 1.5);
  o += uStraw * vec3(1.0, 0.82, 0.55) * E * uTrans * back * (1.0 - 0.7 * frost);
  // Frost sheen towards the light.
  vec3 h = normalize(l + V);
  o += uFrostCol * E * frost * 0.5 * pow(abs(dot(N, h)), 30.0);
  return o;
}

void main() {
  vec3 V = normalize(cameraPosition - vPos);
  vec3 N = normalize(vN);
  if (dot(N, V) < 0.0) N = -N;
  float v = vUV.y;
  float edge = abs(vUV.x);
  // Soft blade edge (antialiasing inside the ribbon).
  float tone = vLook.x;
  vec3 alb = mix(uStrawDark, uStraw, smoothstep(0.0, 0.7, v)) * (0.75 + 0.5 * tone);
  // A darker midrib and paler edges.
  alb *= 0.85 + 0.15 * smoothstep(0.0, 0.6, edge);
  float frost = clamp(uFrostAmt * vLook.y * (smoothstep(0.45, 1.0, edge) + 0.8 * smoothstep(0.55, 1.0, v)), 0.0, 1.0);
  alb = mix(alb, uFrostCol, frost);
  // Sky light reaches into a dense canopy only near the top: the lower blade is in shade.
  vec3 L = alb * uSkyAmb * (0.12 + 0.88 * v * v) * (0.6 + 0.4 * max(N.y, 0.0));
  for (int i = 0; i < 4; i++) {
    if (i >= uLightCount) break;
    L += shadeLight(vPos, N, V, alb, frost, uLightPos[i].xyz, uLightCol[i], uLightPos[i].w, uLightRange[i]);
  }
  for (int i = 0; i < ${MAX_GROUND}; i++) {
    if (i >= uGroundCount) break;
    // A glowing patch on the ground: light from slightly above its centre, softened by its radius.
    L += shadeLight(vPos, N, V, alb, frost, uGroundPos[i].xyz, uGroundCol[i], uGroundPos[i].w, uGroundRange);
  }
  gl_FragColor = vec4(L * uGain * mix(1.0, uTallGain, vLook.w), 1.0);
}
`;

export class GrassField {
  readonly mesh: THREE.Mesh;
  readonly material: THREE.ShaderMaterial;
  readonly count: number;
  /** Blade data (for frost crystals etc.): base xyz, height, lean azimuth, lean angle, curl. */
  readonly blades: { x: number; y: number; z: number; h: number; az: number; lean: number; curl: number; frost: number }[] = [];

  constructor(spec: GrassSpec) {
    const rng = new Rng(`${spec.key}/grass`);
    const K = spec.segments ?? 7;
    const n = spec.count;
    const [h0, h1] = spec.height ?? [0.03, 0.09];
    const [w0, w1] = spec.width ?? [0.0018, 0.0042];
    const [t0, t1] = spec.tallHeight ?? [0.12, 0.24];
    const tallF = spec.tallFraction ?? 0.06;
    const leanMax = spec.lean ?? 0.5;
    const wd = spec.wedge;

    // Base strip.
    const uv = new Float32Array((K + 1) * 2 * 2);
    const idx: number[] = [];
    for (let k = 0; k <= K; k++) {
      const v = k / K;
      uv.set([-1, v, 1, v], k * 4);
      if (k < K) {
        const a = k * 2;
        idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
      }
    }
    const geo = new THREE.InstancedBufferGeometry();
    geo.setAttribute('aUV', new THREE.BufferAttribute(uv, 2));
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array((K + 1) * 2 * 3), 3));
    geo.setIndex(idx);

    const base = new Float32Array(n * 4);
    const shape = new Float32Array(n * 4);
    const look = new Float32Array(n * 4);
    // Clump centres (blades of a tussock lean outwards from it).
    const area = wd.halfAngle * (wd.far * wd.far - wd.near * wd.near);
    const nClumps = Math.max(1, Math.round((spec.clumpsPerM2 ?? 0) * area));
    const clumps: [number, number, number][] = [];
    const sampleWedge = (): [number, number] => {
      const r = Math.sqrt(wd.near * wd.near + rng.next() * (wd.far * wd.far - wd.near * wd.near));
      const th = wd.yaw + (rng.next() * 2 - 1) * wd.halfAngle;
      return [wd.x + Math.sin(th) * r, wd.z - Math.cos(th) * r];
    };
    if (spec.clumpsPerM2) for (let i = 0; i < nClumps; i++) clumps.push([...sampleWedge(), 0.04 + 0.08 * rng.next()]);
    for (let i = 0; i < n; i++) {
      let x: number;
      let z: number;
      let az = rng.next() * Math.PI * 2;
      if (clumps.length && rng.next() < 0.75) {
        const c = clumps[Math.floor(rng.next() * clumps.length)];
        const rr = c[2] * Math.sqrt(rng.next());
        const aa = rng.next() * Math.PI * 2;
        x = c[0] + Math.cos(aa) * rr;
        z = c[1] + Math.sin(aa) * rr;
        az = aa + (rng.next() - 0.5) * 0.8; // lean outwards
      } else {
        [x, z] = sampleWedge();
      }
      const tall = rng.next() < tallF;
      // Never degenerate (a zero-height blade would put NaNs in the HDR target).
      const h = Math.max((tall ? t0 + (t1 - t0) * rng.next() : h0 + (h1 - h0) * Math.pow(rng.next(), 0.8)) * (spec.heightScale?.(x, z, tall) ?? 1), 0.003);
      const y = spec.heightAt(x, z);
      const lean = (tall ? 0.25 : 1) * leanMax * Math.pow(rng.next(), 0.7);
      const curl = (rng.next() - 0.3) * 0.6;
      const frost = 0.5 + 0.5 * rng.next();
      base.set([x, y, z, h], i * 4);
      const wr = rng.next();
      const bw = tall && spec.tallWidth ? spec.tallWidth[0] + (spec.tallWidth[1] - spec.tallWidth[0]) * wr * wr : (w0 + (w1 - w0) * wr) * (tall ? 0.7 : 1);
      shape.set([bw, az, lean, curl], i * 4);
      look.set([rng.next(), frost, rng.next(), tall ? 1 : 0], i * 4);
      this.blades.push({ x, y, z, h, az, lean, curl, frost });
    }
    geo.setAttribute('iBase', new THREE.InstancedBufferAttribute(base, 4));
    geo.setAttribute('iShape', new THREE.InstancedBufferAttribute(shape, 4));
    geo.setAttribute('iLook', new THREE.InstancedBufferAttribute(look, 4));
    geo.instanceCount = n;
    this.count = n;

    this.material = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: {
        uSkyAmb: { value: new THREE.Vector3() },
        uStraw: { value: new THREE.Vector3() },
        uStrawDark: { value: new THREE.Vector3() },
        uFrostCol: { value: new THREE.Vector3() },
        uFrostAmt: { value: 0.7 },
        uTrans: { value: 0.5 },
        uLightPos: { value: [0, 1, 2, 3].map(() => new THREE.Vector4()) },
        uLightCol: { value: [0, 1, 2, 3].map(() => new THREE.Vector3()) },
        uLightRange: { value: [1e6, 1e6, 1e6, 1e6] },
        uLightCount: { value: 0 },
        uGroundPos: { value: Array.from({ length: MAX_GROUND }, () => new THREE.Vector4()) },
        uGroundCol: { value: Array.from({ length: MAX_GROUND }, () => new THREE.Vector3()) },
        uGroundCount: { value: 0 },
        uGroundRange: { value: 0.4 },
        uGain: { value: 1 },
        uTallGain: { value: 1 },
      },
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.frustumCulled = false;
  }

  /** Position of a point along a blade (v in 0..1), CPU twin of the centre line. */
  bladePoint(i: number, v: number): THREE.Vector3 {
    const b = this.blades[i];
    const ld = new THREE.Vector3(Math.cos(b.az), 0, Math.sin(b.az));
    const p0 = new THREE.Vector3(b.x, b.y, b.z);
    const p1 = p0.clone().addScaledVector(new THREE.Vector3(0, 1, 0), b.h * 0.62).addScaledVector(ld, b.h * Math.sin(b.lean) * 0.15);
    const p2 = p0.clone().addScaledVector(new THREE.Vector3(0, 1, 0), b.h * Math.cos(b.lean + b.curl)).addScaledVector(ld, b.h * Math.sin(b.lean + b.curl));
    return p0.multiplyScalar((1 - v) * (1 - v)).addScaledVector(p1, 2 * (1 - v) * v).addScaledVector(p2, v * v);
  }

  /**
   * lights: character lights (<= 4). ground: glowing patches on the ground (<= 12), e.g. footprints,
   * as point lights placed a little above the patch with nearRadius = patch radius; groundRange is
   * their soft effective radius (m).
   */
  update(look: GrassLook, lights: PointLightSpec[], ground: PointLightSpec[], gain = 1, groundRange = 0.4): void {
    const u = this.material.uniforms;
    u.uGroundRange.value = groundRange;
    u.uSkyAmb.value.set(...look.skyAmbient);
    u.uStraw.value.set(...look.straw);
    u.uStrawDark.value.set(...look.strawDark);
    u.uFrostCol.value.set(...look.frost);
    u.uFrostAmt.value = look.frostAmount;
    u.uTrans.value = look.translucency;
    u.uGain.value = gain;
    u.uTallGain.value = look.tallGain ?? 1;
    const n = Math.min(lights.length, 4);
    u.uLightCount.value = n;
    for (let i = 0; i < 4; i++) {
      const l = lights[i];
      if (!l) continue;
      u.uLightPos.value[i].set(l.position.x, l.position.y, l.position.z, l.nearRadius);
      u.uLightCol.value[i].set(...l.color).multiplyScalar(l.intensity);
      u.uLightRange.value[i] = l.range ?? 1e6;
    }
    const m = Math.min(ground.length, MAX_GROUND);
    u.uGroundCount.value = m;
    for (let i = 0; i < m; i++) {
      const g = ground[i];
      u.uGroundPos.value[i].set(g.position.x, g.position.y, g.position.z, g.nearRadius);
      u.uGroundCol.value[i].set(...g.color).multiplyScalar(g.intensity);
    }
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.material.dispose();
  }
}
