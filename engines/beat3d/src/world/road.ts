// RoadOfLight: 「路」, a permanent path of light. A first road, a network of roads, a road under snow are
// the same thing: a ribbon draped on the ground along a centripetal
// Catmull-Rom spline (or a polyline), with width, brightness and colour given as functions of the
// arc length, glowing additively in the HDR world layer.
//
// Across the road: a trodden core plus a soft shoulder (the light spilling on the ground). Along it:
// a fine worn grain and, optionally, the footprints it was made of (beads alternating left/right at
// the stride), so a young road still reads as "many steps". `reveal` draws the road on between two
// arc lengths (in metres), with a soft head; `lights()` gives point lights spaced along the lit part
// so terrain and grass (Terrain / GrassField) are lit by it.
//
// Far roads go sub-pixel: the ribbon is widened in the vertex shader to at least `minPx` canvas
// pixels and dimmed by the same factor, so it stays a continuous, energy-correct line (no crawling).
//
// Everything is a pure function of the spec and the look you pass per frame.

import * as THREE from 'three';
import { NOISE_GLSL } from '../gl/noise.glsl';
import { seedFloat } from '../lib/rng';
import { PALETTE } from './lights';
import type { PointLightSpec } from './terrain';

type RGB = [number, number, number];

export interface RoadSpec {
  /** Seed key for the grain (e.g. `${shotId}/road`). */
  key: string;
  /** Control points on the ground, XZ (metres). At least 2. */
  points: [number, number][];
  /** Ground height (normally `(x, z) => terrainHeight(x, z, shape)`). */
  heightAt: (x: number, z: number) => number;
  /** Centripetal Catmull-Rom through the points (default true); false = polyline. */
  smooth?: boolean;
  /** Sample spacing along the road (m, default 0.5). Use ~1/100 of the visible length. */
  spacing?: number;
  /** Full width (m): a number, or a function of (u in 0..1, s in metres). Default 1.2. */
  width?: number | ((u: number, s: number) => number);
  /** Relative brightness along the road (default 1). */
  brightness?: (u: number, s: number) => number;
  /** Linear colour along the road (default PALETTE.warmHaloIn). */
  color?: (u: number, s: number) => RGB;
  /** Vertices across (odd, default 5): more for wide roads on relief. */
  across?: number;
  /** Lift above the ground (m, default 0.02); a polygon offset handles the rest. */
  lift?: number;
}

export interface RoadLook {
  /** Overall gain (HDR linear). ~1–3 reads as "glowing" in the night grade. */
  gain: number;
  /** Lit part [from, to] in metres of arc length (default: all). Animate `to` to lay the road. */
  reveal?: [number, number];
  /** Softness of the reveal head and tail (m, default 2). */
  revealSoft?: number;
  /** Optional: a different softness at the `to` end (default = revealSoft); e.g. a comet tail: long soft tail, short head. */
  revealHeadSoft?: number;
  /** Share of the bright trodden core vs the soft shoulder (0..1, default 0.6). */
  core?: number;
  /** Spill on the ground beyond the edge, 0..2 (default 1.0: a soft glow about half a width past the edge). */
  shoulder?: number;
  /** Worn grain along the road (0..1, default 0.25). */
  grain?: number;
  /** Footprint beads: stride (m) and amount (0..1). Default off. */
  stride?: number;
  strideAmount?: number;
  /** Minimum on-screen width (canvas px at this resolution, default 1.5 x scale). */
  minPx?: number;
  // ---- walked-path options (all default off, so the default look is unchanged) ----
  /** Trampled ground under the road: linear colour and opacity (0..1). The path's own surface
   *  (crushed frost, flattened grass) replaces the plain's texture there; the glow is added on top. */
  base?: [number, number, number];
  baseAlpha?: number;
  /** Footprints the road is made of: overlapping ellipses alternating left/right. stride = one
   *  foot's step (m); size = [width, length] (m); spread = sideways offset (m); amount 0..1. */
  prints?: { stride: number; size: [number, number]; spread: number; amount: number };
  /** Ragged edges: the road's width wanders by this fraction (0..1). */
  edge?: number;
  /** Light living in the crushed frost crystals: sparse sparkles (0..1), averaged out when far. */
  sparkle?: number;
  /** Freshest steps brightest: extra gain `gain` at arc length `at`, falling off behind over `length` m. */
  fresh?: { at: number; length: number; gain: number };
  /** The spill lands on frosted grass (0..1, default 0): tufts catch it, small shadows run away from
   *  the line, so the glow reads as light on the ground rather than a flat haze. */
  spillGrass?: number;
}

const VERT = /* glsl */ `
attribute vec3 aAcross;     // unit, horizontal
attribute vec4 aRoad;       // side (-1..1 over the drawn width), s (m), half drawn width (m), brightness
attribute vec3 aColor;
uniform vec2 uRes;
uniform float uMinPx;
varying vec3 vCol;
varying vec3 vRoad;         // side, s, energy
varying vec3 vW;            // world position
varying float vHW;          // road half-width (m)
void main() {
  // Pixel width of the drawn half-width at this vertex; widen sub-pixel roads, dim to conserve energy.
  vec3 centre = position - aAcross * aRoad.x * aRoad.z;
  vec4 c0 = projectionMatrix * viewMatrix * vec4(centre, 1.0);
  vec4 c1 = projectionMatrix * viewMatrix * vec4(centre + aAcross * aRoad.z, 1.0);
  vec2 p0 = c0.xy / max(c0.w, 1e-4) * 0.5 * uRes;
  vec2 p1 = c1.xy / max(c1.w, 1e-4) * 0.5 * uRes;
  float px = max(length(p1 - p0), 1e-3);
  // The ribbon is drawn 3 road half-widths wide (room for the spill); compare the road itself.
  float k = max(1.0, 0.5 * uMinPx / (px / 3.0));
  vec3 p = position + aAcross * aRoad.x * aRoad.z * (k - 1.0);
  vCol = aColor;
  vRoad = vec3(aRoad.x, aRoad.y, aRoad.w / k);
  vW = p;
  vHW = aRoad.z * k / 3.0;
  gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
}
`;

const FRAG = /* glsl */ `
${NOISE_GLSL}
uniform float uGain, uCore, uShoulder, uGrain, uStride, uStrideAmt, uSeed;
uniform vec3 uReveal;       // from, to, soft (m)
uniform float uRevealHead;  // softness at the head (to) end, m
uniform vec4 uBase;         // trampled ground colour, opacity
uniform vec4 uPrints;       // stride, width, length, amount
uniform float uPrintSpread;
uniform float uEdge, uSparkle, uSpillGrass;
uniform vec3 uFresh;        // at, length, gain
varying vec3 vCol;
varying vec3 vRoad;
varying vec3 vW;
varying float vHW;
float h21(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
void main() {
  float x = vRoad.x * 3.0;   // road units: 0 at the centre, ±1 at the road edge, ±3 at the ribbon edge
  float s = vRoad.y;
  // Ragged edges: the width wanders (two scales) where the walkers strayed or the frost held.
  float wf = 1.0 + uEdge * (0.55 * snoise(vec2(s * 0.45 + uSeed, 3.1)) + 0.45 * snoise(vec2(s * 2.3 - uSeed, 8.7 + x * 0.6)));
  // Across: trodden core (flat-topped) + spill on the ground beyond the edge.
  float core = exp(-pow(abs(x) / (0.85 * max(wf, 0.25)), 4.0));
  float sg = 0.45 + 0.6 * uShoulder;
  float spill = exp(-x * x / (2.0 * sg * sg));
  if (uSpillGrass > 0.0) {
    // Light on frosted grass: tufts (fine along the road), shadows stretched across it (away from
    // the line light); averaged to its mean once a tuft is under a pixel.
    float xm0 = x * vHW;
    float n1 = snoise(vec2(s / 0.07 + uSeed, xm0 / 0.32));
    float n2 = snoise(vec2(s / 0.028 + 7.0, xm0 / 0.1 - uSeed));
    float gp = clamp(0.5 + 0.85 * n1 + 0.4 * n2, 0.0, 1.8);
    float fwg = length(fwidth(vec2(s / 0.07, xm0 / 0.32)));
    gp = mix(gp, 0.62, smoothstep(0.6, 1.6, fwg));
    spill *= mix(1.0, gp * smoothstep(0.6, 1.4, abs(x)) + (1.0 - smoothstep(0.6, 1.4, abs(x))), uSpillGrass);
  }
  float across = (uCore * core + (1.0 - uCore) * spill) * (1.0 - smoothstep(0.7, 1.0, abs(vRoad.x)));
  // Along: worn grain (two scales, stretched along the road) and the footprints it is made of.
  float g = 0.6 * snoise(vec2(s * 0.35 + uSeed, x * 1.3)) + 0.4 * snoise(vec2(s * 1.9 - uSeed, x * 3.1 + 7.0));
  float grain = 1.0 + uGrain * g;
  float beads = 1.0;
  if (uStride > 0.0) {
    float t = s / uStride;
    float k = floor(t + 0.5);
    float side = mod(k, 2.0) < 0.5 ? 0.32 : -0.32;
    float d = (t - k) * uStride;
    float b = exp(-d * d / (0.06 * uStride * uStride) - (x - side) * (x - side) / 0.05);
    beads = 1.0 - uStrideAmt + uStrideAmt * 2.2 * b;
  }
  // The footprints it is made of: overlapping soft ellipses, alternating feet, a little irregular.
  float pr = 1.0;
  if (uPrints.w > 0.0) {
    float xm = x * vHW;
    float half_ = 0.5 * uPrints.x;
    float k0 = floor(s / half_ + 0.5);
    float m = 0.0;
    for (int j = -1; j <= 1; j++) {
      float k = k0 + float(j);
      float hk = h21(vec2(k, uSeed));
      float sk = k * half_ + (hk - 0.5) * 0.25 * half_;
      float side = (mod(k, 2.0) < 0.5 ? 1.0 : -1.0) * uPrintSpread * (0.7 + 0.6 * h21(vec2(uSeed, k)));
      vec2 q = vec2((s - sk) / (0.5 * uPrints.z), (xm - side) / (0.5 * uPrints.y));
      m = max(m, exp(-1.6 * dot(q, q)) * (0.75 + 0.5 * hk));
    }
    pr = mix(1.0, 0.45 + 1.5 * m, uPrints.w);
  }
  // Crushed frost: sparse crystals that hold the light; averaged out once a cell is under a pixel.
  float sp = 1.0;
  if (uSparkle > 0.0) {
    vec2 c = vW.xz / 0.045;
    vec2 ci = floor(c);
    float hc = h21(ci + uSeed);
    vec2 f = fract(c) - 0.5 - (vec2(h21(ci + 3.1), h21(ci + 7.7)) - 0.5) * 0.6;
    float pt = step(0.86, hc) * exp(-dot(f, f) * 40.0) * (hc - 0.86) * 7.0 * 9.0;
    float fw = length(fwidth(c));
    sp = 1.0 + uSparkle * mix(pt, 0.25, smoothstep(0.35, 1.2, fw));
  }
  float fr = 1.0 + uFresh.z * exp(-max(uFresh.x - s, 0.0) / max(uFresh.y, 1e-3));
  float rv = smoothstep(uReveal.x - uReveal.z, uReveal.x, s) * (1.0 - smoothstep(uReveal.y - uRevealHead, uReveal.y, s));
  float e = across * grain * beads * pr * sp * fr * rv * vRoad.z * uGain;
  // Trampled ground (premultiplied over the plain; 0 opacity = purely additive, as before).
  float a = uBase.a * core * rv;
  gl_FragColor = vec4(uBase.rgb * a + vCol * max(e, 0.0), a);
}
`;

interface Sample {
  x: number;
  z: number;
  s: number;
  tx: number;
  tz: number;
}

/** Centripetal Catmull-Rom (alpha 0.5) or polyline, resampled at ~`spacing` metres. */
function resample(points: [number, number][], smooth: boolean, spacing: number): Sample[] {
  const dense: [number, number][] = [];
  if (!smooth || points.length < 3) {
    for (let i = 0; i < points.length - 1; i++) {
      const [ax, az] = points[i];
      const [bx, bz] = points[i + 1];
      const n = Math.max(1, Math.ceil(Math.hypot(bx - ax, bz - az) / (spacing * 0.25)));
      for (let j = 0; j < n; j++) dense.push([ax + ((bx - ax) * j) / n, az + ((bz - az) * j) / n]);
    }
    dense.push(points[points.length - 1]);
  } else {
    const P = [
      [2 * points[0][0] - points[1][0], 2 * points[0][1] - points[1][1]] as [number, number],
      ...points,
      [2 * points[points.length - 1][0] - points[points.length - 2][0], 2 * points[points.length - 1][1] - points[points.length - 2][1]] as [number, number],
    ];
    for (let i = 1; i < P.length - 2; i++) {
      const [p0, p1, p2, p3] = [P[i - 1], P[i], P[i + 1], P[i + 2]];
      const td = (a: number[], b: number[]): number => Math.max(Math.sqrt(Math.hypot(b[0] - a[0], b[1] - a[1])), 1e-4);
      const t0 = 0;
      const t1 = t0 + td(p0, p1);
      const t2 = t1 + td(p1, p2);
      const t3 = t2 + td(p2, p3);
      const n = Math.max(2, Math.ceil(Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / (spacing * 0.25)));
      for (let j = 0; j < n; j++) {
        const t = t1 + ((t2 - t1) * j) / n;
        const pt = (k: number): number => {
          const A1 = p0[k] * ((t1 - t) / (t1 - t0)) + p1[k] * ((t - t0) / (t1 - t0));
          const A2 = p1[k] * ((t2 - t) / (t2 - t1)) + p2[k] * ((t - t1) / (t2 - t1));
          const A3 = p2[k] * ((t3 - t) / (t3 - t2)) + p3[k] * ((t - t2) / (t3 - t2));
          const B1 = A1 * ((t2 - t) / (t2 - t0)) + A2 * ((t - t0) / (t2 - t0));
          const B2 = A2 * ((t3 - t) / (t3 - t1)) + A3 * ((t - t1) / (t3 - t1));
          return B1 * ((t2 - t) / (t2 - t1)) + B2 * ((t - t1) / (t2 - t1));
        };
        dense.push([pt(0), pt(1)]);
      }
    }
    dense.push(points[points.length - 1]);
  }
  // Arc length over the dense polyline, then resample uniformly.
  const acc = [0];
  for (let i = 1; i < dense.length; i++) acc.push(acc[i - 1] + Math.hypot(dense[i][0] - dense[i - 1][0], dense[i][1] - dense[i - 1][1]));
  const L = acc[acc.length - 1];
  const n = Math.max(2, Math.ceil(L / spacing) + 1);
  const out: Sample[] = [];
  let j = 0;
  for (let i = 0; i < n; i++) {
    const s = (L * i) / (n - 1);
    while (j < dense.length - 2 && acc[j + 1] < s) j++;
    const seg = Math.max(acc[j + 1] - acc[j], 1e-9);
    const f = Math.min(Math.max((s - acc[j]) / seg, 0), 1);
    out.push({ x: dense[j][0] + (dense[j + 1][0] - dense[j][0]) * f, z: dense[j][1] + (dense[j + 1][1] - dense[j][1]) * f, s, tx: 0, tz: 0 });
  }
  for (let i = 0; i < n; i++) {
    const a = out[Math.max(0, i - 1)];
    const b = out[Math.min(n - 1, i + 1)];
    const l = Math.max(Math.hypot(b.x - a.x, b.z - a.z), 1e-9);
    out[i].tx = (b.x - a.x) / l;
    out[i].tz = (b.z - a.z) / l;
  }
  return out;
}

export class RoadOfLight {
  readonly mesh: THREE.Mesh;
  readonly material: THREE.ShaderMaterial;
  /** Total arc length (m). */
  readonly length: number;
  private readonly samples: Sample[];
  private readonly spec: RoadSpec;

  constructor(spec: RoadSpec) {
    if (spec.points.length < 2) throw new Error('RoadOfLight: need at least 2 points');
    this.spec = spec;
    const spacing = spec.spacing ?? 0.5;
    this.samples = resample(spec.points, spec.smooth ?? true, spacing);
    const n = this.samples.length;
    this.length = this.samples[n - 1].s;
    const M = Math.max(3, (spec.across ?? 5) | 1);
    const lift = spec.lift ?? 0.02;
    const widthOf = typeof spec.width === 'function' ? spec.width : ((w: number) => () => w)(spec.width ?? 1.2);
    // The drawn ribbon is 3x the road's width, leaving room for the spill on either side (the
    // fragment shader works in road units: ±1 = road edge, ±3 = ribbon edge).
    const pos = new Float32Array(n * M * 3);
    const acr = new Float32Array(n * M * 3);
    const road = new Float32Array(n * M * 4);
    const col = new Float32Array(n * M * 3);
    for (let i = 0; i < n; i++) {
      const p = this.samples[i];
      const u = this.length > 0 ? p.s / this.length : 0;
      const hw = 0.5 * widthOf(u, p.s) * 3.0;
      const b = spec.brightness?.(u, p.s) ?? 1;
      const c = spec.color?.(u, p.s) ?? PALETTE.warmHaloIn;
      const ax = -p.tz;
      const az = p.tx;
      for (let j = 0; j < M; j++) {
        const side = -1 + (2 * j) / (M - 1);
        const x = p.x + ax * side * hw;
        const z = p.z + az * side * hw;
        const k = i * M + j;
        pos.set([x, spec.heightAt(x, z) + lift, z], k * 3);
        acr.set([ax, 0, az], k * 3);
        road.set([side, p.s, hw, b], k * 4);
        col.set(c, k * 3);
      }
    }
    const idx: number[] = [];
    for (let i = 0; i < n - 1; i++) {
      for (let j = 0; j < M - 1; j++) {
        const a = i * M + j;
        idx.push(a, a + M, a + 1, a + 1, a + M, a + M + 1);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('aAcross', new THREE.BufferAttribute(acr, 3));
    geo.setAttribute('aRoad', new THREE.BufferAttribute(road, 4));
    geo.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
    geo.setIndex(idx);
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: {
        uRes: { value: new THREE.Vector2(1920, 1080) },
        uMinPx: { value: 1.5 },
        uGain: { value: 1 },
        uCore: { value: 0.6 },
        uShoulder: { value: 1.0 },
        uGrain: { value: 0.25 },
        uStride: { value: 0 },
        uStrideAmt: { value: 0 },
        uSeed: { value: seedFloat(spec.key) % 97 },
        uReveal: { value: new THREE.Vector3(-1e9, 1e9, 2) },
        uRevealHead: { value: 2 },
        uBase: { value: new THREE.Vector4(0, 0, 0, 0) },
        uPrints: { value: new THREE.Vector4(1, 0.2, 0.3, 0) },
        uPrintSpread: { value: 0.12 },
        uEdge: { value: 0 },
        uSparkle: { value: 0 },
        uSpillGrass: { value: 0 },
        uFresh: { value: new THREE.Vector3(0, 1, 0) },
      },
      transparent: true,
      // Premultiplied "over": with zero opacity (the default) this is plain additive glow.
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
      depthWrite: false,
      depthTest: true,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -4,
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.frustumCulled = false;
  }

  /** Per frame. W, H: canvas size; scale: canvas height / 1080. */
  update(look: RoadLook, W: number, H: number, scale = H / 1080): void {
    const u = this.material.uniforms;
    u.uRes.value.set(W, H);
    u.uMinPx.value = look.minPx ?? 1.5 * scale;
    u.uGain.value = look.gain;
    u.uCore.value = look.core ?? 0.6;
    u.uShoulder.value = Math.min(Math.max(look.shoulder ?? 1.0, 0), 2);
    u.uGrain.value = look.grain ?? 0.25;
    u.uStride.value = look.stride ?? 0;
    u.uStrideAmt.value = look.stride ? (look.strideAmount ?? 0.5) : 0;
    u.uBase.value.set(...(look.base ?? [0, 0, 0]), look.base ? (look.baseAlpha ?? 0.8) : 0);
    const pr = look.prints;
    u.uPrints.value.set(pr?.stride ?? 1, pr?.size[0] ?? 0.2, pr?.size[1] ?? 0.3, pr?.amount ?? 0);
    u.uPrintSpread.value = pr?.spread ?? 0.12;
    u.uEdge.value = look.edge ?? 0;
    u.uSparkle.value = look.sparkle ?? 0;
    u.uSpillGrass.value = look.spillGrass ?? 0;
    u.uFresh.value.set(look.fresh?.at ?? 0, look.fresh?.length ?? 1, look.fresh?.gain ?? 0);
    const [a, b] = look.reveal ?? [-1e9, 1e9];
    u.uReveal.value.set(a, b, Math.max(look.revealSoft ?? 2, 1e-3));
    u.uRevealHead.value = Math.max(look.revealHeadSoft ?? look.revealSoft ?? 2, 1e-3);
    this.mesh.visible = look.gain > 0 && b > a;
  }

  /** Point on the road centre (on the ground) at arc length s (clamped), with its unit tangent. */
  sample(s: number): { position: THREE.Vector3; tangent: THREE.Vector3; u: number } {
    const S = this.samples;
    const t = Math.min(Math.max(s, 0), this.length);
    let lo = 0;
    let hi = S.length - 1;
    while (hi - lo > 1) {
      const m = (lo + hi) >> 1;
      if (S[m].s <= t) lo = m;
      else hi = m;
    }
    const f = (t - S[lo].s) / Math.max(S[hi].s - S[lo].s, 1e-9);
    const x = S[lo].x + (S[hi].x - S[lo].x) * f;
    const z = S[lo].z + (S[hi].z - S[lo].z) * f;
    const tx = S[lo].tx + (S[hi].tx - S[lo].tx) * f;
    const tz = S[lo].tz + (S[hi].tz - S[lo].tz) * f;
    const l = Math.max(Math.hypot(tx, tz), 1e-9);
    return {
      position: new THREE.Vector3(x, this.spec.heightAt(x, z), z),
      tangent: new THREE.Vector3(tx / l, 0, tz / l),
      u: this.length > 0 ? t / this.length : 0,
    };
  }

  /**
   * `count` point lights spaced along the lit part [from, to] (default: the look's reveal, or all),
   * `height` m above the road, coloured by the road's colour, for Terrain (max 4) / GrassField (max 12).
   */
  lights(count: number, intensity: number, from = 0, to = this.length, height = 0.4, range?: number): PointLightSpec[] {
    const a = Math.max(0, from);
    const b = Math.min(this.length, to);
    const out: PointLightSpec[] = [];
    if (count <= 0 || b <= a) return out;
    for (let i = 0; i < count; i++) {
      const s = a + ((b - a) * (i + 0.5)) / count;
      const { position, u } = this.sample(s);
      const w = typeof this.spec.width === 'function' ? this.spec.width(u, s) : (this.spec.width ?? 1.2);
      const br = this.spec.brightness?.(u, s) ?? 1;
      out.push({
        position: position.add(new THREE.Vector3(0, height, 0)),
        color: this.spec.color?.(u, s) ?? PALETTE.warmHaloIn,
        intensity: intensity * br,
        nearRadius: Math.max(0.5 * w, 0.05),
        range,
      });
    }
    return out;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.material.dispose();
  }
}
