// 霜晶: frost sparkles. Hoar-frost plates on grass tips are tiny mirrors; each one flashes only when
// the half-vector between its light and the eye lines up with its own normal. We scatter crystals
// on the terrain once (seeded), and in the vertex shader evaluate, for every crystal and every point
// light, the mirror lobe exp(-angle(n, h)^2 / tau^2) times the light's irradiance. Moving lights or a
// moving eye make different crystals flash, so the sparkle "walks" with the light and is never
// random over time: it is a pure function of the geometry.
//
// Crystals are point sprites with a resolution-independent flux (like stars), depth-tested against
// the scene (so terrain and grass hide them) and drawn additively; DOF turns the defocused ones
// into bokeh discs.

import * as THREE from 'three';
import { Rng } from '../lib/rng';
import { terrainHeight, type PointLightSpec, type TerrainShape } from './terrain';

export interface FrostRegion {
  /** Centre and half-extents (metres) of the elliptical patch, with an optional rotation about +Y. */
  x: number;
  z: number;
  rx: number;
  rz: number;
  angle?: number;
}

export interface FrostScatterOptions {
  /** Crystals in the region. */
  count: number;
  /** Height range above the ground (frost sits on blade tips). Default [0.0, 0.18] m. */
  height?: [number, number];
  /** Spread of the plate normals around +Y (radians, gaussian sigma). Default 0.55. */
  normalSpread?: number;
  /** Crystal size range (metres): only matters close up, where a crystal covers pixels. Default [0.6, 2.2] mm. */
  size?: [number, number];
  /** Optional height function override (e.g. a macro ground with its own relief). */
  heightAt?: (x: number, z: number) => number;
}

const VERT = /* glsl */ `
attribute vec3 aNormal;
attribute float aSize;
attribute float aRand;
uniform vec4 uLightPos[4];
uniform vec3 uLightCol[4];
uniform float uLightRange[4];
uniform int uLightCount;
uniform float uTau;        // mirror lobe width (radians)
uniform float uGain;
uniform float uPxPerRad;   // canvas px per radian at the image centre
uniform float uScale;
uniform float uDistRef;    // flux falls as (ref / d)^pow beyond ref (keeps far sparkle visible but smaller)
uniform float uDistPow;
uniform vec3 uSkyCol;      // optional faint sky glints (cool), 0 = off
uniform vec4 uBokeh;       // focus (m), CoC K (diameter px = K |z - F| / z), max radius px, on (0/1)
uniform vec4 uRingA[2];    // centre xyz, radius (m)
uniform vec4 uRingB[2];    // colour rgb, width (m); colour 0 = off
varying vec3 vCol;
varying float vSig2;
varying float vR;
varying float vHex;        // > 0: draw as a hexagonal bokeh disc of this radius (px)
void main() {
  vec3 P = position;
  vec3 V = cameraPosition - P;
  float dc = length(V);
  V /= dc;
  vec3 n = normalize(aNormal);
  vec3 sum = vec3(0.0);
  for (int i = 0; i < 4; i++) {
    if (i >= uLightCount) break;
    vec3 lv = uLightPos[i].xyz - P;
    float d2 = dot(lv, lv);
    vec3 l = lv * inversesqrt(d2);
    vec3 h = normalize(l + V);
    float c = clamp(dot(n, h), -1.0, 1.0);
    float a = acos(c);
    float lobe = exp(-a * a / (uTau * uTau));
    float r0 = uLightPos[i].w;
    float rq = d2 / (uLightRange[i] * uLightRange[i]);
    sum += uLightCol[i] / (d2 + r0 * r0) / (1.0 + rq * rq) * lobe * step(0.0, dot(n, l)) * step(0.0, dot(n, V));
  }
  // Rings: a wave of light running out through the frost (a landing footfall); every crystal it passes
  // lights up, whatever its tilt (some more than others).
  for (int i = 0; i < 2; i++) {
    if (uRingB[i].x + uRingB[i].y + uRingB[i].z <= 0.0) continue;
    float dd = abs(length(P.xz - uRingA[i].xz) - uRingA[i].w) / uRingB[i].w;
    sum += uRingB[i].rgb * exp(-dd * dd) * (0.3 + 1.4 * aRand * aRand);
  }
  // Faint sky glints: plates tilted towards the bright horizon behind the eye's line of sight.
  if (uSkyCol.x + uSkyCol.y + uSkyCol.z > 0.0) {
    vec3 r = reflect(-V, n);
    float hor = exp(-pow(r.y / 0.08, 2.0)) * step(0.0, r.y + 0.02);
    sum += uSkyCol * hor * step(0.92, aRand);
  }
  float att = pow(uDistRef / max(dc, uDistRef), uDistPow);
  // PSF: at least ~0.6 px (1080p), or the crystal's own projected size close up.
  float sz = aSize / dc * uPxPerRad;
  float s1080 = max(0.6 * uScale, 0.5 * sz);
  vSig2 = s1080 * s1080 + 1.0 / 12.0;
  float flux = uGain * att * (0.6 + 0.8 * aRand);
  vCol = sum * flux / (6.2831853 * vSig2 / (uScale * uScale));
  float lum = vCol.r + vCol.g + vCol.b;
  vR = ceil(3.0 * sqrt(vSig2)) + 1.0;
  vHex = 0.0;
  vec4 Pv = viewMatrix * vec4(P, 1.0);
  float z = -Pv.z;
  if (uBokeh.w > 0.5 && z < uBokeh.x) {
    // Defocused crystal in front of the focus plane: our own aperture-shaped disc, drawn on the
    // focus plane along the same ray (so the post DOF leaves it alone), flux spread over its area.
    float r = min(0.5 * uBokeh.y * (uBokeh.x - z) / max(z, 1e-3), uBokeh.z);
    if (r > 1.2 * uScale) {
      vHex = r;
      vR = r + 2.0 * uScale;
      vCol = sum * uGain * att * (0.6 + 0.8 * aRand) / (3.14159265 * r * r / (uScale * uScale));
      Pv.xyz *= uBokeh.x / max(z, 1e-3);
      lum = vCol.r + vCol.g + vCol.b;
      // A disc writes the focus depth (keeps the post DOF off it), so a faint one would show the
      // sharp scene behind it as a dark hexagon: only discs that clearly read are drawn.
      if (lum < 0.015) lum = 0.0;
    } else lum = 0.0;
  } else if (uBokeh.w > 0.5) lum = 0.0;
  gl_Position = projectionMatrix * Pv;
  gl_PointSize = lum > 1e-5 ? 2.0 * vR : 0.0;
}
`;

const FRAG = /* glsl */ `
varying vec3 vCol;
varying float vSig2;
varying float vR;
varying float vHex;
// Six-bladed aperture: 1 inside, soft 1 px edge, a faint brighter rim, the red disc a touch larger
// than the blue one (lateral colour), so a defocused sparkle reads as glass, not as fog.
float hexDisc(vec2 p, float r) {
  vec2 a = abs(p);
  float h = max(a.x * 0.8660254 + a.y * 0.5, a.y);
  float e = h - 0.8660254 * r;
  return (1.0 - smoothstep(-0.8, 0.8, e)) * (1.0 + 0.35 * smoothstep(-0.18 * r, 0.0, e));
}
void main() {
  vec2 d = (gl_PointCoord - 0.5) * 2.0 * vR;
  if (vHex > 0.0) {
    vec3 m = vec3(hexDisc(d, vHex * 1.025), hexDisc(d, vHex), hexDisc(d, vHex * 0.975));
    if (m.r < 0.02) discard;
    gl_FragColor = vec4(vCol * m, 1.0);
    return;
  }
  float g = exp(-0.5 * dot(d, d) / vSig2);
  gl_FragColor = vec4(vCol * g, 1.0);
}
`;

/**
 * Shaped bokeh for crystals in front of the focus plane (match the shot's post `dof`). Use it on a
 * FrostGlints that holds only foreground crystals: in this mode crystals that are not clearly
 * defocused (or lie behind the focus plane) are not drawn.
 */
export interface FrostBokeh {
  focus: number;
  focalMm: number;
  fStop: number;
  /** Canvas width (px). */
  width: number;
  /** Max disc radius (1080p px). */
  maxRadius: number;
}

/** A ring of light running out through the frost (centre on the ground, radius and width in m). */
export interface FrostRing {
  center: THREE.Vector3;
  radius: number;
  width: number;
  /** Linear colour x intensity (irradiance-like, as a point light's colour x intensity / d^2). */
  color: [number, number, number];
}

export class FrostGlints {
  readonly points: THREE.Points;
  readonly material: THREE.ShaderMaterial;
  readonly count: number;

  constructor(positions: Float32Array, normals: Float32Array, sizes: Float32Array, rands: Float32Array) {
    this.count = sizes.length;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('aNormal', new THREE.BufferAttribute(normals, 3));
    geo.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
    geo.setAttribute('aRand', new THREE.BufferAttribute(rands, 1));
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: {
        uLightPos: { value: [0, 1, 2, 3].map(() => new THREE.Vector4()) },
        uLightCol: { value: [0, 1, 2, 3].map(() => new THREE.Vector3()) },
        uLightRange: { value: [1e6, 1e6, 1e6, 1e6] },
        uLightCount: { value: 0 },
        uTau: { value: 0.05 },
        uGain: { value: 1 },
        uPxPerRad: { value: 1000 },
        uScale: { value: 1 },
        uDistRef: { value: 10 },
        uDistPow: { value: 1 },
        uSkyCol: { value: new THREE.Vector3() },
        uBokeh: { value: new THREE.Vector4(1, 0, 0, 0) },
        uRingA: { value: [new THREE.Vector4(), new THREE.Vector4()] },
        uRingB: { value: [new THREE.Vector4(0, 0, 0, 1), new THREE.Vector4(0, 0, 0, 1)] },
      },
      blending: THREE.AdditiveBlending,
      depthTest: true,
      depthWrite: false,
      transparent: true,
      toneMapped: false,
    });
    this.points = new THREE.Points(geo, this.material);
    this.points.frustumCulled = false;
  }

  /** Scatter crystals over an elliptical patch of terrain. Deterministic for a given key. */
  static scatter(key: string, shape: TerrainShape | null, region: FrostRegion, o: FrostScatterOptions): FrostGlints {
    const rng = new Rng(`${key}/frost`);
    const n = o.count;
    const pos = new Float32Array(n * 3);
    const nor = new Float32Array(n * 3);
    const size = new Float32Array(n);
    const rand = new Float32Array(n);
    const [h0, h1] = o.height ?? [0, 0.18];
    const [s0, s1] = o.size ?? [0.0006, 0.0022];
    const spread = o.normalSpread ?? 0.55;
    const ca = Math.cos(region.angle ?? 0);
    const sa = Math.sin(region.angle ?? 0);
    const hAt = o.heightAt ?? ((x: number, z: number) => (shape ? terrainHeight(x, z, shape) : 0));
    for (let i = 0; i < n; i++) {
      // Uniform in the ellipse.
      const r = Math.sqrt(rng.next());
      const t = rng.next() * Math.PI * 2;
      const u = r * Math.cos(t) * region.rx;
      const v = r * Math.sin(t) * region.rz;
      const x = region.x + ca * u - sa * v;
      const z = region.z + sa * u + ca * v;
      pos[i * 3] = x;
      pos[i * 3 + 1] = hAt(x, z) + h0 + (h1 - h0) * Math.pow(rng.next(), 1.6);
      pos[i * 3 + 2] = z;
      const nx = rng.normal() * spread;
      const nz = rng.normal() * spread;
      const len = Math.hypot(nx, 1, nz);
      nor[i * 3] = nx / len;
      nor[i * 3 + 1] = 1 / len;
      nor[i * 3 + 2] = nz / len;
      size[i] = s0 + (s1 - s0) * rng.next();
      rand[i] = rng.next();
    }
    return new FrostGlints(pos, nor, size, rand);
  }

  /**
   * gain: overall sparkle brightness. tau: mirror lobe (radians; 0.03-0.08). distRef/distPow: far
   * crystals dim as (ref/d)^pow. skyGlint: colour of faint horizon-sky glints (default off).
   */
  update(camera: THREE.PerspectiveCamera, height: number, scale: number, lights: PointLightSpec[], o: { gain: number; tau?: number; distRef?: number; distPow?: number; skyGlint?: [number, number, number]; bokeh?: FrostBokeh; rings?: FrostRing[] }): void {
    const u = this.material.uniforms;
    const n = Math.min(lights.length, 4);
    u.uLightCount.value = n;
    for (let i = 0; i < 4; i++) {
      const l = lights[i];
      if (!l) {
        u.uLightCol.value[i].set(0, 0, 0);
        continue;
      }
      u.uLightPos.value[i].set(l.position.x, l.position.y, l.position.z, l.nearRadius);
      u.uLightCol.value[i].set(...l.color).multiplyScalar(l.intensity);
      u.uLightRange.value[i] = l.range ?? 1e6;
    }
    u.uTau.value = o.tau ?? 0.05;
    u.uGain.value = o.gain;
    u.uPxPerRad.value = height / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    u.uScale.value = scale;
    u.uDistRef.value = o.distRef ?? 10;
    u.uDistPow.value = o.distPow ?? 1;
    u.uSkyCol.value.set(...(o.skyGlint ?? [0, 0, 0]));
    for (let i = 0; i < 2; i++) {
      const r = o.rings?.[i];
      if (r) {
        u.uRingA.value[i].set(r.center.x, r.center.y, r.center.z, r.radius);
        u.uRingB.value[i].set(r.color[0], r.color[1], r.color[2], Math.max(r.width, 1e-3));
      } else u.uRingB.value[i].set(0, 0, 0, 1);
    }
    const b = o.bokeh;
    if (b) {
      // Same thin-lens CoC as the post DOF (post/FilmPost.ts): diameter px = K |z - F| / z.
      const f = b.focalMm / 1000;
      const K = ((f / b.fStop) * f) / Math.max(b.focus - f, 1e-3) / 0.02489 * b.width;
      u.uBokeh.value.set(b.focus, K, b.maxRadius * scale, 1);
      // Front crystals are drawn on the focus plane: they must not be hidden by nearer grass, and they
      // write that depth so the post DOF does not blur them a second time.
      this.material.depthFunc = THREE.AlwaysDepth;
      this.material.depthWrite = true;
    } else {
      u.uBokeh.value.set(1, 0, 0, 0);
      this.material.depthFunc = THREE.LessEqualDepth;
      this.material.depthWrite = false;
    }
    this.points.visible = o.gain > 0 && (n > 0 || !!o.skyGlint || !!o.rings?.length);
  }

  dispose(): void {
    this.points.geometry.dispose();
    this.material.dispose();
  }
}
