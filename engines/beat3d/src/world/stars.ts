// Bright-star catalog (to about magnitude 7), generated from a seed: realistic number counts
// (about x2.9 per magnitude), mild concentration to the galactic plane, B-V colours, extinction and
// reddening near the horizon. Drawn as gaussian point sprites whose flux is resolution-independent.

import * as THREE from 'three';
import { Rng } from '../lib/rng';
import { galaxyDensity, type GalaxyFrame } from './sky';

export interface StarCatalog {
  positions: Float32Array; // unit vectors
  mags: Float32Array;
  colors: Float32Array; // linear, luminance-normalised
  count: number;
}

const lobe = (l: number, mu: number, s1: number, s2: number): number => {
  const t = (l - mu) / (l < mu ? s1 : s2);
  return Math.exp(-0.5 * t * t);
};

/** Blackbody colour in linear sRGB, unit luminance (Planck x CIE 1931 multi-lobe fit, Wyman 2013). */
export function blackbodyRGB(T: number): [number, number, number] {
  let X = 0;
  let Y = 0;
  let Z = 0;
  for (let l = 380; l <= 780; l += 5) {
    const B = 1 / (Math.pow(l, 5) * (Math.exp(1.4388e7 / (l * T)) - 1));
    X += B * (1.056 * lobe(l, 599.8, 37.9, 31.0) + 0.362 * lobe(l, 442.0, 16.0, 26.7) - 0.065 * lobe(l, 501.1, 20.4, 26.2));
    Y += B * (0.821 * lobe(l, 568.8, 46.9, 40.5) + 0.286 * lobe(l, 530.9, 16.3, 31.1));
    Z += B * (1.217 * lobe(l, 437.0, 11.8, 36.0) + 0.681 * lobe(l, 459.0, 26.0, 13.8));
  }
  const r = Math.max(3.2406 * X - 1.5372 * Y - 0.4986 * Z, 0);
  const g = Math.max(-0.9689 * X + 1.8758 * Y + 0.0415 * Z, 0);
  const b = Math.max(0.0557 * X - 0.204 * Y + 1.057 * Z, 0);
  const L = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return [r / L, g / L, b / L];
}

/** Star colour from B-V (Ballesteros 2012 temperature), with 55 % of the chroma kept. */
export function starColorBV(bv: number): [number, number, number] {
  const T = 4600 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62));
  const [r, g, b] = blackbodyRGB(T);
  const k = 0.55;
  return [1 + k * (r - 1), 1 + k * (g - 1), 1 + k * (b - 1)];
}

function sampleBV(rng: Rng): number {
  const u = rng.next();
  if (u < 0.1) return rng.range(-0.25, 0.05);
  if (u < 0.3) return rng.range(0.05, 0.45);
  if (u < 0.75) return rng.normal(0.85, 0.22);
  if (u < 0.93) return rng.normal(1.2, 0.15);
  return rng.range(1.45, 1.85);
}

export function makeCatalog(seed: string, galaxy: GalaxyFrame, count = 15000, magMax = 7.0): StarCatalog {
  const rng = new Rng(seed);
  const positions = new Float32Array(count * 3);
  const mags = new Float32Array(count);
  const colors = new Float32Array(count * 3);
  const v = new THREE.Vector3();
  const slope = 0.46; // log10 N(<m) grows by ~0.46 per magnitude
  for (let i = 0; i < count; i++) {
    // Mild galactic concentration for this bright population.
    for (;;) {
      const [x, y, z] = rng.unitVector();
      v.set(x, y, z);
      const accept = 0.45 + 0.55 * Math.min(1, galaxyDensity(v, galaxy));
      if (rng.next() < accept) break;
    }
    positions.set([v.x, v.y, v.z], i * 3);
    const m = magMax + Math.log10(Math.max(rng.next(), 1e-9)) / slope;
    mags[i] = Math.max(m, -1.4);
    colors.set(starColorBV(sampleBV(rng)), i * 3);
  }
  return { positions, mags, colors, count };
}

const VERT = /* glsl */ `
attribute float aMag;
attribute vec3 aColor;
uniform float uGain;
uniform float uMagLimit;
uniform float uScale;
uniform float uExtinction;
uniform float uRadius;
uniform float uCull;     // 1 = hide stars below the world horizon (y < 0), 0 = show the whole sphere
uniform vec2 uBokeh;     // disc radius (canvas px, 0 = off), focus distance (m)
varying vec3 vCol;
varying float vSig2;
varying float vHex;
void main() {
  vec3 d = normalize(position);
  float alt = asin(clamp(d.y, -1.0, 1.0));
  float altDeg = max(degrees(alt), 0.0);
  float X = 1.0 / (sin(max(alt, 0.0)) + 0.50572 * pow(altDeg + 6.07995, -1.6364));
  vec3 ext = pow(vec3(10.0), -0.4 * uExtinction * X * vec3(1.35, 1.0, 0.72));
  float vis = clamp((uMagLimit - aMag) / 1.5, 0.0, 1.0) * mix(1.0, step(-0.002, d.y), uCull);
  float flux = pow(10.0, -0.4 * (aMag - 6.0)) * vis * uGain;
  // PSF: slightly wider for bright stars (optics), defined in 1080p pixels plus pixel footprint.
  float s1080 = 0.55 + 0.35 * clamp((3.5 - aMag) / 5.0, 0.0, 1.0);
  float sEff2 = s1080 * s1080 + 1.0 / 12.0;
  vSig2 = sEff2 * uScale * uScale;
  vCol = aColor * ext * flux / (6.2831853 * sEff2);
  // Place on a sphere around the camera so translation never moves the sky.
  vec4 p = projectionMatrix * vec4(mat3(viewMatrix) * d * uRadius, 1.0);
  vHex = 0.0;
  if (uBokeh.x > 0.0) {
    // Defocused star: an aperture-shaped disc, drawn on the focus plane so the post DOF keeps it.
    vHex = uBokeh.x;
    vCol = aColor * ext * flux / (3.14159265 * uBokeh.x * uBokeh.x / (uScale * uScale));
    p = projectionMatrix * vec4(mat3(viewMatrix) * d * uBokeh.y, 1.0);
  }
  gl_Position = p;
  gl_PointSize = flux > 1e-6 ? (vHex > 0.0 ? 2.0 * vHex + 4.0 : 2.0 * ceil(3.2 * sqrt(vSig2)) + 2.0) : 0.0;
}
`;

const FRAG = /* glsl */ `
varying vec3 vCol;
varying float vSig2;
varying float vHex;
float hexDisc(vec2 p, float r) {
  vec2 a = abs(p);
  float h = max(a.x * 0.8660254 + a.y * 0.5, a.y);
  float e = h - 0.8660254 * r;
  return (1.0 - smoothstep(-0.8, 0.8, e)) * (1.0 + 0.3 * smoothstep(-0.2 * r, 0.0, e));
}
void main() {
  if (vHex > 0.0) {
    vec2 q = (gl_PointCoord - 0.5) * (2.0 * vHex + 4.0);
    vec3 m = vec3(hexDisc(q, vHex * 1.025), hexDisc(q, vHex), hexDisc(q, vHex * 0.975));
    if (m.r < 0.02) discard;
    gl_FragColor = vec4(vCol * m, 1.0);
    return;
  }
  vec2 d = (gl_PointCoord - 0.5) * (2.0 * ceil(3.2 * sqrt(vSig2)) + 2.0);
  float g = exp(-0.5 * dot(d, d) / vSig2);
  gl_FragColor = vec4(vCol * g, 1.0);
}
`;

export class StarPoints {
  readonly points: THREE.Points;
  readonly material: THREE.ShaderMaterial;
  /** opts.horizonCull (default true): hide stars below the world horizon; false for orbital shots. */
  constructor(cat: StarCatalog, opts: { horizonCull?: boolean } = {}) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(cat.positions, 3));
    geo.setAttribute('aMag', new THREE.BufferAttribute(cat.mags, 1));
    geo.setAttribute('aColor', new THREE.BufferAttribute(cat.colors, 3));
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: {
        uGain: { value: 0 },
        uMagLimit: { value: 12 },
        uScale: { value: 1 },
        uExtinction: { value: 0.25 },
        uRadius: { value: 1000 },
        uCull: { value: opts.horizonCull === false ? 0 : 1 },
        uBokeh: { value: new THREE.Vector2(0, 1) },
      },
      blending: THREE.AdditiveBlending,
      depthTest: false,
      depthWrite: false,
      transparent: true,
      toneMapped: false,
    });
    this.points = new THREE.Points(geo, this.material);
    this.points.frustumCulled = false;
    this.points.renderOrder = -90;
  }
  /**
   * Optional (default off): draw every star as a defocused, aperture-shaped disc of `radius` (1080p
   * px; for a shot focused at `focus` m, stars at infinity have CoC radius ~ min(K/2, dof.maxRadius)).
   * The discs are written at the focus depth so the post DOF does not blur them again. null = off.
   */
  setBokeh(b: { radius: number; focus: number; scale: number } | null): void {
    const u = this.material.uniforms;
    if (b) {
      u.uBokeh.value.set(b.radius * b.scale, b.focus);
      this.material.depthTest = true;
      this.material.depthFunc = THREE.AlwaysDepth;
      this.material.depthWrite = true;
    } else {
      u.uBokeh.value.set(0, 1);
      this.material.depthTest = false;
      this.material.depthFunc = THREE.LessEqualDepth;
      this.material.depthWrite = false;
    }
  }

  /** Toggle the below-horizon cull after construction. */
  set horizonCull(on: boolean) {
    this.material.uniforms.uCull.value = on ? 1 : 0;
  }

  update(gain: number, magLimit: number, scale: number, extinction: number, radius: number): void {
    const u = this.material.uniforms;
    u.uGain.value = gain;
    u.uMagLimit.value = magLimit;
    u.uScale.value = scale;
    u.uExtinction.value = extinction;
    u.uRadius.value = radius;
  }
  dispose(): void {
    this.points.geometry.dispose();
    this.material.dispose();
  }
}
