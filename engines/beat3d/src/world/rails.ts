// Light running along straight rails (a corridor's chords, a string): each pulse is a short bright slug
// with a tail, drawn as a camera-facing ribbon along the rail whose width is the slug's image plus its
// circle of confusion, at constant energy (out of focus = wide and faint). The pulse is evaluated per
// pixel in metres along the rail, so it stays exact however the ribbon is tessellated; only the stretch
// of each rail around its pulses is drawn. It dims with distance through the haze. Drive the pulses
// from the score (CLOCK.eventFrames) with a beat-grid fallback, so the picture plays what is heard.

import * as THREE from 'three';
import type { StageContext } from '../core/types';
import { cocK, type LensState } from '../lib/lens';
import { emissive } from './stage';
import type { RGB } from '../lib/color';

export const MAX_PULSES = 16;
const SEGMENTS = 64;

export interface RailSpec {
  /** Start and end of the line (world). */
  a: THREE.Vector3;
  b: THREE.Vector3;
  /** Tube radius (m). */
  radius: number;
}

export interface RailPulse {
  rail: number;
  /** Head position along the line (m from `a`) and its length (m, head to end of tail). */
  head: number;
  length: number;
  /** Linear colour times intensity of the head. */
  color: RGB;
}

export interface RailLook {
  lens: LensState;
  pulses: RailPulse[];
  /** Haze extinction (1/m). */
  extinction: number;
}

const VERT = /* glsl */ `
attribute vec3 iA;
attribute vec3 iB;
attribute vec4 iRange;    // near, tip (m along the line), radius (m), rail index
uniform vec2 uRes;
uniform float uScale;
uniform float uCocK;
uniform float uFocus;
varying float vM;         // metres along the line
varying float vAcross;    // -1..1 across the ribbon
varying float vK;         // energy factor: tube width / drawn width
varying float vRail;
varying float vDist;      // distance to the lens
void main() {
  vec3 d = iB - iA;
  float len = length(d);
  vec3 t = d / len;
  float m = mix(iRange.x, iRange.y, position.x);
  vec3 p = iA + t * m;
  vec4 c0 = modelViewMatrix * vec4(p, 1.0);
  vec4 c1 = modelViewMatrix * vec4(p + t * 0.05, 1.0);
  float z = max(-c0.z, 0.02);
  // Screen direction of the line here, and its perpendicular.
  vec4 q0 = projectionMatrix * c0;
  vec4 q1 = projectionMatrix * c1;
  vec2 s0 = q0.xy / max(q0.w, 1e-4) * 0.5 * uRes;
  vec2 s1 = q1.xy / max(q1.w, 1e-4) * 0.5 * uRes;
  vec2 dir = s1 - s0;
  dir = length(dir) > 1e-6 ? normalize(dir) : vec2(1.0, 0.0);
  vec2 nrm = vec2(-dir.y, dir.x);
  float ppm = 0.5 * uRes.y * projectionMatrix[1][1] / z;
  float tube = iRange.z * ppm;
  float coc = 0.5 * uCocK * abs(z - uFocus) / z;
  float R = max(sqrt(tube * tube + coc * coc), 0.7 * uScale);
  vK = tube / R;
  float w = R * 1.6 + 1.0 * uScale;
  q0.xy += nrm * position.y * w / (0.5 * uRes) * q0.w;
  gl_Position = q0;
  vAcross = position.y * w / R;
  vM = m;
  vRail = iRange.w;
  vDist = length(c0.xyz);
}
`;

const FRAG = /* glsl */ `
#define MAX_PULSES ${MAX_PULSES}
uniform int uNP;
uniform vec4 uP[MAX_PULSES];     // rail, head (m), length (m), unused
uniform vec3 uPC[MAX_PULSES];
uniform float uExt;
varying float vM;
varying float vAcross;
varying float vK;
varying float vRail;
varying float vDist;
void main() {
  // Cross profile of a defocused tube: a soft-shouldered band over [-1, 1].
  float x = abs(vAcross);
  float prof = clamp(1.0 - x * x, 0.0, 1.0);
  prof = prof * sqrt(prof) * 1.6977;   // (1 - x^2)^1.5 integrates to 3pi/8: mean 1 over the band
  vec3 c = vec3(0.0);
  for (int i = 0; i < MAX_PULSES; i++) {
    if (i >= uNP) break;
    vec4 p = uP[i];
    if (abs(p.x - vRail) > 0.5) continue;
    float b = p.y - vM;              // metres behind the head
    if (b < -0.12 || b > p.z) continue;
    float head = exp(-b * b / 0.0045);
    float tail = b > 0.0 ? exp(-b / (0.28 * p.z)) * (1.0 - b / p.z) : 0.0;
    c += uPC[i] * (head + 0.35 * tail);
  }
  gl_FragColor = vec4(c * prof * vK * exp(-uExt * vDist), 0.0);
}
`;

export class Rails {
  readonly mesh: THREE.Mesh;
  private readonly geo: THREE.InstancedBufferGeometry;
  private readonly range: THREE.InstancedBufferAttribute;
  private readonly mat: THREE.ShaderMaterial;

  constructor(
    private readonly ctx: StageContext,
    readonly rails: RailSpec[],
  ) {
    const n = rails.length;
    const pos: number[] = [];
    const idx: number[] = [];
    for (let i = 0; i <= SEGMENTS; i++) {
      const u = i / SEGMENTS;
      pos.push(u, -1, 0, u, 1, 0);
      if (i < SEGMENTS) {
        const k = i * 2;
        idx.push(k, k + 1, k + 2, k + 1, k + 3, k + 2);
      }
    }
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
    g.setIndex(idx);
    g.setAttribute('iA', new THREE.InstancedBufferAttribute(new Float32Array(rails.flatMap((r) => [r.a.x, r.a.y, r.a.z])), 3));
    g.setAttribute('iB', new THREE.InstancedBufferAttribute(new Float32Array(rails.flatMap((r) => [r.b.x, r.b.y, r.b.z])), 3));
    this.range = new THREE.InstancedBufferAttribute(new Float32Array(n * 4), 4);
    this.range.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('iRange', this.range);
    g.instanceCount = n;
    this.geo = g;
    this.mat = emissive(
      new THREE.ShaderMaterial({
        vertexShader: VERT,
        fragmentShader: FRAG,
        uniforms: {
          uRes: { value: new THREE.Vector2(ctx.width, ctx.height) },
          uScale: { value: ctx.scale },
          uCocK: { value: 0 },
          uFocus: { value: 10 },
          uNP: { value: 0 },
          uP: { value: Array.from({ length: MAX_PULSES }, () => new THREE.Vector4()) },
          uPC: { value: Array.from({ length: MAX_PULSES }, () => new THREE.Vector3()) },
          uExt: { value: 0 },
        },
        depthTest: false,
        side: THREE.DoubleSide,
        toneMapped: false,
      }),
    );
    this.mesh = new THREE.Mesh(g, this.mat);
    this.mesh.frustumCulled = false;
  }

  /** Length of rail i (m). */
  length(i: number): number {
    return this.rails[i].a.distanceTo(this.rails[i].b);
  }

  /** World point `m` metres along rail i. */
  at(i: number, m: number): THREE.Vector3 {
    const r = this.rails[i];
    return r.a.clone().lerp(r.b, m / this.length(i));
  }

  update(look: RailLook): void {
    const u = this.mat.uniforms;
    u.uCocK.value = cocK(look.lens, this.ctx.width);
    u.uFocus.value = look.lens.focus;
    u.uExt.value = look.extinction;
    // Draw each chord only around its pulses (an empty stretch where it has none).
    const r = this.range.array as Float32Array;
    this.rails.forEach((rail, i) => {
      const len = this.length(i);
      const mine = look.pulses.filter((p) => p.rail === i);
      const near = mine.length ? Math.max(0, Math.min(...mine.map((p) => p.head - p.length - 0.2))) : 0;
      const tip = mine.length ? Math.min(len, Math.max(...mine.map((p) => p.head + 0.2))) : 0;
      r.set([near, Math.max(near, tip), rail.radius, i], i * 4);
    });
    this.range.needsUpdate = true;
    const n = Math.min(look.pulses.length, MAX_PULSES);
    u.uNP.value = n;
    for (let i = 0; i < n; i++) {
      const p = look.pulses[i];
      u.uP.value[i].set(p.rail, p.head, p.length, 0);
      u.uPC.value[i].set(...p.color);
    }
  }

  dispose(): void {
    this.geo.dispose();
    this.mat.dispose();
  }
}
