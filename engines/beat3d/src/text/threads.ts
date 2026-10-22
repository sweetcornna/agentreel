// ThreadFlow: "光聚成字" (assemble) and "字化为光" (dissolve), with paths in 3D (text/screenThreads.ts
// is the screen-space original it grew from).
// Each thread is a short glowing filament whose head travels along a bent path (quadratic Bezier +
// curl-noise wander) between a glyph point and a far end:
//   mode 'assemble': leaves its far end `dur` frames before its frame, lands on the glyph at its frame.
//   mode 'dissolve': leaves the glyph at its frame, reaches its far end `dur` later.
// Positions are evaluated in the vertex shader as a pure function of the frame: no simulation state.
//
// Filaments are screen-aligned ribbons with a gaussian cross-section (sigma `width`, 1080p px), widened
// by their circle of confusion at their depth with the energy kept (out of focus = wide and faint), so
// a thread that streams past the lens is a soft wisp, not a hard line. Energy is normalised to a 1-px
// line of intensity `gain`. Light dims with distance through the haze (extinction) and fades out the
// last few tens of centimetres before the lens (nothing is drawn behind it).

import * as THREE from 'three';
import { NOISE_GLSL } from '../gl/noise.glsl';
import { cocK, type LensState } from '../lib/lens';
import { Rng } from '../lib/rng';

export interface ThreadOrigin {
  /** The glyph point (world). */
  p: THREE.Vector3;
  /** The far end (world): where an assembling thread comes from, where a dissolving one goes. */
  end: THREE.Vector3;
  /** Landing frame (assemble) or release frame (dissolve). */
  rel: number;
  /** Travel time (frames). */
  dur: number;
  amp: number;
}

export interface ThreadOptions {
  mode: 'dissolve' | 'assemble';
  /** Polyline segments per thread. */
  segments?: number;
  /** Trail length as a fraction of the path. */
  trail?: number;
  /** Bend of the path, fraction of its length, to a side picked per thread. */
  bend?: number;
  /** Curl-noise wander amplitude (m) and its scale (m). */
  wander?: number;
  wanderScale?: number;
  /** Filament cross-section sigma in 1080p px (default 0.5). */
  width?: number;
}

export interface ThreadLook {
  lens: LensState;
  /** Intensity of a filament, as a 1-px line. */
  gain: number;
  /** Colour at the far end and at the glyph (linear). */
  colFar: [number, number, number];
  colNear: [number, number, number];
  /** Haze extinction (1/m). */
  extinction: number;
  /** 离镜头这段距离（m）内淡掉：从 .y 起变暗，到 .x 没了（默认 [0.12, 0.6]）。镜头从光丝中间穿过时，
   *  贴着镜头的那一段弥散圆极大、跑得极快，留着只会是一团糊的烟。 */
  near?: [number, number];
}

const VERT = /* glsl */ `
${NOISE_GLSL}
attribute vec3 aStart;   // glyph point
attribute vec3 aEnd;     // far end
attribute float aRel;
attribute float aDur;
attribute float aSeed;
attribute float aAmp;
attribute float aLag;
attribute float aLagO;   // lag of the other end of this segment (for the tangent)
attribute float aSide;   // -1 / +1 across the ribbon
uniform float uFrame;
uniform float uTrail;
uniform float uScale;
uniform float uGain;
uniform float uMode;
uniform float uBend;
uniform float uWander;
uniform float uWanderScale;
uniform float uWidth;
uniform vec2 uRes;
uniform float uCocK;
uniform float uFocus;
uniform float uExt;
uniform vec3 uColA;
uniform vec3 uColB;
uniform vec2 uNear;
varying vec3 vCol;
varying float vAcross;

vec3 pathAt(float s) {
  vec3 p0 = aStart;
  vec3 p1 = aEnd;
  vec3 d = p1 - p0;
  float len = max(length(d), 1e-4);
  vec3 dn = d / len;
  vec3 e1 = normalize(cross(dn, abs(dn.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
  vec3 e2 = cross(dn, e1);
  float a = aSeed * 6.2831853;
  vec3 ctrl = mix(p0, p1, 0.45) + (cos(a) * e1 + sin(a) * e2) * len * uBend;
  vec3 b = (1.0 - s) * (1.0 - s) * p0 + 2.0 * (1.0 - s) * s * ctrl + s * s * p1;
  // Coherent flow: the wander is sampled where the thread is, so neighbours drift together.
  vec3 w = curl3(b / uWanderScale + vec3(aSeed * 0.35, 0.0, 0.0));
  return b + w * uWander * sin(3.14159265 * s);
}

float easeS(float s) {
  // Dissolve accelerates away like sand; assembly decelerates onto the glyph.
  return uMode > 0.5 ? 1.0 - (1.0 - s) * (1.0 - s) : s * s * (3.0 - 2.0 * s) * 0.7 + s * s * 0.3;
}

vec3 posAt(float u, float lag) {
  float e = easeS(clamp(u - lag * uTrail, 0.0, 1.0));
  return uMode > 0.5 ? pathAt(1.0 - e) : pathAt(e);
}

void main() {
  float start = uMode > 0.5 ? aRel - aDur : aRel;
  float u = (uFrame - start) / aDur;
  float uh = u - aLag * uTrail;
  vec3 p = posAt(u, aLag);
  vec3 q = posAt(u, aLagO);
  vec4 cp = projectionMatrix * viewMatrix * vec4(p, 1.0);
  vec4 cq = projectionMatrix * viewMatrix * vec4(q, 1.0);
  float alive = step(0.0, uh) * step(u, 1.0 + uTrail);
  // Both ends must be in front of the lens, or the segment is not drawn at all.
  if (alive <= 0.0 || cp.w < 0.05 || cq.w < 0.05) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vCol = vec3(0.0); vAcross = 0.0; return; }
  vec2 sp = cp.xy / cp.w * uRes;
  vec2 sq = cq.xy / cq.w * uRes;
  vec2 t = sq - sp;
  float tl = length(t);
  t = tl > 1e-4 ? t / tl : vec2(1.0, 0.0);
  // Keep a consistent winding: the tangent always points from the smaller lag to the larger.
  if (aLagO < aLag) t = -t;
  vec2 n = vec2(-t.y, t.x);
  float z = cp.w;
  float coc = 0.5 * uCocK * abs(z - uFocus) / z;
  float sig = sqrt(uWidth * uWidth * uScale * uScale + 0.25 * coc * coc);
  cp.xy += n * aSide * 3.0 * sig / (0.5 * uRes) * cp.w;
  gl_Position = cp;
  float tail = pow(1.0 - aLag, 1.5);
  float arrive = uMode > 0.5 ? (1.0 - smoothstep(0.92, 1.0, u)) * smoothstep(0.0, 0.2, u) : 1.0 - smoothstep(0.5, 0.97, clamp(uh, 0.0, 1.0));
  float nearFade = smoothstep(uNear.x, uNear.y, z);
  // Energy of a 1-px line of intensity uGain spread over a gaussian of sigma sig: peak = g / (sig sqrt(2 pi)).
  float I = aAmp * uGain * tail * arrive * nearFade * exp(-uExt * length(p - cameraPosition)) / (sig / uScale * 2.5066283);
  float s = clamp(uh, 0.0, 1.0);
  vCol = mix(uColA, uColB, smoothstep(0.0, 0.65, uMode > 0.5 ? s : 1.0 - s)) * I;
  vAcross = aSide * 3.0;
}
`;

const FRAG = /* glsl */ `
varying vec3 vCol;
varying float vAcross;
void main() {
  float g = exp(-0.5 * vAcross * vAcross);
  gl_FragColor = vec4(vCol * g, 0.0);
}
`;

export class ThreadFlow {
  readonly mesh: THREE.Mesh;
  private readonly u: Record<string, THREE.IUniform>;

  constructor(
    private readonly ctxSize: { width: number; height: number; scale: number },
    origins: ThreadOrigin[],
    readonly opts: ThreadOptions,
    seedKey: string,
  ) {
    const rng = new Rng(`${seedKey}/threads`);
    const K = opts.segments ?? 16;
    const n = origins.length;
    const vcount = n * K * 4;
    const aStart = new Float32Array(vcount * 3);
    const aEnd = new Float32Array(vcount * 3);
    const aRel = new Float32Array(vcount);
    const aDur = new Float32Array(vcount);
    const aSeed = new Float32Array(vcount);
    const aAmp = new Float32Array(vcount);
    const aLag = new Float32Array(vcount);
    const aLagO = new Float32Array(vcount);
    const aSide = new Float32Array(vcount);
    const index = new Uint32Array(n * K * 6);
    for (let i = 0; i < n; i++) {
      const o = origins[i];
      const seed = rng.next();
      for (let k = 0; k < K; k++) {
        const seg = i * K + k;
        // 4 vertices: (end a, -), (end a, +), (end b, -), (end b, +)
        for (let e = 0; e < 2; e++) {
          for (let sd = 0; sd < 2; sd++) {
            const v = seg * 4 + e * 2 + sd;
            aStart.set([o.p.x, o.p.y, o.p.z], v * 3);
            aEnd.set([o.end.x, o.end.y, o.end.z], v * 3);
            aRel[v] = o.rel;
            aDur[v] = o.dur;
            aSeed[v] = seed;
            aAmp[v] = o.amp;
            aLag[v] = (k + e) / K;
            aLagO[v] = (k + 1 - e) / K;
            aSide[v] = sd === 0 ? -1 : 1;
          }
        }
        const b = seg * 4;
        index.set([b, b + 1, b + 2, b + 1, b + 3, b + 2], seg * 6);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vcount * 3), 3));
    geo.setAttribute('aStart', new THREE.BufferAttribute(aStart, 3));
    geo.setAttribute('aEnd', new THREE.BufferAttribute(aEnd, 3));
    geo.setAttribute('aRel', new THREE.BufferAttribute(aRel, 1));
    geo.setAttribute('aDur', new THREE.BufferAttribute(aDur, 1));
    geo.setAttribute('aSeed', new THREE.BufferAttribute(aSeed, 1));
    geo.setAttribute('aAmp', new THREE.BufferAttribute(aAmp, 1));
    geo.setAttribute('aLag', new THREE.BufferAttribute(aLag, 1));
    geo.setAttribute('aLagO', new THREE.BufferAttribute(aLagO, 1));
    geo.setAttribute('aSide', new THREE.BufferAttribute(aSide, 1));
    geo.setIndex(new THREE.BufferAttribute(index, 1));
    this.u = {
      uFrame: { value: 0 },
      uTrail: { value: opts.trail ?? 0.2 },
      uScale: { value: ctxSize.scale },
      uGain: { value: 1 },
      uMode: { value: opts.mode === 'assemble' ? 1 : 0 },
      uBend: { value: opts.bend ?? 0.15 },
      uWander: { value: opts.wander ?? 0.05 },
      uWanderScale: { value: opts.wanderScale ?? 1.2 },
      uWidth: { value: opts.width ?? 0.5 },
      uRes: { value: new THREE.Vector2(ctxSize.width, ctxSize.height) },
      uCocK: { value: 0 },
      uFocus: { value: 10 },
      uExt: { value: 0 },
      uColA: { value: new THREE.Vector3(1, 0.6, 0.2) },
      uColB: { value: new THREE.Vector3(1, 1, 1) },
      uNear: { value: new THREE.Vector2(0.12, 0.6) },
    };
    const mat = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: this.u,
      transparent: true,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneFactor,
      blendSrcAlpha: THREE.ZeroFactor,
      blendDstAlpha: THREE.OneFactor,
      depthTest: false,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.frustumCulled = false;
  }

  update(frame: number, look: ThreadLook): void {
    const u = this.u;
    u.uFrame.value = frame;
    u.uGain.value = look.gain;
    u.uCocK.value = cocK(look.lens, this.ctxSize.width);
    u.uFocus.value = look.lens.focus;
    u.uExt.value = look.extinction;
    u.uColA.value.set(...look.colFar);
    u.uColB.value.set(...look.colNear);
    u.uNear.value.set(...(look.near ?? [0.12, 0.6]));
    this.mesh.visible = look.gain > 0;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
  }
}
