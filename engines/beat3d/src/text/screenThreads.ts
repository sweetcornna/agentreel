// ScreenThreadFlow, the screen-space ThreadFlow (canvas pixels, for the screen layer): "字化为光" and
// "光聚成字" on a flat picture (text/threads.ts is the 3D version, with depth of field). Each thread is a
// short glowing filament whose head travels
// along a bent path (quadratic Bezier + curl-noise wander) between a glyph pixel and a point.
//   mode 'dissolve': leaves the glyph at its release frame, arrives at the target `dur` later.
//   mode 'assemble': leaves the target `dur` before its frame, lands on the glyph at its frame.
// Positions are evaluated in the vertex shader as a pure function of the frame: no simulation state.
//
// Filaments are screen-aligned ribbons with a gaussian cross-section (sigma `width`,
// 1080p px, x scale) instead of 1-px GL lines, so they are smooth, silky and identical in look at 4K.
// Energy is normalised to a 1-px line of intensity `gain`, so the calling convention is unchanged:
// pass gain x layout.scale, exactly as before. A thread is born as a hair-thin, faint wisp that
// peels off its stroke and only brightens once it is clear of the glyph (no scratchy starts).

import * as THREE from 'three';
import { NOISE_GLSL } from '../gl/noise.glsl';
import { Rng } from '../lib/rng';

export interface ScreenThreadOrigin {
  x: number;
  y: number;
  /** Release frame (dissolve) or landing frame (assemble). */
  rel: number;
  amp: number;
  /** Optional: this thread's own far end (canvas px) instead of the shared target (e.g. threads that
   *  rise out of a road, or fall back into it). */
  end?: { x: number; y: number };
}

export interface ScreenThreadOptions {
  mode: 'dissolve' | 'assemble';
  /** Polyline segments per thread. */
  segments?: number;
  /** Trail length as a fraction of the path. */
  trail?: number;
  /** Travel time range in frames. */
  durMin: number;
  durMax: number;
  /** Bend of the path, fraction of its length; sag pulls it down-screen (sand) or up (rising). */
  bend?: number;
  sag?: number;
  /** Curl-noise wander amplitude in 1080p px. */
  wander?: number;
  /** If given, travel time scales with the distance to this point (similar speeds, no jagged chords). */
  target?: { x: number; y: number };
  /** Distance (canvas px) that maps to durMax. */
  refLength?: number;
  /** Filament cross-section sigma in 1080p px (default 0.55). */
  width?: number;
  /** Distance (1080p px) over which a thread fades in after leaving its glyph (default 22). */
  emerge?: number;
  /** 0 = arc (default); 1 = pour: fall to the target's level first, then run along it. */
  pour?: number;
}

const VERT = /* glsl */ `
${NOISE_GLSL}
attribute vec2 aStart;
attribute float aRel;
attribute float aDur;
attribute float aSeed;
attribute float aAmp;
attribute float aLag;
attribute float aLagO;   // lag of the other end of this segment (for the tangent)
attribute float aSide;   // -1 / +1 across the ribbon
attribute vec3 aEnd;     // own far end (xy) if z > 0.5, else the shared uTarget
uniform float uFrame;
uniform float uTrail;
uniform float uScale;
uniform float uGain;
uniform float uMode;
uniform float uBend;
uniform float uSag;
uniform float uWander;
uniform float uWidth;
uniform float uEmerge;
uniform float uPour;
uniform vec2 uTarget;
uniform vec3 uColA;
uniform vec3 uColB;
varying vec3 vCol;
varying float vAcross;

vec2 farEnd() { return aEnd.z > 0.5 ? aEnd.xy : uTarget; }

vec2 pathAt(float s) {
  vec2 p0 = aStart;
  vec2 p1 = farEnd();
  vec2 d = p1 - p0;
  float len = max(length(d), 1e-3);
  vec2 perp = vec2(-d.y, d.x) / len;
  float side = aSeed * 2.0 - 1.0;
  // Sand pours: the control point sits below the lower end (screen y grows downwards).
  vec2 ctrl = vec2(mix(p0.x, p1.x, 0.45), max(p0.y, p1.y) + uSag * len) + perp * side * len * uBend;
  // Pour: fall onto the target's level first, then run along it (sand landing on a line).
  vec2 ctrlPour = vec2(mix(p0.x, p1.x, 0.1 + 0.15 * aSeed), p1.y + uSag * len * 0.15);
  ctrl = mix(ctrl, ctrlPour, uPour);
  vec2 b = (1.0 - s) * (1.0 - s) * p0 + 2.0 * (1.0 - s) * s * ctrl + s * s * p1;
  // Coherent flow: the wander is sampled where the thread is, so neighbours drift together.
  vec2 w = curl2(b / (220.0 * uScale) + vec2(0.0, aSeed * 0.35));
  return b + w * uWander * uScale * sin(3.14159265 * s);
}

float easeS(float s) {
  // Dissolve accelerates into the point like sand; assembly decelerates onto the glyph.
  return uMode > 0.5 ? 1.0 - (1.0 - s) * (1.0 - s) : s * s * (3.0 - 2.0 * s) * 0.7 + s * s * 0.3;
}

vec2 posAt(float u, float lag) {
  float s = clamp(u - lag * uTrail, 0.0, 1.0);
  float e = easeS(s);
  return uMode > 0.5 ? pathAt(1.0 - e) : pathAt(e);
}
// Distance (canvas px) a point of the filament has travelled from its glyph along the chord.
float travelled(float u, float lag) {
  float s = clamp(u - lag * uTrail, 0.0, 1.0);
  return easeS(s) * length(farEnd() - aStart);
}

void main() {
  float start = uMode > 0.5 ? aRel - aDur : aRel;
  float u = (uFrame - start) / aDur;
  float uh = u - aLag * uTrail;
  float s = clamp(uh, 0.0, 1.0);
  vec2 p = posAt(u, aLag);
  vec2 q = posAt(u, aLagO);
  vec2 t = q - p;
  float tl = length(t);
  // Degenerate (not yet unrolled) segments take the direction of the path.
  t = tl > 1e-4 ? t / tl : normalize(farEnd() - aStart + vec2(1e-4, 0.0));
  // Keep a consistent winding: the tangent always points from the smaller lag to the larger.
  if (aLagO < aLag) t = -t;
  vec2 n = vec2(-t.y, t.x);
  float sig = uWidth * uScale;
  float hw = 3.0 * sig;
  vec2 pos = p + n * aSide * hw;

  float alive = step(0.0, uh) * step(u, 1.0 + uTrail);
  // Emerge: a thread is born faint and thin at its glyph and brightens as it clears it.
  // Each point of the filament is dim where it is still near its glyph (its own s small), so the
  // root fades into the stroke it peels from and nothing scratches across the remaining text.
  float em = uMode > 0.5 ? 1.0 : smoothstep(1.5 * uScale, uEmerge * uScale, travelled(u, aLag));
  em *= em;
  float tail = pow(1.0 - aLag, 1.5);
  float arrive = uMode > 0.5 ? (1.0 - smoothstep(0.92, 1.0, u)) * smoothstep(0.0, 0.2, u) : 1.0 - smoothstep(0.5, 0.97, s);
  // Energy of a 1-px line of intensity uGain spread over a gaussian of sigma sig: peak = g / (sig sqrt(2 pi)).
  float I = aAmp * uGain * em * tail * arrive * alive / (sig * 2.5066283);
  vCol = mix(uColA, uColB, smoothstep(0.0, 0.65, uMode > 0.5 ? 1.0 - s : s)) * I;
  vAcross = aSide * 3.0;
  gl_Position = projectionMatrix * viewMatrix * vec4(pos, 0.0, 1.0);
}
`;

const FRAG = /* glsl */ `
varying vec3 vCol;
varying float vAcross;
void main() {
  float g = exp(-0.5 * vAcross * vAcross);
  gl_FragColor = vec4(vCol * g, 1.0);
}
`;

export class ScreenThreadFlow {
  /** The filament mesh (add it to a screen scene). Named `lines` for backward compatibility. */
  readonly lines: THREE.Mesh;
  private readonly u: Record<string, THREE.IUniform>;
  private readonly rel: Float32Array;
  private readonly dur: Float32Array;
  private readonly amp: Float32Array;
  private readonly totalAmp: number;

  constructor(
    origins: ScreenThreadOrigin[],
    readonly opts: ScreenThreadOptions,
    seedKey: string,
  ) {
    const rng = new Rng(`${seedKey}/threads`);
    const K = opts.segments ?? 10;
    const n = origins.length;
    const vcount = n * K * 4;
    const aStart = new Float32Array(vcount * 2);
    const aRel = new Float32Array(vcount);
    const aDur = new Float32Array(vcount);
    const aSeed = new Float32Array(vcount);
    const aAmp = new Float32Array(vcount);
    const aLag = new Float32Array(vcount);
    const aLagO = new Float32Array(vcount);
    const aSide = new Float32Array(vcount);
    const aEnd = new Float32Array(vcount * 3);
    const index = new Uint32Array(n * K * 6);
    this.rel = new Float32Array(n);
    this.dur = new Float32Array(n);
    this.amp = new Float32Array(n);
    let total = 0;
    for (let i = 0; i < n; i++) {
      const o = origins[i];
      const far = o.end ?? opts.target;
      const len = Math.hypot(o.x - (far?.x ?? o.x), o.y - (far?.y ?? o.y));
      const lenT = far ? Math.min(len / (opts.refLength ?? 240), 1) : rng.next();
      const dur = opts.durMin + (opts.durMax - opts.durMin) * (0.75 * lenT + 0.25 * rng.next());
      const seed = rng.next();
      this.rel[i] = o.rel;
      this.dur[i] = dur;
      this.amp[i] = o.amp;
      total += o.amp;
      for (let k = 0; k < K; k++) {
        const seg = i * K + k;
        // 4 vertices: (end a, -), (end a, +), (end b, -), (end b, +)
        for (let e = 0; e < 2; e++) {
          for (let sd = 0; sd < 2; sd++) {
            const v = seg * 4 + e * 2 + sd;
            aStart[v * 2] = o.x;
            aStart[v * 2 + 1] = o.y;
            aRel[v] = o.rel;
            aDur[v] = dur;
            aSeed[v] = seed;
            aAmp[v] = o.amp;
            aLag[v] = (k + e) / K;
            aLagO[v] = (k + 1 - e) / K;
            aSide[v] = sd === 0 ? -1 : 1;
            if (o.end) aEnd.set([o.end.x, o.end.y, 1], v * 3);
          }
        }
        const b = seg * 4;
        index.set([b, b + 1, b + 2, b + 1, b + 3, b + 2], seg * 6);
      }
    }
    this.totalAmp = total;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vcount * 3), 3));
    geo.setAttribute('aStart', new THREE.BufferAttribute(aStart, 2));
    geo.setAttribute('aRel', new THREE.BufferAttribute(aRel, 1));
    geo.setAttribute('aDur', new THREE.BufferAttribute(aDur, 1));
    geo.setAttribute('aSeed', new THREE.BufferAttribute(aSeed, 1));
    geo.setAttribute('aAmp', new THREE.BufferAttribute(aAmp, 1));
    geo.setAttribute('aLag', new THREE.BufferAttribute(aLag, 1));
    geo.setAttribute('aLagO', new THREE.BufferAttribute(aLagO, 1));
    geo.setAttribute('aSide', new THREE.BufferAttribute(aSide, 1));
    geo.setAttribute('aEnd', new THREE.BufferAttribute(aEnd, 3));
    geo.setIndex(new THREE.BufferAttribute(index, 1));
    this.u = {
      uFrame: { value: 0 },
      uTrail: { value: opts.trail ?? 0.2 },
      uScale: { value: 1 },
      uGain: { value: 1 },
      uMode: { value: opts.mode === 'assemble' ? 1 : 0 },
      uBend: { value: opts.bend ?? 0.2 },
      uSag: { value: opts.sag ?? 0.1 },
      uWander: { value: opts.wander ?? 12 },
      uWidth: { value: opts.width ?? 0.55 },
      uEmerge: { value: opts.emerge ?? 22 },
      uPour: { value: opts.pour ?? 0 },
      uTarget: { value: new THREE.Vector2() },
      uColA: { value: new THREE.Vector3(1, 1, 1) },
      uColB: { value: new THREE.Vector3(1, 0.6, 0.2) },
    };
    const mat = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: this.u,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthTest: false,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    this.lines = new THREE.Mesh(geo, mat);
    this.lines.frustumCulled = false;
  }

  /**
   * gain: integrated intensity of a filament, as for a 1-px line. Multiply by layout.scale (as
   * before) so the same apparent weight holds at 4K.
   */
  update(frame: number, target: { x: number; y: number }, scale: number, gain: number, colA: [number, number, number], colB: [number, number, number]): void {
    this.u.uFrame.value = frame;
    this.u.uTarget.value.set(target.x, target.y);
    this.u.uScale.value = scale;
    this.u.uGain.value = gain;
    this.u.uColA.value.set(...colA);
    this.u.uColB.value.set(...colB);
    this.lines.visible = gain > 0;
  }

  /** Fraction (0..1, amplitude-weighted) of threads that have arrived by `frame`. */
  arrived(frame: number): number {
    let a = 0;
    const assemble = this.opts.mode === 'assemble';
    for (let i = 0; i < this.rel.length; i++) {
      const end = assemble ? this.rel[i] : this.rel[i] + this.dur[i];
      const t = Math.min(Math.max((frame - (end - 4)) / 6, 0), 1);
      a += this.amp[i] * t;
    }
    return this.totalAmp > 0 ? a / this.totalAmp : 0;
  }

  /** Last arrival frame. */
  get endFrame(): number {
    let m = -Infinity;
    for (let i = 0; i < this.rel.length; i++) m = Math.max(m, this.opts.mode === 'assemble' ? this.rel[i] : this.rel[i] + this.dur[i]);
    return m;
  }

  dispose(): void {
    this.lines.geometry.dispose();
    (this.lines.material as THREE.Material).dispose();
  }
}
