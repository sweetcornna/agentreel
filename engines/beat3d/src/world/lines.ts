// Lines of light with a shape that changes every instant: strings (taut, sagging, humming, plucked into
// standing waves), a track lit along its length, a pulse running down it. Each line is a
// polyline whose vertices (position, colour times intensity) are set on the CPU per call, a pure
// function of the frame, and uploaded as a float texture; the GPU draws every segment as a
// camera-facing ribbon with a gaussian cross-section (sigma in 1080p px), widened by its circle of
// confusion at constant energy (out of focus = wide and faint, as ThreadFlow does), clipped at the
// lens, dimmed by the haze. Intensities are those of a 1-px line. Additive, no depth.

import * as THREE from 'three';
import type { StageContext } from '../core/types';
import { cocK, type LensState } from '../lib/lens';
import { emissive } from './stage';

const VERT = /* glsl */ `
precision highp sampler2D;
uniform sampler2D tPos;   // xyz, w unused
uniform sampler2D tCol;   // rgb intensity
attribute vec3 aSeg;      // line, vertex index of the segment start, end (0 / 1)
attribute float aSide;    // -1 / +1
uniform vec2 uRes;
uniform float uScale;
uniform float uCocK;
uniform float uFocus;
uniform float uWidth;
uniform float uExt;
uniform float uNear;
varying vec3 vCol;
varying float vAcross;
void main() {
  ivec2 ia = ivec2(int(aSeg.y), int(aSeg.x));
  ivec2 ib = ivec2(int(aSeg.y) + 1, int(aSeg.x));
  vec3 A = texelFetch(tPos, ia, 0).xyz;
  vec3 B = texelFetch(tPos, ib, 0).xyz;
  vec3 ca = texelFetch(tCol, ia, 0).rgb;
  vec3 cb = texelFetch(tCol, ib, 0).rgb;
  vec4 va = viewMatrix * vec4(A, 1.0);
  vec4 vb = viewMatrix * vec4(B, 1.0);
  // Clip the segment at the near plane, in view space.
  float za = -va.z;
  float zb = -vb.z;
  if ((za < uNear && zb < uNear) || max(max(ca.r, ca.g), ca.b) + max(max(cb.r, cb.g), cb.b) <= 0.0) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vCol = vec3(0.0); vAcross = 0.0; return;
  }
  if (za < uNear) { float k = (uNear - za) / (zb - za); va = mix(va, vb, k); ca = mix(ca, cb, k); }
  if (zb < uNear) { float k = (uNear - zb) / (za - zb); vb = mix(vb, va, k); cb = mix(cb, ca, k); }
  vec4 pa = projectionMatrix * va;
  vec4 pb = projectionMatrix * vb;
  vec2 sa = pa.xy / pa.w * uRes;
  vec2 sb = pb.xy / pb.w * uRes;
  vec2 t = sb - sa;
  float tl = length(t);
  t = tl > 1e-4 ? t / tl : vec2(1.0, 0.0);
  vec2 n = vec2(-t.y, t.x);
  bool atB = aSeg.z > 0.5;
  vec4 cp = atB ? pb : pa;
  float z = atB ? -vb.z : -va.z;
  float coc = 0.5 * uCocK * abs(z - uFocus) / z;
  float sig = sqrt(uWidth * uWidth * uScale * uScale + 0.25 * coc * coc);
  cp.xy += n * aSide * 3.0 * sig / (0.5 * uRes) * cp.w;
  gl_Position = cp;
  vec3 c = atB ? cb : ca;
  vec3 wp = (inverse(viewMatrix) * (atB ? vb : va)).xyz;
  float nearFade = smoothstep(uNear, uNear + 0.25, z);
  vCol = c * nearFade * exp(-uExt * length(wp - cameraPosition)) / (sig / uScale * 2.5066283);
  vAcross = aSide * 3.0;
}
`;

const FRAG = /* glsl */ `
varying vec3 vCol;
varying float vAcross;
void main() {
  gl_FragColor = vec4(vCol * exp(-0.5 * vAcross * vAcross), 0.0);
}
`;

export interface LinesLook {
  lens: LensState;
  /** Cross-section sigma, 1080p px (default 0.55). */
  width?: number;
  extinction: number;
  /** Lines nearer than this are cut (m, default 0.08). */
  near?: number;
}

export class LightLines {
  readonly mesh: THREE.Mesh;
  private readonly pos: Float32Array;
  private readonly col: Float32Array;
  private readonly tPos: THREE.DataTexture;
  private readonly tCol: THREE.DataTexture;
  private readonly mat: THREE.ShaderMaterial;

  constructor(
    private readonly ctx: StageContext,
    readonly lines: number,
    readonly verts: number,
  ) {
    this.pos = new Float32Array(lines * verts * 4);
    this.col = new Float32Array(lines * verts * 4);
    this.tPos = new THREE.DataTexture(this.pos, verts, lines, THREE.RGBAFormat, THREE.FloatType);
    this.tCol = new THREE.DataTexture(this.col, verts, lines, THREE.RGBAFormat, THREE.FloatType);
    for (const t of [this.tPos, this.tCol]) {
      t.minFilter = THREE.NearestFilter;
      t.magFilter = THREE.NearestFilter;
      t.generateMipmaps = false;
      t.needsUpdate = true;
    }
    const seg: number[] = [];
    const side: number[] = [];
    const idx: number[] = [];
    let k = 0;
    for (let l = 0; l < lines; l++) {
      for (let i = 0; i + 1 < verts; i++) {
        for (const [e, s] of [
          [0, -1],
          [0, 1],
          [1, 1],
          [1, -1],
        ]) {
          seg.push(l, i, e);
          side.push(s);
        }
        idx.push(k, k + 1, k + 2, k, k + 2, k + 3);
        k += 4;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(seg.length), 3));
    g.setAttribute('aSeg', new THREE.Float32BufferAttribute(seg, 3));
    g.setAttribute('aSide', new THREE.Float32BufferAttribute(side, 1));
    g.setIndex(idx);
    this.mat = emissive(
      new THREE.ShaderMaterial({
        vertexShader: VERT,
        fragmentShader: FRAG,
        uniforms: {
          tPos: { value: this.tPos },
          tCol: { value: this.tCol },
          uRes: { value: new THREE.Vector2(ctx.width, ctx.height) },
          uScale: { value: ctx.scale },
          uCocK: { value: 0 },
          uFocus: { value: 10 },
          uWidth: { value: 0.55 },
          uExt: { value: 0 },
          uNear: { value: 0.08 },
        },
        depthTest: false,
        toneMapped: false,
        // The ribbons' winding follows the segment's screen direction: draw both faces.
        side: THREE.DoubleSide,
      }),
    );
    this.mesh = new THREE.Mesh(g, this.mat);
    this.mesh.frustumCulled = false;
  }

  /** Vertex i of line l: position (world) and colour times intensity (as a 1-px line). */
  set(l: number, i: number, p: THREE.Vector3, r: number, g: number, b: number): void {
    const o = (l * this.verts + i) * 4;
    this.pos[o] = p.x;
    this.pos[o + 1] = p.y;
    this.pos[o + 2] = p.z;
    this.col[o] = r;
    this.col[o + 1] = g;
    this.col[o + 2] = b;
  }

  /** Every vertex of line l dark (it is not drawn). */
  clear(l: number): void {
    this.col.fill(0, l * this.verts * 4, (l + 1) * this.verts * 4);
  }

  update(look: LinesLook): void {
    const u = this.mat.uniforms;
    u.uCocK.value = cocK(look.lens, this.ctx.width);
    u.uFocus.value = look.lens.focus;
    u.uWidth.value = look.width ?? 0.55;
    u.uExt.value = look.extinction;
    u.uNear.value = look.near ?? 0.08;
    this.tPos.needsUpdate = true;
    this.tCol.needsUpdate = true;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.mat.dispose();
    this.tPos.dispose();
    this.tCol.dispose();
  }
}
