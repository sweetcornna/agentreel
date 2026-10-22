// Text written in light, in 3D: every character is its own quad (so it can sit anywhere in depth and
// ignite on its own frame), cut from one glyph atlas (by default JetBrains Mono for Latin, Noto Sans SC
// for CJK; any loaded faces). Defocus goes through the same aperture as everything else
// (lib/aperture.ts): the atlas is blurred once through LensBlur at a few radii and each character
// blends the levels its circle of confusion asks for, so a character close to the lens opens into the
// same soft shape as a lamp would.
//
// Build: new LightText(ctx, chars), lay out with advance(), then place(). Light is emission only: it
// adds to the target and leaves its alpha alone. It dims through the haze with distance, and as the
// lens goes through a line the characters near it are pushed out around it (they part, not clip).
// inkPoints() samples points on a glyph's strokes (for the threads that write it).

import * as THREE from 'three';
import type { StageContext } from '../core/types';
import { hdrTarget } from '../post/FilmPost';
import { LensBlur } from '../post/LensBlur';
import { cocK, type LensState } from '../lib/lens';
import { defaultAperture, type ApertureSpec } from '../lib/aperture';
import type { Rng } from '../lib/rng';
import { FONT, fontCss, isCJK } from './fonts';
import { emissive } from '../world/stage';
import type { RGB } from '../lib/color';

/** Atlas cell: glyphs at this many px per em, with room around them for the widest blur. */
const EM_PX = 112;
const PAD = 40;
const CELL_H = EM_PX + 2 * PAD;
/** Blur radii (atlas px) of the levels after the sharp one. */
const LEVEL_R = [4, 11, 30];

interface Glyph {
  /** Advance in em. */
  advance: number;
  /** Atlas cell x and width (px). */
  x: number;
  w: number;
}

export interface CharPlacement {
  ch: string;
  /** Centre of the character's em box (world). */
  position: THREE.Vector3;
  /** Unit vectors of the character's plane (world): reading direction and up. */
  right: THREE.Vector3;
  up: THREE.Vector3;
  /** Em size (m). */
  em: number;
  /** Frame it lights (fractional allowed). */
  ignite: number;
  /** Linear colour times intensity once it has settled. */
  color: RGB;
}

const VERT = /* glsl */ `
attribute vec3 iPos;
attribute vec3 iRight;
attribute vec3 iUp;
attribute vec4 iRect;     // atlas u0, u1 of the cell; half-width, half-height of the quad (m)
attribute vec4 iCol;      // rgb, ignite frame
uniform float uFrame;
uniform vec2 uRes;
uniform float uCocK;
uniform float uFocus;
uniform float uGain;
uniform float uHotGain;
uniform vec3 uHot;
uniform float uExt;
uniform float uScatter;
uniform vec2 uFade;
varying vec2 vUv;
varying float vRA;        // CoC radius in atlas px
varying vec3 vCol;
void main() {
  float age = uFrame - iCol.a;
  // Dark until its frame; then a white-hot flash that cools into the character's own colour.
  float on = smoothstep(-0.5, 0.5, age);
  float hot = age >= 0.0 ? exp(-age / 2.5) : 0.0;
  vCol = (iCol.rgb * (1.0 - 0.6 * hot) + uHot * uHotGain * hot) * on * uGain;
  vec4 c = modelViewMatrix * vec4(iPos, 1.0);
  float z = -c.z;
  if (z < 0.03 || on <= 0.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); vCol = vec3(0.0); return; }
  vCol *= exp(-uExt * length(c.xyz));
  // Gone to light before the lens reaches it (uFade: fully there beyond .y, gone by .x; off when x = y = 0).
  if (uFade.y > 0.0) vCol *= smoothstep(uFade.x, uFade.y, z);
  vec3 p = iPos + iRight * position.x * iRect.z + iUp * position.y * iRect.w;
  vec4 mvp = modelViewMatrix * vec4(p, 1.0);
  // Parting around the lens, out from the view axis: within the last metre and a half, or, for a
  // line that gives way to light (uFade), over that same stretch.
  vec2 part = uFade.y > 0.0 ? vec2(uFade.y, uFade.x - 0.4) : vec2(1.6, 0.2);
  mvp.xy += c.xy * uScatter * smoothstep(part.x, part.y, z);
  gl_Position = projectionMatrix * mvp;
  float ppm = 0.5 * uRes.y * projectionMatrix[1][1] / z;
  float coc = 0.5 * uCocK * abs(z - uFocus) / z;
  vRA = coc * ${CELL_H.toFixed(1)} / max(2.0 * iRect.w * ppm, 1e-3);
  vUv = vec2(mix(iRect.x, iRect.y, position.x * 0.5 + 0.5), position.y * 0.5 + 0.5);
}
`;

const r0 = LEVEL_R[0].toFixed(1);
const r1 = LEVEL_R[1].toFixed(1);
const r2 = LEVEL_R[2].toFixed(1);
const FRAG = /* glsl */ `
uniform sampler2D tL0;
uniform sampler2D tL1;
uniform sampler2D tL2;
uniform sampler2D tL3;
varying vec2 vUv;
varying float vRA;
varying vec3 vCol;
void main() {
  if (max(vCol.r, max(vCol.g, vCol.b)) <= 0.0) discard;
  float r = vRA;
  float a;
  if (r < ${r0}) a = mix(texture2D(tL0, vUv).r, texture2D(tL1, vUv).r, r / ${r0});
  else if (r < ${r1}) a = mix(texture2D(tL1, vUv).r, texture2D(tL2, vUv).r, (r - ${r0}) / (${r1} - ${r0}));
  else a = mix(texture2D(tL2, vUv).r, texture2D(tL3, vUv).r, clamp((r - ${r1}) / (${r2} - ${r1}), 0.0, 1.0));
  // Beyond the widest level: the same energy, spread thinner.
  a *= ${r2} * ${r2} / max(r * r, ${r2} * ${r2});
  gl_FragColor = vec4(vCol * a, 0.0);
}
`;

/** The faces a LightText draws with: CJK and the rest, each a family and weight (default: Noto Sans SC
 *  400 and JetBrains Mono 300). */
export interface LightFaces {
  cjk?: [string, number];
  latin?: [string, number];
  /** Aperture the defocused levels are blurred through (default: the film's). */
  aperture?: ApertureSpec;
}
const DEFAULT_FACES: Required<Omit<LightFaces, 'aperture'>> = { cjk: [FONT.sansSC, 400], latin: [FONT.mono, 300] };

export class LightText {
  readonly mesh: THREE.Mesh;
  private readonly glyphs = new Map<string, Glyph>();
  private readonly sharp: THREE.CanvasTexture;
  private readonly targets: THREE.WebGLRenderTarget[] = [];
  private readonly mat: THREE.ShaderMaterial;
  private geo: THREE.InstancedBufferGeometry | null = null;
  private readonly atlasW: number;
  /** The sharp atlas's coverage (row-major, top row first), for sampling glyph strokes. */
  private readonly ink: Uint8Array;

  constructor(
    private readonly ctx: StageContext,
    chars: string,
    faces: LightFaces = {},
  ) {
    const cjk = faces.cjk ?? DEFAULT_FACES.cjk;
    const latin = faces.latin ?? DEFAULT_FACES.latin;
    const fontOf = (ch: string): string => (isCJK(ch) ? fontCss(cjk[0], cjk[1], EM_PX) : fontCss(latin[0], latin[1], EM_PX));
    // Atlas: each distinct character once, in a row.
    const g0 = document.createElement('canvas').getContext('2d')!;
    let x = 0;
    for (const ch of new Set(Array.from(chars))) {
      g0.font = fontOf(ch);
      const advance = g0.measureText(ch).width / EM_PX;
      const w = Math.ceil(advance * EM_PX) + 2 * PAD;
      this.glyphs.set(ch, { advance, x, w });
      x += w;
    }
    this.atlasW = x;
    const cv = document.createElement('canvas');
    cv.width = this.atlasW;
    cv.height = CELL_H;
    const g = cv.getContext('2d')!;
    g.fillStyle = '#000';
    g.fillRect(0, 0, cv.width, cv.height);
    g.fillStyle = '#fff';
    g.textBaseline = 'alphabetic';
    for (const [ch, gl] of this.glyphs) {
      g.font = fontOf(ch);
      // Baseline so the em box (ascent .8, descent .2) is centred in the cell.
      g.fillText(ch, gl.x + PAD, PAD + 0.8 * EM_PX);
    }
    const img = g.getImageData(0, 0, cv.width, cv.height).data;
    this.ink = new Uint8Array(cv.width * cv.height);
    for (let i = 0; i < this.ink.length; i++) this.ink[i] = img[i * 4];
    this.sharp = new THREE.CanvasTexture(cv);
    this.sharp.colorSpace = THREE.NoColorSpace;
    this.sharp.generateMipmaps = true;
    this.sharp.minFilter = THREE.LinearMipmapLinearFilter;
    this.sharp.anisotropy = 8;

    // Defocused levels through the aperture.
    const renderer = ctx.renderer;
    const blur = new LensBlur(renderer, this.atlasW, CELL_H, 256, faces.aperture ?? defaultAperture());
    const src = hdrTarget(this.atlasW, CELL_H);
    const copy = new THREE.MeshBasicMaterial({ map: this.sharp, toneMapped: false });
    const plane = new THREE.PlaneGeometry(2, 2);
    const sc = new THREE.Scene().add(new THREE.Mesh(plane, copy));
    renderer.setRenderTarget(src);
    renderer.render(sc, new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1));
    copy.dispose();
    plane.dispose();
    for (const r of LEVEL_R) {
      const t = hdrTarget(this.atlasW, CELL_H);
      blur.applyRadius(src.texture, t, r);
      this.targets.push(t);
    }
    blur.dispose();
    src.dispose();

    this.mat = emissive(
      new THREE.ShaderMaterial({
        vertexShader: VERT,
        fragmentShader: FRAG,
        uniforms: {
          tL0: { value: this.sharp },
          tL1: { value: this.targets[0].texture },
          tL2: { value: this.targets[1].texture },
          tL3: { value: this.targets[2].texture },
          uFrame: { value: 0 },
          uRes: { value: new THREE.Vector2(ctx.width, ctx.height) },
          uCocK: { value: 0 },
          uFocus: { value: 10 },
          uGain: { value: 1 },
          uHotGain: { value: 1 },
          uHot: { value: new THREE.Vector3(1, 1, 1) },
          uExt: { value: 0 },
          uScatter: { value: 0 },
          uFade: { value: new THREE.Vector2(0, 0) },
        },
        depthTest: false,
        toneMapped: false,
      }),
    );
    this.mesh = new THREE.Mesh(new THREE.BufferGeometry(), this.mat);
    this.mesh.frustumCulled = false;
  }

  /** Advance of a character (em). */
  advance(ch: string): number {
    const g = this.glyphs.get(ch);
    if (!g) throw new Error(`LightText: no glyph for "${ch}"`);
    return g.advance;
  }

  /** `n` points on the strokes of `ch`, in em from the centre of its em box (x right, y up). */
  inkPoints(ch: string, n: number, rng: Rng): [number, number][] {
    const gl = this.glyphs.get(ch);
    if (!gl) throw new Error(`LightText: no glyph for "${ch}"`);
    const hits: [number, number][] = [];
    for (let y = 0; y < CELL_H; y++) for (let x = gl.x; x < gl.x + gl.w; x++) if (this.ink[y * this.atlasW + x] > 140) hits.push([x, y]);
    const out: [number, number][] = [];
    for (let i = 0; i < n && hits.length; i++) {
      const [x, y] = hits[rng.int(0, hits.length)];
      out.push([(x + 0.5 - (gl.x + gl.w / 2)) / EM_PX, (CELL_H / 2 - y - 0.5) / EM_PX]);
    }
    return out;
  }

  /** Set the characters (once). */
  place(chars: CharPlacement[]): void {
    const n = chars.length;
    const pos = new Float32Array(n * 3);
    const right = new Float32Array(n * 3);
    const up = new Float32Array(n * 3);
    const rect = new Float32Array(n * 4);
    const col = new Float32Array(n * 4);
    chars.forEach((p, i) => {
      const gl = this.glyphs.get(p.ch);
      if (!gl) throw new Error(`LightText: no glyph for "${p.ch}"`);
      pos.set([p.position.x, p.position.y, p.position.z], i * 3);
      right.set([p.right.x, p.right.y, p.right.z], i * 3);
      up.set([p.up.x, p.up.y, p.up.z], i * 3);
      const k = p.em / EM_PX;
      rect.set([gl.x / this.atlasW, (gl.x + gl.w) / this.atlasW, 0.5 * gl.w * k, 0.5 * CELL_H * k], i * 4);
      col.set([p.color[0], p.color[1], p.color[2], p.ignite], i * 4);
    });
    const geo = new THREE.InstancedBufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]), 3));
    geo.setIndex([0, 1, 2, 0, 2, 3]);
    geo.setAttribute('iPos', new THREE.InstancedBufferAttribute(pos, 3));
    geo.setAttribute('iRight', new THREE.InstancedBufferAttribute(right, 3));
    geo.setAttribute('iUp', new THREE.InstancedBufferAttribute(up, 3));
    geo.setAttribute('iRect', new THREE.InstancedBufferAttribute(rect, 4));
    geo.setAttribute('iCol', new THREE.InstancedBufferAttribute(col, 4));
    geo.instanceCount = n;
    this.geo?.dispose();
    this.geo = geo;
    this.mesh.geometry = geo;
  }

  /** Frame, lens, overall gain, and the colour (times intensity) of the ignition flash; haze
   *  extinction (1/m), how far characters part around the lens (fraction of their offset) and the
   *  distances (m, gone by the first, whole beyond the second) over which they give way to light. */
  update(frame: number, lens: LensState, gain: number, hot: RGB, opts: { hotGain?: number; extinction?: number; scatter?: number; fadeNear?: [number, number] } = {}): void {
    const u = this.mat.uniforms;
    u.uExt.value = opts.extinction ?? 0;
    u.uScatter.value = opts.scatter ?? 0;
    u.uFade.value.set(...(opts.fadeNear ?? [0, 0]));
    u.uFrame.value = frame;
    u.uCocK.value = cocK(lens, this.ctx.width);
    u.uFocus.value = lens.focus;
    u.uGain.value = gain;
    u.uHot.value.set(...hot);
    u.uHotGain.value = opts.hotGain ?? 1;
  }

  dispose(): void {
    this.geo?.dispose();
    this.mesh.geometry.dispose();
    this.mat.dispose();
    this.sharp.dispose();
    for (const t of this.targets) t.dispose();
  }
}
