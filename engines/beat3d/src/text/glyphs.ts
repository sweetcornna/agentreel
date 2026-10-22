// GlyphField: a line (or two) of typed text living in the HDR screen layer, rasterised once with
// Canvas2D into a float data texture that also carries, per pixel:
//   R = frame the glyph is typed (keystroke), G = frame the pixel lets go (turns into a thread),
//   B = line index, A = coverage.
// So typing, re-centring while typing, "字化为光" and "光聚成字" are all one shader and one map.
// Glyph positions are the final centred layout; while a line is being typed it is shifted so the
// typed part stays centred (an editor with centred alignment), caret at its right end.

import * as THREE from 'three';
import type { StageContext } from '../core/types';
import { Rng, hashString } from '../lib/rng';
import { fbm2 } from '../lib/noise';
import { FONT, fontCss } from './fonts';
import { measureTypedLine } from './cards';

export interface GlyphLine {
  text: string;
  /** Font size in 1080p px. */
  px: number;
  /** Baseline offset from the film line in 1080p px (negative = above the line). */
  dy: number;
  monoWeight?: number;
  /** 'typed': CJK in Noto Sans SC, Latin in JetBrains Mono (default). 'garamond': EB Garamond 400.
   *  'serifSC': Noto Serif SC 300 (titles), tracked like 'garamond'. */
  font?: 'typed' | 'garamond' | 'serifSC';
  /** Extra letter spacing in em (0.12 = +12 %). */
  tracking?: number;
  /** Not typed: the whole line fades in over this many frames starting at `appearAt`. */
  fadeIn?: number;
  appearAt?: number;
  /** Frames over which a pixel fades when it lets go (default 3; a line that just fades out: ~16). */
  fadeOut?: number;
  /** Brightening just before letting go (default 0.7; 0 for lines that simply fade). */
  glow?: number;
}

export interface GlyphFieldSpec {
  key: string;
  /** Centre x in canvas px. */
  cx: number;
  /** The film line, canvas px. */
  lineY: number;
  lines: GlyphLine[];
  /** Linear HDR colour (x intensity). */
  color: [number, number, number];
  /** Keystroke frame for every glyph of the typed lines, concatenated (spaces included). */
  keystrokes: number[];
}

export interface ReleaseSpec {
  start: number;
  span: number;
  /** Pixels nearer this side let go first. */
  from: 'right' | 'left';
  /** Weight of stroke depth: thin ends and edges release before stroke interiors. */
  edge: number;
  noise: number;
}

interface GlyphBox {
  line: number;
  ch: string;
  x0: number;
  x1: number;
  key: number;
}

const NEVER = 1e9;

export class GlyphField {
  readonly mesh: THREE.Mesh;
  readonly glyphs: GlyphBox[] = [];
  readonly lineWidths: number[] = [];
  readonly lineBaselines: number[] = [];
  readonly capHeights: number[] = [];
  readonly box: { x: number; y: number; w: number; h: number };
  private readonly data: Float32Array;
  private readonly tex: THREE.DataTexture;
  private readonly u: Record<string, THREE.IUniform>;
  private lineSplit = 0;

  constructor(
    readonly ctx: StageContext,
    readonly spec: GlyphFieldSpec,
  ) {
    const s = ctx.scale;
    const meas = document.createElement('canvas').getContext('2d')!;
    let gi = 0;
    let minY = Infinity;
    let maxY = -Infinity;
    let minX = Infinity;
    let maxX = -Infinity;
    const runs = spec.lines.map((ln, li) => {
      const px = ln.px * s;
      const run =
        ln.font === 'garamond'
          ? measureTrackedLine(meas, ln.text, px, (ln.tracking ?? 0) * px)
          : ln.font === 'serifSC'
            ? measureTrackedLine(meas, ln.text, px, (ln.tracking ?? 0) * px, fontCss(FONT.serifSC, 300, px))
            : measureTypedLine(meas, ln.text, px, ln.monoWeight ?? 300);
      const w = run.length ? run[run.length - 1].x + run[run.length - 1].w : 0;
      const baseline = spec.lineY + ln.dy * s;
      meas.font = fontCss(FONT.mono, ln.monoWeight ?? 300, px);
      const cap = meas.measureText('H').actualBoundingBoxAscent;
      this.lineWidths.push(w);
      this.lineBaselines.push(baseline);
      this.capHeights.push(cap);
      const x0 = spec.cx - w / 2;
      for (const r of run) {
        const typed = ln.fadeIn === undefined;
        const key = typed ? (spec.keystrokes[gi] ?? NEVER) : (ln.appearAt ?? 0);
        this.glyphs.push({ line: li, ch: r.ch, x0: x0 + r.x, x1: x0 + r.x + r.w, key });
        if (typed) gi++;
      }
      minX = Math.min(minX, x0);
      maxX = Math.max(maxX, x0 + w);
      minY = Math.min(minY, baseline - px * 1.05);
      maxY = Math.max(maxY, baseline + px * 0.35);
      return { run, px, baseline, x0, li };
    });
    const pad = Math.ceil(6 * s);
    const bx = Math.floor(minX) - pad;
    const by = Math.floor(minY) - pad;
    const bw = Math.ceil(maxX) + pad - bx;
    const bh = Math.ceil(maxY) + pad - by;
    this.box = { x: bx, y: by, w: bw, h: bh };
    if (runs.length > 1) this.lineSplit = Math.round((runs[0].baseline + runs[0].px * 0.35 + runs[1].baseline - runs[1].px * 1.05) / 2) - by;
    else this.lineSplit = bh + 1;

    // Rasterise (white on transparent) at the final layout.
    const cv = document.createElement('canvas');
    cv.width = bw;
    cv.height = bh;
    const g = cv.getContext('2d', { willReadFrequently: true })!;
    g.fillStyle = '#ffffff';
    g.textBaseline = 'alphabetic';
    g.textAlign = 'left';
    for (const r of runs) {
      for (const c of r.run) {
        g.font = c.font;
        g.fillText(c.ch, r.x0 + c.x - bx, r.baseline - by);
      }
    }
    const img = g.getImageData(0, 0, bw, bh).data;
    this.data = new Float32Array(bw * bh * 4);
    for (let y = 0; y < bh; y++) {
      const cy = y + by;
      const line = runs.length > 1 && y >= this.lineSplit ? 1 : 0;
      for (let x = 0; x < bw; x++) {
        const i = (y * bw + x) * 4;
        const a = img[i + 3] / 255;
        const cxp = x + bx + 0.5;
        let key = NEVER;
        if (a > 0) {
          for (const gb of this.glyphs) {
            if (gb.line === line && cxp >= gb.x0 - 1 && cxp < gb.x1 + 1) {
              key = gb.key;
              if (cxp < gb.x1) break;
            }
          }
        }
        void cy;
        this.data[i] = key;
        this.data[i + 1] = NEVER;
        this.data[i + 2] = line;
        this.data[i + 3] = a;
      }
    }
    this.tex = new THREE.DataTexture(this.data, bw, bh, THREE.RGBAFormat, THREE.FloatType);
    this.tex.minFilter = THREE.NearestFilter;
    this.tex.magFilter = THREE.NearestFilter;
    this.tex.generateMipmaps = false;
    this.tex.needsUpdate = true;

    this.u = {
      tMap: { value: this.tex },
      uBox: { value: new THREE.Vector4(bx, by, bw, bh) },
      uShift: { value: new THREE.Vector2() },
      uSplit: { value: this.lineSplit },
      uFrame: { value: 0 },
      uColor: { value: new THREE.Vector3(...spec.color) },
      uGain: { value: 1 },
      uGlow: { value: new THREE.Vector2(spec.lines[0]?.glow ?? 0.7, spec.lines[1]?.glow ?? 0.7) },
      uFadeIn: { value: new THREE.Vector2(spec.lines[0]?.fadeIn ?? 1, spec.lines[1]?.fadeIn ?? 1) },
      uFadeOut: { value: new THREE.Vector2(spec.lines[0]?.fadeOut ?? 3, spec.lines[1]?.fadeOut ?? 3) },
    };
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]), 3));
    geo.setIndex([0, 1, 2, 0, 2, 3]);
    const mat = new THREE.ShaderMaterial({
      vertexShader: /* glsl */ `
        uniform vec4 uBox;
        varying vec2 vLocal;
        void main() {
          vLocal = position.xy * uBox.zw;
          gl_Position = projectionMatrix * viewMatrix * vec4(uBox.xy + vLocal, 0.0, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform sampler2D tMap;
        uniform vec4 uBox;
        uniform vec2 uShift;
        uniform float uSplit, uFrame, uGain;
        uniform vec2 uGlow, uFadeIn, uFadeOut;
        uniform vec3 uColor;
        varying vec2 vLocal;
        void main() {
          bool l0 = vLocal.y < uSplit;
          float shift = l0 ? uShift.x : uShift.y;
          float fin = l0 ? uFadeIn.x : uFadeIn.y;
          float fout = l0 ? uFadeOut.x : uFadeOut.y;
          float gl = l0 ? uGlow.x : uGlow.y;
          ivec2 ip = ivec2(floor(vLocal.x - shift), floor(vLocal.y));
          if (ip.x < 0 || ip.x >= int(uBox.z) || ip.y < 0 || ip.y >= int(uBox.w)) discard;
          vec4 m = texelFetch(tMap, ip, 0);
          if (m.a <= 0.0) discard;
          float typed = fin <= 1.0 ? clamp(uFrame - m.r + 1.0, 0.0, 1.0) : smoothstep(m.r, m.r + fin, uFrame);
          float gone = smoothstep(m.g, m.g + fout, uFrame);
          float glow = smoothstep(m.g - 8.0, m.g, uFrame) * (1.0 - gone);
          gl_FragColor = vec4(uColor * m.a * typed * (1.0 - gone) * (1.0 + gl * glow) * uGain, 1.0);
        }`,
      uniforms: this.u,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthTest: false,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.frustumCulled = false;
  }

  /** Every pixel of a line lets go at the same frame (a line that simply fades out). */
  setReleaseLine(line: number, frame: number): void {
    const { w, h } = this.box;
    for (let i = 0; i < w * h; i++) if (this.data[i * 4 + 2] === line && this.data[i * 4 + 3] > 0) this.data[i * 4 + 1] = frame;
    this.tex.needsUpdate = true;
  }

  /** Fill the release map. Deterministic: depends on layout, the spec and the field key. */
  setRelease(r: ReleaseSpec, lines?: number[]): void {
    const { w, h, x: bx } = this.box;
    const a = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) a[i] = this.data[i * 4 + 3];
    // Stroke depth: separable box blur (twice) of coverage.
    const rad = Math.max(1, Math.round(2.4 * this.ctx.scale));
    const depth = boxBlur(boxBlur(a, w, h, rad), w, h, rad);
    let xMin = Infinity;
    let xMax = -Infinity;
    for (const gb of this.glyphs) {
      if (lines && !lines.includes(gb.line)) continue;
      xMin = Math.min(xMin, gb.x0);
      xMax = Math.max(xMax, gb.x1);
    }
    const s = this.ctx.scale;
    const seed = (hashString(this.spec.key) % 997) / 10;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (a[i] <= 0) continue;
        if (lines && !lines.includes(this.data[i * 4 + 2])) continue;
        const px = x + bx + 0.5;
        let order = (xMax - px) / Math.max(1, xMax - xMin);
        if (r.from === 'left') order = 1 - order;
        const n = 0.5 + 0.5 * fbm2(px / (9 * s) + seed, (y + this.box.y) / (9 * s), 3);
        const t = Math.min(Math.max((1 - r.edge - r.noise) * order + r.edge * Math.min(depth[i] * 1.6, 1) + r.noise * n, 0), 1);
        this.data[i * 4 + 1] = r.start + r.span * t;
      }
    }
    this.tex.needsUpdate = true;
  }

  /** Release frame at a raster pixel (box-local integer coords). */
  releaseAt(x: number, y: number): number {
    return this.data[(y * this.box.w + x) * 4 + 1];
  }

  /** Pick n thread origins weighted by coverage. Positions are canvas px of the final layout. */
  sampleThreads(n: number, rng: Rng, lines?: number[]): { x: number; y: number; rel: number; amp: number }[] {
    const { w, h, x: bx, y: by } = this.box;
    const cum: number[] = [];
    const idx: number[] = [];
    let total = 0;
    for (let i = 0; i < w * h; i++) {
      const a = this.data[i * 4 + 3];
      if (a > 0.08 && (!lines || lines.includes(this.data[i * 4 + 2]))) {
        total += a;
        cum.push(total);
        idx.push(i);
      }
    }
    const out: { x: number; y: number; rel: number; amp: number }[] = [];
    for (let k = 0; k < n; k++) {
      const t = rng.next() * total;
      let lo = 0;
      let hi = cum.length - 1;
      while (lo < hi) {
        const m = (lo + hi) >> 1;
        if (cum[m] < t) lo = m + 1;
        else hi = m;
      }
      const i = idx[lo];
      const x = i % w;
      const y = Math.floor(i / w);
      out.push({ x: bx + x + rng.next(), y: by + y + rng.next(), rel: this.data[i * 4 + 1], amp: this.data[i * 4 + 3] });
    }
    void h;
    return out;
  }

  private typedWidth(line: number, frame: number): { width: number; started: boolean; done: boolean } {
    const gl = this.glyphs.filter((g) => g.line === line);
    let width = 0;
    let count = 0;
    const x0 = this.spec.cx - this.lineWidths[line] / 2;
    for (const g of gl) {
      if (g.key <= frame) {
        width = g.x1 - x0;
        count++;
      }
    }
    return { width, started: count > 0, done: count === gl.length };
  }

  /** Horizontal shift (integer px) of a line so its typed part is centred. */
  lineShift(line: number, frame: number): number {
    if (this.spec.lines[line].fadeIn !== undefined) return 0;
    const t = this.typedWidth(line, frame);
    if (t.done) return 0;
    return Math.round((this.lineWidths[line] - t.width) / 2);
  }

  /**
   * Caret for a frame: on the line being typed, at the right end of its typed text. `returnAt`:
   * after this frame the caret goes back to the end of the first line (it stands on the film line).
   */
  caret(frame: number, returnAt = NEVER): { x: number; bottom: number; height: number; line: number; typing: boolean } {
    let line = 0;
    for (let li = 0; li < this.lineWidths.length; li++) {
      if (this.spec.lines[li].fadeIn !== undefined) continue;
      const t = this.typedWidth(li, frame);
      line = li;
      if (!t.done) break;
    }
    if (frame >= returnAt) line = 0;
    const t = this.typedWidth(line, frame);
    const x = this.spec.cx - this.lineWidths[line] / 2 + t.width + this.lineShift(line, frame) + 1.5 * this.ctx.scale;
    const bottom = line === 0 ? this.spec.lineY : this.lineBaselines[line];
    const lastKey = Math.max(...this.glyphs.filter((g) => g.key <= frame && this.spec.lines[g.line].fadeIn === undefined).map((g) => g.key), -NEVER);
    return { x: t.started ? x : this.spec.cx, bottom, height: this.capHeights[line], line, typing: frame - lastKey < 10 };
  }

  update(frame: number, gain = 1): void {
    this.u.uFrame.value = frame;
    this.u.uGain.value = gain;
    this.u.uShift.value.set(this.lineShift(0, frame), this.lineWidths.length > 1 ? this.lineShift(1, frame) : 0);
    this.mesh.visible = gain > 0;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
    this.tex.dispose();
  }
}

function boxBlur(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  const n = 2 * r + 1;
  for (let y = 0; y < h; y++) {
    let acc = 0;
    for (let x = -r; x <= r; x++) acc += src[y * w + Math.min(Math.max(x, 0), w - 1)];
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = acc / n;
      acc += src[y * w + Math.min(x + r + 1, w - 1)] - src[y * w + Math.max(x - r, 0)];
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let y = -r; y <= r; y++) acc += tmp[Math.min(Math.max(y, 0), h - 1) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = acc / n;
      acc += tmp[Math.min(y + r + 1, h - 1) * w + x] - tmp[Math.max(y - r, 0) * w + x];
    }
  }
  return out;
}

/**
 * Humanised keystroke frames: one glyph per `step` frames (a 16th note is 5 frames) with seeded
 * jitter; `breaks` adds a pause before the given glyph indices (line changes).
 */
export function humanKeystrokes(key: string, start: number, count: number, step: number, opts: { jitter?: number; breaks?: Record<number, number> } = {}): number[] {
  const rng = new Rng(`${key}/keys`);
  const out: number[] = [];
  let t = start;
  for (let i = 0; i < count; i++) {
    t += opts.breaks?.[i] ?? 0;
    const j = (rng.next() - 0.5) * 2 * (opts.jitter ?? 0.35) * step;
    out.push(Math.round(t + (i === 0 ? 0 : j)));
    t += step;
  }
  for (let i = 1; i < out.length; i++) out[i] = Math.max(out[i], out[i - 1] + 1);
  return out;
}

/** EB Garamond line with tracking; positions from prefix measurement so kerning survives. */
export function measureTrackedLine(g: CanvasRenderingContext2D, text: string, px: number, tracking: number, fontIn?: string): { ch: string; x: number; w: number; font: string }[] {
  const font = fontIn ?? fontCss(FONT.garamond, 400, px);
  g.font = font;
  const chars = Array.from(text);
  const out: { ch: string; x: number; w: number; font: string }[] = [];
  let prefix = '';
  for (let i = 0; i < chars.length; i++) {
    const x = g.measureText(prefix).width + tracking * i;
    const w = g.measureText(chars[i]).width + (i < chars.length - 1 ? tracking : 0);
    out.push({ ch: chars[i], x, w, font });
    prefix += chars[i];
  }
  return out;
}
