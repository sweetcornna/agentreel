// 功能卡 (a typed caption card): a film's interface layer. A cursor comes in at the head of a chapter
// and types one line, small, below the guide line (layout.lineY) on the left: Chinese above, English
// below. It holds for a couple of beats and fades. The picture style may change from chapter to
// chapter; this card does not.
//
//   const card = new FeatureCard(ctx, { zh, en, at: '13.2' });
//   screen.add(card.object);           // the screen layer (canvas pixels, y down)
//   card.update(frame);                // every frame; a pure function of the frame
//
// Keystrokes fall on the beat grid (`zhRate` characters per beat: 4 = sixteenths, 8 = thirty-seconds),
// so the typing is in time with the music without a key sound. English is typed after the Chinese
// (`en: 'type'`, `enRate` per beat) or fades in once the Chinese is done (`en: 'fade'`, the default).
// `ink: true` draws it as ink on paper (multiplied, for the chapters on Paper); otherwise it is light.

import * as THREE from 'three';
import type { StageContext } from '../core/types';
import { CLOCK, posToBeat, type Pos } from '../lib/time';
import { LINE_Y_FRAC } from '../lib/layout';
import { FONT, fontCss } from './fonts';
import { measureTrackedLine } from './glyphs';

type RGB = [number, number, number];

export interface FeatureCardSpec {
  zh: string;
  en: string;
  /** First keystroke (小节.拍). The cursor appears one beat earlier and blinks. */
  at: Pos;
  /** Chinese characters per beat (default 4: sixteenths; 8 for the one-bar cards). */
  zhRate?: number;
  /** English typed after the Chinese, or faded in once it is done (default 'fade'). */
  enMode?: 'type' | 'fade';
  /** English characters per beat when typed (default 8). */
  enRate?: number;
  /** Fade mode: how much of the Chinese is typed when the English starts to fade in (0..1; 0.5 = half
   *  typed; default 1 = once it is all typed). The English must then stay up, fully in, for at least
   *  its character count / 17 seconds before the card fades (a reading-time rule). */
  enAt?: number;
  /** Fade mode: beats the English takes to fade in (default 0.5). */
  enFade?: number;
  /** Beats the finished card holds before it fades (default 2.5). */
  hold?: number;
  /** Beats of the fade-out (default 1). */
  fade?: number;
  /** Ink on paper (multiplied) instead of light (added). */
  ink?: boolean;
  /** Left edge as a fraction of the width (default 0.1) and the first baseline below the line (1080p px, default 40). */
  x?: number;
  dy?: number;
  /** Light colour (linear HDR) or, with `ink`, what the ink absorbs. */
  color?: RGB;
  /** Type sizes, 1080p px (default 21 and 15). */
  zhPx?: number;
  enPx?: number;
}

const VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
}
`;

const FRAG = /* glsl */ `
uniform sampler2D tText;
uniform vec2 uBox;        // box size (px)
uniform float uSplit;     // y (px from the box top) between the two lines
uniform vec2 uCut;        // typed extent of line 0 and line 1 (px from the box left)
uniform float uEn;        // English opacity (fade mode)
uniform float uAlpha;     // whole-card opacity
uniform vec4 uCaret;      // x, top, height (px in the box), on 0..1
uniform float uCaretW;
uniform vec3 uColor;
varying vec2 vUv;
void main() {
  vec2 p = vec2(vUv.x, 1.0 - vUv.y) * uBox;
  float a = texture2D(tText, vUv).a;
  bool first = p.y < uSplit;
  float cut = first ? uCut.x : uCut.y;
  a *= step(p.x, cut);
  if (!first) a *= uEn;
  float c = uCaret.w * step(uCaret.x, p.x) * step(p.x, uCaret.x + uCaretW) * step(uCaret.y, p.y) * step(p.y, uCaret.y + uCaret.z);
  float v = max(a, c) * uAlpha;
  if (v <= 0.0) discard;
  gl_FragColor = vec4(uColor * v, 1.0);
}
`;

interface Key {
  line: 0 | 1;
  /** Right edge of the character (px from the box left). */
  x1: number;
  frame: number;
}

export class FeatureCard {
  readonly object: THREE.Mesh;
  /** Keystroke frames (fractional), in order: for a key sound, if one is ever wanted. */
  readonly keyFrames: number[];
  /** First and last frame at which anything is drawn. */
  readonly from: number;
  readonly to: number;
  private readonly keys: Key[] = [];
  private readonly u: Record<string, THREE.IUniform>;
  private readonly tex: THREE.CanvasTexture;
  private readonly enMode: 'type' | 'fade';
  /** Beat at which the English starts to fade in, and its fade-in length (fade mode). */
  private readonly enFrom: number;
  private readonly enFade: number;
  private readonly fadeFrom: number;
  private readonly lineEnd: [number, number];
  private readonly caretTop: [number, number];
  private readonly caretH: [number, number];
  private readonly ox: number;

  constructor(
    private readonly ctx: StageContext,
    readonly spec: FeatureCardSpec,
  ) {
    const s = ctx.scale;
    const W = ctx.width;
    const H = ctx.height;
    const zhPx = (spec.zhPx ?? 21) * s;
    const enPx = (spec.enPx ?? 15) * s;
    const zhFont = fontCss(FONT.sansSC, 400, zhPx);
    const enFont = fontCss(FONT.mono, 300, enPx);
    const meas = document.createElement('canvas').getContext('2d')!;
    const zhRun = measureTrackedLine(meas, spec.zh, zhPx, 0.04 * zhPx, zhFont);
    const enRun = measureTrackedLine(meas, spec.en, enPx, 0, enFont);
    const end = (r: { x: number; w: number }[]) => (r.length ? r[r.length - 1].x + r[r.length - 1].w : 0);

    const lineY = LINE_Y_FRAC * H;
    const base0 = lineY + (spec.dy ?? 40) * s;
    const base1 = base0 + 1.55 * enPx + 0.35 * zhPx;
    const pad = Math.ceil(6 * s);
    const left = Math.round((spec.x ?? 0.1) * W);
    const bx = left - pad;
    const by = Math.floor(base0 - zhPx * 1.1) - pad;
    const bw = Math.ceil(Math.max(end(zhRun), end(enRun))) + 2 * pad + Math.ceil(6 * s);
    const bh = Math.ceil(base1 + enPx * 0.4) + pad - by;

    const cv = document.createElement('canvas');
    cv.width = bw;
    cv.height = bh;
    const g = cv.getContext('2d')!;
    g.fillStyle = '#ffffff';
    g.textBaseline = 'alphabetic';
    for (const c of zhRun) {
      g.font = c.font;
      g.fillText(c.ch, left - bx + c.x, base0 - by);
    }
    for (const c of enRun) {
      g.font = c.font;
      g.fillText(c.ch, left - bx + c.x, base1 - by);
    }
    this.tex = new THREE.CanvasTexture(cv);
    this.tex.colorSpace = THREE.NoColorSpace;
    this.tex.generateMipmaps = false;
    this.tex.minFilter = THREE.LinearFilter;
    this.tex.magFilter = THREE.LinearFilter;
    this.tex.flipY = true;

    // Keystrokes on the beat grid.
    const b0 = posToBeat(spec.at);
    const zr = spec.zhRate ?? 4;
    const er = spec.enRate ?? 8;
    this.enMode = spec.enMode ?? 'fade';
    const ox = left - bx;
    this.ox = ox;
    zhRun.forEach((c, i) => this.keys.push({ line: 0, x1: ox + c.x + c.w, frame: CLOCK.beatToFrame(b0 + i / zr) }));
    const bZhDone = b0 + zhRun.length / zr;
    let bDone = bZhDone;
    if (this.enMode === 'type') {
      // Return to the second line on the next beat subdivision.
      const b1 = bZhDone + 1 / er;
      enRun.forEach((c, i) => this.keys.push({ line: 1, x1: ox + c.x + c.w, frame: CLOCK.beatToFrame(b1 + i / er) }));
      bDone = b1 + enRun.length / er;
    } else bDone = Math.max(bZhDone, b0 + (spec.enAt ?? 1) * (bZhDone - b0) + (spec.enFade ?? 0.5));
    this.enFrom = b0 + (spec.enAt ?? 1) * (bZhDone - b0);
    this.enFade = spec.enFade ?? 0.5;
    this.keyFrames = this.keys.map((k) => k.frame);
    const hold = spec.hold ?? 2.5;
    this.fadeFrom = CLOCK.beatToFrame(bDone + hold);
    this.from = Math.floor(CLOCK.beatToFrame(b0 - 1));
    this.to = Math.ceil(CLOCK.beatToFrame(bDone + hold + (spec.fade ?? 1)));
    this.lineEnd = [ox + end(zhRun), ox + end(enRun)];
    this.caretTop = [base0 - by - zhPx * 0.92, base1 - by - enPx * 0.92];
    this.caretH = [zhPx * 1.12, enPx * 1.12];

    const color = spec.color ?? (spec.ink ? [0.86, 0.86, 0.88] : [1.25, 1.14, 0.98]);
    this.u = {
      tText: { value: this.tex },
      uBox: { value: new THREE.Vector2(bw, bh) },
      uSplit: { value: (base0 - by + zhPx * 0.3 + (base1 - by - enPx * 1.0)) / 2 },
      uCut: { value: new THREE.Vector2(0, 0) },
      uEn: { value: 0 },
      uAlpha: { value: 0 },
      uCaret: { value: new THREE.Vector4(0, 0, 0, 0) },
      uCaretW: { value: Math.max(1, 1.6 * s) },
      uColor: { value: new THREE.Vector3(...color) },
    };
    const mat = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: this.u,
      depthTest: false,
      depthWrite: false,
      transparent: true,
      side: THREE.DoubleSide,
      ...(spec.ink
        ? { blending: THREE.CustomBlending, blendEquation: THREE.AddEquation, blendSrc: THREE.ZeroFactor, blendDst: THREE.OneMinusSrcColorFactor, blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.OneFactor }
        : { blending: THREE.AdditiveBlending }),
    });
    const geo = new THREE.PlaneGeometry(bw, bh);
    geo.translate(bx + bw / 2, by + bh / 2, 0);
    // The screen camera is y down: flip the uvs so the texture reads upright.
    const uv = geo.getAttribute('uv') as THREE.BufferAttribute;
    for (let i = 0; i < uv.count; i++) uv.setY(i, 1 - uv.getY(i));
    this.object = new THREE.Mesh(geo, mat);
    this.object.frustumCulled = false;
    this.object.renderOrder = 900;
  }

  /** True while the card draws anything. */
  active(frame: number): boolean {
    return frame >= this.from && frame <= this.to;
  }

  update(frame: number): void {
    const u = this.u;
    if (!this.active(frame)) {
      u.uAlpha.value = 0;
      this.object.visible = false;
      return;
    }
    this.object.visible = true;
    let n = 0;
    while (n < this.keys.length && this.keys[n].frame <= frame) n++;
    let cut0 = 0;
    let cut1 = 0;
    for (let i = 0; i < n; i++) {
      const k = this.keys[i];
      if (k.line === 0) cut0 = k.x1;
      else cut1 = k.x1;
    }
    const enTyped = this.enMode === 'type';
    if (!enTyped) cut1 = this.lineEnd[1] + 10;
    (u.uCut.value as THREE.Vector2).set(cut0, cut1);
    const beat = CLOCK.frameToBeat(frame);
    // English fades in over `enFade` beats from `enAt` of the way through the Chinese.
    u.uEn.value = enTyped ? 1 : Math.min(1, Math.max(0, (beat - this.enFrom) / this.enFade));
    // Fade-out; a short fade-in of the cursor itself.
    const fadeB = this.spec.fade ?? 1;
    const out = 1 - Math.min(1, Math.max(0, (beat - CLOCK.frameToBeat(this.fadeFrom)) / fadeB));
    u.uAlpha.value = out * out;

    // The cursor: on the line being typed; steady while keys fall, blinking on the beat (first half
    // lit) before the first key and after the last; gone once the card fades.
    const typing = n > 0 && n < this.keys.length;
    const onLine: 0 | 1 = n > 0 && this.keys[n - 1].line === 1 ? 1 : 0;
    const caretX = (n === 0 ? this.ox : onLine === 0 ? cut0 : cut1) + 2 * this.ctx.scale;
    const blink = beat - Math.floor(beat) < 0.5 ? 1 : 0;
    const on = frame >= this.fadeFrom ? 0 : typing ? 1 : blink;
    (u.uCaret.value as THREE.Vector4).set(caretX, this.caretTop[onLine], this.caretH[onLine], on);
  }

  dispose(): void {
    this.object.geometry.dispose();
    (this.object.material as THREE.Material).dispose();
    this.tex.dispose();
  }
}
