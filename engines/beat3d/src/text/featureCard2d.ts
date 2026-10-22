// 功能卡 drawn on the engine's 2D overlay (composited after the grade and the negative, before grain),
// so one card reads the same over any shot: ink on paper, pale gold on lacquer, light on night blue.
// Same layout, sizes and beat-grid timing as `FeatureCard` (the GL card a program draws in its own
// screen layer); only the compositing differs: the colour is painted as is (alpha over), not added.
//
//   const card = new FeatureCard2D(spec, width, height);
//   card.draw(g, frame);   // returns false (and draws nothing) outside the card's window

import { CLOCK, posToBeat } from '../lib/time';
import { LINE_Y_FRAC } from '../lib/layout';
import { FONT, fontCss } from './fonts';
import { measureTrackedLine } from './glyphs';
import type { FeatureCardSpec } from './featureCard';

export interface FeatureCard2DSpec extends Omit<FeatureCardSpec, 'ink' | 'color'> {
  /** CSS colour of the text and cursor (sRGB, painted as is). */
  css: string;
  /** Beats the cursor shows (blinking) before the first key (default 1). */
  lead?: number;
  /** A soft knock-out of the ground colour behind the glyphs (blur radius in 1080p px), so a line or
   *  a bright patch passing under the text does not cut through it. */
  halo?: { css: string; px: number };
}

interface Key {
  line: 0 | 1;
  x1: number;
  frame: number;
}

export class FeatureCard2D {
  readonly from: number;
  readonly to: number;
  readonly keyFrames: number[];
  private readonly keys: Key[] = [];
  private readonly cv: HTMLCanvasElement;
  private readonly bx: number;
  private readonly by: number;
  private readonly bw: number;
  private readonly bh: number;
  private readonly split: number;
  private readonly ox: number;
  private readonly enFrom: number;
  private readonly enFade: number;
  private readonly fadeFrom: number;
  private readonly enMode: 'type' | 'fade';
  private readonly lineEnd: [number, number];
  private readonly caretTop: [number, number];
  private readonly caretH: [number, number];
  private readonly caretW: number;
  private readonly scale: number;

  constructor(
    readonly spec: FeatureCard2DSpec,
    width: number,
    height: number,
  ) {
    const s = height / 1080;
    this.scale = s;
    const zhPx = (spec.zhPx ?? 21) * s;
    const enPx = (spec.enPx ?? 15) * s;
    const zhFont = fontCss(FONT.sansSC, 400, zhPx);
    const enFont = fontCss(FONT.mono, 300, enPx);
    const meas = document.createElement('canvas').getContext('2d')!;
    const zhRun = measureTrackedLine(meas, spec.zh, zhPx, 0.04 * zhPx, zhFont);
    const enRun = measureTrackedLine(meas, spec.en, enPx, 0, enFont);
    const end = (r: { x: number; w: number }[]) => (r.length ? r[r.length - 1].x + r[r.length - 1].w : 0);

    const lineY = LINE_Y_FRAC * height;
    const base0 = lineY + (spec.dy ?? 40) * s;
    const base1 = base0 + 1.55 * enPx + 0.35 * zhPx;
    const pad = Math.ceil((6 + 2 * (spec.halo?.px ?? 0)) * s);
    const left = Math.round((spec.x ?? 0.1) * width);
    this.bx = left - pad;
    this.by = Math.floor(base0 - zhPx * 1.1) - pad;
    this.bw = Math.ceil(Math.max(end(zhRun), end(enRun))) + 2 * pad + Math.ceil(6 * s);
    this.bh = Math.ceil(base1 + enPx * 0.4) + pad - this.by;

    this.cv = document.createElement('canvas');
    this.cv.width = this.bw;
    this.cv.height = this.bh;
    const g = this.cv.getContext('2d')!;
    g.textBaseline = 'alphabetic';
    const paint = (css: string): void => {
      g.fillStyle = css;
      for (const c of zhRun) {
        g.font = c.font;
        g.fillText(c.ch, left - this.bx + c.x, base0 - this.by);
      }
      for (const c of enRun) {
        g.font = c.font;
        g.fillText(c.ch, left - this.bx + c.x, base1 - this.by);
      }
    };
    if (spec.halo) {
      g.save();
      g.shadowColor = spec.halo.css;
      g.shadowBlur = spec.halo.px * s;
      // Twice: a blurred shadow alone is too faint to hold back a full-strength line.
      paint(spec.halo.css);
      paint(spec.halo.css);
      g.restore();
    }
    paint(spec.css);

    const b0 = posToBeat(spec.at);
    const zr = spec.zhRate ?? 4;
    const er = spec.enRate ?? 8;
    this.enMode = spec.enMode ?? 'fade';
    const ox = left - this.bx;
    this.ox = ox;
    zhRun.forEach((c, i) => this.keys.push({ line: 0, x1: ox + c.x + c.w, frame: CLOCK.beatToFrame(b0 + i / zr) }));
    const bZhDone = b0 + zhRun.length / zr;
    let bDone: number;
    if (this.enMode === 'type') {
      const b1 = bZhDone + 1 / er;
      enRun.forEach((c, i) => this.keys.push({ line: 1, x1: ox + c.x + c.w, frame: CLOCK.beatToFrame(b1 + i / er) }));
      bDone = b1 + enRun.length / er;
    } else bDone = Math.max(bZhDone, b0 + (spec.enAt ?? 1) * (bZhDone - b0) + (spec.enFade ?? 0.5));
    this.enFrom = b0 + (spec.enAt ?? 1) * (bZhDone - b0);
    this.enFade = spec.enFade ?? 0.5;
    this.keyFrames = this.keys.map((k) => k.frame);
    const hold = spec.hold ?? 2.5;
    this.fadeFrom = CLOCK.beatToFrame(bDone + hold);
    this.from = Math.floor(CLOCK.beatToFrame(b0 - (spec.lead ?? 1)));
    this.to = Math.ceil(CLOCK.beatToFrame(bDone + hold + (spec.fade ?? 1)));
    this.lineEnd = [ox + end(zhRun), ox + end(enRun)];
    this.caretTop = [base0 - this.by - zhPx * 0.92, base1 - this.by - enPx * 0.92];
    this.caretH = [zhPx * 1.12, enPx * 1.12];
    this.caretW = Math.max(1, 1.6 * s);
    this.split = Math.round((base0 - this.by + zhPx * 0.3 + (base1 - this.by - enPx * 1.0)) / 2);
  }

  active(frame: number): boolean {
    return frame >= this.from && frame <= this.to;
  }

  /** Draw the card at `frame` (absolute film frame). False, and nothing drawn, outside its window. */
  draw(g: CanvasRenderingContext2D, frame: number): boolean {
    if (!this.active(frame)) return false;
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
    const beat = CLOCK.frameToBeat(frame);
    const en = enTyped ? 1 : Math.min(1, Math.max(0, (beat - this.enFrom) / this.enFade));
    const out = 1 - Math.min(1, Math.max(0, (beat - CLOCK.frameToBeat(this.fadeFrom)) / (this.spec.fade ?? 1)));
    const alpha = out * out;
    if (alpha <= 0) return false;

    g.save();
    g.globalAlpha = alpha;
    const w0 = Math.min(this.bw, Math.ceil(cut0));
    if (w0 > 0) g.drawImage(this.cv, 0, 0, w0, this.split, this.bx, this.by, w0, this.split);
    const w1 = Math.min(this.bw, Math.ceil(cut1));
    if (w1 > 0 && en > 0) {
      g.globalAlpha = alpha * en;
      g.drawImage(this.cv, 0, this.split, w1, this.bh - this.split, this.bx, this.by + this.split, w1, this.bh - this.split);
    }
    // Cursor: steady while keys fall, blinking on the beat before the first key and after the last.
    const typing = n > 0 && n < this.keys.length;
    const onLine: 0 | 1 = n > 0 && this.keys[n - 1].line === 1 ? 1 : 0;
    const caretX = (n === 0 ? this.ox : onLine === 0 ? cut0 : cut1) + 2 * this.scale;
    const blink = beat - Math.floor(beat) < 0.5;
    const on = frame < this.fadeFrom && (typing || blink);
    if (on) {
      g.globalAlpha = alpha;
      g.fillStyle = this.spec.css;
      g.fillRect(this.bx + caretX, this.by + this.caretTop[onLine], this.caretW, this.caretH[onLine]);
    }
    g.restore();
    return true;
  }
}
