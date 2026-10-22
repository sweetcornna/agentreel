// Display-referred text drawn with Canvas2D into the overlay layer (which WebGL composites after the
// display transform, before grain). Cinema cards, the title fallback, end credits, and a simple
// typed-card fallback for placeholders. All sizes are 1080p pixels times layout.scale.

import type { Layout } from '../lib/layout';
import { clamp, easeInOutSine, envelope } from '../lib/easing';
import { FONT, fontCss, isCJK } from './fonts';

export const CARD_FADE_IN = 10;
export const CARD_FADE_OUT = 16;
/** Letter-spacing growth over the life of a card, as a fraction of the line width. */
export const CARD_TRACK_GROWTH = 0.02;

export const INK = {
  cardZh: '#EEE9E0',
  cardEn: '#D8D2C8',
  warm: '#F2EDE4',
  cool: '#E8F4FF',
  neutral: '#EDEDEA',
};

/**
 * Draw `text` centred on cx with extra tracking (px between glyphs). Positions come from measuring
 * prefixes, so the font's own kerning survives.
 */
export function drawTracked(g: CanvasRenderingContext2D, text: string, cx: number, baseline: number, tracking: number): number {
  const chars = Array.from(text);
  const natural = g.measureText(text).width;
  const total = natural + tracking * Math.max(0, chars.length - 1);
  let x0 = cx - total / 2;
  let prefix = '';
  g.textAlign = 'left';
  g.textBaseline = 'alphabetic';
  for (let i = 0; i < chars.length; i++) {
    const x = x0 + g.measureText(prefix).width + tracking * i;
    g.fillText(chars[i], x, baseline);
    prefix += chars[i];
  }
  return total;
}

const isAllCaps = (s: string): boolean => /[A-Z]/.test(s) && s === s.toUpperCase();

export interface CinemaCardText {
  zh?: string;
  en?: string;
  source?: { zh: string; en: string };
}

export interface CinemaCardStyle {
  zhPx: number;
  enPx: number;
  zhWeight: number;
  /** Gap between the line and the bottom of the Chinese ink / top of the English caps (1080p px). */
  gapAbove: number;
  gapBelow: number;
  /** Optional (default CARD_FADE_IN / CARD_FADE_OUT, linear): fade lengths in frames, eased (sine). */
  fadeIn?: number;
  fadeOut?: number;
  /** Optional: the English line comes in this many frames after the Chinese (both leave together). Default 0. */
  enDelay?: number;
}

export const CINEMA_STYLE: CinemaCardStyle = { zhPx: 44, enPx: 24, zhWeight: 300, gapAbove: 20, gapBelow: 22 };
export const TITLE_STYLE: CinemaCardStyle = { zhPx: 96, enPx: 22, zhWeight: 300, gapAbove: 26, gapBelow: 26 };

/** Cinema card: Chinese above the line, English below, fade 10 in / 16 out, tracking opens 2 %. */
export function drawCinemaCard(
  g: CanvasRenderingContext2D,
  L: Layout,
  card: CinemaCardText,
  frame: number,
  from: number,
  to: number,
  style: CinemaCardStyle = CINEMA_STYLE,
): void {
  const eased = style.fadeIn !== undefined || style.fadeOut !== undefined;
  const env = (lag: number): number => {
    const e = envelope(frame - lag, from, to, style.fadeIn ?? CARD_FADE_IN, style.fadeOut ?? CARD_FADE_OUT);
    return eased ? easeInOutSine(e) : e;
  };
  const a = env(0);
  // The English comes in a little after the Chinese and leaves with it.
  const lagIn = style.enDelay ? envelope(frame - style.enDelay, from, to + style.enDelay, style.fadeIn ?? CARD_FADE_IN, 0) : 1;
  const aEn = Math.min(a, eased ? easeInOutSine(lagIn) : lagIn);
  if (a <= 0 && aEn <= 0) return;
  const grow = easeInOutSine(clamp((frame - from) / Math.max(1, to - from)));
  const s = L.scale;
  const cx = L.width / 2;
  g.save();
  if (card.zh) {
    g.font = fontCss(FONT.serifSC, style.zhWeight, style.zhPx * s);
    g.fillStyle = INK.cardZh;
    g.globalAlpha = a;
    const n = Array.from(card.zh).length;
    const base = 0.12 * style.zhPx * s;
    const w0 = g.measureText(card.zh).width + base * (n - 1);
    const tracking = base + (CARD_TRACK_GROWTH * w0 * grow) / Math.max(1, n - 1);
    const m = g.measureText(card.zh);
    const baseline = L.lineY - style.gapAbove * s - m.actualBoundingBoxDescent;
    drawTracked(g, card.zh, cx, baseline, tracking);
  }
  if (card.en) {
    const caps = isAllCaps(card.en);
    g.font = fontCss(FONT.garamond, 400, style.enPx * s);
    g.fillStyle = INK.cardEn;
    g.globalAlpha = aEn * 0.92;
    const n = Array.from(card.en).length;
    const base = (caps ? 0.12 : 0.015) * style.enPx * s;
    const w0 = g.measureText(card.en).width + base * (n - 1);
    const tracking = base + (CARD_TRACK_GROWTH * w0 * grow) / Math.max(1, n - 1);
    const m = g.measureText('H');
    const baseline = L.lineY + style.gapBelow * s + m.actualBoundingBoxAscent;
    drawTracked(g, card.en, cx, baseline, tracking);
  }
  if (card.source) {
    g.globalAlpha = a * 0.7;
    g.font = fontCss(FONT.serifSC, 300, 20 * s);
    g.fillStyle = INK.cardEn;
    const yZh = L.lineY + (style.gapBelow + style.enPx + 44) * s;
    drawTracked(g, card.source.zh, cx, yZh, 0.1 * 20 * s);
    g.font = fontCss(FONT.garamond, 400, 14 * s);
    drawTracked(g, card.source.en, cx, yZh + 26 * s, 0.14 * 14 * s);
  }
  g.restore();
}

/** Mixed-script line: CJK in Noto Sans SC 400, everything else in JetBrains Mono. */
export function measureTypedLine(g: CanvasRenderingContext2D, text: string, px: number, monoWeight = 300): { ch: string; x: number; w: number; font: string }[] {
  const out: { ch: string; x: number; w: number; font: string }[] = [];
  let x = 0;
  for (const ch of Array.from(text)) {
    const font = isCJK(ch) ? fontCss(FONT.sansSC, 400, px) : fontCss(FONT.mono, monoWeight, px);
    g.font = font;
    const w = g.measureText(ch).width;
    out.push({ ch, x, w, font });
    x += w;
  }
  return out;
}

/** Placeholder typing: characters revealed evenly across the card, no particles. */
export function drawTypedFallback(g: CanvasRenderingContext2D, L: Layout, zh: string, en: string, frame: number, from: number, to: number, color: string): void {
  if (frame < from || frame >= to) return;
  const s = L.scale;
  const all = Array.from(zh).length + Array.from(en).length;
  const shown = Math.floor(clamp((frame - from) / Math.max(1, (to - from) * 0.8)) * all);
  const zhRun = measureTypedLine(g, zh, 28 * s);
  const enRun = measureTypedLine(g, en, 17 * s);
  const zhW = zhRun.length ? zhRun[zhRun.length - 1].x + zhRun[zhRun.length - 1].w : 0;
  const enW = enRun.length ? enRun[enRun.length - 1].x + enRun[enRun.length - 1].w : 0;
  g.save();
  g.fillStyle = color;
  g.textBaseline = 'alphabetic';
  g.textAlign = 'left';
  zhRun.forEach((c, i) => {
    if (i >= shown) return;
    g.font = c.font;
    g.fillText(c.ch, L.width / 2 - zhW / 2 + c.x, L.lineY - 9 * s);
  });
  enRun.forEach((c, i) => {
    if (i + zhRun.length >= shown) return;
    g.font = c.font;
    g.fillText(c.ch, L.width / 2 - enW / 2 + c.x, L.lineY + 30 * s);
  });
  g.restore();
}

export interface CreditsData {
  cards: (
    | { rows: { role: { zh: string; en: string }; name: { zh: string; en: string } }[] }
    | { paragraphs: { zh: string; en: string }[] }
    | { lines: string[] }
  )[];
}

function wrap(g: CanvasRenderingContext2D, text: string, maxW: number): string[] {
  const tokens: string[] = [];
  let buf = '';
  for (const ch of Array.from(text)) {
    if (isCJK(ch)) {
      if (buf) tokens.push(buf);
      buf = '';
      tokens.push(ch);
    } else if (ch === ' ') {
      buf += ch;
      tokens.push(buf);
      buf = '';
    } else buf += ch;
  }
  if (buf) tokens.push(buf);
  const lines: string[] = [];
  let line = '';
  for (const t of tokens) {
    const trial = line + t;
    if (g.measureText(trial.trimEnd()).width > maxW && line) {
      lines.push(line.trimEnd());
      line = t.trimStart();
    } else line = trial;
  }
  if (line.trim()) lines.push(line.trimEnd());
  return lines;
}

/** End credits: three cards spread evenly over [from, to), each with the card fades. */
export function drawCredits(g: CanvasRenderingContext2D, L: Layout, frame: number, from: number, to: number, data: CreditsData): void {
  const n = data.cards.length;
  const span = (to - from) / n;
  const idx = Math.floor((frame - from) / span);
  if (idx < 0 || idx >= n) return;
  const c0 = from + idx * span;
  const a = envelope(frame, c0, c0 + span, CARD_FADE_IN + 6, CARD_FADE_OUT + 4);
  if (a <= 0) return;
  const s = L.scale;
  const cx = L.width / 2;
  const card = data.cards[idx];
  g.save();
  g.globalAlpha = a;
  g.textAlign = 'center';
  g.textBaseline = 'alphabetic';
  if ('rows' in card) {
    let y = L.lineY - 90 * s;
    for (const r of card.rows) {
      g.fillStyle = INK.cardEn;
      g.font = fontCss(FONT.serifSC, 300, 18 * s);
      g.fillText(r.role.zh, cx, y);
      g.font = fontCss(FONT.serifSC, 400, 30 * s);
      g.fillStyle = INK.cardZh;
      g.fillText(r.name.zh, cx, y + 44 * s);
      y += 120 * s;
    }
  } else if ('paragraphs' in card) {
    const maxW = 980 * s;
    let y = L.lineY - 150 * s;
    for (const p of card.paragraphs) {
      g.fillStyle = INK.cardZh;
      g.font = fontCss(FONT.serifSC, 300, 21 * s);
      for (const ln of wrap(g, p.zh, maxW)) {
        g.fillText(ln, cx, y);
        y += 36 * s;
      }
      y += 22 * s;
      g.fillStyle = INK.cardEn;
      g.font = fontCss(FONT.garamond, 400, 19 * s);
      for (const ln of wrap(g, p.en, maxW)) {
        g.fillText(ln, cx, y);
        y += 29 * s;
      }
    }
  } else {
    let y = L.lineY - 110 * s;
    g.fillStyle = INK.cardEn;
    g.font = fontCss(FONT.serifSC, 300, 18 * s);
    for (const ln of card.lines) {
      g.fillText(ln, cx, y);
      y += 40 * s;
    }
  }
  g.restore();
}
