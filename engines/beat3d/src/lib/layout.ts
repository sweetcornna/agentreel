// Screen geometry every shot shares: the letterbox, the 1080p-relative scale, and (for films that keep
// one) the guide line.
// Rule: anything sized "in 1080p pixels" is multiplied by `scale` so 4K looks identical.

import { ASPECT } from './curves';
import { TIMELINE } from './time';

export interface Layout {
  width: number;
  height: number;
  /** height / 1080. Multiply every 1080p pixel measurement by this. */
  scale: number;
  /** y of the film's guide line in canvas pixels from the top (timeline `line.yFrac`, default the middle). */
  lineY: number;
  /** Visible picture aspect at this frame (2.39, 1.7778 or in between). */
  aspect: number;
  /** Top/bottom of the visible (non-letterboxed) band, canvas pixels from the top. Integers. */
  visibleTop: number;
  visibleBottom: number;
}

/** The guide line's height as a fraction of the canvas (a film that keeps a horizon at one height sets it). */
export const LINE_Y_FRAC = TIMELINE.line?.yFrac ?? 0.5;

export function letterboxBand(width: number, height: number, aspect: number): { top: number; bottom: number } {
  const visH = Math.min(height, width / aspect);
  const top = Math.round((height - visH) / 2);
  return { top, bottom: height - top };
}

export function layoutAt(width: number, height: number, frame: number): Layout {
  const aspect = ASPECT.at(frame);
  const { top, bottom } = letterboxBand(width, height, aspect);
  return { width, height, scale: height / 1080, lineY: LINE_Y_FRAC * height, aspect, visibleTop: top, visibleBottom: bottom };
}
