// Film look parameters. One default set for the engine; a film overrides it (FilmDef.post), per shot
// (FilmDef.shotPost), per program (ProgramDef.post) and per frame (FrameResult.post), in that order.
// Every length is in 1080p pixels and is scaled by the engine, so 1080p and 4K share one set of numbers.

export type Vec3 = [number, number, number];

export interface PostParams {
  /** Exposure in stops applied to the scene before the display transform. */
  exposure: number;
  bloom: {
    /** Energy-conserving mix of the multi-scale blur into the image (0.04 = 4 %). */
    mix: number;
    /** Soft threshold (scene linear) and knee; threshold 0 = all light scatters (physical). */
    threshold: number;
    knee: number;
    /** Relative weight of each scale, finest first. Level k ~ radius 2^(k+1) px at 1080p. */
    weights: number[];
    tint: Vec3;
  };
  halation: {
    /** Additive amount of the red-orange glow that bright edges leak into the film base. */
    amount: number;
    threshold: number;
    tint: Vec3;
  };
  streak: {
    /** Anamorphic horizontal flare from very bright points. */
    amount: number;
    threshold: number;
    /** Falloff per octave of length (0.5 short, 0.85 long). */
    falloff: number;
    tint: Vec3;
    /** 0..1: threshold a 5-tap mean so thin lit stems and 1-px glints do not streak. Default 0. */
    despeckle?: number;
  };
  grain: {
    /** Std-dev of the grain in display code values at mid-grey (0.035 = fine 35 mm stock). */
    amount: number;
    /** Grain cell size in 1080p pixels. */
    size: number;
    /** 0 = monochrome grain, 1 = fully independent per channel. */
    chroma: number;
  };
  vignette: { amount: number; softness: number };
  /** Lateral chromatic aberration at the frame corner, in 1080p pixels. */
  chromatic: number;
  /**
   * 'warm': the long wavelengths image larger and spill outward over the edge (red, then orange) and
   * nothing is taken away, so a fringe never turns cyan or blue (for films that keep blue out of the
   * frame). 'rgb': the classic split, red out and blue in.
   */
  chromaticStyle: 'warm' | 'rgb';
  grade: {
    /** AgX look in encoded space (ASC-CDL). */
    slope: Vec3;
    offset: Vec3;
    power: Vec3;
    lookSaturation: number;
    /** Display-referred lift/gamma/gain + saturation, after the display transform. */
    lift: Vec3;
    gamma: Vec3;
    gain: Vec3;
    saturation: number;
  };
  /**
   * Depth of field from the depth buffer of the world (single-pass spiral gather, thin lens). Off by
   * default: most shots do their defocus optically per layer (post/LensBlur.ts) and per point
   * (world/bokeh.ts). With it on, the screen layer is drawn apart and added after the blur, sharp.
   */
  dof: {
    enabled: boolean;
    focalMm: number;
    fStop: number;
    /** Focus distance in metres (world units). */
    focus: number;
    /** Clamp of the circle of confusion radius, 1080p pixels (behind the focus plane). */
    maxRadius: number;
    /** Optional clamp in front of the focus plane (foreground), 1080p pixels; defaults to maxRadius. */
    maxRadiusNear?: number;
  };
  motionBlur: {
    /** Temporal sub-samples per frame (1 = off). */
    samples: number;
    /** Shutter as a fraction of the frame interval (0.5 = 180 degrees). */
    shutter: number;
  };
  /** Draw the letterbox from the aspect curve. */
  letterbox: boolean;
  /**
   * 0..1: how far the 2D overlay is kept clean of grain and dither, times its alpha. 1 = an opaque overlay pixel
   * leaves the composite exactly as drawn (a logo that must match its source file pixel for pixel).
   */
  overlayClean: number;
  /**
   * Negative to positive (light on black -> ink on paper): the picture's luminance is remapped, black
   * to `paper` and light to `ink` (linear display colours), mixed in by `amount` (0..1).
   */
  negative: {
    amount: number;
    paper: Vec3;
    ink: Vec3;
    /** Optional radial wipe: the turn happens inside a circle around `center` (canvas fractions, y
     *  from the top) of `radius` (fraction of the canvas height) with a soft edge `soft` (same
     *  unit). radius < 0 (default) = the whole frame at once. */
    wipe: { center: [number, number]; radius: number; soft: number };
  };
}

export const DEFAULT_POST: PostParams = {
  exposure: 0,
  bloom: { mix: 0.045, threshold: 0, knee: 0.5, weights: [0.6, 0.8, 1.0, 1.0, 0.9, 0.75, 0.6], tint: [1, 1, 1] },
  halation: { amount: 0.18, threshold: 0.9, tint: [1.0, 0.3, 0.08] },
  streak: { amount: 0.12, threshold: 3.0, falloff: 0.82, tint: [1.0, 0.62, 0.4] },
  grain: { amount: 0.011, size: 0.8, chroma: 0.06 },
  vignette: { amount: 0.22, softness: 0.65 },
  chromatic: 0.9,
  chromaticStyle: 'warm',
  grade: {
    slope: [1, 1, 1],
    offset: [0, 0, 0],
    power: [1, 1, 1],
    lookSaturation: 1.0,
    lift: [0.0006, 0.0008, 0.0014],
    gamma: [1, 1, 1],
    gain: [1, 1, 1],
    saturation: 1.0,
  },
  dof: { enabled: false, focalMm: 50, fStop: 2.8, focus: 10, maxRadius: 18 },
  motionBlur: { samples: 1, shutter: 0.5 },
  letterbox: true,
  overlayClean: 0,
  negative: { amount: 0, paper: [0.8, 0.74, 0.6], ink: [0.028, 0.028, 0.032], wipe: { center: [0.5, 0.5], radius: -1, soft: 0.04 } },
};

export type DeepPartial<T> = { [K in keyof T]?: T[K] extends number[] ? T[K] : T[K] extends string ? T[K] : T[K] extends object ? DeepPartial<T[K]> : T[K] };
export type PostOverrides = DeepPartial<PostParams>;

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/** Merge overrides left to right onto a copy of `base`. Arrays replace, objects merge. */
export function mergePost(base: PostParams, ...overrides: (PostOverrides | undefined)[]): PostParams {
  const out = structuredClone(base) as unknown as Record<string, unknown>;
  for (const o of overrides) {
    if (!o) continue;
    mergeInto(out, o as Record<string, unknown>);
  }
  return out as unknown as PostParams;
}
function mergeInto(dst: Record<string, unknown>, src: Record<string, unknown>): void {
  for (const [k, v] of Object.entries(src)) {
    if (v === undefined) continue;
    if (isPlainObject(v) && isPlainObject(dst[k])) mergeInto(dst[k] as Record<string, unknown>, v);
    else dst[k] = Array.isArray(v) ? [...v] : v;
  }
}
