// The contract between the engine, a film and its shot programs.
//
// A film (films/<name>/scenes/index.ts) default-exports a FilmDef: its programs and its look. A program
// renders one or more consecutive shots (a long take that runs through three shots without a cut
// is one program that owns all three). The engine calls frame() with an absolute film frame; the program poses its
// scenes for that instant and returns them. frame() MUST be a pure function of `req.frame`: no state
// carried between calls, no clocks, no Math.random. Frames are rendered out of order, in parallel,
// and (for motion blur) at fractional times.

import type * as THREE from 'three';
import type { ApertureSpec } from '../lib/aperture';
import type { Layout } from '../lib/layout';
import type { ShotTiming } from '../lib/time';
import type { PostOverrides } from '../post/params';
import type { CreditsData } from '../text/cards';

export interface StageContext {
  renderer: THREE.WebGLRenderer;
  width: number;
  height: number;
  /** height / 1080 */
  scale: number;
  /** Orthographic camera in canvas pixels, origin top-left, y down. For FrameResult.screen. */
  screenCamera: THREE.OrthographicCamera;
}

export interface FrameRequest {
  /** Absolute film frame. Fractional during motion-blur sub-samples. */
  frame: number;
  shot: ShotTiming;
  layout: Layout;
  /** True for motion-blur sub-samples: only scene poses are used, overlay/post are ignored. */
  subSample: boolean;
}

export interface FrameResult {
  /** 3D world, drawn first into the HDR target (with depth, used by the post `dof`). */
  world?: { scene: THREE.Scene; camera: THREE.PerspectiveCamera };
  /** Screen-space HDR layer in canvas pixels (y down), drawn over the world, before post. */
  screen?: THREE.Scene;
  /** Display-referred 2D layer (slates, subtitles). Composited after the display transform, before grain. */
  overlay?: (g: CanvasRenderingContext2D, layout: Layout) => void;
  /** Per-frame post overrides (merged over program and shot overrides). */
  post?: PostOverrides;
  /** HDR clear colour, linear. Default black. */
  clear?: [number, number, number];
}

export interface ShotProgram {
  frame(req: FrameRequest): FrameResult;
  dispose(): void;
}

export interface ProgramDef {
  /** Stable id; also the root of every rng seed in the program. */
  id: string;
  /** Timeline shot ids this program renders. */
  shots: string[];
  /** Static post overrides for all its shots. */
  post?: PostOverrides;
  /** Build GPU resources. May be async (glyph rasters, precomputed data). Must be deterministic. */
  create(ctx: StageContext): ShotProgram | Promise<ShotProgram>;
  /** Card ids from the timeline's `cards` this program draws itself (the engine then skips them). */
  handlesCards?: string[];
}

/** What a film's scenes/index.ts default-exports. Everything but `programs` is optional. */
export interface FilmDef {
  /** Shot programs. A shot claimed twice is an error; a shot nobody claims gets the placeholder slate. */
  programs: ProgramDef[];
  /** Style tests (`--lab`): these stand in for the shots they claim. Never in a final render. */
  lab?: ProgramDef[];
  /** The film's look over the engine defaults (under shotPost, program and frame overrides). */
  post?: PostOverrides;
  /** Per-shot look overrides. */
  shotPost?: Record<string, PostOverrides>;
  /** The film's lens aperture: the default of every BokehPoints, LensBlur and LightText (default circle). */
  aperture?: ApertureSpec;
  /** Film-level 2D layer drawn over whatever shot is on screen (captions that span shots). True if it drew. */
  overlay?: (g: CanvasRenderingContext2D, frame: number, layout: Layout) => boolean;
  /** Data for the timeline's `credits` cards. */
  credits?: CreditsData;
}
