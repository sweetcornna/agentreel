// FilmEngine: one WebGL2 context, one HDR target, one post chain, a small cache of shot programs.
// renderFrame(n) renders frame n of the film completely (world, screen layer, cards and overlays,
// post, letterbox) into the canvas. It depends on n only. Frames that no shot covers (the tail after
// the last shot) are pure black: no grain, no dither.

import * as THREE from 'three';
import type { FrameResult, ProgramDef, ShotProgram, StageContext } from './types';
import { CLOCK, TIMELINE, shotAt, type ShotTiming } from '../lib/time';
import { layoutAt, type Layout } from '../lib/layout';
import { hashString } from '../lib/rng';
import { FilmPost } from '../post/FilmPost';
import { DEFAULT_POST, mergePost, type PostOverrides } from '../post/params';
import { loadFonts } from '../text/fonts';
import { CINEMA_STYLE, INK, TITLE_STYLE, drawCinemaCard, drawCredits, drawTypedFallback } from '../text/cards';
import { currentFilm, filmPost, placeholderProgram, programFor, shotPost } from './registry';

export interface EngineOptions {
  /** Force placeholders for every shot (fast "time skeleton" render). */
  skeleton?: boolean;
  /** MSAA samples of the HDR scene target. */
  msaa?: number;
  /** Keep at most this many programs alive (each may hold large GPU buffers). */
  programCache?: number;
  /** Post overrides applied last (look development, tests). */
  postOverride?: PostOverrides;
}

export interface FrameStats {
  frame: number;
  /** Shot id, or 'black' where the timeline has no picture. */
  shot: string;
  program: string;
  /** ms spent creating a program this frame (0 when cached). */
  createMs: number;
  /** ms from start to GPU completion (includes a readback fence). */
  renderMs: number;
}

export class FilmEngine {
  readonly renderer: THREE.WebGLRenderer;
  readonly width: number;
  readonly height: number;
  readonly ctx: StageContext;
  private readonly post: FilmPost;
  private readonly sceneRT: THREE.WebGLRenderTarget;
  private readonly overlayCanvas: HTMLCanvasElement;
  private readonly overlayCtx: CanvasRenderingContext2D;
  private readonly overlayTex: THREE.CanvasTexture;
  private readonly programs = new Map<string, { prog: ShotProgram; lastUse: number }>();
  private useCounter = 0;
  lastStats: FrameStats | null = null;

  constructor(
    readonly canvas: HTMLCanvasElement,
    width: number,
    height: number,
    readonly opts: EngineOptions = {},
  ) {
    this.width = width;
    this.height = height;
    canvas.width = width;
    canvas.height = height;
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false,
      alpha: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(width, height, false);
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.autoClear = false;

    this.sceneRT = new THREE.WebGLRenderTarget(width, height, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      samples: opts.msaa ?? 4,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      generateMipmaps: false,
      depthBuffer: true,
      // Depth of the world, for the post chain's depth-of-field pass (`dof`).
      depthTexture: new THREE.DepthTexture(width, height, THREE.UnsignedIntType),
    });
    this.post = new FilmPost(this.renderer, width, height);

    this.overlayCanvas = document.createElement('canvas');
    this.overlayCanvas.width = width;
    this.overlayCanvas.height = height;
    const g = this.overlayCanvas.getContext('2d', { willReadFrequently: false });
    if (!g) throw new Error('2D canvas unavailable');
    this.overlayCtx = g;
    this.overlayTex = new THREE.CanvasTexture(this.overlayCanvas);
    this.overlayTex.colorSpace = THREE.NoColorSpace;
    this.overlayTex.generateMipmaps = false;
    this.overlayTex.minFilter = THREE.LinearFilter;
    this.overlayTex.magFilter = THREE.LinearFilter;

    const screenCamera = new THREE.OrthographicCamera(0, width, 0, height, -1000, 1000);
    screenCamera.position.set(0, 0, 10);
    screenCamera.updateMatrixWorld(true);
    this.ctx = { renderer: this.renderer, width, height, scale: height / 1080, screenCamera };
  }

  async init(): Promise<void> {
    const gl = this.renderer.getContext();
    if (!(gl instanceof WebGL2RenderingContext)) throw new Error('WebGL2 is required');
    await loadFonts();
  }

  private async program(def: ProgramDef): Promise<ShotProgram> {
    const hit = this.programs.get(def.id);
    if (hit) {
      hit.lastUse = ++this.useCounter;
      return hit.prog;
    }
    const prog = await def.create(this.ctx);
    this.programs.set(def.id, { prog, lastUse: ++this.useCounter });
    const max = this.opts.programCache ?? 2;
    while (this.programs.size > max) {
      let oldest: string | null = null;
      let t = Infinity;
      for (const [k, v] of this.programs) if (v.lastUse < t) [oldest, t] = [k, v.lastUse];
      if (!oldest) break;
      this.programs.get(oldest)!.prog.dispose();
      this.programs.delete(oldest);
    }
    return prog;
  }

  defFor(shot: ShotTiming): ProgramDef {
    return this.opts.skeleton ? placeholderProgram : programFor(shot.id);
  }

  private readonly emptyScene = new THREE.Scene();
  /** The screen layer on its own target, for shots with depth of field (added after the DOF pass). */
  private screenRT: THREE.WebGLRenderTarget | null = null;

  private drawScene(res: FrameResult, screenApart = false): void {
    const r = this.renderer;
    r.setRenderTarget(this.sceneRT);
    const c = res.clear ?? [0, 0, 0];
    r.setClearColor(new THREE.Color(c[0], c[1], c[2]), 1);
    r.clear(true, true, false);
    if (res.world) r.render(res.world.scene, res.world.camera);
    const screen = screenApart ? undefined : res.screen;
    if (screen) {
      r.clearDepth();
      r.render(screen, this.ctx.screenCamera);
    }
    // The scene target is multisampled: a bare clear never resolves into its texture, so a frame
    // that draws nothing would show the last frame this page drew. Rendering an empty scene resolves it.
    if (!res.world && !screen) r.render(this.emptyScene, this.ctx.screenCamera);
  }

  /**
   * Screen layer (text, threads, lines, glow points: all additive, no depth) on its own target. With
   * depth of field on, the post chain adds it after the DOF pass, so it is never blurred by the depth
   * of whatever lies under it.
   */
  private drawScreenApart(screen: THREE.Scene): THREE.Texture {
    const r = this.renderer;
    this.screenRT ??= new THREE.WebGLRenderTarget(this.width, this.height, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      samples: this.opts.msaa ?? 4,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      generateMipmaps: false,
      depthBuffer: false,
    });
    r.setRenderTarget(this.screenRT);
    r.setClearColor(0x000000, 1);
    r.clear(true, false, false);
    r.render(screen, this.ctx.screenCamera);
    return this.screenRT.texture;
  }

  /** The timeline's cards (cinema/title/credits, typed fallbacks) unless the program draws them. */
  private drawCards(g: CanvasRenderingContext2D, L: Layout, frame: number, def: ProgramDef): boolean {
    let drew = false;
    for (const card of TIMELINE.cards ?? []) {
      if (def.handlesCards?.includes(card.id)) continue;
      const from = CLOCK.f(card.from);
      const to = CLOCK.f(card.to);
      if (frame < from || frame >= to) continue;
      drew = true;
      if (card.kind === 'cinema') drawCinemaCard(g, L, card, frame, from, to, CINEMA_STYLE);
      else if (card.kind === 'title') drawCinemaCard(g, L, card, frame, from, to, TITLE_STYLE);
      else if (card.kind === 'credits') {
        const credits = currentFilm().credits;
        if (credits) drawCredits(g, L, frame, from, to, credits);
      } else if (card.kind === 'typed') drawTypedFallback(g, L, card.zh ?? '', card.en ?? '', frame, from, to, card.tone === 'cool' ? INK.cool : card.tone === 'warm' ? INK.warm : INK.neutral);
    }
    return drew;
  }

  /** Pure black on the canvas (frames outside every shot: the hard cut to black and the tail). */
  private drawBlack(): void {
    const r = this.renderer;
    r.setRenderTarget(null);
    r.setClearColor(0x000000, 1);
    r.clear(true, false, false);
  }

  async renderFrame(frameIn: number): Promise<FrameStats> {
    const t0 = performance.now();
    const frame = Math.floor(frameIn);
    const shot = shotAt(frame);
    if (!shot) {
      this.drawBlack();
      return this.fence(t0, { frame, shot: 'black', program: 'black', createMs: 0 });
    }
    const def = this.defFor(shot);
    const tc = performance.now();
    const prog = await this.program(def);
    const createMs = performance.now() - tc;

    const layout = layoutAt(this.width, this.height, frame);
    const res = prog.frame({ frame, shot, layout, subSample: false });
    const params = mergePost(DEFAULT_POST, filmPost(), shotPost(shot.id), def.post, res.post, this.opts.postOverride);
    const screenApart = params.dof.enabled && !!res.world;

    const n = Math.max(1, Math.round(params.motionBlur.samples));
    let hdr: THREE.Texture = this.sceneRT.texture;
    if (n === 1) {
      this.drawScene(res, screenApart);
    } else {
      // Motion blur: the same frame at N instants inside the shutter, averaged in HDR.
      const acc = this.post.accumRT;
      this.renderer.setRenderTarget(acc);
      this.renderer.setClearColor(0x000000, 1);
      this.renderer.clear(true, false, false);
      for (let i = 0; i < n; i++) {
        const t = frame + ((i + 0.5) / n - 0.5) * params.motionBlur.shutter;
        // Sub-samples share the frame's layout: the picture's aspect belongs to the frame, not the instant.
        const sub = prog.frame({ frame: t, shot, layout, subSample: true });
        this.drawScene(sub, screenApart);
        this.post.mAccum.uniforms.tSrc.value = this.sceneRT.texture;
        this.post.mAccum.uniforms.uWeight.value = 1 / n;
        this.post.pass(this.post.mAccum, acc, false);
      }
      hdr = acc.texture;
    }

    const g = this.overlayCtx;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, this.width, this.height);
    // Cards draw in skeleton renders too: their timing is what a skeleton is for.
    let hasOverlay = this.drawCards(g, layout, frame, def);
    if (res.overlay) {
      res.overlay(g, layout);
      hasOverlay = true;
    }
    // The film's own layer over whatever shot is on screen (captions that span shots).
    const filmOverlay = currentFilm().overlay;
    if (!this.opts.skeleton && filmOverlay && filmOverlay(g, frame, layout)) hasOverlay = true;
    if (hasOverlay) this.overlayTex.needsUpdate = true;

    // With motion blur the separate screen layer is drawn once, at the frame's own time.
    const screenTex = screenApart && res.screen ? this.drawScreenApart(res.screen) : null;

    this.post.render({
      scene: hdr,
      screen: screenTex,
      depth: this.sceneRT.depthTexture,
      camera: res.world ? { near: res.world.camera.near, far: res.world.camera.far } : null,
      overlay: hasOverlay ? this.overlayTex : null,
      params,
      frame,
      seed: hashString(shot.id) & 0xffff,
      band: { top: layout.visibleTop, bottom: layout.visibleBottom },
    });
    return this.fence(t0, { frame, shot: shot.id, program: def.id, createMs });
  }

  /** A 1-pixel readback waits for the GPU so the timing is honest. */
  private fence(t0: number, s: Omit<FrameStats, 'renderMs'>): FrameStats {
    const px = new Uint8Array(4);
    const gl = this.renderer.getContext();
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    this.lastStats = { ...s, renderMs: performance.now() - t0 };
    return this.lastStats;
  }

  /** RGBA8 of the final frame, rows top-to-bottom (WebGL reads bottom-up; we flip). */
  readPixels(): Uint8Array {
    const gl = this.renderer.getContext();
    const w = this.width;
    const h = this.height;
    const raw = new Uint8Array(w * h * 4);
    this.renderer.setRenderTarget(null);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, raw);
    const out = new Uint8Array(w * h * 4);
    const row = w * 4;
    for (let y = 0; y < h; y++) out.set(raw.subarray((h - 1 - y) * row, (h - y) * row), y * row);
    return out;
  }

  dispose(): void {
    for (const { prog } of this.programs.values()) prog.dispose();
    this.programs.clear();
    this.post.dispose();
    this.sceneRT.dispose();
    this.screenRT?.dispose();
    this.overlayTex.dispose();
    this.renderer.dispose();
  }
}
