// 页面入口。scripts/build.ts 把它和片子的镜头（`@film` 模块，films/<name>/scenes/index.ts）一起打成
// film.js，public/index.html 加载它。页面给渲染脚本和预览器暴露 window.film：
//   await film.init({ width, height, shot?, skeleton?, lab? })
//   await film.renderFrame(n)    // n: absolute film frame, or shot-local if init got a shot
//   film.totalFrames
//   await film.postFrame(url, n) // renderFrame + readPixels + POST raw RGBA to the local server

import filmDef from '@film';
import { FilmEngine, type FrameStats } from './core/engine';
import { useFilm, useLab } from './core/registry';
import { CLOCK, FILM_FRAMES, SHOTS, shotById } from './lib/time';
import type { PostOverrides } from './post/params';

useFilm(filmDef);

interface InitOptions {
  width: number;
  height: number;
  shot?: string;
  skeleton?: boolean;
  /** Style tests: the film's lab programs replace the shots they claim. */
  lab?: boolean;
  /** Put the canvas in this element (viewer); otherwise appended to <body>. */
  mount?: HTMLElement;
  /** Post overrides applied after everything else (look development, tests). */
  post?: PostOverrides;
}

let engine: FilmEngine | null = null;
let offset = 0;

const film = {
  totalFrames: FILM_FRAMES,
  framesPerBeat: CLOCK.framesPerBeat,
  shots: SHOTS.map((s) => ({ id: s.id, zh: s.zh, from: s.from, to: s.to, fromFrame: s.fromFrame, toFrame: s.toFrame })),
  clockSource: CLOCK.source,
  canvas: null as HTMLCanvasElement | null,

  async init(o: InitOptions): Promise<{ width: number; height: number; totalFrames: number; offset: number }> {
    engine?.dispose();
    const canvas = document.createElement('canvas');
    canvas.style.width = '100%';
    canvas.style.height = 'auto';
    canvas.style.display = 'block';
    (o.mount ?? document.body).appendChild(canvas);
    this.canvas?.remove();
    this.canvas = canvas;
    useLab(!!o.lab);
    engine = new FilmEngine(canvas, o.width, o.height, { skeleton: !!o.skeleton, postOverride: o.post });
    await engine.init();
    if (o.shot) {
      const s = shotById(o.shot);
      offset = s.fromFrame;
      this.totalFrames = s.duration;
    } else {
      offset = 0;
      this.totalFrames = FILM_FRAMES;
    }
    return { width: o.width, height: o.height, totalFrames: this.totalFrames, offset };
  },

  async renderFrame(n: number): Promise<FrameStats> {
    if (!engine) throw new Error('film.init() first');
    return engine.renderFrame(n + offset);
  },

  /** Render frame n, read it back (top-down RGBA8) and POST it. Returns timings. */
  async postFrame(url: string, n: number): Promise<{ renderMs: number; readMs: number; postMs: number; createMs: number; shot: string }> {
    if (!engine) throw new Error('film.init() first');
    const st = await engine.renderFrame(n + offset);
    const t1 = performance.now();
    const px = engine.readPixels();
    const t2 = performance.now();
    const r = await fetch(url, { method: 'POST', body: px as unknown as BodyInit, headers: { 'content-type': 'application/octet-stream' } });
    if (!r.ok) throw new Error(`POST ${url} -> ${r.status}`);
    const t3 = performance.now();
    return { renderMs: st.renderMs, readMs: t2 - t1, postMs: t3 - t2, createMs: st.createMs, shot: st.shot };
  },

  /** Render [from, to) in order, POSTing each frame to `${base}/frame/${job}/${n}`. Returns totals. */
  async renderRange(base: string, job: string, from: number, to: number): Promise<{ frames: number; renderMs: number; readMs: number; postMs: number; createMs: number; maxMs: number }> {
    const acc = { frames: 0, renderMs: 0, readMs: 0, postMs: 0, createMs: 0, maxMs: 0 };
    for (let n = from; n < to; n++) {
      const t = await this.postFrame(`${base}/frame/${job}/${n}`, n);
      acc.frames++;
      acc.renderMs += t.renderMs;
      acc.readMs += t.readMs;
      acc.postMs += t.postMs;
      acc.createMs += t.createMs;
      acc.maxMs = Math.max(acc.maxMs, t.renderMs);
      if ((n - from) % 96 === 95) console.info(`progress ${job} ${n - from + 1}/${to - from}`);
    }
    return acc;
  },

  frameAt(pos: string): number {
    return CLOCK.f(pos);
  },
  label(n: number): string {
    return CLOCK.label(n + offset);
  },
};

declare global {
  interface Window {
    film: typeof film;
  }
}
window.film = film;
