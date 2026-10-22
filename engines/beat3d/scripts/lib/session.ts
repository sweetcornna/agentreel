// Shared plumbing for still/render/check scripts: argument parsing, sizes, page setup, ffmpeg.

import { rmSync } from 'node:fs';
import { resolve } from 'node:path';
import { Chrome, type Page } from './chrome';
import { build } from '../build';
import { startServer, type RunningServer } from './server';
import { OUT, TIMELINE_JSON } from './film';
import { ffmpeg } from './tools';

export const SIZES = {
  '1080': [1920, 1080],
  '4k': [3840, 2160],
  '540': [960, 540],
} as const;
export type SizeName = keyof typeof SIZES;

export function parseArgs(argv = process.argv.slice(2)): Record<string, string | true> {
  const out: Record<string, string | true> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next !== undefined && !next.startsWith('--')) {
      out[key] = next;
      i++;
    } else out[key] = true;
  }
  return out;
}

export function sizeOf(name: string | true | undefined): { name: SizeName; width: number; height: number } {
  const n = (typeof name === 'string' ? name.toLowerCase() : '1080').replace('p', '').replace('2160', '4k') as SizeName;
  const s = SIZES[n];
  if (!s) throw new Error(`--size must be 1080 | 4k | 540 (got ${String(name)})`);
  return { name: n, width: s[0], height: s[1] };
}

export interface FilmSession {
  chrome: Chrome;
  pages: Page[];
  server: RunningServer;
  logs: string[];
  /** Score events baked into this session's bundle (null = beat grid only). */
  musicEvents: number | null;
  close(): Promise<void>;
}

export async function openSession(o: {
  pages: number;
  width: number;
  height: number;
  skeleton?: boolean;
  lab?: boolean;
  post?: unknown;
  onFrame?: (job: string, n: number, data: Uint8Array) => Promise<void> | void;
  verbose?: boolean;
}): Promise<FilmSession> {
  // Private build per session: a viewer rebuilding meanwhile must not change what this session renders.
  const distDir = resolve(OUT, '.build', `${process.pid}-${Date.now().toString(36)}`);
  const dropBuild = () => rmSync(distDir, { recursive: true, force: true });
  let musicEvents: number | null = null;
  try {
    ({ musicEvents } = await build(false, distDir));
  } catch (e) {
    dropBuild();
    throw e;
  }
  const server = startServer({ onFrame: o.onFrame, distDir });
  let chrome: Chrome;
  try {
    chrome = await Chrome.launch();
  } catch (e) {
    server.stop();
    dropBuild();
    throw e;
  }
  const logs: string[] = [];
  const pages: Page[] = [];
  try {
    for (let i = 0; i < o.pages; i++) {
      const p = await chrome.newPage(o.width, o.height);
      p.onConsole((type, text) => {
        const line = `[page ${i}] ${type}: ${text}`;
        logs.push(line);
        if (o.verbose || type === 'error' || type === 'exception' || type === 'warning') console.error(line);
      });
      await p.goto(`${server.url}/index.html`);
      await p.eval(`film.init(${JSON.stringify({ width: o.width, height: o.height, skeleton: !!o.skeleton, lab: !!o.lab, post: o.post })})`);
      pages.push(p);
    }
  } catch (e) {
    await chrome.close();
    server.stop();
    dropBuild();
    throw e;
  }
  return {
    chrome,
    pages,
    server,
    logs,
    musicEvents,
    async close() {
      await chrome.close();
      server.stop();
      dropBuild();
    },
  };
}

/** Encode one RGBA8 frame (top-down rows) to PNG with ffmpeg. */
export async function writePng(path: string, data: Uint8Array, width: number, height: number): Promise<void> {
  const p = Bun.spawn([ffmpeg(), '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${width}x${height}`, '-i', '-', '-frames:v', '1', '-pix_fmt', 'rgb24', path], {
    stdin: 'pipe',
    stderr: 'pipe',
  });
  p.stdin.write(data);
  await p.stdin.end();
  const code = await p.exited;
  if (code !== 0) throw new Error(`ffmpeg png failed: ${await new Response(p.stderr).text()}`);
}

/** Streaming H.264 encoder fed with raw RGBA frames. */
export class H264Writer {
  private proc: ReturnType<typeof Bun.spawn>;
  private err = '';
  constructor(
    readonly path: string,
    width: number,
    height: number,
    crf: number,
    preset = 'medium',
  ) {
    this.proc = Bun.spawn(
      [
        ffmpeg(), '-v', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${width}x${height}`, '-framerate', String(TIMELINE_JSON.fps), '-i', '-',
        '-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p',
        '-c:v', 'libx264', '-preset', preset, '-crf', String(crf), '-profile:v', 'high',
        // Dark, grainy gradients: spend bits in flat dark areas and keep the grain (no banding).
        '-x264-params', 'aq-mode=3:aq-strength=0.9:deblock=-1,-1', '-tune', 'grain',
        '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
        '-movflags', '+faststart',
        path,
      ],
      { stdin: 'pipe', stderr: 'pipe' },
    );
    const stderr = this.proc.stderr as ReadableStream;
    (async () => {
      this.err = await new Response(stderr).text();
    })();
  }
  async write(frame: Uint8Array): Promise<void> {
    const sink = this.proc.stdin as import('bun').FileSink;
    sink.write(frame);
    await sink.flush();
  }
  async close(): Promise<void> {
    await (this.proc.stdin as import('bun').FileSink).end();
    const code = await this.proc.exited;
    if (code !== 0) throw new Error(`ffmpeg failed for ${this.path}: ${this.err}`);
  }
}

export const fmtMs = (ms: number): string => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(0)} ms`);
