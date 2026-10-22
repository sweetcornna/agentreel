// Local HTTP server for the render page and the viewer. Static files from public/ and the build
// directory, the film's own assets (<film>/assets, served at /film-assets/: textures, recordings),
// plus POST /frame/<job>/<n> which hands raw RGBA frames to a callback.

import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { DIST, build } from '../build';
import { FILM_DIR } from './film';

export const ROOT = resolve(import.meta.dir, '../..');
/** The film's assets, served at /film-assets/. */
export const FILM_ASSETS = resolve(FILM_DIR, 'assets');

export interface ServerOptions {
  port?: number;
  /** Rebuild film.js whenever it is requested (viewer dev loop). */
  rebuildOnRequest?: boolean;
  /** Directory holding film.js and its assets (default <film>/out/.build/dist). */
  distDir?: string;
  onFrame?: (job: string, n: number, data: Uint8Array) => Promise<void> | void;
}

export interface RunningServer {
  url: string;
  stop(): void;
}

const TYPES: Record<string, string> = {
  html: 'text/html; charset=utf-8',
  js: 'text/javascript; charset=utf-8',
  map: 'application/json',
  json: 'application/json',
  woff: 'font/woff',
  woff2: 'font/woff2',
  txt: 'text/plain; charset=utf-8',
  png: 'image/png',
  jpg: 'image/jpeg',
  webp: 'image/webp',
  otf: 'font/otf',
  ttf: 'font/ttf',
};

export function startServer(opts: ServerOptions = {}): RunningServer {
  let building: Promise<unknown> | null = null;
  const dirs = [resolve(ROOT, 'public'), opts.distDir ?? DIST];
  const server = Bun.serve({
    port: opts.port ?? 0,
    hostname: '127.0.0.1',
    maxRequestBodySize: 64 * 1024 * 1024,
    async fetch(req) {
      const url = new URL(req.url);
      const path = decodeURIComponent(url.pathname);
      if (req.method === 'POST' && path.startsWith('/frame/')) {
        const [, , job, nStr] = path.split('/');
        const data = new Uint8Array(await req.arrayBuffer());
        try {
          await opts.onFrame?.(job, Number(nStr), data);
        } catch (e) {
          console.error(e);
          return new Response(String(e), { status: 500 });
        }
        return new Response('ok');
      }
      if (path === '/film.js' && opts.rebuildOnRequest) {
        building ??= build().finally(() => (building = null));
        try {
          await building;
        } catch (e) {
          return new Response(`console.error(${JSON.stringify(String(e))})`, { headers: { 'content-type': TYPES.js } });
        }
      }
      const rel = path === '/' ? 'index.html' : path.slice(1);
      if (rel.includes('..')) return new Response('no', { status: 403 });
      if (rel.startsWith('film-assets/')) {
        const f = resolve(FILM_ASSETS, rel.slice('film-assets/'.length));
        if (!existsSync(f)) return new Response('not found', { status: 404 });
        return new Response(Bun.file(f), { headers: { 'content-type': TYPES[rel.split('.').pop() ?? ''] ?? 'application/octet-stream', 'cache-control': 'no-store' } });
      }
      for (const dir of dirs) {
        const f = resolve(dir, rel);
        if (existsSync(f)) {
          const ext = rel.split('.').pop() ?? '';
          return new Response(Bun.file(f), { headers: { 'content-type': TYPES[ext] ?? 'application/octet-stream', 'cache-control': 'no-store' } });
        }
      }
      return new Response('not found', { status: 404 });
    },
  });
  return { url: `http://127.0.0.1:${server.port}`, stop: () => server.stop(true) };
}
