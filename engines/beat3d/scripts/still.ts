// Render stills as PNG.
//   bun scripts/still.ts [--film <dir>] --at 6.1,13.4 [--size 1080|4k|540] [--out <dir>] [--frames 400,401]
//        [--skeleton] [--lab] [--prefix name] [--post '<json>'] [--sheet <file.png> [--cols 3]]
// --out defaults to <film>/out/stills.
// --sheet also writes a contact sheet: every still at half size, cropped to its letterbox band, in the
// order given, each labelled with its position.
// Each still is rendered twice in the same page: the first render builds the shot program (cold),
// the second is the steady-state cost that a full render pays per frame (warm). Both are logged
// to <out>/timings.json.

import { OUT, selectionLine } from './lib/film';
import { mkdirSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { CLOCK } from '../src/lib/time';
import { layoutAt } from '../src/lib/layout';
import { fmtMs, openSession, parseArgs, sizeOf, writePng } from './lib/session';
import type { Page } from './lib/chrome';

const args = parseArgs();
const size = sizeOf(args.size);
const outDir = typeof args.out === 'string' ? resolve(args.out) : resolve(OUT, 'stills');
mkdirSync(outDir, { recursive: true });

const targets: { label: string; frame: number }[] = [];
if (typeof args.at === 'string') for (const p of args.at.split(',')) targets.push({ label: p.trim(), frame: CLOCK.f(p.trim()) });
if (typeof args.frames === 'string') for (const f of args.frames.split(',')) targets.push({ label: `f${f.trim()}`, frame: Number(f) });
if (!targets.length) {
  console.error('usage: bun scripts/still.ts [--film <dir>] --at 6.1,13.4 [--size 1080|4k] [--out <dir>]');
  process.exit(2);
}

console.log(selectionLine());
let pending: { resolve: (d: Uint8Array) => void } | null = null;
const t0 = performance.now();
const session = await openSession({
  pages: 1,
  width: size.width,
  height: size.height,
  skeleton: !!args.skeleton,
  lab: !!args.lab,
  post: typeof args.post === 'string' ? JSON.parse(args.post) : undefined,
  verbose: !!args.verbose,
  onFrame: (_job, _n, data) => {
    pending?.resolve(data);
    pending = null;
  },
});
const startupMs = performance.now() - t0;
const page = session.pages[0];
const prefix = typeof args.prefix === 'string' ? `${args.prefix}_` : '';
const results: Record<string, unknown>[] = [];
const sheetFile = typeof args.sheet === 'string' ? resolve(args.sheet) : null;
const sheetTiles: { label: string; frame: number; data: Uint8Array }[] = [];

try {
  for (const t of targets) {
    const cold = await page.eval<{ renderMs: number; createMs: number; shot: string }>(`film.renderFrame(${t.frame})`);
    const got = new Promise<Uint8Array>((res) => (pending = { resolve: res }));
    const warm = await page.eval<{ renderMs: number; readMs: number; postMs: number; createMs: number; shot: string }>(
      `film.postFrame('${session.server.url}/frame/still/${t.frame}', ${t.frame})`,
    );
    const data = await got;
    const file = resolve(outDir, `${prefix}${t.label}_${size.name === '4k' ? '2160p' : size.name + 'p'}.png`);
    const te = performance.now();
    await writePng(file, data, size.width, size.height);
    const encMs = performance.now() - te;
    const r = {
      at: t.label,
      frame: t.frame,
      shot: warm.shot,
      size: `${size.width}x${size.height}`,
      coldMs: Math.round(cold.renderMs),
      programCreateMs: Math.round(cold.createMs),
      warmRenderMs: Math.round(warm.renderMs),
      readbackMs: Math.round(warm.readMs),
      transferMs: Math.round(warm.postMs),
      pngMs: Math.round(encMs),
      file,
    };
    results.push(r);
    if (sheetFile) sheetTiles.push({ label: t.label, frame: t.frame, data });
    console.log(`${t.label.padEnd(6)} f${String(t.frame).padEnd(5)} ${warm.shot.padEnd(5)} warm ${fmtMs(warm.renderMs).padStart(8)}  cold ${fmtMs(cold.renderMs).padStart(8)} (create ${fmtMs(cold.createMs)})  read ${fmtMs(warm.readMs)}  post ${fmtMs(warm.postMs)}  -> ${file}`);
  }
  if (sheetFile) await writeSheet(page, sheetFile, sheetTiles, Number(args.cols ?? 3));
} finally {
  await session.close();
}

/** Half-size tiles cropped to the letterbox band, a label strip above each, `cols` per row. */
async function writeSheet(pg: Page, file: string, tiles: { label: string; frame: number; data: Uint8Array }[], cols: number): Promise<void> {
  const W = size.width;
  const tw = W / 2;
  const LH = 30;
  const bands = tiles.map((t) => layoutAt(W, size.height, t.frame));
  const th = Math.max(...bands.map((b) => (b.visibleBottom - b.visibleTop) / 2)) | 0;
  const rows = Math.ceil(tiles.length / cols);
  const SW = cols * tw;
  const SH = rows * (th + LH);
  const out = new Uint8Array(SW * SH * 4);
  for (let i = 3; i < out.length; i += 4) out[i] = 255;
  for (let k = 0; k < tiles.length; k++) {
    const { data, label } = tiles[k];
    const b = bands[k];
    const x0 = (k % cols) * tw;
    const y0 = Math.floor(k / cols) * (th + LH);
    // Label, drawn by the page in the film's mono face.
    const strip = await pg.eval<string>(`(() => {
      const c = document.createElement('canvas'); c.width = ${tw}; c.height = ${LH};
      const g = c.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, c.width, c.height);
      g.fillStyle = '#8c8781'; g.font = '400 15px "B3D Mono"'; g.textBaseline = 'middle';
      g.fillText(${JSON.stringify(label)}, 12, ${LH / 2});
      const d = g.getImageData(0, 0, c.width, c.height).data; let s = '';
      for (let i = 0; i < d.length; i++) s += String.fromCharCode(d[i]);
      return btoa(s);
    })()`);
    const lab = Uint8Array.from(atob(strip), (ch) => ch.charCodeAt(0));
    for (let y = 0; y < LH; y++) out.set(lab.subarray(y * tw * 4, (y + 1) * tw * 4), ((y0 + y) * SW + x0) * 4);
    // 2x2 box filter of the visible band.
    for (let y = 0; y < th; y++) {
      const sy = b.visibleTop + 2 * y;
      if (sy + 1 >= b.visibleBottom) break;
      for (let x = 0; x < tw; x++) {
        const o = ((y0 + LH + y) * SW + x0 + x) * 4;
        for (let c = 0; c < 3; c++) {
          const i0 = (sy * W + 2 * x) * 4 + c;
          const i1 = ((sy + 1) * W + 2 * x) * 4 + c;
          out[o + c] = (data[i0] + data[i0 + 4] + data[i1] + data[i1 + 4] + 2) >> 2;
        }
      }
    }
  }
  await writePng(file, out, SW, SH);
  console.log(`contact sheet ${cols}x${rows} -> ${file}`);
}

const timingFile = resolve(outDir, 'timings.json');
const prev = existsSync(timingFile) ? JSON.parse(readFileSync(timingFile, 'utf8')) : { runs: [] };
prev.runs.push({ date: new Date().toISOString(), size: size.name, startupMs: Math.round(startupMs), stills: results });
writeFileSync(timingFile, JSON.stringify(prev, null, 2));
console.log(`startup (bundle + chrome + page init) ${fmtMs(startupMs)}; timings -> ${timingFile}`);
