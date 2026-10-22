// Order-dependence probe: render one frame on two pages after different warm-up frames and report
// the box where they differ (check-determinism says *whether*, this says *where*).
//   bun scripts/diff-order.ts [--film <dir>] --frame 300 [--warmA 260] [--warm 100,500,30] [--out <dir>]
// --out defaults to <film>/out/diff-order; both renders are written there as PNG.
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { OUT } from './lib/film';
import { FILM_FRAMES } from '../src/lib/time';
import { openSession, parseArgs, writePng } from './lib/session';
const args = parseArgs();
const n = Number(args.frame ?? Math.floor(FILM_FRAMES / 2));
const warm = String(args.warm ?? `${Math.floor(FILM_FRAMES / 4)},${Math.floor((3 * FILM_FRAMES) / 4)},30`).split(',').map(Number);
const warmA = typeof args.warmA === 'string' ? args.warmA.split(',').map(Number) : [];
const out = typeof args.out === 'string' ? resolve(args.out) : resolve(OUT, 'diff-order');
mkdirSync(out, { recursive: true });
let pending: ((d: Uint8Array) => void) | null = null;
const W = 1920, H = 1080;
const s = await openSession({ pages: 2, width: W, height: H, onFrame: (_j, _n, d) => { pending?.(d); pending = null; } });
const grab = async (page: number, f: number) => { const p = new Promise<Uint8Array>((r) => (pending = r)); await s.pages[page].eval(`film.postFrame('${s.server.url}/frame/x/${f}', ${f})`); return p; };
try {
  for (const f of warmA) await grab(0, f);
  const a = await grab(0, n);
  for (const f of warm) await grab(1, f);
  const b = await grab(1, n);
  let x0 = W, y0 = H, x1 = -1, y1 = -1, cnt = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const k = (y * W + x) * 4; if (a[k] !== b[k] || a[k + 1] !== b[k + 1] || a[k + 2] !== b[k + 2]) { cnt++; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); } }
  console.log(`frame ${n} warm ${warm}: ${cnt} px differ, box x ${x0}..${x1} y ${y0}..${y1}`);
  await writePng(`${out}/a_${n}.png`, a, W, H);
  await writePng(`${out}/b_${n}.png`, b, W, H);
} finally { await s.close(); }
