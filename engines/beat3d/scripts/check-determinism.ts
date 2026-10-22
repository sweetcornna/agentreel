// Determinism self-check: render the same frames twice in two independent pages (different render
// order, the second page warmed up on other frames first) and compare every byte.
//   bun scripts/check-determinism.ts [--film <dir>] [--at 3.1+2f,6.1] [--size 1080]
// Default frames: two per shot (its first beat and a frame past its middle), so every program is checked.
import { selectionLine } from './lib/film';
import { CLOCK, SHOTS } from '../src/lib/time';
import { openSession, parseArgs, sizeOf } from './lib/session';

const args = parseArgs();
const size = sizeOf(args.size ?? '1080');
const positions =
  typeof args.at === 'string'
    ? args.at.split(',').map((p) => p.trim())
    : SHOTS.flatMap((s) => [CLOCK.label(Math.min(s.fromFrame + CLOCK.framesPerBeat, s.toFrame - 1)), CLOCK.label(Math.floor((s.fromFrame + s.toFrame) / 2) + 3)]);
const frames = positions.map((p) => CLOCK.f(p));

console.log(selectionLine());
const got = new Map<string, Uint8Array>();
let pending: ((d: Uint8Array) => void) | null = null;
const session = await openSession({ pages: 2, width: size.width, height: size.height, onFrame: (_j, _n, d) => { pending?.(d); pending = null; } });
const grab = async (page: number, n: number): Promise<Uint8Array> => {
  const p = new Promise<Uint8Array>((r) => (pending = r));
  await session.pages[page].eval(`film.postFrame('${session.server.url}/frame/check/${n}', ${n})`);
  return p;
};
let bad = 0;
try {
  for (const n of frames) got.set(`a${n}`, await grab(0, n));
  // Page B: warm up on unrelated frames, then visit the targets in reverse order.
  for (const n of [...SHOTS].reverse().map((s) => s.fromFrame + 5)) await grab(1, n);
  for (const n of [...frames].reverse()) got.set(`b${n}`, await grab(1, n));
} finally {
  await session.close();
}
for (let i = 0; i < frames.length; i++) {
  const n = frames[i];
  const a = got.get(`a${n}`)!;
  const b = got.get(`b${n}`)!;
  let diff = 0;
  let maxd = 0;
  for (let k = 0; k < a.length; k++) {
    const d = Math.abs(a[k] - b[k]);
    if (d) {
      diff++;
      maxd = Math.max(maxd, d);
    }
  }
  if (diff) bad++;
  console.log(`${positions[i].padEnd(7)} f${String(n).padEnd(5)} ${diff === 0 ? 'identical' : `DIFFERS: ${diff} bytes, max ${maxd}`}`);
}
console.log(bad ? `FAIL: ${bad} frame(s) not deterministic` : `OK: ${frames.length} frames bit-identical across pages and render order`);
process.exit(bad ? 1 : 0);
