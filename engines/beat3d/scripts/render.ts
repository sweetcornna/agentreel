// Render a range of the timeline to H.264 (default: all of it, picture and black tail).
//   bun scripts/render.ts [--film <dir>] [--from <bar.beat>] [--to <bar.beat>] [--size 1080|4k|540]
//        [--out <file.mp4> | --out-name <file.mp4>] [--workers 3] [--skeleton] [--lab] [--crf N]
//        [--preset medium|slow] [--no-audio] [--post '<json>'] [--score <wav>] [--events <json>]
// Output defaults to <film>/out/<film>-<size>.mp4 (--out-name: a file name inside <film>/out).
// Each worker is a page in one headless Chrome; it renders a contiguous slice of frames and streams
// raw RGBA to a Bun server, which pipes them in order into its own ffmpeg (libx264, yuv420p, BT.709).
// The slices are joined losslessly with the concat demuxer; the film's score (<film>/out/music/score.wav,
// see scripts/lib/film.ts) is muxed if present (unless --no-audio). Every run appends its timings to
// <film>/out/render-log.json.

import { FILM_NAME, MUSIC, OUT, TIMELINE_PATH, selectionLine } from './lib/film';
import { existsSync, mkdirSync, readFileSync, renameSync, rmdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { CLOCK, FILM_FRAMES } from '../src/lib/time';
import { H264Writer, fmtMs, openSession, parseArgs, sizeOf } from './lib/session';
import { ffmpeg } from './lib/tools';

const args = parseArgs();
const size = sizeOf(args.size);
const from = typeof args.from === 'string' ? CLOCK.f(args.from) : 0;
const to = typeof args.to === 'string' ? CLOCK.f(args.to) : FILM_FRAMES;
const out = typeof args.out === 'string' ? resolve(args.out) : resolve(OUT, typeof args['out-name'] === 'string' ? args['out-name'] : `${FILM_NAME}-${size.name === '4k' ? '2160' : size.name}p.mp4`);
const workers = Math.max(1, Math.min(Number(args.workers ?? 3), to - from));
const crf = Number(args.crf ?? (size.name === '4k' ? 14 : 16));
const preset = typeof args.preset === 'string' ? args.preset : 'slow';
if (to <= from) throw new Error(`empty range ${from}..${to}`);

const tmp = resolve(OUT, 'render-tmp', `${process.pid}-${Date.now()}`);
mkdirSync(tmp, { recursive: true });
mkdirSync(dirname(out), { recursive: true });

const per = Math.ceil((to - from) / workers);
const slices = Array.from({ length: workers }, (_, i) => ({ from: from + i * per, to: Math.min(to, from + (i + 1) * per) })).filter((s) => s.to > s.from);
const writers = slices.map((_, i) => new H264Writer(resolve(tmp, `seg_${String(i).padStart(3, '0')}.mp4`), size.width, size.height, crf, preset));

console.log(selectionLine());
console.log(`render ${CLOCK.label(from)}..${CLOCK.label(to)} = frames ${from}..${to - 1} (${to - from}) at ${size.width}x${size.height}, ${slices.length} workers, crf ${crf} ${preset}`);
const t0 = performance.now();
const session = await openSession({
  pages: slices.length,
  width: size.width,
  height: size.height,
  skeleton: !!args.skeleton,
  lab: !!args.lab,
  post: typeof args.post === 'string' ? JSON.parse(args.post) : undefined,
  onFrame: (job, _n, data) => writers[Number(job)].write(data),
});
const tReady = performance.now();
let totals: { frames: number; renderMs: number; readMs: number; postMs: number; createMs: number; maxMs: number }[] = [];
try {
  totals = await Promise.all(
    slices.map((s, i) => session.pages[i].eval<(typeof totals)[number]>(`film.renderRange('${session.server.url}', '${i}', ${s.from}, ${s.to})`)),
  );
} finally {
  await session.close();
}
for (const w of writers) await w.close();
const tFrames = performance.now();

// Lossless join.
const list = resolve(tmp, 'list.txt');
writeFileSync(list, writers.map((w) => `file '${w.path}'`).join('\n'));
const joined = resolve(tmp, 'joined.mp4');
const run = async (cmd: string[]): Promise<void> => {
  const p = Bun.spawn(cmd, { stdout: 'ignore', stderr: 'pipe' });
  if ((await p.exited) !== 0) throw new Error(`${cmd.join(' ')}\n${await new Response(p.stderr).text()}`);
};
await run([ffmpeg(), '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', '-movflags', '+faststart', joined]);

const score = MUSIC.score;
const muxed = existsSync(score) && !args['no-audio'] ? score : null;
if (muxed) {
  const start = from / CLOCK.fps;
  const dur = (to - from) / CLOCK.fps;
  // No -shortest: the AAC track can come out a few samples short of the score and would cut the last frame.
  await run([ffmpeg(), '-v', 'error', '-y', '-i', joined, '-ss', start.toFixed(4), '-t', dur.toFixed(4), '-i', score, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '320k', '-movflags', '+faststart', out]);
  console.log(`muxed audio from ${score} at ${start.toFixed(3)} s`);
} else {
  renameSync(joined, out);
}
rmSync(tmp, { recursive: true, force: true });
try {
  rmdirSync(resolve(OUT, 'render-tmp'));
} catch {
  /* other renders still running */
}
const tEnd = performance.now();

const n = totals.reduce((a, t) => a + t.frames, 0);
const sum = (k: keyof (typeof totals)[number]) => totals.reduce((a, t) => a + t[k], 0);
const report = {
  date: new Date().toISOString(),
  timeline: TIMELINE_PATH,
  range: [CLOCK.label(from), CLOCK.label(to)],
  frames: n,
  size: `${size.width}x${size.height}`,
  workers: slices.length,
  skeleton: !!args.skeleton,
  lab: !!args.lab,
  audio: muxed,
  musicEvents: session.musicEvents,
  wallSeconds: +((tEnd - t0) / 1000).toFixed(2),
  startupSeconds: +((tReady - t0) / 1000).toFixed(2),
  framesSeconds: +((tFrames - tReady) / 1000).toFixed(2),
  wallMsPerFrame: +((tFrames - tReady) / n).toFixed(1),
  avgRenderMs: +(sum('renderMs') / n).toFixed(1),
  avgReadbackMs: +(sum('readMs') / n).toFixed(1),
  avgTransferMs: +(sum('postMs') / n).toFixed(1),
  maxRenderMs: +Math.max(...totals.map((t) => t.maxMs)).toFixed(1),
  programCreateMs: +sum('createMs').toFixed(0),
  out,
};
const logFile = resolve(OUT, 'render-log.json');
const log = existsSync(logFile) ? JSON.parse(readFileSync(logFile, 'utf8')) : [];
log.push(report);
writeFileSync(logFile, JSON.stringify(log, null, 2));
console.log(`done: ${n} frames in ${fmtMs(tEnd - t0)} (startup ${fmtMs(tReady - t0)}), ${report.wallMsPerFrame} ms/frame wall, render ${report.avgRenderMs} ms, readback ${report.avgReadbackMs} ms, transfer ${report.avgTransferMs} ms -> ${out}`);
