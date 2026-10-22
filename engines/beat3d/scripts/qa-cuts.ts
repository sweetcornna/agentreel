// Find picture jumps in a rendered video and check them against the timeline's cuts.
//   bun scripts/qa-cuts.ts [--film <dir>] --in <video.mp4> [--start <bar.beat>] [--threshold 1] [--ratio 3]
// The video's first frame is --start (default 1.1). Expected cuts are every shot boundary of the
// timeline (including the end of the last shot: the cut to black), every montage cut inside a shot
// (shots[].cuts) and every hard sync point.
// A jump is a frame whose mean absolute luma difference from the previous frame is at least
// --threshold (0..255) and at least --ratio times the median of its ten neighbours.
// Jumps off a cut are pops (exit code 1). Cuts without a jump are listed as soft (match cuts, holds).

import './lib/film';
import { spawnSync } from 'node:child_process';
import { CLOCK, SHOTS, TIMELINE, syncStart } from '../src/lib/time';
import { parseArgs } from './lib/session';
import { ffmpeg } from './lib/tools';

const args = parseArgs();
if (typeof args.in !== 'string') throw new Error('usage: bun scripts/qa-cuts.ts --in <video.mp4> [--start <bar.beat>]');
const start = typeof args.start === 'string' ? CLOCK.f(args.start) : 0;
const threshold = Number(args.threshold ?? 1);
const ratio = Number(args.ratio ?? 3);

const ff = spawnSync(
  ffmpeg(),
  ['-v', 'error', '-i', args.in, '-vf', 'format=gray,tblend=all_mode=difference,signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-', '-f', 'null', '-'],
  { encoding: 'utf8', maxBuffer: 64 << 20 },
);
if (ff.status !== 0) throw new Error(`ffmpeg failed: ${ff.stderr}`);
// tblend emits one value per adjacent pair: value i is the step from frame i to frame i+1.
const diffs = [...ff.stdout.matchAll(/YAVG=([\d.]+)/g)].map((m) => Number(m[1]));
const frames = diffs.length + 1;
const end = start + frames;

const expected = new Map<number, string>();
const mark = (f: number, what: string) => {
  if (f > start && f < end && !expected.has(f)) expected.set(f, what);
};
for (const s of SHOTS) {
  mark(s.fromFrame, s.id);
  mark(s.toFrame, `end of ${s.id}`);
  for (const c of s.cuts ?? []) mark(CLOCK.f(c.at), `${s.id} cut ${c.at}`);
}
for (const s of TIMELINE.sync ?? []) if (s.hard) mark(CLOCK.f(syncStart(s.at)), `${s.id ?? 'sync'} ${s.at}`);

const median = (xs: number[]) => {
  const v = [...xs].sort((a, b) => a - b);
  return v.length ? v[v.length >> 1] : 0;
};
const jumps: { frame: number; d: number; base: number }[] = [];
for (let i = 0; i < diffs.length; i++) {
  const around = [...diffs.slice(Math.max(0, i - 5), i), ...diffs.slice(i + 1, i + 6)];
  const base = median(around);
  if (diffs[i] >= threshold && diffs[i] >= ratio * Math.max(base, 0.5)) jumps.push({ frame: start + i + 1, d: diffs[i], base });
}

const label = (f: number) => `${f} (${CLOCK.label(f)})`;
const hit = new Set<number>();
const pops: string[] = [];
for (const j of jumps) {
  if (expected.has(j.frame)) {
    hit.add(j.frame);
    continue;
  }
  const near = [-2, -1, 1, 2].map((o) => j.frame + o).find((f) => expected.has(f));
  const why = near !== undefined ? `, ${j.frame - near > 0 ? '+' : ''}${j.frame - near} f off the ${expected.get(near)} cut` : '';
  pops.push(`  ${label(j.frame)}  step ${j.d.toFixed(1)} vs ${j.base.toFixed(1)}${why}`);
}
const soft = [...expected].filter(([f]) => !hit.has(f)).map(([f, id]) => `  ${label(f)}  ${id}`);

console.log(`${args.in}: frames ${start}..${end - 1} (${frames}), ${expected.size} cuts expected, ${jumps.length} jumps`);
console.log(`on cut: ${hit.size}`);
if (soft.length) console.log(`soft cuts (no jump):\n${soft.join('\n')}`);
if (pops.length) console.log(`POPS (jump off any cut):\n${pops.join('\n')}`);
else console.log('no pops');
process.exit(pops.length ? 1 : 0);
