// Walkers: lights that move like people. Footfalls land on the music (quarter notes, eighths, any
// beats per step), with a seeded human sway; between footfalls the light rises and sinks, lowest at
// the moment of the step.

import { CLOCK, posToBeat, type Pos } from '../lib/time';
import { Rng } from '../lib/rng';

export interface StepPlan {
  /** Footfall frames (fractional), ascending. */
  frames: number[];
}

/**
 * Footfalls every `beatsPerStep` beats from `from` (inclusive) to `to` (exclusive), on the beat clock
 * (so they follow events.json when it exists). `sway` is the max human offset in frames.
 */
export function planSteps(key: string, from: Pos, to: Pos, beatsPerStep = 1, sway = 1): StepPlan {
  const rng = new Rng(`${key}/steps`);
  const b0 = posToBeat(from);
  const b1 = posToBeat(to);
  const frames: number[] = [];
  for (let b = b0; b < b1 - 1e-6; b += beatsPerStep) frames.push(CLOCK.beatToFrame(b) + (rng.next() * 2 - 1) * sway);
  return { frames };
}

/** Index of the last footfall at or before `frame` (-1 if none). */
export function lastStep(plan: StepPlan, frame: number): number {
  let lo = -1;
  let hi = plan.frames.length - 1;
  while (lo < hi) {
    const m = (lo + hi + 1) >> 1;
    if (plan.frames[m] <= frame) lo = m;
    else hi = m - 1;
  }
  return lo;
}

/** Vertical bob (metres, <= 0): sinks to -depth at each footfall, back to 0 halfway between. */
export function bob(plan: StepPlan, frame: number, depth: number): number {
  const i = lastStep(plan, frame);
  if (i < 0 || i >= plan.frames.length - 1) {
    if (i < 0) return 0;
    const since = frame - plan.frames[i];
    return -depth * Math.exp(-since / 6);
  }
  const a = plan.frames[i];
  const b = plan.frames[i + 1];
  const ph = (frame - a) / (b - a);
  return -depth * (0.5 + 0.5 * Math.cos(2 * Math.PI * ph));
}
