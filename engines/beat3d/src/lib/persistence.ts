// "回声就是脚印" (the echo is the footprint): one persistence value p in [0,1] (the timeline's
// `persistence` curve) drives both the visual trail decay and the delay feedback of plucks in the
// score. The score must use the same two functions (keep a mirror of them in the film's score).
//
//   half-life h = 0.4 s x 10^(3p)       feedback fb = 0.15 + 0.45 p
//   p = 0    -> h 0.4 s                  fb 0.15  (forgotten almost at once)
//   p = 0.08 -> h 0.70 s                 fb 0.186
//   p = 0.6  -> h 25 s                   fb 0.42
//   p = 1    -> permanent                fb 0.6

import { PERSISTENCE } from './curves';

const BASE_HALF_LIFE_S = 0.4;

/** Half-life in seconds of a light trail / footprint laid down while persistence is p. */
export function trailHalfLife(p: number): number {
  const q = Math.min(Math.max(p, 0), 1);
  if (q >= 0.999) return Infinity;
  return BASE_HALF_LIFE_S * Math.pow(10, 3 * q);
}

/** Remaining brightness (0..1) of a mark laid at persistence p, `ageSeconds` later. */
export function trailRemaining(ageSeconds: number, p: number): number {
  const hl = trailHalfLife(p);
  if (!Number.isFinite(hl) || ageSeconds <= 0) return 1;
  return Math.pow(0.5, ageSeconds / hl);
}

/** Pluck delay feedback gain for persistence p (kept in sync with music/score). */
export function delayFeedback(p: number): number {
  const q = Math.min(Math.max(p, 0), 1);
  return 0.15 + 0.45 * q;
}

/** Persistence in effect at a film frame. A mark keeps the p of the moment it was laid. */
export const persistenceAt = (frame: number): number => PERSISTENCE.at(frame);
