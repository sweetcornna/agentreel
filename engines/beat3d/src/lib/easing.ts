// Scalar helpers shared by every shot. All are pure functions of their arguments.

export const clamp = (x: number, lo = 0, hi = 1): number => (x < lo ? lo : x > hi ? hi : x);
export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
export const invLerp = (a: number, b: number, x: number): number => (b === a ? 0 : (x - a) / (b - a));
/** Map x from [a0,a1] to [b0,b1], clamped. */
export const remap = (x: number, a0: number, a1: number, b0 = 0, b1 = 1): number =>
  lerp(b0, b1, clamp(invLerp(a0, a1, x)));

export const smoothstep = (e0: number, e1: number, x: number): number => {
  const t = clamp(invLerp(e0, e1, x));
  return t * t * (3 - 2 * t);
};
export const smootherstep = (e0: number, e1: number, x: number): number => {
  const t = clamp(invLerp(e0, e1, x));
  return t * t * t * (t * (t * 6 - 15) + 10);
};

export const easeInCubic = (t: number): number => t * t * t;
export const easeOutCubic = (t: number): number => 1 - Math.pow(1 - t, 3);
export const easeInOutCubic = (t: number): number =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
export const easeOutExpo = (t: number): number => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t));
export const easeInExpo = (t: number): number => (t <= 0 ? 0 : Math.pow(2, 10 * t - 10));
export const easeInOutSine = (t: number): number => -(Math.cos(Math.PI * t) - 1) / 2;

export type EasingName = 'linear' | 'smooth' | 'easeInOutCubic' | 'easeOutCubic' | 'easeInCubic' | 'easeOutExpo' | 'step';

export const EASINGS: Record<EasingName, (t: number) => number> = {
  linear: (t) => t,
  smooth: (t) => t * t * (3 - 2 * t),
  easeInOutCubic,
  easeOutCubic,
  easeInCubic,
  easeOutExpo,
  step: (t) => (t < 1 ? 0 : 1),
};

/** Fade envelope: 0 before `a`, ramps to 1 over `inLen`, holds, ramps to 0 over `outLen` ending at `b`. */
export const envelope = (x: number, a: number, b: number, inLen: number, outLen: number): number => {
  if (x < a || x >= b) return 0;
  const i = inLen > 0 ? clamp((x - a) / inLen) : 1;
  const o = outLen > 0 ? clamp((b - x) / outLen) : 1;
  return Math.min(i, o);
};

/** CSS cubic-bezier(x1, y1, x2, y2) at progress t in [0, 1] (motion curves of a design system). */
export function cubicBezier(x1: number, y1: number, x2: number, y2: number, t: number): number {
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  const bx = (u: number) => 3 * x1 * u * (1 - u) * (1 - u) + 3 * x2 * u * u * (1 - u) + u * u * u;
  const by = (u: number) => 3 * y1 * u * (1 - u) * (1 - u) + 3 * y2 * u * u * (1 - u) + u * u * u;
  // x(u) is monotone for 0 <= x1, x2 <= 1: bisection is exact enough and branch-free of NaNs.
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    if (bx(mid) < t) lo = mid;
    else hi = mid;
  }
  return by((lo + hi) / 2);
}

/** The standard ease for big things, cubic-bezier(.2, .8, .2, 1). */
export const ease = (t: number): number => cubicBezier(0.2, 0.8, 0.2, 1, t);
/** A small spring for small things, cubic-bezier(.34, 1.56, .64, 1): about 10 % overshoot. */
export const spring = (t: number): number => cubicBezier(0.34, 1.56, 0.64, 1, t);
