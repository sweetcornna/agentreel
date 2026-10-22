// Numerical part of the API: interpolate, Easing (with the cubic-bezier solver), spring, measureSpring and random.
//
// Where the code comes from (details and licence texts: ../THIRD_PARTY_NOTICES.md):
//   · the cubic-bezier solver and the Easing functions follow React Native's Animated library (Libraries/Animated/
//     bezier.js and Easing.js, MIT, Copyright (c) Meta Platforms, Inc. and affiliates); React Native's bezier.js is
//     itself a port of bezier-easing (MIT, Copyright (c) 2014 Gaëtan Renaudeau);
//   · interpolate, spring, measureSpring and random are written for this engine from the documented behaviour of
//     the API, not from any other implementation: spring evaluates the closed-form solution of the damped oscillator
//     at t = frame / fps, measureSpring uses an envelope bound to know when to stop looking, random hashes the seed
//     with FNV-1a (public domain) into one step of mulberry32 (public domain).
// Numbers can therefore differ from other implementations of the same API in the last digits (spring), by a frame
// (measureSpring, and so the timing of a spring stretched with durationInFrames or played in reverse) or entirely
// (random gives a different sequence for the same seed).

// ---------------------------------------------------------------- cubic bezier (CSS timing function solver)
// After React Native's Libraries/Animated/bezier.js, a port of bezier-easing (both MIT; ../THIRD_PARTY_NOTICES.md):
// same constants, float32 sample table, Newton steps with a bisection fallback, exact end points.
const NEWTON_ITERATIONS = 4;
const NEWTON_MIN_SLOPE = 0.001;
const SUBDIVISION_PRECISION = 0.0000001;
const SUBDIVISION_MAX_ITERATIONS = 10;
const SPLINE_TABLE_SIZE = 11;
const SAMPLE_STEP = 1 / (SPLINE_TABLE_SIZE - 1);

const coefA = (a1: number, a2: number) => 1 - 3 * a2 + 3 * a1;
const coefB = (a1: number, a2: number) => 3 * a2 - 6 * a1;
const coefC = (a1: number) => 3 * a1;
/** x(t) or y(t) of the curve with control values a1, a2 (end points 0 and 1). */
const calcBezier = (t: number, a1: number, a2: number) => ((coefA(a1, a2) * t + coefB(a1, a2)) * t + coefC(a1)) * t;
const slopeAt = (t: number, a1: number, a2: number) => 3 * coefA(a1, a2) * t * t + 2 * coefB(a1, a2) * t + coefC(a1);

const binarySubdivide = (x: number, lo: number, hi: number, x1: number, x2: number) => {
	let a = lo;
	let b = hi;
	let t = 0;
	let dx = 0;
	let i = 0;
	do {
		t = a + (b - a) / 2;
		dx = calcBezier(t, x1, x2) - x;
		if (dx > 0) b = t;
		else a = t;
	} while (Math.abs(dx) > SUBDIVISION_PRECISION && ++i < SUBDIVISION_MAX_ITERATIONS);
	return t;
};

const newton = (x: number, guess: number, x1: number, x2: number) => {
	let t = guess;
	for (let i = 0; i < NEWTON_ITERATIONS; ++i) {
		const slope = slopeAt(t, x1, x2);
		if (slope === 0) return t;
		t -= (calcBezier(t, x1, x2) - x) / slope;
	}
	return t;
};

export const bezier = (x1: number, y1: number, x2: number, y2: number): ((t: number) => number) => {
	if (!(x1 >= 0 && x1 <= 1 && x2 >= 0 && x2 <= 1)) throw new Error(`Easing.bezier: x1 and x2 must lie in [0, 1], got ${x1}, ${x2}`);
	if (x1 === y1 && x2 === y2) return (t) => t;
	// sample table in float32, as in bezier-easing: it only seeds the Newton step
	const samples = new Float32Array(SPLINE_TABLE_SIZE);
	for (let i = 0; i < SPLINE_TABLE_SIZE; ++i) samples[i] = calcBezier(i * SAMPLE_STEP, x1, x2);
	const tForX = (x: number) => {
		let start = 0;
		let k = 1;
		const last = SPLINE_TABLE_SIZE - 1;
		for (; k !== last && samples[k] <= x; ++k) start += SAMPLE_STEP;
		--k;
		const dist = (x - samples[k]) / (samples[k + 1] - samples[k]);
		const guess = start + dist * SAMPLE_STEP;
		const slope = slopeAt(guess, x1, x2);
		if (slope >= NEWTON_MIN_SLOPE) return newton(x, guess, x1, x2);
		if (slope === 0) return guess;
		return binarySubdivide(x, start, start + SAMPLE_STEP, x1, x2);
	};
	return (t: number) => {
		// exact end points despite floating point
		if (t === 0 || t === 1) return t;
		return calcBezier(tForX(t), y1, y2);
	};
};

// ---------------------------------------------------------------- Easing
// The functions of React Native's Libraries/Animated/Easing.js (MIT; ../THIRD_PARTY_NOTICES.md), same formulas.
export type EasingFunction = (t: number) => number;

let easeCurve: EasingFunction | null = null;

export class Easing {
	static step0(n: number): number {
		return n > 0 ? 1 : 0;
	}

	static step1(n: number): number {
		return n >= 1 ? 1 : 0;
	}

	static linear(t: number): number {
		return t;
	}

	static ease(t: number): number {
		easeCurve ??= bezier(0.42, 0, 1, 1);
		return easeCurve(t);
	}

	static quad(t: number): number {
		return t * t;
	}

	static cubic(t: number): number {
		return t * t * t;
	}

	static poly(n: number): EasingFunction {
		return (t) => t ** n;
	}

	static sin(t: number): number {
		return 1 - Math.cos((t * Math.PI) / 2);
	}

	static circle(t: number): number {
		return 1 - Math.sqrt(1 - t * t);
	}

	static exp(t: number): number {
		return 2 ** (10 * (t - 1));
	}

	static elastic(bounciness = 1): EasingFunction {
		const p = bounciness * Math.PI;
		return (t) => 1 - Math.cos((t * Math.PI) / 2) ** 3 * Math.cos(t * p);
	}

	static back(s = 1.70158): EasingFunction {
		return (t) => t * t * ((s + 1) * t - s);
	}

	static bounce(t: number): number {
		if (t < 1 / 2.75) return 7.5625 * t * t;
		if (t < 2 / 2.75) {
			const u = t - 1.5 / 2.75;
			return 7.5625 * u * u + 0.75;
		}
		if (t < 2.5 / 2.75) {
			const u = t - 2.25 / 2.75;
			return 7.5625 * u * u + 0.9375;
		}
		const u = t - 2.625 / 2.75;
		return 7.5625 * u * u + 0.984375;
	}

	static bezier(x1: number, y1: number, x2: number, y2: number): EasingFunction {
		return bezier(x1, y1, x2, y2);
	}

	static in(easing: EasingFunction): EasingFunction {
		return easing;
	}

	static out(easing: EasingFunction): EasingFunction {
		return (t) => 1 - easing(1 - t);
	}

	static inOut(easing: EasingFunction): EasingFunction {
		return (t) => (t < 0.5 ? easing(t * 2) / 2 : 1 - easing((1 - t) * 2) / 2);
	}
}

// ---------------------------------------------------------------- interpolate
export type ExtrapolateType = 'extend' | 'identity' | 'clamp' | 'wrap';
export type InterpolateOptions = Partial<{
	easing: EasingFunction;
	extrapolateLeft: ExtrapolateType;
	extrapolateRight: ExtrapolateType;
}>;

const MODES: readonly ExtrapolateType[] = ['extend', 'identity', 'clamp', 'wrap'];

/** Throws unless `r` is a list of at least two finite numbers. */
const requireNumbers = (name: string, r: readonly number[]) => {
	if (!Array.isArray(r)) throw new TypeError(`interpolate: ${name} must be an array`);
	if (r.length < 2) throw new Error(`interpolate: ${name} needs at least two values, got ${r.length}`);
	r.forEach((v, i) => {
		if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`interpolate: ${name}[${i}] is ${String(v)}, not a finite number`);
	});
};

/**
 * Piecewise mapping of `input` from inputRange to outputRange, with an easing applied inside each segment. Same
 * meaning as React Native's Animated interpolation (extend / identity / clamp on either side), plus `wrap`.
 */
export const interpolate = (
	input: number,
	inputRange: readonly number[],
	outputRange: readonly number[],
	options: InterpolateOptions = {},
): number => {
	if (typeof input !== 'number') throw new TypeError(`interpolate: input must be a number, got ${typeof input}`);
	requireNumbers('inputRange', inputRange);
	requireNumbers('outputRange', outputRange);
	if (inputRange.length !== outputRange.length)
		throw new Error(`interpolate: inputRange and outputRange differ in length (${inputRange.length} and ${outputRange.length})`);
	for (let i = 1; i < inputRange.length; i++)
		if (inputRange[i] <= inputRange[i - 1]) throw new Error(`interpolate: inputRange must strictly increase, got [${inputRange.join(', ')}]`);
	const easing = options.easing ?? Easing.linear;
	const left = options.extrapolateLeft ?? 'extend';
	const right = options.extrapolateRight ?? 'extend';
	if (!MODES.includes(left) || !MODES.includes(right)) throw new Error(`interpolate: unknown extrapolation "${!MODES.includes(left) ? left : right}"`);

	// outside the range: identity hands the input back; clamp holds it at the end; wrap folds it back into
	// [first, last) periodically; extend leaves it, so the end segment's line carries on
	const first = inputRange[0];
	const last = inputRange[inputRange.length - 1];
	let x = input;
	if (x < first || x > last) {
		const mode = x < first ? left : right;
		if (mode === 'identity') return input;
		if (mode === 'clamp') x = x < first ? first : last;
		else if (mode === 'wrap') x -= (last - first) * Math.floor((x - first) / (last - first));
	}
	// the segment: the first whose right end is at or beyond x (a knot belongs to the segment on its left); below the
	// range the first segment, above it the last
	let k = 0;
	while (k < inputRange.length - 2 && inputRange[k + 1] < x) k++;
	const a = inputRange[k];
	const b = inputRange[k + 1];
	const lo = outputRange[k];
	const hi = outputRange[k + 1];
	if (lo === hi) return lo;
	return lo + (hi - lo) * easing((x - a) / (b - a));
};

// ---------------------------------------------------------------- spring
export type SpringConfig = {
	damping: number;
	mass: number;
	stiffness: number;
	overshootClamping: boolean;
};

const SPRING_DEFAULTS: SpringConfig = {damping: 10, mass: 1, stiffness: 100, overshootClamping: false};

const springConfig = (config: Partial<SpringConfig>): SpringConfig => {
	const c = {...SPRING_DEFAULTS, ...config};
	for (const name of ['damping', 'mass', 'stiffness'] as const) {
		const v = c[name];
		if (typeof v !== 'number' || !Number.isFinite(v) || v <= 0)
			throw new Error(`spring: config.${name} must be a positive finite number, got ${String(v)}${name === 'damping' ? ' (an undamped spring never comes to rest)' : ''}`);
	}
	return c;
};

/**
 * The spring as a damped harmonic oscillator m·y'' + d·y' + k·y = 0, where y is the distance to the target, started at
 * rest one unit short of it (y(0) = −1, y'(0) = 0). `offset(t)` is y at t seconds, closed form for each regime;
 * `bound(t)` is an upper bound of |y| from t on, valid once t ≥ `boundFrom` (where it starts to decrease).
 */
type Oscillator = {offset: (t: number) => number; bound: (t: number) => number; boundFrom: number};

const oscillator = (c: SpringConfig): Oscillator => {
	const y0 = -1;
	const v0 = 0;
	const w0 = Math.sqrt(c.stiffness / c.mass); // natural angular frequency
	const zeta = c.damping / (2 * Math.sqrt(c.stiffness * c.mass)); // damping ratio
	if (Math.abs(zeta - 1) < 1e-6) {
		// critically damped: y = e^(−w0 t) (p + q t)
		const p = y0;
		const q = v0 + w0 * y0;
		const qa = Math.abs(q);
		return {
			offset: (t) => Math.exp(-w0 * t) * (p + q * t),
			bound: (t) => Math.exp(-w0 * t) * (Math.abs(p) + qa * t),
			boundFrom: qa > 0 ? Math.max(0, 1 / w0 - Math.abs(p) / qa) : 0,
		};
	}
	if (zeta < 1) {
		// under-damped: y = e^(−σ t) (A cos(wd t) + B sin(wd t))
		const sigma = zeta * w0;
		const wd = w0 * Math.sqrt(1 - zeta * zeta);
		const A = y0;
		const B = (v0 + sigma * y0) / wd;
		const amp = Math.hypot(A, B);
		return {
			offset: (t) => Math.exp(-sigma * t) * (A * Math.cos(wd * t) + B * Math.sin(wd * t)),
			bound: (t) => amp * Math.exp(-sigma * t),
			boundFrom: 0,
		};
	}
	// over-damped: y = C1 e^(r1 t) + C2 e^(r2 t), r2 < r1 < 0 (r1 taken as w0² / r2 to avoid cancellation)
	const r2 = -w0 * (zeta + Math.sqrt(zeta * zeta - 1));
	const r1 = (w0 * w0) / r2;
	const C1 = (v0 - r2 * y0) / (r1 - r2);
	const C2 = y0 - C1;
	return {
		offset: (t) => C1 * Math.exp(r1 * t) + C2 * Math.exp(r2 * t),
		bound: (t) => Math.abs(C1) * Math.exp(r1 * t) + Math.abs(C2) * Math.exp(r2 * t),
		boundFrom: 0,
	};
};

const requireFps = (fps: number, where: string) => {
	if (typeof fps !== 'number' || !Number.isFinite(fps) || fps <= 0) throw new Error(`${where}: fps must be a positive finite number, got ${String(fps)}`);
};

/** Longest a spring may take to settle before measureSpring gives up (an hour of frames). */
const SETTLE_LIMIT_SECONDS = 3600;
const settled = new Map<string, number>();

/**
 * How many frames the spring (0 → 1, at rest at both ends) takes to settle: the first frame from which every later
 * frame stays within `threshold` of the target. Frames are walked from 0; the walk stops as soon as the oscillator's
 * decay envelope is below the threshold, after which no frame can leave it again.
 */
export const measureSpring = ({
	fps,
	config = {},
	threshold = 0.005,
}: {
	fps: number;
	config?: Partial<SpringConfig>;
	threshold?: number;
}): number => {
	requireFps(fps, 'measureSpring');
	if (typeof threshold !== 'number' || !Number.isFinite(threshold) || threshold <= 0)
		throw new Error(`measureSpring: threshold must be a positive finite number, got ${String(threshold)}`);
	const c = springConfig(config);
	const key = `${fps}|${threshold}|${c.stiffness}|${c.mass}|${c.damping}`;
	const known = settled.get(key);
	if (known !== undefined) return known;
	const osc = oscillator(c);
	let lastOutside = -1;
	for (let n = 0; ; n++) {
		const t = n / fps;
		if (Math.abs(osc.offset(t)) >= threshold) lastOutside = n;
		if (t >= osc.boundFrom && osc.bound(t) < threshold) break;
		if (t > SETTLE_LIMIT_SECONDS) throw new Error(`measureSpring: the spring takes longer than ${SETTLE_LIMIT_SECONDS} s to settle`);
	}
	settled.set(key, lastOutside + 1);
	return lastOutside + 1;
};

/**
 * Value of a spring animation at `frame`: from `from` to `to`, starting `delay` frames late. The spring is evaluated
 * in closed form at t = frame / fps (fractional frames included), not stepped frame by frame.
 *   durationInFrames  stretches the spring's own settling time (measureSpring with durationRestThreshold) to that many
 *                     frames; from then on the value is exactly `to`
 *   reverse           plays the same animation backwards: `to` until the delay is over, `from` at the end
 *   overshootClamping (in config) never goes past `to`
 */
export const spring = ({
	frame,
	fps,
	config = {},
	from = 0,
	to = 1,
	durationInFrames,
	durationRestThreshold,
	delay = 0,
	reverse = false,
}: {
	frame: number;
	fps: number;
	config?: Partial<SpringConfig>;
	from?: number;
	to?: number;
	durationInFrames?: number;
	durationRestThreshold?: number;
	delay?: number;
	reverse?: boolean;
}): number => {
	if (typeof frame !== 'number' || Number.isNaN(frame)) throw new TypeError(`spring: frame must be a number, got ${String(frame)}`);
	requireFps(fps, 'spring');
	if (durationInFrames !== undefined && !(typeof durationInFrames === 'number' && durationInFrames > 0 && Number.isFinite(durationInFrames)))
		throw new TypeError(`spring: durationInFrames must be a positive finite number, got ${String(durationInFrames)}`);
	const c = springConfig(config);
	const natural = durationInFrames !== undefined || reverse ? measureSpring({fps, config: c, threshold: durationRestThreshold}) : 0;
	// frames into the forward animation
	const elapsed = frame - delay;
	const forward = reverse ? (durationInFrames ?? natural) - elapsed : elapsed;
	let progress: number;
	if (durationInFrames !== undefined && forward >= durationInFrames) progress = 1;
	else {
		const springFrames = durationInFrames === undefined ? forward : (forward * natural) / durationInFrames;
		progress = 1 + oscillator(c).offset(Math.max(0, springFrames) / fps);
	}
	if (c.overshootClamping && progress > 1) progress = 1;
	return from === 0 && to === 1 ? progress : from + (to - from) * progress;
};

// ---------------------------------------------------------------- random
// FNV-1a, 32 bit (Glenn Fowler, Landon Curt Noll, Kiem-Phong Vo; public domain), over the bytes of the seed.
const FNV_OFFSET = 0x811c9dc5;
const FNV_PRIME = 0x01000193;
const fnvByte = (h: number, byte: number) => Math.imul(h ^ byte, FNV_PRIME);

const hashString = (s: string): number => {
	let h = FNV_OFFSET;
	for (let i = 0; i < s.length; i++) {
		const unit = s.charCodeAt(i); // UTF-16 code unit, low byte first
		h = fnvByte(fnvByte(h, unit & 0xff), unit >>> 8);
	}
	return h >>> 0;
};

const numberBytes = new DataView(new ArrayBuffer(8));
const hashNumber = (x: number): number => {
	numberBytes.setFloat64(0, x === 0 ? 0 : x, true); // −0 and 0 are the same seed
	let h = FNV_OFFSET;
	for (let i = 0; i < 8; i++) h = fnvByte(h, numberBytes.getUint8(i));
	return h >>> 0;
};

// mulberry32 (Tommy Ettinger, 2017; public domain): one output of the generator seeded with `state`, in [0, 1).
const mulberry32 = (state: number): number => {
	const a = (state + 0x6d2b79f5) | 0;
	let t = Math.imul(a ^ (a >>> 15), a | 1);
	t = (t + Math.imul(t ^ (t >>> 7), t | 61)) ^ t;
	return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

/** Deterministic number in [0, 1) for a seed (a string or a number); `null` gives Math.random(). */
export const random = (seed: number | string | null): number => {
	if (seed === null) return Math.random();
	if (typeof seed === 'string') return mulberry32(hashString(seed));
	if (typeof seed === 'number') {
		if (Number.isNaN(seed)) throw new TypeError('random(seed): seed is NaN');
		return mulberry32(hashNumber(seed));
	}
	throw new TypeError(`random(seed): seed must be a string, a number or null, got ${typeof seed}`);
};
