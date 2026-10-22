// 动画小工具：钳位、插值、缓动、按帧号的确定性随机、呼吸与拍点脉冲。全部是帧号的纯函数。
import {Easing, interpolate, random} from '../shim/remotion';

export const clamp = (x: number, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export const ease = {
	inOut: Easing.bezier(0.65, 0, 0.35, 1),
	out: Easing.bezier(0.16, 1, 0.3, 1), // expo-like out
	in: Easing.bezier(0.7, 0, 0.84, 0),
	soft: Easing.bezier(0.45, 0, 0.2, 1),
	linear: (t: number) => t,
};

/** 0→1 progress of `frame` between a and b, eased. */
export const ramp = (frame: number, a: number, b: number, e: (t: number) => number = ease.inOut) =>
	e(clamp((frame - a) / Math.max(1, b - a)));

/**
 * Fade in over [a, a+inLen], hold, fade out over [b-outLen, b]. (Named `window` in the source film; renamed so an
 * import does not shadow the global.)
 */
export const fadeWindow = (frame: number, a: number, b: number, inLen = 12, outLen = 12) =>
	interpolate(frame, [a, a + inLen, b - outLen, b], [0, 1, 1, 0], {
		extrapolateLeft: 'clamp',
		extrapolateRight: 'clamp',
		easing: ease.soft,
	});

/** Deterministic pseudo random in [0,1). */
export const rnd = (seed: string | number) => random(String(seed));

/** Smooth deterministic wobble for organic drift. */
export const wobble = (frame: number, seed: number, speed = 0.02, amp = 1) =>
	amp *
	(Math.sin(frame * speed + seed * 12.9898) * 0.6 + Math.sin(frame * speed * 1.7 + seed * 78.233) * 0.4);

/** Beat pulse: 1 on the beat, decaying over `len` frames. */
export const pulse = (frame: number, at: number, len = 16) => {
	const d = frame - at;
	if (d < 0 || d > len) return 0;
	return Math.exp(-d / (len / 4));
};

/**
 * 一拍二 / 一拍三：把帧号压到每 `n` 帧一格（从 `from` 起算），手作感的图形段用（镜头可以照常每帧走）。
 * onTwos(f) 在 f = 0,1 → 0；2,3 → 2；…
 */
export const onTwos = (frame: number, n = 2, from = 0) => from + Math.floor((frame - from) / n) * n;
