// A camera for flat pages: a path through keys that it passes through rather than stops at (Catmull-Rom, non-uniform
// spacing), so the camera keeps its speed and only changes its mind; the first and last keys are eased out of and
// into. Kept inside the page: zoom ≥ 1 and the centre where the scaled page still covers the screen. The world layer
// is plain DOM transformed with lensTransform() (transformOrigin '0 0').
export type Lens = {z: number; cx: number; cy: number};
/** [frame, zoom, centre x, centre y] */
export type LensKey = [number, number, number, number];
/** Screen size the lens maps onto (the composition's). */
export type Screen = {width: number; height: number};

const FULL_HD: Screen = {width: 1920, height: 1080};

export const lensPath = (keys: LensKey[], f: number): Lens => {
	const n = keys.length;
	if (f <= keys[0][0]) return {z: keys[0][1], cx: keys[0][2], cy: keys[0][3]};
	if (f >= keys[n - 1][0]) return {z: keys[n - 1][1], cx: keys[n - 1][2], cy: keys[n - 1][3]};
	let i = 0;
	while (i < n - 2 && f >= keys[i + 1][0]) i++;
	const k = (j: number) => keys[Math.max(0, Math.min(n - 1, j))];
	const [t0] = k(i);
	const [t1] = k(i + 1);
	const u = Math.min(1, Math.max(0, (f - t0) / (t1 - t0)));
	const h = (c: number) => {
		const [a, b, cc, d] = [k(i - 1), k(i), k(i + 1), k(i + 2)];
		// the first and last keys are eased out of and into; every other key is passed through
		const m0 = i === 0 ? 0 : ((cc[c] - a[c]) / (cc[0] - a[0])) * (t1 - t0);
		const m1 = i + 1 >= n - 1 ? 0 : ((d[c] - b[c]) / (d[0] - b[0])) * (t1 - t0);
		const u2 = u * u;
		const u3 = u2 * u;
		return (2 * u3 - 3 * u2 + 1) * b[c] + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * cc[c] + (u3 - u2) * m1;
	};
	return {z: h(1), cx: h(2), cy: h(3)};
};

/** Never past the picture's edge. */
export const clampLens = (l: Lens, s: Screen = FULL_HD): Lens => {
	const z = Math.max(1, l.z);
	const c = (v: number, half: number, size: number) => Math.min(size - half / z, Math.max(half / z, v));
	return {z, cx: c(l.cx, s.width / 2, s.width), cy: c(l.cy, s.height / 2, s.height)};
};

/** CSS transform for a world layer seen through the lens (use with transformOrigin '0 0'). */
export const lensTransform = (l: Lens, s: Screen = FULL_HD) =>
	`translate(${s.width / 2}px, ${s.height / 2}px) scale(${l.z}) translate(${-l.cx}px, ${-l.cy}px)`;

/** World point → screen point. */
export const toScreenLens = (l: Lens, [x, y]: [number, number], s: Screen = FULL_HD): [number, number] => [
	s.width / 2 + (x - l.cx) * l.z,
	s.height / 2 + (y - l.cy) * l.z,
];
