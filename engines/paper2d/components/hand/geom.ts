// 手绘笔触的几何层：纯函数、确定性（同参数同输出），不依赖 React。
// Pipeline: input points → spline (smooth) → uniform arc-length samples → per-sample half width
// (hand speed, seeded pressure, start press, end taper) → outline polygon + dry-brush gap strips.
export type Pt = [number, number];
export type Texture = 'ink' | 'pencil' | 'brush' | 'gold' | 'cord';
export type Pace = 'hand' | 'flick' | 'linear';

// ---------------------------------------------------------------- deterministic noise
const toInt = (x: number) => Math.round(x * 7919) | 0;
/** Integer hash → [0,1). Pure integer arithmetic, identical on every run. */
export const hash = (seed: number, i: number): number => {
	let h = Math.imul(toInt(seed) ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(i | 0, 0xc2b2ae35);
	h ^= h >>> 16;
	h = Math.imul(h, 0x7feb352d);
	h ^= h >>> 15;
	h = Math.imul(h, 0x846ca68b);
	h ^= h >>> 16;
	return (h >>> 0) / 4294967296;
};
/** Smooth 1-D value noise in [-1, 1]. */
export const noise1 = (x: number, seed: number) => {
	const i = Math.floor(x);
	const f = x - i;
	const u = f * f * f * (f * (f * 6 - 15) + 10);
	const a = hash(seed, i) * 2 - 1;
	const b = hash(seed, i + 1) * 2 - 1;
	return a + (b - a) * u;
};
/** Fractal 1-D noise in about [-1, 1]. */
export const fbm1 = (x: number, seed: number, octaves = 3) => {
	let sum = 0;
	let amp = 1;
	let norm = 0;
	let fx = x;
	for (let k = 0; k < octaves; k++) {
		sum += amp * noise1(fx, seed + k * 101);
		norm += amp;
		amp *= 0.5;
		fx *= 2.03;
	}
	return sum / norm;
};
const smoothstep = (a: number, b: number, x: number) => {
	const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
	return t * t * (3 - 2 * t);
};
const clamp01 = (x: number) => Math.min(1, Math.max(0, x));

// ---------------------------------------------------------------- hand speed
// Human point-to-point movements follow the minimum-jerk profile (slow start, fast middle, slow
// end). A little constant speed is mixed in so the pen never stalls completely at either end.
const minJerk = (t: number) => t * t * t * (10 + t * (-15 + 6 * t));
const minJerkV = (t: number) => 30 * t * t * (1 - t) * (1 - t);
const PACE: Record<Pace, {pos: (t: number) => number; vel: (t: number) => number; vmax: number}> = {
	hand: {pos: (t) => 0.2 * t + 0.8 * minJerk(t), vel: (t) => 0.2 + 0.8 * minJerkV(t), vmax: 1.7},
	// decisive cut: a brief press, then it accelerates and lifts off still moving
	flick: {pos: (t) => 0.3 * t + 0.7 * t * t, vel: (t) => 0.3 + 1.4 * t, vmax: 1.7},
	linear: {pos: (t) => t, vel: () => 1, vmax: 1},
};
/** Map time-linear progress 0..1 to the drawn fraction of the stroke's length. */
export const pacePos = (t: number, pace: Pace = 'hand') => PACE[pace].pos(clamp01(t));
/** Normalised pen speed (0..1) at drawn fraction u of the stroke. */
const speedAt = (pace: Pace) => {
	const p = PACE[pace];
	const n = 64;
	const ts = Array.from({length: n + 1}, (_, i) => i / n);
	const us = ts.map(p.pos);
	return (u: number) => {
		let i = 0;
		while (i < n - 1 && us[i + 1] < u) i++;
		const k = us[i + 1] > us[i] ? (u - us[i]) / (us[i + 1] - us[i]) : 0;
		const t = ts[i] + (ts[i + 1] - ts[i]) * clamp01(k);
		return p.vel(t) / p.vmax;
	};
};

// ---------------------------------------------------------------- polylines
const dist = (a: Pt, b: Pt) => Math.hypot(b[0] - a[0], b[1] - a[1]);
export const polyLength = (pts: Pt[]) => {
	let l = 0;
	for (let i = 1; i < pts.length; i++) l += dist(pts[i - 1], pts[i]);
	return l;
};

/** Uniform arc-length resample; returns points and their arc positions. */
export const resample = (pts: Pt[], step: number): {P: Pt[]; s: number[]} => {
	const P: Pt[] = [];
	const s: number[] = [];
	if (pts.length === 0) return {P, s};
	const L = polyLength(pts);
	const n = Math.max(1, Math.round(L / step));
	const ds = L / n;
	let seg = 0;
	let segStart = 0;
	let segLen = pts.length > 1 ? dist(pts[0], pts[1]) : 0;
	for (let k = 0; k <= n; k++) {
		const target = k * ds;
		while (seg < pts.length - 2 && segStart + segLen < target) {
			segStart += segLen;
			seg++;
			segLen = dist(pts[seg], pts[seg + 1]);
		}
		const a = pts[seg];
		const b = pts[Math.min(seg + 1, pts.length - 1)];
		const f = segLen > 1e-9 ? clamp01((target - segStart) / segLen) : 0;
		P.push([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]);
		s.push(target);
	}
	return {P, s};
};

/**
 * Cardinal spline through the points. `k` scales the tangents: 0 keeps straight segments
 * (a polyline), 1 is a chord-normalised Catmull-Rom that does not overshoot on uneven spacing.
 */
const spline = (pts: Pt[], k: number, step = 1): Pt[] => {
	if (pts.length < 2) return pts.slice();
	const n = pts.length;
	const get = (i: number): Pt => {
		if (i < 0) return [2 * pts[0][0] - pts[1][0], 2 * pts[0][1] - pts[1][1]];
		if (i >= n) return [2 * pts[n - 1][0] - pts[n - 2][0], 2 * pts[n - 1][1] - pts[n - 2][1]];
		return pts[i];
	};
	const out: Pt[] = [pts[0]];
	for (let i = 0; i < n - 1; i++) {
		const p0 = get(i - 1);
		const p1 = pts[i];
		const p2 = pts[i + 1];
		const p3 = get(i + 2);
		const d01 = Math.max(1e-6, dist(p0, p1));
		const d12 = Math.max(1e-6, dist(p1, p2));
		const d23 = Math.max(1e-6, dist(p2, p3));
		const m1: Pt = [(k * (p2[0] - p0[0]) * d12) / (d01 + d12), (k * (p2[1] - p0[1]) * d12) / (d01 + d12)];
		const m2: Pt = [(k * (p3[0] - p1[0]) * d12) / (d12 + d23), (k * (p3[1] - p1[1]) * d12) / (d12 + d23)];
		const m = Math.max(2, Math.ceil(d12 / step));
		for (let j = 1; j <= m; j++) {
			const t = j / m;
			const t2 = t * t;
			const t3 = t2 * t;
			const h00 = 2 * t3 - 3 * t2 + 1;
			const h10 = t3 - 2 * t2 + t;
			const h01 = -2 * t3 + 3 * t2;
			const h11 = t3 - t2;
			out.push([
				h00 * p1[0] + h10 * m1[0] + h01 * p2[0] + h11 * m2[0],
				h00 * p1[1] + h10 * m1[1] + h01 * p2[1] + h11 * m2[1],
			]);
		}
	}
	return out;
};

/** Low-pass with a window that shrinks at the ends, so the end points stay where they were put. */
const lowpass = (P: Pt[], radius: number): Pt[] => {
	const r = Math.round(radius);
	if (r < 1) return P;
	const n = P.length;
	const cx = [0];
	const cy = [0];
	for (let i = 0; i < n; i++) {
		cx.push(cx[i] + P[i][0]);
		cy.push(cy[i] + P[i][1]);
	}
	return P.map((_, i) => {
		const w = Math.min(r, i, n - 1 - i);
		const a = i - w;
		const b = i + w + 1;
		return [(cx[b] - cx[a]) / (b - a), (cy[b] - cy[a]) / (b - a)] as Pt;
	});
};

/**
 * The stroke's centre line. smooth: 0 = straight segments through the points, 0.5 (default) =
 * spline through the points, 1 = spline plus low-pass that irons out small wiggles.
 */
export const centerline = (pts: Pt[], smooth = 0.5): Pt[] => {
	const clean = pts.filter((p, i) => i === 0 || dist(p, pts[i - 1]) > 1e-3);
	if (clean.length < 2) return clean;
	const dense = spline(clean, Math.min(1, smooth * 2));
	if (smooth <= 0.5) return dense;
	const L = polyLength(dense);
	const {P} = resample(dense, 1);
	return lowpass(P, (smooth - 0.5) * 2 * Math.min(24, 0.06 * L));
};

/**
 * Make a regular path irregular: resample it and push each point along its normal by smooth
 * noise of amplitude `amp` px (wavelength scales with amp). End points move half as much.
 */
export function roughen(pts: Pt[], amp: number, seed: number): Pt[] {
	if (pts.length < 2 || !(amp > 0)) return pts.map((p) => [p[0], p[1]] as Pt);
	const lambda = Math.max(18, amp * 10);
	const {P, s} = resample(pts, lambda / 3);
	const L = s[s.length - 1] || 1;
	return P.map((p, i) => {
		const a = P[Math.max(0, i - 1)];
		const b = P[Math.min(P.length - 1, i + 1)];
		const len = Math.max(1e-6, dist(a, b));
		const nx = -(b[1] - a[1]) / len;
		const ny = (b[0] - a[0]) / len;
		const endFade = 0.5 + 0.5 * smoothstep(0, Math.min(lambda, L / 3), Math.min(s[i], L - s[i]));
		const d = amp * endFade * fbm1(s[i] / lambda, seed * 13 + 5, 2);
		const t = amp * 0.35 * endFade * noise1(s[i] / lambda + 7.3, seed * 17 + 3);
		const tx = (b[0] - a[0]) / len;
		const ty = (b[1] - a[1]) / len;
		return [p[0] + nx * d + tx * t, p[1] + ny * d + ty * t] as Pt;
	});
}

// ---------------------------------------------------------------- texture presets
export type Preset = {
	thin: number; // how much speed thins the line
	press: number; // seeded pressure variation (±)
	startW: number; // width at the very first touch, relative
	bump: number; // extra weight just after the touch (起笔略重)
	taper: [number, number]; // default taper lengths as a fraction of the stroke
	dry: number; // dry-brush amount (飞白)
	edge: number; // geometric edge wobble, relative to half width
	head: number; // ink bead / bright head size relative to the half width at the tip
	entry: number; // how far the first touch juts out toward the upper left (a right hand's brush entry)
	opacity: number; // default opacity
};
export const PRESETS: Record<Texture, Preset> = {
	ink: {thin: 0.32, press: 0.16, startW: 0.55, bump: 0.14, taper: [0.09, 0.32], dry: 1, edge: 0.07, head: 1.08, entry: 0.7, opacity: 1},
	brush: {thin: 0.34, press: 0.2, startW: 0.6, bump: 0.18, taper: [0.1, 0.34], dry: 0.3, edge: 0.05, head: 1.1, entry: 0.55, opacity: 1},
	pencil: {thin: 0.12, press: 0.1, startW: 0.6, bump: 0.05, taper: [0.07, 0.2], dry: 0, edge: 0.05, head: 1.04, entry: 0, opacity: 0.86},
	gold: {thin: 0.18, press: 0.07, startW: 0.45, bump: 0.06, taper: [0.12, 0.24], dry: 0, edge: 0.02, head: 1.7, entry: 0, opacity: 1},
	// a coloured cord drawn with a felt pen: even, clean, just enough edge to be made by hand (the paper scenes' thread)
	cord: {thin: 0.05, press: 0.04, startW: 0.9, bump: 0.02, taper: [0.02, 0.05], dry: 0, edge: 0.012, head: 1, entry: 0, opacity: 1},
};

// ---------------------------------------------------------------- stroke geometry
export type StrokeGeo = {
	P: Pt[]; // centre samples
	N: Pt[]; // unit normals
	s: number[]; // arc position
	hw: number[]; // half width
	eL: number[]; // edge wobble factors
	eR: number[];
	dry: number[]; // dry-brush amount 0..1
	curv: number[]; // |curvature|
	L: number;
	bbox: [number, number, number, number];
};

export type GeoInput = {
	pts: Pt[];
	width: number;
	texture: Texture;
	taper?: [number, number];
	seed: number;
	smooth: number;
	pace: Pace;
	dry?: number;
};

/** Taper values ≤ 1 are fractions of the stroke length, larger values are pixels. */
const taperFrac = (v: number, L: number) => (v <= 1 ? v : v / Math.max(1, L));

export const strokeGeo = ({pts, width, texture, taper, seed, smooth, pace, dry}: GeoInput): StrokeGeo | null => {
	const line = centerline(pts, smooth);
	const L0 = polyLength(line);
	if (line.length < 2 || L0 < 0.5) return null;
	const step = Math.max(0.75, Math.min(3, L0 / 500));
	const {P, s} = resample(line, step);
	const n = P.length;
	const L = s[n - 1];
	const pre = PRESETS[texture];
	const [ta, tb] = taper ?? pre.taper;
	const t0 = Math.max(0.002, Math.min(0.5, taperFrac(ta, L)));
	const t1 = Math.max(0.002, Math.min(0.8, taperFrac(tb, L)));
	const speed = speedAt(pace);

	// tangents, normals, curvature
	const N: Pt[] = [];
	const ang: number[] = [];
	for (let i = 0; i < n; i++) {
		const a = P[Math.max(0, i - 1)];
		const b = P[Math.min(n - 1, i + 1)];
		const len = Math.max(1e-9, dist(a, b));
		const tx = (b[0] - a[0]) / len;
		const ty = (b[1] - a[1]) / len;
		N.push([-ty, tx]);
		ang.push(Math.atan2(ty, tx));
	}
	const curv = ang.map((_, i) => {
		const a = ang[Math.max(0, i - 2)];
		const b = ang[Math.min(n - 1, i + 2)];
		let d = b - a;
		while (d > Math.PI) d -= 2 * Math.PI;
		while (d < -Math.PI) d += 2 * Math.PI;
		const span = s[Math.min(n - 1, i + 2)] - s[Math.max(0, i - 2)];
		return span > 0 ? Math.abs(d) / span : 0;
	});

	// width: pen speed (fast = thinner), seeded pressure, a pressed start, a long lift-off taper
	// hairlines need a larger relative swing to read as drawn rather than plotted
	const pressAmp = pre.press * (1 + 1.2 * (1 - smoothstep(1.5, 6, width)));
	const raw = s.map((si, i) => {
		const u = si / L;
		const v = speed(u);
		const thin = 1 - pre.thin * v;
		const pressure = 1 + pressAmp * fbm1(si / 42, seed * 7 + 1, 3);
		const slow = 1 + 0.12 * Math.min(1, curv[i] * width * 2); // the hand slows in a turn
		let env: number;
		if (u < t0) {
			const a = smoothstep(0, 1, u / t0);
			env = pre.startW + (1 + pre.bump - pre.startW) * Math.sqrt(a);
		} else env = 1 + pre.bump * Math.exp(-(u - t0) / Math.max(0.02, t0 * 1.5));
		if (u > 1 - t1) {
			const a = clamp01((1 - u) / t1);
			env *= 0.03 + 0.97 * Math.pow(a, 0.85);
		}
		return Math.max(0, thin * pressure * slow * env);
	});
	const peak = Math.max(1e-6, ...raw);
	const hw = raw.map((r) => (0.5 * width * r) / peak);

	const eAmp = pre.edge;
	const eL = s.map((si) => eAmp * (fbm1(si / 6, seed * 11 + 21, 2) + 0.6 * noise1(si / 31, seed * 3 + 5)));
	const eR = s.map((si) => eAmp * (fbm1(si / 6, seed * 11 + 47, 2) + 0.6 * noise1(si / 31, seed * 3 + 9)));

	// dry brush: where the pen is fast and in the second half of a long stroke
	const dryAmt = (dry ?? pre.dry) * smoothstep(40, 170, L) * smoothstep(3, 7, width);
	const dryArr = s.map((si) => {
		if (dryAmt <= 0) return 0;
		const u = si / L;
		const v = speed(u);
		const d = 1.1 * (v - 0.5) + 0.9 * (u - 0.3) + 0.32 * fbm1(si / 55, seed * 5 + 3, 2);
		return clamp01(d * 1.3) * dryAmt;
	});

	let x0 = Infinity,
		y0 = Infinity,
		x1 = -Infinity,
		y1 = -Infinity;
	for (let i = 0; i < n; i++) {
		const r = hw[i] * 1.3 + 1;
		x0 = Math.min(x0, P[i][0] - r);
		y0 = Math.min(y0, P[i][1] - r);
		x1 = Math.max(x1, P[i][0] + r);
		y1 = Math.max(y1, P[i][1] + r);
	}
	return {P, N, s, hw, eL, eR, dry: dryArr, curv, L, bbox: [x0, y0, x1, y1]};
};

// ---------------------------------------------------------------- outlines
const fmt = (v: number) => (Math.round(v * 100) / 100).toString();
const polyD = (poly: Pt[]) => {
	if (poly.length < 3) return '';
	let d = 'M' + fmt(poly[0][0]) + ' ' + fmt(poly[0][1]);
	for (let i = 1; i < poly.length; i++) d += 'L' + fmt(poly[i][0]) + ' ' + fmt(poly[i][1]);
	return d + 'Z';
};
const signedArea = (poly: Pt[]) => {
	let a = 0;
	for (let i = 0; i < poly.length; i++) {
		const p = poly[i];
		const q = poly[(i + 1) % poly.length];
		a += p[0] * q[1] - q[0] * p[1];
	}
	return a / 2;
};
const circlePoly = (c: Pt, r: number, ccw: boolean): Pt[] => {
	const m = Math.max(8, Math.min(24, Math.ceil(r * 2)));
	return Array.from({length: m}, (_, k) => {
		const a = ((ccw ? -1 : 1) * 2 * Math.PI * k) / m;
		return [c[0] + r * Math.cos(a), c[1] + r * Math.sin(a)] as Pt;
	});
};

/** Samples up to the drawn tip (arc position sTip), with an interpolated last sample. */
export type Cut = {idx: number[]; tipP: Pt; tipN: Pt; tipHW: number; tipDry: number; tipE: [number, number]; tipS: number};
export const cutAt = (g: StrokeGeo, sTip: number): Cut => {
	const idx: number[] = [];
	let i = 0;
	while (i < g.s.length && g.s[i] <= sTip) idx.push(i++);
	const a = Math.max(0, Math.min(g.s.length - 1, i - 1));
	const b = Math.min(g.s.length - 1, i);
	const f = g.s[b] > g.s[a] ? clamp01((sTip - g.s[a]) / (g.s[b] - g.s[a])) : 0;
	const mix = (x: number, y: number) => x + (y - x) * f;
	return {
		idx,
		tipP: [mix(g.P[a][0], g.P[b][0]), mix(g.P[a][1], g.P[b][1])],
		tipN: g.N[b],
		tipHW: mix(g.hw[a], g.hw[b]),
		tipDry: mix(g.dry[a], g.dry[b]),
		tipE: [mix(g.eL[a], g.eL[b]), mix(g.eR[a], g.eR[b])],
		tipS: sTip,
	};
};

/**
 * Filled outline of the stroke up to the cut: left side, round tip cap, right side back, round
 * start cap. Where a tight turn would fold the inner edge back over itself, discs of the same
 * orientation are added so the nonzero fill never opens a hole.
 */
export const bodyPath = (g: StrokeGeo, cut: Cut, jitter: Pt[] | null, hwScale = 1, edge = true, entry = 0): string => {
	const pts: Pt[] = [];
	const nrm: Pt[] = [];
	const hws: number[] = [];
	const eLs: number[] = [];
	const eRs: number[] = [];
	for (const i of cut.idx) {
		const j = jitter ? jitter[i] : ([0, 0] as Pt);
		pts.push([g.P[i][0] + j[0], g.P[i][1] + j[1]]);
		nrm.push(g.N[i]);
		hws.push(g.hw[i] * hwScale);
		eLs.push(edge ? g.eL[i] : 0);
		eRs.push(edge ? g.eR[i] : 0);
	}
	const last = cut.idx.length ? cut.idx[cut.idx.length - 1] : 0;
	if (!cut.idx.length || cut.tipS - g.s[last] > 0.05) {
		const j = jitter ? jitter[Math.min(jitter.length - 1, last + 1)] : ([0, 0] as Pt);
		pts.push([cut.tipP[0] + j[0], cut.tipP[1] + j[1]]);
		nrm.push(cut.tipN);
		hws.push(cut.tipHW * hwScale);
		eLs.push(edge ? cut.tipE[0] : 0);
		eRs.push(edge ? cut.tipE[1] : 0);
	}
	if (pts.length < 2) return '';
	const m = pts.length;
	const left: Pt[] = [];
	const right: Pt[] = [];
	for (let i = 0; i < m; i++) {
		const hl = hws[i] * (1 + eLs[i]);
		const hr = hws[i] * (1 + eRs[i]);
		left.push([pts[i][0] + nrm[i][0] * hl, pts[i][1] + nrm[i][1] * hl]);
		right.push([pts[i][0] - nrm[i][0] * hr, pts[i][1] - nrm[i][1] * hr]);
	}
	const cap = (c: Pt, nv: Pt, r: number, forward: boolean): Pt[] => {
		// half circle from nv to -nv bulging along the tangent (tip) or against it (start). The start
		// is pushed out toward the upper left, where a right hand's brush comes down from.
		const t: Pt = forward ? [nv[1], -nv[0]] : [-nv[1], nv[0]]; // unit tangent of the stroke
		const tv: Pt = forward ? t : [-t[0], -t[1]];
		const k = Math.max(6, Math.min(16, Math.ceil(r * 1.6)));
		const out: Pt[] = [];
		for (let j = 1; j < k; j++) {
			const a = (Math.PI * j) / k;
			const dx = nv[0] * Math.cos(a) + tv[0] * Math.sin(a);
			const dy = nv[1] * Math.cos(a) + tv[1] * Math.sin(a);
			const push = forward ? 0 : entry * Math.pow(Math.max(0, -0.7071 * dx - 0.7071 * dy), 3);
			out.push([c[0] + r * dx * (1 + push), c[1] + r * dy * (1 + push)]);
		}
		return out;
	};
	const poly: Pt[] = [
		...left,
		...cap(pts[m - 1], nrm[m - 1], hws[m - 1], true),
		...right.reverse(),
		...cap(pts[0], [-nrm[0][0], -nrm[0][1]], hws[0], false),
	];
	let d = polyD(poly);
	const ccw = signedArea(poly) < 0;
	for (let i = 0; i < m; i += 2) {
		const gi = cut.idx[i] ?? last;
		if (g.curv[gi] * hws[i] > 0.7) d += polyD(circlePoly(pts[i], hws[i], ccw));
	}
	return d;
};

/**
 * Dry-brush gaps (飞白): a few bristle-parallel strips inside the body whose width follows the
 * dryness and a broken on/off noise, so the paper shows through in thin interrupted threads.
 */
export const gapPath = (g: StrokeGeo, cut: Cut, jitter: Pt[] | null, seed: number, width: number): string => {
	if (!cut.idx.length || Math.max(...cut.idx.map((i) => g.dry[i])) <= 0.02) return '';
	const K = Math.max(3, Math.min(10, Math.round(width / 1.4)));
	let d = '';
	for (let k = 0; k < K; k++) {
		const h = (q: number) => hash(seed * 31 + k, q);
		const o = -0.95 + (1.9 * (k + 0.5 + (h(1) - 0.5) * 0.7)) / K;
		const lam = 9 + 22 * h(2);
		const th = -0.3 + 0.6 * h(3);
		const gmax = width * (0.035 + 0.075 * h(4));
		let run: {c: Pt; n: Pt; w: number}[] = [];
		const flush = () => {
			if (run.length >= 3) {
				const left = run.map((r) => [r.c[0] + r.n[0] * r.w, r.c[1] + r.n[1] * r.w] as Pt);
				const right = run.map((r) => [r.c[0] - r.n[0] * r.w, r.c[1] - r.n[1] * r.w] as Pt).reverse();
				d += polyD([...left, ...right]);
			}
			run = [];
		};
		for (const i of cut.idx) {
			const on = smoothstep(th, th + 0.28, noise1(g.s[i] / lam, seed * 37 + k * 5));
			const w = 0.5 * g.dry[i] * on * gmax * Math.min(1, (2 * g.hw[i]) / width + 0.25);
			const j = jitter ? jitter[i] : ([0, 0] as Pt);
			const c: Pt = [g.P[i][0] + j[0] + g.N[i][0] * o * g.hw[i], g.P[i][1] + j[1] + g.N[i][1] * o * g.hw[i]];
			if (w > 0.06) run.push({c, n: g.N[i], w});
			else {
				if (run.length) run.push({c, n: g.N[i], w: 0});
				flush();
			}
		}
		flush();
	}
	return d;
};

/** Boil offsets: a smooth wander along the line, |offset| ≤ 0.6 px, new every boil tick. */
export const boilJitter = (g: StrokeGeo, tick: number, seed: number): Pt[] =>
	g.s.map((si) => [
		0.42 * noise1(si / 57 + tick * 3.1, seed * 19 + tick * 7 + 1),
		0.42 * noise1(si / 57 + tick * 5.7, seed * 23 + tick * 11 + 2),
	]);

/** First crossing of two polylines (for placing a light where two strokes meet). */
export const intersect = (a: Pt[], b: Pt[]): Pt | null => {
	for (let i = 1; i < a.length; i++)
		for (let j = 1; j < b.length; j++) {
			const p = a[i - 1];
			const r: Pt = [a[i][0] - p[0], a[i][1] - p[1]];
			const q = b[j - 1];
			const sv: Pt = [b[j][0] - q[0], b[j][1] - q[1]];
			const den = r[0] * sv[1] - r[1] * sv[0];
			if (Math.abs(den) < 1e-12) continue;
			const t = ((q[0] - p[0]) * sv[1] - (q[1] - p[1]) * sv[0]) / den;
			const u = ((q[0] - p[0]) * r[1] - (q[1] - p[1]) * r[0]) / den;
			if (t >= 0 && t <= 1 && u >= 0 && u <= 1) return [p[0] + t * r[0], p[1] + t * r[1]];
		}
	return null;
};
