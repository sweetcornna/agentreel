// A thread that stays alive after it is drawn: it keeps a wave travelling along its own length, so next to a still
// card the line is the one thing that breathes. The centre line is resampled by arc length and every point is pushed
// along the local normal by a wave running down the thread, zero at pinned ends, plus optional plucks (a bump that
// runs both ways from a point and dies away: the twang when it is pulled taut or something lands on it). Pure
// function of the frame.
import type {Pt} from './hand';

export type Pluck = {at: number; u: number; amp: number};
export type AliveOpts = {
	amp: number; // px
	wavelength?: number; // px
	speed?: number; // radians per frame
	seed?: number;
	pinStart?: boolean;
	pinEnd?: boolean;
	step?: number; // resample spacing, px
	plucks?: Pluck[];
	/** keep straight segments between the points (a drawn box's walls) instead of a spline through them */
	straight?: boolean;
};

/** Catmull-Rom through the control points (the hand strokes draw a spline through them too), densely. */
const spline = (pts: Pt[], sub = 10): Pt[] => {
	if (pts.length < 3) return pts;
	const out: Pt[] = [];
	for (let i = 0; i < pts.length - 1; i++) {
		const p0 = pts[Math.max(0, i - 1)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(pts.length - 1, i + 2)];
		for (let k = 0; k < sub; k++) {
			const t = k / sub, t2 = t * t, t3 = t2 * t;
			const c = (a: number, b: number, c_: number, d: number) => 0.5 * (2 * b + (-a + c_) * t + (2 * a - 5 * b + 4 * c_ - d) * t2 + (-a + 3 * b - 3 * c_ + d) * t3);
			out.push([c(p0[0], p1[0], p2[0], p3[0]), c(p0[1], p1[1], p2[1], p3[1])]);
		}
	}
	out.push(pts[pts.length - 1]);
	return out;
};

const resample = (pts: Pt[], step: number) => {
	const seg = pts.slice(1).map((q, i) => Math.hypot(q[0] - pts[i][0], q[1] - pts[i][1]));
	const total = seg.reduce((a, b) => a + b, 0);
	const n = Math.max(2, Math.round(total / step));
	const out: Pt[] = [];
	let i = 0;
	let acc = 0;
	for (let k = 0; k <= n; k++) {
		const d = (k / n) * total;
		while (i < seg.length - 1 && acc + seg[i] < d) acc += seg[i++];
		const t = seg[i] ? (d - acc) / seg[i] : 0;
		out.push([pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t]);
	}
	return {out, total};
};

export const alive = (pts: Pt[], f: number, o: AliveOpts): Pt[] => {
	const {amp, wavelength = 320, speed = 0.18, seed = 0, pinStart = true, pinEnd = true, step = 10, plucks = []} = o;
	const {out, total} = resample(o.straight ? pts : spline(pts), step);
	const n = out.length - 1;
	return out.map((p, k) => {
		const u = k / n;
		const a = out[Math.max(0, k - 1)];
		const b = out[Math.min(n, k + 1)];
		let nx = -(b[1] - a[1]);
		let ny = b[0] - a[0];
		const l = Math.hypot(nx, ny) || 1;
		nx /= l;
		ny /= l;
		const env = (pinStart ? Math.min(1, u * 6) : 1) * (pinEnd ? Math.min(1, (1 - u) * 6) : 1);
		const s = u * total;
		let d = amp * env * (Math.sin((2 * Math.PI * s) / wavelength - speed * f + seed) * 0.75 + Math.sin((2 * Math.PI * s) / (wavelength * 0.43) - speed * 1.7 * f + seed * 2.3) * 0.25);
		for (const pk of plucks) {
			const age = f - pk.at;
			if (age < 0 || age > 40) continue;
			const front = age * 0.035; // the bump runs out both ways
			const w = 0.05;
			const bump = Math.exp(-(((u - pk.u - front) / w) ** 2)) + Math.exp(-(((u - pk.u + front) / w) ** 2));
			d += pk.amp * env * bump * Math.exp(-age / 14) * Math.sin(age * 1.3);
		}
		return [p[0] + nx * d, p[1] + ny * d] as Pt;
	});
};
