// Thread helpers for paper scenes. What a hand-drawn thread does on paper:
//   · it weaves — over one card, under the next — so it is drawn in two layers: a back pass under the cards and front
//     pieces over them (`Threaded`);
//   · it goes slack (sags under its own weight) and twangs taut (`sag`, and alive() plucks);
//   · it wraps a card and becomes its outline (`outline`);
//   · when something lands, a few pencilled strokes burst round it (`Burst`), on twos;
//   · it changes its mind: one shape pulled into another, point for point along its length (`blendPaths`).
import React from 'react';
import {Stroke} from './hand';
import type {Pt} from './hand';
import {palette} from './palette';

/** Evenly resampled points (the output of alive() or resampleN()) sliced by fraction of length. */
export const slice = (pts: Pt[], u0: number, u1: number): Pt[] => {
	const n = pts.length - 1;
	const a = Math.max(0, Math.floor(u0 * n));
	const b = Math.min(n, Math.ceil(u1 * n));
	return pts.slice(a, b + 1);
};

/** Polyline length. */
export const pathLength = (pts: Pt[]): number => {
	let l = 0;
	for (let i = 1; i < pts.length; i++) l += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
	return l;
};

/** Exactly `n` + 1 points evenly spaced by arc length along a polyline (ends kept). */
export const resampleN = (pts: Pt[], n: number): Pt[] => {
	if (pts.length < 2) return Array.from({length: n + 1}, () => [pts[0]?.[0] ?? 0, pts[0]?.[1] ?? 0] as Pt);
	const seg = pts.slice(1).map((q, i) => Math.hypot(q[0] - pts[i][0], q[1] - pts[i][1]));
	const total = seg.reduce((a, b) => a + b, 0);
	const out: Pt[] = [];
	let i = 0;
	let acc = 0;
	for (let k = 0; k <= n; k++) {
		const d = (k / n) * total;
		while (i < seg.length - 1 && acc + seg[i] < d) acc += seg[i++];
		const t = seg[i] ? Math.min(1, Math.max(0, (d - acc) / seg[i])) : 0;
		out.push([pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t]);
	}
	return out;
};

/**
 * One path pulled into another: both are resampled to the same number of points by arc length and blended point for
 * point. `k` is 0..1; `lag(u)` (u = 0..1 along the path) delays parts of the line so it is pulled from somewhere
 * rather than morphed all at once: e.g. u => Math.abs(u - 0.5) * 0.4 pulls from both ends toward the middle last.
 */
export const blendPaths = (a: Pt[], b: Pt[], k: number, o: {n?: number; lag?: (u: number) => number; ease?: (t: number) => number} = {}): Pt[] => {
	const n = o.n ?? 240;
	const A = resampleN(a, n);
	const B = resampleN(b, n);
	const e = o.ease ?? ((t: number) => t * t * (3 - 2 * t));
	return A.map((p, i) => {
		const u = i / n;
		const lag = o.lag ? o.lag(u) : 0;
		const t = e(Math.min(1, Math.max(0, (k - lag) / Math.max(1e-6, 1 - lag))));
		return [p[0] + (B[i][0] - p[0]) * t, p[1] + (B[i][1] - p[1]) * t] as Pt;
	});
};

/**
 * A thread drawn at an even pace (`progress` 0..1 of its length) with the stretches in `front` drawn over the cards;
 * render <Threaded layer="back"/> before the cards and <Threaded layer="front"/> after them.
 */
export const Threaded: React.FC<{
	pts: Pt[];
	progress: number;
	front?: [number, number][];
	layer: 'back' | 'front';
	width?: number;
	seed?: number;
	opacity?: number;
	color?: string;
}> = ({pts, progress, front = [], layer, width = 3.6, seed = 1, opacity = 1, color = palette.thread}) => {
	if (progress <= 0) return null;
	if (layer === 'back') return <Stroke pts={pts} progress={progress} width={width} color={color} texture="cord" seed={seed} taper={[0.015, 0]} pace="linear" opacity={opacity} />;
	return (
		<>
			{front.map(([u0, u1], i) => {
				if (progress <= u0) return null;
				const piece = slice(pts, u0, u1);
				const k = Math.min(1, (progress - u0) / (u1 - u0));
				return <Stroke key={i} pts={piece} progress={k} width={width} color={color} texture="cord" seed={seed + i + 1} taper={[0, 0]} pace="linear" opacity={opacity} />;
			})}
		</>
	);
};

/** Sag: push points down by a catenary-ish bow between the ends, `amt` px at the middle. */
export const sag = (pts: Pt[], amt: number): Pt[] => {
	const n = pts.length - 1;
	return pts.map(([x, y], i) => {
		const u = i / n;
		return [x, y + amt * 4 * u * (1 - u)] as Pt;
	});
};

/** A hand-drawn rounded rectangle round a w × h box at the origin, starting at the lower left and overshooting a little. */
export const outline = (w: number, h: number, pad = 10, r = 18, seed = 1): Pt[] => {
	const x0 = -pad,
		y0 = -pad,
		x1 = w + pad,
		y1 = h + pad;
	const pts: Pt[] = [];
	const arc = (cx: number, cy: number, a0: number, a1: number) => {
		for (let i = 0; i <= 4; i++) {
			const a = a0 + ((a1 - a0) * i) / 4;
			pts.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]);
		}
	};
	const j = (k: number) => Math.sin(seed * 12.9 + k * 7.3) * 2.2; // the hand is not a ruler
	pts.push([x0 + r + 6, y1 + j(1)]);
	arc(x0 + r, y1 - r, Math.PI / 2, Math.PI);
	arc(x0 + r, y0 + r, Math.PI, (3 * Math.PI) / 2);
	pts.push([(x0 + x1) / 2, y0 + j(2)]);
	arc(x1 - r, y0 + r, (3 * Math.PI) / 2, 2 * Math.PI);
	arc(x1 - r, y1 - r, 0, Math.PI / 2);
	pts.push([x0 + r + 40, y1 + 4 + j(3)]); // past the start, as a hand closing a loop does
	return pts;
};

/** Pencilled strokes bursting round a point: they shoot out over a few frames and are gone in half a second. */
export const Burst: React.FC<{
	f: number;
	at: number;
	x: number;
	y: number;
	r?: number;
	n?: number;
	seed?: number;
	width?: number;
	color?: string;
	texture?: 'cord' | 'pencil' | 'ink';
}> = ({f, at, x, y, r = 60, n = 8, seed = 1, width = 2.6, color = palette.thread, texture = 'cord'}) => {
	const t = Math.floor((f - at) / 2) * 2; // on twos
	if (t < 0 || t > 14) return null;
	const out = Math.min(1, t / 5);
	const gone = Math.max(0, (t - 6) / 8);
	return (
		<>
			{Array.from({length: n}, (_, i) => {
				const a = (i / n) * Math.PI * 2 + Math.sin(seed * 3.7 + i * 1.9) * 0.25;
				const r0 = r * (0.62 + 0.5 * gone);
				const r1 = r * (0.62 + 0.45 * out) + (i % 2 ? 0 : r * 0.18);
				if (r1 - r0 < 2) return null;
				const p0: Pt = [x + Math.cos(a) * r0, y + Math.sin(a) * r0];
				const p1: Pt = [x + Math.cos(a) * r1, y + Math.sin(a) * r1];
				return <Stroke key={i} pts={[p0, p1]} progress={1} width={width} color={color} texture={texture} seed={seed * 10 + i} taper={[0.2, 0.3]} pace="linear" smooth={0} />;
			})}
		</>
	);
};
