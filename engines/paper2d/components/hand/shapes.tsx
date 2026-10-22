// 常用手绘标记：圈注、下划线、箭头、勾、切断的一笔。都是 Stroke 的薄封装，
// 形状里的不规则全部由 seed 决定。
import React from 'react';
import {hash, roughen} from './geom';
import type {Pace, Pt, Texture} from './geom';
import {Stroke} from './Stroke';
import {palette} from '../palette';

/** Props every hand mark shares (beyond its own geometry). */
export type MarkProps = {
	progress: number;
	width?: number;
	color?: string;
	seed?: number;
	texture?: Texture;
	taper?: [number, number];
	boil?: boolean;
	boilStep?: number;
	opacity?: number;
	smooth?: number;
	pace?: Pace;
	dry?: number;
	head?: boolean;
	inline?: boolean;
	style?: React.CSSProperties;
};
const INK = palette.ink;
const seg = (p: number, a: number, b: number) => Math.min(1, Math.max(0, (p - a) / (b - a)));
const pick = ({width = 3, color = INK, seed = 1, ...rest}: MarkProps) => ({width, color, seed, ...rest});

/** Loose ring: slightly elliptical, tilted, the end overlaps the start by 15–25° on a slightly different radius. */
export const HandCircle: React.FC<MarkProps & {cx: number; cy: number; r: number; squash?: number}> = ({cx, cy, r, squash, ...m}) => {
	const o = pick(m);
	const h = (k: number) => hash(o.seed * 3 + 11, k);
	const e = squash ?? 0.04 + 0.08 * h(1);
	const rot = (h(2) - 0.5) * 0.42;
	const start = (-150 + 50 * h(3)) * (Math.PI / 180); // upper left, the way a right hand starts a ring
	const sweep = (2 * Math.PI * (360 + 15 + 10 * h(4))) / 360;
	const drift = (h(5) < 0.5 ? -1 : 1) * (0.045 + 0.04 * h(6)); // end lands inside or outside the start
	const n = Math.max(24, Math.ceil(sweep * r * 0.2));
	const pts: Pt[] = [];
	for (let i = 0; i <= n; i++) {
		const t = i / n;
		const a = start - sweep * t; // counter-clockwise on screen
		const rr = r * (1 + drift * t * t + 0.025 * Math.sin(a * 2 + h(7) * 6));
		const x = rr * (1 + e) * Math.cos(a);
		const y = rr * (1 - e) * Math.sin(a);
		pts.push([cx + x * Math.cos(rot) - y * Math.sin(rot), cy + x * Math.sin(rot) + y * Math.cos(rot)]);
	}
	return <Stroke {...o} pts={roughen(pts, Math.min(2.2, r * 0.03), o.seed)} taper={o.taper ?? [0.05, 0.1]} />;
};

/** Underline from x1 to x2 at y: a slight belly, rising a little to the right, ends overshoot. */
export const HandUnderline: React.FC<MarkProps & {x1: number; x2: number; y: number}> = ({x1, x2, y, ...m}) => {
	const o = pick(m);
	const h = (k: number) => hash(o.seed * 5 + 7, k);
	const L = Math.abs(x2 - x1);
	const rise = L * (0.006 + 0.012 * h(1));
	const sag = L * (0.004 + 0.01 * h(2));
	const a = x1 - L * (0.01 + 0.02 * h(3));
	const b = x2 + L * (0.015 + 0.03 * h(4));
	const pts: Pt[] = Array.from({length: 9}, (_, i) => {
		const t = i / 8;
		return [a + (b - a) * t, y + rise * (0.5 - t) + sag * Math.sin(Math.PI * t)] as Pt;
	});
	return <Stroke {...o} pts={roughen(pts, Math.min(1.6, L * 0.006), o.seed)} taper={o.taper ?? [0.04, 0.22]} />;
};

/** Arrow: a bent shaft, then the head drawn as one quick stroke (wing → tip → wing). */
export const HandArrow: React.FC<MarkProps & {from: Pt; to: Pt; bend?: number; headSize?: number}> = ({
	from,
	to,
	bend = 0.12,
	headSize,
	...m
}) => {
	const o = pick(m);
	const h = (k: number) => hash(o.seed * 7 + 3, k);
	const dx = to[0] - from[0];
	const dy = to[1] - from[1];
	const L = Math.max(1, Math.hypot(dx, dy));
	const nx = -dy / L;
	const ny = dx / L;
	const shaft: Pt[] = Array.from({length: 13}, (_, i) => {
		const t = i / 12;
		const off = bend * L * 4 * t * (1 - t) * 0.5;
		const stop = 0.985;
		return [from[0] + dx * t * stop + nx * off, from[1] + dy * t * stop + ny * off] as Pt;
	});
	// head along the shaft's arrival direction
	const e = shaft[shaft.length - 1];
	const f = shaft[shaft.length - 3];
	const ang = Math.atan2(e[1] - f[1], e[0] - f[0]);
	const hs = headSize ?? Math.min(34, Math.max(11, L * 0.11)) + o.width * 1.4;
	const wing = (side: number, k: number): Pt => {
		const a = ang + Math.PI + side * (0.42 + 0.14 * h(k));
		const l = hs * (0.85 + 0.3 * h(k + 1));
		return [to[0] + l * Math.cos(a), to[1] + l * Math.sin(a)];
	};
	const headPts: Pt[] = [wing(1, 1), [to[0], to[1]], wing(-1, 3)];
	const pShaft = seg(o.progress, 0, 0.72);
	const pHead = seg(o.progress, 0.76, 1);
	return (
		<>
			<Stroke {...o} pts={roughen(shaft, Math.min(2, L * 0.008), o.seed)} progress={pShaft} taper={o.taper ?? [0.05, 0.12]} />
			<Stroke {...o} pts={headPts} progress={pHead} seed={o.seed + 0.5} smooth={o.smooth ?? 0.12} taper={[0.1, 0.3]} dry={0} />
		</>
	);
};

/** Tick (勾) centred on (x, y), `size` wide: a short press down, then a long flick up to the right. */
export const HandTick: React.FC<MarkProps & {x: number; y: number; size: number}> = ({x, y, size, ...m}) => {
	const o = pick(m);
	const h = (k: number) => hash(o.seed * 11 + 5, k);
	const s = size;
	const j = (k: number) => (h(k) - 0.5) * 0.08 * s;
	const pts: Pt[] = [
		[x - 0.46 * s + j(1), y - 0.02 * s + j(2)],
		[x - 0.3 * s + j(3), y + 0.16 * s + j(4)],
		[x - 0.14 * s, y + 0.34 * s],
		[x + 0.12 * s + j(5), y + 0.02 * s + j(6)],
		[x + 0.5 * s + j(7), y - 0.44 * s + j(8)],
	];
	return <Stroke {...o} pts={pts} smooth={o.smooth ?? 0.34} taper={o.taper ?? [0.06, 0.42]} />;
};

/** One decisive cut: nearly straight with a faint bow, pressed in and lifted off at speed. */
export const HandSlash: React.FC<MarkProps & {from: Pt; to: Pt}> = ({from, to, ...m}) => {
	const o = pick(m);
	const h = (k: number) => hash(o.seed * 13 + 9, k);
	const dx = to[0] - from[0];
	const dy = to[1] - from[1];
	const L = Math.max(1, Math.hypot(dx, dy));
	const bow = (h(1) - 0.5) * 0.05 * L;
	const pts: Pt[] = Array.from({length: 7}, (_, i) => {
		const t = i / 6;
		const off = bow * 4 * t * (1 - t);
		return [from[0] + dx * t - (dy / L) * off, from[1] + dy * t + (dx / L) * off] as Pt;
	});
	return <Stroke {...o} pts={roughen(pts, Math.min(1.4, L * 0.004), o.seed)} pace={o.pace ?? 'flick'} taper={o.taper ?? [0.04, 0.45]} />;
};
