// Film grain and vignette, laid over the whole picture at low strength so vector drawing reads as printed / filmed.
//
// The grain is generated in the page, not shipped as pictures: two tiles of pixel-sized grey noise (seeded mulberry32,
// identical on every run) larger than the frame, drawn 1:1 with putImageData at a fresh offset every frame. The tile
// and the offset are functions of the frame number only (rnd('gx' + frame), the source film's seeding), so a frame's
// grain does not depend on what was rendered before it. putImageData copies the bytes exactly (no filtering), and the
// tiles are opaque, so no premultiplied-alpha rounding enters.
import React, {useLayoutEffect, useRef} from 'react';
import {useCurrentFrame, useVideoConfig} from '../shim/remotion';
import {rnd} from './anim';

const PAD = 384;

const mulberry32 = (seed: number) => {
	let a = seed | 0;
	return () => {
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), a | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
};

const tiles = new Map<string, ImageData[]>();

/** Grey noise tiles of (w + PAD) × (h + PAD), mean 128; a sum of two uniforms (triangular), so grain clumps a little. */
const tilesFor = (w: number, h: number, seed: number, count: number): ImageData[] => {
	const key = `${w}x${h}:${seed}:${count}`;
	const hit = tiles.get(key);
	if (hit) return hit;
	const tw = w + PAD;
	const th = h + PAD;
	const out: ImageData[] = [];
	for (let k = 0; k < count; k++) {
		const r = mulberry32(seed * 7919 + k * 104729 + 1);
		const img = new ImageData(tw, th);
		const d = img.data;
		for (let i = 0; i < d.length; i += 4) {
			const v = Math.round(128 + 118 * (r() + r() - 1));
			d[i] = v;
			d[i + 1] = v;
			d[i + 2] = v;
			d[i + 3] = 255;
		}
		out.push(img);
	}
	tiles.set(key, out);
	return out;
};

export const Grain: React.FC<{
	opacity?: number;
	blend?: 'overlay' | 'soft-light' | 'multiply' | 'screen';
	seed?: number;
	/** the frame the grain follows (default: the current frame; pass a rounded output frame inside a Retimed picture) */
	frame?: number;
}> = ({opacity = 0.1, blend = 'overlay', seed = 1, frame}) => {
	const current = useCurrentFrame();
	const {width, height} = useVideoConfig();
	const f = frame ?? current;
	const ref = useRef<HTMLCanvasElement>(null);
	useLayoutEffect(() => {
		const c = ref.current;
		if (!c) return;
		const ctx = c.getContext('2d');
		if (!ctx) return;
		const set = tilesFor(width, height, seed, 2);
		const tile = set[Math.abs(Math.round(f)) % set.length];
		const dx = Math.floor(rnd('gx' + f) * PAD);
		const dy = Math.floor(rnd('gy' + f) * PAD);
		ctx.putImageData(tile, -dx, -dy);
	}, [f, width, height, seed]);
	if (!(opacity > 0)) return null;
	return (
		<canvas
			ref={ref}
			width={width}
			height={height}
			style={{position: 'absolute', left: 0, top: 0, width: '100%', height: '100%', mixBlendMode: blend, opacity, pointerEvents: 'none'}}
		/>
	);
};

export const Vignette: React.FC<{strength?: number; color?: string; inner?: number}> = ({strength = 0.55, color = '0,0,0', inner = 55}) => (
	<div
		style={{
			position: 'absolute',
			inset: 0,
			pointerEvents: 'none',
			background: `radial-gradient(ellipse 75% 70% at 50% 50%, rgba(${color},0) ${inner}%, rgba(${color},${strength}) 100%)`,
		}}
	/>
);
