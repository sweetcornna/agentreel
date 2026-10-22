// The page itself, drawn by hand, and the cards that lie on it. A real sheet is not a flat colour with a CSS grid: it
// has its texture at the middle scales (broad soft blotches, fibres laid one way, a few specks — not the even fine noise
// of a digital grain), and its ruling is pencilled — lines a little crooked, some darker, some fading out — and the
// ruling can change pattern when the state changes (square grid → diagonal lattice → dots). All deterministic: the
// noise is seeded SVG turbulence, the pencil wobble a hash of the line's index.
import React, {useId} from 'react';
import {Img, staticFile, useCurrentFrame, useVideoConfig} from '../shim/remotion';
import {ease, ramp} from './anim';
import {palette} from './palette';

export type Ruling = 'grid' | 'lattice' | 'dots';

const hash = (n: number) => {
	const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
	return x - Math.floor(x);
};
/** A pencilled line from a to b: a slow wander plus a hand tremor, as an SVG path. */
const pencil = (ax: number, ay: number, bx: number, by: number, seed: number) => {
	const n = 14;
	const len = Math.hypot(bx - ax, by - ay);
	const nx = -(by - ay) / len;
	const ny = (bx - ax) / len;
	let d = '';
	for (let i = 0; i <= n; i++) {
		const t = i / n;
		const w = 1.6 * Math.sin(t * 5.3 + seed * 3.1) + 0.7 * Math.sin(t * 17 + seed * 7.7);
		d += `${i ? 'L' : 'M'}${(ax + (bx - ax) * t + nx * w).toFixed(1)} ${(ay + (by - ay) * t + ny * w).toFixed(1)}`;
	}
	return d;
};

const rgbOf = (hex: string): [number, number, number] => {
	const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
	if (!m) throw new Error(`HandPaper colours must be #rrggbb, got ${hex}`);
	const v = parseInt(m[1], 16);
	return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
};

/**
 * The sheet. Size: the composition's (or width / height).
 * @param shiftX/shiftY  world scroll (px) — the ruling moves with the camera
 * @param drawn  0..1: the ruling being pencilled in (lines appear in a quick stagger); 0 = no ruling
 * @param grain  0..1: the texture developing; at 0 the page is the flat `flat` colour, at 1 the textured `stock`
 * @param mix    weights of the three rulings (they cross-fade on a state change)
 * @param step   ruling pitch in px (about 80 at 1080p reads as a notebook grid)
 */
export const HandPaper: React.FC<{
	shiftX?: number;
	shiftY?: number;
	drawn?: number;
	grain?: number;
	mix?: Partial<Record<Ruling, number>>;
	step?: number;
	width?: number;
	height?: number;
	flat?: string;
	stock?: string;
	/** rgb triple of the ruling's pencil, e.g. '92,78,60' */
	pencilRgb?: string;
	/** texture strength multiplier (1 = the source film's sheet) */
	texture?: number;
}> = ({shiftX = 0, shiftY = 0, drawn = 0, grain = 1, mix = {}, step = 80, width, height, flat = palette.paper, stock = palette.stock, pencilRgb = '92,78,60', texture = 1}) => {
	const cfg = useVideoConfig();
	const W = width ?? cfg.width;
	const H = height ?? cfg.height;
	const id = useId().replace(/[^a-zA-Z0-9_-]/g, '');
	const STEP = step;
	const m = {grid: mix.grid ?? 0, lattice: mix.lattice ?? 0, dots: mix.dots ?? 0};
	const ox = ((-shiftX % STEP) + STEP) % STEP;
	const oy = ((-shiftY % STEP) + STEP) % STEP;
	// which line of the page this is, so the same line keeps its wobble while the page scrolls
	const col0 = Math.floor(shiftX / STEP);
	const row0 = Math.floor(shiftY / STEP);
	const lines: React.ReactNode[] = [];
	if (m.grid > 0 && drawn > 0) {
		for (let i = -1; i <= W / STEP + 1; i++) {
			const lid = col0 + i;
			const k = hash(lid * 1.3);
			const show = Math.min(1, Math.max(0, drawn * 1.6 - k * 0.6));
			if (show <= 0) continue;
			const x = ox + i * STEP;
			lines.push(<path key={`v${lid}`} d={pencil(x, -10, x + (k - 0.5) * 3, -10 + (H + 20) * show, lid)} stroke={`rgba(${pencilRgb},${0.16 + 0.1 * hash(lid * 7.1)})`} strokeWidth={1.5 + 0.7 * hash(lid * 3.3)} fill="none" opacity={m.grid} />);
		}
		for (let j = -1; j <= H / STEP + 1; j++) {
			const lid = row0 + j;
			const k = hash(lid * 2.9 + 50);
			const show = Math.min(1, Math.max(0, drawn * 1.6 - k * 0.6));
			if (show <= 0) continue;
			const y = oy + j * STEP;
			lines.push(<path key={`h${lid}`} d={pencil(-10, y, -10 + (W + 20) * show, y + (k - 0.5) * 3, lid + 99)} stroke={`rgba(${pencilRgb},${0.16 + 0.1 * hash(lid * 5.7)})`} strokeWidth={1.5 + 0.7 * hash(lid * 4.1)} fill="none" opacity={m.grid} />);
		}
	}
	if (m.lattice > 0 && drawn > 0) {
		const L = STEP * 1.2;
		const lx = ((-shiftX % L) + L) % L;
		for (let i = -Math.ceil(H / L) - 2; i <= Math.ceil(W / L) + 2; i++) {
			const x = lx + i * L;
			lines.push(<path key={`a${i}`} d={pencil(x, -20, x + H + 40, H + 20, i + 300)} stroke={`rgba(${pencilRgb},0.2)`} strokeWidth={1.6} fill="none" opacity={m.lattice * Math.min(1, drawn)} />);
			lines.push(<path key={`b${i}`} d={pencil(x, H + 20, x + H + 40, -20, i + 600)} stroke={`rgba(${pencilRgb},0.2)`} strokeWidth={1.6} fill="none" opacity={m.lattice * Math.min(1, drawn)} />);
		}
	}
	// the sheet settles from the flat colour to the textured stock as `grain` rises
	const [fr, fg, fb] = rgbOf(flat);
	const [sr, sg, sb] = rgbOf(stock);
	const mixC = (a: number, b: number) => Math.round(a + (b - a) * grain);
	const base = `rgb(${mixC(fr, sr)},${mixC(fg, sg)},${mixC(fb, sb)})`;
	const tx = Math.max(0, texture);
	return (
		<svg width={W} height={H} style={{position: 'absolute', left: 0, top: 0}}>
			<defs>
				<filter id={`${id}m`} x="0" y="0" width="100%" height="100%">
					<feTurbulence type="fractalNoise" baseFrequency="0.006 0.008" numOctaves={4} seed={11} />
					<feColorMatrix type="matrix" values="0 0 0 0 0.42  0 0 0 0 0.36  0 0 0 0 0.26  1.3 0 0 0 -0.5" />
				</filter>
				<filter id={`${id}f`} x="0" y="0" width="100%" height="100%">
					<feTurbulence type="fractalNoise" baseFrequency="0.09 0.1" numOctaves={3} seed={5} />
					<feColorMatrix type="matrix" values="0 0 0 0 0.38  0 0 0 0 0.32  0 0 0 0 0.22  1.6 0 0 0 -0.72" />
				</filter>
				<filter id={`${id}l`} x="0" y="0" width="100%" height="100%">
					<feTurbulence type="fractalNoise" baseFrequency="0.035 0.045" numOctaves={2} seed={23} />
					<feColorMatrix type="matrix" values="0 0 0 0 1  0 0 0 0 0.99  0 0 0 0 0.95  1.8 0 0 0 -0.95" />
				</filter>
				<filter id={`${id}s`} x="0" y="0" width="100%" height="100%">
					<feTurbulence type="fractalNoise" baseFrequency="0.55" numOctaves={1} seed={31} />
					<feColorMatrix type="matrix" values="0 0 0 0 0.25  0 0 0 0 0.2  0 0 0 0 0.15  0 30 0 0 -22.4" />
				</filter>
				{/* ruling printed on the sheet: a little soft, not a vector hairline */}
				<filter id={`${id}i`} x="-2%" y="-2%" width="104%" height="104%">
					<feGaussianBlur stdDeviation={0.7} />
				</filter>
				<pattern id={`${id}d`} width={24} height={24} patternUnits="userSpaceOnUse" x={-shiftX % 24} y={-shiftY % 24}>
					<circle cx={12} cy={12} r={1.9} fill={`rgba(${pencilRgb},0.3)`} />
				</pattern>
			</defs>
			<rect width={W} height={H} fill={base} />
			{grain > 0 && tx > 0 ? (
				<g opacity={grain}>
					<rect width={W} height={H} filter={`url(#${id}m)`} opacity={Math.min(1, 0.32 * tx)} />
					<rect width={W} height={H} filter={`url(#${id}f)`} opacity={Math.min(1, 0.3 * tx)} />
					<rect width={W} height={H} filter={`url(#${id}l)`} opacity={Math.min(1, 0.35 * tx)} />
					<rect width={W} height={H} filter={`url(#${id}s)`} opacity={Math.min(1, 0.5 * tx)} />
				</g>
			) : null}
			<g filter={`url(#${id}i)`}>{lines}</g>
			{m.dots > 0 ? <rect width={W} height={H} fill={`url(#${id}d)`} opacity={m.dots} /> : null}
		</svg>
	);
};

/** A big soft shadow (a card lying on the page) and a small one (a chip). */
export const shadow = (k = 1, rgb = palette.shadow) =>
	k <= 0 ? 'none' : `0 ${16 * k}px ${38 * k}px rgba(${rgb},${0.15 * k}), 0 ${2 * k}px ${6 * k}px rgba(${rgb},${0.08 * k})`;
export const SHADOW = shadow(1);
export const SHADOW_S = `0 8px 18px rgba(${palette.shadow},0.13), 0 1px 3px rgba(${palette.shadow},0.08)`;

/** A card lying on the page, positioned by its centre, with the big soft shadow (`lift` scales it; 0 = flat). */
export const PaperCard: React.FC<{
	x: number;
	y: number;
	w: number;
	h: number;
	rot?: number;
	scale?: number;
	opacity?: number;
	blur?: number;
	origin?: string;
	radius?: number;
	bg?: string;
	lift?: number;
	border?: string;
	children?: React.ReactNode;
	style?: React.CSSProperties;
}> = ({x, y, w, h, rot = 0, scale = 1, opacity = 1, blur = 0, origin = '50% 50%', radius = 14, bg = palette.card, lift = 1, border = `1px solid rgba(${palette.shadow},0.07)`, children, style}) => (
	<div
		style={{
			position: 'absolute', left: x - w / 2, top: y - h / 2, width: w, height: h, borderRadius: radius, background: bg, boxShadow: shadow(lift),
			border, boxSizing: 'border-box', transform: `rotate(${rot}deg) scale(${scale})`, transformOrigin: origin,
			opacity, filter: blur > 0.3 ? `blur(${blur}px)` : undefined, ...style,
		}}
	>
		{children}
	</div>
);

/** An image from public/ filling the frame, with a slow camera move (zoom and pan, eased). */
export const Plate: React.FC<{
	src: string;
	from?: number; // local frame the move starts
	dur: number; // move length in frames
	zoom?: [number, number];
	pan?: [[number, number], [number, number]]; // px offsets at start/end
	origin?: string;
	style?: React.CSSProperties;
	e?: (t: number) => number;
}> = ({src, from = 0, dur, zoom = [1.04, 1.1], pan = [[0, 0], [0, 0]], origin = '50% 50%', style, e = ease.soft}) => {
	const frame = useCurrentFrame();
	const t = ramp(frame, from, from + dur, e);
	const s = zoom[0] + (zoom[1] - zoom[0]) * t;
	const px = pan[0][0] + (pan[1][0] - pan[0][0]) * t;
	const py = pan[0][1] + (pan[1][1] - pan[0][1]) * t;
	return (
		<div style={{position: 'absolute', inset: 0, overflow: 'hidden'}}>
			<Img
				src={staticFile(src)}
				style={{width: '100%', height: '100%', objectFit: 'cover', transform: `translate(${px}px, ${py}px) scale(${s})`, transformOrigin: origin, ...style}}
			/>
		</div>
	);
};
