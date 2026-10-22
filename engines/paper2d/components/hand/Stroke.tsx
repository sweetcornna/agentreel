// 一笔手绘笔触。几何在 geom.ts（宽度随笔速与压力起伏、起笔略重、收笔出锋、干笔飞白缝），
// 材质在这里用 SVG 滤镜完成（feTurbulence + feDisplacementMap + 遮罩），全部种子化、逐帧确定。
import React, {useId} from 'react';
import {useCurrentFrame} from '../../shim/remotion';
import {bodyPath, boilJitter, cutAt, gapPath, pacePos, PRESETS, strokeGeo} from './geom';
import type {Pace, Pt, Texture} from './geom';

export type {Pace, Pt, Texture} from './geom';

/** A seed that changes every `step` frames (default 3) — drives the line boil. */
export function useBoil(step = 3): number {
	const f = useCurrentFrame();
	return Math.floor(f / Math.max(1, step));
}

export type StrokeProps = {
	/** Centre line in px, in the coordinates of the nearest positioned ancestor (or parent <svg> when inline). */
	pts: Pt[];
	/** 0..1, time-linear; the hand-speed curve (slow start, fast middle, slow end) is applied inside. */
	progress: number;
	/** Widest point of the stroke in px. */
	width: number;
	color: string;
	texture?: Texture;
	/** [start, end] taper; values ≤ 1 are fractions of the length, larger values are px. */
	taper?: [number, number];
	seed?: number;
	/** Redraw the tiny wobble every 3 frames (≤ 0.6 px). */
	boil?: boolean;
	opacity?: number;
	/** 0 = straight segments through pts, 0.5 = spline through pts (default), 1 = extra smoothing. */
	smooth?: number;
	// ---- extras
	/** Speed curve used for drawing and for the width: 'hand' (default), 'flick' (accelerates off the page), 'linear'. */
	pace?: Pace;
	boilStep?: number;
	/** Dry-brush amount override (0 = none; default depends on texture). */
	dry?: number;
	/** Show the ink bead / bright head at the pen tip while drawing (default true). */
	head?: boolean;
	/** Render a bare <g> for use inside an existing <svg> (same coordinates). */
	inline?: boolean;
	style?: React.CSSProperties;
};

const lighten = (hex: string, k: number) => {
	const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
	if (!m) return '#FFF4DE';
	const v = parseInt(m[1], 16);
	const c = [(v >> 16) & 255, (v >> 8) & 255, v & 255].map((x) => Math.round(x + (255 - x) * k));
	return '#' + c.map((x) => x.toString(16).padStart(2, '0')).join('');
};

const matrixA = (k: number, c: number) => `0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  ${k} 0 0 0 ${c}`;

/** The texture filter. All primitives work in user space so the noise stays glued to the page. */
const TextureFilter: React.FC<{
	id: string;
	texture: Texture;
	box: [number, number, number, number];
	width: number;
	seed: number;
}> = ({id, texture, box, width, seed}) => {
	const [x, y, w, h] = box;
	const s = (k: number) => (Math.abs(Math.round(seed * 97 + k * 131)) % 9973) + 1;
	// fibre displacement: enough to feel the paper, never enough to fuzz a hairline
	const fib = Math.min(1.1 + width * 0.07, 0.6 + width * 0.45);
	const hair = 1 - Math.min(1, Math.max(0, (width - 2) / 4)); // 1 for hairlines, 0 from 6 px up
	const common = {id, filterUnits: 'userSpaceOnUse' as const, x, y, width: w, height: h, colorInterpolationFilters: 'sRGB' as const};
	if (texture === 'ink')
		return (
			<filter {...common}>
				{/* paper fibres tug at the edge; a slower wave keeps it from looking machine-cut */}
				<feTurbulence type="fractalNoise" baseFrequency="1.15" numOctaves={2} seed={s(1)} result="fib" />
				<feDisplacementMap in="SourceGraphic" in2="fib" scale={fib} xChannelSelector="R" yChannelSelector="G" result="edge" />
				<feTurbulence type="fractalNoise" baseFrequency="0.12" numOctaves={2} seed={s(2)} result="wave" />
				<feDisplacementMap in="edge" in2="wave" scale={0.7 + width * 0.05} xChannelSelector="G" yChannelSelector="B" result="edge2" />
				{/* ink density: dense, with a slow mottle */}
				<feTurbulence type="fractalNoise" baseFrequency="0.035" numOctaves={3} seed={s(3)} result="tone" />
				<feColorMatrix in="tone" type="matrix" values={matrixA(0.6 + hair * 1.2, 0.68 - hair * 0.5)} result="toneA" />
				<feComposite in="edge2" in2="toneA" operator="in" result="body" />
				{/* the thin wet margin the paper drinks from the line */}
				<feGaussianBlur in="edge2" stdDeviation={0.55 + width * 0.03} result="soak" />
				<feColorMatrix in="soak" type="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 0.3 0" result="soakA" />
				<feMerge>
					<feMergeNode in="soakA" />
					<feMergeNode in="body" />
				</feMerge>
			</filter>
		);
	if (texture === 'cord')
		return (
			<filter {...common}>
				{/* the pen's edge on paper: a whisper of fibre, and a faint density wave along the line */}
				<feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves={1} seed={s(1)} result="fib" />
				<feDisplacementMap in="SourceGraphic" in2="fib" scale={0.55 + width * 0.04} xChannelSelector="R" yChannelSelector="G" result="edge" />
				<feTurbulence type="fractalNoise" baseFrequency="0.02" numOctaves={2} seed={s(3)} result="tone" />
				<feColorMatrix in="tone" type="matrix" values={matrixA(0.25, 0.86)} result="toneA" />
				<feComposite in="edge" in2="toneA" operator="in" />
			</filter>
		);
	if (texture === 'pencil')
		return (
			<filter {...common}>
				<feTurbulence type="fractalNoise" baseFrequency="0.42" numOctaves={2} seed={s(1)} result="en" />
				<feDisplacementMap in="SourceGraphic" in2="en" scale={1.6 + width * 0.25} xChannelSelector="R" yChannelSelector="G" result="d" />
				{/* graphite sits on the paper tooth: a hard-thresholded fine grain */}
				<feTurbulence type="fractalNoise" baseFrequency="1.25" numOctaves={2} seed={s(2)} result="gn" />
				<feColorMatrix in="gn" type="matrix" values={matrixA(4.2, -1.35)} result="gA" />
				<feTurbulence type="fractalNoise" baseFrequency="0.09" numOctaves={2} seed={s(3)} result="lay" />
				<feColorMatrix in="lay" type="matrix" values={matrixA(1.2, 0.2)} result="layA" />
				<feComposite in="gA" in2="layA" operator="arithmetic" k1={1} k2={0} k3={0} k4={0} result="grain" />
				<feComposite in="d" in2="grain" operator="in" />
			</filter>
		);
	if (texture === 'brush')
		return (
			<filter {...common}>
				<feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves={2} seed={s(1)} result="fib" />
				<feDisplacementMap in="SourceGraphic" in2="fib" scale={fib * 0.8} xChannelSelector="R" yChannelSelector="G" result="core" />
				{/* wet bleed: the edge creeps out into the paper in an irregular feathered band */}
				<feGaussianBlur in="SourceGraphic" stdDeviation={0.8 + width * 0.08} result="blur" />
				<feTurbulence type="fractalNoise" baseFrequency="0.09" numOctaves={3} seed={s(2)} result="wet" />
				<feDisplacementMap in="blur" in2="wet" scale={3 + width * 0.45} xChannelSelector="R" yChannelSelector="G" result="spread" />
				<feTurbulence type="fractalNoise" baseFrequency="0.6" numOctaves={2} seed={s(3)} result="feather" />
				<feDisplacementMap in="spread" in2="feather" scale={2.4} xChannelSelector="R" yChannelSelector="G" result="spread2" />
				<feColorMatrix in="spread2" type="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 1.6 -0.35" result="bleed" />
				<feColorMatrix in="bleed" type="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 0.3 0" result="bleedA" />
				{/* pigment pools at the edge as the water dries */}
				<feMorphology in="core" operator="erode" radius={Math.max(0.5, width * 0.09)} result="inner" />
				<feComposite in="core" in2="inner" operator="out" result="rim" />
				<feColorMatrix in="core" type="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 0.8 0" result="body" />
				<feMerge>
					<feMergeNode in="bleedA" />
					<feMergeNode in="body" />
					<feMergeNode in="rim" />
				</feMerge>
			</filter>
		);
	// gold: a grainy metal thread with two layers of glow and a brighter core
	return (
		<filter {...common}>
			<feTurbulence type="fractalNoise" baseFrequency="1.1" numOctaves={1} seed={s(1)} result="gr" />
			<feColorMatrix in="gr" type="matrix" values={matrixA(1.6, 0.2)} result="grA" />
			<feComposite in="SourceGraphic" in2="grA" operator="in" result="line" />
			<feGaussianBlur in="SourceGraphic" stdDeviation={0.9 + width * 0.9} result="b1" />
			<feColorMatrix in="b1" type="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 1.5 0" result="g1" />
			<feGaussianBlur in="SourceGraphic" stdDeviation={4 + width * 3.2} result="b2" />
			<feColorMatrix in="b2" type="matrix" values="0 0 0 0 0.96  0 0 0 0 0.68  0 0 0 0 0.36  0 0 0 2.6 0" result="g2" />
			<feMerge>
				<feMergeNode in="g2" />
				<feMergeNode in="g1" />
				<feMergeNode in="line" />
			</feMerge>
		</filter>
	);
};

const idOf = (raw: string) => 'hand' + raw.replace(/[^a-zA-Z0-9_-]/g, '');

export const Stroke: React.FC<StrokeProps> = ({
	pts,
	progress,
	width,
	color,
	texture = 'ink',
	taper,
	seed = 1,
	boil = false,
	opacity = 1,
	smooth = 0.5,
	pace = 'hand',
	boilStep = 3,
	dry,
	head = true,
	inline = false,
	style,
}) => {
	const uid = idOf(useId());
	const tick = useBoil(boilStep);
	if (!(progress > 0) || pts.length < 2 || !(width > 0)) return null;
	const g = strokeGeo({pts, width, texture, taper, seed, smooth, pace, dry});
	if (!g) return null;
	const pre = PRESETS[texture];
	const p = Math.min(1, progress);
	const cut = cutAt(g, g.L * pacePos(p, pace));
	const jitter = boil ? boilJitter(g, tick, seed) : null;

	const body = bodyPath(g, cut, jitter, 1, true, pre.entry);
	const gaps = gapPath(g, cut, jitter, seed, width);
	const core = texture === 'gold' ? bodyPath(g, cut, jitter, 0.36, false) : '';

	// the pen tip while drawing: a wet bead of ink, or a bright head on a gold line
	const headT = head ? Math.min(1, p / 0.04) * (1 - Math.min(1, Math.max(0, (p - 0.9) / 0.1))) : 0;
	const tipJ = jitter ? jitter[Math.min(jitter.length - 1, cut.idx.length)] : ([0, 0] as Pt);
	const tip: Pt = [cut.tipP[0] + tipJ[0], cut.tipP[1] + tipJ[1]];
	const tipR = Math.max(width * 0.12, cut.tipHW) * pre.head;
	const tipT: Pt = [cut.tipN[1], -cut.tipN[0]]; // unit tangent at the tip

	const margin = texture === 'gold' ? 14 + width * 11 : 6 + width * 0.6;
	const bx = Math.floor(g.bbox[0] - margin);
	const by = Math.floor(g.bbox[1] - margin);
	const bw = Math.ceil(g.bbox[2] + margin) - bx;
	const bh = Math.ceil(g.bbox[3] + margin) - by;
	const box: [number, number, number, number] = [bx, by, bw, bh];
	const fid = uid + 'f';
	const mid = uid + 'm';
	const light = texture === 'gold' ? lighten(color, 0.72) : color;

	const content = (
		<>
			<defs>
				<TextureFilter id={fid} texture={texture} box={box} width={width} seed={seed} />
				{gaps ? (
					<mask id={mid} maskUnits="userSpaceOnUse" x={bx} y={by} width={bw} height={bh}>
						<rect x={bx} y={by} width={bw} height={bh} fill="#fff" />
						<path d={gaps} fill="#000" />
					</mask>
				) : null}
			</defs>
			<g filter={`url(#${fid})`}>
				<g mask={gaps ? `url(#${mid})` : undefined}>
					<path d={body} fill={color} />
					{core ? <path d={core} fill={light} /> : null}
				</g>
				{headT > 0.001 ? (
					texture === 'gold' ? (
						<>
							<circle cx={tip[0]} cy={tip[1]} r={tipR * 1.9} fill={light} opacity={0.35 * headT} />
							<circle cx={tip[0]} cy={tip[1]} r={tipR} fill="#FFFBF2" opacity={headT} />
						</>
					) : (
						// wet ink at the moving tip: a little wider than the line, dragged back along it
						<ellipse
							cx={tip[0] - tipT[0] * tipR * 0.9}
							cy={tip[1] - tipT[1] * tipR * 0.9}
							rx={tipR * 1.9}
							ry={tipR}
							transform={`rotate(${((Math.atan2(tipT[1], tipT[0]) * 180) / Math.PI).toFixed(2)} ${(tip[0] - tipT[0] * tipR * 0.9).toFixed(2)} ${(tip[1] - tipT[1] * tipR * 0.9).toFixed(2)})`}
							fill={color}
							opacity={headT}
						/>
					)
				) : null}
			</g>
		</>
	);
	const op = opacity * pre.opacity;
	if (inline) return <g opacity={op}>{content}</g>;
	return (
		<svg
			width={bw}
			height={bh}
			viewBox={`${bx} ${by} ${bw} ${bh}`}
			style={{position: 'absolute', left: bx, top: by, overflow: 'visible', pointerEvents: 'none', opacity: op, ...style}}
		>
			{content}
		</svg>
	);
};
