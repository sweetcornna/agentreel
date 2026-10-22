// One way for a film's words to move. Every caption and running note can go through Kinetic, so a whole film speaks
// with one voice (the source film's lead, on the first cut: "don't show the subtitles so stiffly"):
//
//   · it is spoken, not printed: characters arrive phrase by phrase, with a breath at each comma, each one carried in
//     a short way on a curve and settling with a touch of overshoot, sharpening as it lands;
//   · it is never pinned: the block drifts slowly on, as if the lens were still moving past it, and each character
//     breathes a fraction of a pixel on its own;
//   · it leaves the way it came, swept on in the same direction, character after character, not faded as one;
//   · a key phrase can be marked with the thread (a cord underline drawn in once the phrase has landed, pulled out
//     again on the way out);
//   · the English follows the Chinese, uncovered left to right, and goes with it.
//
// Pure function of the frame; no layout measurement: CJK characters are one em wide plus the tracking, so the place
// of any character is known in advance (kineticLayout) and other drawing (a thread running to a word) can meet it.
import React from 'react';
import {useCurrentFrame} from '../shim/remotion';
import {alive} from './alive';
import {ease, ramp} from './anim';
import {Stroke} from './hand';
import type {Pt} from './hand';
import {fonts, palette} from './palette';

const cl = (x: number) => Math.min(1, Math.max(0, x));
const outBack = (u: number, s = 1.15) => 1 + (s + 1) * Math.pow(u - 1, 3) + s * Math.pow(u - 1, 2);
const BREATH = /[，。、；：！？,.;:!?]/;
/**
 * a stop to breathe at, and to give a full em: punctuation, except a decimal point or a thousands comma between two
 * digits (815.1, 1,500), which is part of the number
 */
const breathAt = (chars: string[], i: number) =>
	BREATH.test(chars[i]) && !(/[.,]/.test(chars[i]) && /\d/.test(chars[i - 1] ?? '') && /\d/.test(chars[i + 1] ?? ''));

export type KineticProps = {
	/** Chinese lines, top to bottom */
	zh: string[];
	en?: string;
	/** local frame the first character starts to arrive */
	at: number;
	/** local frame the sweep out starts */
	out: number;
	x: number;
	y: number;
	align?: 'left' | 'center';
	/** vertical anchor of (x, y): the top of the block or its middle */
	anchor?: 'top' | 'middle';
	size?: number;
	font?: string;
	weight?: number;
	color?: string;
	enFont?: string;
	enColor?: string;
	enSize?: number;
	tracking?: number; // em
	shadow?: string;
	/** direction the words travel (in, drift, out), unit-ish vector; default rightwards and a little up */
	dir?: [number, number];
	/** a phrase to mark with the thread, and its colour */
	mark?: string;
	markColor?: string;
	/** frames between characters inside a phrase */
	step?: number;
	/** extra frames of breath after a comma or full stop */
	breath?: number;
	/** screen offset added to the block (e.g. parallax against the camera) */
	shift?: [number, number];
	width?: number;
	/** earliest start for each Chinese line (a second line held back for the voice) */
	lineAt?: number[];
	/** when the English starts (default: once the Chinese has landed) */
	enAt?: number;
};

export type KineticLayout = {
	/** arrival frame of every character, line by line */
	times: number[][];
	/** frame the last character has settled */
	landed: number;
	/** characters in all lines */
	count: number;
	/** one character's advance (size · (1 + tracking)) and the line height */
	em: number;
	lineH: number;
	/** the block's top-left before drift */
	left: number;
	top: number;
	/** left edge (block coordinates) and width of each line */
	lines: {x: number; w: number; chars: string[]}[];
	/** the block's drift at frame f (add it to anything that must stay with the words) */
	drift: (f: number) => [number, number];
	/**
	 * Where a phrase of line `li` sits, in world coordinates at frame f (drift included): the underline span the mark
	 * uses, [x0, x1] at baseline height y, and the frame its last character arrives. null if the phrase is not there.
	 */
	phrase: (li: number, text: string, f: number) => {x0: number; x1: number; y: number; at: number} | null;
};

/** Timing and geometry of a Kinetic block (the same numbers Kinetic draws with). */
export function kineticLayout(p: KineticProps): KineticLayout {
	const {zh, en, at, out, x, y, align = 'left', anchor = 'top', size = 36, tracking = 0.06, dir = [1, -0.35], step = 1.3, breath = 5, shift = [0, 0], width = 1200, lineAt} = p;
	const [dx, dy] = dir;
	const eSize = p.enSize ?? Math.round(size * 0.62);
	// arrival times: phrase by phrase, a breath after each stop
	const times: number[][] = [];
	let t = at;
	zh.forEach((line, li) => {
		const row: number[] = [];
		if (lineAt?.[li] !== undefined) t = Math.max(t, lineAt[li]);
		const cs = Array.from(line);
		cs.forEach((_, i) => {
			row.push(t);
			t += breathAt(cs, i) ? step + breath : step;
		});
		times.push(row);
		if (li < zh.length - 1) t += breath * 0.6;
	});
	const landed = t + 8;
	const count = times.reduce((a, r) => a + r.length, 0);
	const lineH = size * 1.5;
	const blockH = zh.length * lineH + (en ? size * 0.4 + eSize * 1.35 : 0);
	const top = anchor === 'top' ? y : y - blockH / 2;
	const left = align === 'left' ? x : x - width / 2;
	const em = size * (1 + tracking);
	const lines = zh.map((line) => {
		const chars = Array.from(line);
		const w = chars.length * em;
		return {x: align === 'center' ? (width - w) / 2 : 0, w, chars};
	});
	// the block never stops: a slow drift along the travel direction, faster once it is leaving
	const drift = (f: number): [number, number] => {
		const life = f - at;
		const leave = cl((f - out) / 26);
		return [dx * (life * 0.1 + leave * leave * 40) + shift[0], dy * (life * 0.1 + leave * leave * 40) + shift[1]];
	};
	const phrase = (li: number, text: string, f: number) => {
		const ln = lines[li];
		if (!ln) return null;
		const idx = zh[li].indexOf(text);
		if (idx < 0) return null;
		const ci = Array.from(zh[li].slice(0, idx)).length;
		const len = Array.from(text).length;
		const [ddx, ddy] = drift(f);
		const m0 = ci * em;
		const m1 = (ci + len) * em - size * tracking;
		return {x0: left + ln.x + m0 + ddx, x1: left + ln.x + m1 + ddx, y: top + li * lineH + size * 1.28 + ddy, at: times[li][ci + len - 1]};
	};
	return {times, landed, count, em, lineH, left, top, lines, drift, phrase};
}

export const Kinetic: React.FC<KineticProps> = (props) => {
	const {
		zh, en, at, out, align = 'left', size = 36, font = fonts.serif, weight = 500, color = palette.ink, enFont = fonts.latin, enColor = palette.mute,
		tracking = 0.06, shadow, dir = [1, -0.35], mark, markColor = palette.thread, width = 1200, enAt,
	} = props;
	const f = useCurrentFrame();
	if (f < at - 1 || f > out + 40) return null;
	const [dx, dy] = dir;
	const eSize = props.enSize ?? Math.round(size * 0.62);
	const L = kineticLayout(props);
	const {times, landed, count: n, em, lineH, left, top} = L;
	const sw = Math.min(0.8, 12 / Math.max(1, n)); // the sweep out takes at most ~21 frames however long the line
	const [driftX, driftY] = L.drift(f);

	let k = 0;
	const rows = zh.map((line, li) => {
		const chars = Array.from(line);
		const lineW = L.lines[li].w;
		let markAt = -1;
		if (mark) markAt = line.indexOf(mark);
		const spans = chars.map((c, i) => {
			const idx = k++;
			const u = cl((f - times[li][i]) / 11);
			const e = outBack(u);
			// out: in reading order, each one swept on a little after the last
			const v = cl((f - out - idx * sw) / 9);
			const ev = v * v;
			const breathe = Math.sin(f * 0.07 + idx * 0.9) * 0.7;
			const tx = (1 - e) * -22 * dx + ev * 70 * dx;
			const ty = (1 - e) * (16 - 22 * dy) + ev * 70 * dy + breathe;
			const rot = (1 - e) * -7 * Math.sign(dx || 1) + ev * 6;
			const blur = (1 - u) * (1 - u) * 7 + ev * 7;
			return (
				<span
					key={i}
					style={{
						display: 'inline-block', width: breathAt(chars, i) ? em : undefined, opacity: Math.min(1, u * 2.2) * (1 - ev),
						transform: `translate(${tx}px, ${ty}px) rotate(${rot}deg) scale(${0.92 + 0.08 * e})`,
						filter: blur > 0.25 ? `blur(${blur}px)` : undefined,
					}}
				>
					{c}
				</span>
			);
		});
		let underline: React.ReactNode = null;
		if (markAt >= 0 && mark) {
			const ci = Array.from(line.slice(0, markAt)).length;
			const len = Array.from(mark).length;
			const m0 = ci * em;
			const m1 = (ci + len) * em - size * tracking;
			const drawIn = cl((f - (times[li][ci + len - 1] + 6)) / 9);
			const pullOut = cl((f - out + 2) / 8);
			const yy = size * 1.28;
			const base: Pt[] = [[m0 - 6, yy + 2], [(m0 + m1) / 2, yy - 1], [m1 + 10, yy + 3]];
			const pts = alive(base, f, {amp: 1.2, wavelength: 160, speed: 0.12, seed: li + 3});
			const ox = align === 'center' ? (width - lineW) / 2 : 0;
			if (drawIn > 0 && pullOut < 1)
				underline = (
					<div style={{position: 'absolute', left: ox, top: 0, width: lineW, height: lineH}}>
						<Stroke pts={pts} progress={drawIn * (1 - pullOut)} width={Math.max(2.4, size * 0.085)} color={markColor} texture="cord" seed={31 + li} taper={[0.05, 0.1]} pace="linear" />
					</div>
				);
		}
		return (
			<div key={li} style={{position: 'relative', height: lineH, whiteSpace: 'pre', textAlign: align}}>
				{spans}
				{underline}
			</div>
		);
	});

	// the English: uncovered left to right after the Chinese has landed, swept out with it
	const w = cl((f - (enAt ?? landed - 4)) / 14);
	const we = 1 - Math.pow(1 - w, 3);
	const ve = Math.pow(cl((f - out - n * sw) / 9), 2);

	return (
		<div
			style={{
				position: 'absolute', left, top, width, transform: `translate(${driftX}px, ${driftY}px)`,
				fontFamily: font, fontWeight: weight, fontSize: size, color, letterSpacing: `${tracking}em`, textShadow: shadow, lineHeight: `${lineH}px`,
			}}
		>
			{rows}
			{en ? (
				<div style={{marginTop: size * 0.4, textAlign: align}}>
					<div
						style={{
							display: 'inline-block', fontFamily: enFont, fontStyle: 'italic', fontWeight: 400, fontSize: eSize, lineHeight: 1.35,
							letterSpacing: '0.01em', color: enColor, textAlign: 'left',
							clipPath: `inset(-10px ${(1 - we) * 100}% -10px -10px)`,
							transform: `translate(${(1 - we) * -10 * dx + ve * 70 * dx}px, ${(1 - we) * 6 + ve * 70 * dy}px)`,
							opacity: 1 - ve, filter: ve > 0.05 ? `blur(${ve * 6}px)` : undefined,
						}}
					>
						{en}
					</div>
				</div>
			) : null}
		</div>
	);
};

type CaptionProps = {
	zh: string;
	en?: string;
	from: number; // local frame reveal starts
	to: number; // local frame fully gone
	tone?: 'night' | 'paper';
	y?: number; // vertical centre in px
	x?: number; // horizontal centre in px (default: the middle of a 1920 frame)
	size?: number;
	align?: 'center' | 'left';
	stagger?: number; // frames between characters
	weight?: number;
	enSize?: number;
	glow?: boolean;
	exit?: 'fade' | 'fly'; // 'fly': tracking opens and the line rushes past the camera
	exitLen?: number;
};

/** A one-line caption through Kinetic, in a night or paper tone; a 'fly' exit rushes the whole line past the camera. */
export const Caption: React.FC<CaptionProps> = ({
	zh, en, from, to, tone = 'night', y = 540, x = 960, size = 40, align = 'center', stagger = 2, weight = 300, enSize, glow = tone === 'night', exit = 'fade', exitLen = 14,
}) => {
	const frame = useCurrentFrame();
	if (frame < from - 1 || frame > to + 1) return null;
	const color = tone === 'night' ? palette.nightText : palette.ink;
	const enColor = tone === 'night' ? 'rgba(236,230,218,0.62)' : 'rgba(36,33,31,0.58)';
	const chars = Array.from(zh).length;
	const fly = exit === 'fly' ? ramp(frame, to - exitLen, to, ease.in) : 0;
	const out = exit === 'fly' ? to + 100 : to - Math.max(exitLen, Math.min(22, Math.round(chars * 0.8 + 10)));
	return (
		<div
			style={{
				position: 'absolute', inset: 0, opacity: 1 - fly, transformOrigin: `${x}px ${y}px`,
				filter: fly > 0 ? `blur(${fly * 16}px)` : undefined, scale: fly > 0 ? String(1 + fly * 0.9) : undefined,
			}}
		>
			<Kinetic
				zh={zh ? [zh] : []} en={en} at={from} out={out} x={x} y={y} anchor="middle" align={align}
				width={align === 'center' ? 1800 : 1920 - x} size={size} font={fonts.serif} weight={weight} color={color} tracking={0.28}
				enFont={fonts.latin} enColor={enColor} enSize={enSize ?? Math.round(size * 0.58)} step={stagger * 0.75} dir={[1, -0.25]}
				shadow={glow ? '0 0 18px rgba(229,178,93,0.25)' : undefined} enAt={zh ? undefined : from}
			/>
		</div>
	);
};

/** Small mono label, e.g. addresses and terms. */
export const Mono: React.FC<{children: React.ReactNode; size?: number; color?: string; style?: React.CSSProperties}> = ({children, size = 18, color = 'currentColor', style}) => (
	<span style={{fontFamily: fonts.mono, fontSize: size, color, letterSpacing: '0.01em', whiteSpace: 'nowrap', ...style}}>{children}</span>
);

/** Typewriter: reveals `text` one character every `every` frames starting at `from`; optional caret. */
export const Typed: React.FC<{text: string; from: number; every?: number; caret?: boolean; style?: React.CSSProperties}> = ({text, from, every = 2, caret = true, style}) => {
	const frame = useCurrentFrame();
	const chars = Array.from(text);
	const n = Math.max(0, Math.min(chars.length, Math.floor((frame - from) / every)));
	const done = n >= chars.length;
	const blink = done ? Math.floor(frame / 12) % 2 === 0 : true;
	return (
		<span style={{fontFamily: fonts.mono, whiteSpace: 'pre', ...style}}>
			{chars.slice(0, n).join('')}
			{caret && frame >= from ? <span style={{opacity: blink ? 0.85 : 0, marginLeft: 1, borderLeft: '2px solid currentColor'}}>&nbsp;</span> : null}
		</span>
	);
};
