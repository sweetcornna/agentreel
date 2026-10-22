// 红线：一根线贯穿 P02–P03。
//   3.1   从画面左下甩进来：线头快、线身带着一道衰减的摆动跟上；
//         沿卡片外绕一圈（成了卡片的描边），在左下角折回卡内，斜着升上来，从左边接成「帧号」的下划线（4.1 到位，轻颤）；
//   4.3   下划线换到第二句「拍子表」下：先整段落到第二行字的下面，再滑过去（线尾先走，后面的线被拖着跟上），
//         4.4+0.5「拍子表」落定后贴上来；线的上沿和字形最低点之间始终留着一道 6–8 px 的纸色；
//   5.1   卡片收起，绕在卡片上的线跟着压扁，再被拉直成一条横线（5.2+20f 绷直，回颤）；
//   6.4   淡出。
// 每一帧只由帧号算出：路径是控制点的样条，摆动、颤动都是帧号的函数，不读上一帧。
import React from 'react';
import {useCurrentFrame} from 'remotion';
import {alive, centerline, clamp, ease, lerp, palette, pathLength, ramp, resampleN, Stroke, type Pt} from 'paper2d/components';
import {COPY} from './copy';
import {AT, CAPTION_LAYOUT, CARD_BOX, END_LINE_Y} from './timing';

const PAD = 24; // 线绕在卡片外多远
const R = 30; // 绕过卡角的半径
const LX0 = CARD_BOX.x0 - PAD;
const LX1 = CARD_BOX.x1 + PAD;
const LY0 = CARD_BOX.y0 - PAD;
const LY1 = CARD_BOX.y1 + PAD;
/** 折回卡内那一下（发夹弯）的位置：在卡片下沿外从左往回、向上、再向右 */
const HAIR_X = CARD_BOX.x0 + 80;
const HAIR_R = (LY1 - (CARD_BOX.y1 - 33)) / 2;
const LEAD_Y = LY1 - 2 * HAIR_R; // 卡内那一段走的高度

const arc = (cx: number, cy: number, r: number, a0: number, a1: number, n = 10): Pt[] =>
	Array.from({length: n + 1}, (_, i) => {
		const a = a0 + ((a1 - a0) * i) / n;
		return [cx + Math.cos(a) * r, cy + Math.sin(a) * r] as Pt;
	});
const line = (a: Pt, b: Pt, step = 12): Pt[] => {
	const n = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / step));
	return Array.from({length: n + 1}, (_, i) => [lerp(a[0], b[0], i / n), lerp(a[1], b[1], i / n)] as Pt);
};
const join = (...parts: Pt[][]): Pt[] => parts.reduce((acc, p) => (acc.length ? acc.concat(p.slice(1)) : p.slice()), [] as Pt[]);

/** 从画外甩进来，接到卡片外框的左下角，方向朝上 */
const ENTRY: Pt[] = centerline(
	[
		[-90, 948],
		[110, 912],
		[282, 852],
		[350, 796],
		[LX0, LY1 - R],
	],
	0.5,
);

/** 绕卡片一圈：左边往上、上边往右、右边往下、下边往左，再在 HAIR_X 处折回卡内 */
const LOOP: Pt[] = join(
	line([LX0, LY1 - R], [LX0, LY0 + R]),
	arc(LX0 + R, LY0 + R, R, Math.PI, 1.5 * Math.PI),
	line([LX0 + R, LY0], [LX1 - R, LY0]),
	arc(LX1 - R, LY0 + R, R, -0.5 * Math.PI, 0),
	line([LX1, LY0 + R], [LX1, LY1 - R]),
	arc(LX1 - R, LY1 - R, R, 0, 0.5 * Math.PI),
	line([LX1 - R, LY1], [HAIR_X, LY1]),
	arc(HAIR_X, LY1 - HAIR_R, HAIR_R, 0.5 * Math.PI, 1.5 * Math.PI, 12),
);
const FIXED = join(ENTRY, LOOP);

/**
 * 下划线的中心线在词的 y（kineticLayout 给的高度，实测就是字形的最低点）下面多远：线宽 4.6、活线的摆动 ±1.5，
 * 字底和线的上沿之间留出 6–8 px 的纸色（1080p，镜头放大约 1.03 倍）
 */
const UNDER = 10;
/** 换行滑动时线在字底下多走的深度：正在落下的字从下面 20 px 处带着模糊进来，线要让开 */
const TRAVEL = 20;

/**
 * 卡内那一段：折回卡内后斜着升上来，从词的左边接成下划线，往右划过整个词。box 是词的位置（含字幕的漂移）。
 * 第一句时斜线从第二句的位置穿过（那时第二句还没出来）；换到第二句时整段落到第二句下面。
 */
const tail = (box: {x0: number; x1: number; y: number}): Pt[] => {
	const y = box.y + UNDER;
	return centerline(
		[
			[HAIR_X, LEAD_Y],
			[HAIR_X + 180, LEAD_Y - 6],
			[box.x0 - 150, y + 62],
			[box.x0 - 40, y + 9],
			[box.x0 - 8, y + 2],
			[(box.x0 + box.x1) / 2, y],
			[box.x1 + 12, y + 3],
		],
		0.5,
	);
};

/** 0..1 的平滑台阶 */
const smooth = (u: number, a: number, b: number) => {
	const t = clamp((u - a) / (b - a));
	return t * t * (3 - 2 * t);
};

/** 片尾的横线：从画外左边到画外右边 */
const STRAIGHT: Pt[] = line([-90, END_LINE_Y], [2010, END_LINE_Y], 8);

const N = 900;

/** 这一帧线的形状（还没加摆动与颤动），世界坐标 */
export const threadBase = (f: number): Pt[] => {
	const b1 = CAPTION_LAYOUT.phrase(0, COPY.mark1, f)!;
	const b2 = CAPTION_LAYOUT.phrase(1, COPY.mark2, f)!;
	const S = AT.slide;
	if (f < S.from) return join(FIXED, tail(b1));
	if (f < AT.end) {
		if (f >= S.tuck[1]) return join(FIXED, tail(b2));
		// 先整段落下（7 帧，赶在第二行的字走到那儿之前），落到字底下再低 TRAVEL 的地方；
		// 横向滑过去时线尾先走（u 接近 1 的地方先动），后面的线被拖着跟过去；字落定后再贴上来
		const drop = ease.out(ramp(f, S.from, S.drop, ease.linear));
		const along = ramp(f, S.from, S.along, ease.linear);
		const deep = TRAVEL * (1 - ramp(f, S.tuck[0], S.tuck[1], ease.inOut));
		const A = resampleN(tail(b1), 260);
		const B = resampleN(tail(b2), 260);
		const moved = A.map((p, i) => {
			const u = i / 260;
			const lag = 0.32 * (1 - u);
			const t = ease.inOut(clamp((along - lag) / (1 - lag)));
			return [lerp(p[0], B[i][0], t), lerp(p[1], B[i][1] + deep * smooth(u, 0.3, 0.75), drop)] as Pt;
		});
		return join(FIXED, moved);
	}
	// 片尾：先跟着卡片压扁到横线的高度，再被拉直（两头先走、中间最后）
	const base = resampleN(join(FIXED, tail(b2)), N);
	const flat = resampleN(STRAIGHT, N);
	const ky = ramp(f, AT.end, AT.end + 26, ease.inOut);
	return base.map((p, i) => {
		const u = i / N;
		const lag = 0.45 * (1 - Math.abs(2 * u - 1));
		const kx = ease.inOut(clamp((ramp(f, AT.end + 10, AT.taut, ease.linear) - lag) / (1 - lag)));
		const kyy = Math.max(ky, kx);
		return [lerp(p[0], flat[i][0], kx), lerp(p[1], flat[i][1], kyy)] as Pt;
	});
};

/** 线头走到哪儿了（弧长，像素）和此刻的速度（0..1）。3.1 起甩进来，4.1 停在词下；先快后慢。 */
const DRAW = (t: number) => 1 - Math.pow(1 - t, 2.2);
const head = (f: number, total: number): {s: number; v: number} => {
	const t = clamp((f - AT.thread) / (AT.word1 - AT.thread));
	// DRAW'(t) / DRAW'(0)
	return {s: total * DRAW(t), v: Math.pow(1 - t, 1.2)};
};

/** 刚放下的那段线还在摆：离线头越远摆得越小，线头本身不动 */
const whip = (pts: Pt[], sHead: number, v: number): Pt[] => {
	if (v <= 0.001) return pts;
	const out: Pt[] = [];
	let s = 0;
	for (let i = 0; i < pts.length; i++) {
		if (i) s += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
		const a = pts[Math.max(0, i - 1)];
		const b = pts[Math.min(pts.length - 1, i + 1)];
		const l = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
		const behind = sHead - s;
		let d = 0;
		if (behind > 0) {
			const env = clamp(behind / 60) * Math.exp(-behind / 300);
			d = 18 * v * env * Math.sin((2 * Math.PI * behind) / 260);
		}
		out.push([pts[i][0] - ((b[1] - a[1]) / l) * d, pts[i][1] + ((b[0] - a[0]) / l) * d]);
	}
	return out;
};

export const Thread: React.FC = () => {
	const f = useCurrentFrame();
	if (f < AT.thread || f >= AT.cut) return null;
	const base = threadBase(f);
	const total = pathLength(base);
	let pts: Pt[];
	let progress = 1;
	if (f < AT.word1) {
		const h = head(f, total);
		pts = whip(resampleN(base, Math.ceil(total / 6)), h.s, h.v);
		progress = h.s / total;
	} else {
		// 放下之后线还活着：一道沿线走的细波；4.1 落到词下轻颤一下，5.2+20f 绷直时整根回颤
		const live = ramp(f, AT.word1, AT.word1 + 20, ease.soft);
		pts = alive(resampleN(base, Math.ceil(total / 6)), f, {
			amp: 1.5 * live,
			wavelength: 380,
			speed: 0.1,
			seed: 2,
			pinStart: true,
			pinEnd: false,
			straight: true,
			step: 6,
			plucks: [
				// 落到词下的两下轻颤压得小，免得颤到字上
				{at: AT.word1, u: 0.985, amp: 1.8},
				{at: AT.slide.tuck[1], u: 0.985, amp: 1.5},
				{at: AT.taut, u: 0.5, amp: 11},
			],
		});
	}
	const fade = 1 - ramp(f, AT.fade, AT.cut, ease.inOut);
	return (
		<Stroke
			pts={pts}
			progress={progress}
			width={4.6}
			color={palette.thread}
			texture="cord"
			seed={7}
			taper={[0.004, 0.006]}
			pace="linear"
			smooth={0}
			head={false}
			opacity={fade}
		/>
	);
};

