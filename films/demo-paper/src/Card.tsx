// P01 的卡片：1.3 起铅笔四笔画出一个框（每拍一笔），2.3「成纸」——框里落下一张有纸纹的卡片，轻抬、落定、长出投影，
// 铅笔线随后淡去；落定那一下，对角两个角上弹出几笔铅笔短线（一拍二）。5.1 卡片收进红线（以片尾横线的高度为轴压扁）。
import React from 'react';
import {useCurrentFrame} from 'remotion';
import {Burst, clamp, ease, HandPaper, PaperCard, palette, ramp, roughen, Stroke, type Pt} from 'paper2d/components';
import {AT, CARD, CARD_BOX, END_LINE_Y, fold} from './timing';

const {x0, x1, y0, y1} = CARD_BOX;
/** 四笔：上、右、下、左；每笔都越过角一点，像手画的框 */
const EDGES: {from: Pt; to: Pt; seed: number}[] = [
	{from: [x0 - 22, y0 + 4], to: [x1 + 18, y0 - 3], seed: 3},
	{from: [x1 - 3, y0 - 20], to: [x1 + 4, y1 + 16], seed: 5},
	{from: [x1 + 20, y1 - 3], to: [x0 - 18, y1 + 4], seed: 8},
	{from: [x0 + 4, y1 + 21], to: [x0 - 3, y0 - 17], seed: 11},
];
const edgePts = EDGES.map((e) => {
	const n = 8;
	const pts = Array.from({length: n + 1}, (_, i) => [e.from[0] + ((e.to[0] - e.from[0]) * i) / n, e.from[1] + ((e.to[1] - e.from[1]) * i) / n] as Pt);
	return roughen(pts, 1.6, e.seed);
});

export const Card: React.FC = () => {
	const f = useCurrentFrame();
	if (f < AT.strokes[0] || f >= AT.end + 30) return null;
	// 成纸：纸面 8 帧内填满框，同时轻轻抬起再落定（投影跟着长出来）
	const fill = ramp(f, AT.paper, AT.paper + 8, ease.out);
	const lift = ramp(f, AT.paper, AT.paper + 26, ease.out);
	const settle = 1 + 0.014 * Math.sin(Math.PI * ramp(f, AT.paper, AT.paper + 20, ease.linear)) * (1 - lift * 0.5);
	const pencilFade = 1 - ramp(f, AT.paper + 4, AT.paper + 36, ease.inOut);
	// 片尾：以横线为轴压扁
	const {k: folded, out: foldOut} = fold(f);
	const originY = END_LINE_Y - y0;
	return (
		<>
			{fill > 0 ? (
				<PaperCard
					x={CARD.cx}
					y={CARD.cy}
					w={CARD.w}
					h={CARD.h}
					radius={6}
					lift={0.25 + 0.75 * lift * (1 - folded)}
					opacity={fill * foldOut}
					origin={`50% ${originY}px`}
					style={{overflow: 'hidden', transform: `scale(${settle}, ${settle * (1 - folded)})`}}
				>
					{/* 卡片自己也是一张纸：更白、纹理更淡 */}
					<HandPaper width={CARD.w} height={CARD.h} flat={palette.card} stock={palette.card} grain={1} texture={0.45} />
				</PaperCard>
			) : null}
			<Burst f={f} at={AT.paper + 2} x={x0 - 6} y={y0 - 6} r={40} n={7} seed={2} width={2.4} color={palette.graphite} texture="pencil" />
			<Burst f={f} at={AT.paper + 4} x={x1 + 6} y={y1 + 6} r={40} n={7} seed={5} width={2.4} color={palette.graphite} texture="pencil" />
			{pencilFade > 0
				? EDGES.map((e, i) => {
						const p = clamp((f - AT.strokes[i]) / AT.strokeLen);
						if (p <= 0) return null;
						return (
							<Stroke
								key={i}
								pts={edgePts[i]}
								progress={p}
								width={3.4}
								color={palette.graphite}
								texture="pencil"
								seed={e.seed}
								boil
								opacity={pencilFade}
							/>
						);
					})
				: null}
		</>
	);
};
