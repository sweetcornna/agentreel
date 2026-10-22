// 整片的分层：纸面世界（纸、卡片、红线、字）在一个镜头里，镜头只做极慢的推拉；颗粒和暗角在画面空间盖在最上面。
// 7.1 切到纸色（平铺，无纹理、无颗粒），停到 8.1（配乐的余音）。
import React from 'react';
import {AbsoluteFill, Sequence, useCurrentFrame, useVideoConfig} from 'remotion';
import {clampLens, ease, Grain, HandPaper, Kinetic, lensPath, lensTransform, palette, ramp, Vignette, type LensKey} from 'paper2d/components';
import {Card} from './Card';
import {EndTitle} from './EndTitle';
import {Thread} from './Thread';
import {AT, CAPTION, fold, T, WORDS_FOLD_Y} from './timing';

/** 镜头：成纸时推近一点，红线走字时略向右下，片尾退回原位 */
const LENS: LensKey[] = [
	[T.at('1.1'), 1.0, 960, 540],
	[T.at('3.1'), 1.035, 960, 532],
	[T.at('5.1'), 1.03, 978, 548],
	[T.at('7.1'), 1.0, 960, 540],
];

const Paper: React.FC = () => {
	const f = useCurrentFrame();
	// 纸纹随帧号显影；铅笔方格轻轻画上去；片尾方格换成点阵（换了状态）
	const grain = ramp(f, AT.textureIn[0], AT.textureIn[1], ease.out);
	const drawn = ramp(f, T.at('1.1+8f'), T.at('1.3+10f'), ease.inOut);
	const swap = ramp(f, AT.end, AT.end + 30, ease.inOut);
	return <HandPaper grain={grain} drawn={drawn} texture={0.55} mix={{grid: 0.4 * (1 - swap), dots: 0.5 * swap}} />;
};

/** 卡片上的两句字：印在卡片上，5.1 跟卡片一起压扁（同时按 Kinetic 的方式扫出） */
const CardWords: React.FC = () => {
	const f = useCurrentFrame();
	const {k, out} = fold(f);
	return (
		<div style={{position: 'absolute', inset: 0, transformOrigin: `0 ${WORDS_FOLD_Y}px`, transform: k > 0 ? `scale(1, ${1 - k})` : undefined, opacity: out}}>
			<Kinetic {...CAPTION} />
		</div>
	);
};

export const Film: React.FC = () => {
	const f = useCurrentFrame();
	const {width, height} = useVideoConfig();
	const screen = {width, height};
	const lens = clampLens(lensPath(LENS, f), screen);
	return (
		<AbsoluteFill style={{background: palette.paper, overflow: 'hidden'}}>
			<Sequence durationInFrames={AT.cut} name="P01–P03">
				<div style={{position: 'absolute', left: 0, top: 0, width, height, transformOrigin: '0 0', transform: lensTransform(lens, screen)}}>
					<Paper />
					<Card />
					<Thread />
					<CardWords />
					<Sequence from={AT.end} name="P03 片尾">
						<EndTitle />
					</Sequence>
				</div>
				<Vignette strength={0.1} color={palette.shadow} inner={48} />
				<Grain opacity={0.085} blend="overlay" seed={3} />
			</Sequence>
		</AbsoluteFill>
	);
};
