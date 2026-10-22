// P03 片尾：红线绷直后，上方落下「agentreel」，再打出仓库地址；6.4 起淡出。放在 <Sequence from={5.1}> 里，帧号从 5.1 起算。
import React from 'react';
import {useCurrentFrame} from 'remotion';
import {ease, fonts, Kinetic, palette, ramp, Typed} from 'paper2d/components';
import {COPY} from './copy';
import {AT, END_LINE_Y} from './timing';

const NAME_Y = END_LINE_Y - 200;
const REPO_Y = END_LINE_Y - 56;
const REPO_SIZE = 30;
/** JetBrains Mono 的字宽是 0.6 em：地址按整串的宽度定左边，打字时不会跟着居中左右挪 */
const REPO_W = Array.from(COPY.repo).length * REPO_SIZE * 0.62;

export const EndTitle: React.FC = () => {
	const f = useCurrentFrame();
	const t0 = AT.end;
	const fade = 1 - ramp(f + t0, AT.fade, AT.cut, ease.inOut);
	return (
		<div style={{position: 'absolute', inset: 0, opacity: fade}}>
			<Kinetic
				zh={[COPY.name]}
				at={AT.name - t0}
				out={10000}
				x={960}
				y={NAME_Y}
				anchor="middle"
				align="center"
				width={1400}
				size={136}
				tracking={0.01}
				font={fonts.latin}
				weight={500}
				color={palette.ink}
				step={2.4}
				dir={[1, -0.15]}
			/>
			<div
				style={{
					position: 'absolute', left: 960 - REPO_W / 2, top: REPO_Y - REPO_SIZE * 0.7, fontSize: REPO_SIZE, color: palette.soft, letterSpacing: '0.02em',
					opacity: ramp(f + t0, AT.repo, AT.repo + 6, ease.out),
				}}
			>
				<Typed text={COPY.repo} from={AT.repo - t0} every={1} caret={false} />
			</div>
		</div>
	);
};
