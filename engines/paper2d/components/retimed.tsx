// 按别的网格写的画面，按合成自己的 fps 取样。例如场景写在 24 格网格上（72 BPM 一拍 20 格）、成片按 30 fps 出（移植来源的原片就是这样）：
// 输出帧 n 显示网格时间 n · 24 / 30（小数帧，每 5 帧落一次整格），所以两格之间是中间态而不是停格。配乐留在外面，按真实时间走。
import React from 'react';
import {Freeze, useCurrentFrame, useVideoConfig} from '../shim/remotion';

/** Output frame n shows grid time n · gridFps / fps (children see fractional frames; Sequences inside count from it). */
export const Retimed: React.FC<{gridFps: number; children: React.ReactNode}> = ({gridFps, children}) => {
	const frame = useCurrentFrame();
	const {fps} = useVideoConfig();
	return <Freeze frame={(frame * gridFps) / fps}>{children}</Freeze>;
};

/** 一拍二：子树每 `n` 帧才换一格画面（从所在段的第 0 帧起算）。镜头若要照常每帧走，放在外面。 */
export const OnTwos: React.FC<{n?: number; children: React.ReactNode}> = ({n = 2, children}) => {
	const frame = useCurrentFrame();
	return <Freeze frame={Math.floor(frame / n) * n}>{children}</Freeze>;
};
