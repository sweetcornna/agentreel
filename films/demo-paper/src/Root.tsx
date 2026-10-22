// 片里的组合。只有一个：整片 DemoPaper（命令行不给 --comp 时就用它）。尺寸、帧率、总帧数都从 timeline.json 来。
import React from 'react';
import {Composition} from 'remotion';
import {loadFonts} from 'paper2d/components';
import {Film} from './Film';
import {T} from './timing';

// public/fonts 里是 OFL 字体的子集（见 public/fonts/OFL.txt）；登记成常用的 family 名，组件的字体栈直接能用
loadFonts([
	{family: 'Noto Serif SC', weight: '500', file: 'fonts/NotoSerifSC-500-subset.woff2'},
	{family: 'EB Garamond', weight: '500', file: 'fonts/EBGaramond-500-subset.woff2'},
	{family: 'JetBrains Mono', weight: '400', file: 'fonts/JetBrainsMono-400-subset.woff2'},
]);

const [W, H] = (T.timeline.canvas as {final: [number, number]}).final;

export const Root: React.FC = () => <Composition id="DemoPaper" component={Film} durationInFrames={T.total} fps={T.fps} width={W} height={H} />;
