// 拍点与版面：所有时间都从 timeline.json 的位置换算（T.at('3.1') = 200 帧），版面常量写在这里，各镜头共用。
import {beatGrid, ease, kineticLayout, ramp, type KineticProps, type Timeline} from 'paper2d/components';
import timeline from '../timeline.json';
import {COPY} from './copy';

export const T = beatGrid(timeline as Timeline);

const pencil = T.cue('pencil');
/** 帧号：每个关键时刻对应 timeline 里的一个位置或 cue */
export const AT = {
	/** 纸纹显影 1.1 → 1.2+20f */
	textureIn: [T.at('1.1'), T.at('1.2+20f')] as const,
	/** 铅笔四笔，各起在一拍上：1.3 1.4 2.1 2.2（cue pencil = 1.3-2.3） */
	strokes: Array.from({length: (pencil.end - pencil.start) / T.beat}, (_, i) => pencil.start + i * T.beat),
	/** 一笔画多久（不到一拍，留出抬笔） */
	strokeLen: 19,
	/** 成纸（cue paper） */
	paper: T.cue('paper').start,
	/** 红线进场（cue line），到「帧号」下（cue word1） */
	thread: T.cue('line').start,
	word1: T.cue('word1').start,
	/** 第二句（cue word2 = 4.3-5.1） */
	line2: T.cue('word2').start,
	/**
	 * 下划线换到「拍子表」下：4.3 先落到第二行字的下面（字还没到那儿），沿着字底下更低一点的地方滑过去（线尾先走），
	 * 等「拍子表」落定，4.4+0.5 贴上来、轻颤。全程不碰到正在落下的字。
	 */
	slide: {from: T.at('4.3'), drop: T.at('4.3+7f'), along: T.at('4.4+7f'), tuck: [T.at('4.4+3f'), T.at('4.4+0.5')] as const},
	/** 片尾：卡片收起、线绷直（cue end） */
	end: T.cue('end').start,
	taut: T.at('5.2+20f'),
	name: T.at('5.3'),
	repo: T.at('5.3+18f'),
	/** 6.4 起淡出，7.1 切到纸色（cue cut） */
	fade: T.at('6.4'),
	cut: T.cue('cut').start,
};

/** 卡片（世界坐标，按中心与宽高） */
export const CARD = {cx: 960, cy: 530, w: 1140, h: 430};
export const CARD_BOX = {x0: CARD.cx - CARD.w / 2, x1: CARD.cx + CARD.w / 2, y0: CARD.cy - CARD.h / 2, y1: CARD.cy + CARD.h / 2};

/** 卡片上的两句字幕（Kinetic 的参数；红线按 kineticLayout 算出的字位落到词下） */
export const CAPTION: KineticProps = {
	zh: [COPY.line1, COPY.line2],
	at: AT.thread + 10,
	lineAt: [AT.thread + 10, AT.line2],
	out: AT.end,
	x: CARD.cx,
	y: CARD.cy - 8,
	anchor: 'middle',
	align: 'center',
	width: 1120,
	size: 62,
	tracking: 0.1,
	weight: 500,
	step: 2.2,
	breath: 6,
	dir: [1, -0.2],
};
export const CAPTION_LAYOUT = kineticLayout(CAPTION);

/** 片尾的横线高度 */
export const END_LINE_Y = 622;
/** 卡片上的字收起时的轴：比横线高一点，压扁的字落在线的上方，不和线挤在一起 */
export const WORDS_FOLD_Y = END_LINE_Y - 12;

/** 5.1 卡片收起：k 是压扁的程度（以横线为轴，0 → 1），out 是剩下的不透明度。卡片和卡片上的字共用这一条。 */
export const fold = (f: number) => ({
	k: ramp(f, AT.end, AT.end + 26, ease.inOut),
	out: 1 - ramp(f, AT.end + 14, AT.end + 28, ease.linear),
});
