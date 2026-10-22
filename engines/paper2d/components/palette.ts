// 纸面的中性默认色板与字体栈。组件的颜色参数都以这里为默认值；一部片用自己的颜色时直接传参数，或者在片里另写一份
// 色板（不改这里）。数值是为示例片调的通用暖纸色，不对应任何产品或品牌。
export const palette = {
	/** 平铺的纸色（纸纹显影之前、片尾切到的纸色） */
	paper: '#F2EBDD',
	/** 带纹理的纸底：比平铺纸色略灰、略暖 */
	stock: '#ECE4D3',
	/** 卡片纸面 */
	card: '#FBF7EF',
	ink: '#24211F',
	soft: '#4C4640',
	mute: '#8C8477',
	/** 铅笔石墨 */
	graphite: '#3E3A35',
	/** 红线（朱红偏橙） */
	thread: '#C8462E',
	/** 投影的颜色（rgb 三元组，组件里配透明度） */
	shadow: '70,52,34',
	/** 夜色（暗场组件的默认值） */
	night: '#0A0C10',
	nightText: '#ECE6DA',
	gold: '#E5B25D',
};

export type Palette = typeof palette;

/** 字体栈。示例片在 public/fonts 里放了 OFL 字体的子集，并用 loadFonts 登记同名 family。 */
export const fonts = {
	serif: "'Noto Serif SC', 'Songti SC', 'SimSun', serif",
	latin: "'EB Garamond', 'Noto Serif SC', Georgia, serif",
	mono: "'JetBrains Mono', ui-monospace, 'SFMono-Regular', Menlo, monospace",
	sans: "'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif",
};
