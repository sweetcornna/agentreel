// 拍网格：读一部片的 timeline.json，把「小节.拍+偏移」换成帧号，反过来把帧号写成「小节.拍+偏移」。纯函数，不依赖 React，
// 场景（浏览器里）和命令行（still / scan 的帧列表与标注）共用这一份。
//
// 写法（docs/TIMELINE.md）：位置写成 "小节.拍"，从 1 起；"小节.拍+0.5" 晚半拍，"小节.拍+4f" 晚 4 帧，可以连写
// "3.1+0.5+2f"。帧号 frame = ((bar − 1) · beatsPerBar + (beat − 1)) · framesPerBeat + 偏移。范围 "1.3-2.3" 含起点、
// 不含终点（镜头的 from / to 同样）。速度固定：framesPerBeat 必须等于 fps · 60 / bpm，否则画面与配乐对不上同一张网格。

export type Span = {id: string; zh?: string; from: string; to: string; [k: string]: unknown};
export type Cue = {id: string; at: string; kind?: string; note?: string; [k: string]: unknown};

export type Timeline = {
	version?: number;
	fps: number;
	bpm: number;
	beatsPerBar: number;
	framesPerBeat: number;
	framesPerBar?: number;
	bars?: number;
	totalFrames: number;
	filmEnd?: string;
	sections?: Span[];
	shots?: Span[];
	cues?: Cue[];
	[k: string]: unknown;
};

export type Pos = {bar: number; beat: number; beats: number; frames: number};

const POS = /^\s*(\d+)\.(\d+)((?:\+\d+(?:\.\d+)?f?)*)\s*$/;

/** "3.1+0.5+4f" → {bar 3, beat 1, beats 0.5, frames 4}. Throws on anything else. */
export function parsePos(s: string): Pos {
	const m = POS.exec(s);
	if (!m) throw new Error(`bad position "${s}" (write bar.beat, e.g. 3.1, 3.1+0.5 or 3.1+4f)`);
	let beats = 0;
	let frames = 0;
	for (const term of m[3].split('+').filter(Boolean)) {
		if (term.endsWith('f')) frames += Number(term.slice(0, -1));
		else beats += Number(term);
	}
	return {bar: Number(m[1]), beat: Number(m[2]), beats, frames};
}

export type BeatGrid = {
	timeline: Timeline;
	fps: number;
	bpm: number;
	beatsPerBar: number;
	/** frames per beat / per bar */
	beat: number;
	barLen: number;
	total: number;
	/** frame of a position ("3.1+4f"); numbers pass through. May be fractional (e.g. "+0.3" of a 25-frame beat). */
	at: (p: string | number) => number;
	/** frame of beat `beat` (1-based) of bar `n` (1-based) */
	bar: (n: number, beat?: number) => number;
	/** frame → "bar.beat" or "bar.beat+Nf" (the beat at or before the frame) */
	pos: (frame: number) => string;
	/** [from, to) of "1.3-2.3", or [at, at] for a single position */
	range: (s: string) => [number, number];
	shot: (id: string) => Span & {start: number; end: number};
	section: (id: string) => Span & {start: number; end: number};
	cue: (id: string) => Cue & {start: number; end: number};
	/** the shot / section a frame belongs to */
	shotAt: (frame: number) => (Span & {start: number; end: number}) | undefined;
	sectionAt: (frame: number) => (Span & {start: number; end: number}) | undefined;
	seconds: (frame: number) => number;
};

export function beatGrid(tl: Timeline): BeatGrid {
	const {fps, bpm, beatsPerBar, framesPerBeat} = tl;
	for (const [k, v] of Object.entries({fps, bpm, beatsPerBar, framesPerBeat, totalFrames: tl.totalFrames}))
		if (!(typeof v === 'number' && v > 0)) throw new Error(`timeline.${k} must be a positive number, got ${String(v)}`);
	if (Math.abs((fps * 60) / bpm - framesPerBeat) > 1e-9)
		throw new Error(`timeline: framesPerBeat ${framesPerBeat} ≠ fps·60/bpm = ${(fps * 60) / bpm}; picture and score would drift apart`);
	const barLen = framesPerBeat * beatsPerBar;
	if (tl.framesPerBar !== undefined && tl.framesPerBar !== barLen) throw new Error(`timeline.framesPerBar ${tl.framesPerBar} ≠ beatsPerBar·framesPerBeat = ${barLen}`);
	const bar = (n: number, beat = 1) => ((n - 1) * beatsPerBar + (beat - 1)) * framesPerBeat;
	const at = (p: string | number): number => {
		if (typeof p === 'number') return p;
		const q = parsePos(p);
		if (q.beat < 1 || q.beat > beatsPerBar) throw new Error(`position "${p}": beat must be 1–${beatsPerBar}`);
		return bar(q.bar, q.beat) + q.beats * framesPerBeat + q.frames;
	};
	const pos = (frame: number): string => {
		const b = Math.floor(frame / framesPerBeat + 1e-9);
		const rest = frame - b * framesPerBeat;
		const head = `${Math.floor(b / beatsPerBar) + 1}.${(b % beatsPerBar) + 1}`;
		if (Math.abs(rest) < 1e-9) return head;
		return `${head}+${Number.isInteger(rest) ? rest : rest.toFixed(2)}f`;
	};
	const range = (s: string): [number, number] => {
		const m = /^(.+?)-(.+)$/.exec(s.trim());
		if (!m) {
			const f = at(s);
			return [f, f];
		}
		return [at(m[1]), at(m[2])];
	};
	const spanOf = <T extends Span>(list: T[] | undefined, what: string) => (id: string) => {
		const s = (list ?? []).find((x) => x.id === id);
		if (!s) throw new Error(`timeline has no ${what} "${id}"; have: ${(list ?? []).map((x) => x.id).join(', ') || 'none'}`);
		return {...s, start: at(s.from), end: at(s.to)};
	};
	const shot = spanOf(tl.shots, 'shot');
	const section = spanOf(tl.sections, 'section');
	const cue = (id: string) => {
		const c = (tl.cues ?? []).find((x) => x.id === id);
		if (!c) throw new Error(`timeline has no cue "${id}"; have: ${(tl.cues ?? []).map((x) => x.id).join(', ') || 'none'}`);
		const [start, end] = range(c.at);
		return {...c, start, end};
	};
	const within = (list: Span[] | undefined) => (frame: number) => {
		const s = (list ?? []).find((x) => frame >= at(x.from) && frame < at(x.to));
		return s ? {...s, start: at(s.from), end: at(s.to)} : undefined;
	};
	return {
		timeline: tl,
		fps,
		bpm,
		beatsPerBar,
		beat: framesPerBeat,
		barLen,
		total: tl.totalFrames,
		at,
		bar,
		pos,
		range,
		shot,
		section,
		cue,
		shotAt: within(tl.shots),
		sectionAt: within(tl.sections),
		seconds: (frame: number) => frame / fps,
	};
}
