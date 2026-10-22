// Parallel capture in frame order, and the H.264 writer it feeds.
//
// renderOrdered: frames are dealt to the tabs round-robin (tab k renders frames k, k+N, k+2N, …); finished pictures wait
// in a small reorder buffer and are handed to `onFrame` strictly in order. Tabs never run more than `window` frames ahead
// of the writer, which bounds memory (a 1080p PNG is 2–4 MB).
// 固定分帧而不是共享队列：Chrome 的栅格化在一个标签里多少带着它之前渲过的帧的痕迹（见 README「确定性与浮点」，
// 极个别像素差 1–4/255），共享队列下每个标签渲哪些帧随调度变，两次整片渲染会有几帧不一样；固定分帧后同样的
// --from/--to/--tabs 每次的标签历史都相同，整片重渲通常逐帧哈希相同（metal；偶有 1–2 帧差几个像素，swangle 差得多些）。
import type {Page} from './chrome';
import {captureFrame, type CompMeta} from './session';
import {ffmpeg} from './tools';

export type CaptureFormat = 'png' | 'jpeg';

export type OrderedStats = {
	frames: number;
	wallMs: number;
	/** sums over all frames, ms */
	settleMs: number;
	captureMs: number;
	/** time the writer spent in onFrame (ffmpeg back-pressure included) */
	writeMs: number;
	/** time tabs sat idle because the reorder window was full */
	stallMs: number;
	perTab: number[];
};

export async function renderOrdered(o: {
	pages: Page[];
	comp: CompMeta;
	frames: number[];
	format: CaptureFormat;
	quality?: number;
	window?: number;
	onFrame: (n: number, data: Uint8Array, index: number) => Promise<void> | void;
	onProgress?: (done: number, total: number, elapsedMs: number) => void;
}): Promise<OrderedStats> {
	const {pages, comp, frames} = o;
	const window = o.window ?? pages.length * 3;
	const ready = new Map<number, Uint8Array>();
	const stats: OrderedStats = {frames: frames.length, wallMs: 0, settleMs: 0, captureMs: 0, writeMs: 0, stallMs: 0, perTab: pages.map(() => 0)};
	let written = 0;
	let failure: unknown = null;
	let waiters: (() => void)[] = [];
	const pulse = () => {
		const w = waiters;
		waiters = [];
		for (const f of w) f();
	};
	const changed = () => new Promise<void>((res) => waiters.push(res));
	const t0 = performance.now();

	const worker = async (page: Page, tab: number) => {
		for (let i = tab; i < frames.length; i += pages.length) {
			const ts = performance.now();
			while (!failure && i - written >= window) await changed();
			stats.stallMs += performance.now() - ts;
			if (failure) return;
			try {
				const r = await captureFrame(page, comp, frames[i], o.format, o.quality);
				stats.settleMs += r.settleMs;
				stats.captureMs += r.captureMs;
				stats.perTab[tab]++;
				ready.set(i, r.data);
			} catch (e) {
				failure ??= new Error(`frame ${frames[i]} (tab ${tab}): ${e instanceof Error ? e.message : String(e)}`);
			}
			pulse();
		}
	};

	const writer = async () => {
		while (written < frames.length) {
			while (!failure && !ready.has(written)) await changed();
			if (failure) return;
			const data = ready.get(written)!;
			ready.delete(written);
			const tw = performance.now();
			try {
				await o.onFrame(frames[written], data, written);
			} catch (e) {
				failure ??= e;
				pulse();
				return;
			}
			stats.writeMs += performance.now() - tw;
			written++;
			o.onProgress?.(written, frames.length, performance.now() - t0);
			pulse();
		}
	};

	await Promise.all([writer(), ...pages.map((p, i) => worker(p, i))]);
	if (failure) throw failure;
	stats.wallMs = performance.now() - t0;
	return stats;
}

/**
 * ffmpeg fed with a stream of PNG or JPEG pictures on stdin, writing H.264 (High, yuv420p, BT.709 limited range,
 * tagged), no audio. Pictures are converted from RGB to BT.709 by swscale; JPEG input is first decoded to RGB so the
 * BT.601 YCbCr of the JPEG is not reinterpreted as BT.709.
 */
export class H264Writer {
	private proc: ReturnType<typeof Bun.spawn>;
	private stderr: Promise<string>;
	private exited = false;
	readonly args: string[];

	constructor(o: {path: string; fps: number; format: CaptureFormat; crf: number; preset: string; width: number; height: number}) {
		const decoder = o.format === 'png' ? 'png' : 'mjpeg';
		this.args = [
			ffmpeg(), '-v', 'error', '-y',
			'-f', 'image2pipe', '-c:v', decoder, '-framerate', String(o.fps), '-i', '-',
			'-an',
			// setparams: ffmpeg 9 (measured) takes the colour tags from the frames, and scale only sets matrix and range
			'-vf',
			`format=rgb24,scale=w=${o.width}:h=${o.height}:out_color_matrix=bt709:out_range=tv:flags=bicubic+accurate_rnd+full_chroma_int,format=yuv420p,` +
				'setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv',
			'-c:v', 'libx264', '-preset', o.preset, '-crf', String(o.crf), '-profile:v', 'high', '-pix_fmt', 'yuv420p',
			'-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv',
			'-r', String(o.fps), '-movflags', '+faststart', '-f', 'mp4', o.path,
		];
		const proc = Bun.spawn(this.args, {stdin: 'pipe', stdout: 'ignore', stderr: 'pipe'});
		this.proc = proc;
		this.stderr = new Response(proc.stderr as ReadableStream).text();
		proc.exited.then(() => {
			this.exited = true;
		});
	}

	async write(picture: Uint8Array): Promise<void> {
		if (this.exited) throw new Error(`ffmpeg exited early (${this.proc.exitCode}): ${await this.stderr}`);
		const sink = this.proc.stdin as import('bun').FileSink;
		sink.write(picture);
		await sink.flush();
	}

	async close(): Promise<void> {
		await (this.proc.stdin as import('bun').FileSink).end();
		const code = await this.proc.exited;
		if (code !== 0) throw new Error(`ffmpeg failed (${code}): ${await this.stderr}`);
	}

	kill(): void {
		try {
			this.proc.kill();
		} catch {
			/* gone */
		}
	}
}
