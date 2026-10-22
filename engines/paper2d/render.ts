// Render a composition (or a frame range of it) to a silent H.264 MP4, and optionally mux the score in.
//   bun render.ts --film <dir> [--comp <id>] [--out <film>/out/<comp>.mp4] [--from 0] [--to last] [--tabs 6]
//                 [--crf 15] [--preset slow] [--capture png|jpeg] [--quality 100] [--scale 1]
//                 [--keep 10,500 --keep-dir dir] [--hashes file.tsv] [--props json] [--gl name] [--verbose]
//                 [--audio score.wav [--with-audio-out film.mp4]]
// --from/--to are inclusive composition frames (or bar.beat positions with the film's timeline.json). Tabs capture
// interleaved frames in parallel; pictures go to ffmpeg's stdin in frame order (no temporary picture folder). Output: H.264 High, yuv420p, BT.709 limited range with colour
// tags, the composition's fps, no audio track; written to <out>.part and renamed when ffmpeg has finished cleanly.
// --keep also saves those frames' captured pictures (for comparing against still.ts); --hashes writes one line per
// frame with a hash of the captured picture (PNG bytes are a function of the pixels) to compare runs frame by frame.
// A JSON report with timings is written next to the video (<out>.render.json). --audio also writes the film with
// sound (default <out minus .mp4>_audio.mp4; see lib/mux.ts and mux.ts).
import {existsSync, mkdirSync, renameSync, unlinkSync, writeFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {resolveFilm} from './lib/film';
import {muxAudio} from './lib/mux';
import {H264Writer, renderOrdered, type CaptureFormat} from './lib/pipeline';
import {fmtMs, openSession, parseArgs, parseFrames, runCli, sessionArgs, UsageError} from './lib/session';

const USAGE =
	'usage: bun render.ts --film <dir> [--comp <id>] [--out <file.mp4>] [--from n] [--to n] [--tabs 6] [--crf 15] [--preset slow] [--capture png|jpeg] [--quality q] [--keep a,b --keep-dir dir] [--hashes file] [--audio score.wav]';

await runCli(async () => {
	const args = parseArgs();
	if (args.help) throw new UsageError(USAGE);
	const film = resolveFilm(args);
	const num = (k: string, d: number) => {
		const v = args[k];
		if (v === undefined) return d;
		const x = Number(v);
		if (typeof v !== 'string' || !Number.isFinite(x)) throw new UsageError(`--${k} needs a number`);
		return x;
	};
	const tabs = num('tabs', 6);
	const crf = num('crf', 15);
	const preset = typeof args.preset === 'string' ? args.preset : 'slow';
	const format = (typeof args.capture === 'string' ? args.capture : 'png') as CaptureFormat;
	if (format !== 'png' && format !== 'jpeg') throw new UsageError('--capture must be png or jpeg');
	const quality = format === 'jpeg' ? num('quality', 100) : undefined;
	const scale = num('scale', 1);
	if (!(tabs >= 1 && tabs <= 16 && Number.isInteger(tabs))) throw new UsageError('--tabs must be an integer in 1–16');
	const audio = typeof args.audio === 'string' ? resolve(args.audio) : null;
	if (audio && !existsSync(audio)) throw new UsageError(`--audio ${args.audio} does not exist`);
	const hashFile = typeof args.hashes === 'string' ? resolve(args.hashes) : null;
	const hashes: string[] = [];

	const t0 = performance.now();
	const session = await openSession({film, pages: tabs, scale, ...sessionArgs(args)});
	const {comp} = session;
	const out = resolve(typeof args.out === 'string' ? args.out : resolve(film.outDir, `${comp.id}.mp4`));
	mkdirSync(dirname(out), {recursive: true});
	const keep = new Set(typeof args.keep === 'string' ? parseFrames(args.keep, film.grid) : []);
	const keepDir = typeof args['keep-dir'] === 'string' ? resolve(args['keep-dir']) : resolve(dirname(out), 'keep');
	if (keep.size) mkdirSync(keepDir, {recursive: true});
	const at = (k: 'from' | 'to', d: number) => {
		const v = args[k];
		if (typeof v === 'string' && /^\d+\.\d+/.test(v)) return parseFrames(v, film.grid)[0];
		return num(k, d);
	};
	const from = at('from', 0);
	const to = at('to', comp.durationInFrames - 1);
	if (!(Number.isInteger(from) && Number.isInteger(to) && from >= 0 && to >= from && to < comp.durationInFrames)) {
		await session.close();
		throw new UsageError(`--from/--to must be integers with 0 ≤ from ≤ to ≤ ${comp.durationInFrames - 1}`);
	}
	const frames = Array.from({length: to - from + 1}, (_, i) => from + i);
	const width = Math.round(comp.width * scale);
	const height = Math.round(comp.height * scale);
	const startMs = performance.now() - t0;
	console.log(
		`${comp.id}: frames ${from}–${to} (${frames.length}) at ${width}x${height} ${comp.fps} fps; ${tabs} tabs, gl ${session.gl}, capture ${format}${quality ? ` q${quality}` : ''}, ` +
			`x264 crf ${crf} ${preset}; start-up ${fmtMs(startMs)}`,
	);

	const part = `${out}.part`;
	const enc = new H264Writer({path: part, fps: comp.fps, format, crf, preset, width, height});
	const pad = Math.max(4, String(comp.durationInFrames - 1).length);
	let lastLog = 0;
	let stats;
	const tr = performance.now();
	let wallMs = 0;
	try {
		stats = await renderOrdered({
			pages: session.pages,
			comp,
			frames,
			format,
			quality,
			onFrame: async (n, data) => {
				if (keep.has(n)) await Bun.write(resolve(keepDir, `${comp.id}_${String(n).padStart(pad, '0')}.${format === 'png' ? 'png' : 'jpg'}`), data);
				if (hashFile) hashes.push(`${n}\t${Bun.hash(data).toString(16)}`);
				await enc.write(data);
			},
			onProgress: (done, total, ms) => {
				if (ms - lastLog < 5000 && done !== total) return;
				lastLog = ms;
				const fps = (done / ms) * 1000;
				console.log(`  ${String(done).padStart(5)}/${total}  ${fps.toFixed(1)} fps  ETA ${fmtMs(((total - done) / fps) * 1000)}`);
			},
		});
		await enc.close();
		// wall time of the render includes ffmpeg finishing the last frames
		wallMs = performance.now() - tr;
	} catch (e) {
		enc.kill();
		if (existsSync(part)) unlinkSync(part);
		throw e;
	} finally {
		await session.close();
	}
	renameSync(part, out);
	if (hashFile) writeFileSync(hashFile, `${hashes.join('\n')}\n`);
	const totalMs = performance.now() - t0;
	const n = stats.frames;
	let withAudio: string | null = null;
	if (audio) {
		withAudio = resolve(typeof args['with-audio-out'] === 'string' ? args['with-audio-out'] : out.replace(/\.mp4$/i, '') + '_audio.mp4');
		const m = await muxAudio({video: out, audio, out: withAudio});
		console.log(`muxed ${audio} (${m.audioSeconds.toFixed(3)} s) under the picture (${m.videoSeconds.toFixed(3)} s) -> ${withAudio}`);
	}
	const report = {
		comp: comp.id,
		gl: session.gl,
		frames: {from, to, count: n},
		size: `${width}x${height}`,
		fps: comp.fps,
		tabs,
		capture: format,
		quality,
		x264: {crf, preset},
		ffmpeg: enc.args.join(' '),
		startupMs: Math.round(startMs),
		renderWallMs: Math.round(wallMs),
		totalMs: Math.round(totalMs),
		throughputFps: Number(((n / wallMs) * 1000).toFixed(2)),
		meanPerFrameInTab: {settleMs: Math.round(stats.settleMs / n), captureMs: Math.round(stats.captureMs / n)},
		writerMsPerFrame: Number((stats.writeMs / n).toFixed(1)),
		tabStallMs: Math.round(stats.stallMs),
		framesPerTab: stats.perTab,
		output: out,
		bytes: Bun.file(out).size,
		withAudio,
	};
	writeFileSync(out.replace(/\.mp4$/i, '') + '.render.json', `${JSON.stringify(report, null, 2)}\n`);
	console.log(
		`done: ${n} frames in ${fmtMs(wallMs)} (${report.throughputFps} fps; total with start-up ${fmtMs(totalMs)}); ` +
			`per frame in a tab: render ${report.meanPerFrameInTab.settleMs} ms + capture ${report.meanPerFrameInTab.captureMs} ms; ` +
			`writer ${report.writerMsPerFrame} ms/frame; ${(report.bytes / 1e6).toFixed(1)} MB -> ${out}`,
	);
});
