// Frame-to-frame difference scan of a rendered video, to find jumps.
//   bun scan.ts <video.mp4> [--top 15] [--film <dir> | --timeline <file>] [--start <first frame>] [--out <prefix>]
// Every frame is decoded to luma at 480x270 (area-averaged, which also averages the film grain away) and compared with
// the frame before it. Per step n-1 -> n:
//   mad    mean absolute luma difference, 0–255
//   moved  share of pixels whose luma changed by more than 24
//   spike  mad over the median mad of the moving steps among the ten around it (held pictures, e.g. on twos, left
//          out): a hard cut or a jump stands alone (high spike); fast continuous motion is high mad, spike near 1
// Writes <prefix>_scan.tsv (all steps) and <prefix>_scan_top.png (one row per top step, ranked: frame n-1 | frame n |
// difference x4), and prints the top steps by mad. With the film's timeline.json (--film or --timeline) each step is
// annotated with its shot and bar.beat position (--start: the composition frame of the video's first frame, when the
// video is a range of the film); without it, with the time in seconds only.
import {existsSync, readFileSync, writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {beatGrid, type Timeline} from './components/timeline';
import {parseArgs, runCli, UsageError} from './lib/session';
import {ffmpeg, probeStream, run} from './lib/tools';

const W = 480;
const H = 270;
const TW = 320;
const TH = 180;
/** a step below this mad is a held picture (encoder noise only) */
const HELD = 0.3;

await runCli(async () => {
	const video = process.argv[2];
	if (!video || video.startsWith('--')) throw new UsageError('usage: bun scan.ts <video.mp4> [--top 15] [--film dir | --timeline file] [--start n] [--out prefix]');
	const args = parseArgs(process.argv.slice(3));
	const top = typeof args.top === 'string' ? Number(args.top) : 15;
	const file = resolve(video);
	const prefix = typeof args.out === 'string' ? resolve(args.out) : file.replace(/\.mp4$/i, '');
	const tlFile = typeof args.timeline === 'string' ? resolve(args.timeline) : typeof args.film === 'string' ? resolve(args.film, 'timeline.json') : null;
	if (tlFile && !existsSync(tlFile)) throw new UsageError(`no timeline at ${tlFile}`);
	const grid = tlFile ? beatGrid(JSON.parse(readFileSync(tlFile, 'utf8')) as Timeline) : null;
	const start = typeof args.start === 'string' ? Number(args.start) : 0;
	const probe = await probeStream(file, 'v', 'r_frame_rate,color_space,color_range');
	if (!probe) throw new Error(`${file} has no video stream`);
	const [fa, fb] = String(probe.r_frame_rate).split('/').map(Number);
	const fps = fa / fb;

	// pass 1: stream luma thumbnails, difference consecutive frames
	const t0 = performance.now();
	const p = Bun.spawn([ffmpeg(), '-v', 'error', '-i', file, '-vf', `scale=${W}:${H}:flags=area,format=gray`, '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], {stdout: 'pipe', stderr: 'pipe'});
	const size = W * H;
	let prev: Uint8Array | null = null;
	let buf = new Uint8Array(0);
	const mad: number[] = [0];
	const moved: number[] = [0];
	let frames = 0;
	const reader = (p.stdout as ReadableStream<Uint8Array>).getReader();
	for (;;) {
		const {value, done} = await reader.read();
		if (done) break;
		const joined = new Uint8Array(buf.length + value.length);
		joined.set(buf);
		joined.set(value, buf.length);
		let off = 0;
		while (joined.length - off >= size) {
			const cur = joined.slice(off, off + size);
			off += size;
			if (prev) {
				let sum = 0;
				let big = 0;
				for (let i = 0; i < size; i++) {
					const d = cur[i] > prev[i] ? cur[i] - prev[i] : prev[i] - cur[i];
					sum += d;
					if (d > 24) big++;
				}
				mad.push(sum / size);
				moved.push(big / size);
			}
			prev = cur;
			frames++;
		}
		buf = joined.slice(off);
	}
	const code = await p.exited;
	if (code !== 0) throw new Error(`ffmpeg decode failed: ${await new Response(p.stderr).text()}`);

	const spike = mad.map((m, n) => {
		if (n === 0) return 0;
		// held frames (pictures on twos repeat; mad near 0) are left out, or every step of a twos passage is a "spike"
		const around: number[] = [];
		for (let k = n - 5; k <= n + 5; k++) if (k !== n && k >= 1 && k < mad.length && mad[k] >= HELD) around.push(mad[k]);
		around.sort((a, b) => a - b);
		const med = around.length ? around[Math.floor(around.length / 2)] : 0;
		return m / Math.max(med, 0.05);
	});
	if (grid && Math.abs(grid.fps - fps) > 1e-6) console.warn(`[scan] the video is ${fps} fps, the timeline ${grid.fps} fps; positions assume the timeline's rate`);
	const where = (n: number) => {
		if (!grid) return '';
		const f = n + start;
		const shot = grid.shotAt(f) ?? grid.sectionAt(f);
		return `${shot ? `${shot.id} ` : ''}${grid.pos(f)}`;
	};
	const rows = ['step\tframe\ttime_s\tmad\tmoved\tspike\twhere'];
	for (let n = 1; n < mad.length; n++)
		rows.push(`${n - 1}->${n}\t${n}\t${(n / fps).toFixed(3)}\t${mad[n].toFixed(3)}\t${moved[n].toFixed(4)}\t${spike[n].toFixed(2)}\t${where(n)}`);
	writeFileSync(`${prefix}_scan.tsv`, `${rows.join('\n')}\n`);
	const ranked = Array.from({length: mad.length - 1}, (_, i) => i + 1)
		.sort((a, b) => mad[b] - mad[a])
		.slice(0, top);
	const sorted = [...mad.slice(1)].sort((a, b) => a - b);
	console.log(
		`${file}: ${frames} frames at ${fps} fps decoded in ${((performance.now() - t0) / 1000).toFixed(1)} s; ` +
			`step mad median ${sorted[Math.floor(sorted.length / 2)].toFixed(2)}, 99th pct ${sorted[Math.floor(sorted.length * 0.99)].toFixed(2)}`,
	);
	console.log(`top ${top} steps by mean absolute luma difference (0–255):`);
	console.log('rank  step          time     mad    moved  spike  where');
	ranked.forEach((n, i) =>
		console.log(
			`${String(i + 1).padStart(4)}  ${`${n - 1}->${n}`.padEnd(12)} ${(n / fps).toFixed(2).padStart(6)}s ${mad[n].toFixed(2).padStart(6)} ${(moved[n] * 100).toFixed(1).padStart(6)}% ${spike[n].toFixed(1).padStart(6)}  ${where(n)}`,
		),
	);

	// pass 2: contact sheet of the ranked steps
	const rgbAt = (k: number) =>
		run([
			ffmpeg(), '-v', 'error', '-ss', Math.max(0, (k - 0.25) / fps).toFixed(6), '-i', file, '-frames:v', '1',
			'-vf', `scale=${TW}:${TH}:flags=area:in_color_matrix=${probe.color_space === 'bt709' ? 'bt709' : 'bt601'}:in_range=${probe.color_range === 'pc' ? 'pc' : 'tv'}:out_range=pc,format=rgb24`,
			'-f', 'rawvideo', '-pix_fmt', 'rgb24', '-',
		]);
	const sheet = new Uint8Array(TW * 3 * TH * ranked.length * 3);
	const rowBytes = TW * 3 * 3;
	for (const [r, n] of ranked.entries()) {
		const [a, b] = await Promise.all([rgbAt(n - 1), rgbAt(n)]);
		for (let y = 0; y < TH; y++) {
			const o = (r * TH + y) * rowBytes;
			sheet.set(a.subarray(y * TW * 3, (y + 1) * TW * 3), o);
			sheet.set(b.subarray(y * TW * 3, (y + 1) * TW * 3), o + TW * 3);
			for (let x = 0; x < TW * 3; x++) sheet[o + TW * 6 + x] = Math.min(255, Math.abs(a[y * TW * 3 + x] - b[y * TW * 3 + x]) * 4);
		}
	}
	const sheetFile = `${prefix}_scan_top.png`;
	const enc = Bun.spawn([ffmpeg(), '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', `${TW * 3}x${TH * ranked.length}`, '-i', '-', '-frames:v', '1', sheetFile], {stdin: 'pipe', stderr: 'pipe'});
	const sink = enc.stdin as import('bun').FileSink;
	sink.write(sheet);
	await sink.end();
	if ((await enc.exited) !== 0) throw new Error(`sheet encode failed: ${await new Response(enc.stderr).text()}`);
	console.log(`all steps -> ${prefix}_scan.tsv; contact sheet (rows in rank order: n-1 | n | diff x4) -> ${sheetFile}`);
});
