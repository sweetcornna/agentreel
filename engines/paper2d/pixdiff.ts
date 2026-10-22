// Compare rendered pictures.
//   bun pixdiff.ts <dirA> <dirB>
//       every picture (png/jpg) present in both directories under the same name: exact pixel comparison
//   bun pixdiff.ts --video <file.mp4> --start <comp frame of the video's first frame> <dir>
//       every <comp>_<frame>.png in dir against that frame of the video, decoded with the video's own matrix and range
// Per picture: identical, or the number of differing channel values, the largest difference and the RGB PSNR.
import {existsSync, readdirSync} from 'node:fs';
import {resolve} from 'node:path';
import {parseArgs, runCli, UsageError} from './lib/session';
import {ffmpeg, probeStream, run} from './lib/tools';

const decodeFile = (file: string) => run([ffmpeg(), '-v', 'error', '-i', file, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']);

type Diff = {identical: boolean; differing: number; max: number; psnr: number};
function diff(a: Uint8Array, b: Uint8Array): Diff {
	if (a.length !== b.length) throw new Error(`sizes differ (${a.length} vs ${b.length} bytes)`);
	let differing = 0;
	let max = 0;
	let se = 0;
	for (let i = 0; i < a.length; i++) {
		const d = a[i] - b[i];
		if (d !== 0) {
			differing++;
			const ad = d < 0 ? -d : d;
			if (ad > max) max = ad;
			se += d * d;
		}
	}
	return {identical: differing === 0, differing, max, psnr: se === 0 ? 99 : 10 * Math.log10((255 * 255) / (se / a.length))};
}

const line = (name: string, d: Diff) =>
	`${name.padEnd(28)} ${d.identical ? 'identical' : `differs: ${d.differing} channel values, max ${d.max}, PSNR ${d.psnr.toFixed(2)} dB`}`;

await runCli(async () => {
	const {args, dirs} = splitArgs(process.argv.slice(2), ['video', 'start']);
	if (typeof args.video === 'string') {
		if (dirs.length !== 1 || typeof args.start !== 'string') throw new UsageError('usage: bun pixdiff.ts --video <file.mp4> --start <frame> <dir>');
		const video = resolve(args.video);
		const start = Number(args.start);
		const probe = await probeStream(video, 'v', 'r_frame_rate,color_range,color_space');
		if (!probe) throw new Error(`${video} has no video stream`);
		const [a, b] = String(probe.r_frame_rate).split('/').map(Number);
		const fps = a / b;
		const matrix = probe.color_space === 'bt709' ? 'bt709' : 'bt601';
		const range = probe.color_range === 'pc' ? 'pc' : 'tv';
		console.log(`${video}: ${fps} fps, decoding as ${matrix} ${range} range`);
		let worst = 99;
		for (const f of readdirSync(dirs[0]).filter((x) => /_(\d+)\.png$/.test(x)).sort()) {
			const n = Number(/_(\d+)\.png$/.exec(f)![1]);
			const k = n - start;
			const ref = await decodeFile(resolve(dirs[0], f));
			const got = await run([
				ffmpeg(), '-v', 'error', '-ss', Math.max(0, (k - 0.25) / fps).toFixed(6), '-i', video, '-frames:v', '1',
				'-vf', `scale=in_color_matrix=${matrix}:in_range=${range}:out_range=pc:flags=bicubic+accurate_rnd+full_chroma_int,format=rgb24`,
				'-f', 'rawvideo', '-pix_fmt', 'rgb24', '-',
			]);
			const d = diff(ref, got);
			worst = Math.min(worst, d.psnr);
			console.log(line(`${f} (video frame ${k})`, d));
		}
		console.log(`lowest PSNR ${worst.toFixed(2)} dB`);
		return;
	}
	if (dirs.length !== 2) throw new UsageError('usage: bun pixdiff.ts <dirA> <dirB>');
	const [da, db] = dirs.map((d) => resolve(d));
	const names = readdirSync(da).filter((f) => /\.(png|jpe?g)$/i.test(f) && existsSync(resolve(db, f))).sort();
	if (!names.length) throw new Error(`no pictures with the same name in ${da} and ${db}`);
	let same = 0;
	for (const f of names) {
		const d = diff(await decodeFile(resolve(da, f)), await decodeFile(resolve(db, f)));
		if (d.identical) same++;
		console.log(line(f, d));
	}
	console.log(`${same}/${names.length} identical`);
});

/** Positional directories mixed with flags; only the flags in `valued` take the next argument as their value. */
function splitArgs(argv: string[], valued: string[]): {args: Record<string, string | true>; dirs: string[]} {
	const flags: string[] = [];
	const dirs: string[] = [];
	for (let i = 0; i < argv.length; i++) {
		const a = argv[i];
		if (!a.startsWith('--')) dirs.push(a);
		else if (valued.includes(a.slice(2)) && i + 1 < argv.length) flags.push(a, argv[++i]);
		else flags.push(a);
	}
	return {args: parseArgs(flags), dirs};
}
