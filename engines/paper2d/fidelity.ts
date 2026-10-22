// Fidelity check: render frames of a composition and compare each with the same frame of a reference master.
//   bun fidelity.ts --film <dir> [--comp <id>] --ref <master.mp4> --frames 0,150,305 [--out dir] [--gl name]
//                   [--mine dir] [--images] [--threshold 34] [--ref-offset k]
// The reference is decoded to its native 4:2:0 planes (no ffmpeg colour conversion). Its matrix and range come from
// the stream: BT.709 when tagged so (render.ts writes BT.709 limited range), BT.601 otherwise; full range for
// yuvj420p or color_range=pc. Metrics, all in dB:
//   Y      luma PSNR: our RGB -> Y with the reference's matrix and range, against the reference Y plane (no chroma
//          subsampling involved)
//   YUV    ffmpeg-style weighted (4 Y + U + V) / 6, our chroma box-averaged 2x2 like a JPEG encoder
//   RGB    the reference converted to RGB (its matrix and range, center-sited bilinear chroma), against our RGB
// --mine compares PNGs already rendered (named <comp>_<frame>.png) instead of rendering. --images writes, for every
// frame, <out>/cmp_<frame>.png (ours | reference, half size, over the 8x amplified difference). With --mine the
// composition id (--comp) names the files; no browser is started.
import {existsSync, mkdirSync, writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {resolveFilm} from './lib/film';
import {captureFrame, fmtMs, openSession, parseArgs, parseFrames, runCli, sessionArgs, type CompMeta} from './lib/session';
import {ffmpeg, probeStream, run} from './lib/tools';

const args = parseArgs();
if (typeof args.frames !== 'string' || typeof args.ref !== 'string' || (typeof args.mine === 'string' && typeof args.comp !== 'string')) {
	console.error('usage: bun fidelity.ts --film <dir> [--comp <id>] --ref <master.mp4> --frames a,b,c [--out dir] [--mine dir --comp <id>] [--images] [--gl name]');
	process.exit(2);
}
const ref = resolve(args.ref);
const threshold = typeof args.threshold === 'string' ? Number(args.threshold) : 34;
/** compare our frame n with reference frame n + offset (alignment check: the best match should be offset 0) */
const refOffset = typeof args['ref-offset'] === 'string' ? Number(args['ref-offset']) : 0;

type Probe = {width: number; height: number; fps: number; pixFmt: string; range: string; space: string};
async function probe(file: string): Promise<Probe> {
	const s = await probeStream(file, 'v', 'width,height,r_frame_rate,pix_fmt,color_range,color_space');
	if (!s) throw new Error(`${file} has no video stream`);
	const [a, b] = String(s.r_frame_rate).split('/').map(Number);
	return {width: s.width, height: s.height, fps: a / b, pixFmt: s.pix_fmt, range: s.color_range ?? 'unknown', space: s.color_space ?? 'unknown'};
}

/** Frame n of the reference as raw 4:2:0 planes, exactly as decoded. */
async function refPlanes(n: number, p: Probe): Promise<Uint8Array> {
	// accurate input seek: the first frame whose pts is at or after (n - 0.25) / fps is frame n
	const t = Math.max(0, (n - 0.25) / p.fps).toFixed(6);
	const data = await run([ffmpeg(), '-v', 'error', '-ss', t, '-i', ref, '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', p.pixFmt, '-']);
	const need = (p.width * p.height * 3) / 2;
	if (data.length !== need) throw new Error(`reference frame ${n}: got ${data.length} bytes, expected ${need}`);
	return data;
}

async function pngToRgb(png: Uint8Array, w: number, h: number): Promise<Uint8Array> {
	const data = await run([ffmpeg(), '-v', 'error', '-f', 'png_pipe', '-i', '-', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], png);
	if (data.length !== w * h * 3) throw new Error(`our frame decodes to ${data.length} bytes, expected ${w * h * 3} (rendered at another size?)`);
	return data;
}

async function writeRgbPng(path: string, rgb: Uint8Array, w: number, h: number) {
	await run([ffmpeg(), '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', `${w}x${h}`, '-i', '-', '-frames:v', '1', path], rgb);
}

const psnr = (mse: number) => (mse === 0 ? 99 : 10 * Math.log10((255 * 255) / mse));
const clamp8 = (v: number) => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v));

/**
 * The reference's matrix (BT.601: Kr 0.299, Kb 0.114; BT.709: Kr 0.2126, Kb 0.0722) in its range: full ("pc", JPEG)
 * or limited ("tv"). For BT.601 these are the classic JPEG constants (0.168736, 0.331264, 1.402, 0.344136, …).
 */
function matrix(range: string, space: 'bt601' | 'bt709') {
	const full = range === 'pc';
	const [kr, kb] = space === 'bt709' ? [0.2126, 0.0722] : [0.299, 0.114];
	const kg = 1 - kr - kb;
	const luma = (r: number, g: number, b: number) => kr * r + kg * g + kb * b;
	return {
		y: (r: number, g: number, b: number) => (full ? luma(r, g, b) : 16 + (219 / 255) * luma(r, g, b)),
		cb: (r: number, g: number, b: number) => 128 + (full ? 1 : 224 / 255) * ((b - luma(r, g, b)) / (2 * (1 - kb))),
		cr: (r: number, g: number, b: number) => 128 + (full ? 1 : 224 / 255) * ((r - luma(r, g, b)) / (2 * (1 - kr))),
		rgb: (y: number, cb: number, cr: number): [number, number, number] => {
			const Y = full ? y : ((y - 16) * 255) / 219;
			const u = full ? cb - 128 : ((cb - 128) * 255) / 224;
			const v = full ? cr - 128 : ((cr - 128) * 255) / 224;
			const r = Y + 2 * (1 - kr) * v;
			const b = Y + 2 * (1 - kb) * u;
			return [r, (Y - kr * r - kb * b) / kg, b];
		},
	};
}

type Result = {frame: number; y: number; yuv: number; rgb: number; maxDiff: number; over16: number; renderMs?: number};

function compare(n: number, mine: Uint8Array, planes: Uint8Array, w: number, h: number, range: string, space: 'bt601' | 'bt709'): {r: Result; refRgb: Uint8Array} {
	const m = matrix(range, space);
	const cw = w / 2;
	const ch = h / 2;
	const Yp = planes.subarray(0, w * h);
	const Up = planes.subarray(w * h, w * h + cw * ch);
	const Vp = planes.subarray(w * h + cw * ch);
	let seY = 0;
	for (let i = 0, j = 0; i < w * h; i++, j += 3) {
		const d = clamp8(m.y(mine[j], mine[j + 1], mine[j + 2])) - Yp[i];
		seY += d * d;
	}
	let seU = 0;
	let seV = 0;
	for (let cy = 0; cy < ch; cy++)
		for (let cx = 0; cx < cw; cx++) {
			let cb = 0;
			let cr = 0;
			for (let dy = 0; dy < 2; dy++)
				for (let dx = 0; dx < 2; dx++) {
					const j = ((cy * 2 + dy) * w + cx * 2 + dx) * 3;
					cb += m.cb(mine[j], mine[j + 1], mine[j + 2]);
					cr += m.cr(mine[j], mine[j + 1], mine[j + 2]);
				}
			const k = cy * cw + cx;
			const du = clamp8(cb / 4) - Up[k];
			const dv = clamp8(cr / 4) - Vp[k];
			seU += du * du;
			seV += dv * dv;
		}
	// reference to RGB: center-sited chroma, bilinear (JPEG "fancy" upsampling)
	const refRgb = new Uint8Array(w * h * 3);
	const chroma = (P: Uint8Array, x: number, y: number) => {
		const fx = Math.min(Math.max((x + 0.5) / 2 - 0.5, 0), cw - 1);
		const fy = Math.min(Math.max((y + 0.5) / 2 - 0.5, 0), ch - 1);
		const x0 = Math.floor(fx);
		const y0 = Math.floor(fy);
		const x1 = Math.min(x0 + 1, cw - 1);
		const y1 = Math.min(y0 + 1, ch - 1);
		const ax = fx - x0;
		const ay = fy - y0;
		const top = P[y0 * cw + x0] * (1 - ax) + P[y0 * cw + x1] * ax;
		const bot = P[y1 * cw + x0] * (1 - ax) + P[y1 * cw + x1] * ax;
		return top * (1 - ay) + bot * ay;
	};
	let seRGB = 0;
	let maxDiff = 0;
	let over16 = 0;
	for (let y = 0; y < h; y++)
		for (let x = 0; x < w; x++) {
			const i = y * w + x;
			const [r, g, b] = m.rgb(Yp[i], chroma(Up, x, y), chroma(Vp, x, y));
			const j = i * 3;
			refRgb[j] = clamp8(r);
			refRgb[j + 1] = clamp8(g);
			refRgb[j + 2] = clamp8(b);
			let px = 0;
			for (let c = 0; c < 3; c++) {
				const d = Math.abs(mine[j + c] - refRgb[j + c]);
				seRGB += d * d;
				if (d > px) px = d;
			}
			if (px > maxDiff) maxDiff = px;
			if (px > 16) over16++;
		}
	const mseY = seY / (w * h);
	const mseU = seU / (cw * ch);
	const mseV = seV / (cw * ch);
	return {
		r: {frame: n, y: psnr(mseY), yuv: psnr((4 * mseY + mseU + mseV) / 6), rgb: psnr(seRGB / (w * h * 3)), maxDiff, over16: over16 / (w * h)},
		refRgb,
	};
}

/** A w x h sheet: ours | reference at half size on top, the 8x amplified absolute difference (half size) below. */
function sheet(mine: Uint8Array, refRgb: Uint8Array, w: number, h: number) {
	const hw = w / 2;
	const hh = h / 2;
	const out = new Uint8Array(w * h * 3);
	const tile = (ox: number, oy: number, f: (x: number, y: number, c: number) => number) => {
		for (let y = 0; y < hh; y++)
			for (let x = 0; x < hw; x++)
				for (let c = 0; c < 3; c++) out[((oy + y) * w + ox + x) * 3 + c] = f(x, y, c);
	};
	const at = (x: number, y: number, dx: number, dy: number, c: number) => ((2 * y + dy) * w + 2 * x + dx) * 3 + c;
	const avg = (src: Uint8Array, x: number, y: number, c: number) =>
		(src[at(x, y, 0, 0, c)] + src[at(x, y, 1, 0, c)] + src[at(x, y, 0, 1, c)] + src[at(x, y, 1, 1, c)]) / 4;
	tile(0, 0, (x, y, c) => avg(mine, x, y, c));
	tile(hw, 0, (x, y, c) => avg(refRgb, x, y, c));
	tile(hw / 2, hh, (x, y, c) => {
		let d = 0;
		for (let dy = 0; dy < 2; dy++)
			for (let dx = 0; dx < 2; dx++) d = Math.max(d, Math.abs(mine[at(x, y, dx, dy, c)] - refRgb[at(x, y, dx, dy, c)]));
		return Math.min(255, d * 8);
	});
	return {data: out, w, h};
}

await runCli(async () => {
	const film = resolveFilm(args);
	const frames = parseFrames(String(args.frames), film.grid);
	const p = await probe(ref);
	console.log(`reference ${ref}: ${p.width}x${p.height} ${p.fps} fps ${p.pixFmt} range=${p.range} matrix=${p.space}`);
	if (p.pixFmt !== 'yuvj420p' && p.pixFmt !== 'yuv420p') throw new Error(`reference pixel format ${p.pixFmt} not handled`);
	const space = p.space === 'bt709' ? 'bt709' : 'bt601';
	if (p.space !== 'bt709' && p.space !== 'bt470bg' && p.space !== 'smpte170m' && p.space !== 'unknown')
		console.warn(`reference matrix is ${p.space}; this tool converts with BT.601`);
	const range = p.pixFmt === 'yuvj420p' ? 'pc' : p.range === 'pc' ? 'pc' : 'tv';
	console.log(`comparing with ${space} ${range === 'pc' ? 'full' : 'limited'} range`);

	const session = typeof args.mine === 'string' ? null : await openSession({film, ...sessionArgs(args)});
	const comp: CompMeta | null = session?.comp ?? null;
	const compId = comp?.id ?? String(args.comp);
	const outDir = resolve(typeof args.out === 'string' ? args.out : resolve(film.outDir, 'fidelity', compId));
	mkdirSync(outDir, {recursive: true});
	if (comp && (comp.width !== p.width || comp.height !== p.height)) throw new Error(`composition is ${comp.width}x${comp.height}, reference ${p.width}x${p.height}`);
	const results: Result[] = [];
	try {
		for (const n of frames) {
			let png: Uint8Array;
			let renderMs: number | undefined;
			if (session && comp) {
				const r = await captureFrame(session.pages[0], comp, n);
				png = r.data;
				renderMs = r.settleMs + r.captureMs;
				await Bun.write(resolve(outDir, `${compId}_${String(n).padStart(4, '0')}.png`), png);
			} else {
				const f = resolve(String(args.mine), `${compId}_${String(n).padStart(4, '0')}.png`);
				if (!existsSync(f)) throw new Error(`missing ${f}`);
				png = new Uint8Array(await Bun.file(f).arrayBuffer());
			}
			const [mine, planes] = await Promise.all([pngToRgb(png, p.width, p.height), refPlanes(n + refOffset, p)]);
			const {r, refRgb} = compare(n, mine, planes, p.width, p.height, range, space);
			r.renderMs = renderMs;
			results.push(r);
			if (args.images || r.rgb < threshold) {
				const s = sheet(mine, refRgb, p.width, p.height);
				await writeRgbPng(resolve(outDir, `cmp_${String(n).padStart(4, '0')}.png`), s.data, s.w, s.h);
			}
			console.log(
				`f${String(n).padEnd(5)} Y ${r.y.toFixed(2).padStart(6)}  YUV ${r.yuv.toFixed(2).padStart(6)}  RGB ${r.rgb.toFixed(2).padStart(6)}  max ${String(r.maxDiff).padStart(3)}  >16: ${(r.over16 * 100).toFixed(3)}%${renderMs !== undefined ? `  (${fmtMs(renderMs)})` : ''}${r.rgb < threshold ? '  < threshold' : ''}`,
			);
		}
	} finally {
		await session?.close();
	}
	const lines = ['frame\tY\tYUV\tRGB\tmax\tover16\trender_ms', ...results.map((r) => `${r.frame}\t${r.y.toFixed(2)}\t${r.yuv.toFixed(2)}\t${r.rgb.toFixed(2)}\t${r.maxDiff}\t${r.over16.toFixed(6)}\t${r.renderMs?.toFixed(0) ?? ''}`)];
	writeFileSync(resolve(outDir, 'psnr.tsv'), `${lines.join('\n')}\n`);
	const below = results.filter((r) => r.rgb < threshold);
	console.log(`${results.length} frames; min RGB ${Math.min(...results.map((r) => r.rgb)).toFixed(2)} dB, min Y ${Math.min(...results.map((r) => r.y)).toFixed(2)} dB; below ${threshold} dB (RGB): ${below.map((r) => r.frame).join(', ') || 'none'}; table -> ${resolve(outDir, 'psnr.tsv')}`);
});
