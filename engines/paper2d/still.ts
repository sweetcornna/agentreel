// Render single frames of a composition as PNG.
//   bun still.ts --film <dir> [--comp <id>] --frames 0,150,600-700:20 [--out dir] [--scale 0.5] [--grid 24]
//                [--props '{"k":1}'] [--gl metal|angle|d3d11|vulkan|egl|swangle] [--prefix name] [--verbose]
// Frames are the composition's own frame numbers; with the film's timeline.json they can also be bar.beat positions
// ("2.3", "3.1+12f", "1.1-3.1:25"). --grid <fps> reads plain numbers as frames of another grid (e.g. a picture
// written on a 24-unit grid inside a 30 fps film) and rounds them to the nearest output frame.
// Files: <out>/<prefix or comp>_<frame, 4 digits>.png at width*scale x height*scale (default <film>/out/stills/<comp>).
// Any failure prints one "error: …" line and exits 1; session start-up is retried (lib/session.ts openSession).
import {mkdirSync} from 'node:fs';
import {resolve} from 'node:path';
import {resolveFilm} from './lib/film';
import {captureFrame, fmtMs, openSession, parseArgs, parseFrames, runCli, sessionArgs, UsageError} from './lib/session';

const USAGE = 'usage: bun still.ts --film <dir> [--comp <id>] --frames a,b,c|from-to:step|bar.beat [--out dir] [--scale s] [--grid fps] [--props json] [--gl name]';

await runCli(async () => {
	const args = parseArgs();
	if (typeof args.frames !== 'string') throw new UsageError(USAGE);
	const film = resolveFilm(args);
	const scale = typeof args.scale === 'string' ? Number(args.scale) : 1;
	const gridFps = typeof args.grid === 'string' ? Number(args.grid) : null;
	if (gridFps !== null && !(gridFps > 0)) throw new UsageError('--grid needs the grid rate, e.g. --grid 24');

	const t0 = performance.now();
	const session = await openSession({film, scale, ...sessionArgs(args)});
	const {comp, timings} = session;
	const requested = parseFrames(args.frames, film.grid);
	const outDir = resolve(typeof args.out === 'string' ? args.out : resolve(film.outDir, 'stills', comp.id));
	const prefix = typeof args.prefix === 'string' ? args.prefix : comp.id;
	mkdirSync(outDir, {recursive: true});
	console.log(
		`${comp.id}: ${comp.width}x${comp.height} @ ${comp.fps} fps, ${comp.durationInFrames} frames; gl ${session.gl}; ` +
			`bundle ${fmtMs(timings.bundleMs)}, chrome ${fmtMs(timings.launchMs)}, page ${fmtMs(timings.pagesMs)}`,
	);
	const frames = requested.map((f) => (gridFps ? Math.round((f * comp.fps) / gridFps) : f));
	const page = session.pages[0];
	const pad = Math.max(4, String(comp.durationInFrames - 1).length);
	const perFrame: number[] = [];
	try {
		for (let i = 0; i < frames.length; i++) {
			const n = frames[i];
			const file = resolve(outDir, `${prefix}_${String(n).padStart(pad, '0')}.png`);
			const r = await captureFrame(page, comp, n);
			const tw = performance.now();
			await Bun.write(file, r.data);
			const writeMs = performance.now() - tw;
			perFrame.push(r.settleMs + r.captureMs + writeMs);
			const where = film.grid ? ` ${film.grid.pos(n)}` : '';
			const label = gridFps ? `grid ${requested[i]} -> f${n}` : `f${n}${where}`;
			console.log(
				`${label.padEnd(18)} render ${fmtMs(r.settleMs).padStart(7)} (rounds ${r.rounds})  capture ${fmtMs(r.captureMs).padStart(6)}  write ${fmtMs(writeMs).padStart(6)}  -> ${file}`,
			);
		}
	} finally {
		await session.close();
	}
	if (perFrame.length > 1) {
		const warm = perFrame.slice(1);
		console.log(`${perFrame.length} frames in ${fmtMs(performance.now() - t0)}; first ${fmtMs(perFrame[0])}, then mean ${fmtMs(warm.reduce((a, b) => a + b, 0) / warm.length)}`);
	}
});
