// Shared plumbing for the render CLIs: argument parsing, frame lists, and a session = in-memory bundle + local server
// + headless Chrome + N tabs, each with one composition selected and sized.
import type {BeatGrid} from '../components/timeline';
import {buildBundle} from './build';
import {Chrome, type GlName, GL_FLAGS, type Page} from './chrome';
import type {Film} from './film';
import {startServer, type RunningServer} from './server';
import {defaultGl} from './tools';

/** An error that a retry cannot fix (bad arguments, unknown composition). */
export class UsageError extends Error {}

export type CompMeta = {id: string; width: number; height: number; fps: number; durationInFrames: number; defaultProps: Record<string, unknown>};

export function parseArgs(argv = process.argv.slice(2)): Record<string, string | true> {
	const out: Record<string, string | true> = {};
	for (let i = 0; i < argv.length; i++) {
		const a = argv[i];
		if (!a.startsWith('--')) throw new UsageError(`unexpected argument "${a}"`);
		const eq = a.indexOf('=');
		if (eq > 0) {
			out[a.slice(2, eq)] = a.slice(eq + 1);
			continue;
		}
		const next = argv[i + 1];
		if (next !== undefined && !next.startsWith('--')) {
			out[a.slice(2)] = next;
			i++;
		} else out[a.slice(2)] = true;
	}
	return out;
}

/**
 * "0,150,305", "600-1100:20" (step defaults to 1) or a mix: "0,10-20:5". With the film's beat grid, positions too:
 * "2.3", "3.1+12f", "1.1-3.1:25" (a position range excludes its end, like a shot's [from, to)). Positions that land
 * between frames are rounded to the nearest frame.
 */
export function parseFrames(spec: string, grid?: BeatGrid | null): number[] {
	const frames: number[] = [];
	const pos = (s: string) => {
		if (!grid) throw new UsageError(`"${s}" is a bar.beat position, but the film has no timeline.json`);
		return Math.round(grid.at(s));
	};
	for (const part of spec.split(',').map((s) => s.trim()).filter(Boolean)) {
		const m = /^(\d+)-(\d+)(?::(\d+))?$/.exec(part);
		if (m) {
			const [a, b, step] = [Number(m[1]), Number(m[2]), Number(m[3] ?? 1)];
			if (step <= 0 || b < a) throw new UsageError(`bad frame range "${part}"`);
			for (let f = a; f <= b; f += step) frames.push(f);
			continue;
		}
		if (/^\d+$/.test(part)) {
			frames.push(Number(part));
			continue;
		}
		const r = /^(\d+\.\d+[^-:]*)-(\d+\.\d+[^-:]*)(?::(\d+))?$/.exec(part);
		if (r) {
			const [a, b, step] = [pos(r[1]), pos(r[2]), Number(r[3] ?? 1)];
			if (step <= 0 || b <= a) throw new UsageError(`bad position range "${part}"`);
			for (let f = a; f < b; f += step) frames.push(f);
			continue;
		}
		if (/^\d+\.\d+/.test(part)) {
			try {
				frames.push(pos(part));
			} catch (e) {
				throw e instanceof UsageError ? e : new UsageError(e instanceof Error ? e.message : String(e));
			}
			continue;
		}
		throw new UsageError(`bad frame "${part}" (use 12, 10-20, 10-20:5${grid ? ', 2.3, 3.1+12f or 1.1-3.1:25' : ''})`);
	}
	return frames;
}

export const fmtMs = (ms: number): string => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(0)} ms`);

export type Session = {
	chrome: Chrome;
	server: RunningServer;
	pages: Page[];
	comp: CompMeta;
	gl: GlName;
	logs: string[];
	timings: {bundleMs: number; launchMs: number; pagesMs: number};
	close(): Promise<void>;
};

type SessionOptions = {
	film: Film;
	/** composition id; may be left out when the film registers exactly one */
	comp?: string;
	pages?: number;
	scale?: number;
	gl?: string;
	inputProps?: Record<string, unknown>;
	timeout?: number;
	verbose?: boolean;
	/** extra Chrome command-line flags (also PAPER2D_CHROME_FLAGS, space separated) */
	chromeFlags?: string[];
	/** attempts to start the session (default 3): Chrome launch, the CDP connection, the bundle (a scene file caught
	 * half-written by another agent's editor) and page start-up can fail transiently */
	attempts?: number;
};

export async function openSession(o: SessionOptions): Promise<Session> {
	const attempts = o.attempts ?? 3;
	for (let i = 1; ; i++) {
		try {
			return await startSession(o);
		} catch (e) {
			if (e instanceof UsageError || i >= attempts) throw e;
			const msg = (e instanceof Error ? e.message : String(e)).split('\n').slice(0, 6).join('\n');
			console.error(`[paper2d] session start failed (attempt ${i}/${attempts}), retrying in ${i} s:\n${msg}`);
			await Bun.sleep(1000 * i);
		}
	}
}

/** The CLI options every command shares, read into session options. */
export function sessionArgs(args: Record<string, string | true>): Omit<SessionOptions, 'film'> {
	return {
		comp: typeof args.comp === 'string' ? args.comp : undefined,
		gl: typeof args.gl === 'string' ? args.gl : undefined,
		inputProps: typeof args.props === 'string' ? (JSON.parse(args.props) as Record<string, unknown>) : {},
		timeout: typeof args.timeout === 'string' ? Number(args.timeout) : undefined,
		verbose: !!args.verbose,
		chromeFlags: typeof args['chrome-flags'] === 'string' ? args['chrome-flags'].split(/\s+/).filter(Boolean) : [],
	};
}

async function startSession(o: SessionOptions): Promise<Session> {
	const gl = (o.gl ?? defaultGl()) as GlName;
	if (!(gl in GL_FLAGS)) throw new UsageError(`--gl must be one of ${Object.keys(GL_FLAGS).join(', ')} (got ${gl})`);
	const scale = o.scale ?? 1;
	if (!(scale > 0 && scale <= 4)) throw new UsageError(`--scale must be in (0, 4], got ${scale}`);
	const bundle = await buildBundle(o.film.entry);
	const server = startServer(bundle, {publicDir: o.film.publicDir, inputProps: o.inputProps, timeout: o.timeout});
	const t1 = performance.now();
	const flags = [...(process.env.PAPER2D_CHROME_FLAGS ?? '').split(/\s+/).filter(Boolean), ...(o.chromeFlags ?? [])];
	let chrome: Chrome;
	try {
		chrome = await Chrome.launch(gl, flags);
	} catch (e) {
		server.stop();
		throw e;
	}
	const kill = () => chrome.killNow();
	process.on('exit', kill);
	const onSignal = () => {
		kill();
		process.exit(130);
	};
	process.on('SIGINT', onSignal);
	process.on('SIGTERM', onSignal);
	const t2 = performance.now();
	const logs: string[] = [];
	const pages: Page[] = [];
	let comp: CompMeta | null = null;
	const close = async () => {
		await chrome.close();
		server.stop();
		process.off('exit', kill);
		process.off('SIGINT', onSignal);
		process.off('SIGTERM', onSignal);
	};
	try {
		for (let i = 0; i < (o.pages ?? 1); i++) {
			const page = await chrome.newPage(1920, 1080, scale);
			page.onConsole((type, text) => {
				const line = `[tab ${i}] ${type}: ${text}`;
				logs.push(line);
				if (o.verbose || type === 'error' || type === 'exception' || type === 'warning' || type === 'assert') console.error(line);
			});
			await page.goto(`${server.url}/`);
			const ok = await page.eval<boolean>(`typeof window.paper2d === 'object'`);
			if (!ok) throw new Error(`render page did not start; console:\n${logs.join('\n')}`);
			const all = await page.eval<CompMeta[]>('paper2d.compositions()');
			let id = o.comp;
			if (!id) {
				if (all.length !== 1) throw new UsageError(`give --comp: the film registers ${all.length} compositions (${all.map((c) => c.id).join(', ')})`);
				id = all[0].id;
			}
			const meta = all.find((c) => c.id === id);
			if (!meta) throw new UsageError(`no composition "${id}"; have: ${all.map((c) => c.id).join(', ')}`);
			await page.setViewport(meta.width, meta.height, scale);
			comp = await page.eval<CompMeta>(`paper2d.select(${JSON.stringify(id)})`);
			pages.push(page);
		}
	} catch (e) {
		await close();
		throw e;
	}
	return {
		chrome,
		server,
		pages,
		comp: comp!,
		gl,
		logs,
		timings: {bundleMs: bundle.ms, launchMs: t2 - t1, pagesMs: performance.now() - t2},
		close,
	};
}

/**
 * Runs a CLI body and turns every failure into one clear "error: …" line and exit code 1 (stack with --verbose), so
 * nothing ends in Bun's bare uncaught-exception banner.
 */
export async function runCli(main: () => Promise<number | void>): Promise<never> {
	const verbose = process.argv.includes('--verbose');
	const fail = (what: string, e: unknown) => {
		const err = e instanceof Error ? e : new Error(String(e));
		console.error(`error${what}: ${verbose ? (err.stack ?? err.message) : err.message}`);
		process.exit(1);
	};
	process.on('unhandledRejection', (e) => fail(' (unhandled rejection)', e));
	process.on('uncaughtException', (e) => fail(' (uncaught exception)', e));
	let code = 0;
	try {
		code = (await main()) ?? 0;
	} catch (e) {
		fail('', e);
	}
	process.exit(code);
}

/** Render frame n in a tab and return the composited picture. */
export async function captureFrame(page: Page, comp: CompMeta, n: number, format: 'png' | 'jpeg' = 'png', quality?: number) {
	const t0 = performance.now();
	const r = await page.eval<{frame: number; ms: number; rounds: number}>(`paper2d.frame(${n})`);
	const t1 = performance.now();
	const data = await page.screenshot(comp.width, comp.height, format, quality);
	const t2 = performance.now();
	return {data, settleMs: t1 - t0, pageMs: r.ms, rounds: r.rounds, captureMs: t2 - t1};
}
