// 一部片 = 一个目录：<film>/src/index.ts（registerRoot 注册组合）、可选的 <film>/public/（staticFile 指向这里）、
// 可选的 <film>/timeline.json（拍网格；有它时帧列表可以写「小节.拍」）、输出 <film>/out/（不入库）。
// 不按片目录组织时用 --entry <file> [--public <dir>]，输出落在当前目录的 out/。
import {existsSync, readFileSync, statSync} from 'node:fs';
import {resolve} from 'node:path';
import {beatGrid, type BeatGrid, type Timeline} from '../components/timeline';
import {UsageError} from './session';

export type Film = {
	/** the film directory, or null with --entry */
	dir: string | null;
	entry: string;
	publicDir: string | null;
	timeline: Timeline | null;
	grid: BeatGrid | null;
	outDir: string;
};

const isDir = (p: string) => existsSync(p) && statSync(p).isDirectory();

export function resolveFilm(args: Record<string, string | true>): Film {
	const filmArg = typeof args.film === 'string' ? args.film : null;
	const entryArg = typeof args.entry === 'string' ? args.entry : null;
	if (!filmArg && !entryArg) throw new UsageError('give --film <film directory> (or --entry <file> [--public <dir>])');
	const dir = filmArg ? resolve(filmArg) : null;
	if (dir && !isDir(dir)) throw new UsageError(`--film ${filmArg}: not a directory`);
	let entry: string;
	if (entryArg) entry = resolve(entryArg);
	else {
		const cands = ['src/index.ts', 'src/index.tsx'].map((p) => resolve(dir!, p));
		const hit = cands.find((p) => existsSync(p));
		if (!hit) throw new UsageError(`--film ${filmArg}: no src/index.ts (the file that calls registerRoot)`);
		entry = hit;
	}
	if (!existsSync(entry)) throw new UsageError(`entry ${entry} does not exist`);
	const publicDir =
		typeof args.public === 'string' ? resolve(args.public) : dir && isDir(resolve(dir, 'public')) ? resolve(dir, 'public') : null;
	if (publicDir && !isDir(publicDir)) throw new UsageError(`--public ${String(args.public)}: not a directory`);
	const tlFile = typeof args.timeline === 'string' ? resolve(args.timeline) : dir ? resolve(dir, 'timeline.json') : null;
	let timeline: Timeline | null = null;
	if (tlFile && existsSync(tlFile)) timeline = JSON.parse(readFileSync(tlFile, 'utf8')) as Timeline;
	else if (typeof args.timeline === 'string') throw new UsageError(`--timeline ${args.timeline} does not exist`);
	return {dir, entry, publicDir, timeline, grid: timeline ? beatGrid(timeline) : null, outDir: dir ? resolve(dir, 'out') : resolve('out')};
}
