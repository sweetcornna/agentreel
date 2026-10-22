// 脚本在拍哪部片。每个脚本都要**最先** import 这个文件（在任何 import 了 src/lib/time.ts 的东西之前）：
// 它读 --film，把片子的时间线（以及 tempo 为 "score" 的时间线要的乐谱拍点）挂到 globalThis 上，
// Bun 这一侧的 time.ts 从那里取。页面包里的同一份选择由 scripts/build.ts 烘进去。
//   --film <dir>        片子目录（默认本引擎旁边的 ../../films/demo-light）
//   --timeline <json>   默认 <film>/timeline.json
//   --score <wav>       默认 <film>/out/music/score.wav
//   --events <json>     默认 <film>/out/music/events.json
//   --facts <json>      默认 <film>/facts.json（可选）
// 命令行里的路径相对当前目录，或者给绝对路径。
//
// 片子目录里有 timeline.json、scenes/index.ts（默认导出 FilmDef；可以没有，那样每个镜头都是占位牌）、
// out/music/（配乐，可选），渲染结果写进 out/。

import { existsSync, readFileSync } from 'node:fs';
import { isAbsolute, relative, resolve } from 'node:path';

/** 引擎目录（engines/beat3d）。 */
export const ENGINE = resolve(import.meta.dir, '../..');

export function arg(name: string): string | undefined {
  const argv = process.argv;
  const i = argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < argv.length && !argv[i + 1].startsWith('--') ? argv[i + 1] : undefined;
}
const cliPath = (p: string): string => (isAbsolute(p) ? p : resolve(process.cwd(), p));

export const FILM_DIR = arg('film') ? cliPath(arg('film')!) : resolve(ENGINE, '../../films/demo-light');
if (!existsSync(FILM_DIR)) throw new Error(`--film ${FILM_DIR} does not exist`);
export const FILM_NAME = FILM_DIR.split(/[\\/]/).filter(Boolean).pop() ?? 'film';

export const TIMELINE_PATH = arg('timeline') ? cliPath(arg('timeline')!) : resolve(FILM_DIR, 'timeline.json');
if (!existsSync(TIMELINE_PATH)) throw new Error(`timeline ${TIMELINE_PATH} does not exist`);
export const TIMELINE_JSON = JSON.parse(readFileSync(TIMELINE_PATH, 'utf8')) as { tempo?: string; bpm: number; fps: number; [k: string]: unknown };

/** 片子的镜头入口（默认导出 FilmDef）；没有就是 null，每个镜头都是占位牌。 */
export const SCENES_ENTRY: string | null = (() => {
  for (const name of ['index.ts', 'index.tsx', 'index.js']) {
    const p = resolve(FILM_DIR, 'scenes', name);
    if (existsSync(p)) return p;
  }
  return null;
})();

/** 这部片每次运行写出的东西：静帧、成片、日志、各自的构建。 */
export const OUT = resolve(FILM_DIR, 'out');

export const MUSIC = {
  score: arg('score') ? cliPath(arg('score')!) : resolve(OUT, 'music/score.wav'),
  events: arg('events') ? cliPath(arg('events')!) : resolve(OUT, 'music/events.json'),
};
export const FACTS_PATH = arg('facts') ? cliPath(arg('facts')!) : resolve(FILM_DIR, 'facts.json');

/** 乐谱的拍点时间（秒），给 tempo 为 "score" 的时间线用；其他情况或还没有配乐时是 null。 */
export function scoreBeats(): number[] | null {
  if (TIMELINE_JSON.tempo !== 'score' || !existsSync(MUSIC.events)) return null;
  const ev = JSON.parse(readFileSync(MUSIC.events, 'utf8'));
  const raw: unknown[] = Array.isArray(ev.beats) ? ev.beats : [];
  const beats = raw.map((b) => (typeof b === 'number' ? b : (b as { t: number }).t));
  return beats.length >= 2 ? beats : null;
}

const g = globalThis as { __B3D_TIMELINE__?: unknown; __B3D_BEATS__?: number[] | null };
g.__B3D_TIMELINE__ = TIMELINE_JSON;
g.__B3D_BEATS__ = scoreBeats();

/** 日志里显示的路径：相对当前目录更短时用相对路径。 */
export const shown = (p: string): string => {
  const r = relative(process.cwd(), p);
  return r && !r.startsWith('..' + '/..') && r.length < p.length ? r : p;
};

/** 日志里的一行：这次运行用的是哪部片、哪份时间线、配乐与 facts。 */
export function selectionLine(): string {
  const opt = (p: string, none: string) => (existsSync(p) ? shown(p) : none);
  return `film ${shown(FILM_DIR)}, timeline ${shown(TIMELINE_PATH)}, scenes ${SCENES_ENTRY ? shown(SCENES_ENTRY) : 'none (placeholders)'}, score ${opt(MUSIC.score, 'none')}, events ${opt(MUSIC.events, 'none (beat grid)')}, facts ${opt(FACTS_PATH, 'none')}`;
}
