// 把 src/film.ts 和片子的镜头打成 film.js（浏览器用，单个 IIFE 文件），并把这次运行的选择烘进去
// （scripts/lib/film.ts）：时间线 → __TIMELINE__；配乐的事件（存在的话）→ __MUSIC_EVENTS__，烘之前先检查
// 它的拍点落在时间线的网格上（见 src/lib/time.ts；tempo 为 "score" 的时间线改为烘拍点时间 __MUSIC_BEATS__）；
// 片子的 facts.json（存在的话）→ __FACTS__（src/lib/facts.ts）；public/fonts 里有哪些字体文件 → __FONT_FILES__。
//
// 模块解析（用插件做，所以片子目录不需要 node_modules，也不需要构建配置）：
//   @film          -> <film>/scenes/index.ts（没有就是一部空片：每个镜头都是占位牌）
//   @beat3d/<path> -> engines/beat3d/src/<path>
//   three, three/* -> 引擎自己的 three.js
//   bun scripts/build.ts [--film <dir>] [--minify]
import { FACTS_PATH, MUSIC, OUT, SCENES_ENTRY, TIMELINE_JSON, selectionLine, shown } from './lib/film';
import { existsSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import { resolve } from 'node:path';
import type { BunPlugin } from 'bun';

const root = resolve(import.meta.dir, '..');
const timeline = TIMELINE_JSON;
export const MUSIC_EVENTS = MUSIC.events;

// Bun's file loader emits content-hashed copies of binary assets; drop stale ones before each build.
function pruneDist(dist: string): void {
  if (!existsSync(dist)) return;
  for (const name of readdirSync(dist)) {
    if (name === 'film.js' || name === 'film.js.map') continue;
    rmSync(resolve(dist, name), { recursive: true, force: true });
  }
}

type Ev = { t: number; type: string; [k: string]: unknown };

/**
 * The score's note events (and beat times), or null before the score exists. On a fixed-tempo
 * timeline, throws if its beat grid disagrees with the timeline by more than 1 ms: the picture must
 * never drift from the music silently.
 */
export function loadMusicEvents(): { events: Ev[]; beats: number[] } | null {
  if (!existsSync(MUSIC_EVENTS)) return null;
  const ev = JSON.parse(readFileSync(MUSIC_EVENTS, 'utf8'));
  const raw: unknown[] = Array.isArray(ev.beats) ? ev.beats : [];
  const beats = raw.map((b) => (typeof b === 'number' ? b : (b as { t: number }).t));
  if (timeline.tempo === 'score') {
    beats.forEach((t, i) => {
      if (typeof t !== 'number' || (i > 0 && !(t > beats[i - 1]))) throw new Error(`${MUSIC_EVENTS}: beat ${i} at ${t} s is not after beat ${i - 1}`);
    });
  } else {
    const spb = 60 / timeline.bpm;
    beats.forEach((t, i) => {
      if (typeof t !== 'number' || Math.abs(t - i * spb) > 0.001) throw new Error(`${MUSIC_EVENTS}: beat ${i} at ${t} s is off the ${timeline.bpm} BPM grid (${(i * spb).toFixed(4)} s)`);
    });
  }
  const events: Ev[] = Array.isArray(ev.events) ? ev.events.filter((e: Ev) => typeof e.t === 'number' && typeof e.type === 'string') : [];
  return { events: events.sort((a, b) => a.t - b.t), beats };
}

export const DIST = resolve(OUT, '.build/dist');

/** The film's facts.json as text (validated JSON), or null before it exists. */
export function loadFacts(): string | null {
  if (!existsSync(FACTS_PATH)) return null;
  const text = readFileSync(FACTS_PATH, 'utf8');
  JSON.parse(text);
  return text;
}

/** Resolves the film, the engine alias and three.js (see the header). */
export const filmPlugin: BunPlugin = {
  name: 'beat3d-film',
  setup(b) {
    b.onResolve({ filter: /^@film$/ }, () => (SCENES_ENTRY ? { path: SCENES_ENTRY } : { path: 'empty-film', namespace: 'beat3d' }));
    b.onLoad({ filter: /^empty-film$/, namespace: 'beat3d' }, () => ({ contents: 'export default { programs: [] };', loader: 'ts' }));
    b.onResolve({ filter: /^@beat3d\// }, (a) => ({ path: Bun.resolveSync(`./src/${a.path.slice('@beat3d/'.length)}`, root) }));
    b.onResolve({ filter: /^three(\/.*)?$/ }, (a) => ({ path: Bun.resolveSync(a.path, root) }));
  },
};

/**
 * Build into `outdir` (default <film>/out/.build/dist, used by the viewer). Render sessions pass a
 * private directory, so a rebuild for the viewer never changes what a running render loads.
 */
export async function build(minify = false, outdir = DIST): Promise<{ musicEvents: number | null; facts: boolean }> {
  pruneDist(outdir);
  const music = loadMusicEvents();
  const facts = loadFacts();
  const fonts = readdirSync(resolve(root, 'public/fonts')).filter((f) => /\.(woff2?|otf|ttf)$/.test(f));
  const define: Record<string, string> = {
    __TIMELINE__: JSON.stringify(JSON.stringify(TIMELINE_JSON)),
    __FONT_FILES__: JSON.stringify(JSON.stringify(fonts)),
  };
  if (music) define.__MUSIC_EVENTS__ = JSON.stringify(JSON.stringify(music.events));
  if (music && timeline.tempo === 'score') define.__MUSIC_BEATS__ = JSON.stringify(JSON.stringify(music.beats));
  if (facts) define.__FACTS__ = JSON.stringify(facts);
  const r = await Bun.build({
    entrypoints: [resolve(root, 'src/film.ts')],
    outdir,
    target: 'browser',
    format: 'iife',
    naming: 'film.js',
    minify,
    sourcemap: 'linked',
    define,
    plugins: [filmPlugin],
  });
  if (!r.success) {
    for (const l of r.logs) console.error(l);
    throw new Error('build failed');
  }
  return { musicEvents: music ? music.events.length : null, facts: facts !== null };
}

if (import.meta.main) {
  const t = performance.now();
  const { musicEvents } = await build(process.argv.includes('--minify'));
  console.log(`${shown(resolve(DIST, 'film.js'))} built in ${Math.round(performance.now() - t)} ms; ${selectionLine()}; ${musicEvents === null ? 'no score events, beat grid only' : `${musicEvents} score events baked in`}`);
}
