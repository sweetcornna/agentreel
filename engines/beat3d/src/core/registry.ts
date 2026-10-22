// 哪个程序拍哪个镜头：来自片子的 FilmDef（films/<name>/scenes/index.ts）。一个镜头被两个程序认领直接报错；
// 没人认领的镜头画占位牌，所以时间线比镜头程序写得快时也能整片渲染。

import type { FilmDef, ProgramDef } from './types';
import type { PostOverrides } from '../post/params';
import { placeholderProgram } from '../scenes/placeholder';
import { SHOTS } from '../lib/time';
import { setDefaultAperture } from '../lib/aperture';

export { placeholderProgram };

let film: FilmDef = { programs: [] };
const byShot = new Map<string, ProgramDef>();
let lab: Map<string, ProgramDef> | null = null;

/** 登记片子（一次，在引擎创建任何程序之前）。 */
export function useFilm(def: FilmDef): void {
  film = def;
  byShot.clear();
  const known = new Set(SHOTS.map((s) => s.id));
  for (const p of def.programs) {
    for (const s of p.shots) {
      if (byShot.has(s)) throw new Error(`Shot ${s} claimed by both ${byShot.get(s)!.id} and ${p.id}`);
      if (!known.has(s)) console.warn(`program ${p.id} claims ${s}, which the timeline does not have`);
      byShot.set(s, p);
    }
  }
  setDefaultAperture(def.aperture);
}

/** 片子的风格试验（`--lab`）顶替它们认领的镜头；传 false 关掉。 */
export function useLab(on: boolean): void {
  lab = on && film.lab ? new Map(film.lab.flatMap((p) => p.shots.map((s) => [s, p] as const))) : null;
}

export function programFor(shotId: string): ProgramDef {
  return lab?.get(shotId) ?? byShot.get(shotId) ?? placeholderProgram;
}

/** 片子的整体质感与逐镜头覆盖。 */
export const filmPost = (): PostOverrides | undefined => film.post;
export const shotPost = (shotId: string): PostOverrides | undefined => film.shotPost?.[shotId];
export const currentFilm = (): FilmDef => film;
