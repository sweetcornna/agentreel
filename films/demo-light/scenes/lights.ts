// S01「暗处的光」：雾里六粒暖光，随乐谱的 pulse 事件一明一灭（没有乐谱就按八分音符网格）。
// 镜头对焦在 60 m 外，这几粒光在镜头前 3–6 m，所以全是光斑：1.3–2.1 是叶形光圈，2.1 起虹膜在
// 十帧里打开成圆形（同一支镜头，光圈形状的两种都演示一次）。雾里的微尘被它们照亮。2.4 起六粒光
// 沿弧线飞向字心，越远光斑越小，3.1 合成一点，交给 S02 的字。

import * as THREE from 'three';
import type { StageContext } from '@beat3d/core/types';
import type { ApertureSpec } from '@beat3d/lib/aperture';
import { BokehPoints } from '@beat3d/world/bokeh';
import { Dust } from '@beat3d/world/dust';
import type { HazeLight } from '@beat3d/world/haze';
import { clamp, easeInOutCubic, easeOutCubic, smoothstep } from '@beat3d/lib/easing';
import { cocPixels, pxPerMetre, type LensState } from '@beat3d/lib/lens';
import { snoise2 } from '@beat3d/lib/noise';
import { toneOf } from '@beat3d/lib/color';
import { CUE, p1Pulses, type Pulse } from './cues';
import { DIM_HUE, PAL } from './palette';

/** 字心：光点最后汇到这里，S02 的字在这里点亮。 */
export const WORD_CENTRE = new THREE.Vector3(0, 0.02, 0);

/** 六粒光：相对起始机位（z = 9.2）的 x、y 与距离（m）。屏幕上左右错开、远近错开。 */
const SPOTS: [number, number, number][] = [
  [-0.42, 0.055, 3.6],
  [0.3, -0.075, 4.4],
  [-0.1, 0.105, 5.6],
  [0.38, 0.085, 3.2],
  [-0.33, -0.09, 5.0],
  [0.075, -0.018, 6.4],
];
/** 网格回退时轮流点亮的顺序（相邻两下不落在同一侧）。 */
const ORDER = [2, 0, 3, 5, 1, 4];
const CAM_Z0 = 9.2;
/** 一粒光的通量（线性 HDR × 1080p 像素²）：光斑越大越暗，总能量不变。 */
const FLUX = 9000;

/** 帧 f 时镜头的光圈：先叶形，再张开成圆。 */
export function apertureAt(f: number): ApertureSpec {
  const u = clamp((f - CUE.irisOpen[0]) / (CUE.irisOpen[1] - CUE.irisOpen[0]));
  // 量化到 1/32：编译过的光圈按值缓存，运动模糊的子帧不会每次都重算。
  const open = Math.round(easeInOutCubic(u) * 32) / 32;
  return open >= 1 ? { shape: 'circle' } : { shape: 'leaf', open };
}

interface Lamp {
  base: THREE.Vector3;
  /** 屏幕上的横向位置（-1..1），按它把 pan 分给最近的一粒 */
  screenX: number;
  pulses: Pulse[];
  first: number;
}

export class DarkLights {
  readonly points: BokehPoints;
  readonly dustPoints: BokehPoints;
  private readonly dust: Dust;
  private readonly lamps: Lamp[];

  constructor(private readonly ctx: StageContext, key: string) {
    this.points = new BokehPoints(ctx, SPOTS.length, { profile: 'hero', aperture: apertureAt(0) });
    this.lamps = SPOTS.map(([x, y, d]) => ({ base: new THREE.Vector3(x, y, CAM_Z0 - d), screenX: x / (d * (24.89 / 2 / 85)), pulses: [], first: Infinity }));
    // 把每一下分给一粒光：事件带 light（音高序号）就按音高从左到右排（低音在左，像琴键）；只有 pan / side
    // 就给屏幕上那一侧最近的一粒（不连着两下给同一粒）；什么都没有（网格回退）就按 ORDER 轮流。
    const byX = this.lamps.map((l, i) => ({ i, x: l.screenX })).sort((a, b) => a.x - b.x).map((q) => q.i);
    let last = -1;
    for (const p of p1Pulses()) {
      let k: number;
      if (p.light !== null) k = byX[((Math.round(p.light) % byX.length) + byX.length) % byX.length];
      else if (p.pan !== null) {
        const pan = p.pan * 0.6;
        k = this.lamps.map((l, i) => ({ i, d: Math.abs(l.screenX - pan) + (i === last ? 9 : 0) })).sort((a, b) => a.d - b.d)[0].i;
      } else k = ORDER[p.index % ORDER.length];
      this.lamps[k].pulses.push(p);
      this.lamps[k].first = Math.min(this.lamps[k].first, p.frame);
      last = k;
    }
    // 还没轮到的光，在 P1 结束时也亮起来（它们一起聚拢）。
    for (const l of this.lamps) if (!Number.isFinite(l.first)) l.first = CUE.gather - 4;
    this.dust = new Dust(`${key}/S01`, 1400, new THREE.Vector3(-1.5, -0.62, 1.0), new THREE.Vector3(1.5, 0.62, 8.4));
    this.dustPoints = new BokehPoints(ctx, this.dust.count, { aperture: apertureAt(0) });
  }

  /** 一粒光在帧 f 的亮度（0..1+）：每一下 3 帧亮起、13 帧回落到余辉；第一下之前是黑的。 */
  private level(l: Lamp, f: number): number {
    if (f < l.first - 0.5) return 0;
    let k = 0;
    for (const p of l.pulses) {
      const d = f - p.frame;
      if (d < -0.5 || d > 16) continue;
      const env = d < 0 ? 0 : Math.min(easeOutCubic(clamp(d / 3)), 1 - smoothstep(3, 14, d));
      k = Math.max(k, env * (0.55 + 0.6 * p.bright));
    }
    // 余辉：亮过一次就留一点；聚拢时整体抬起来一些。
    const rest = 0.11 * easeOutCubic(clamp((f - l.first + 0.5) / 3));
    const gather = 0.35 * smoothstep(CUE.gather - 2, CUE.gather + 6, f);
    return Math.max(k, rest, gather);
  }

  /** 光在帧 f 的位置：原位轻微浮动；2.4 起沿弧线飞向字心，3.1 汇成一点。 */
  private position(l: Lamp, i: number, f: number): THREE.Vector3 {
    const t = f / 24;
    const p = l.base.clone();
    p.x += 0.01 * snoise2(t * 0.4, i * 7.1);
    p.y += 0.006 * snoise2(t * 0.35, i * 3.3 + 40);
    const u = easeInOutCubic(clamp((f - CUE.gather) / (CUE.word - CUE.gather)));
    if (u <= 0) return p;
    // 弧线：往外绕一点再收进来，不走直线。
    const side = new THREE.Vector3(-(p.y - WORD_CENTRE.y), p.x - WORD_CENTRE.x, 0).multiplyScalar(0.9 * Math.sin(Math.PI * u));
    return p.lerp(WORD_CENTRE, u).add(side);
  }

  /** 摆好帧 f 的光与微尘；返回给雾的光源。3.1 之后光点交给字，几帧内熄掉。`extra`：也照亮微尘的
   *  别的光（S02 的字）。 */
  update(f: number, lens: LensState, camera: THREE.PerspectiveCamera, extinction: number, energy: number, extra: HazeLight[] = []): HazeLight[] {
    const lights: HazeLight[] = [];
    const ap = apertureAt(f);
    const gone = smoothstep(CUE.word - 1, CUE.word + 6, f);
    let n = 0;
    this.lamps.forEach((l, i) => {
      const lv = this.level(l, f) * (1 - gone);
      if (lv <= 0) return;
      const p = this.position(l, i, f);
      // 汇到一点时能量不叠加：合成的那一点不比最亮的一粒更亮。
      const merge = smoothstep(CUE.word - 6, CUE.word, f);
      const k = lv * (1 - 0.8 * merge);
      // AgX 会把很暗的饱和光推红：余辉里掺一点更黄的暖色，明灭只改能量、不改色相。
      const warm = 0.2 * (1 - smoothstep(0.1, 0.6, lv));
      const L = PAL.lamp;
      const c: [number, number, number] = [L[0] * FLUX * k, (L[1] + (PAL.amber[1] - L[1]) * warm) * FLUX * k, (L[2] + (PAL.amber[2] - L[2]) * warm) * FLUX * k];
      this.points.set(n++, p.x, p.y, p.z, c[0], c[1], c[2], 0.002);
      // 雾里的光被它自己的失焦柔化：雾里的光晕不比光斑更锐。
      lights.push({ position: p, color: toneOf(PAL.amber, 0.3 * k, DIM_HUE), radius: Math.max(0.012, this.defocusMetres(p, lens, camera)) });
    });
    this.points.update(n, { lens, chroma: 0.035, extinction, aperture: ap, maxRadius: 160 });
    const nd = this.dust.fill({ time: f / 24, camera: camera.position, lights: [...lights, ...extra], planes: [], gain: 40 * (0.6 + 0.8 * energy), extinction, maxFlux: 900 }, this.dustPoints, null);
    this.dustPoints.update(nd.back, { lens, extinction, aperture: ap });
    return lights;
  }

  /** 一点在镜头里的弥散圆半径（m）：它在雾里该有多软。 */
  private defocusMetres(p: THREE.Vector3, lens: LensState, cam: THREE.PerspectiveCamera): number {
    const z = -p.clone().applyMatrix4(cam.matrixWorldInverse).z;
    if (z <= 0.05) return 0.05;
    return (0.5 * cocPixels(z, lens, this.ctx.width)) / pxPerMetre(z, cam.fov, this.ctx.height);
  }

  dispose(): void {
    this.points.dispose();
    this.dustPoints.dispose();
  }
}
