// S04「片尾」：固定机位。走廊的弦在 7.1 收进纵深里的一点（corridor.ts 的 SPOT），这一点在 H3 落成
// 一枚光斑：7.1 时焦点还在它身上（一粒很亮的点），之后十几帧把焦点拉到镜头前 1.4 m 的片名上，
// 这一点就在圆形光圈里张开成一枚大光斑，随乐谱 S04 的 pulse 轻轻呼吸。片名「agentreel」和仓库地址
// 在 7.1+4f 到 7.3 之间淡入（不闪），8.4 起整体淡出到三成，9.1 引擎切黑。

import * as THREE from 'three';
import type { StageContext } from '@beat3d/core/types';
import { LightText, type CharPlacement } from '@beat3d/text/lighttext';
import { FONT } from '@beat3d/text/fonts';
import { BokehPoints } from '@beat3d/world/bokeh';
import { Dust } from '@beat3d/world/dust';
import type { HazeLight } from '@beat3d/world/haze';
import { clamp, easeInOutCubic, easeOutCubic, smoothstep } from '@beat3d/lib/easing';
import type { LensState } from '@beat3d/lib/lens';
import { snoise2 } from '@beat3d/lib/noise';
import { scaled, toneOf, type RGB } from '@beat3d/lib/color';
import { CLOCK } from '@beat3d/lib/time';
import { CUE, endPulses, type Pulse } from './cues';
import { DIM_HUE, PAL } from './palette';
import { SPOT } from './corridor';
import { STOP, STOP_PITCH } from './rig';

export const END_TITLE = 'agentreel';
export const END_URL = 'github.com/sweetcornna/agentreel';

/** 片名离镜头的距离（m），沿停住的机位的视轴。 */
const TITLE_DIST = 1.4;
/** 光斑的通量（线性 HDR × 1080p 像素²）与发光体半径（m）。 */
const SPOT_FLUX = 12000;
const SPOT_EMITTER = 0.02;

/** 停住的机位的三个轴。 */
const FWD = new THREE.Vector3(0, Math.sin(STOP_PITCH), -Math.cos(STOP_PITCH));
const UP = new THREE.Vector3(0, Math.cos(STOP_PITCH), Math.sin(STOP_PITCH));
const RIGHT = new THREE.Vector3(1, 0, 0);

const F = (p: string): number => CLOCK.frameAt(p);

/** S04 的焦点：7.1 在光斑上，7.1 → 7.2+8f 按屈光度拉到片名上。 */
export function endFocus(f: number): number {
  const spot = STOP.distanceTo(SPOT);
  const u = easeInOutCubic(clamp((f - CUE.spot) / (F('7.2') + 8 - CUE.spot)));
  return 1 / (1 / spot + (1 / TITLE_DIST - 1 / spot) * u);
}

/** 8.4 起淡出到三成（9.1 之后引擎画纯黑）。 */
export const endFade = (f: number): number => 1 - 0.7 * smoothstep(CUE.fadeOut, CUE.cut, f);

export class EndCard {
  readonly spot: BokehPoints;
  readonly motes: BokehPoints;
  readonly objects: THREE.Object3D[];
  private readonly title: LightText;
  private readonly url: LightText;
  private readonly dust: Dust;
  private readonly pulses: Pulse[];

  constructor(ctx: StageContext, key: string) {
    this.spot = new BokehPoints(ctx, 1, { profile: 'hero', aperture: { shape: 'circle' } });
    this.title = new LightText(ctx, END_TITLE, { latin: [FONT.mono, 400], aperture: { shape: 'circle' } });
    this.url = new LightText(ctx, END_URL, { latin: [FONT.mono, 300], aperture: { shape: 'circle' } });
    const fadeIn = [CUE.spot + 4, F('7.3')];
    // 字号与行距按距离给（在画面上：片名的 em 约为画高的 7.5%）。
    const k = TITLE_DIST / 2.2;
    this.layout(this.title, END_TITLE, 0.046 * k, -0.028 * k, fadeIn[0], scaled(PAL.warmWhite, 1.9));
    this.layout(this.url, END_URL, 0.0165 * k, -0.083 * k, fadeIn[0] + 6, scaled(PAL.amber, 1.25));
    // 几粒微尘：停住的机位前 0.5–7 m 的一个盒子里（机位坐标），被光斑从后面照亮。
    const frame = new THREE.Matrix4().makeBasis(RIGHT, UP, FWD.clone().negate()).setPosition(STOP);
    this.dust = new Dust(`${key}/S04`, 260, new THREE.Vector3(-1.2, -0.7, -7), new THREE.Vector3(1.2, 0.7, -0.5), frame);
    this.motes = new BokehPoints(ctx, this.dust.count, { aperture: { shape: 'circle' } });
    this.pulses = endPulses();
    this.objects = [this.spot.mesh, this.motes.mesh, this.title.mesh, this.url.mesh];
  }

  /** 一行字摆在视轴上，左右居中，`y` 是这一行相对视轴的高度（m）。 */
  private layout(lt: LightText, text: string, em: number, y: number, ignite: number, color: RGB): void {
    const chars = Array.from(text);
    const width = chars.reduce((a, ch) => a + lt.advance(ch), 0);
    const centre = STOP.clone().addScaledVector(FWD, TITLE_DIST).addScaledVector(UP, y);
    let pen = -width / 2;
    const placed: CharPlacement[] = chars.map((ch, i) => {
      const adv = lt.advance(ch);
      const x = (pen + adv / 2) * em;
      pen += adv;
      // 从中间往两边依次亮起（相差不到两帧），淡入本身由 update 的 gain 给。
      const ignite_i = ignite + 1.6 * Math.abs(i - (chars.length - 1) / 2) / chars.length;
      return { ch, position: centre.clone().addScaledVector(RIGHT, x), right: RIGHT.clone(), up: UP.clone(), em, ignite: ignite_i, color };
    });
    lt.place(placed);
  }

  /** 光斑的呼吸：每一下 2 帧涨起、两拍里落回。 */
  private breath(f: number): number {
    let k = 0;
    for (const p of this.pulses) {
      const d = f - p.frame;
      if (d < 0 || d > 40) continue;
      k += p.bright * Math.min(easeOutCubic(clamp(d / 2)), Math.exp(-d / 12));
    }
    return k;
  }

  update(f: number, lens: LensState, extinction: number, energy: number): HazeLight[] {
    const lights: HazeLight[] = [];
    const on = f >= CUE.zoom;
    this.spot.mesh.visible = on;
    this.motes.mesh.visible = f >= CUE.spot - 8;
    this.title.mesh.visible = f >= CUE.spot;
    this.url.mesh.visible = f >= CUE.spot;
    if (!on) return lights;
    const fade = endFade(f);
    // 汇聚时一点点攒起来，7.1 落成时满；之后按 pulse 呼吸，并有一点极慢的闪烁。
    const gather = smoothstep(CUE.zoom, CUE.spot, f);
    const land = Math.exp(-Math.max(0, f - CUE.spot) / 6) * smoothstep(CUE.spot - 2, CUE.spot, f);
    const level = (gather * (0.85 + 0.25 * land) + 0.32 * this.breath(f)) * (0.97 + 0.03 * snoise2(f / 24, 5.5)) * fade;
    const c = scaled(PAL.lamp, SPOT_FLUX * level);
    this.spot.set(0, SPOT.x, SPOT.y, SPOT.z, c[0], c[1], c[2], SPOT_EMITTER);
    this.spot.update(1, { lens, extinction, chroma: 0.03, maxRadius: 220 });
    lights.push({ position: SPOT, color: toneOf(PAL.amber, 0.5 * level, DIM_HUE), radius: 0.35 });

    // 片名：淡入，不闪；跟着整体淡出。
    const tIn = smoothstep(CUE.spot + 4, F('7.3'), f);
    const uIn = smoothstep(CUE.spot + 10, F('7.3') + 6, f);
    this.title.update(f, lens, tIn * fade, [0, 0, 0], { hotGain: 0, extinction });
    this.url.update(f, lens, uIn * fade, [0, 0, 0], { hotGain: 0, extinction });
    if (tIn > 0) lights.push({ position: STOP.clone().addScaledVector(FWD, TITLE_DIST + 0.3), color: toneOf(PAL.warmWhite, 0.05 * tIn * fade, DIM_HUE), radius: 0.4 });

    const nd = this.dust.fill({ time: f / 24, camera: STOP, lights, planes: [], gain: 30 * (0.6 + 0.8 * energy) * fade, extinction, maxFlux: 700 }, this.motes, null);
    this.motes.update(nd.back, { lens, extinction, aperture: { shape: 'circle' } });
    return lights;
  }

  dispose(): void {
    this.spot.dispose();
    this.motes.dispose();
    this.title.dispose();
    this.url.dispose();
  }
}
