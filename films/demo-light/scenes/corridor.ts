// S03「字化为光」之后：一条由光丝组成的走廊。两侧各五根弦，平行于镜头的航向伸向纵深，被字化成的
// 光丝落上去点亮（从近到远），乐谱的 pulse（没有就是八分音符网格）沿两侧的弦往前跑，左右轮流。
// 6.1 起弦的远端弯向纵深里的一点，越收越近，7.1 全部收进那一点：它在 S04 落成一枚光斑（end.ts）。
//
// 弦是 LightLines（每帧在 CPU 上摆每个顶点，帧号的纯函数），脉冲是 Rails（沿直线逐像素求值，
// 所以不受顶点疏密影响）加一颗 BokehPoints 火花作脉冲头。都按 CoC 加宽、能量守恒，穿过雾时消光。

import * as THREE from 'three';
import type { StageContext } from '@beat3d/core/types';
import { BokehPoints } from '@beat3d/world/bokeh';
import type { HazeLight } from '@beat3d/world/haze';
import { LightLines } from '@beat3d/world/lines';
import { MAX_PULSES, Rails, type RailPulse } from '@beat3d/world/rails';
import { clamp, easeInOutCubic, smoothstep } from '@beat3d/lib/easing';
import type { LensState } from '@beat3d/lib/lens';
import { snoise2 } from '@beat3d/lib/noise';
import { scaled, toneOf, type RGB } from '@beat3d/lib/color';
import { blurSamples, CUE, railPulses, SHUTTER, type Pulse } from './cues';
import { DIM_HUE, PAL } from './palette';
import { STOP, STOP_PITCH } from './rig';

/** 两侧弦的位置（x, y，m）：左右对称，两层。 */
const WALL: [number, number][] = [
  [1.45, -0.62],
  [1.62, -0.05],
  [1.45, 0.52],
  [1.95, -0.34],
  [1.95, 0.26],
];
export const STRINGS: { x: number; y: number; side: -1 | 1 }[] = [
  ...WALL.map(([x, y]) => ({ x: -x, y, side: -1 as const })),
  ...WALL.map(([x, y]) => ({ x, y, side: 1 as const })),
];
/** 弦从字面后一点起，伸到 150 m 外。 */
export const Z_NEAR = 0.6;
const LENGTH = 150;
const VERTS = 140;
/** 顶点沿弦的分布：近处密、远处疏（z = Z_NEAR − LENGTH · s^1.7）。 */
const zOf = (s: number): number => Z_NEAR - LENGTH * Math.pow(s, 1.7);

/** 汇聚点：停住的机位正前方 9 m，经 7.1 的俯角落在画面上方。 */
export const SPOT = (() => {
  const d = 9;
  return new THREE.Vector3(STOP.x, STOP.y - Math.sin(STOP_PITCH) * d, STOP.z - Math.cos(STOP_PITCH) * d);
})();

/** 字化成的光丝在 5.1 前几帧开始落到弦上；点亮的前沿从字面往前跑。 */
const REVEAL_START = CUE.pass - 6;
const REVEAL_SPEED = 2.2;

/** 脉冲：头从镜头前 1 m 起往前跑（m/帧），拖一条尾巴。 */
const PULSE_LEAD = 2.0;
const PULSE_SPEED = 1.5;
const PULSE_LEN = 3.2;
const RAIL_LEN = 70;
/** 一颗火花在一个子帧的快门里摆几颗。 */
const SMEAR = 6;

export class Corridor {
  readonly lines: LightLines;
  readonly rails: Rails;
  readonly sparks: BokehPoints;
  private readonly pulses: (Pulse & { string: number; start: number })[];

  constructor(ctx: StageContext, private readonly camZ: (f: number) => number) {
    this.lines = new LightLines(ctx, STRINGS.length, VERTS);
    this.rails = new Rails(
      ctx,
      STRINGS.map((s) => ({ a: new THREE.Vector3(s.x, s.y, Z_NEAR), b: new THREE.Vector3(s.x, s.y, Z_NEAR - RAIL_LEN), radius: 0.01 })),
    );
    // 脉冲左右轮流；同一侧依次换一根弦。
    let left = 0;
    let right = 0;
    this.pulses = railPulses().map((p) => {
      const side = p.pan ? (p.pan < 0 ? -1 : 1) : p.index % 2 ? 1 : -1;
      const k = side < 0 ? left++ : right++;
      const string = (side < 0 ? 0 : WALL.length) + ((k * 2) % WALL.length);
      return { ...p, string, start: camZ(p.frame) - PULSE_LEAD };
    });
    this.sparks = new BokehPoints(ctx, this.pulses.length * SMEAR);
  }

  /**
   * 弦上某点在帧 f 的位置。汇聚（c 从 0 到 1）时每根弦从直线变成一条收向汇聚点的曲线：镜头旁边的
   * 一段还平行着，越往前越收，到汇聚点的深度正好落进那一点；比汇聚点更远的部分沿视轴缩进它。
   */
  private bend(s: { x: number; y: number }, z: number, f: number, c: number, out: THREE.Vector3): THREE.Vector3 {
    out.set(s.x, s.y, z);
    if (c <= 0) return out;
    const camZ = this.camZ(f);
    const u = (camZ - z) / Math.max(1, camZ - SPOT.z);
    if (u <= 0) return out;
    const g = easeInOutCubic(clamp(u));
    const fz = u >= 1 ? SPOT.z : z;
    return out.set(s.x + (SPOT.x - s.x) * g * c, s.y + (SPOT.y - s.y) * g * c, z + (fz - z) * c);
  }

  /** 走廊在帧 f 的亮度、脉冲与汇聚；返回给雾的光源（最近的几颗脉冲头、走廊尽头的远光）。 */
  update(f: number, lens: LensState, extinction: number, energy: number): HazeLight[] {
    const lights: HazeLight[] = [];
    const t = f / 24;
    const c = easeInOutCubic(clamp((f - CUE.converge) / (CUE.spot - CUE.converge)));
    const front = f < REVEAL_START ? Infinity : Z_NEAR - 1 - REVEAL_SPEED * (f - REVEAL_START);
    const on = f >= REVEAL_START - 0.5 && f < CUE.spot + 2;
    // 收进汇聚点时弦变暗（能量交给光斑），最后两帧完全消失。
    const fade = (1 - c * c * c) * (1 - smoothstep(CUE.spot - 2, CUE.spot + 1, f));
    const base = 0.32 * (0.7 + 0.6 * energy) * fade;

    // 脉冲：头和尾巴，按弦分组，也用来让弦在经过处轻轻颤动、变亮。
    const railPulses: RailPulse[] = [];
    const camZ = this.camZ(f);
    const heads: { string: number; z: number; frame: number; k: number }[] = [];
    this.pulses.forEach((p) => {
      if (f < p.frame) return;
      const h = p.start - PULSE_SPEED * (f - p.frame);
      const run = Z_NEAR - h;
      const k = (0.6 + 1.2 * p.bright) * smoothstep(p.frame, p.frame + 1.5, f) * (1 - smoothstep(RAIL_LEN - 25, RAIL_LEN - 5, run)) * (1 - c * c);
      if (k <= 0.002) return;
      // 脉冲的尾巴（Rails）是直线：弦一弯就让它退掉，只留沿弦走的火花和弦上的余亮。
      // 尾巴不伸到镜头后面去（投影过镜头平面会翻折成一个 V）。
      const tail = Math.min(PULSE_LEN, camZ - 0.4 - h);
      if (tail > 0.05) railPulses.push({ rail: p.string, head: run, length: tail, color: scaled(PAL.amber, 11 * k * (1 - c) * (1 - c)) });
      heads.push({ string: p.string, z: h, frame: p.frame, k });
    });

    const v = new THREE.Vector3();
    STRINGS.forEach((s, l) => {
      if (!on) {
        this.lines.clear(l);
        return;
      }
      for (let i = 0; i < VERTS; i++) {
        const z = zOf(i / (VERTS - 1));
        // 点亮的前沿：从字面往前跑，前沿柔和 6 m。
        const lit = front === Infinity ? 0 : smoothstep(front - 6, front, z);
        // 被脉冲经过的地方亮一点并颤动。
        let hum = 0;
        for (const p of this.pulses) {
          if (p.string !== l || f < p.frame) continue;
          const h = p.start - PULSE_SPEED * (f - p.frame);
          const d = z - h;
          if (d > 0 && d < 9) hum += Math.exp(-d / 3) * (0.6 + p.bright);
        }
        this.bend(s, z, f, c, v);
        v.y += 0.012 * hum * Math.sin(z * 2.3 - t * 40 + l) * (1 - c);
        // 远处略带深琥珀，近处是琥珀；颜色 × 强度（1 px 线的单位）。
        const far = smoothstep(10, 80, -z);
        const k = base * lit * (1 + 0.7 * hum) * (0.92 + 0.08 * snoise2(z * 0.07, l * 3.1));
        const col: RGB = [
          (PAL.amber[0] + (PAL.deep[0] - PAL.amber[0]) * far) * k,
          (PAL.amber[1] + (PAL.deep[1] - PAL.amber[1]) * far) * k,
          (PAL.amber[2] + (PAL.deep[2] - PAL.amber[2]) * far) * k,
        ];
        this.lines.set(l, i, v, col[0], col[1], col[2]);
      }
    });
    this.lines.update({ lens, extinction, width: 0.6 });
    // Rails 一次最多 MAX_PULSES 个脉冲：留最近发出的那些（老的已经跑远、快灭了）。
    this.rails.update({ lens, pulses: railPulses.slice(-MAX_PULSES), extinction });
    // 火花：一个子帧的快门里它要跑一米多，画成一颗点会在运动模糊里断成一串点。所以在这个子帧的快门
    // 区间里摆 SMEAR 颗（每颗 1/SMEAR 的能量），沿弦（含汇聚时的弯）连成一道。
    const dt = SHUTTER / blurSamples(f);
    let n = 0;
    const q = new THREE.Vector3();
    for (const h of heads) {
      for (let j = 0; j < SMEAR; j++) {
        const tf = f + ((j + 0.5) / SMEAR - 0.5) * dt;
        const z = h.z - PULSE_SPEED * (tf - f);
        this.bend(STRINGS[h.string], z, tf, c, q);
        this.sparks.set(n++, q.x, q.y, q.z, ...scaled(PAL.amber, (2600 * h.k) / SMEAR), 0.005);
      }
    }
    this.sparks.update(n, { lens, extinction });
    const headPos = heads.map((h) => ({ p: this.bend(STRINGS[h.string], h.z, f, c, new THREE.Vector3()), k: h.k }));

    // 雾：走廊尽头一盏很远的暖光（逆光，给弦勾出纵深），汇聚时它就是那一点。
    if (on) {
      const glow = (0.35 + 0.65 * smoothstep(REVEAL_START, REVEAL_START + 30, f)) * (1 - 0.6 * c) * fade;
      lights.push({ position: new THREE.Vector3(0, 0.1, camZ - 55).lerp(SPOT, c), color: toneOf(PAL.deep, 1.6 * glow, DIM_HUE), radius: 3.5 });
      for (const h of headPos.filter((q) => q.p.z < camZ - 0.4).sort((a, b) => b.p.z - a.p.z).slice(0, 3)) lights.push({ position: h.p, color: toneOf(PAL.amber, 0.16 * h.k, DIM_HUE), radius: 0.12 });
    }
    return lights;
  }

  dispose(): void {
    this.lines.dispose();
    this.rails.dispose();
    this.sparks.dispose();
  }
}
