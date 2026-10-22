// 一镜到底的摄影机，(小数)帧号的纯函数。
//
// 世界坐标：镜头朝 −Z 看，字面在 z = 0。S01 在 9 m 外的雾里极慢地推进；S02 继续推，4.4 起加速，
// 5.1 正好穿过两行字之间；S03 进光丝走廊后减速；7.1 停住，同时 35 → 50 mm 变焦、微微低头，
// 让汇聚点落在画面上方，S04 固定机位。速度在每个关键帧上给定（Path3），所以没有走走停停。

import * as THREE from 'three';
import type { CameraPose } from '@beat3d/world/stage';
import { Path1, Path3 } from '@beat3d/lib/path';
import { clamp, easeInOutCubic, smoothstep } from '@beat3d/lib/easing';
import { snoise2 } from '@beat3d/lib/noise';
import { CUE } from './cues';

/** 字面上，镜头穿过的那一点：两行字之间。 */
export const THROUGH = new THREE.Vector3(0, -0.075, 0);

/** 镜头位置（m），速度单位 m/帧。 */
const POS = new Path3([
  { pos: '1.1', p: [0.05, 0.012, 9.2], v: [-0.0002, 0, -0.0085] },
  { pos: '3.1', p: [0.02, -0.01, 8.05], v: [-0.0002, -0.0002, -0.022] },
  { pos: '4.4', p: [0.0, -0.06, 4.6], v: [0, -0.0002, -0.03] },
  { pos: '5.1', p: [THROUGH.x, THROUGH.y, 0], v: [0, 0, -0.545] },
  { pos: '5.3', p: [0, -0.06, -11.3], v: [0, 0.0004, -0.16] },
  { pos: '6.1', p: [0, -0.03, -15.8], v: [0, 0.0004, -0.12] },
  { pos: '7.1', p: [0, 0, -19.6], v: [0, 0, 0] },
  { pos: '9.1', p: [0, 0, -19.6], v: [0, 0, 0] },
]);

/** 停住的机位（S04）。 */
export const STOP = POS.at(CUE.spot);

/** 7.1 停住时的俯角（rad）：汇聚点落在画面上方三分之一附近。 */
export const STOP_PITCH = -2.4 * (Math.PI / 180);

/** S04 的焦段与光圈。 */
export const END_LENS = { focalMm: 50, fStop: 1.3 };

/** 焦距：S01–S02 85 mm；5.1 那一帧起 35 mm；6.3 起推到 50 mm，7.1 到位。 */
function focalAt(f: number): number {
  // 硬切的量按整数帧决定：同一帧的运动模糊子帧不会一半 85 一半 35。
  if (Math.round(f) < CUE.pass) return 85;
  return 35 + 15 * easeInOutCubic(clamp((f - CUE.zoom) / (CUE.spot - CUE.zoom)));
}

/** 光圈：S01 f/1.2 全开；S02 f/1.4；穿过字面后 f/2；S04 f/1.3。 */
const FSTOP = new Path1([
  { pos: '1.1', value: 1.2, slope: 0 },
  { pos: '3.1', value: 1.2, slope: 0 },
  { pos: '3.2', value: 1.4, slope: 0 },
  { pos: '6.4', value: 2.0, slope: 0 },
  { pos: '7.1', value: END_LENS.fStop, slope: 0 },
]);

export interface Shot extends CameraPose {
  /** 镜头到字面的距离（m，穿过之后为负） */
  toText: number;
}

const tmpE = new THREE.Euler(0, 0, 0, 'YXZ');

/** 帧 f 的镜头：位置、朝向、焦段、光圈、对焦距离。 */
export function shotAt(f: number, focusOverride?: number): Shot {
  const position = POS.at(f);
  // 雾里的微漂：手持感很弱的慢噪声；S04 固定机位，漂移在 7.1 前收掉。
  const t = f / 24;
  const drift = 1 - smoothstep(CUE.converge, CUE.spot, f);
  const yaw = (0.12 * snoise2(t * 0.21, 3.7) + 0.05 * snoise2(t * 0.6, 9.1)) * (Math.PI / 180) * drift;
  const pitch = (0.08 * snoise2(t * 0.19, 13.3) + 0.04 * snoise2(t * 0.55, 1.9)) * (Math.PI / 180) * drift + STOP_PITCH * easeInOutCubic(clamp((f - CUE.converge) / (CUE.spot - CUE.converge)));
  const roll = 0.15 * snoise2(t * 0.12, 21.5) * (Math.PI / 180) * drift;
  tmpE.set(pitch, yaw, roll, 'YXZ');
  const quaternion = new THREE.Quaternion().setFromEuler(tmpE);
  const toText = position.z;
  const focalMm = focalAt(f);
  return { position, quaternion, lens: { focalMm, fStop: FSTOP.at(f), focus: focusOverride ?? focusAt(f, toText) }, toText };
}

/** 对焦：S01 对在 60 m 外（前景光点全是光斑）；3.1 起 10 帧拉到字面并跟着它；穿过后 5 m；S04 由片尾程序给。 */
function focusAt(f: number, toText: number): number {
  const far = 60;
  if (f < CUE.word) return far;
  if (Math.round(f) < CUE.pass) {
    const u = easeInOutCubic(clamp((f - CUE.word) / CUE.rack));
    const near = Math.max(toText, 0.4);
    // 按屈光度插值，焦点移动在画面上才均匀。
    return 1 / (1 / far + (1 / near - 1 / far) * u);
  }
  return 5;
}
