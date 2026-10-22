// 这部片的全部时刻，都从时间线的拍位换算（CLOCK），scenes/ 里没有写死的帧号。
// 乐谱的 pulse 事件（out/music/events.json，构建时烘进包）驱动 S01 的光点明灭和 S03 弦上的脉冲；
// 没有乐谱时走八分音符网格，画面照样成立，只是不再跟着具体的音。

import { smoothstep } from '@beat3d/lib/easing';
import { CLOCK, shotById } from '@beat3d/lib/time';

const F = (p: string): number => CLOCK.frameAt(p);

export const BEAT = CLOCK.framesPerBeat;

export const CUE = {
  /** 1.3 之前纯黑：第一粒光 */
  firstLight: F('1.3'),
  /** P1：光点明灭的区间（含 2.4） */
  p1: [F('1.3'), F('2.4')] as [number, number],
  /** 光圈从叶形打开成圆形（虹膜张开）的起止 */
  irisOpen: [F('2.1'), F('2.1+10f')] as [number, number],
  /** 光点开始向画面中心聚拢 */
  gather: F('2.4'),
  /** H1：光聚成字（硬同步） */
  word: F('3.1'),
  /** 拉焦到字上用的帧数 */
  rack: 10,
  /** 副标题晚一拍点亮 */
  subtitle: F('3.2'),
  /** 推进转成加速 */
  accel: F('4.4'),
  /** H2：镜头穿过字面，画幅打开（硬同步） */
  pass: F('5.1'),
  /** 光丝开始向纵深汇聚 */
  converge: F('6.1'),
  /** 35 → 50 mm 的变焦从这里开始，7.1 停稳 */
  zoom: F('6.3'),
  /** H3：汇聚点落成光斑，片名淡入（软同步） */
  spot: F('7.1'),
  /** 片尾淡出 */
  fadeOut: F('8.4'),
  /** 9.1 一刀切黑（S04 的终点，之后引擎画纯黑） */
  cut: shotById('S04').toFrame,
};

export interface Pulse {
  frame: number;
  /** 0..1：乐谱里这一下有多亮（力度） */
  bright: number;
  /** -1 左 … 1 右（事件的 pan，或 side 的 L / C / R）；没有就是 null */
  pan: number | null;
  /** S01 的事件带 light：这一下的音高在 P1 里从低到高排第几（0..5），没有就是 null */
  light: number | null;
  /** 第几个（同一区间内的序号） */
  index: number;
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const SIDE: Record<string, number> = { L: -1, C: 0, R: 1 };

/** 一段网格：从 from 起每 step 帧一下，到 to 为止（不含），力度按 ramp 渐强。 */
export interface GridRun {
  from: number;
  to: number;
  step: number;
}

/**
 * 镜头 `shot` 在 [from, to) 里乐谱的 pulse 事件（事件带 shot 字段时按它分，S03 的脉冲不会跑去驱动
 * S01 的光点）；没有乐谱时按 `grid`（默认整段八分音符）出网格，力度按 `ramp` 渐强。
 * 力度取事件的 bright，否则 vel，否则 0.7。
 */
export function pulsesIn(shot: string, from: number, to: number, grid: GridRun[] = [{ from, to, step: BEAT / 2 }], ramp: [number, number] = [0.55, 0.85]): Pulse[] {
  const ev = CLOCK.eventFrames('pulse', from - 0.5, to - 0.5).filter((e) => typeof e.data.shot !== 'string' || e.data.shot === shot);
  if (ev.length)
    return ev.map((e, index) => ({
      // events.json 的时间精确到微秒：帧号取到千分之一，落在整帧上的就是整帧。
      frame: Math.round(e.frame * 1000) / 1000,
      bright: Math.min(1, Math.max(0.15, num(e.data.bright) ?? num(e.data.vel) ?? 0.7)),
      pan: num(e.data.pan) ?? (typeof e.data.side === 'string' ? (SIDE[e.data.side] ?? null) : null),
      light: num(e.data.light),
      index,
    }));
  const frames: number[] = [];
  for (const g of grid) for (let f = g.from; f < g.to - 1e-6; f += g.step) frames.push(f);
  const n = frames.length;
  return frames.map((frame, i) => ({ frame, bright: ramp[0] + ((ramp[1] - ramp[0]) * i) / Math.max(1, n - 1), pan: null, light: null, index: i }));
}

/** P1（S01）：1.3–2.4 的八分音符，含 2.4 后半拍那一下。 */
export const p1Pulses = (): Pulse[] => pulsesIn('S01', CUE.p1[0], CUE.word, [{ from: CUE.p1[0], to: CUE.p1[1] + BEAT, step: BEAT / 2 }]);

/** S03 弦上的脉冲：5.1–7.1。网格回退：第 5 小节十六分音符（从 5.1 后的第一个十六分起），第 6 小节八分。 */
export const railPulses = (): Pulse[] =>
  pulsesIn('S03', CUE.pass, CUE.spot, [
    { from: CUE.pass + BEAT / 4, to: CUE.converge, step: BEAT / 4 },
    { from: CUE.converge, to: CUE.spot, step: BEAT / 2 },
  ], [0.45, 0.95]);

/** S04 光斑的呼吸：7.1–9.1。网格回退：第 7 小节四分，第 8 小节前半的两个二分。 */
export const endPulses = (): Pulse[] =>
  pulsesIn('S04', CUE.spot, CUE.cut, [
    { from: CUE.spot, to: F('8.1'), step: BEAT },
    { from: F('8.1'), to: F('8.4'), step: 2 * BEAT },
  ], [0.35, 0.22]);

/**
 * S02 的上行（4.4 起四个十六分音符，乐谱里是 run 事件）：推进加速时字一下一下更亮。没有乐谱就是
 * 4.4 起的四个十六分。返回帧 f 的增亮量（0..1+）。
 */
export function runLift(f: number): number {
  const ev = CLOCK.eventFrames('run', CUE.accel - 0.5, CUE.pass - 0.5);
  const hits = ev.length
    ? ev.map((e, i) => ({ frame: Math.round(e.frame * 1000) / 1000, k: num(e.data.vel) ?? 0.36 + 0.06 * i }))
    : [0, 1, 2, 3].map((i) => ({ frame: CUE.accel + (i * BEAT) / 4, k: 0.36 + 0.06 * i }));
  let lift = 0;
  for (const h of hits) {
    const d = f - h.frame;
    if (d >= 0 && d < 16) lift += h.k * Math.min(1, d / 1.5) * Math.exp(-d / 3.5);
  }
  return lift;
}

/** 两次硬同步的闪：H1 字落成（曝光档数：+1.1 两帧，12 帧内退掉），H2 穿过字面（0..1，穿过那一刻最强）。 */
export function flashes(f: number): { word: number; pass: number } {
  const d = f - CUE.word;
  const word = d >= 0 && d < 14 ? (d < 2 ? 1.1 : 1.1 * (1 - smoothstep(2, 14, d))) : 0;
  const p = f - CUE.pass;
  const pass = p >= -2 && p < 12 ? Math.exp(-((p - 0.5) ** 2) / 5) : 0;
  return { word, pass };
}

/** 快门（帧间隔的比例，0.5 = 180°）。 */
export const SHUTTER = 0.5;

/** 运动模糊的子帧数：推得慢的地方少，光点聚拢与 4.4 加速到穿过字面多。按整数帧取（子帧里不跳）。 */
export function blurSamples(frame: number): number {
  const f = Math.round(frame);
  if (f < CUE.gather) return 2;
  // 2.4 起光点飞向字心，跑得快：子帧少了每粒光会变成两三个叠着的圆。
  if (f < CUE.word + 4) return 8;
  if (f < CUE.accel) return 3;
  if (f < CUE.pass + 8) return 12;
  if (f < CUE.converge) return 8;
  if (f < CUE.spot) return 5;
  return 2;
}
