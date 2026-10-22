// 片子镜头的光圈：一切失焦的东西都取它的形状。点光斑（world/bokeh.ts）、层的虚化（post/LensBlur.ts）
// 和预虚化的字图（text/lighttext.ts）都通过同一个 ApertureSpec 画，CPU 与 GLSL（gl/aperture.glsl.ts）
// 各一份孪生实现，所以失焦的灯、被照亮的面和一行字读起来是同一支镜头。
//
// 形状（片子选一种，不同镜头也可以各用各的）：
//   circle   全开的圆形虹膜
//   leaf     四个角各自给圆角半径的圆角方形；默认左上、右下大圆角，另一条对角线小圆角，就是叶形
//   polygon  `sides` 片直叶片的虹膜，角可以再倒圆
// `open`（0..1）把任何形状往圆形渐变（虹膜张开），`rotation` 转动它。
// 每种形状都缩放到与单位圆同面积（不管什么形状能量守恒）；形状内的亮度分布（大光圈镜头带一点球差：
// 平静的中间、一圈细亮的边、很淡的洋葱环）在形状上归一到均值 1。两种归一都在这里的固定网格上
// 数值测量（确定性），作为 uniform 交给着色器。坐标 y 朝上（与 GL 一致），单位是弥散圆半径。

import { sdRoundRect, type CornerRadii } from './leaf';

export type ApertureSpec =
  | { shape: 'circle'; open?: number }
  | { shape: 'leaf'; radii?: CornerRadii; rotation?: number; open?: number }
  | { shape: 'polygon'; sides: number; rotation?: number; roundness?: number; open?: number };

/** 默认叶形的四角圆角（半边长的比例）：左上、右下大。 */
export const LEAF_RADII: CornerRadii = { tl: 1, tr: 1 / 3, br: 1, bl: 1 / 3 };

export const CIRCLE: ApertureSpec = { shape: 'circle' };
export const LEAF: ApertureSpec = { shape: 'leaf' };
export const HEXAGON: ApertureSpec = { shape: 'polygon', sides: 6, rotation: Math.PI / 6, roundness: 0.08 };

/** 着色器拿到的东西（gl/aperture.glsl.ts）：三个 vec4 uniform。 */
export interface ApertureGL {
  /** 种类（0 圆、1 叶形、2 多边形）、边数、旋转（rad）、张开（0..1）。 */
  shape: [number, number, number, number];
  /** 叶形：圆角按（右上、右下、左上、左下）。多边形：x 为倒角。 */
  radii: [number, number, number, number];
  /** k（形状 → 单位面积）、k2（渐变之后 → 单位面积）、亮度分布的归一、外延（最大半径）。 */
  norm: [number, number, number, number];
}

const KIND = { circle: 0, leaf: 1, polygon: 2 } as const;

function sdPolygon(x: number, y: number, n: number, round: number): number {
  // 外接圆半径为 1 的正多边形，一个顶点在 +y，角按边心距的 `round` 倒圆。
  const an = Math.PI / n;
  const c = round * Math.cos(an);
  const r = 1 - c / Math.cos(an);
  const bn = mod(Math.atan2(x, y), 2 * an) - an;
  const len = Math.hypot(x, y);
  let px = len * Math.cos(bn);
  let py = Math.abs(len * Math.sin(bn));
  px -= r * Math.cos(an);
  py -= r * Math.sin(an);
  py += Math.min(Math.max(-py, 0), r * Math.sin(an));
  return Math.hypot(px, py) * Math.sign(px) - c;
}
const mod = (a: number, b: number): number => a - b * Math.floor(a / b);

/** 原始形状（未归一）的有符号距离，y 朝上。 */
function sdRaw(spec: ApertureSpec, x: number, y: number): number {
  if (spec.shape === 'circle') return Math.hypot(x, y) - 1;
  if (spec.shape === 'leaf') return sdRoundRect(x, y, 1, 1, spec.radii ?? LEAF_RADII);
  return sdPolygon(x, y, Math.max(3, Math.round(spec.sides)), spec.roundness ?? 0);
}

/** 归一之前的亮度分布：离边缘往里 `d`（弥散圆半径）处。 */
export function rawProfile(d: number): number {
  const rim = Math.exp(-d / 0.1);
  const u = Math.min(Math.max(d / 0.5, 0), 1);
  return 0.72 + 0.1 * u + 0.75 * rim + 0.05 * Math.cos(d * 18) * u;
}

const GRID = 320;
const BOUND = 2.0;

/** 固定网格上一个 sd 函数按覆盖率加权的面积、亮度均值与外延。 */
function measure(sd: (x: number, y: number) => number, profile?: (d: number) => number): { area: number; prof: number; extent: number } {
  const cell = (2 * BOUND) / GRID;
  let area = 0;
  let prof = 0;
  let extent = 0;
  for (let j = 0; j < GRID; j++) {
    const y = -BOUND + (j + 0.5) * cell;
    for (let i = 0; i < GRID; i++) {
      const x = -BOUND + (i + 0.5) * cell;
      const s = sd(x, y);
      const cov = Math.min(Math.max(0.5 - s / cell, 0), 1);
      if (cov <= 0) continue;
      area += cov;
      if (profile) prof += cov * profile(Math.max(-s, 0));
      extent = Math.max(extent, Math.hypot(x, y) + 0.71 * cell);
    }
  }
  return { area: area * cell * cell, prof: profile ? prof / area : 0, extent };
}

const cache = new Map<string, ApertureGL>();

/** 编译给着色器用的光圈（按值缓存）。 */
export function compileAperture(spec: ApertureSpec): ApertureGL {
  const key = JSON.stringify(spec);
  const hit = cache.get(key);
  if (hit) return hit;
  const rot = 'rotation' in spec ? (spec.rotation ?? 0) : 0;
  const open = Math.min(Math.max(spec.open ?? 0, 0), 1);
  const k = Math.sqrt(Math.PI / measure((x, y) => sdRaw(spec, x, y)).area);
  const unit = (x: number, y: number): number => {
    const c = Math.cos(-rot);
    const s = Math.sin(-rot);
    const qx = c * x - s * y;
    const qy = s * x + c * y;
    const a = sdRaw(spec, qx / k, qy / k) * k;
    return open > 0 ? a + (Math.hypot(qx, qy) - 1 - a) * open : a;
  };
  const k2 = open > 0 ? Math.sqrt(Math.PI / measure(unit).area) : 1;
  const sd = (x: number, y: number): number => unit(x / k2, y / k2) * k2;
  const m = measure(sd, rawProfile);
  const r = spec.shape === 'leaf' ? (spec.radii ?? LEAF_RADII) : null;
  const gl: ApertureGL = {
    shape: [KIND[spec.shape], spec.shape === 'polygon' ? Math.max(3, Math.round(spec.sides)) : 0, rot, open],
    radii: r ? [r.tr, r.br, r.tl, r.bl] : [spec.shape === 'polygon' ? (spec.roundness ?? 0) : 0, 0, 0, 0],
    norm: [k, k2, 1 / m.prof, m.extent],
  };
  cache.set(key, gl);
  return gl;
}

/** 到单位面积光圈的有符号距离（弥散圆半径为单位，里面为负）。GLSL apertureSd() 的 CPU 孪生。 */
export function apertureSd(spec: ApertureSpec, x: number, y: number): number {
  const g = compileAperture(spec);
  const [k, k2] = g.norm;
  const rot = g.shape[2];
  const open = g.shape[3];
  const px = x / k2;
  const py = y / k2;
  const c = Math.cos(-rot);
  const s = Math.sin(-rot);
  const qx = c * px - s * py;
  const qy = s * px + c * py;
  const a = sdRaw(spec, qx / k, qy / k) * k;
  return (open > 0 ? a + (Math.hypot(qx, qy) - 1 - a) * open : a) * k2;
}

/** 离边缘往里 `d` 处的亮度（在形状上均值为 1）。GLSL apertureProfile() 的孪生。 */
export function apertureProfile(spec: ApertureSpec, d: number): number {
  return compileAperture(spec).norm[2] * rawProfile(d);
}

/** 光圈的最大半径（弥散圆半径为单位）：光斑的四边形要画多大。 */
export const apertureExtent = (spec: ApertureSpec): number => compileAperture(spec).norm[3];

let filmDefault: ApertureSpec = CIRCLE;
/** 片子的光圈（FilmDef.aperture，页面在创建任何程序之前设好）。 */
export const defaultAperture = (): ApertureSpec => filmDefault;
export function setDefaultAperture(spec: ApertureSpec | undefined): void {
  filmDefault = spec ?? CIRCLE;
}
