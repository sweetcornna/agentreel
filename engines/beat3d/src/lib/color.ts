// 给片子用的颜色工具：设计色 hex → 线性 Rec.709、缩放、亮度、「换色调」（同一亮度换一种色度），以及
// 显示变换的反解：sceneFor(hex) 是后期链（AgX + 调色，gl/color.glsl.ts）显示成那个 hex 的场景线性值，
// 片子要在必须准的地方（没被照亮的空间、主体、一行字）正好落在设计色上就用它。是合成着色器
// （post/shaders.ts）颜色数学的 CPU 孪生，不含依赖邻域的效果（辉光、光晕、暗角、颗粒）。

export type RGB = [number, number, number];

/** sRGB hex（"#E0A040"）→ 线性 Rec.709。 */
export function lin(hex: string): RGB {
  const n = parseInt(hex.replace('#', ''), 16);
  const c = (v: number) => {
    const s = v / 255;
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return [c((n >> 16) & 255), c((n >> 8) & 255), c(n & 255)];
}

export const scaled = (c: RGB, k: number): RGB => [c[0] * k, c[1] * k, c[2] * k];
export const luminance = (c: RGB): number => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
/** `c` 的色度（亮度归一到 1）。 */
export const unitLum = (c: RGB): RGB => scaled(c, 1 / Math.max(luminance(c), 1e-12));
/** `c` × `k` 的亮度、换成 `tone` 的色度：例如把暗光推黄一点，免得 AgX 的趾部把它显示成红色。 */
export const toneOf = (c: RGB, k: number, tone: RGB): RGB => scaled(unitLum(tone), luminance(c) * k);

// ---- 显示变换（gl/color.glsl.ts 与合成着色器的孪生）

const INSET = [
  [0.842479062253094, 0.0784335999999992, 0.0792237451477643],
  [0.0423282422610123, 0.878468636469772, 0.0791661274605434],
  [0.0423756549057051, 0.0784336, 0.879142973793104],
];
const OUTSET = [
  [1.19687900512017, -0.0980208811401368, -0.0990297440797205],
  [-0.0528968517574562, 1.15190312990417, -0.0989611768448433],
  [-0.0529716355144438, -0.0980434501171241, 1.15107367264116],
];
const mul = (m: number[][], v: RGB): RGB => [
  m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
  m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
  m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
];
const MIN_EV = -12.47393;
const MAX_EV = 4.026069;
const contrast = (x: number): number => {
  const x2 = x * x;
  const x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
};
const toSrgb = (c: number): number => {
  const x = Math.max(c, 0);
  return x <= 0.0031308 ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055;
};

export interface DisplayGrade {
  exposure?: number;
  slope?: RGB;
  offset?: RGB;
  power?: RGB;
  lookSaturation?: number;
  lift?: RGB;
  gamma?: RGB;
  gain?: RGB;
  saturation?: number;
}

/** 场景线性颜色 → 显示出来的 sRGB（0..1）：曝光、AgX、look（ASC-CDL 与饱和度）、lift/gamma/gain。 */
export function displayOf(c: RGB, g: DisplayGrade = {}): RGB {
  const e = 2 ** (g.exposure ?? 0);
  const anyLight = Math.max(c[0], c[1], c[2]) * e > 1e-4 ? 1 : 0;
  let v = mul(INSET, [Math.max(c[0] * e, 1e-10), Math.max(c[1] * e, 1e-10), Math.max(c[2] * e, 1e-10)]);
  v = v.map((x) => contrast((Math.min(Math.max(Math.log2(x), MIN_EV), MAX_EV) - MIN_EV) / (MAX_EV - MIN_EV))) as RGB;
  const slope = g.slope ?? [1, 1, 1];
  const offset = g.offset ?? [0, 0, 0];
  const power = g.power ?? [1, 1, 1];
  const l0 = luminance(v);
  const looked = v.map((x, i) => Math.max(x * slope[i] + offset[i], 0) ** power[i]) as RGB;
  const ls = g.lookSaturation ?? 1;
  v = looked.map((x) => l0 + ls * (x - l0)) as RGB;
  let d = mul(OUTSET, v).map((x) => Math.max(x, 0) ** 2.2) as RGB;
  const lift = g.lift ?? [0.0006, 0.0008, 0.0014];
  const gamma = g.gamma ?? [1, 1, 1];
  const gain = g.gain ?? [1, 1, 1];
  d = d.map((x, i) => {
    const y = Math.min(Math.max(x, 0), 1);
    return Math.max(gain[i] * (y + lift[i] * anyLight * (1 - y)), 0) ** (1 / Math.max(gamma[i], 1e-3));
  }) as RGB;
  const s = g.saturation ?? 1;
  const ld = luminance(d);
  d = d.map((x) => Math.max(ld + s * (x - ld), 0)) as RGB;
  return d.map(toSrgb) as RGB;
}

/**
 * 显示成 `target`（hex 或 0..1 的 sRGB）的场景线性颜色：对正向变换做牛顿迭代求得。变换够得着的颜色，
 * sRGB 上误差约 1e-4。`g` 要与片子 post 的 grade 一致（例如片子调了 lookSaturation，这里也给同样的值）。
 */
export function sceneFor(target: string | RGB, g: DisplayGrade = {}): RGB {
  const t: RGB = typeof target === 'string' ? (lin(target).map(toSrgb) as RGB) : target;
  // 从目标的线性值起步：AgX 会压暗，所以解比它亮。
  let x: RGB = (typeof target === 'string' ? lin(target) : (t.map((v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4)) as RGB)).map((v) => Math.max(v * 2, 1e-5)) as RGB;
  for (let it = 0; it < 40; it++) {
    // 在对数空间里迭代：变换在那里接近线性，x 也始终为正。
    const y = displayOf(x, g);
    const r = [y[0] - t[0], y[1] - t[1], y[2] - t[2]];
    if (Math.max(...r.map(Math.abs)) < 1e-6) break;
    const J: number[][] = [[], [], []];
    for (let j = 0; j < 3; j++) {
      const h = 1e-4;
      const xp = [...x] as RGB;
      xp[j] *= Math.exp(h);
      const yp = displayOf(xp, g);
      for (let i = 0; i < 3; i++) J[i][j] = (yp[i] - y[i]) / h;
    }
    const step = solve3(J, r);
    for (let j = 0; j < 3; j++) x[j] *= Math.exp(-Math.max(-1, Math.min(1, step[j])));
  }
  return x;
}

function solve3(a: number[][], b: number[]): number[] {
  const det = (m: number[][]) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const d = det(a);
  if (Math.abs(d) < 1e-14) return [0, 0, 0];
  return [0, 1, 2].map((k) => det(a.map((row, i) => row.map((v, j) => (j === k ? b[i] : v)))) / d);
}
