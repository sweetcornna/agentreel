// CPU twins of the GLSL noise in src/gl/noise.glsl.ts. Same algorithms, same constants, same octave
// rotation, so a terrain height or a sky density computed here matches the shader to ~1e-6.
// Simplex noise follows the well-known Gustavson/McEwan formulation (permutation polynomial,
// no lookup tables), which is what makes a bit-faithful CPU twin practical.

const mod289 = (x: number): number => x - Math.floor(x * (1 / 289)) * 289;
const permute = (x: number): number => mod289((x * 34 + 1) * x);
const fract = (x: number): number => x - Math.floor(x);

/** 2D simplex noise in [-1,1]. */
export function snoise2(x: number, y: number): number {
  const C0 = 0.211324865405187;
  const C1 = 0.366025403784439;
  const C2 = -0.577350269189626;
  const C3 = 0.024390243902439;
  const s = (x + y) * C1;
  let ix = Math.floor(x + s);
  let iy = Math.floor(y + s);
  const t = (ix + iy) * C0;
  const x0x = x - ix + t;
  const x0y = y - iy + t;
  const i1x = x0x > x0y ? 1 : 0;
  const i1y = 1 - i1x;
  const x1x = x0x + C0 - i1x;
  const x1y = x0y + C0 - i1y;
  const x2x = x0x + C2;
  const x2y = x0y + C2;
  ix = mod289(ix);
  iy = mod289(iy);
  const p0 = permute(permute(iy) + ix);
  const p1 = permute(permute(iy + i1y) + ix + i1x);
  const p2 = permute(permute(iy + 1) + ix + 1);
  let n = 0;
  const corner = (p: number, cx: number, cy: number): number => {
    let m = Math.max(0.5 - (cx * cx + cy * cy), 0);
    m = m * m;
    m = m * m;
    const gx = 2 * fract(p * C3) - 1;
    const h = Math.abs(gx) - 0.5;
    const a = gx - Math.floor(gx + 0.5);
    m *= 1.79284291400159 - 0.85373472095314 * (a * a + h * h);
    return m * (a * cx + h * cy);
  };
  n += corner(p0, x0x, x0y);
  n += corner(p1, x1x, x1y);
  n += corner(p2, x2x, x2y);
  return 130 * n;
}

/** 3D simplex noise in [-1,1]. */
export function snoise3(x: number, y: number, z: number): number {
  const s = (x + y + z) / 3;
  let i = Math.floor(x + s);
  let j = Math.floor(y + s);
  let k = Math.floor(z + s);
  const t = (i + j + k) / 6;
  const x0 = x - i + t;
  const y0 = y - j + t;
  const z0 = z - k + t;
  const gx = x0 >= y0 ? 1 : 0;
  const gy = y0 >= z0 ? 1 : 0;
  const gz = z0 >= x0 ? 1 : 0;
  const lx = 1 - gx;
  const ly = 1 - gy;
  const lz = 1 - gz;
  const i1x = Math.min(gx, lz);
  const i1y = Math.min(gy, lx);
  const i1z = Math.min(gz, ly);
  const i2x = Math.max(gx, lz);
  const i2y = Math.max(gy, lx);
  const i2z = Math.max(gz, ly);
  i = mod289(i);
  j = mod289(j);
  k = mod289(k);
  const offs: [number, number, number][] = [
    [0, 0, 0],
    [i1x, i1y, i1z],
    [i2x, i2y, i2z],
    [1, 1, 1],
  ];
  const cs: [number, number, number][] = [
    [x0, y0, z0],
    [x0 - i1x + 1 / 6, y0 - i1y + 1 / 6, z0 - i1z + 1 / 6],
    [x0 - i2x + 1 / 3, y0 - i2y + 1 / 3, z0 - i2z + 1 / 3],
    [x0 - 0.5, y0 - 0.5, z0 - 0.5],
  ];
  const nsx = 2 / 7;
  const nsy = 0.5 / 7 - 1;
  const nsz = 1 / 7;
  let sum = 0;
  for (let c = 0; c < 4; c++) {
    const [ox, oy, oz] = offs[c];
    const p = permute(permute(permute(k + oz) + j + oy) + i + ox);
    const jj = p - 49 * Math.floor(p * nsz * nsz);
    const xq = Math.floor(jj * nsz);
    const yq = Math.floor(jj - 7 * xq);
    const gxv = xq * nsx + nsy;
    const gyv = yq * nsx + nsy;
    const h = 1 - Math.abs(gxv) - Math.abs(gyv);
    const sh = h <= 0 ? -1 : 0;
    let px = gxv + (Math.floor(gxv) * 2 + 1) * sh;
    let py = gyv + (Math.floor(gyv) * 2 + 1) * sh;
    let pz = h;
    const norm = 1.79284291400159 - 0.85373472095314 * (px * px + py * py + pz * pz);
    px *= norm;
    py *= norm;
    pz *= norm;
    const [cx, cy, cz] = cs[c];
    let m = Math.max(0.6 - (cx * cx + cy * cy + cz * cz), 0);
    m = m * m;
    sum += m * m * (px * cx + py * cy + pz * cz);
  }
  return 42 * sum;
}

/** fBm over 2D simplex; octave rotation matches GLSL fbm2(). Output roughly in [-1,1]. */
export function fbm2(x: number, y: number, octaves: number): number {
  let f = 0;
  let a = 0.5;
  let px = x;
  let py = y;
  for (let o = 0; o < octaves; o++) {
    f += a * snoise2(px, py);
    const nx = (0.8 * px - 0.6 * py) * 2.03 + 17.1;
    const ny = (0.6 * px + 0.8 * py) * 2.03 + 9.7;
    px = nx;
    py = ny;
    a *= 0.5;
  }
  return f;
}

/** fBm over 3D simplex; matches GLSL fbm3(). */
export function fbm3(x: number, y: number, z: number, octaves: number): number {
  let f = 0;
  let a = 0.5;
  let px = x;
  let py = y;
  let pz = z;
  for (let o = 0; o < octaves; o++) {
    f += a * snoise3(px, py, pz);
    px = px * 2.02 + 31.4;
    py = py * 2.02 + 17.3;
    pz = pz * 2.02 + 7.9;
    a *= 0.5;
  }
  return f;
}

/** Divergence-free 2D flow: curl of a scalar simplex potential. Matches GLSL curl2(). */
export function curl2(x: number, y: number): [number, number] {
  const e = 0.01;
  const dx = (snoise2(x + e, y) - snoise2(x - e, y)) / (2 * e);
  const dy = (snoise2(x, y + e) - snoise2(x, y - e)) / (2 * e);
  return [dy, -dx];
}
