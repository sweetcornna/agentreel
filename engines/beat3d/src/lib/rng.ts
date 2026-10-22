// Deterministic randomness. Math.random is banned in this engine: every random number comes from a
// stream seeded by a string key, normally `${shotId}/${purpose}`. Same key -> same numbers, on every
// machine, in every browser tab, regardless of render order.

/** 32-bit FNV-1a over UTF-16 code units, then avalanche-mixed. */
export function hashString(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return mix32(h);
}

/** Murmur3 finaliser: good avalanche for integer keys. */
export function mix32(x: number): number {
  x |= 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x85ebca6b);
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35);
  x ^= x >>> 16;
  return x >>> 0;
}

/** Stateless hash of up to three integers -> [0,1). Use for per-index randomness without a stream. */
export function hash01(a: number, b = 0, c = 0, seed = 0): number {
  let h = mix32(seed ^ 0x9e3779b9);
  h = mix32(h ^ Math.imul(a | 0, 0x27d4eb2d));
  h = mix32(h ^ Math.imul(b | 0, 0x165667b1));
  h = mix32(h ^ Math.imul(c | 0, 0x61c88647));
  return h / 4294967296;
}

/** sfc32 generator: fast, 128-bit state, passes PractRand far beyond our needs. */
export class Rng {
  private a: number;
  private b: number;
  private c: number;
  private d: number;
  private spareNormal: number | null = null;

  constructor(seed: number | string) {
    const s = typeof seed === 'string' ? hashString(seed) : seed >>> 0;
    this.a = mix32(s ^ 0xa3c59ac3);
    this.b = mix32(s ^ 0x3c6ef372);
    this.c = mix32(s ^ 0x9e3779b9);
    this.d = 1;
    for (let i = 0; i < 12; i++) this.nextU32();
  }

  nextU32(): number {
    const t = (((this.a + this.b) | 0) + this.d) | 0;
    this.d = (this.d + 1) | 0;
    this.a = this.b ^ (this.b >>> 9);
    this.b = (this.c + (this.c << 3)) | 0;
    this.c = (this.c << 21) | (this.c >>> 11);
    this.c = (this.c + t) | 0;
    return t >>> 0;
  }

  /** Uniform in [0,1). */
  next(): number {
    return this.nextU32() / 4294967296;
  }
  range(lo: number, hi: number): number {
    return lo + (hi - lo) * this.next();
  }
  int(lo: number, hiExclusive: number): number {
    return lo + Math.floor(this.next() * (hiExclusive - lo));
  }
  chance(p: number): boolean {
    return this.next() < p;
  }
  /** Standard normal via Box-Muller (cached pair). */
  normal(mean = 0, sd = 1): number {
    if (this.spareNormal !== null) {
      const v = this.spareNormal;
      this.spareNormal = null;
      return mean + sd * v;
    }
    let u = 0;
    while (u <= 1e-12) u = this.next();
    const v = this.next();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spareNormal = r * Math.sin(2 * Math.PI * v);
    return mean + sd * r * Math.cos(2 * Math.PI * v);
  }
  pick<T>(arr: readonly T[]): T {
    return arr[Math.floor(this.next() * arr.length)];
  }
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
  /** Uniform direction on the unit sphere. */
  unitVector(): [number, number, number] {
    const z = this.range(-1, 1);
    const a = this.range(0, Math.PI * 2);
    const r = Math.sqrt(1 - z * z);
    return [r * Math.cos(a), z, r * Math.sin(a)];
  }
  /** Fork an independent child stream (stable: depends only on this stream's seed + label). */
  fork(label: string): Rng {
    return new Rng(mix32(this.nextU32() ^ hashString(label)));
  }
}

/** Canonical constructor: one stream per shot and purpose. */
export const rngFor = (shotId: string, purpose: string): Rng => new Rng(`${shotId}/${purpose}`);

/** Numeric seed for GLSL uniforms (kept < 2^24 so it is exact in float32). */
export const seedFloat = (key: string): number => (hashString(key) & 0xffffff) / 4096;
