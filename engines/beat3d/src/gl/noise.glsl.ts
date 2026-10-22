// GLSL noise library (GLSL ES 3.00; three.js ShaderMaterial on WebGL2).
// snoise(vec2/vec3): simplex noise after Gustavson & McEwan (permutation polynomial form).
// fbm2/fbm3/curl2/curl3, integer PCG hashes. CPU twins live in src/lib/noise.ts; keep them in sync.

export const NOISE_GLSL = /* glsl */ `
#ifndef B3D_NOISE
#define B3D_NOISE
vec2 b3d_mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 b3d_mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 b3d_mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 b3d_permute(vec3 x) { return b3d_mod289(((x * 34.0) + 1.0) * x); }
vec4 b3d_permute(vec4 x) { return b3d_mod289(((x * 34.0) + 1.0) * x); }

float snoise(vec2 v) {
  const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
  vec2 i = floor(v + dot(v, C.yy));
  vec2 x0 = v - i + dot(i, C.xx);
  vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = b3d_mod289(i);
  vec3 p = b3d_permute(b3d_permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
  vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
  m = m * m;
  m = m * m;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
  vec3 g;
  g.x = a0.x * x0.x + h.x * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

float snoise(vec3 v) {
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;
  i = b3d_mod289(i);
  vec4 p = b3d_permute(b3d_permute(b3d_permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = 1.79284291400159 - 0.85373472095314 * vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

float fbm2(vec2 p, int octaves) {
  float f = 0.0;
  float a = 0.5;
  for (int o = 0; o < 12; o++) {
    if (o >= octaves) break;
    f += a * snoise(p);
    p = vec2(0.8 * p.x - 0.6 * p.y, 0.6 * p.x + 0.8 * p.y) * 2.03 + vec2(17.1, 9.7);
    a *= 0.5;
  }
  return f;
}

float fbm3(vec3 p, int octaves) {
  float f = 0.0;
  float a = 0.5;
  for (int o = 0; o < 12; o++) {
    if (o >= octaves) break;
    f += a * snoise(p);
    p = p * 2.02 + vec3(31.4, 17.3, 7.9);
    a *= 0.5;
  }
  return f;
}

/** Ridged fbm (filaments, dust lanes, ridgelines). 0..~1. */
float ridged3(vec3 p, int octaves) {
  float f = 0.0;
  float a = 0.5;
  for (int o = 0; o < 12; o++) {
    if (o >= octaves) break;
    float n = 1.0 - abs(snoise(p));
    f += a * n * n;
    p = p * 2.02 + vec3(31.4, 17.3, 7.9);
    a *= 0.5;
  }
  return f;
}

/** Divergence-free 2D flow field. */
vec2 curl2(vec2 p) {
  const float e = 0.01;
  float dx = (snoise(p + vec2(e, 0.0)) - snoise(p - vec2(e, 0.0))) / (2.0 * e);
  float dy = (snoise(p + vec2(0.0, e)) - snoise(p - vec2(0.0, e))) / (2.0 * e);
  return vec2(dy, -dx);
}

/** Divergence-free 3D flow field (Bridson): curl of three offset simplex potentials. */
vec3 curl3(vec3 p) {
  const float e = 0.01;
  vec3 dx = vec3(e, 0.0, 0.0);
  vec3 dy = vec3(0.0, e, 0.0);
  vec3 dz = vec3(0.0, 0.0, e);
  vec3 o1 = vec3(0.0);
  vec3 o2 = vec3(123.4, 0.0, 0.0);
  vec3 o3 = vec3(0.0, 0.0, 271.8);
  float p1_y = snoise(p + o1 + dy) - snoise(p + o1 - dy);
  float p1_z = snoise(p + o1 + dz) - snoise(p + o1 - dz);
  float p2_x = snoise(p + o2 + dx) - snoise(p + o2 - dx);
  float p2_z = snoise(p + o2 + dz) - snoise(p + o2 - dz);
  float p3_x = snoise(p + o3 + dx) - snoise(p + o3 - dx);
  float p3_y = snoise(p + o3 + dy) - snoise(p + o3 - dy);
  return vec3(p3_y - p2_z, p1_z - p3_x, p2_x - p1_y) / (2.0 * e);
}

// Integer hashes (exact, platform-independent). pcg3d after Jarzynski & Olano 2020.
uvec3 pcg3d(uvec3 v) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v ^= v >> 16u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}
vec3 hash33i(ivec3 p) { return vec3(pcg3d(uvec3(p))) * (1.0 / 4294967296.0); }
vec3 hash33(vec3 p) { return hash33i(ivec3(floor(p))); }
float hash13(vec3 p) { return hash33(p).x; }
float hash12(vec2 p) { return hash33i(ivec3(ivec2(floor(p)), 7)).x; }
#endif
`;
