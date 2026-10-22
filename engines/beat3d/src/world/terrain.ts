// Procedural ground: one height function with a GLSL and a CPU implementation (same noise, same
// constants), a polar grid mesh that is dense near the camera, and a night lighting model for a
// frosted winter plain of dry grass:
//   sky ambient + a weak raking "Milky Way" key, frost that reflects the sky in patches,
//   tussocks (clumps with dark gaps between them) and grass tips catching the bright horizon,
//   up to four point lights with inverse-square falloff, where vertical blades still catch
//   near-horizontal light and dry grass glows when lit from behind (translucency),
//   a thin ground mist that scatters the point lights into a flat pool hugging the ground
//   (closed-form line integral, so it is smooth and exact at any distance),
//   then aerial perspective. Far terrain is flattened so the true horizon stays on the line.
// Frost sparkles are points, not shading: see world/frost.ts.

import * as THREE from 'three';
import { NOISE_GLSL } from '../gl/noise.glsl';
import { COLOR_GLSL } from '../gl/color.glsl';
import { fbm2 } from '../lib/noise';

/** A gaussian swell (or hollow, height < 0) added to the relief. Rotation `angle` in radians about +Y. */
export interface Swell {
  x: number;
  z: number;
  /** Radii (1/e) along the swell's own x and z axes, metres. */
  rx: number;
  rz: number;
  height: number;
  angle?: number;
}

export interface TerrainShape {
  /** Broad swells: amplitude (m) and wavelength (m). */
  bigAmp: number;
  bigScale: number;
  midAmp: number;
  midScale: number;
  smallAmp: number;
  smallScale: number;
  /** Distance range (m) over which relief flattens to `farFactor` (keeps the horizon straight). */
  flatFrom: number;
  flatTo: number;
  farFactor: number;
  /** Point the flattening is measured from (usually the camera's xz). */
  origin: [number, number];
  /** Up to four placed swells (shot design: a rise for a light to walk on). */
  swells?: Swell[];
}

export const PLAIN_SHAPE: TerrainShape = {
  bigAmp: 3.6,
  bigScale: 420,
  midAmp: 2.1,
  midScale: 75,
  smallAmp: 0.35,
  smallScale: 13,
  flatFrom: 1500,
  flatTo: 9000,
  farFactor: 0.22,
  origin: [0, 0],
};

const smoothstep = (a: number, b: number, x: number): number => {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
  return t * t * (3 - 2 * t);
};

function swellSum(x: number, z: number, sw: Swell[] | undefined): number {
  if (!sw) return 0;
  let h = 0;
  for (let i = 0; i < Math.min(sw.length, 4); i++) {
    const s = sw[i];
    const c = Math.cos(s.angle ?? 0);
    const n = Math.sin(s.angle ?? 0);
    const dx = x - s.x;
    const dz = z - s.z;
    const u = (c * dx + n * dz) / s.rx;
    const v = (-n * dx + c * dz) / s.rz;
    h += s.height * Math.exp(-(u * u + v * v));
  }
  return h;
}

/** CPU twin of terrainHeight() in GLSL. */
export function terrainHeight(x: number, z: number, s: TerrainShape): number {
  const big = fbm2(x / s.bigScale, z / s.bigScale, 4) * s.bigAmp;
  const mid = fbm2(x / s.midScale + 3.7, z / s.midScale + 3.7, 4) * s.midAmp;
  const small = fbm2(x / s.smallScale + 11.3, z / s.smallScale + 11.3, 3) * s.smallAmp;
  const d = Math.hypot(x - s.origin[0], z - s.origin[1]);
  const far = smoothstep(s.flatFrom, s.flatTo, d);
  return (big + mid + small + swellSum(x, z, s.swells)) * (1 + (s.farFactor - 1) * far);
}

/** First intersection of a ray with the terrain (ray march + bisection). */
export function raycastTerrain(ray: THREE.Ray, s: TerrainShape, maxDist = 5000): THREE.Vector3 | null {
  let tPrev = 0;
  let step = 0.5;
  const p = new THREE.Vector3();
  for (let t = 0.5; t < maxDist; t += step) {
    ray.at(t, p);
    if (p.y < terrainHeight(p.x, p.z, s)) {
      let lo = tPrev;
      let hi = t;
      for (let i = 0; i < 40; i++) {
        const m = (lo + hi) / 2;
        ray.at(m, p);
        if (p.y < terrainHeight(p.x, p.z, s)) hi = m;
        else lo = m;
      }
      return ray.at(hi, new THREE.Vector3());
    }
    tPrev = t;
    step = Math.max(0.25, t * 0.01);
  }
  return null;
}

/**
 * Highest canvas point (smallest y) of the terrain seen from `camera`, scanning columns of rays.
 * Use it to prove a framing keeps the horizon on the line: result.y must be >= lineY - 0.5.
 */
export function terrainSkylineY(camera: THREE.PerspectiveCamera, s: TerrainShape, width: number, height: number, columns = 24, maxDist = 4000): number {
  camera.updateMatrixWorld(true);
  let best = Infinity;
  const pos = new THREE.Vector3().setFromMatrixPosition(camera.matrixWorld);
  const v = new THREE.Vector3();
  for (let c = 0; c <= columns; c++) {
    const x = (c / columns) * width;
    for (let d = 2; d < maxDist; d *= 1.04) {
      // Ground point straight "under" this column at distance d: project it.
      const ndc = new THREE.Vector3((x / width) * 2 - 1, 0, 0.5).unproject(camera).sub(pos);
      ndc.y = 0;
      ndc.normalize();
      const gx = pos.x + ndc.x * d;
      const gz = pos.z + ndc.z * d;
      v.set(gx, terrainHeight(gx, gz, s), gz).project(camera);
      if (v.z > 1) continue;
      best = Math.min(best, (1 - (v.y * 0.5 + 0.5)) * height);
    }
  }
  return best;
}

const HEIGHT_GLSL = /* glsl */ `
uniform vec4 uShapeA;   // bigAmp, bigScale, midAmp, midScale
uniform vec4 uShapeB;   // smallAmp, smallScale, flatFrom, flatTo
uniform vec3 uShapeC;   // farFactor, origin.x, origin.z
uniform vec4 uSwellA[4]; // x, z, rx, rz
uniform vec4 uSwellB[4]; // height, cos, sin, -
uniform int uSwellN;
float terrainHeight(vec2 p) {
  float big = fbm2(p / uShapeA.y, 4) * uShapeA.x;
  float mid = fbm2(p / uShapeA.w + 3.7, 4) * uShapeA.z;
  float small = fbm2(p / uShapeB.y + 11.3, 3) * uShapeB.x;
  float sw = 0.0;
  for (int i = 0; i < 4; i++) {
    if (i >= uSwellN) break;
    vec2 d = p - uSwellA[i].xy;
    float u = (uSwellB[i].y * d.x + uSwellB[i].z * d.y) / uSwellA[i].z;
    float v = (-uSwellB[i].z * d.x + uSwellB[i].y * d.y) / uSwellA[i].w;
    sw += uSwellB[i].x * exp(-(u * u + v * v));
  }
  float d = length(p - uShapeC.yz);
  float far = smoothstep(uShapeB.z, uShapeB.w, d);
  return (big + mid + small + sw) * mix(1.0, uShapeC.x, far);
}
`;

const VERT = /* glsl */ `
${NOISE_GLSL}
${HEIGHT_GLSL}
varying vec3 vWorld;
void main() {
  vec3 p = position;
  p.y = terrainHeight(p.xz);
  vWorld = p;
  gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
}
`;

const FRAG = /* glsl */ `
${NOISE_GLSL}
${COLOR_GLSL}
${HEIGHT_GLSL}
uniform vec3 uSkyAmb;     // irradiance from the sky dome (already x 1/pi)
uniform vec3 uKeyDir;     // weak raking light from the bright part of the sky
uniform vec3 uKeyCol;
uniform vec3 uAlbedoA;    // dry grass
uniform vec3 uAlbedoB;    // bare earth
uniform vec3 uFrostAlb;   // hoar frost
uniform float uFrost;     // frost coverage 0..1
uniform float uTrans;     // dry-grass translucency (lit from behind)
uniform vec2 uMist;       // ground mist: layer height (m), scattering coefficient (1/m)
uniform vec3 uTus;        // tussocks: max height (m, 0 = off), cell size (m), fade-in distance (m)
uniform vec2 uVar;        // tussock variety, ground variety (0 = the original look)
uniform float uSheen;     // grazing sky sheen scale (1 = original)
uniform vec3 uFogCol;
uniform float uFogDist;
uniform vec4 uLightPos[4];   // xyz, w = near radius (m)
uniform vec3 uLightCol[4];   // colour x intensity (W-ish units; E = I / d^2)
uniform float uLightRange[4]; // soft effective radius (m): E x 1/(1 + (d/R)^4)
uniform int uLightCount;
uniform float uWrap;         // foliage wrap: vertical grass blades catch grazing light
uniform float uReveal;       // 0 = ground black, 1 = fully lit by the sky
varying vec3 vWorld;

// Light scattered towards the eye by a uniform slab of mist along the last s1 metres of the view
// ray before P, from a point light: integral of 1/(|X-L|^2 + r0^2) ds, in closed form.
float mistLine(vec3 P, vec3 V, float s1, vec3 Lp, float r0) {
  vec3 w = Lp - P;
  float s0 = dot(V, w);
  float b2 = max(dot(w, w) - s0 * s0, 0.0);
  float c = sqrt(b2 + r0 * r0);
  return (atan((s1 - s0) / c) - atan(-s0 / c)) / c;
}

// Ground variety (uVar.y): bare frozen soil (no clumps) and frost-filled hollows, at metres' scale.
float groundBare(vec2 p) {
  return uVar.y * smoothstep(0.22, 0.5, fbm2(p / 7.0 + 201.0, 3) + 0.15 * snoise(p / 0.8 + 9.0));
}
float groundHollow(vec2 p) {
  return uVar.y * smoothstep(0.3, 0.55, fbm2(p / 4.0 + 307.0, 3));
}
// A stone now and then (uVar.y): returns height (m) and writes its local normal tilt.
float groundStone(vec2 p, out vec2 tilt) {
  tilt = vec2(0.0);
  if (uVar.y <= 0.0) return 0.0;
  vec2 g = p / 2.3;
  vec2 i = floor(g);
  float best = 0.0;
  for (int y = -1; y <= 1; y++)
  for (int x = -1; x <= 1; x++) {
    vec2 c = i + vec2(float(x), float(y));
    vec3 h = hash33i(ivec3(int(c.x), int(c.y), 577));
    if (h.z > 0.07 * uVar.y) continue;
    vec2 cc = (c + 0.2 + 0.6 * h.xy) * 2.3;
    float r = 0.08 + 0.22 * fract(h.x * 13.7);
    vec2 d = (p - cc) / r;
    float q = dot(d, d);
    if (q >= 1.0) continue;
    float ht = r * 0.55 * sqrt(1.0 - q);
    if (ht > best) { best = ht; tilt = d * 0.9; }
  }
  return best;
}

// Tussock height field: ragged domes of dry grass on a jittered grid, with gaps of short stubble.
// info: x = height as a fraction of its clump (0 base .. 1 tip), y = angle around the clump centre,
// z = clump id; w = 1 inside a clump.
float tusField(vec2 p, out vec4 info) {
  vec2 g = p / uTus.y;
  vec2 i = floor(g);
  float best = uTus.x * 0.08;
  // Meadow structure: tall dense stands, short sparse turf, bare scrapes, at a few metres' scale.
  float stand = fbm2(p / 2.6 + 41.0, 3);
  float tall = 0.35 + 0.85 * smoothstep(-0.45, 0.5, stand);
  float keep = smoothstep(-0.65, -0.2, stand + 0.35 * snoise(p / 0.9 + 3.0));
  if (uVar.x > 0.0) {
    // Variety: density from low-frequency noise (dense stands, thin turf, open ground at metres to
    // tens of metres), so no lattice of equal clumps survives.
    float dens = 0.6 * fbm2(p / 9.0 + 77.0, 3) + 0.4 * fbm2(p / 2.2 + 19.0, 2);
    keep = mix(keep, smoothstep(-0.55, 0.25, dens + 0.25 * stand), uVar.x);
  }
  if (uVar.y > 0.0) keep *= 1.0 - groundBare(p);
  info = vec4(0.25, 0.0, 0.0, 0.0);
  for (int y = -1; y <= 1; y++)
  for (int x = -1; x <= 1; x++) {
    vec2 c = i + vec2(float(x), float(y));
    vec3 h = hash33i(ivec3(int(c.x), int(c.y), 913));
    if (h.z > keep + 0.05) continue;
    vec2 cc = (c + 0.15 + 0.7 * h.xy) * uTus.y;
    vec2 d = p - cc;
    float r = uTus.y * (0.42 + 0.4 * h.z);
    float hv = 1.0;
    if (uVar.x > 0.0) {
      // Each clump its own size, lean and outline: some dropped, some squat, some long and narrow.
      vec3 g2 = hash33i(ivec3(int(c.x), int(c.y), 421));
      if (g2.x < 0.3 * uVar.x) continue;
      float an = g2.y * 6.2831853;
      float asp = mix(1.0, 0.55 + 0.9 * g2.z, uVar.x);
      vec2 rd = vec2(cos(an) * d.x + sin(an) * d.y, -sin(an) * d.x + cos(an) * d.y);
      d = vec2(rd.x / asp, rd.y * asp);
      r *= mix(1.0, 0.5 + 0.8 * fract(g2.x * 7.31 + g2.z), uVar.x);
      hv = mix(1.0, 0.45 + 1.1 * fract(g2.y * 5.17 + g2.x), uVar.x);
    }
    float ang = atan(d.y, d.x);
    // Ragged rim and a fringe of blade tips: the clump's top is spiky, not a dome.
    float rr = r * (0.78 + 0.22 * sin(ang * 5.0 + h.x * 40.0) * sin(ang * 3.0 + h.y * 20.0));
    float q = dot(d, d) / (rr * rr);
    if (q >= 1.0) continue;
    float tips = 0.62 + 0.38 * abs(sin(ang * 23.0 + h.y * 70.0) * sin(ang * 37.0 + h.x * 90.0 + q * 3.0));
    float hc = uTus.x * (0.35 + 0.65 * h.y) * tall * hv;
    float ht = hc * pow(1.0 - q, 0.35) * tips;
    if (ht > best) {
      best = ht;
      info = vec4(ht / hc, ang, h.x, 1.0);
    }
  }
  return best;
}

void main() {
  vec3 P0 = vWorld;
  vec3 V = normalize(cameraPosition - P0);
  float dist = length(cameraPosition - P0);
  // Footprints: overall, and across the view (the narrow side of a grazing pixel).
  vec2 dx = dFdx(P0.xz);
  vec2 dy = dFdy(P0.xz);
  float fp = max(length(fwidth(P0.xz)), 1e-4);
  float fpX = max(min(length(dx), length(dy)), 1e-4);
  // Per-pixel normal from the full height function; finest octaves fade with footprint.
  float e = max(fp * 0.75, 0.02);
  float h0 = terrainHeight(P0.xz);
  float hx = terrainHeight(P0.xz + vec2(e, 0.0));
  float hz = terrainHeight(P0.xz + vec2(0.0, e));
  vec3 Ng = normalize(vec3(h0 - hx, e, h0 - hz));

  // ---- tussock layer: parallax-march the view ray through the grass height field.
  float tusLod = uTus.x > 0.0 ? (1.0 - smoothstep(uTus.y * 0.2, uTus.y * 0.45, fpX)) * smoothstep(uTus.z, uTus.z * 1.4 + 0.01, dist) : 0.0;
  vec3 P = P0;
  vec4 tinfo = vec4(0.25, 0.0, 0.0, 0.0);
  float tusH = 0.0;
  if (tusLod > 0.0) {
    float vn = max(dot(V, Ng), 0.03);
    float sTop = min(uTus.x / vn, 3.0);
    float sPrev = sTop;
    float sHit = -1.0;
    vec4 inf;
    for (int k = 1; k <= 24; k++) {
      float sk = sTop * (1.0 - float(k) / 24.0);
      float hr = sk * vn;
      float hf = tusField(P0.xz + V.xz * sk, inf);
      if (hr <= hf) { sHit = sk; break; }
      sPrev = sk;
    }
    if (sHit < 0.0) sHit = 0.0;
    float lo = sHit, hi = sPrev;
    for (int k = 0; k < 5; k++) {
      float m = 0.5 * (lo + hi);
      if (m * vn <= tusField(P0.xz + V.xz * m, inf)) lo = m; else hi = m;
    }
    tusH = tusField(P0.xz + V.xz * lo, tinfo);
    P = P0 + V * lo * tusLod;
  }

  // ---- tufts (fine bump) and occlusion between clumps
  float tuftFade = 1.0 - smoothstep(0.03, 0.12, fpX);
  vec2 tg = vec2(snoise(P.xz * 3.1), snoise(P.xz * 3.1 + 17.0));
  vec3 N = normalize(Ng + vec3(tg.x, 0.0, tg.y) * 0.22 * tuftFade);
  float hfrac = tinfo.x;
  vec2 radial = vec2(cos(tinfo.y), sin(tinfo.y));
  vec3 Nt = normalize(Ng + vec3(radial.x, 0.0, radial.y) * (0.55 - 0.35 * hfrac) * tinfo.w);
  N = normalize(mix(N, Nt, tusLod));
  // Fibres: radial streaks on a clump read as upright blades from the side.
  float fib = snoise(vec2(tinfo.y * 11.0 + tinfo.z * 60.0, hfrac * 2.5)) * 0.5 + 0.5 * snoise(vec2(tinfo.y * 27.0 + tinfo.z * 13.0, hfrac * 5.0));
  float aoT = (0.5 + 0.5 * pow(hfrac, 0.8)) * (0.78 + 0.22 * fib);
  float ao = mix(0.86, aoT, tusLod);

  // ---- albedo and frost (octaves fade with the footprint so nothing aliases into dashes)
  float lod1 = 1.0 - smoothstep(0.25, 0.9, fp);
  float lod2 = 1.0 - smoothstep(0.08, 0.3, fp);
  float patchN = fbm2(P.xz / 7.0 + 5.0, 4);
  float grassy = smoothstep(-0.45, 0.45, patchN);
  vec3 albedo = mix(uAlbedoB, uAlbedoA, grassy) * (0.86 + 0.28 * (0.5 + 0.5 * snoise(P.xz * 0.9) * lod1));
  albedo *= mix(1.0, 0.85 + 0.3 * (0.5 + 0.5 * fib), tusLod * tinfo.w);
  float fr = uFrost + 0.42 * fbm2(P.xz / 4.5 + 11.0, 2) + 0.12 * snoise(P.xz / 1.1 + 4.0) * lod1 + 0.12 * snoise(P.xz / 0.35 + 2.0) * lod2;
  fr += tusLod * (0.45 * smoothstep(0.35, 0.95, hfrac) - 0.25);
  if (uVar.y > 0.0) {
    // Bare frozen soil: darker, finer, a hoar rime; hollows hold frost; stones are dark with a cap.
    float bare = groundBare(P.xz);
    float hol = groundHollow(P.xz);
    albedo = mix(albedo, uAlbedoB * (0.7 + 0.3 * (0.5 + 0.5 * snoise(P.xz * 6.0) * lod2)), bare * (1.0 - tusLod * tinfo.w));
    fr = mix(fr, 0.55 + 0.25 * snoise(P.xz * 2.3) * lod1, bare * 0.6);
    fr += 0.6 * hol;
    vec2 tilt;
    float st = groundStone(P.xz, tilt);
    if (st > 0.0) {
      N = normalize(N + vec3(tilt.x, 0.0, tilt.y));
      albedo = mix(albedo, vec3(0.09, 0.09, 0.1), smoothstep(0.0, 0.01, st));
      fr = mix(fr, 0.9 * smoothstep(0.3, 0.9, N.y), smoothstep(0.0, 0.01, st));
    }
  }
  float frost = clamp(fr, 0.0, 1.0);
  albedo = mix(albedo, uFrostAlb, frost * 0.7);

  vec3 L = vec3(0.0);
  float skyView = 0.55 + 0.45 * N.y;
  L += albedo * uSkyAmb * skyView * ao * uReveal;
  L += albedo * uKeyCol * max(dot(N, uKeyDir) * 0.8 + 0.2, 0.0) * ao * uReveal;
  // Grass tips and frost catch the bright horizon sky at grazing angles: a faint sheen that shapes
  // the swells and keeps the far plain from going to a flat black.
  float graze = pow(1.0 - clamp(dot(Ng, V), 0.0, 1.0), 4.0);
  vec3 sheenCol = mix(uFogCol, uFogCol * vec3(0.9, 1.0, 1.15), frost);
  L += uSheen * sheenCol * (0.75 + 0.6 * frost) * graze * smoothstep(-0.2, 0.6, dot(Ng, normalize(vec3(-V.x, 0.0, -V.z)))) * mix(1.0, 0.4 + 0.6 * hfrac, tusLod) * uReveal;

  // Point lights.
  vec3 mist = vec3(0.0);
  float s1 = min(dist, uMist.x / max(dot(V, Ng), uMist.x / max(dist, 1e-3)));
  for (int i = 0; i < 4; i++) {
    if (i >= uLightCount) break;
    vec3 lv = uLightPos[i].xyz - P;
    float d2 = dot(lv, lv);
    vec3 l = lv * inversesqrt(d2);
    float r0 = uLightPos[i].w;
    float rq = d2 / (uLightRange[i] * uLightRange[i]);
    vec3 E = uLightCol[i] / (d2 + r0 * r0) / (1.0 + rq * rq);
    float ndl = (dot(N, l) + uWrap) / (1.0 + uWrap);
    // Blades stand up: horizontal light still lands on them, more so where we look across them.
    float blade = sqrt(max(1.0 - l.y * l.y, 0.0)) * (0.35 + 0.65 * graze);
    L += albedo * E * (max(ndl, 0.0) * 0.7 + 0.45 * blade) * ao;
    // Dry grass between the eye and the light glows through (translucency, forward lobe).
    float back = pow(max(dot(-V, l), 0.0), 2.0) * (0.4 + 0.6 * graze);
    L += uAlbedoA * vec3(1.0, 0.85, 0.6) * E * uTrans * back * grassy * (0.6 + 0.4 * ao) * mix(1.0, hfrac, tusLod);
    // Frost: a soft sheen towards the light (the individual sparkles are frost.ts points).
    vec3 hv = normalize(l + V);
    L += uFrostAlb * E * frost * 0.12 * pow(max(dot(N, hv), 0.0), 24.0) * ao;
    // Ground mist in-scatter (isotropic phase).
    if (uMist.y > 0.0) mist += uLightCol[i] * uMist.y * 0.0796 * mistLine(P0, V, s1, uLightPos[i].xyz, max(r0, 0.3));
  }
  // Aerial perspective: distant ground lifts towards (dimmer than) the horizon sky.
  float T = exp(-dist / uFogDist);
  L = L * T + uFogCol * (1.0 - T) * uReveal + mist;
  gl_FragColor = vec4(L, 1.0);
}
`;

/** GLSL of `terrainHeight(vec2)` (same formula as the CPU twin) for other ground materials; set its
 *  uniforms with `terrainShapeUniforms(shape)`. */
export const TERRAIN_HEIGHT_GLSL = HEIGHT_GLSL;

/** Uniforms for TERRAIN_HEIGHT_GLSL. */
export function terrainShapeUniforms(shape: TerrainShape): Record<string, THREE.IUniform> {
  const sw = shape.swells ?? [];
  return {
    uShapeA: { value: new THREE.Vector4(shape.bigAmp, shape.bigScale, shape.midAmp, shape.midScale) },
    uShapeB: { value: new THREE.Vector4(shape.smallAmp, shape.smallScale, shape.flatFrom, shape.flatTo) },
    uShapeC: { value: new THREE.Vector3(shape.farFactor, shape.origin[0], shape.origin[1]) },
    uSwellA: { value: [0, 1, 2, 3].map((i) => (sw[i] ? new THREE.Vector4(sw[i].x, sw[i].z, sw[i].rx, sw[i].rz) : new THREE.Vector4(0, 0, 1, 1))) },
    uSwellB: { value: [0, 1, 2, 3].map((i) => (sw[i] ? new THREE.Vector4(sw[i].height, Math.cos(sw[i].angle ?? 0), Math.sin(sw[i].angle ?? 0), 0) : new THREE.Vector4(0, 1, 0, 0))) },
    uSwellN: { value: Math.min(sw.length, 4) },
  };
}

export interface TerrainLook {
  skyAmbient: [number, number, number];
  keyDir: THREE.Vector3;
  keyColor: [number, number, number];
  albedoGrass: [number, number, number];
  albedoEarth: [number, number, number];
  fogColor: [number, number, number];
  fogDistance: number;
  wrap: number;
  /** Frost coverage bias (default 0.45) and frost albedo (default bluish white). */
  frost?: number;
  frostAlbedo?: [number, number, number];
  /** Dry-grass translucency under back light (default 0.5). */
  translucency?: number;
  /** Ground mist: layer height in metres (default 0.7) and scattering coefficient per metre (default 0.02; 0 = off). */
  mistHeight?: number;
  mistDensity?: number;
  /** Tussocks of dry grass, parallax-marched: max height (m, default 0.17; 0 = off) and cell size (m, default 0.3). */
  tussockHeight?: number;
  tussockCell?: number;
  /** Distance (m) from the eye where tussocks start (fade in to 1.4x); default 0. Use it when real blades cover the near field. */
  tussockFrom?: number;
  /** Tussock variety 0..1 (default 0 = original): density from low-frequency noise, each clump its own size, height, outline and lean. */
  tussockVariety?: number;
  /** Ground variety 0..1 (default 0): patches of bare frozen soil, frost-filled hollows, the odd stone. */
  groundVariety?: number;
  /** Scale of the grazing sky sheen on grass tips and frost (default 1). Night aerials want ~0.2. */
  sheen?: number;
}

/**
 * A frosted winter plain under starlight (a base look; snow fields start from it).
 * keyDir is per shot (towards the bright part of the Milky Way).
 */
export const FROST_PLAIN_LOOK: Omit<TerrainLook, 'keyDir'> = {
  skyAmbient: [0.0105, 0.0138, 0.0225],
  keyColor: [0.075, 0.068, 0.058],
  albedoGrass: [0.33, 0.26, 0.16],
  albedoEarth: [0.15, 0.13, 0.11],
  fogColor: [0.0072, 0.0102, 0.019],
  fogDistance: 2600,
  wrap: 0.7,
  frost: 0.45,
  frostAlbedo: [0.64, 0.67, 0.71],
  translucency: 0.5,
  mistHeight: 0.7,
  mistDensity: 0.02,
  tussockHeight: 0.17,
  tussockCell: 0.3,
};

export interface PointLightSpec {
  position: THREE.Vector3;
  color: [number, number, number];
  /** Radiant intensity scale: irradiance at 1 m is color * intensity. */
  intensity: number;
  nearRadius: number;
  /** Optional soft effective radius (m): irradiance x 1/(1 + (d/R)^4). About 5 m for a hand-held light. */
  range?: number;
}

export class Terrain {
  readonly mesh: THREE.Mesh;
  readonly material: THREE.ShaderMaterial;
  constructor(
    readonly shape: TerrainShape,
    center: [number, number],
    opts: { rings?: number; segments?: number; r0?: number; r1?: number } = {},
  ) {
    const rings = opts.rings ?? 420;
    const segs = opts.segments ?? 900;
    const r0 = opts.r0 ?? 0.5;
    const r1 = opts.r1 ?? 25000;
    const pos = new Float32Array((rings * segs + 1) * 3);
    pos.set([center[0], 0, center[1]], 0);
    for (let i = 0; i < rings; i++) {
      const r = r0 * Math.pow(r1 / r0, i / (rings - 1));
      for (let j = 0; j < segs; j++) {
        const a = (j / segs) * Math.PI * 2;
        const k = 1 + i * segs + j;
        pos[k * 3] = center[0] + Math.cos(a) * r;
        pos[k * 3 + 2] = center[1] + Math.sin(a) * r;
      }
    }
    const idx: number[] = [];
    for (let j = 0; j < segs; j++) idx.push(0, 1 + ((j + 1) % segs), 1 + j);
    for (let i = 0; i < rings - 1; i++) {
      for (let j = 0; j < segs; j++) {
        const a = 1 + i * segs + j;
        const b = 1 + i * segs + ((j + 1) % segs);
        const c = 1 + (i + 1) * segs + j;
        const d = 1 + (i + 1) * segs + ((j + 1) % segs);
        idx.push(a, b, c, b, d, c);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setIndex(idx);
    const sw = shape.swells ?? [];
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: {
        uShapeA: { value: new THREE.Vector4(shape.bigAmp, shape.bigScale, shape.midAmp, shape.midScale) },
        uShapeB: { value: new THREE.Vector4(shape.smallAmp, shape.smallScale, shape.flatFrom, shape.flatTo) },
        uShapeC: { value: new THREE.Vector3(shape.farFactor, shape.origin[0], shape.origin[1]) },
        uSwellA: { value: [0, 1, 2, 3].map((i) => (sw[i] ? new THREE.Vector4(sw[i].x, sw[i].z, sw[i].rx, sw[i].rz) : new THREE.Vector4(0, 0, 1, 1))) },
        uSwellB: { value: [0, 1, 2, 3].map((i) => (sw[i] ? new THREE.Vector4(sw[i].height, Math.cos(sw[i].angle ?? 0), Math.sin(sw[i].angle ?? 0), 0) : new THREE.Vector4(0, 1, 0, 0))) },
        uSwellN: { value: Math.min(sw.length, 4) },
        uSkyAmb: { value: new THREE.Vector3() },
        uKeyDir: { value: new THREE.Vector3(0, 1, 0) },
        uKeyCol: { value: new THREE.Vector3() },
        uAlbedoA: { value: new THREE.Vector3() },
        uAlbedoB: { value: new THREE.Vector3() },
        uFrostAlb: { value: new THREE.Vector3(0.62, 0.68, 0.78) },
        uFrost: { value: 0.45 },
        uTrans: { value: 0.5 },
        uMist: { value: new THREE.Vector2(0.7, 0.02) },
        uTus: { value: new THREE.Vector3(0.17, 0.3, 0) },
        uVar: { value: new THREE.Vector2(0, 0) },
        uSheen: { value: 1 },
        uFogCol: { value: new THREE.Vector3() },
        uFogDist: { value: 3000 },
        uLightPos: { value: [0, 1, 2, 3].map(() => new THREE.Vector4()) },
        uLightCol: { value: [0, 1, 2, 3].map(() => new THREE.Vector3()) },
        uLightRange: { value: [1e6, 1e6, 1e6, 1e6] },
        uLightCount: { value: 0 },
        uWrap: { value: 0.6 },
        uReveal: { value: 1 },
      },
      side: THREE.FrontSide,
      toneMapped: false,
    });
    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.frustumCulled = false;
  }

  update(look: TerrainLook, lights: PointLightSpec[], reveal: number, gain: number): void {
    const u = this.material.uniforms;
    u.uSkyAmb.value.set(...look.skyAmbient).multiplyScalar(gain);
    u.uKeyDir.value.copy(look.keyDir).normalize();
    u.uKeyCol.value.set(...look.keyColor).multiplyScalar(gain);
    u.uAlbedoA.value.set(...look.albedoGrass);
    u.uAlbedoB.value.set(...look.albedoEarth);
    u.uFrostAlb.value.set(...(look.frostAlbedo ?? [0.62, 0.68, 0.78]));
    u.uFrost.value = look.frost ?? 0.45;
    u.uTrans.value = look.translucency ?? 0.5;
    u.uMist.value.set(look.mistHeight ?? 0.7, look.mistDensity ?? 0.02);
    u.uTus.value.set(look.tussockHeight ?? 0.17, look.tussockCell ?? 0.3, look.tussockFrom ?? 0);
    u.uVar.value.set(look.tussockVariety ?? 0, look.groundVariety ?? 0);
    u.uSheen.value = look.sheen ?? 1;
    u.uFogCol.value.set(...look.fogColor).multiplyScalar(gain);
    u.uFogDist.value = look.fogDistance;
    u.uWrap.value = look.wrap;
    u.uReveal.value = reveal;
    const n = Math.min(lights.length, 4);
    u.uLightCount.value = n;
    for (let i = 0; i < 4; i++) {
      const l = lights[i];
      if (!l) {
        u.uLightCol.value[i].set(0, 0, 0);
        continue;
      }
      u.uLightPos.value[i].set(l.position.x, l.position.y, l.position.z, l.nearRadius);
      u.uLightCol.value[i].set(...l.color).multiplyScalar(l.intensity);
      u.uLightRange.value[i] = l.range ?? 1e6;
    }
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.material.dispose();
  }
}
