// 以光为墨 ("light as ink"): a shared brush for films whose world is drawn only with light on darkness
// (or ink on paper). Every shot draws with these things, so the line has one hand:
//
//   LightBrush  - strokes: polylines in world space drawn as screen-space ribbons with a gaussian core
//                 and a soft halo, like a brush stroke of light: tapered ends, brush pressure (width and
//                 brightness breathe along the stroke), a slight hand tremor, and a draw-on reveal.
//   LightPoints - points: lamps, stars, walkers, snow, rain, fireflies. Optional cross flare, and a motion
//                 streak from a world velocity (rain, driven snow, sparks).
//   Wash        - colour fields: a soft vertical gradient with the horizon on the film line plus soft
//                 blobs. It sets the hour and the season; it never draws a shape.
//   Occluder    - the land, never drawn: in the depth buffer it removes hidden lines; as a depth pre-pass
//                 it decides per point whether a lamp is seen, so a glow is never cut by the ground.
//   Paper       - the ground of the chapters that are not light on black: rice
//                 paper for ink, black lacquer for gold.
//
// Two modes. 'light' (default) adds light. 'ink' multiplies: the colour is what the stroke absorbs, so
// strokes darken the paper and overlapping strokes darken further, like ink; use it on Paper.
// `edge` turns the soft gaussian core into a hard edge (gold leaf, the carving knife).
// Depth of field is computed per element (circle of confusion from the view depth), so a defocused line
// becomes a wider, dimmer line and a defocused point becomes a disc with the same energy. Shots drawn
// with the brush do not use post.dof (strokes write no depth). Distance fades and cools every element
// (aerial perspective). Lines thinner than `minPx` keep their energy instead of crawling.
// Everything is a pure function of what you pass per frame; noise is seeded per stroke.

import * as THREE from 'three';
import { NOISE_GLSL } from '../gl/noise.glsl';
import { Rng, hashString } from '../lib/rng';
import { layoutAt } from '../lib/layout';
import type { StageContext } from '../core/types';

export type RGB = [number, number, number];

/** Super 35 width, as in post/FilmPost (mm). */
const SENSOR_MM = 24.89;

/** Lens and atmosphere shared by brush, points and (for the horizon) wash. Set per frame. */
export interface InkLook {
  /** Focus distance (m). Omit or 0 for no defocus. */
  focus?: number;
  focalMm?: number;
  fStop?: number;
  /** Largest circle of confusion radius (1080p px, default 24). */
  maxCoc?: number;
  /** Distance (m) over which elements fade to 1/e (0 = off, default 0). */
  fogDistance?: number;
  /** Colour multiplier reached at full fog (default a cool blue-grey). */
  fogTint?: RGB;
  /** Master gain (default 1). */
  gain?: number;
  /** Film frame, for anything that shimmers (default 0). */
  frame?: number;
}

const LENS_GLSL = /* glsl */ `
uniform vec2 uRes;
uniform float uScale;
uniform float uMinPx;
uniform float uCocK;
uniform float uFocus;
uniform float uCocMax;
uniform float uFogDist;
uniform vec3 uFogTint;
uniform float uGain;
uniform float uTime;
float cocRadius(float z) {
  if (uCocK <= 0.0) return 0.0;
  return min(0.5 * uCocK * abs(z - uFocus) / max(z, 1e-3), uCocMax);
}
float fogOf(float z) { return uFogDist > 0.0 ? exp(-z / uFogDist) : 1.0; }
vec2 toPx(vec4 c) { return (c.xy / c.w) * 0.5 * uRes; }
uniform float uDepthBias;
// Clip-space z (for a vertex whose clip w is w) of a view-space point pulled toward the camera by a
// fraction of its distance: a glow is hidden by a hill well in front of it, not cut by the ground it
// stands on.
float biasedClipZ(vec4 vp, float w) {
  vec4 b = projectionMatrix * vec4(vp.xy, min(vp.z * (1.0 - uDepthBias), -1e-3), 1.0);
  return b.z / b.w * w;
}
`;

function lensUniforms(ctx: StageContext): Record<string, THREE.IUniform> {
  return {
    uRes: { value: new THREE.Vector2(ctx.width, ctx.height) },
    uScale: { value: ctx.scale },
    uMinPx: { value: 0.6 * ctx.scale },
    uCocK: { value: 0 },
    uFocus: { value: 10 },
    uCocMax: { value: 24 * ctx.scale },
    uFogDist: { value: 0 },
    uFogTint: { value: new THREE.Vector3(0.55, 0.65, 0.85) },
    uGain: { value: 1 },
    uTime: { value: 0 },
    uDepthBias: { value: 0 },
  };
}

function applyLook(u: Record<string, THREE.IUniform>, ctx: StageContext, look: InkLook): void {
  const focus = look.focus ?? 0;
  if (focus > 0) {
    const f = (look.focalMm ?? 50) / 1000;
    const A = f / (look.fStop ?? 2.8);
    u.uCocK.value = ((A * f) / Math.max(focus - f, 1e-3) / (SENSOR_MM / 1000)) * ctx.width;
    u.uFocus.value = focus;
  } else u.uCocK.value = 0;
  u.uCocMax.value = (look.maxCoc ?? 24) * ctx.scale;
  u.uFogDist.value = look.fogDistance ?? 0;
  const t = look.fogTint ?? [0.55, 0.65, 0.85];
  (u.uFogTint.value as THREE.Vector3).set(t[0], t[1], t[2]);
  u.uGain.value = look.gain ?? 1;
  u.uTime.value = (look.frame ?? 0) / 24;
}

export type InkMode = 'light' | 'ink';

/** Blending for a mode: light adds; ink multiplies the target by (1 - colour). */
function blendOf(mode: InkMode): Partial<THREE.ShaderMaterialParameters> {
  if (mode === 'light') return { blending: THREE.AdditiveBlending };
  return {
    blending: THREE.CustomBlending,
    blendEquation: THREE.AddEquation,
    blendSrc: THREE.ZeroFactor,
    blendDst: THREE.OneMinusSrcColorFactor,
    blendSrcAlpha: THREE.ZeroFactor,
    blendDstAlpha: THREE.OneFactor,
  };
}

// ------------------------------------------------------------------------------------------------
// LightBrush

export interface Stroke {
  /** World-space points: Vector3s or a flat xyz array. At least 2. */
  points: THREE.Vector3[] | ArrayLike<number>;
  /** Linear HDR colour at the start (default warm amber) and, optionally, at the end. */
  color?: RGB;
  color2?: RGB;
  /** Brightness of the core (default 1). */
  gain?: number;
  /** Core width, 1080p px (default 1.2). */
  width?: number;
  /** Halo strength relative to the core peak (default 0.18) and its radius in 1080p px (default 7). */
  halo?: number;
  haloPx?: number;
  /** Tapered fraction of the length at the start and at the end (default [0.06, 0.1]). */
  taper?: [number, number];
  /** Brush pressure 0..1: how much width and brightness breathe along the stroke (default 0.35). */
  pressure?: number;
  /** Wavelength of the pressure in metres (default 6). */
  pressureScale?: number;
  /** Hand tremor, 1080p px (default 0.35). */
  wobble?: number;
  /** 0 = constant pixel width; 1 = width grows as the stroke comes closer (default 0.3). */
  perspective?: number;
  /** Distance (m) at which `width` is exact when perspective > 0 (default 20). */
  refDistance?: number;
  /** Fade out with distance: full up to [0] m, gone at [1] m (level of detail for dense line fields). */
  fade?: [number, number];
  /** Fade in with distance: gone below [0] m, full from [1] m (lines that should not reach the lens). */
  fadeNear?: [number, number];
  /** 飞白, dry brush 0..1: the core breaks into bristle lanes, more so toward the end of the stroke.
   *  Shows on strokes wider than ~3 px (hero strokes: roads, the title); default 0. */
  dry?: number;
  /** Length of the bristle streaks along the stroke (m, default 8). */
  dryScale?: number;
  /** 0 = soft gaussian core (light); 1 = hard edge, anti-aliased (gold leaf, carving) (default 0). */
  edge?: number;
  /** Width in metres at the stroke, in true perspective (roads, paths, stems); overrides `width`,
   *  `perspective` and `refDistance`. Thinner than `minPx` it keeps its energy. */
  widthM?: number;
  /** Lies flat on the ground (with `widthM`): the width runs across the stroke in the model's XZ plane,
   *  so it foreshortens with the ground (a road across the view is thin, one going away is wide)
   *  instead of facing the camera. Never thinner on screen than the core at `minPx`. Default false. */
  flat?: boolean;
  /** 'round': an end that keeps its width (taper 0 there) closes in a half disc, the round touch of a
   *  brush, instead of a square cut (branches leaving a trunk, roof strokes, road ends). Default flat. */
  cap?: 'round' | 'flat';
  /** Per-point multipliers of the width and of the strength (same length as `points`): the brush
   *  pressing harder here, a road brighter near a town, an engraved line swelling with the tone. */
  widths?: ArrayLike<number>;
  gains?: ArrayLike<number>;
  /** Seed key (default the stroke index). */
  seed?: string;
}

/**
 * Per-stroke values that may change every frame. Lengths are fractions of the stroke (use `length(i)`
 * to work in metres). A draw-on is `{ reveal }`; a pulse running along the line (a letter, a spark)
 * is `{ reveal, trail: 0, tail, head }`: nothing but the head and a fading tail behind it.
 */
export interface StrokeState {
  /** Head along the stroke, 0..1 (default past the end = fully drawn). */
  reveal?: number;
  /** Softness of the head (default 0.03). */
  soft?: number;
  /** Extra glow at the head (default 0). */
  head?: number;
  /** Brightness multiplier (default 1). */
  gain?: number;
  /** Brightness of the drawn part far behind the head, 0..1 (default 1). */
  trail?: number;
  /** Length over which the drawn part falls from the head to `trail` (default 0.1). */
  tail?: number;
  /** Width multiplier (default 1): a path that thickens as more walkers pass. */
  width?: number;
  /** Halo multiplier (default 1): light soaking outward like ink into paper. */
  halo?: number;
  /** Where the writing starts, 0..1 (default 0). In between, the stroke is written outward in both
   *  directions from here (reveal is then the distance covered on each side), with no seam. */
  from?: number;
  /** Distance written back toward the start, from `from` (default: the same as `reveal`), so the two
   *  sides of a two-way stroke can grow at their own pace. `reveal` then covers the forward side. */
  back?: number;
}

/** Texels per stroke in the state texture: [reveal, gain, soft, head], [trail, tail, width, halo], [from, back, -, -].
 *  back < 0 means the same as reveal. */
const STATE_DEFAULT = [2, 1, 0.03, 0, 1, 0.1, 1, 1, 0, -1, 0, 0];
const STATE_TEXELS = 3;

const BRUSH_VERT = /* glsl */ `
${NOISE_GLSL}
${LENS_GLSL}
uniform sampler2D tState;
uniform vec2 uStateSize;
uniform float uInkDensity;
attribute vec3 aPrev;
attribute vec3 aNext;
attribute float aSide;   // ±1; ±2 / ±3 mark the extra point of a round cap at the start / end
attribute float aU;
attribute float aS;
attribute float aStroke;
attribute vec3 aColor;
attribute vec4 aShape;   // width px, halo gain, halo px, gain
attribute vec4 aStyle;   // taper start, taper end, pressure, wobble px
attribute vec4 aExtra;   // seed, pressure wavelength (m), perspective, ref distance
attribute vec4 aFade;    // near gone, near full, far full, far gone (m)
attribute vec4 aInk;     // dry brush amount, streak length (m), edge, width in metres (0 = px)
attribute vec3 aMod;     // per-point width and strength; lies flat (1)
varying vec3 vColor;
varying vec4 vDry;       // amount, streak length, seed, edge
varying float vS;
varying float vAcross;
varying float vHalfW;
varying float vSigma;
varying float vHaloPx;
varying float vHalo;
varying float vU;
varying vec4 vState;
varying vec4 vState2;
varying float vFrom;
varying float vBack;
varying float vCap;
void main() {
  float sideS = aSide > 0.0 ? 1.0 : -1.0;
  float aCap = abs(aSide) < 1.5 ? 0.0 : (abs(aSide) < 2.5 ? -1.0 : 1.0);
  // uStateSize = strokes per row, rows; three texels per stroke.
  float idx = aStroke;
  float col = mod(idx, uStateSize.x);
  float row = (floor(idx / uStateSize.x) + 0.5) / uStateSize.y;
  vState = texture2D(tState, vec2((3.0 * col + 0.5) / (3.0 * uStateSize.x), row));
  vState2 = texture2D(tState, vec2((3.0 * col + 1.5) / (3.0 * uStateSize.x), row));
  vec4 s3 = texture2D(tState, vec2((3.0 * col + 2.5) / (3.0 * uStateSize.x), row));
  vFrom = s3.x;
  vBack = s3.y;

  mat4 mv = modelViewMatrix;
  vec4 vp = mv * vec4(position, 1.0);
  float z = max(-vp.z, 1e-3);
  vec4 c = projectionMatrix * vp;
  vec4 cp = projectionMatrix * (mv * vec4(aPrev, 1.0));
  vec4 cn = projectionMatrix * (mv * vec4(aNext, 1.0));
  // A vertex behind the camera slides along its segment to just in front of it (the stroke is cut
  // at the near plane instead of flipping across the screen).
  const float EPS = 0.05;
  bool slid = c.w < EPS;
  if (slid) {
    vec4 o = cn.w > EPS ? cn : cp;
    if (o.w <= EPS) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
    c = mix(c, o, (EPS - c.w) / (o.w - c.w));
  }
  cp.w = max(cp.w, 1e-3);
  cn.w = max(cn.w, 1e-3);
  vec2 s = toPx(vec4(c.xy, c.z, max(c.w, 1e-3)));
  vec2 d1 = s - toPx(cp);
  vec2 d2 = toPx(cn) - s;
  vec2 t1 = length(d1) > 1e-4 ? normalize(d1) : vec2(0.0);
  vec2 t2 = length(d2) > 1e-4 ? normalize(d2) : vec2(0.0);
  vec2 dir = t1 + t2;
  dir = length(dir) > 1e-4 ? normalize(dir) : (length(t2) > 0.0 ? t2 : vec2(1.0, 0.0));
  vec2 nrm = vec2(-dir.y, dir.x);
  // Miter: keep the ribbon width across sharp bends, limited to 2x.
  float miter = 1.0 / max(dot(nrm, length(t1) > 0.0 ? vec2(-t1.y, t1.x) : nrm), 0.5);

  float seed = aExtra.x;
  // A taper of 0 means none: the end keeps its full width (joins, closed loops, abutting strokes).
  float taper = (aStyle.x > 0.0 ? smoothstep(0.0, aStyle.x, aU) : 1.0) * (aStyle.y > 0.0 ? 1.0 - smoothstep(1.0 - aStyle.y, 1.0, aU) : 1.0);
  float n1 = snoise(vec2(aS / max(aExtra.y, 1e-3), seed));
  float n2 = snoise(vec2(aS / max(aExtra.y * 0.23, 1e-3), seed + 17.0));
  float press = 1.0 + aStyle.z * (0.75 * n1 + 0.25 * n2);
  float persp = mix(1.0, clamp(aExtra.w / z, 0.35, 3.0), aExtra.z);
  float basePx = aInk.w > 0.0 ? aInk.w * projectionMatrix[1][1] * 0.5 * uRes.y / z : aShape.x * uScale * persp;
  float wPx = basePx * max(press, 0.25) * max(taper, 0.0) * vState2.z * aMod.x;

  float coc = cocRadius(z);
  float sigma0 = max(0.5 * wPx, 1e-4);
  float sigma = sqrt(sigma0 * sigma0 + 0.25 * coc * coc);
  float sigmaMin = 0.5 * uMinPx;
  float energy = sigma0 / max(sigma, sigmaMin);
  sigma = max(sigma, sigmaMin);
  float haloPx = aShape.z * uScale + coc;
  // A ribbon much wider than its segments are long folds over at every bend and draws spikes (lines
  // seen edge-on, far contours). The halo is windowed to the ribbon, so where segments are short on
  // screen it just narrows; the core always fits.
  float l1 = length(d1);
  float l2 = length(d2);
  float segPx = min(l1 > 1e-4 ? l1 : 1e6, l2 > 1e-4 ? l2 : 1e6);
  // Ink density k amplifies the gaussian's skirt about k times: the ribbon (and the window that
  // closes it) reaches out until that is negligible, so a soft stroke's edge never shows as a box.
  float reach = uInkDensity > 1.0 ? sqrt(2.0 * log(uInkDensity / 0.004)) / 0.8 : 3.2;
  float coreW = reach * sigma + 1.0;
  float haloW = aShape.y > 0.0 ? min(3.5 * haloPx, max(1.5 * segPx, coreW)) : 0.0;
  float halfW = max(coreW, haloW);

  float wob = aStyle.w * uScale * snoise(vec2(aS * 0.8, seed + 41.0));
  vec2 off = nrm * (sideS * halfW * min(miter, 2.0) + wob);
  // Lying flat: the same half-width, in metres at this depth, along the ground across the stroke, then
  // projected. The side that lands on +nrm stays side +1, and the ribbon never closes below the core.
  if (aMod.z > 0.5 && aInk.w > 0.0 && !slid) {
    vec3 lat = cross(aNext - aPrev, vec3(0.0, 1.0, 0.0));
    vec3 lat1 = cross(position - aPrev, vec3(0.0, 1.0, 0.0));
    if (dot(lat, lat) > 1e-12) {
      lat = normalize(lat);
      float miterG = dot(lat1, lat1) > 1e-12 ? 1.0 / max(dot(lat, normalize(lat1)), 0.5) : 1.0;
      float mpp = z / (projectionMatrix[1][1] * 0.5 * uRes.y);
      vec4 ct = projectionMatrix * (mv * vec4(position + lat * halfW * mpp, 1.0));
      if (ct.w > EPS) {
        if (dot(toPx(ct) - s, nrm) < 0.0) lat = -lat;
        vec4 cw = projectionMatrix * (mv * vec4(position + lat * (sideS * halfW * min(miterG, 2.0) + wob) * mpp, 1.0));
        if (cw.w > EPS) {
          vec2 o = toPx(cw) - s;
          float minHalf = min(halfW, 3.2 * sigmaMin + 1.0);
          off = o + nrm * sideS * max(minHalf - sideS * dot(o, nrm), 0.0);
        }
      }
    }
  }
  // Round cap: the extra end point steps outward along the stroke by the half-width; the fragment
  // measures the distance from the end point there, so the end closes in a half disc.
  if (aCap != 0.0) off += dir * aCap * halfW;
  vCap = abs(aCap) * halfW;
  c.xy += off / (0.5 * uRes) * c.w;
  c.z = biasedClipZ(vp, c.w);
  gl_Position = c;

  float fog = fogOf(z);
  float lod = smoothstep(aFade.x, aFade.y, z) * (1.0 - smoothstep(aFade.z, aFade.w, z));
  vColor = aColor * mix(uFogTint, vec3(1.0), fog) * (uGain * aShape.w * aMod.y * energy * fog * lod * taper * (1.0 + 0.6 * aStyle.z * n1));
  vAcross = sideS * halfW;
  vHalfW = halfW;
  vSigma = sigma;
  vHaloPx = haloPx;
  vHalo = aShape.y * energy * vState2.w;
  vU = aU;
  vS = aS;
  vDry = vec4(aInk.xy, seed, aInk.z);
}
`;

const BRUSH_FRAG = /* glsl */ `
${NOISE_GLSL}
varying vec3 vColor;
varying vec4 vDry;
varying float vS;
varying float vAcross;
varying float vHalfW;
varying float vSigma;
varying float vHaloPx;
varying float vHalo;
varying float vU;
varying vec4 vState;   // reveal, gain, soft, head
varying vec4 vState2;  // trail, tail, width, halo
varying float vFrom;   // where the writing starts
varying float vBack;   // reveal on the side toward the start (< 0: same as reveal)
varying float vCap;    // px beyond the end point, inside a round cap
uniform float uInkGrain;
uniform float uInkScale;
uniform float uInkDensity;
uniform float uFrameFade;  // px
uniform vec4 uBand;        // visible picture: left, bottom, right, top (px, from the bottom)
void main() {
  float d = vCap > 0.0 ? length(vec2(vAcross, vCap)) : abs(vAcross);
  // Both terms reach zero at the ribbon edge: no visible border however the ribbon is cut.
  float gauss = exp(-0.5 * d * d / (vSigma * vSigma));
  // Hard edge at the gaussian's half maximum, anti-aliased over ~1.5 px.
  float hw = 1.1774 * vSigma;
  float hard = 1.0 - smoothstep(hw - 0.75, hw + 0.75, d);
  float core = mix(gauss, hard, vDry.w) * (1.0 - smoothstep(0.8 * vHalfW, vHalfW, d));
  float halo = vHalo * exp(-d / max(vHaloPx, 1e-3)) * (1.0 - smoothstep(0.5 * vHalfW, vHalfW, d));
  // 飞白: bristle lanes along the stroke (noise stretched along it), drier toward the end; only where
  // the core is wide enough to show lanes, so thin lines never sparkle.
  if (vDry.x > 0.0) {
    float t = vAcross / max(vSigma, 1e-3);
    float n = 0.65 * snoise(vec2(t * 1.6 + vDry.z, vS / max(vDry.y, 1e-3))) + 0.35 * snoise(vec2(t * 3.7 - vDry.z, vS / max(0.35 * vDry.y, 1e-3)));
    float lanes = smoothstep(-0.3, 0.45, n);
    float dryness = vDry.x * mix(0.35, 1.0, smoothstep(0.2, 1.0, vU)) * smoothstep(1.0, 2.5, vSigma);
    core *= mix(1.0, lanes, dryness);
    halo *= mix(1.0, lanes, 0.4 * dryness);
  }
  float soft = max(vState.b, 1e-4);
  // Distance written from the start point (one-way from 0 by default, both ways from vFrom).
  float u = abs(vU - vFrom);
  float r = (vU < vFrom && vBack >= 0.0) ? vBack : vState.r;
  float shown = 1.0 - smoothstep(r - soft, r, u);
  shown *= mix(exp(-max(r - u, 0.0) / max(vState2.y, 1e-4)), 1.0, vState2.x);
  float headGlow = vState.a * exp(-abs(u - r) / soft);
  vec3 col = vColor * vState.g * ((core + halo) * shown + headGlow * (core + 2.0 * halo));
  if (uFrameFade > 0.0) {
    vec2 p = gl_FragCoord.xy;
    col *= smoothstep(0.0, uFrameFade, min(min(p.x - uBand.x, uBand.z - p.x), min(p.y - uBand.y, uBand.w - p.y)));
  }
#ifdef INK
  // Ink sits unevenly in the paper: the fine tooth (Paper's scale) and a soft pooling mottle. Both are
  // isotropic; the fibres come through on their own, since ink multiplies the paper under it. (A
  // directional fibre term read as scan lines at 4K.)
  vec2 gpx = gl_FragCoord.xy / uInkScale;
  float grain = 0.55 * snoise(gpx / 1.6) + 0.45 * snoise(gpx / 9.0 + 7.3);
  col *= 1.0 + uInkGrain * grain;
  // Beer–Lambert: the stroke's strength is optical depth, so a core saturates to black softly and
  // overlapping strokes add their depths (the blend multiplies the transmittances).
  if (uInkDensity > 0.0) col = 1.0 - exp(-uInkDensity * col);
  col = clamp(col, vec3(0.0), vec3(0.995));
#endif
  gl_FragColor = vec4(col, 1.0);
}
`;

export interface BrushOptions {
  /** Draw order inside the world scene (default 0). */
  renderOrder?: number;
  /** Depth test against occluders that write depth only (default true). */
  depthTest?: boolean;
  /** Pull the depth toward the camera by this fraction of the distance (default 0.004) so a line's halo
   *  is not cut by the ground it lies on. */
  depthBias?: number;
  /** 'light' adds (default); 'ink' multiplies: colours are what the ink absorbs. */
  mode?: InkMode;
  /** Ink mode: how much the ink follows the paper's grain (default 0.3; 0 = flat ink). */
  grain?: number;
  /** Ink mode: optical density k. A stroke absorbs 1 − e^(−k·colour) instead of its colour (clipped):
   *  cores go to a soft full black and shoulders fill out, as ink does on paper. 0 = linear (default);
   *  In linear light a clean black core needs ~5 (2–3 is a dark grey). The ribbon widens with it so
   *  soft skirts never end in a visible edge. */
  density?: number;
  /** Strokes fade out toward the edges of the visible picture (inside the letterbox) over this
   *  fraction of its height (default 0 = off): lines dissolve into the paper at the frame edge instead
   *  of being cut by it. */
  frameFade?: number;
}

export class LightBrush {
  readonly object: THREE.Mesh;
  readonly count: number;
  private readonly material: THREE.ShaderMaterial;
  private readonly state: THREE.DataTexture;
  private readonly stateData: Float32Array;
  private readonly lengths: number[] = [];
  private readonly frameFade: number;

  constructor(
    private readonly ctx: StageContext,
    strokes: Stroke[],
    opts: BrushOptions = {},
  ) {
    this.count = strokes.length;
    this.frameFade = opts.frameFade ?? 0;
    let verts = 0;
    let quads = 0;
    const pts = strokes.map((s) => toPoints(s.points));
    // A round cap is one extra point on each end (same place as the end point, pushed outward on screen).
    const capped = strokes.map((s, k) => s.cap === 'round' && pts[k].length >= 2);
    pts.forEach((p, k) => {
      if (p.length < 2) return;
      const m = p.length + (capped[k] ? 2 : 0);
      verts += m * 2;
      quads += m - 1;
    });
    const pos = new Float32Array(verts * 3);
    const prev = new Float32Array(verts * 3);
    const next = new Float32Array(verts * 3);
    const side = new Float32Array(verts);
    const uu = new Float32Array(verts);
    const ss = new Float32Array(verts);
    const sid = new Float32Array(verts);
    const col = new Float32Array(verts * 3);
    const shape = new Float32Array(verts * 4);
    const style = new Float32Array(verts * 4);
    const extra = new Float32Array(verts * 4);
    const fade = new Float32Array(verts * 4);
    const ink = new Float32Array(verts * 4);
    const mod = new Float32Array(verts * 3);
    const index = new Uint32Array(quads * 6);
    let v = 0;
    let q = 0;
    strokes.forEach((st, k) => {
      const p = pts[k];
      if (p.length < 2) {
        this.lengths.push(0);
        return;
      }
      const cum = [0];
      for (let i = 1; i < p.length; i++) cum.push(cum[i - 1] + p[i].distanceTo(p[i - 1]));
      const L = Math.max(cum[cum.length - 1], 1e-6);
      this.lengths.push(L);
      const c1 = st.color ?? [1.0, 0.62, 0.3];
      const c2 = st.color2 ?? c1;
      const seed = (hashString(st.seed ?? `stroke-${k}`) % 9973) * 0.731;
      const taper = st.taper ?? [0.06, 0.1];
      // smoothstep needs edge0 < edge1; the defaults never fade.
      const fn = st.fadeNear ?? [-2, -1];
      const ff = st.fade ?? [1e8, 2e8];
      const fd = [fn[0], Math.max(fn[1], fn[0] + 1e-3), ff[0], Math.max(ff[1], ff[0] + 1e-3)];
      const base = v;
      const n = p.length;
      const order = capped[k] ? [0, ...Array.from({ length: n }, (_, i) => i), n - 1] : Array.from({ length: n }, (_, i) => i);
      order.forEach((i, j) => {
        const a = p[i];
        const b = i > 0 ? p[i - 1] : a.clone().multiplyScalar(2).sub(p[1]);
        const c = i < p.length - 1 ? p[i + 1] : a.clone().multiplyScalar(2).sub(p[p.length - 2]);
        const u = cum[i] / L;
        for (const sd of [-1, 1]) {
          pos.set([a.x, a.y, a.z], v * 3);
          prev.set([b.x, b.y, b.z], v * 3);
          next.set([c.x, c.y, c.z], v * 3);
          side[v] = sd;
          uu[v] = u;
          ss[v] = cum[i];
          sid[v] = k;
          col.set([c1[0] + (c2[0] - c1[0]) * u, c1[1] + (c2[1] - c1[1]) * u, c1[2] + (c2[2] - c1[2]) * u], v * 3);
          shape.set([st.width ?? 1.2, st.halo ?? 0.18, st.haloPx ?? 7, st.gain ?? 1], v * 4);
          style.set([taper[0], taper[1], st.pressure ?? 0.35, st.wobble ?? 0.35], v * 4);
          extra.set([seed, st.pressureScale ?? 6, st.perspective ?? 0.3, st.refDistance ?? 20], v * 4);
          fade.set(fd, v * 4);
          ink.set([st.dry ?? 0, st.dryScale ?? 8, st.edge ?? 0, st.widthM ?? 0], v * 4);
          mod.set([st.widths?.[i] ?? 1, st.gains?.[i] ?? 1, st.flat ? 1 : 0], v * 3);
          // A cap point is marked in the side (WebGL has no attribute slot left): ±2 start, ±3 end.
          if (capped[k] && (j === 0 || j === order.length - 1)) side[v] = sd * (j === 0 ? 2 : 3);
          v++;
        }
      });
      for (let i = 0; i < order.length - 1; i++) {
        const a = base + i * 2;
        index.set([a, a + 1, a + 2, a + 1, a + 3, a + 2], q * 6);
        q++;
      }
    });

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aPrev', new THREE.BufferAttribute(prev, 3));
    g.setAttribute('aNext', new THREE.BufferAttribute(next, 3));
    g.setAttribute('aSide', new THREE.BufferAttribute(side, 1));
    g.setAttribute('aU', new THREE.BufferAttribute(uu, 1));
    g.setAttribute('aS', new THREE.BufferAttribute(ss, 1));
    g.setAttribute('aStroke', new THREE.BufferAttribute(sid, 1));
    g.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
    g.setAttribute('aShape', new THREE.BufferAttribute(shape, 4));
    g.setAttribute('aStyle', new THREE.BufferAttribute(style, 4));
    g.setAttribute('aExtra', new THREE.BufferAttribute(extra, 4));
    g.setAttribute('aFade', new THREE.BufferAttribute(fade, 4));
    g.setAttribute('aInk', new THREE.BufferAttribute(ink, 4));
    g.setAttribute('aMod', new THREE.BufferAttribute(mod, 3));
    g.setIndex(new THREE.BufferAttribute(index, 1));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e9);

    const W = Math.min(Math.max(this.count, 1), 2048);
    const H = Math.max(1, Math.ceil(Math.max(this.count, 1) / W));
    this.stateData = new Float32Array(W * H * 4 * STATE_TEXELS);
    for (let k = 0; k < W * H; k++) this.stateData.set(STATE_DEFAULT, k * 4 * STATE_TEXELS);
    this.state = new THREE.DataTexture(this.stateData, STATE_TEXELS * W, H, THREE.RGBAFormat, THREE.FloatType);
    this.state.minFilter = THREE.NearestFilter;
    this.state.magFilter = THREE.NearestFilter;
    this.state.needsUpdate = true;

    this.material = new THREE.ShaderMaterial({
      vertexShader: BRUSH_VERT,
      fragmentShader: BRUSH_FRAG,
      uniforms: {
        ...lensUniforms(ctx),
        tState: { value: this.state },
        uStateSize: { value: new THREE.Vector2(W, H) },
        uInkGrain: { value: opts.grain ?? 0.3 },
        uInkScale: { value: ctx.scale },
        uInkDensity: { value: opts.density ?? 0 },
        uFrameFade: { value: 0 },
        uBand: { value: new THREE.Vector4(0, 0, ctx.width, ctx.height) },
      },
      defines: opts.mode === 'ink' ? { INK: 1 } : {},
      ...blendOf(opts.mode ?? 'light'),
      depthTest: opts.depthTest ?? true,
      depthWrite: false,
      transparent: true,
      side: THREE.DoubleSide,
    });
    this.material.uniforms.uDepthBias.value = opts.depthBias ?? 0.004;
    this.object = new THREE.Mesh(g, this.material);
    this.object.frustumCulled = false;
    this.object.renderOrder = opts.renderOrder ?? 0;
  }

  /** Length of stroke i in metres (e.g. to turn a walking speed into a reveal). */
  length(i: number): number {
    return this.lengths[i] ?? 0;
  }

  /** Set the per-frame state of stroke i. Unset fields go back to their defaults, so a second `set` on
   *  the same stroke in a frame undoes the first (use `patch` for that). */
  set(i: number, s: StrokeState): void {
    if (i < 0 || i >= this.count) return;
    this.stateData.set(stateOf(s), i * 4 * STATE_TEXELS);
    this.state.needsUpdate = true;
  }

  /** Change only the given fields of stroke i's state; the others keep their current values (unlike
   *  `set`, which resets them). Use it when two places drive the same stroke in one frame. */
  patch(i: number, s: StrokeState): void {
    if (i < 0 || i >= this.count) return;
    const o = i * 4 * STATE_TEXELS;
    for (let k = 0; k < STATE_KEYS.length; k++) {
      const v = s[STATE_KEYS[k]];
      if (v !== undefined) this.stateData[o + k] = v;
    }
    this.state.needsUpdate = true;
  }

  /** Same state for every stroke. */
  setAll(s: StrokeState): void {
    const v = stateOf(s);
    for (let i = 0; i < this.count; i++) this.stateData.set(v, i * 4 * STATE_TEXELS);
    this.state.needsUpdate = true;
  }

  update(look: InkLook = {}): void {
    applyLook(this.material.uniforms, this.ctx, look);
    if (this.frameFade > 0) {
      // The visible band follows the aspect curve (2.39 opening to 16:9 at 30.1).
      const l = layoutAt(this.ctx.width, this.ctx.height, look.frame ?? 0);
      const u = this.material.uniforms;
      (u.uBand.value as THREE.Vector4).set(0, this.ctx.height - l.visibleBottom, this.ctx.width, this.ctx.height - l.visibleTop);
      u.uFrameFade.value = this.frameFade * (l.visibleBottom - l.visibleTop);
    }
  }

  dispose(): void {
    this.object.geometry.dispose();
    this.material.dispose();
    this.state.dispose();
  }
}

/** StrokeState fields in state-texture order (see STATE_DEFAULT). */
const STATE_KEYS: (keyof StrokeState)[] = ['reveal', 'gain', 'soft', 'head', 'trail', 'tail', 'width', 'halo', 'from', 'back'];

function stateOf(s: StrokeState): number[] {
  const d = STATE_DEFAULT;
  return [s.reveal ?? d[0], s.gain ?? d[1], s.soft ?? d[2], s.head ?? d[3], s.trail ?? d[4], s.tail ?? d[5], s.width ?? d[6], s.halo ?? d[7], s.from ?? d[8], s.back ?? d[9], 0, 0];
}

export interface BristleOptions {
  /** Width of the whole bundle (m), across the stroke in the surface it lies on. */
  widthM: number;
  /** Number of bristle strokes (default 7). */
  count?: number;
  /** Normal of that surface (default +Y: a path on the ground). */
  up?: THREE.Vector3;
  /** Each bristle's width relative to the spacing between bristles (default 1.3: they overlap a little). */
  fill?: number;
  /** Random variation of each bristle's strength (default 0.35) and of its offset (a fifth of that). */
  jitter?: number;
}

/**
 * A wide stroke drawn as a flat brush (排笔): `count` narrow strokes side by side across `widthM`.
 * A single ribbon much wider than its bends are tight folds over and draws shards (a road near the
 * lens); narrow bristles never do, and their uneven strength reads as dry brush. Each bristle keeps
 * the base stroke's settings (taper, dry, pressure, per-point `widths` and `gains` ...) with its own
 * seed; `widths` also scales the spread. Pass the result to one LightBrush.
 */
export function bristles(base: Stroke, opts: BristleOptions): Stroke[] {
  const pts = toPoints(base.points);
  const n = Math.max(1, Math.round(opts.count ?? 7));
  const up = (opts.up ?? new THREE.Vector3(0, 1, 0)).clone().normalize();
  const jitter = opts.jitter ?? 0.35;
  const gap = opts.widthM / n;
  const rng = new Rng(`${base.seed ?? 'bristles'}/bundle`);
  // Lateral direction at each point: across the tangent, in the surface.
  const side: THREE.Vector3[] = [];
  const t = new THREE.Vector3();
  for (let i = 0; i < pts.length; i++) {
    t.subVectors(pts[Math.min(i + 1, pts.length - 1)], pts[Math.max(i - 1, 0)]);
    const l = new THREE.Vector3().crossVectors(t, up);
    side.push(l.lengthSq() > 1e-12 ? l.normalize() : i > 0 ? side[i - 1].clone() : new THREE.Vector3(1, 0, 0));
  }
  const out: Stroke[] = [];
  for (let k = 0; k < n; k++) {
    const o = ((k + 0.5) / n - 0.5) * opts.widthM + (rng.next() - 0.5) * 0.2 * jitter * gap;
    const p = pts.map((q, i) => q.clone().addScaledVector(side[i], o * (base.widths ? base.widths[i] : 1)));
    out.push({ ...base, points: p, widthM: gap * (opts.fill ?? 1.3), gain: (base.gain ?? 1) * (1 - jitter * rng.next()), seed: `${base.seed ?? 'bristles'}/${k}` });
  }
  return out;
}

function toPoints(p: THREE.Vector3[] | ArrayLike<number>): THREE.Vector3[] {
  if (Array.isArray(p) && (p.length === 0 || p[0] instanceof THREE.Vector3)) return p as THREE.Vector3[];
  const a = p as ArrayLike<number>;
  const out: THREE.Vector3[] = [];
  for (let i = 0; i + 2 < a.length; i += 3) out.push(new THREE.Vector3(a[i], a[i + 1], a[i + 2]));
  return out;
}

// ------------------------------------------------------------------------------------------------
// LightPoints

export interface LightPoint {
  position: [number, number, number];
  /** Linear HDR colour (default warm lamp). */
  color?: RGB;
  /** Core size, 1080p px (default 2.2). */
  size?: number;
  gain?: number;
  /** Halo strength (default 0.25) and radius in 1080p px (default 10). */
  halo?: number;
  haloPx?: number;
  /** Cross flare 0..1 (default 0). */
  star?: number;
}

export interface PointsOptions {
  renderOrder?: number;
  depthTest?: boolean;
  /** Shutter (s) for motion streaks from `setVelocities` (default 1/48). */
  shutter?: number;
  /** Pull the depth toward the camera by this fraction of the distance (default 0.02), for points
   *  depth-tested against the scene without an occluder. */
  depthBias?: number;
  /** Decide visibility per point from this occluder's depth pre-pass (and skip the depth test). */
  occluder?: Occluder;
  /** 'light' adds (default); 'ink' multiplies (ink dots, 苔点; stippling on paper). */
  mode?: InkMode;
  /** Sizes (core, halo, flare) are as given at this distance (m) and scale with 1/distance elsewhere,
   *  clamped to 0.02–4x (default 0: fixed pixel sizes). A lantern that shrinks as the camera flies off. */
  refDistance?: number;
  /** 0 = a gaussian glow (default); 1 = a crisp disc of the same core size: pointillist dabs (点彩),
   *  gold flakes, stipple. Halo and flare are unchanged. */
  edge?: number;
  /** Ink mode: optical density, as `BrushOptions.density` (default 0 = linear). */
  density?: number;
}

const POINTS_VERT = /* glsl */ `
${LENS_GLSL}
uniform float uShutter;
uniform float uInkDensity;
uniform float uRefDist;
uniform sampler2D tOcc;
uniform float uOcc;
uniform vec2 uNearFar;
// View distance of the land at a pixel (px from the centre), from the occluder's depth pre-pass.
float landAt(vec2 px) {
  vec2 uv = px / uRes + 0.5;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return 1e9;
  float d = texture2D(tOcc, uv).x;
  return uNearFar.x * uNearFar.y / (uNearFar.y - d * (uNearFar.y - uNearFar.x));
}
attribute vec3 aPos;
attribute vec3 aVel;
attribute vec3 aColor;
attribute vec4 aShape;    // size px, gain, halo gain, halo px
attribute float aStar;
attribute float aDyn;     // per-frame gain
varying vec3 vColor;
varying vec2 vP;          // px from the streak's centre line, in the streak frame
varying float vHalfLen;   // half streak length px
varying float vSigma;
varying float vR;         // bokeh disc radius
varying float vHalo;
varying float vHaloPx;
varying float vStar;
varying float vW0;        // share of energy left in the sharp core
varying float vExt;       // quad half-size across the streak (px)
void main() {
  vec4 vp = modelViewMatrix * vec4(aPos, 1.0);
  if (-vp.z < 0.05) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  float z = max(-vp.z, 1e-3);
  vec4 c = projectionMatrix * vp;
  vec4 c2 = projectionMatrix * (modelViewMatrix * vec4(aPos - aVel * uShutter, 1.0));
  c2.w = max(c2.w, 1e-3);
  vec2 a = toPx(vec4(c.xy, c.z, max(c.w, 1e-3)));
  vec2 b = toPx(c2);
  vec2 axis = b - a;
  float len = length(axis);
  vec2 dir = len > 1e-3 ? axis / len : vec2(1.0, 0.0);
  vec2 nrm = vec2(-dir.y, dir.x);
  vec2 mid = 0.5 * (a + b);

  float coc = cocRadius(z);
  float persp = uRefDist > 0.0 ? clamp(uRefDist / z, 0.02, 4.0) : 1.0;
  float sigma0 = max(0.5 * aShape.x * persp * uScale, 0.5 * uMinPx);
  float w0 = sigma0 * sigma0 / (sigma0 * sigma0 + 0.25 * coc * coc);
  float haloPx = aShape.w * persp * uScale + coc;
  float starLen = aStar * 18.0 * persp * uScale * w0;
  // The halo and the flare are windowed to the quad (fragment shader), so the quad is cut where they
  // are already faint and the window hides the rest: never a square.
  // As in the brush: under ink density the quad reaches past the amplified skirt (no square edge).
  float reach = uInkDensity > 1.0 ? sqrt(2.0 * log(uInkDensity / 0.004)) / 0.8 : 3.2;
  float ext = max(max(reach * sigma0, coc * 1.15), max(aShape.z > 0.0 ? 4.5 * haloPx : 0.0, 1.6 * starLen)) + 1.0;
  float halfLen = 0.5 * len;
  // Quad corners from gl_VertexID-free attribute: position.xy in {-1, 1}.
  vec2 corner = position.xy;
  vec2 offPx = dir * corner.x * (halfLen + ext) + nrm * corner.y * ext;
  vec2 ndcMid = mid / (0.5 * uRes);
  gl_Position = vec4((ndcMid + offPx / (0.5 * uRes)) * c.w, biasedClipZ(vp, c.w), c.w);

  // Seen or hidden as a whole, by five taps of the land's depth around the point (soft at edges):
  // the glow belongs to the lens, so it is never cut by the ground in front of it.
  float seen = 1.0;
  if (uOcc > 0.5) {
    float tol = 0.006 * z + 0.5;
    float t = 1.5 * uScale;
    seen = 0.2 * (step(z - tol, landAt(a)) + step(z - tol, landAt(a + vec2(t, 0.0))) + step(z - tol, landAt(a - vec2(t, 0.0)))
      + step(z - tol, landAt(a + vec2(0.0, t))) + step(z - tol, landAt(a - vec2(0.0, t))));
    if (seen <= 0.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  }

  float fog = fogOf(z);
  vColor = aColor * mix(uFogTint, vec3(1.0), fog) * (uGain * aShape.y * aDyn * fog * seen);
  vP = vec2(corner.x * (halfLen + ext), corner.y * ext);
  vHalfLen = halfLen;
  vSigma = sigma0;
  vR = coc;
  vHalo = aShape.z;
  vHaloPx = haloPx;
  vStar = aStar * w0;
  vW0 = w0;
  vExt = ext;
}
`;

const POINTS_FRAG = /* glsl */ `
uniform float uEdge;
uniform float uInkDensity;
varying vec3 vColor;
varying vec2 vP;
varying float vHalfLen;
varying float vSigma;
varying float vR;
varying float vHalo;
varying float vHaloPx;
varying float vStar;
varying float vW0;
varying float vExt;
void main() {
  // Distance to the streak segment (a point when the streak has no length).
  vec2 q = vec2(max(abs(vP.x) - vHalfLen, 0.0), vP.y);
  float d = length(q);
  float win = 1.0 - smoothstep(0.55 * vExt, vExt, d);
  // Energy of a streak is spread along its length: normalise by (length + core) so it stays constant.
  float spread = vSigma * 2.5 / (vSigma * 2.5 + 2.0 * vHalfLen);
  float core = exp(-0.5 * d * d / (vSigma * vSigma));
  if (uEdge > 0.0) {
    float hw = 1.1774 * vSigma;
    core = mix(core, 1.0 - smoothstep(hw - 0.75, hw + 0.75, d), uEdge);
  }
  float disc = 0.0;
  if (vR > 0.5) {
    float edge = smoothstep(vR, vR - 1.2, d);
    float rim = 0.82 + 0.18 * smoothstep(0.55 * vR, vR, d);
    disc = edge * rim * (2.0 * vSigma * vSigma) / (vR * vR);
  }
  float halo = vHalo * exp(-d / max(vHaloPx, 1e-3));
  float star = 0.0;
  if (vStar > 0.0) {
    vec2 p = vec2(abs(vP.x) - vHalfLen, vP.y);
    star = vStar * 0.35 * (exp(-abs(p.y) / 0.6) * exp(-abs(p.x) / (14.0 * max(vSigma, 0.6))) + exp(-abs(p.x) / 0.6) * exp(-abs(p.y) / (14.0 * max(vSigma, 0.6))));
  }
  float e = (vW0 * core + (1.0 - vW0) * disc) * spread + (halo * spread + star) * win;
  vec3 col = vColor * e;
#ifdef INK
  if (uInkDensity > 0.0) col = 1.0 - exp(-uInkDensity * col);
  col = min(col, vec3(0.995));
#endif
  gl_FragColor = vec4(col, 1.0);
}
`;

export class LightPoints {
  readonly object: THREE.Mesh;
  readonly count: number;
  private readonly material: THREE.ShaderMaterial;
  private readonly aPos: THREE.InstancedBufferAttribute;
  private readonly aVel: THREE.InstancedBufferAttribute;
  private readonly aDyn: THREE.InstancedBufferAttribute;

  private readonly occluder?: Occluder;

  constructor(
    private readonly ctx: StageContext,
    points: LightPoint[],
    opts: PointsOptions = {},
  ) {
    this.occluder = opts.occluder;
    const n = points.length;
    this.count = n;
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 1, -1, 0, -1, 1, 0, 1, 1, 0]), 3));
    g.setIndex([0, 1, 2, 1, 3, 2]);
    const pos = new Float32Array(Math.max(n, 1) * 3);
    const col = new Float32Array(Math.max(n, 1) * 3);
    const shape = new Float32Array(Math.max(n, 1) * 4);
    const star = new Float32Array(Math.max(n, 1));
    const dyn = new Float32Array(Math.max(n, 1)).fill(1);
    points.forEach((p, i) => {
      pos.set(p.position, i * 3);
      col.set(p.color ?? [1.0, 0.7, 0.4], i * 3);
      shape.set([p.size ?? 2.2, p.gain ?? 1, p.halo ?? 0.25, p.haloPx ?? 10], i * 4);
      star[i] = p.star ?? 0;
    });
    this.aPos = new THREE.InstancedBufferAttribute(pos, 3);
    this.aVel = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(n, 1) * 3), 3);
    this.aDyn = new THREE.InstancedBufferAttribute(dyn, 1);
    this.aPos.setUsage(THREE.DynamicDrawUsage);
    this.aVel.setUsage(THREE.DynamicDrawUsage);
    this.aDyn.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('aPos', this.aPos);
    g.setAttribute('aVel', this.aVel);
    g.setAttribute('aDyn', this.aDyn);
    g.setAttribute('aColor', new THREE.InstancedBufferAttribute(col, 3));
    g.setAttribute('aShape', new THREE.InstancedBufferAttribute(shape, 4));
    g.setAttribute('aStar', new THREE.InstancedBufferAttribute(star, 1));
    g.instanceCount = n;
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e9);
    this.material = new THREE.ShaderMaterial({
      vertexShader: POINTS_VERT,
      fragmentShader: POINTS_FRAG,
      uniforms: {
        ...lensUniforms(ctx),
        uShutter: { value: opts.shutter ?? 1 / 48 },
        uRefDist: { value: opts.refDistance ?? 0 },
        uEdge: { value: opts.edge ?? 0 },
        uInkDensity: { value: opts.density ?? 0 },
        uDepthBias: { value: opts.depthBias ?? 0.02 },
        tOcc: { value: opts.occluder?.depth ?? null },
        uOcc: { value: opts.occluder ? 1 : 0 },
        uNearFar: { value: new THREE.Vector2(1, 2) },
      },
      defines: opts.mode === 'ink' ? { INK: 1 } : {},
      ...blendOf(opts.mode ?? 'light'),
      depthTest: opts.occluder ? false : (opts.depthTest ?? true),
      depthWrite: false,
      transparent: true,
    });
    this.object = new THREE.Mesh(g, this.material);
    this.object.frustumCulled = false;
    this.object.renderOrder = opts.renderOrder ?? 0;
  }

  /** Per-frame brightness of every point (length = count). */
  setGains(g: ArrayLike<number>): void {
    (this.aDyn.array as Float32Array).set(g);
    this.aDyn.needsUpdate = true;
  }

  /** Per-frame positions, flat xyz (falling snow, walkers). */
  setPositions(p: ArrayLike<number>): void {
    (this.aPos.array as Float32Array).set(p);
    this.aPos.needsUpdate = true;
  }

  /** Per-frame world velocities (m/s), flat xyz: each point becomes a streak over the shutter. */
  setVelocities(v: ArrayLike<number>): void {
    (this.aVel.array as Float32Array).set(v);
    this.aVel.needsUpdate = true;
  }

  update(look: InkLook = {}): void {
    applyLook(this.material.uniforms, this.ctx, look);
    if (this.occluder) (this.material.uniforms.uNearFar.value as THREE.Vector2).copy(this.occluder.nearFar);
  }

  dispose(): void {
    this.object.geometry.dispose();
    this.material.dispose();
  }
}

// ------------------------------------------------------------------------------------------------
// Occluder

/**
 * The land as geometry that is never seen. `object` goes into the world scene: it writes depth only,
 * so strokes behind a hill are hidden (lines lying on the land are kept: the geometry is lowered by
 * `sink` metres). `update(camera)` renders a depth pre-pass that LightPoints with `occluder` read to
 * decide, per point, whether it is seen. Call `update` in frame() after posing the camera.
 */
export class Occluder {
  readonly object: THREE.Mesh;
  readonly nearFar = new THREE.Vector2(1, 2);
  private readonly pre = new THREE.Scene();
  private readonly target: THREE.WebGLRenderTarget;

  constructor(
    private readonly ctx: StageContext,
    geometry: THREE.BufferGeometry,
    sink = 2.5,
  ) {
    const material = new THREE.MeshBasicMaterial({ colorWrite: false });
    this.object = new THREE.Mesh(geometry, material);
    this.object.position.y = -sink;
    this.object.renderOrder = -100;
    this.object.frustumCulled = false;
    const pre = new THREE.Mesh(geometry, material);
    pre.frustumCulled = false;
    this.pre.add(pre);
    this.target = new THREE.WebGLRenderTarget(ctx.width, ctx.height, { depthBuffer: true });
    this.target.depthTexture = new THREE.DepthTexture(ctx.width, ctx.height, THREE.FloatType);
  }

  get depth(): THREE.DepthTexture {
    return this.target.depthTexture as THREE.DepthTexture;
  }

  update(camera: THREE.PerspectiveCamera): void {
    const r = this.ctx.renderer;
    const prev = r.getRenderTarget();
    camera.updateMatrixWorld();
    r.setRenderTarget(this.target);
    r.clear(false, true, false);
    r.render(this.pre, camera);
    r.setRenderTarget(prev);
    this.nearFar.set(camera.near, camera.far);
  }

  dispose(): void {
    this.object.geometry.dispose();
    (this.object.material as THREE.Material).dispose();
    this.target.dispose();
  }
}

// ------------------------------------------------------------------------------------------------
// Wash

export interface WashBlob {
  /** Centre in screen fractions (x 0..1 left to right, y 0..1 top to bottom). */
  x: number;
  y: number;
  /** Radius as a fraction of the frame height. */
  r: number;
  color: RGB;
  gain?: number;
  /** Horizontal stretch (default 1; >1 = wider). */
  stretch?: number;
}

export interface WashSpec {
  /** Linear colours at the top of the frame, at the line (y = lineY) and at the bottom. */
  top: RGB;
  horizon: RGB;
  bottom: RGB;
  /** Film line in screen fraction (default 0.588). */
  lineY?: number;
  /** Softness of the transition through the line (fraction of height, default 0.18). */
  spread?: number;
  blobs?: WashBlob[];
  gain?: number;
}

const WASH_VERT = /* glsl */ `
void main() { gl_Position = vec4(position.xy, 0.99999, 1.0); }
`;

const WASH_FRAG = /* glsl */ `
uniform vec2 uRes;
uniform vec3 uTop;
uniform vec3 uHorizon;
uniform vec3 uBottom;
uniform float uLineY;
uniform float uSpread;
uniform float uGain;
uniform int uBlobs;
uniform vec4 uBlobA[8];   // x, y, r, stretch
uniform vec4 uBlobB[8];   // rgb * gain
void main() {
  vec2 p = vec2(gl_FragCoord.x / uRes.x, 1.0 - gl_FragCoord.y / uRes.y);
  float above = smoothstep(uLineY + 0.25 * uSpread, uLineY - uSpread, p.y);
  float below = smoothstep(uLineY - 0.25 * uSpread, uLineY + uSpread, p.y);
  float upT = clamp(p.y / max(uLineY, 1e-3), 0.0, 1.0);
  vec3 sky = mix(uTop, uHorizon, upT * upT);
  float dnT = clamp((p.y - uLineY) / max(1.0 - uLineY, 1e-3), 0.0, 1.0);
  vec3 ground = mix(uHorizon, uBottom, sqrt(dnT));
  vec3 col = sky * above + ground * below + uHorizon * (1.0 - above - below);
  float aspect = uRes.x / uRes.y;
  for (int i = 0; i < 8; i++) {
    if (i >= uBlobs) break;
    vec2 d = (p - uBlobA[i].xy) * vec2(aspect / max(uBlobA[i].w, 1e-3), 1.0);
    col += uBlobB[i].rgb * exp(-dot(d, d) / max(uBlobA[i].z * uBlobA[i].z, 1e-6));
  }
  col *= uGain;
#ifdef INK
  col = min(col, vec3(0.995));
#endif
  gl_FragColor = vec4(col, 1.0);
}
`;

export class Wash {
  readonly object: THREE.Mesh;
  private readonly material: THREE.ShaderMaterial;

  /** `mode: 'ink'` on Paper: the colours are what the wash absorbs (淡彩; e.g. [0.3, 0.12, 0.02] tints
   *  the paper indigo-blue, [0.02, 0.12, 0.3] ochre). */
  constructor(ctx: StageContext, opts: { mode?: InkMode } = {}) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e9);
    this.material = new THREE.ShaderMaterial({
      vertexShader: WASH_VERT,
      fragmentShader: WASH_FRAG,
      uniforms: {
        uRes: { value: new THREE.Vector2(ctx.width, ctx.height) },
        uTop: { value: new THREE.Vector3() },
        uHorizon: { value: new THREE.Vector3() },
        uBottom: { value: new THREE.Vector3() },
        uLineY: { value: 0.588 },
        uSpread: { value: 0.18 },
        uGain: { value: 1 },
        uBlobs: { value: 0 },
        uBlobA: { value: Array.from({ length: 8 }, () => new THREE.Vector4()) },
        uBlobB: { value: Array.from({ length: 8 }, () => new THREE.Vector4()) },
      },
      depthTest: false,
      depthWrite: false,
      defines: opts.mode === 'ink' ? { INK: 1 } : {},
      ...blendOf(opts.mode ?? 'light'),
      transparent: true,
    });
    this.object = new THREE.Mesh(g, this.material);
    this.object.frustumCulled = false;
    this.object.renderOrder = -1000;
  }

  update(s: WashSpec): void {
    const u = this.material.uniforms;
    (u.uTop.value as THREE.Vector3).set(...s.top);
    (u.uHorizon.value as THREE.Vector3).set(...s.horizon);
    (u.uBottom.value as THREE.Vector3).set(...s.bottom);
    u.uLineY.value = s.lineY ?? 0.588;
    u.uSpread.value = s.spread ?? 0.18;
    u.uGain.value = s.gain ?? 1;
    const blobs = (s.blobs ?? []).slice(0, 8);
    u.uBlobs.value = blobs.length;
    blobs.forEach((b, i) => {
      (u.uBlobA.value as THREE.Vector4[])[i].set(b.x, b.y, b.r, b.stretch ?? 1);
      const k = b.gain ?? 1;
      (u.uBlobB.value as THREE.Vector4[])[i].set(b.color[0] * k, b.color[1] * k, b.color[2] * k, 0);
    });
  }

  dispose(): void {
    this.object.geometry.dispose();
    this.material.dispose();
  }
}

// ------------------------------------------------------------------------------------------------
// Paper

export interface PaperSpec {
  /** Base colour, linear. */
  color: RGB;
  /** Fibre streaks (rice paper), relative brightness (default 0). */
  fiber?: number;
  /** Low-frequency clouding, relative (default 0). */
  mottle?: number;
  /** Fine tooth, relative (default 0). */
  tooth?: number;
  /** Broad soft highlight (lacquer): centre in screen fractions, radius in frame heights, colour. */
  sheen?: { x: number; y: number; r: number; color: RGB };
  /** Darkening toward the corners, 0..1 (default 0). */
  vignette?: number;
}

/** Rice paper (宣纸): warm off-white with fibres and clouding. */
export const PAPER_RICE: PaperSpec = { color: [1.22, 1.1, 0.86], fiber: 0.045, mottle: 0.045, tooth: 0.016, vignette: 0.2 };
/** Black lacquer (漆): black with red in it, uneven depth and one soft highlight. */
export const PAPER_LACQUER: PaperSpec = {
  color: [0.0105, 0.0046, 0.0036],
  mottle: 0.18,
  tooth: 0.03,
  sheen: { x: 0.32, y: 0.3, r: 0.5, color: [0.016, 0.0085, 0.006] },
  vignette: 0.35,
};

/** Interpolate two grounds (dusk: paper darkening into lacquer). */
export function mixPaper(a: PaperSpec, b: PaperSpec, t: number): PaperSpec {
  const m = (x: number, y: number) => x + (y - x) * t;
  const c = (x: RGB, y: RGB): RGB => [m(x[0], y[0]), m(x[1], y[1]), m(x[2], y[2])];
  const sa = a.sheen ?? { x: 0.5, y: 0.5, r: 0.5, color: [0, 0, 0] as RGB };
  const sb = b.sheen ?? { x: 0.5, y: 0.5, r: 0.5, color: [0, 0, 0] as RGB };
  return {
    color: c(a.color, b.color),
    fiber: m(a.fiber ?? 0, b.fiber ?? 0),
    mottle: m(a.mottle ?? 0, b.mottle ?? 0),
    tooth: m(a.tooth ?? 0, b.tooth ?? 0),
    sheen: { x: m(sa.x, sb.x), y: m(sa.y, sb.y), r: m(sa.r, sb.r), color: c(sa.color, sb.color) },
    vignette: m(a.vignette ?? 0, b.vignette ?? 0),
  };
}

const PAPER_FRAG = /* glsl */ `
${NOISE_GLSL}
uniform vec2 uRes;
uniform float uScale;
uniform float uSeed;
uniform vec3 uColor;
uniform float uFiber;
uniform float uMottle;
uniform float uTooth;
uniform vec4 uSheen;      // x, y, r, -
uniform vec3 uSheenColor;
uniform float uVignette;
void main() {
  // Texture in 1080p pixel units so it looks the same at every resolution; fixed to the frame, like a
  // sheet the picture is painted on.
  vec2 px = gl_FragCoord.xy / uScale;
  vec2 p = vec2(gl_FragCoord.x / uRes.x, 1.0 - gl_FragCoord.y / uRes.y);
  float mottle = 0.6 * snoise(px / 430.0 + uSeed) + 0.3 * snoise(px / 160.0 - uSeed) + 0.1 * snoise(px / 55.0 + 2.0 * uSeed);
  vec2 q1 = mat2(0.94, 0.34, -0.34, 0.94) * px;
  vec2 q2 = mat2(0.5, -0.87, 0.87, 0.5) * px;
  float f1 = snoise(vec2(q1.x / 95.0, q1.y / 1.7) + uSeed);
  float f2 = snoise(vec2(q2.x / 70.0, q2.y / 1.4) - uSeed);
  float fiber = smoothstep(0.55, 0.95, f1) + 0.7 * smoothstep(0.6, 0.95, f2);
  float tooth = snoise(px / 1.6 + 3.0 * uSeed);
  vec3 c = uColor * (1.0 + uMottle * mottle + uTooth * tooth + uFiber * fiber);
  vec2 d = (p - uSheen.xy) * vec2(uRes.x / uRes.y, 1.0);
  c += uSheenColor * exp(-dot(d, d) / max(uSheen.z * uSheen.z, 1e-6));
  vec2 v = p - 0.5;
  c *= 1.0 - uVignette * dot(v, v) * 2.0;
  gl_FragColor = vec4(max(c, vec3(0.0)), 1.0);
}
`;

/**
 * The ground of a chapter that is not light on black: rice paper for ink (draw strokes with
 * `mode: 'ink'` on it), black lacquer for gold (draw gold in light mode, `edge: 1`). Opaque, behind
 * everything, fixed to the frame. `update(spec)` every frame it changes (dusk: mixPaper).
 */
export class Paper {
  readonly object: THREE.Mesh;
  private readonly material: THREE.ShaderMaterial;

  constructor(ctx: StageContext, spec: PaperSpec = PAPER_RICE, seed = 'paper') {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e9);
    this.material = new THREE.ShaderMaterial({
      vertexShader: WASH_VERT,
      fragmentShader: PAPER_FRAG,
      uniforms: {
        uRes: { value: new THREE.Vector2(ctx.width, ctx.height) },
        uScale: { value: ctx.scale },
        uSeed: { value: (hashString(seed) % 997) * 0.173 },
        uColor: { value: new THREE.Vector3() },
        uFiber: { value: 0 },
        uMottle: { value: 0 },
        uTooth: { value: 0 },
        uSheen: { value: new THREE.Vector4(0.5, 0.5, 0.5, 0) },
        uSheenColor: { value: new THREE.Vector3() },
        uVignette: { value: 0 },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.object = new THREE.Mesh(g, this.material);
    this.object.frustumCulled = false;
    this.object.renderOrder = -1100;
    this.update(spec);
  }

  update(spec: PaperSpec): void {
    const u = this.material.uniforms;
    (u.uColor.value as THREE.Vector3).set(...spec.color);
    u.uFiber.value = spec.fiber ?? 0;
    u.uMottle.value = spec.mottle ?? 0;
    u.uTooth.value = spec.tooth ?? 0;
    const sh = spec.sheen;
    (u.uSheen.value as THREE.Vector4).set(sh?.x ?? 0.5, sh?.y ?? 0.5, sh?.r ?? 0.5, 0);
    (u.uSheenColor.value as THREE.Vector3).set(...(sh?.color ?? [0, 0, 0]));
    u.uVignette.value = spec.vignette ?? 0;
  }

  dispose(): void {
    this.object.geometry.dispose();
    this.material.dispose();
  }
}
