// Shaders of the film post chain. All passes draw one full-screen triangle.

import { COLOR_GLSL } from '../gl/color.glsl';
import { NOISE_GLSL } from '../gl/noise.glsl';

export const FS_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

/** 13-tap downsample (Jimenez 2014). First level: optional soft threshold + Karis average against fireflies. */
export const DOWN13_FRAG = /* glsl */ `
${COLOR_GLSL}
uniform sampler2D tSrc;
uniform vec2 uTexel;       // source texel size
uniform float uKaris;      // 1 on the first downsample
uniform float uThreshold;  // <= 0 disables
uniform float uKnee;
varying vec2 vUv;
vec3 s(vec2 o) { return texture2D(tSrc, vUv + o * uTexel).rgb; }
vec3 soft(vec3 c) {
  if (uThreshold <= 0.0) return c;
  float br = max(c.r, max(c.g, c.b));
  float k = max(uKnee * uThreshold, 1e-4);
  float rq = clamp(br - uThreshold + k, 0.0, 2.0 * k);
  rq = rq * rq / (4.0 * k);
  return c * max(rq, br - uThreshold) / max(br, 1e-5);
}
float kw(vec3 c) { return 1.0 / (1.0 + luma709(c)); }
void main() {
  vec3 a = s(vec2(-2.0, 2.0)), b = s(vec2(0.0, 2.0)), c = s(vec2(2.0, 2.0));
  vec3 d = s(vec2(-2.0, 0.0)), e = s(vec2(0.0, 0.0)), f = s(vec2(2.0, 0.0));
  vec3 g = s(vec2(-2.0, -2.0)), h = s(vec2(0.0, -2.0)), i = s(vec2(2.0, -2.0));
  vec3 j = s(vec2(-1.0, 1.0)), k = s(vec2(1.0, 1.0)), l = s(vec2(-1.0, -1.0)), m = s(vec2(1.0, -1.0));
  vec3 g0 = (j + k + l + m) * 0.25;
  vec3 g1 = (a + b + d + e) * 0.25;
  vec3 g2 = (b + c + e + f) * 0.25;
  vec3 g3 = (d + e + g + h) * 0.25;
  vec3 g4 = (e + f + h + i) * 0.25;
  vec3 outc;
  if (uKaris > 0.5) {
    float w0 = kw(g0) * 0.5, w1 = kw(g1) * 0.125, w2 = kw(g2) * 0.125, w3 = kw(g3) * 0.125, w4 = kw(g4) * 0.125;
    outc = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
  } else {
    outc = g0 * 0.5 + (g1 + g2 + g3 + g4) * 0.125;
  }
  gl_FragColor = vec4(soft(outc), 1.0);
}
`;

/** 3x3 tent upsample of the coarser level, added to this level's own content times a weight. */
export const UP_FRAG = /* glsl */ `
uniform sampler2D tLow;
uniform sampler2D tCur;
uniform vec2 uLowTexel;
uniform float uCurWeight;
uniform float uLowWeight;
varying vec2 vUv;
void main() {
  vec2 t = uLowTexel;
  vec3 up = texture2D(tLow, vUv + vec2(-t.x, t.y)).rgb + 2.0 * texture2D(tLow, vUv + vec2(0.0, t.y)).rgb + texture2D(tLow, vUv + vec2(t.x, t.y)).rgb
          + 2.0 * texture2D(tLow, vUv + vec2(-t.x, 0.0)).rgb + 4.0 * texture2D(tLow, vUv).rgb + 2.0 * texture2D(tLow, vUv + vec2(t.x, 0.0)).rgb
          + texture2D(tLow, vUv + vec2(-t.x, -t.y)).rgb + 2.0 * texture2D(tLow, vUv + vec2(0.0, -t.y)).rgb + texture2D(tLow, vUv + vec2(t.x, -t.y)).rgb;
  up /= 16.0;
  gl_FragColor = vec4(texture2D(tCur, vUv).rgb * uCurWeight + up * uLowWeight, 1.0);
}
`;

/** Threshold + tint (halation and streak prefilters). */
export const THRESH_FRAG = /* glsl */ `
${COLOR_GLSL}
uniform sampler2D tSrc;
uniform float uThreshold;
uniform vec3 uTint;
uniform vec2 uTexel;
uniform float uDespeckle;  // 0 = off; 1 = threshold the 5-tap cross mean (drops 1-px glints, keeps real lights)
varying vec2 vUv;
void main() {
  vec3 c = texture2D(tSrc, vUv).rgb;
  if (uDespeckle > 0.0) {
    vec3 m = (c + texture2D(tSrc, vUv + vec2(uTexel.x, 0.0)).rgb + texture2D(tSrc, vUv - vec2(uTexel.x, 0.0)).rgb
              + texture2D(tSrc, vUv + vec2(0.0, uTexel.y)).rgb + texture2D(tSrc, vUv - vec2(0.0, uTexel.y)).rgb) * 0.2;
    c = mix(c, m, uDespeckle);
  }
  float br = max(c.r, max(c.g, c.b));
  float k = 0.5 * uThreshold;
  float rq = clamp(br - uThreshold + k, 0.0, 2.0 * k);
  rq = rq * rq / (4.0 * k + 1e-5);
  c *= max(rq, br - uThreshold) / max(br, 1e-5);
  // Halation lives in the red-sensitive layer: weight by the red-ish energy of the source.
  gl_FragColor = vec4(c * uTint, 1.0);
}
`;

/** Horizontal-only 4-tap downsample (anamorphic streak chain). */
export const HDOWN_FRAG = /* glsl */ `
uniform sampler2D tSrc;
uniform vec2 uTexel;
varying vec2 vUv;
void main() {
  float t = uTexel.x;
  vec3 c = texture2D(tSrc, vUv + vec2(-1.5 * t, 0.0)).rgb * 0.125
         + texture2D(tSrc, vUv + vec2(-0.5 * t, 0.0)).rgb * 0.375
         + texture2D(tSrc, vUv + vec2(0.5 * t, 0.0)).rgb * 0.375
         + texture2D(tSrc, vUv + vec2(1.5 * t, 0.0)).rgb * 0.125;
  gl_FragColor = vec4(c, 1.0);
}
`;

/** Horizontal tent upsample with falloff (streak chain). */
export const HUP_FRAG = /* glsl */ `
uniform sampler2D tLow;
uniform sampler2D tCur;
uniform vec2 uLowTexel;
uniform float uFalloff;
varying vec2 vUv;
void main() {
  float t = uLowTexel.x;
  vec3 up = (texture2D(tLow, vUv + vec2(-t, 0.0)).rgb + 2.0 * texture2D(tLow, vUv).rgb + texture2D(tLow, vUv + vec2(t, 0.0)).rgb) * 0.25;
  gl_FragColor = vec4(texture2D(tCur, vUv).rgb + up * uFalloff, 1.0);
}
`;

/** Accumulate a motion-blur sub-sample: additive blending does the sum, this scales it. */
export const COPY_FRAG = /* glsl */ `
uniform sampler2D tSrc;
uniform float uWeight;
varying vec2 vUv;
void main() { gl_FragColor = vec4(texture2D(tSrc, vUv).rgb * uWeight, 1.0); }
`;

/**
 * Depth of field: single-pass spiral gather (after D. Gustafsson's "bokeh in a single pass").
 * CoC from the thin-lens model on linear depth. Background samples are clamped to twice the centre
 * CoC so sharp foreground edges do not get eaten by the blurred background behind them.
 */
export const DOF_FRAG = /* glsl */ `
uniform sampler2D tSrc;
uniform sampler2D tDepth;
uniform vec2 uRes;
uniform float uNear;
uniform float uFar;
uniform float uFocus;      // metres
uniform float uCocK;       // CoC diameter in px = uCocK * |z - focus| / z
uniform float uMaxR;       // max CoC radius behind the focus plane, canvas px
uniform float uMaxRNear;   // max CoC radius in front of the focus plane, canvas px
uniform float uStep;       // spiral step, canvas px
varying vec2 vUv;
float linDepth(float d) {
  float z = d * 2.0 - 1.0;
  return 2.0 * uNear * uFar / (uFar + uNear - z * (uFar - uNear));
}
float cocR(float z) { return min(0.5 * uCocK * abs(z - uFocus) / max(z, 1e-3), z < uFocus ? uMaxRNear : uMaxR); }
void main() {
  float zc = linDepth(texture2D(tDepth, vUv).r);
  float cc = cocR(zc);
  vec3 color = texture2D(tSrc, vUv).rgb;
  float tot = 1.0;
  float radius = uStep;
  float ang = 0.0;
  for (int i = 0; i < 1024; i++) {
    if (radius >= max(uMaxR, uMaxRNear)) break;
    vec2 uv = vUv + vec2(cos(ang), sin(ang)) * radius / uRes;
    vec3 sc = texture2D(tSrc, uv).rgb;
    float zs = linDepth(texture2D(tDepth, uv).r);
    float cs = cocR(zs);
    if (zs > zc) cs = clamp(cs, 0.0, cc * 2.0);
    float m = smoothstep(radius - 0.5, radius + 0.5, cs);
    color += mix(color / tot, sc, m);
    tot += 1.0;
    radius += uStep / radius;
    ang += 2.39996323;
  }
  gl_FragColor = vec4(color / tot, 1.0);
}
`;

export const COMPOSITE_FRAG = /* glsl */ `
${COLOR_GLSL}
${NOISE_GLSL}
uniform sampler2D tScene;
uniform sampler2D tBloom;
uniform sampler2D tHal;
uniform sampler2D tStreak;
uniform sampler2D tOverlay;
uniform float uHasOverlay;
uniform vec2 uRes;
uniform float uScale;
uniform float uBloomMix;
uniform float uBloomNorm;
uniform vec3 uBloomTint;
uniform float uHalAmount;
uniform float uHalNorm;
uniform float uStreakAmount;
uniform float uStreakNorm;
uniform vec3 uStreakTint;
uniform float uExposure;
uniform vec3 uSlope;
uniform vec3 uOffset;
uniform vec3 uPower;
uniform float uLookSat;
uniform vec3 uLift;
uniform vec3 uGamma;
uniform vec3 uGain;
uniform float uSat;
uniform float uVignette;
uniform float uVigK;
uniform float uCA;
uniform float uGrain;
uniform float uGrainSize;
uniform float uGrainChroma;
uniform float uFrame;
uniform float uSeed;
uniform vec2 uBand;        // visible band [top, bottom) in pixels from the top
uniform float uLetterbox;
uniform float uOverlayClean; // grain and dither x (1 - this * overlay alpha)
uniform float uCAWarm;     // 1 = warm-only fringe, 0 = classic RGB split
uniform float uNegative;   // 0..1: light on black -> ink on paper
uniform vec3 uNegPaper;
uniform vec3 uNegInk;
uniform vec4 uNegWipe;     // centre (px from the top-left), radius and soft edge (px); radius < 0 = off
varying vec2 vUv;

// Smoothly interpolated lattice noise with ~unit variance; lattice step = one grain cell.
float grainChannel(vec2 p, int ch) {
  vec2 i = floor(p);
  vec2 f = p - i;
  f = f * f * (3.0 - 2.0 * f);
  int fr = int(uFrame) * 4 + ch + int(uSeed);
  vec3 h00 = hash33i(ivec3(int(i.x), int(i.y), fr));
  vec3 h10 = hash33i(ivec3(int(i.x) + 1, int(i.y), fr));
  vec3 h01 = hash33i(ivec3(int(i.x), int(i.y) + 1, fr));
  vec3 h11 = hash33i(ivec3(int(i.x) + 1, int(i.y) + 1, fr));
  // Sum of three uniforms -> approx. gaussian, sd 0.5; x2 -> sd 1.
  float n00 = (h00.x + h00.y + h00.z - 1.5) * 2.0;
  float n10 = (h10.x + h10.y + h10.z - 1.5) * 2.0;
  float n01 = (h01.x + h01.y + h01.z - 1.5) * 2.0;
  float n11 = (h11.x + h11.y + h11.z - 1.5) * 2.0;
  float n = mix(mix(n00, n10, f.x), mix(n01, n11, f.x), f.y);
  return n * 1.55; // restore variance lost to interpolation
}

void main() {
  vec2 frag = gl_FragCoord.xy;
  float yTop = uRes.y - frag.y;
  if (uLetterbox > 0.5 && (yTop < uBand.x || yTop > uBand.y)) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }
  vec2 uv = frag / uRes;

  // Lateral chromatic aberration grows with r^2 towards the corners. Warm style: the long wavelengths
  // image larger and spill outward (red, then orange) over the edge; nothing is taken away, so the
  // fringe never turns cyan or blue. RGB style: the classic split, red out, blue in.
  vec2 cpx = (uv - 0.5) * uRes;
  float rn = length(cpx) / (0.5 * length(uRes));
  vec2 off = (cpx / max(length(cpx), 1e-3)) * uCA * uScale * rn * rn / uRes;
  vec3 col = texture2D(tScene, uv).rgb;
  if (uCAWarm > 0.5) {
    col.r = max(col.r, texture2D(tScene, uv - off).r);
    col.g = max(col.g, texture2D(tScene, uv - 0.45 * off).g * 0.8);
  } else {
    col = vec3(texture2D(tScene, uv - off).r, col.g, texture2D(tScene, uv + off).b);
  }

  vec3 bloom = texture2D(tBloom, uv).rgb * uBloomNorm * uBloomTint;
  col = mix(col, bloom, uBloomMix);
  col += texture2D(tHal, uv).rgb * uHalNorm * uHalAmount;
  col += texture2D(tStreak, uv).rgb * uStreakNorm * uStreakAmount * uStreakTint;

  col *= exp2(uExposure);
  // The lift is the lens's veiling glare: with no light at all there is none, so a frame with nothing
  // in it (a cold open in the dark) stays the true black of the letterbox.
  float anyLight = smoothstep(0.0, 1e-4, max(col.r, max(col.g, col.b)));
  vec3 a = agxEncode(col);
  a = agxLook(a, uSlope, uOffset, uPower, uLookSat);
  vec3 d = agxDecode(a);
  d = liftGammaGain(d, uLift * anyLight, uGamma, uGain);
  d = saturate709(d, uSat);

  // Natural (cos^4-like) falloff measured on the visible band, corner-normalised.
  float bandH = uBand.y - uBand.x;
  vec2 vc = vec2((frag.x - 0.5 * uRes.x) / (0.5 * uRes.x), (yTop - 0.5 * (uBand.x + uBand.y)) / (0.5 * bandH));
  vc.y *= bandH / uRes.x;
  float r = length(vc) / length(vec2(1.0, bandH / uRes.x));
  float t = r * uVigK;
  float falloff = 1.0 / ((1.0 + t * t) * (1.0 + t * t));
  d *= mix(1.0, falloff, uVignette);

  // Negative to positive: black becomes paper, light becomes ink (sqrt so faint lines still read).
  if (uNegative > 0.0) {
    float Ld = dot(d, vec3(0.2126, 0.7152, 0.0722));
    vec3 inv = mix(uNegPaper, uNegInk, smoothstep(0.0, 0.55, sqrt(max(Ld, 0.0))));
    float negAmt = uNegative;
    // Radial wipe: inside the circle the picture is already the positive, outside still the negative.
    if (uNegWipe.z >= 0.0) {
      float rr = length(vec2(frag.x, yTop) - uNegWipe.xy);
      negAmt *= 1.0 - smoothstep(uNegWipe.z - 0.5 * uNegWipe.w, uNegWipe.z + 0.5 * uNegWipe.w, rr);
    }
    d = mix(d, inv, negAmt);
  }

  float clean = 1.0;
  if (uHasOverlay > 0.5) {
    vec4 o = texture2D(tOverlay, uv);
    d = mix(d, srgbToLinear(o.rgb), o.a);
    clean = 1.0 - uOverlayClean * o.a;
  }

  vec3 s = linearToSrgb(d);

  // Grain: luminance-dependent, deterministic per frame, sized in 1080p pixels.
  float L = luma709(s);
  // Like a negative scanned well: grain lives in the midtones, fades in deep shadow and in highlights.
  float resp = mix(0.18, 1.0, smoothstep(0.05, 0.32, L)) * (1.0 - 0.75 * smoothstep(0.42, 0.92, L));
  vec2 gp = frag / (uGrainSize * uScale);
  float gm = grainChannel(gp, 0);
  vec3 gc = vec3(grainChannel(gp + 37.0, 1), grainChannel(gp + 71.0, 2), grainChannel(gp + 113.0, 3));
  vec3 g = mix(vec3(gm), gc, uGrainChroma);
  // True black stays true black (a cold open and a cut to black match the letterbox exactly);
  // from about one code value up, grain and dither are at full strength.
  float lit = smoothstep(0.0, 1.5 / 255.0, max(s.r, max(s.g, s.b)));
  s += g * uGrain * resp * lit * clean;

  // Triangular dither, one 8-bit code value: no banding in the dark gradients of the haze.
  vec3 h1 = hash33i(ivec3(int(frag.x), int(frag.y), int(uFrame) + 7919));
  s += (h1.x + h1.y - 1.0) / 255.0 * lit * clean;
  gl_FragColor = vec4(clamp(s, 0.0, 1.0), 1.0);
}
`;
