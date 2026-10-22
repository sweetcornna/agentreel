// A scene-linear chroma path, before the film post: a take's own grade for dim light in a dark space.
// Every colour is walked along one path by its luminance m, in units of the ink's (the film's dark,
// scene-linear, e.g. sceneFor('#0D0C10')): at and below the ink it is the ink (the unlit space and the
// structure in it); from m0 it greys, by m1 it is a neutral warm grey and by m2 the film's light hue
// (`hue`); above, it hands back to the light's own colour (between back.x and back.y), so the cores
// stay the white they burn to. Why: lit haze is ink plus a little of a warm light, which AgX's toe shows
// magenta to red; walked through the neutral instead, the dim light reads as the light's hue and the
// dark as the ink. `amt` is how much of a pixel's colour the path takes (0 = untouched).
// Grade the parts of a frame that are seen together by their sum (tWith), see GRADE_FRAG.

import * as THREE from 'three';
import { scaled, unitLum, type RGB } from '../lib/color';

export interface GradePath {
  /** The ink: scene-linear colour of the film's unlit dark (as the display must show it). */
  ink: RGB;
  /** Chromaticities (unit luminance) of the stops. */
  grey: RGB;
  hue: RGB;
  /** m0, m1, m2: where the path leaves the ink, is grey, is the hue. */
  m: RGB;
  /** Hand back to the pixel's own colour between these m. */
  back: [number, number];
  amt: number;
}

/** A path for warm light in a dark space: a little less saturated than the light itself (copper, not red). */
export function warmPath(ink: RGB, hue: RGB = [1, 0.42, 0.14], o: Partial<Omit<GradePath, 'ink' | 'hue'>> = {}): GradePath {
  return { ink, grey: unitLum(o.grey ?? [1, 0.8, 0.65]), hue: unitLum(hue), m: o.m ?? [1, 1.3, 2.4], back: o.back ?? [8, 30], amt: o.amt ?? 0.85 };
}

/** tWith (weight uWith, 0 or 1): light that the frame adds to tSrc's (the other half of the haze). The path
 *  is taken at the luminance of the sum; the grade is linear in tSrc at a fixed sum, so grading each part
 *  this way and adding them is the grade of the whole. */
export const GRADE_FRAG = /* glsl */ `
uniform sampler2D tSrc;
uniform sampler2D tWith;
uniform float uWith;
uniform vec3 uInk;
uniform vec3 uGrey;
uniform vec3 uHue;
uniform vec3 uM;
uniform vec2 uBack;
uniform float uAmt;
varying vec2 vUv;
float lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec3 unit(vec3 c) { return c / max(lum(c), 1e-9); }
void main() {
  vec3 c = max(texture2D(tSrc, vUv).rgb, vec3(0.0));
  if (uAmt <= 0.0) { gl_FragColor = vec4(c, 1.0); return; }
  float y = lum(c);
  float yAll = y + uWith * lum(max(texture2D(tWith, vUv).rgb, vec3(0.0)));
  float lm = log2(max(yAll / lum(uInk), 1e-6));
  vec3 l = log2(uM);
  vec3 tgt = unit(mix(unit(uInk), uGrey, smoothstep(l.x, l.y, lm)));
  tgt = unit(mix(tgt, uHue, smoothstep(l.y, l.z, lm)));
  float a = uAmt * (1.0 - smoothstep(log2(uBack.x), log2(uBack.y), lm));
  gl_FragColor = vec4(mix(c, y * tgt, a), 1.0);
}
`;

export function gradeUniforms(): Record<string, THREE.IUniform> {
  return {
    tSrc: { value: null },
    tWith: { value: null },
    uWith: { value: 0 },
    uInk: { value: new THREE.Vector3() },
    uGrey: { value: new THREE.Vector3() },
    uHue: { value: new THREE.Vector3() },
    uM: { value: new THREE.Vector3() },
    uBack: { value: new THREE.Vector2() },
    uAmt: { value: 0 },
  };
}

/** Set the path (amt scaled by `k`, 0..1). `exposure` (stops) is the post's exposure on this frame: m is
 *  measured as the frame will show it, so a flash that lifts the unlit haze takes it to the hue. */
export function setGrade(u: Record<string, THREE.IUniform>, p: GradePath, k: number, exposure = 0): void {
  u.uInk.value.set(...scaled(p.ink, 2 ** -exposure));
  u.uGrey.value.set(...p.grey);
  u.uHue.value.set(...p.hue);
  u.uM.value.set(...p.m);
  u.uBack.value.set(...p.back);
  u.uAmt.value = p.amt * k;
}
