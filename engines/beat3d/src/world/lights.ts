// Two character lights (a film can make its protagonists lights): the warm light (amber flame, 3-7 %
// irregular flicker) and the cool light (steady white-cyan, faint cross star). Each one is a world-space
// point light for the ground and a screen-space glow sprite whose size follows distance and lens.
// PALETTE is a neutral night palette (warm and cool lights, a night sky); films pass their own colours.

import * as THREE from 'three';
import { snoise2 } from '../lib/noise';
import type { GlowLook } from './screen';
import type { PointLightSpec } from './terrain';

export const srgb = (hex: string): [number, number, number] => {
  const c = new THREE.Color(hex); // three converts sRGB hex to linear working space
  return [c.r, c.g, c.b];
};

export const PALETTE = {
  warmCore: srgb('#FFF4E0'),
  warmHaloIn: srgb('#FFB45E'),
  warmHaloOut: srgb('#E07A2E'),
  coolCore: srgb('#F4FBFF'),
  coolHaloIn: srgb('#A8DCFF'),
  coolHaloOut: srgb('#4D8FD6'),
  lamp: srgb('#FFC27A'),
  skyZenith: srgb('#05070D'),
  skyHorizon: srgb('#1A2233'),
  milkyWay: srgb('#C8C2B8'),
};

/** Irregular flicker of the warm light, 1 +/- ~5 %: a pure function of (frame, seed). */
export function warmFlicker(frame: number, seed: number): number {
  const t = frame / 24;
  const slow = snoise2(t * 1.7, seed) * 0.032;
  const fast = snoise2(t * 7.3, seed + 19.1) * 0.018;
  const gust = Math.max(0, snoise2(t * 0.6, seed + 41.7)) * 0.02;
  return 1 + slow + fast - gust;
}

export interface CharacterLight {
  /** World position of the flame centre. */
  position: THREE.Vector3;
  /** Overall brightness multiplier (fades, dimming when it hesitates, arrival accumulation). */
  gain: number;
}

/** Pixels per metre at distance d for a vertical fov (degrees) and canvas height. */
export const pxPerMetre = (d: number, fovYDeg: number, height: number): number => height / (2 * Math.tan(THREE.MathUtils.degToRad(fovYDeg) / 2) * d);

/**
 * Size rule for every character light: the core is 2-6 px across at 1080p whatever
 * the distance (1/e radius 1.1-3 px, growing with px-per-metre), the halo 8-12x the core (12x for the
 * smallest core, 8x for the largest), so a light reads as a small intense flame in every shot.
 */
export function lightSizes(ppm: number, scale: number, metres = 0.06): { core: number; halo: number } {
  const core = Math.min(Math.max(metres * ppm, 1.1 * scale), 3.0 * scale);
  const t = (core / scale - 1.1) / 1.9;
  return { core, halo: core * (12 - 4 * t) };
}

/** Glow sprite look of the warm light at a given on-screen scale (px per metre), already x gain. */
export function warmGlowLook(ppm: number, scale: number, gain: number, flicker: number): GlowLook {
  const k = gain * flicker;
  const { core, halo } = lightSizes(ppm, scale);
  return {
    coreRadius: core,
    core: PALETTE.warmCore.map((c) => c * 34 * k) as [number, number, number],
    // In-scattering halo of the night air.
    haloRadius: halo,
    haloInner: PALETTE.warmHaloIn.map((c) => c * 0.75 * k) as [number, number, number],
    haloOuter: PALETTE.warmHaloOut.map((c) => c * 0.42 * k) as [number, number, number],
    extent: 12,
  };
}

/** Point-light record of the warm light for the ground shader. */
export function warmGroundLight(position: THREE.Vector3, gain: number, flicker: number, intensity = 22.0, range?: number): PointLightSpec {
  return { position, color: PALETTE.warmHaloIn, intensity: intensity * gain * flicker, nearRadius: 0.45, range };
}

/** Glow look of the cool light: steady (no flicker), white-cyan, with a faint cross star (十). */
export function coolGlowLook(ppm: number, scale: number, gain: number): GlowLook {
  const { core, halo } = lightSizes(ppm, scale);
  return {
    coreRadius: core,
    core: PALETTE.coolCore.map((c) => c * 36 * gain) as [number, number, number],
    haloRadius: halo,
    haloInner: PALETTE.coolHaloIn.map((c) => c * 0.7 * gain) as [number, number, number],
    haloOuter: PALETTE.coolHaloOut.map((c) => c * 0.4 * gain) as [number, number, number],
    extent: 12,
    star: 0.035,
  };
}

/** Point-light record of the cool light for the ground shader. */
export function coolGroundLight(position: THREE.Vector3, gain: number, intensity = 18.0, range?: number): PointLightSpec {
  return { position, color: PALETTE.coolHaloIn, intensity: intensity * gain, nearRadius: 0.45, range };
}
