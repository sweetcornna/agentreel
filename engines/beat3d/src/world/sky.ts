// Night sky: gradient + airglow, the Milky Way (galactic-coordinate model with star clouds, a warm
// bulge, dust lanes and faint emission knots), and four layers of faint procedural stars whose density
// follows the Milky Way. Bright stars are a separate catalog (stars.ts). Everything is generated.
//
// The band is built like a photograph of it, not like smoke:
//   - diffuse light is lognormal multi-scale "star clouds" (bright knots, not a grey haze);
//   - the fine texture comes from real sub-pixel stars (layer 4, ~mag 11.5-13.5, strongly
//     concentrated in the band), never from pixel-scale noise, so it reads as film, not sensor;
//   - the bulge is a warm, bright, elongated glow sliced by its own dust lane;
//   - dust: the Great Rift, domain-warped filaments and patchy dark clouds near the core;
//   - faint pink emission knots (Lagoon/Trifid/Eagle-like) next to the core, very low.
//
// The sky is drawn as a full-screen pass before terrain, with depth writes off, so terrain simply
// covers it. Units: linear HDR radiance; the post chain's AgX maps it to display.

import * as THREE from 'three';
import { NOISE_GLSL } from '../gl/noise.glsl';
import { COLOR_GLSL } from '../gl/color.glsl';
import { fbm3 } from '../lib/noise';

/** Direction from azimuth (radians, 0 = -Z, positive to the right/+X) and altitude (radians). */
export function dirAzAlt(az: number, alt: number): THREE.Vector3 {
  return new THREE.Vector3(Math.sin(az) * Math.cos(alt), Math.sin(alt), -Math.cos(az) * Math.cos(alt));
}

export interface GalaxyFrame {
  /** Unit vector towards the galactic centre (l = 0, b = 0). */
  core: THREE.Vector3;
  /** Galactic north pole. */
  pole: THREE.Vector3;
  /** l = +90 deg direction. */
  side: THREE.Vector3;
}

/** Galactic frame from the core direction and any second point on the galactic equator. */
export function galaxyFrame(core: THREE.Vector3, along: THREE.Vector3): GalaxyFrame {
  const c = core.clone().normalize();
  const pole = new THREE.Vector3().crossVectors(c, along).normalize();
  const side = new THREE.Vector3().crossVectors(pole, c).normalize();
  return { core: c, pole, side };
}

/**
 * A winter-night Milky Way for a camera looking down -Z: core low in the south-west (azimuth -26 deg =
 * left of the -Z view axis, 3.5 deg up), the band climbing to the upper right. With a 14 mm lens level
 * on the horizon the core sits at x ~ 0.2 just above it.
 */
export function filmGalaxy(): GalaxyFrame {
  const d = THREE.MathUtils.degToRad;
  return galaxyFrame(dirAzAlt(d(-26), d(3.5)), dirAzAlt(d(34), d(36)));
}

/** GLSL: galactic structure. galaxyEx returns (brightness, dust optical depth, bulge weight) and the emission amount. */
export const GALAXY_GLSL = /* glsl */ `
uniform vec3 uGalC;
uniform vec3 uGalN;
uniform vec3 uGalS;
uniform float uMWDetail;   // 0 = phase-2 look; 1 = sharp-edged filamentary dust (milkyWayStyle.detail)
vec3 galaxyEx(vec3 d, out float neb) {
  float b = asin(clamp(dot(d, uGalN), -1.0, 1.0));
  float l = atan(dot(d, uGalS), dot(d, uGalC));
  float al = abs(l);
  // Band coordinates: seamless in l, stretched along the plane (features ~3x longer than wide).
  vec3 P = vec3(cos(l) * 3.0, sin(l) * 3.0, b * 9.0);
  // Isotropic sky coordinates for small structure: 1 unit = 1/36 rad (1.6 deg).
  vec3 Q = d * 36.0;
  float warp = 0.03 * snoise(P * 0.45 + 4.1) + 0.01 * snoise(P * 1.6 + 2.0);
  float bb = b - warp;
  float thick = mix(0.065, 0.15, exp(-al * al / 0.3));
  float env = exp(-bb * bb / (thick * thick));
  float along = 0.14 + 0.86 * exp(-al * al / 0.6);
  float halo = exp(-bb * bb / (thick * thick * 6.0)) * 0.06 * (0.3 + 0.7 * exp(-al * al / 1.5));

  // Star clouds: lognormal over three scales, so the band is bright knots on a dimmer floor.
  float c1 = fbm3(P * 0.9 + vec3(3.1, 1.7, 8.2), 4);
  float c2 = fbm3(P * 2.7 + vec3(7.3, 2.2, 1.1), 3);
  float c3 = 0.6 * snoise(Q * 1.1 + 5.0) + 0.4 * snoise(Q * 2.6 + 9.0);
  float c4 = 0.6 * snoise(Q * 5.5 + 1.0) + 0.4 * snoise(Q * 11.0 + 4.0);
  float clouds = exp(1.15 * c1 + 0.6 * c2 + 0.4 * c3 + 0.16 * c4) * 0.78;

  // Bulge: an elongated warm glow plus a brighter inner core, both a touch south of the plane.
  float bo = exp(-pow(l / 0.25, 2.0) - pow((bb + 0.01) / 0.14, 2.0));
  float bc = exp(-pow(l / 0.09, 2.0) - pow((bb + 0.006) / 0.06, 2.0));
  float bulge = 0.7 * bo + 0.75 * bc;
  float lum = env * along * clouds + halo + 3.2 * bulge * (0.6 + 0.4 * clouds);

  // Dust. The Great Rift: a long, clumpy dark lane from the core towards +l.
  float riftC = 0.012 + 0.016 * sin(l * 2.3 + 0.4) + 0.008 * snoise(vec2(l * 4.0, 1.3));
  float riftW = 0.016 + 0.03 * smoothstep(-0.1, 1.3, l);
  float rift = exp(-pow((bb - riftC) / riftW, 2.0)) * smoothstep(-0.45, -0.08, l) * (1.0 - smoothstep(1.5, 2.4, l));
  rift *= 0.45 + 0.75 * smoothstep(-0.35, 0.4, fbm3(P * 3.4 + vec3(2.0, 5.0, 1.0), 3));
  // Wisps: domain-warped fbm, elongated along the plane, with ragged finer edges (no ridges:
  // ridged noise reads as cracks in marble).
  vec3 wq = P * vec3(2.4, 2.4, 1.1) + 0.8 * vec3(snoise(P * 1.1 + 1.0), snoise(P * 1.1 + 7.0), snoise(P * 1.1 + 3.0));
  float wn = fbm3(wq + vec3(5.1, 2.2, 9.4), 5);
  float fil = smoothstep(0.06, 0.45, wn) * (0.8 + 0.2 * snoise(Q * 3.0 + 6.0));
  float fil2 = smoothstep(0.2, 0.6, fbm3(Q * 1.6 + vec3(1.7, 3.3, 0.4), 4));
  // Patchy dark clouds, strongest around the core (Pipe / Dark Horse).
  float patches = smoothstep(0.12, 0.5, fbm3(P * 2.2 + vec3(1.3, 7.7, 2.9), 5));
  float inDisk = exp(-bb * bb / (thick * thick * 1.3));
  float nearCore = exp(-al * al / 0.5);
  float tau = 1.5 * rift + (0.7 * fil * (0.5 + 0.5 * fil2) + 1.0 * patches * patches * (0.35 + 0.65 * nearCore)) * inDisk;
  if (uMWDetail > 0.0) {
    // Sharper dust: the same wisps with a steep, eroded edge, and thin dark tendrils running just
    // outside them (a level set of the same field), stretched along the plane like the wisps.
    float wn2 = wn + 0.09 * fbm3(wq * 3.3 + vec3(2.7, 6.1, 0.9), 3) + 0.05 * fbm3(Q * vec3(5.0, 5.0, 5.0) + vec3(1.0, 9.0, 4.0), 3);
    // Sharp edge, graded body: dense inside, a crisp boundary, fibrous (eroded at two finer scales).
    float filS = smoothstep(0.15, 0.2, wn2) * (0.45 + 0.55 * smoothstep(0.2, 0.42, wn));
    float lv = abs(wn2 - 0.08 - 0.03 * snoise(Q * 2.2 + 1.0));
    float tend = (1.0 - smoothstep(0.003, 0.011, lv)) * smoothstep(0.2, 0.6, fil2 + 0.3) * 0.7;
    float tauFil = 0.7 * fil * (0.5 + 0.5 * fil2);
    float tauSharp = 0.85 * filS * (0.55 + 0.45 * fil2) + 0.45 * tend;
    tau += uMWDetail * (tauSharp - tauFil) * inDisk;
  }
  // The bulge's own lane, slicing it just north of the plane.
  float laneY = 0.016 + 0.008 * snoise(vec2(l * 9.0, 3.0));
  tau += 1.1 * exp(-pow((bb - laneY) / 0.013, 2.0)) * exp(-l * l / 0.06) * (0.6 + 0.4 * smoothstep(-0.5, 0.5, snoise(Q * 1.4 + 2.0)));

  // Emission knots (hydrogen): small pinkish clouds just off the core, ragged edges.
  float k1 = exp(-(pow((l - 0.105) / 0.02, 2.0) + pow((bb + 0.022) / 0.013, 2.0)));
  float k2 = exp(-(pow((l - 0.12) / 0.012, 2.0) + pow((bb + 0.0) / 0.01, 2.0)));
  float k3 = exp(-(pow((l - 0.29) / 0.02, 2.0) + pow((bb - 0.012) / 0.016, 2.0)));
  float rag = 0.5 + 0.5 * smoothstep(-0.4, 0.6, fbm3(Q * 6.0 + 3.0, 3));
  neb = (k1 + 0.6 * k2 + 0.7 * k3) * rag + 0.08 * env * exp(-bb * bb / 0.0016) * smoothstep(0.0, 0.6, c2 + 0.3) * nearCore;
  return vec3(lum, tau, bulge);
}
// Back-compatible entry: (brightness, dust optical depth, bulge weight).
vec3 galaxy(vec3 d) { float n; return galaxyEx(d, n); }
`;

const SKY_FRAG = /* glsl */ `
${NOISE_GLSL}
${COLOR_GLSL}
${GALAXY_GLSL}
uniform mat4 uInvViewProj;
uniform vec3 uZenith;
uniform vec3 uHorizon;
uniform vec3 uAirglow;
uniform float uMW;
uniform vec3 uMWColor;
uniform vec3 uMWCore;
uniform float uCoreGain;
uniform float uNebula;
uniform float uMWExt;      // fraction of the extinction applied to the Milky Way's diffuse light
uniform float uFaint;
uniform float uMagLimit;   // reveal: stars fainter than this are invisible (soft 1.5 mag)
uniform float uPxAngle;    // radians per canvas pixel at the centre
uniform float uScale;
uniform float uExtinction; // magnitudes per airmass
uniform vec3 uMWStyle;     // saturation, cos(rotation), sin(rotation)
varying vec2 vUv;

vec3 starTint(float r) {
  // Rough B-V distribution: mostly G/K, some A/B, a few M. Values are linear, luminance-normalised.
  vec3 c;
  if (r < 0.12) c = vec3(0.70, 0.82, 1.20);
  else if (r < 0.30) c = vec3(0.88, 0.93, 1.08);
  else if (r < 0.62) c = vec3(1.04, 1.00, 0.93);
  else if (r < 0.90) c = vec3(1.16, 0.98, 0.78);
  else c = vec3(1.30, 0.90, 0.62);
  return c / dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// One layer of procedural faint stars on a 3D cell lattice of size 1/K around the unit sphere.
vec3 starLayer(vec3 d, float K, float magLo, float magHi, float baseP, float conc, float dens, int seed) {
  vec3 q = d * K;
  ivec3 b = ivec3(floor(q));
  vec3 sum = vec3(0.0);
  float s1080 = 0.5;                                 // PSF sigma, 1080p pixels
  float sig = s1080 * uScale;
  float sEff2 = sig * sig + 1.0 / 12.0;              // + pixel footprint (canvas px)
  float sigA2 = sEff2 * uPxAngle * uPxAngle;         // in radians^2
  float p = baseP * mix(1.0, dens, conc);
  for (int z = -1; z <= 1; z++)
  for (int y = -1; y <= 1; y++)
  for (int x = -1; x <= 1; x++) {
    ivec3 c = b + ivec3(x, y, z);
    vec3 cc = vec3(c) + 0.5;
    if (abs(length(cc) - K) > 1.4) continue;
    vec3 h = hash33i(c + ivec3(seed * 7919, seed * 104729, seed * 1299709));
    if (h.x > p) continue;
    vec3 h2 = hash33i(c * 3 + ivec3(17 + seed, 5, 11));
    vec3 sp = normalize(vec3(c) + h2);
    float cosA = dot(d, sp);
    if (cosA < 0.999) continue;
    float a = length(cross(d, sp));
    float m = mix(magLo, magHi, pow(h.y, 0.6));
    float vis = clamp((uMagLimit - m) / 1.5, 0.0, 1.0);
    // Flux is conserved across resolutions: peak = flux / (2 pi sigma^2) in 1080p pixel units.
    float flux = pow(10.0, -0.4 * (m - 6.0)) * vis;
    sum += starTint(h.z) * flux * exp(-0.5 * a * a / sigA2) / (6.2831853 * sEff2 / (uScale * uScale));
  }
  return sum;
}

void main() {
  vec4 w = uInvViewProj * vec4(vUv * 2.0 - 1.0, 1.0, 1.0);
  vec3 d = normalize(w.xyz / w.w);
  float alt = asin(clamp(d.y, -1.0, 1.0));
  float up = max(d.y, 0.0);
  // Base gradient, brightest at the horizon; airglow band peaks a few degrees up (van Rhijn).
  float t = pow(1.0 - up, 5.0);
  vec3 col = mix(uZenith, uHorizon, t);
  col += uAirglow * (exp(-pow((alt - 0.07) / 0.08, 2.0)) + 0.35 * exp(-max(alt, 0.0) / 0.012));

  // Atmospheric extinction (Kasten & Young airmass), reddening towards the horizon.
  float altDeg = max(degrees(alt), 0.0);
  float X = 1.0 / (sin(max(alt, 0.0)) + 0.50572 * pow(altDeg + 6.07995, -1.6364));
  vec3 ext = pow(vec3(10.0), -0.4 * uExtinction * X * vec3(1.35, 1.0, 0.72));
  vec3 extMW = pow(vec3(10.0), -0.4 * uExtinction * uMWExt * X * vec3(1.35, 1.0, 0.72));

  float neb;
  // milkyWayStyle.rotation: turn the galaxy about the zenith (azimuth), stars' density with it.
  vec3 dg = vec3(uMWStyle.y * d.x - uMWStyle.z * d.z, d.y, uMWStyle.z * d.x + uMWStyle.y * d.z);
  vec3 g = galaxyEx(dg, neb);
  float trans = exp(-g.y * 1.7);
  float densN = clamp(g.x * exp(-g.y * 1.3), 0.0, 1.8);
  if (uMW > 0.0) {
    // Star clouds read warm-neutral, faint outskirts a touch cooler, the bulge golden.
    vec3 mwc = mix(uMWColor * vec3(0.9, 0.96, 1.08), uMWColor, smoothstep(0.15, 0.9, g.x));
    float coreW = clamp(g.z * 1.6, 0.0, 1.0);
    mwc = mix(mwc, uMWCore, coreW);
    vec3 dustRed = mix(vec3(1.0), vec3(1.0, 0.72, 0.5), 1.0 - exp(-g.y * 0.8));
    float lum = g.x + (uCoreGain - 1.0) * 3.2 * g.z;
    vec3 mw = mwc * dustRed * lum * trans;
    mw += vec3(1.0, 0.36, 0.42) * neb * uNebula * (0.4 + 0.6 * trans);
    // Detail: part of the diffuse light goes into resolved faint stars (below).
    mw *= 1.0 - 0.3 * uMWDetail;
    mw = mix(vec3(dot(mw, vec3(0.2126, 0.7152, 0.0722))), mw, uMWStyle.x);
    col += mw * uMW * extMW;
  }
  if (uFaint > 0.0 && alt > -0.01) {
    vec3 s = starLayer(d, 170.0, 6.8, 8.6, 0.42, 0.55, densN, 1)
           + starLayer(d, 300.0, 8.4, 10.2, 0.50, 0.85, densN, 2)
           + starLayer(d, 520.0, 10.0, 11.8, 0.62, 0.95, densN, 3)
           + starLayer(d, 820.0, 11.4, 13.4, 0.75, 1.0, densN * densN * 0.6, 4);
    // Detail: a fifth, dense layer of very faint stars crowded into the star clouds.
    if (uMWDetail > 0.0) s += uMWDetail * 3.0 * starLayer(d, 950.0, 10.4, 12.6, 0.9, 1.0, clamp(densN * densN * 0.8, 0.0, 1.1), 5);
    col += s * uFaint * ext;
  }
  gl_FragColor = vec4(col, 1.0);
}
`;

export interface SkyLook {
  zenith: [number, number, number];
  horizon: [number, number, number];
  airglow: [number, number, number];
  /** Milky Way radiance scale and colours (disk, core). */
  milkyWay: number;
  mwColor: [number, number, number];
  mwCore: [number, number, number];
  /** Gain of the procedural faint stars (same flux scale as the catalog's starGain). */
  faintStars: number;
  extinction: number;
  /** Extra brightness of the bulge relative to the disk (default 1.7). */
  coreGain?: number;
  /** Strength of the pink emission knots (default 0.35; 0 = off). */
  nebula?: number;
  /** Fraction of the extinction applied to the Milky Way's diffuse light (default 0.55). */
  mwExtinction?: number;
  /** Optional Milky Way styling (all default = unchanged): intensity multiplies `milkyWay`
   *  (default 1); saturation of its colour (default 1; 0 = grey); rotation (radians) turns the
   *  galaxy about the zenith (positive = to the right); detail 0..1 (default 0): sharp-edged
   *  filamentary dust and a dense layer of very faint stars in the star clouds. */
  milkyWayStyle?: { intensity?: number; saturation?: number; rotation?: number; detail?: number };
}

/**
 * A night-sky look to start from (a clear winter night far from any town), tuned with the engine's
 * default post and a 14 mm lens.
 */
export const NIGHT_SKY_LOOK: SkyLook = {
  zenith: [0.0019, 0.0026, 0.005],
  horizon: [0.0113, 0.0168, 0.033],
  airglow: [0.0034, 0.0046, 0.0052],
  milkyWay: 0.045,
  mwColor: [0.62, 0.57, 0.47],
  mwCore: [1.0, 0.58, 0.28],
  faintStars: 0.32,
  extinction: 0.18,
  coreGain: 4.2,
  nebula: 0.35,
  mwExtinction: 0.25,
};

export class NightSky {
  readonly mesh: THREE.Mesh;
  readonly material: THREE.ShaderMaterial;
  readonly galaxy: GalaxyFrame;

  constructor(galaxy: GalaxyFrame) {
    this.galaxy = galaxy;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
    this.material = new THREE.ShaderMaterial({
      vertexShader: /* glsl */ `
        varying vec2 vUv;
        void main() { vUv = position.xy * 0.5 + 0.5; gl_Position = vec4(position.xy, 0.9999, 1.0); }`,
      fragmentShader: SKY_FRAG,
      uniforms: {
        uInvViewProj: { value: new THREE.Matrix4() },
        uZenith: { value: new THREE.Vector3() },
        uHorizon: { value: new THREE.Vector3() },
        uAirglow: { value: new THREE.Vector3() },
        uMW: { value: 0 },
        uMWColor: { value: new THREE.Vector3() },
        uMWCore: { value: new THREE.Vector3() },
        uCoreGain: { value: 1.7 },
        uNebula: { value: 0.35 },
        uMWExt: { value: 0.55 },
        uFaint: { value: 0 },
        uMagLimit: { value: 12 },
        uPxAngle: { value: 0.001 },
        uScale: { value: 1 },
        uExtinction: { value: 0.25 },
        uMWStyle: { value: new THREE.Vector3(1, 1, 0) },
        uMWDetail: { value: 0 },
        uGalC: { value: galaxy.core.clone() },
        uGalN: { value: galaxy.pole.clone() },
        uGalS: { value: galaxy.side.clone() },
      },
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    });
    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = -100;
  }

  /** Pose for a camera and a reveal state. magLimit: -2 = nothing, 12 = everything (13.5 = the faintest layer). */
  update(camera: THREE.PerspectiveCamera, look: SkyLook, magLimit: number, gain: number, scale: number, height: number): void {
    const u = this.material.uniforms;
    const rot = new THREE.Matrix4().extractRotation(camera.matrixWorldInverse);
    const vp = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, rot);
    u.uInvViewProj.value.copy(vp).invert();
    u.uZenith.value.set(...look.zenith).multiplyScalar(gain);
    u.uHorizon.value.set(...look.horizon).multiplyScalar(gain);
    u.uAirglow.value.set(...look.airglow).multiplyScalar(gain);
    const mwReveal = THREE.MathUtils.clamp((magLimit - 4) / 5, 0, 1);
    const st = look.milkyWayStyle;
    u.uMW.value = look.milkyWay * (st?.intensity ?? 1) * gain * mwReveal * mwReveal;
    const mwRot = st?.rotation ?? 0;
    u.uMWStyle.value.set(st?.saturation ?? 1, Math.cos(mwRot), Math.sin(mwRot));
    u.uMWDetail.value = st?.detail ?? 0;
    u.uMWColor.value.set(...look.mwColor);
    u.uMWCore.value.set(...look.mwCore);
    u.uCoreGain.value = look.coreGain ?? 1.7;
    u.uNebula.value = look.nebula ?? 0.35;
    u.uMWExt.value = look.mwExtinction ?? 0.55;
    u.uFaint.value = look.faintStars * gain;
    // The last faint layer reaches mag 13.4: once the reveal passes 12 let it all through.
    u.uMagLimit.value = magLimit >= 12 ? 15 : magLimit;
    u.uPxAngle.value = THREE.MathUtils.degToRad(camera.fov) / height;
    u.uScale.value = scale;
    u.uExtinction.value = look.extinction;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.material.dispose();
  }
}

/** CPU estimate of galactic star density at a direction (coarse twin of galaxy() in GLSL). */
export function galaxyDensity(d: THREE.Vector3, g: GalaxyFrame): number {
  const b = Math.asin(THREE.MathUtils.clamp(d.dot(g.pole), -1, 1));
  const l = Math.atan2(d.dot(g.side), d.dot(g.core));
  const al = Math.abs(l);
  const thick = THREE.MathUtils.lerp(0.08, 0.19, Math.exp((-al * al) / 0.35));
  const disk = Math.exp((-b * b) / (thick * thick)) * (0.16 + 0.84 * Math.exp((-al * al) / 0.8));
  const bulge = 0.75 * Math.exp(-((l / 0.27) ** 2) - ((b + 0.012) / 0.15) ** 2) + 0.55 * Math.exp(-((l / 0.1) ** 2) - ((b + 0.008) / 0.07) ** 2);
  const clouds = Math.exp(1.15 * fbm3(Math.cos(l) * 2.7 + 3.1, Math.sin(l) * 2.7 + 1.7, b * 8.1 + 8.2, 3)) * 0.8;
  return disk * clouds + 0.9 * bulge;
}
