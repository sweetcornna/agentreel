// A thin haze in the space: single scattering of a few point lights (up to MAX_LIGHTS), a faint
// ambient of the unlit haze (the film's "ink"), and optionally a rounded ring of light (a leaf, a
// card, a circle: any rounded rectangle) that runs out through it.
//
// Each light is integrated along the pixel's ray by equiangular sampling (stratified, so the result
// is smooth, not noisy) with a Henyey-Greenstein phase and a slowly drifting fBm density: the glow is
// thick where the haze is thick, the volume shows by depth and by light, never as a flat gradient.
// The haze also absorbs (homogeneous extinction): light from far away arrives dimmer and the unlit
// haze saturates to the ink, so what is far sinks into it and comes out of it as the lens nears.
// The ray is split where it leaves through the layer's planes (a glass slab's; a corridor's walls and
// the face at its end: the nearest one the ray leaves through): `front` integrates the camera side,
// `back` the far side, so a layer drawn in between (world/stage.ts) sits inside the haze, not on top.
// With no planes, `front` is the whole ray.

import * as THREE from 'three';
import { mat } from '../post/FilmPost';
import type { Plane } from '../post/LensBlur';
import { scaled, type RGB } from '../lib/color';
import { NOISE_GLSL } from '../gl/noise.glsl';
import { APERTURE_GLSL } from '../gl/aperture.glsl';

export const MAX_LIGHTS = 8;

/** The unlit haze's mean density is about half the nominal (the fBm is clamped low), and the display
 *  toe takes a little more: measured on the frame, this makes a long unlit ray show as the target. */
const AMBIENT_CAL = 2.5;

/** Ambient that makes the unlit haze, integrated out to `far`, show as `target` (e.g. sceneFor(the film's ink hex)). */
export function ambientFor(target: RGB, extinction: number, far = 60): RGB {
  const len = extinction > 1e-4 ? (1 - Math.exp(-extinction * Math.min(far, 70))) / extinction : Math.min(far, 30);
  return scaled(target, AMBIENT_CAL / (0.12 * len));
}

export interface HazeLight {
  position: THREE.Vector3;
  /** Linear colour times intensity. */
  color: [number, number, number];
  /** Softening radius (m): the light's size. */
  radius: number;
}

export interface HazeRing {
  centre: THREE.Vector3;
  /** In-plane axes (unit) and the plane normal. */
  ax: THREE.Vector3;
  ay: THREE.Vector3;
  /** Half-size of the rounded outline (m), its corner radii (m: tl, tr, br, bl) and the band width (m). */
  half: [number, number];
  radii: [number, number, number, number];
  width: number;
  /** Linear colour times intensity. */
  color: [number, number, number];
}

export interface HazeLook {
  lights: HazeLight[];
  ring?: HazeRing | null;
  /** Split planes (world, 1..3, normals toward the camera): the ray splits where it leaves through the nearest. */
  planes: Plane[];
  side: 'front' | 'back';
  /** Mean scattering density (1/m). */
  density: number;
  /** Extinction (1/m): exp(-extinction * distance) of everything seen through the haze. */
  extinction: number;
  /** Ambient of the unlit haze, linear (its in-scatter: what the dark of the space shows as). */
  ambient: [number, number, number];
  /** Seconds, for the drift of the density. */
  time: number;
  /** Far end of the ray (m). */
  far?: number;
}

const FRAG = /* glsl */ `
${NOISE_GLSL}
${APERTURE_GLSL}
#define MAX_LIGHTS ${MAX_LIGHTS}
uniform mat4 uInvProj;
uniform mat4 uCamWorld;
uniform vec3 uCamPos;
uniform int uNL;
uniform vec3 uLP[MAX_LIGHTS];
uniform vec3 uLC[MAX_LIGHTS];
uniform float uLR[MAX_LIGHTS];
uniform int uNP;
uniform vec3 uPlaneP[3];
uniform vec3 uPlaneN[3];
uniform float uSide;      // 0 front, 1 back
uniform float uFar;
uniform float uDensity;
uniform float uExt;
uniform vec3 uAmbient;
uniform float uTime;
uniform vec2 uRes;
uniform float uFrame;
uniform float uRingOn;
uniform vec3 uRingC;
uniform vec3 uRingX;
uniform vec3 uRingY;
uniform vec2 uRingHalf;
uniform vec4 uRingR;      // tr, br, tl, bl
uniform float uRingW;
uniform vec3 uRingCol;
varying vec2 vUv;

float density(vec3 p) {
  vec3 q = p * 1.1 + vec3(0.06, 0.02, -0.035) * uTime;
  float n = fbm3(q, 3);
  return uDensity * clamp(0.5 + 1.1 * n, 0.04, 2.4);
}
float hg(float c, float g) {
  float g2 = g * g;
  return (1.0 - g2) / (12.566371 * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}

void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  vec4 v = uInvProj * vec4(ndc, 1.0, 1.0);
  vec3 dir = normalize(mat3(uCamWorld) * (v.xyz / v.w));
  vec3 o = uCamPos;
  // Where the ray leaves through the nearest plane (camera on each plane's front side).
  float tp = uFar;
  for (int i = 0; i < 3; i++) {
    if (i >= uNP) break;
    float den = dot(uPlaneN[i], dir);
    if (den < -1e-6 && dot(o - uPlaneP[i], uPlaneN[i]) > 0.0) tp = min(tp, dot(uPlaneP[i] - o, uPlaneN[i]) / den);
  }
  float t0 = 0.02, t1 = uFar;
  if (uSide < 0.5) t1 = tp;
  else t0 = tp;
  vec3 col = vec3(0.0);
  if (t1 > t0) {
    // Per-pixel stratification offset (deterministic): smooth, no fixed pattern.
    float jit = hash33i(ivec3(int(gl_FragCoord.x), int(gl_FragCoord.y), int(uFrame) * 3 + 11)).x;
    const int NS = 10;
    for (int i = 0; i < MAX_LIGHTS; i++) {
      if (i >= uNL) break;
      vec3 L = uLP[i];
      float tc = dot(L - o, dir);
      float h2 = max(dot(L - o, L - o) - tc * tc, 0.0) + uLR[i] * uLR[i];
      float h = sqrt(h2);
      float a0 = atan((t0 - tc) / h);
      float a1 = atan((t1 - tc) / h);
      if (a1 - a0 < 1e-5) continue;
      float acc = 0.0;
      for (int k = 0; k < NS; k++) {
        float th = mix(a0, a1, (float(k) + jit) / float(NS));
        float t = tc + h * tan(th);
        vec3 p = o + dir * t;
        // Scattering angle between the light's travel (L -> p) and the way to the eye (-dir).
        float c = dot(normalize(p - L), -dir);
        acc += density(p) * mix(0.0795775, hg(c, 0.8), 0.8) * exp(-uExt * t);
      }
      // f / pdf for equiangular sampling: (a1 - a0) / h per sample, times the light.
      col += uLC[i] * acc * (a1 - a0) / (h * float(NS));
    }
    // Ambient: the unlit haze, a little denser where the fBm is, saturating with distance.
    float te = min(t1, 30.0 + 40.0 * step(1e-4, uExt));
    if (te > t0) {
      float amb = 0.0;
      for (int k = 0; k < 8; k++) {
        float t = mix(t0, te, (float(k) + jit) / 8.0);
        amb += density(o + dir * t) * exp(-uExt * t);
      }
      col += uAmbient * amb * (te - t0) / 8.0 / max(uDensity, 1e-5) * 0.12;
    }
    // The ring: emission in a thin slab around its plane (back side only, where the plane is).
    if (uRingOn > 0.5) {
      vec3 rn = normalize(cross(uRingX, uRingY));
      float rd = dot(rn, dir);
      float tr = abs(rd) > 1e-5 ? dot(uRingC - o, rn) / rd : -1.0;
      if (tr > t0 - 0.001 && tr < t1 + 0.001) {
        vec3 p = o + dir * tr - uRingC;
        vec2 q = vec2(dot(p, uRingX), dot(p, uRingY));
        float sd = sdRoundRect(q, uRingHalf, uRingR);
        float band = exp(-sd * sd / (uRingW * uRingW));
        float slab = min(0.06 / max(abs(rd), 0.12), 0.5);
        col += uRingCol * band * slab * density(o + dir * tr) / max(uDensity, 1e-5) * exp(-uExt * tr);
      }
    }
  }
  gl_FragColor = vec4(col, 1.0);
}
`;

export class Haze {
  private readonly material: THREE.ShaderMaterial;
  private readonly quad: THREE.Mesh;
  private readonly scene = new THREE.Scene();
  private readonly cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  constructor(
    private readonly renderer: THREE.WebGLRenderer,
    width: number,
    height: number,
  ) {
    this.material = mat(
      FRAG,
      {
        uInvProj: { value: new THREE.Matrix4() },
        uCamWorld: { value: new THREE.Matrix4() },
        uCamPos: { value: new THREE.Vector3() },
        uNL: { value: 0 },
        uLP: { value: Array.from({ length: MAX_LIGHTS }, () => new THREE.Vector3()) },
        uLC: { value: Array.from({ length: MAX_LIGHTS }, () => new THREE.Vector3()) },
        uLR: { value: new Array(MAX_LIGHTS).fill(0.01) },
        uNP: { value: 1 },
        uPlaneP: { value: [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()] },
        uPlaneN: { value: [new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 1)] },
        uSide: { value: 0 },
        uFar: { value: 60 },
        uDensity: { value: 0.03 },
        uExt: { value: 0 },
        uAmbient: { value: new THREE.Vector3() },
        uTime: { value: 0 },
        uRes: { value: new THREE.Vector2(width, height) },
        uFrame: { value: 0 },
        uRingOn: { value: 0 },
        uRingC: { value: new THREE.Vector3() },
        uRingX: { value: new THREE.Vector3(1, 0, 0) },
        uRingY: { value: new THREE.Vector3(0, 1, 0) },
        uRingHalf: { value: new THREE.Vector2(1, 1) },
        uRingR: { value: new THREE.Vector4() },
        uRingW: { value: 0.01 },
        uRingCol: { value: new THREE.Vector3() },
      },
      THREE.AdditiveBlending,
    );
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
    this.quad = new THREE.Mesh(g, this.material);
    this.quad.frustumCulled = false;
    this.scene.add(this.quad);
  }

  /** Add the haze for `look` into the current contents of `target`. */
  render(target: THREE.WebGLRenderTarget, camera: THREE.PerspectiveCamera, look: HazeLook, frame: number): void {
    const u = this.material.uniforms;
    u.uInvProj.value.copy(camera.projectionMatrixInverse);
    u.uCamWorld.value.copy(camera.matrixWorld);
    u.uCamPos.value.setFromMatrixPosition(camera.matrixWorld);
    const n = Math.min(look.lights.length, MAX_LIGHTS);
    u.uNL.value = n;
    for (let i = 0; i < n; i++) {
      const l = look.lights[i];
      u.uLP.value[i].copy(l.position);
      u.uLC.value[i].set(...l.color);
      u.uLR.value[i] = l.radius;
    }
    const planes = look.planes.slice(0, 3);
    u.uNP.value = planes.length;
    planes.forEach((pl, i) => {
      u.uPlaneP.value[i].copy(pl.point);
      u.uPlaneN.value[i].copy(pl.normal).normalize();
    });
    u.uSide.value = look.side === 'front' ? 0 : 1;
    u.uFar.value = look.far ?? 60;
    u.uDensity.value = look.density;
    u.uExt.value = look.extinction;
    u.uAmbient.value.set(...look.ambient);
    u.uTime.value = look.time;
    u.uFrame.value = Math.floor(frame);
    const r = look.ring;
    u.uRingOn.value = r && look.side === 'back' ? 1 : 0;
    if (r) {
      u.uRingC.value.copy(r.centre);
      u.uRingX.value.copy(r.ax);
      u.uRingY.value.copy(r.ay);
      u.uRingHalf.value.set(r.half[0], r.half[1]);
      // sdRoundRect order: tr, br, tl, bl.
      u.uRingR.value.set(r.radii[1], r.radii[2], r.radii[0], r.radii[3]);
      u.uRingW.value = r.width;
      u.uRingCol.value.set(...r.color);
    }
    this.renderer.setRenderTarget(target);
    this.renderer.render(this.scene, this.cam);
  }

  dispose(): void {
    this.material.dispose();
    this.quad.geometry.dispose();
  }
}
