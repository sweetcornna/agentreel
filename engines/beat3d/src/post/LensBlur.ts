// Layer defocus through the film's lens aperture (lib/aperture.ts). A program renders a surface (a glass
// slab, a wall of beams) into its own premultiplied-RGBA HDR layer and calls apply(): every pixel
// gathers the layer through the aperture kernel scaled by its own circle of confusion, with the CoC taken from the thin lens
// at the depth where the pixel's ray meets the layer's plane. Because the plane extends past the
// surface, the blurred silhouette spreads correctly with alpha and composites over whatever is behind
// (no gather halos). A layer may lie on up to three planes (a corridor's two walls and a gate at its
// end): each pixel takes the nearest plane its ray leaves through, so a box of planes works as one.
// Kernel = points filling the unit-area aperture, weighted by the same aperture profile as the point
// bokeh, so a highlight on the surface defocuses into the same shape as a lamp.
// Sampling cost stays flat: each kernel tap reads a pre-filtered level of the layer whose texel matches
// the tap spacing (two levels blended), so wide blurs are smooth without thousands of taps.

import * as THREE from 'three';
import { FS_VERT } from './shaders';
import { hdrTarget, mat } from './FilmPost';
import { APERTURE_GLSL } from '../gl/aperture.glsl';
import { apertureExtent, apertureProfile, apertureSd, defaultAperture, type ApertureSpec } from '../lib/aperture';
import { cocK, type LensState } from '../lib/lens';

const LEVELS = 5;

const DOWN_FRAG = /* glsl */ `
uniform sampler2D tSrc;
uniform vec2 uTexel;
varying vec2 vUv;
vec4 s(vec2 o) { return texture2D(tSrc, vUv + o * uTexel); }
void main() {
  vec4 a = s(vec2(-2.0, 2.0)), b = s(vec2(0.0, 2.0)), c = s(vec2(2.0, 2.0));
  vec4 d = s(vec2(-2.0, 0.0)), e = s(vec2(0.0, 0.0)), f = s(vec2(2.0, 0.0));
  vec4 g = s(vec2(-2.0, -2.0)), h = s(vec2(0.0, -2.0)), i = s(vec2(2.0, -2.0));
  vec4 j = s(vec2(-1.0, 1.0)), k = s(vec2(1.0, 1.0)), l = s(vec2(-1.0, -1.0)), m = s(vec2(1.0, -1.0));
  gl_FragColor = (j + k + l + m) * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + e * 0.125;
}
`;

const GATHER_FRAG = /* glsl */ `
${APERTURE_GLSL}
uniform sampler2D tL0;
uniform sampler2D tL1;
uniform sampler2D tL2;
uniform sampler2D tL3;
uniform sampler2D tL4;
uniform sampler2D tL5;
uniform sampler2D tKernel;
uniform int uN;
uniform vec2 uRes;
uniform vec2 uProj;        // projectionMatrix[0][0], [1][1]
uniform int uNP;           // planes in use (1..3)
uniform vec3 uPlaneN[3];   // view-space planes: dot(n, p) = d, n toward the camera
uniform float uPlaneD[3];
uniform float uCocK;
uniform float uFocus;
uniform float uMaxR;
uniform float uFixedR;     // >= 0: one CoC radius (px) everywhere instead of the plane
varying vec2 vUv;
vec4 lvl(int i, vec2 uv) {
  if (i <= 0) return texture2D(tL0, uv);
  if (i == 1) return texture2D(tL1, uv);
  if (i == 2) return texture2D(tL2, uv);
  if (i == 3) return texture2D(tL3, uv);
  if (i == 4) return texture2D(tL4, uv);
  return texture2D(tL5, uv);
}
void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  vec3 dir = vec3(ndc.x / uProj.x, ndc.y / uProj.y, -1.0);
  float z = 1e4;
  for (int i = 0; i < 3; i++) {
    if (i >= uNP) break;
    float den = dot(uPlaneN[i], dir);
    float t = den < -1e-6 ? uPlaneD[i] / den : -1.0;
    if (t > 0.0) z = min(z, t);
  }
  float R = uFixedR >= 0.0 ? uFixedR : min(cocRadiusPx(z, uCocK, uFocus), uMaxR);
  if (R < 0.35) { gl_FragColor = texture2D(tL0, vUv); return; }
  float spacing = R * sqrt(3.14159265 / float(uN));
  float lf = clamp(log2(max(spacing / 1.3, 1.0)), 0.0, ${(LEVELS - 1).toFixed(1)});
  int l0 = int(floor(lf));
  float fr = lf - float(l0);
  vec4 acc = vec4(0.0);
  float wsum = 0.0;
  for (int i = 0; i < 1024; i++) {
    if (i >= uN) break;
    vec4 k = texelFetch(tKernel, ivec2(i, 0), 0);
    vec2 uv = vUv + k.xy * R / uRes;
    vec4 a = lvl(l0, uv);
    vec4 b = fr > 0.0 ? lvl(l0 + 1, uv) : a;
    acc += k.z * mix(a, b, fr);
    wsum += k.z;
  }
  gl_FragColor = acc / wsum;
}
`;

/** Points filling the unit-area aperture (Halton 2,3), weighted by the aperture profile. */
function apertureKernel(n: number, ap: ApertureSpec): Float32Array {
  const ext = apertureExtent(ap) + 0.01;
  const halton = (i: number, b: number): number => {
    let f = 1;
    let r = 0;
    while (i > 0) {
      f /= b;
      r += f * (i % b);
      i = Math.floor(i / b);
    }
    return r;
  };
  const out = new Float32Array(n * 4);
  let k = 0;
  for (let i = 1; k < n; i++) {
    const x = (halton(i, 2) * 2 - 1) * ext;
    const y = (halton(i, 3) * 2 - 1) * ext;
    const sd = apertureSd(ap, x, y);
    if (sd >= 0) continue;
    out[k * 4] = x;
    out[k * 4 + 1] = y;
    out[k * 4 + 2] = apertureProfile(ap, -sd);
    k++;
  }
  return out;
}

/** A plane in world space: a point on it and its normal, pointing to the camera's side. */
export interface Plane {
  point: THREE.Vector3;
  normal: THREE.Vector3;
}

export interface LensBlurLook {
  lens: LensState;
  /** The layer's camera (for the projection). */
  camera: THREE.PerspectiveCamera;
  /** The plane(s) the layer lies on (1..3; each pixel takes the nearest one its ray leaves through). */
  planes: Plane[];
  /** Largest CoC radius, 1080p px (default 140). */
  maxRadius?: number;
  /** Largest CoC radius actually on the layer (canvas px, from cocRadiusAt): small blurs use fewer taps. */
  radius?: number;
}

export class LensBlur {
  private readonly levels: THREE.WebGLRenderTarget[] = [];
  private kernel: THREE.DataTexture;
  private apertureKey = '';
  private readonly mDown = mat(DOWN_FRAG, { tSrc: { value: null }, uTexel: { value: new THREE.Vector2() } });
  private readonly mGather: THREE.ShaderMaterial;
  private readonly quad: THREE.Mesh;
  private readonly qscene = new THREE.Scene();
  private readonly qcam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly scale: number;
  private readonly samples: number;

  constructor(
    private readonly renderer: THREE.WebGLRenderer,
    private readonly width: number,
    private readonly height: number,
    samples = 384,
    aperture: ApertureSpec = defaultAperture(),
  ) {
    this.scale = height / 1080;
    for (let i = 1; i <= LEVELS; i++) this.levels.push(hdrTarget(Math.ceil(width / 2 ** i), Math.ceil(height / 2 ** i)));
    this.samples = samples;
    this.kernel = this.makeKernel(aperture);
    this.mGather = new THREE.ShaderMaterial({
      vertexShader: FS_VERT,
      fragmentShader: GATHER_FRAG,
      uniforms: {
        tL0: { value: null },
        tL1: { value: this.levels[0].texture },
        tL2: { value: this.levels[1].texture },
        tL3: { value: this.levels[2].texture },
        tL4: { value: this.levels[3].texture },
        tL5: { value: this.levels[4].texture },
        tKernel: { value: this.kernel },
        uN: { value: samples },
        uRes: { value: new THREE.Vector2(width, height) },
        uProj: { value: new THREE.Vector2(1, 1) },
        uNP: { value: 1 },
        uPlaneN: { value: [new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 1)] },
        uPlaneD: { value: [-1, -1, -1] },
        uCocK: { value: 0 },
        uFocus: { value: 10 },
        uMaxR: { value: 140 },
        uFixedR: { value: -1 },
      },
      depthTest: false,
      depthWrite: false,
      blending: THREE.NoBlending,
      toneMapped: false,
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
    this.quad = new THREE.Mesh(g, this.mDown);
    this.quad.frustumCulled = false;
    this.qscene.add(this.quad);
  }

  private makeKernel(ap: ApertureSpec): THREE.DataTexture {
    this.apertureKey = JSON.stringify(ap);
    const t = new THREE.DataTexture(apertureKernel(this.samples, ap), this.samples, 1, THREE.RGBAFormat, THREE.FloatType);
    t.needsUpdate = true;
    return t;
  }

  /** Change the aperture (rebuilds the kernel only when the shape actually changes). */
  setAperture(ap: ApertureSpec): void {
    if (JSON.stringify(ap) === this.apertureKey) return;
    this.kernel.dispose();
    this.kernel = this.makeKernel(ap);
    this.mGather.uniforms.tKernel.value = this.kernel;
  }

  private pass(m: THREE.Material, target: THREE.WebGLRenderTarget): void {
    this.quad.material = m;
    this.renderer.setRenderTarget(target);
    this.renderer.render(this.qscene, this.qcam);
  }

  /** Largest CoC radius (canvas px) over a set of world points, e.g. a surface's corners. */
  cocRadiusAt(look: LensBlurLook, points: THREE.Vector3[]): number {
    const k = cocK(look.lens, this.width);
    const v = new THREE.Vector3();
    let r = 0;
    for (const p of points) {
      const z = -v.copy(p).applyMatrix4(look.camera.matrixWorldInverse).z;
      if (z > 0) r = Math.max(r, (0.5 * k * Math.abs(z - look.lens.focus)) / z);
    }
    return Math.min(r, (look.maxRadius ?? 140) * this.scale);
  }

  /** Defocus `src` (premultiplied RGBA, full size) into `out`, CoC from the lens at the layer's plane. */
  apply(src: THREE.Texture, out: THREE.WebGLRenderTarget, look: LensBlurLook): void {
    this.pyramid(src);
    const cam = look.camera;
    const view = cam.matrixWorldInverse;
    const u = this.mGather.uniforms;
    const planes = look.planes.slice(0, 3);
    u.uNP.value = planes.length;
    planes.forEach((pl, i) => {
      const n = pl.normal.clone().transformDirection(view);
      u.uPlaneN.value[i].copy(n);
      u.uPlaneD.value[i] = n.dot(pl.point.clone().applyMatrix4(view));
    });
    u.tL0.value = src;
    u.uFixedR.value = -1;
    u.uProj.value.set(cam.projectionMatrix.elements[0], cam.projectionMatrix.elements[5]);
    u.uCocK.value = cocK(look.lens, this.width);
    u.uFocus.value = look.lens.focus;
    u.uMaxR.value = (look.maxRadius ?? 140) * this.scale;
    u.uN.value = this.taps(look.radius);
    this.pass(this.mGather, out);
  }

  /** Defocus `src` through the aperture with one CoC radius (px of `src`) everywhere: pre-blurred assets. */
  applyRadius(src: THREE.Texture, out: THREE.WebGLRenderTarget, radius: number): void {
    this.pyramid(src);
    const u = this.mGather.uniforms;
    u.tL0.value = src;
    u.uFixedR.value = radius;
    u.uN.value = this.taps(radius);
    this.pass(this.mGather, out);
  }

  /** Taps for a blur of radius R: about one per pixel of the aperture's area (the kernel is a Halton
   *  sequence, so any prefix of it still fills the shape evenly), never fewer than 64. */
  private taps(radius?: number): number {
    if (radius === undefined) return this.samples;
    return Math.min(this.samples, Math.max(64, Math.ceil(3.5 * radius * radius)));
  }

  private pyramid(src: THREE.Texture): void {
    const md = this.mDown.uniforms;
    for (let i = 0; i < LEVELS; i++) {
      const from = i === 0 ? src : this.levels[i - 1].texture;
      md.tSrc.value = from;
      md.uTexel.value.set(1 / (i === 0 ? this.width : this.levels[i - 1].width), 1 / (i === 0 ? this.height : this.levels[i - 1].height));
      this.pass(this.mDown, this.levels[i]);
    }
  }

  dispose(): void {
    for (const rt of this.levels) rt.dispose();
    this.kernel.dispose();
    this.mDown.dispose();
    this.mGather.dispose();
    this.quad.geometry.dispose();
  }
}
