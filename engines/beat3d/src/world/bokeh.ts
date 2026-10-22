// Points of light seen through the film's lens: every point (a lamp, a mote of dust, a pulse) is drawn
// as the image of the lens aperture (lib/aperture.ts: circle, leaf, polygon), sized by the thin-lens
// circle of confusion at its depth. In focus it is a pixel-sized spark; out of focus it opens into the
// aperture's shape with the same total energy, a softened edge with a brighter rim inside it, faint
// onion rings, two specks of dust on an element, a warm fringe, and near the frame edge the cat's-eye
// clip of a real lens (optical vignetting). So point bokeh and layer defocus (post/LensBlur.ts) read
// as one lens. Additive, HDR, no depth write.
//
// set() the points every frame (a pure function of the frame), then update() with the lens.
// `profile: 'hero'` is the close-up look for a few big discs that fill part of the frame: a crisper
// spherical-aberration rim, finer tooling rings, lighter dust specks.

import * as THREE from 'three';
import type { StageContext } from '../core/types';
import { APERTURE_GLSL, apertureUniforms, setApertureUniforms } from '../gl/aperture.glsl';
import { defaultAperture, type ApertureSpec } from '../lib/aperture';
import { cocK, type LensState } from '../lib/lens';

export interface BokehLook {
  lens: LensState;
  /** Brightness multiplier (default 1). */
  gain?: number;
  /** Largest CoC radius drawn, 1080p px (default 140). */
  maxRadius?: number;
  /** Optical vignetting strength 0..1 (default 0.35). */
  catEye?: number;
  /** Relative radius difference between red and blue images (default 0.018: a warm outer fringe). */
  chroma?: number;
  /** Haze extinction (1/m): light dims with its distance to the lens (default 0). */
  extinction?: number;
  /** Aperture for this frame (default: the one the points were created with). */
  aperture?: ApertureSpec;
}

const VERT = /* glsl */ `
${APERTURE_GLSL}
attribute vec3 iPos;
attribute vec4 iCol;      // rgb flux (linear HDR x 1080p px^2), a = emitter radius (m)
uniform vec2 uRes;
uniform float uScale;
uniform float uCocK;
uniform float uFocus;
uniform float uMaxR;
uniform float uGain;
uniform float uExt;
varying vec2 vP;
varying float vR;
varying vec3 vCol;
varying vec2 vNdc;
void main() {
  vec4 mv = modelViewMatrix * vec4(iPos, 1.0);
  float z = -mv.z;
  vec3 col = iCol.rgb * uGain * exp(-uExt * length(mv.xyz));
  if (z < 0.02 || max(col.r, max(col.g, col.b)) <= 0.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  vec4 clip = projectionMatrix * mv;
  float ppm = 0.5 * uRes.y * projectionMatrix[1][1] / z;
  float coc = cocRadiusPx(z, uCocK, uFocus);
  float em = iCol.a * ppm;
  // Defocus and the emitter's own size add roughly in quadrature; never smaller than ~1 px.
  float R = clamp(sqrt(coc * coc + em * em), 0.85 * uScale, uMaxR);
  vec2 off = position.xy * (R * 1.1 * apertureExtent() + 2.0 * uScale);
  clip.xy += off / (0.5 * uRes) * clip.w;
  gl_Position = clip;
  vP = off;
  vR = R;
  vNdc = clip.xy / clip.w;
  // The flux spread over the aperture (area pi R^2, in 1080p px^2): energy is conserved as it defocuses.
  float r1080 = R / uScale;
  vCol = col / (3.14159265 * r1080 * r1080);
}
`;

const FRAG = /* glsl */ `
${APERTURE_GLSL}
uniform vec2 uRes;
uniform float uScale;
uniform float uCatEye;
uniform float uChroma;
varying vec2 vP;
varying float vR;
varying vec3 vCol;
varying vec2 vNdc;
void main() {
  // Edge softness: the pixel plus the blur of a real lens's edge, which grows with the disc.
  float aa = 0.75 * uScale + 0.05 * vR;
  vec3 c = vec3(
    apertureBokeh(vP, vR * (1.0 + uChroma), aa),
    apertureBokeh(vP, vR, aa),
    apertureBokeh(vP, vR * (1.0 - uChroma), aa));
  // Dust on an element of the lens: two soft specks at fixed places in the aperture, seen in the
  // large discs only.
  vec2 ap = vP / vR;
  float speck = 1.0 - 0.3 * smoothstep(0.075, 0.025, length(ap - vec2(0.31, -0.22))) - 0.2 * smoothstep(0.05, 0.015, length(ap - vec2(-0.38, 0.29)));
  c *= mix(1.0, speck, smoothstep(10.0 * uScale, 24.0 * uScale, vR));
  // Cat's eye: the aperture is clipped by the image of the lens barrel, displaced toward the frame centre.
  // Radial position normalised to the half-width: 1 at the left and right edges.
  vec2 rad = vec2(vNdc.x, vNdc.y * uRes.y / uRes.x);
  vec2 shift = -rad * uCatEye * vR;
  float sd2 = length(vP - shift) - vR * max(1.42, 1.18 * apertureExtent());
  c *= clamp(0.5 - sd2 / aa, 0.0, 1.0);
  gl_FragColor = vec4(vCol * c, 1.0);
}
`;

/** The hero discs: same aperture and thin-lens CoC; the close view resolves the bright spherical-
 *  aberration rim, the aspheric tooling rings and the warm outer fringe. */
const HERO_FRAG = /* glsl */ `
${APERTURE_GLSL}
uniform vec2 uRes;
uniform float uScale;
uniform float uCatEye;
uniform float uChroma;
varying vec2 vP;
varying float vR;
varying vec3 vCol;
varying vec2 vNdc;
float profile(vec2 p, float radius, float aa) {
  float sd = apertureSd(p / radius) * radius;
  float d = max(-sd / radius, 0.0);
  float inside = clamp(0.5 - sd / aa, 0.0, 1.0);
  float rim = exp(-pow((d - 0.045) / 0.045, 2.0));
  float rings = 0.045 * cos(d * 32.0) * smoothstep(0.07, 0.18, d);
  return inside * (0.69 + 1.2 * rim + rings);
}
void main() {
  float aa = 0.75 * uScale + 0.022 * vR;
  vec3 c = vec3(profile(vP, vR * (1.0 + uChroma), aa),
                profile(vP, vR, aa), profile(vP, vR * (1.0 - uChroma), aa));
  vec2 ap = vP / vR;
  float speck = 1.0 - 0.18 * smoothstep(0.075, 0.025, length(ap - vec2(0.31, -0.22)))
                    - 0.12 * smoothstep(0.05, 0.015, length(ap - vec2(-0.38, 0.29)));
  vec2 rad = vec2(vNdc.x, vNdc.y * uRes.y / uRes.x);
  float sd2 = length(vP + rad * uCatEye * vR) - vR * max(1.42, 1.18 * apertureExtent());
  c *= speck * clamp(0.5 - sd2 / aa, 0.0, 1.0);
  gl_FragColor = vec4(vCol * c, 1.0);
}
`;

export class BokehPoints {
  readonly mesh: THREE.Mesh;
  private readonly geo: THREE.InstancedBufferGeometry;
  private readonly pos: THREE.InstancedBufferAttribute;
  private readonly col: THREE.InstancedBufferAttribute;
  private readonly mat: THREE.ShaderMaterial;
  private readonly aperture: ApertureSpec;

  constructor(
    private readonly ctx: StageContext,
    readonly capacity: number,
    opts: { depthTest?: boolean; renderOrder?: number; aperture?: ApertureSpec; profile?: 'standard' | 'hero' } = {},
  ) {
    this.aperture = opts.aperture ?? defaultAperture();
    const g = new THREE.InstancedBufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]), 3));
    g.setIndex([0, 1, 2, 0, 2, 3]);
    this.pos = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
    this.col = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4);
    this.pos.setUsage(THREE.DynamicDrawUsage);
    this.col.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('iPos', this.pos);
    g.setAttribute('iCol', this.col);
    g.instanceCount = 0;
    this.geo = g;
    this.mat = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: opts.profile === 'hero' ? HERO_FRAG : FRAG,
      uniforms: {
        ...apertureUniforms(this.aperture),
        uRes: { value: new THREE.Vector2(ctx.width, ctx.height) },
        uScale: { value: ctx.scale },
        uCocK: { value: 0 },
        uFocus: { value: 10 },
        uMaxR: { value: 140 * ctx.scale },
        uGain: { value: 1 },
        uExt: { value: 0 },
        uCatEye: { value: 0.35 },
        uChroma: { value: 0.018 },
      },
      blending: THREE.AdditiveBlending,
      depthTest: opts.depthTest ?? false,
      depthWrite: false,
      transparent: true,
      toneMapped: false,
    });
    this.mesh = new THREE.Mesh(g, this.mat);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = opts.renderOrder ?? 10;
  }

  /** Point i: world position, flux (linear HDR x 1080p px^2: a value of 1 fills one pixel at 1.0), emitter radius (m). */
  set(i: number, x: number, y: number, z: number, r: number, g: number, b: number, emitter = 0): void {
    const p = this.pos.array as Float32Array;
    const c = this.col.array as Float32Array;
    p[i * 3] = x;
    p[i * 3 + 1] = y;
    p[i * 3 + 2] = z;
    c[i * 4] = r;
    c[i * 4 + 1] = g;
    c[i * 4 + 2] = b;
    c[i * 4 + 3] = emitter;
  }

  /** Upload the first `count` points and set the lens. */
  update(count: number, look: BokehLook): void {
    const u = this.mat.uniforms;
    u.uCocK.value = cocK(look.lens, this.ctx.width);
    u.uFocus.value = look.lens.focus;
    u.uMaxR.value = (look.maxRadius ?? 140) * this.ctx.scale;
    u.uGain.value = look.gain ?? 1;
    u.uExt.value = look.extinction ?? 0;
    u.uCatEye.value = look.catEye ?? 0.35;
    u.uChroma.value = look.chroma ?? 0.018;
    setApertureUniforms(u, look.aperture ?? this.aperture);
    this.geo.instanceCount = Math.min(count, this.capacity);
    this.pos.needsUpdate = true;
    this.col.needsUpdate = true;
    this.pos.clearUpdateRanges();
    this.col.clearUpdateRanges();
    this.pos.addUpdateRange(0, this.geo.instanceCount * 3);
    this.col.addUpdateRange(0, this.geo.instanceCount * 4);
  }

  dispose(): void {
    this.geo.dispose();
    this.mat.dispose();
  }
}
