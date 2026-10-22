// A layered stage for a take: the lens camera, the haze, the aperture defocus, and the three layers a
// frame is composed of:
//   back   haze behind the layer's plane(s) and the points there
//   layer  a surface (glass, a wall of beams), premultiplied, defocused through the aperture
//   front  haze on the camera side, points, text and threads
//   out = back * (1 - layer.a) + layer + front
// The layer targets can be reused by several worlds in turn; each composites into its own output, so a
// transition can mix two of them. A program renders into `out` and shows it with a full-screen pass.

import * as THREE from 'three';
import type { StageContext } from '../core/types';
import { hdrTarget, mat } from '../post/FilmPost';
import { LensBlur } from '../post/LensBlur';
import { fovYDegFromFocal, makeLensCamera, type LensState } from '../lib/lens';
import type { ApertureSpec } from '../lib/aperture';
import { Haze } from './haze';

/** Where the lens is and what it is doing at an instant. */
export interface CameraPose {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
  lens: LensState;
}

const COMPOSITE_FRAG = /* glsl */ `
uniform sampler2D tBack;
uniform sampler2D tLayer;
uniform sampler2D tFront;
varying vec2 vUv;
void main() {
  vec4 l = texture2D(tLayer, vUv);
  gl_FragColor = vec4(texture2D(tBack, vUv).rgb * (1.0 - l.a) + l.rgb + texture2D(tFront, vUv).rgb, 1.0);
}
`;

/** Additive emission that leaves the target's alpha alone (light on a layer does not make it opaque). */
export function emissive(m: THREE.ShaderMaterial): THREE.ShaderMaterial {
  m.blending = THREE.CustomBlending;
  m.blendEquation = THREE.AddEquation;
  m.blendSrc = THREE.OneFactor;
  m.blendDst = THREE.OneFactor;
  m.blendSrcAlpha = THREE.ZeroFactor;
  m.blendDstAlpha = THREE.OneFactor;
  m.transparent = true;
  m.depthWrite = false;
  return m;
}

export class Stage {
  readonly camera: THREE.PerspectiveCamera;
  readonly haze: Haze;
  readonly blur: LensBlur;
  /** Haze and points behind the layer's plane. */
  readonly back: THREE.WebGLRenderTarget;
  /** Haze and points on the camera side. */
  readonly front: THREE.WebGLRenderTarget;
  /** The layer (glass, a wall), premultiplied, multisampled; and its defocused copy. */
  readonly layer: THREE.WebGLRenderTarget;
  readonly layerBlur: THREE.WebGLRenderTarget;
  /** An empty layer, for frames with nothing in it (the multisampled target only resolves on a draw). */
  readonly none: THREE.DataTexture;
  private readonly mComposite: THREE.ShaderMaterial;
  private readonly quad: THREE.Mesh;
  private readonly qscene = new THREE.Scene();
  private readonly qcam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  constructor(
    readonly ctx: StageContext,
    opts: { focalMm?: number; near?: number; far?: number; aperture?: ApertureSpec } = {},
  ) {
    const { width: W, height: H, renderer } = ctx;
    this.camera = makeLensCamera(opts.focalMm ?? 50, W, H, opts.near, opts.far);
    this.haze = new Haze(renderer, W, H);
    this.blur = new LensBlur(renderer, W, H, 384, opts.aperture);
    this.back = hdrTarget(W, H);
    this.front = hdrTarget(W, H);
    this.layer = new THREE.WebGLRenderTarget(W, H, { type: THREE.HalfFloatType, format: THREE.RGBAFormat, samples: 4, depthBuffer: true, generateMipmaps: false });
    this.layerBlur = hdrTarget(W, H);
    this.none = new THREE.DataTexture(new Uint8Array(4), 1, 1);
    this.none.needsUpdate = true;
    this.mComposite = mat(COMPOSITE_FRAG, { tBack: { value: this.back.texture }, tLayer: { value: null }, tFront: { value: this.front.texture } });
    this.quad = new THREE.Mesh(this.fullscreen(), this.mComposite);
    this.quad.frustumCulled = false;
    this.qscene.add(this.quad);
  }

  /** One triangle covering the viewport. */
  fullscreen(): THREE.BufferGeometry {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
    return g;
  }

  /** Full-screen pass with `m` into `target`. */
  pass(m: THREE.Material, target: THREE.WebGLRenderTarget): void {
    this.quad.material = m;
    this.ctx.renderer.setRenderTarget(target);
    this.ctx.renderer.render(this.qscene, this.qcam);
  }

  clear(rt: THREE.WebGLRenderTarget, alpha = 1): void {
    const r = this.ctx.renderer;
    r.setRenderTarget(rt);
    r.setClearColor(0x000000, alpha);
    r.clear(true, true, false);
  }

  /** Put the lens camera where the rig says. */
  pose(s: CameraPose): void {
    const c = this.camera;
    c.position.copy(s.position);
    c.quaternion.copy(s.quaternion);
    c.fov = fovYDegFromFocal(s.lens.focalMm, this.ctx.width / this.ctx.height);
    c.updateProjectionMatrix();
    c.updateMatrixWorld(true);
  }

  /** out = back * (1 - layer.a) + layer + front. */
  composite(layer: THREE.Texture, out: THREE.WebGLRenderTarget): void {
    this.mComposite.uniforms.tLayer.value = layer;
    this.pass(this.mComposite, out);
  }

  dispose(): void {
    this.haze.dispose();
    this.blur.dispose();
    for (const rt of [this.back, this.front, this.layer, this.layerBlur]) rt.dispose();
    this.none.dispose();
    this.mComposite.dispose();
    this.quad.geometry.dispose();
  }
}
