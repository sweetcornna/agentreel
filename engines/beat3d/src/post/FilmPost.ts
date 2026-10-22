// The film post chain, shared by every shot:
//   [depth DOF] -> bloom pyramid -> halation -> anamorphic streak -> composite
//   composite = chromatic aberration, bloom mix, halation, streak, exposure, AgX + look,
//               lift/gamma/gain/saturation, vignette, negative, 2D overlay, grain, dither, letterbox.
// Depth of field is best done optically, per layer before the post chain (post/LensBlur.ts for
// surfaces, world/bokeh.ts for points), both through the one aperture. The depth-buffer DOF here
// (`dof`, off by default) is the cheap alternative for worlds of solid geometry.
// All effect radii are defined at 1080p; at 4K the pyramids simply skip one extra level, so every
// blur covers the same fraction of the frame at both sizes.

import * as THREE from 'three';
import type { PostParams } from './params';
import { COMPOSITE_FRAG, COPY_FRAG, DOF_FRAG, DOWN13_FRAG, FS_VERT, HDOWN_FRAG, HUP_FRAG, THRESH_FRAG, UP_FRAG } from './shaders';
import { SUPER35_WIDTH_MM } from '../lib/lens';

const STREAK_LEVELS = 7;

export function hdrTarget(w: number, h: number): THREE.WebGLRenderTarget {
  return new THREE.WebGLRenderTarget(Math.max(1, w), Math.max(1, h), {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    wrapS: THREE.ClampToEdgeWrapping,
    wrapT: THREE.ClampToEdgeWrapping,
    depthBuffer: false,
    stencilBuffer: false,
    generateMipmaps: false,
  });
}

export function mat(frag: string, uniforms: Record<string, THREE.IUniform>, blending: THREE.Blending = THREE.NoBlending): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: FS_VERT,
    fragmentShader: frag,
    uniforms,
    depthTest: false,
    depthWrite: false,
    blending,
    toneMapped: false,
  });
}

export interface PostInput {
  scene: THREE.Texture;
  /** Additive HDR screen layer drawn apart from the scene; added after depth of field. */
  screen?: THREE.Texture | null;
  /** Depth of the world (for `dof`). */
  depth?: THREE.Texture | null;
  /** Near and far of the world camera (for `dof`). */
  camera?: { near: number; far: number } | null;
  overlay: THREE.Texture | null;
  params: PostParams;
  frame: number;
  seed: number;
  band: { top: number; bottom: number };
}

export class FilmPost {
  readonly width: number;
  readonly height: number;
  readonly scale: number;
  private readonly skip: number;
  private readonly quad: THREE.Mesh;
  private readonly qscene = new THREE.Scene();
  private readonly qcam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  private readonly down: THREE.WebGLRenderTarget[] = [];
  private readonly up: THREE.WebGLRenderTarget[] = [];
  private readonly hal: THREE.WebGLRenderTarget[] = [];
  private readonly halUp: THREE.WebGLRenderTarget[] = [];
  private readonly streak: THREE.WebGLRenderTarget[] = [];
  private readonly streakUp: THREE.WebGLRenderTarget[] = [];
  readonly accumRT: THREE.WebGLRenderTarget;
  private dofRT: THREE.WebGLRenderTarget | null = null;
  private readonly bloomLevels: number;

  private readonly mDown = mat(DOWN13_FRAG, { tSrc: { value: null }, uTexel: { value: new THREE.Vector2() }, uKaris: { value: 0 }, uThreshold: { value: 0 }, uKnee: { value: 0.5 } });
  private readonly mUp = mat(UP_FRAG, { tLow: { value: null }, tCur: { value: null }, uLowTexel: { value: new THREE.Vector2() }, uCurWeight: { value: 1 }, uLowWeight: { value: 1 } });
  private readonly mThresh = mat(THRESH_FRAG, { tSrc: { value: null }, uThreshold: { value: 1 }, uTint: { value: new THREE.Vector3(1, 1, 1) }, uTexel: { value: new THREE.Vector2() }, uDespeckle: { value: 0 } });
  private readonly mHDown = mat(HDOWN_FRAG, { tSrc: { value: null }, uTexel: { value: new THREE.Vector2() } });
  private readonly mHUp = mat(HUP_FRAG, { tLow: { value: null }, tCur: { value: null }, uLowTexel: { value: new THREE.Vector2() }, uFalloff: { value: 0.8 } });
  readonly mAccum = mat(COPY_FRAG, { tSrc: { value: null }, uWeight: { value: 1 } }, THREE.AdditiveBlending);
  private readonly mAddScreen = mat(COPY_FRAG, { tSrc: { value: null }, uWeight: { value: 1 } }, THREE.AdditiveBlending);
  private readonly mDof = mat(DOF_FRAG, {
    tSrc: { value: null },
    tDepth: { value: null },
    uRes: { value: new THREE.Vector2() },
    uNear: { value: 0.1 },
    uFar: { value: 1000 },
    uFocus: { value: 10 },
    uCocK: { value: 0 },
    uMaxR: { value: 18 },
    uMaxRNear: { value: 18 },
    uStep: { value: 0.5 },
  });
  private readonly mComp: THREE.ShaderMaterial;

  constructor(
    private readonly renderer: THREE.WebGLRenderer,
    width: number,
    height: number,
  ) {
    this.width = width;
    this.height = height;
    this.scale = height / 1080;
    this.skip = Math.max(0, Math.round(Math.log2(this.scale)));
    this.quad = new THREE.Mesh(this.fsTriangle(), this.mDown);
    this.quad.frustumCulled = false;
    this.qscene.add(this.quad);

    // Bloom pyramid: level i has size W/2^(i+1). Levels < skip only exist to downsample cleanly.
    this.bloomLevels = 7;
    for (let i = 0; i < this.skip + this.bloomLevels; i++) {
      const w = Math.ceil(width / 2 ** (i + 1));
      const h = Math.ceil(height / 2 ** (i + 1));
      this.down.push(hdrTarget(w, h));
      this.up.push(hdrTarget(w, h));
    }
    // Halation: three levels starting at the 1080p quarter size.
    for (let i = 0; i < 3; i++) {
      const src = this.down[this.skip + 1 + i];
      this.hal.push(hdrTarget(src.width, src.height));
      this.halUp.push(hdrTarget(src.width, src.height));
    }
    // Streak: horizontal-only chain at the 1080p half height.
    const base = this.down[this.skip];
    for (let i = 0; i < STREAK_LEVELS; i++) {
      const w = Math.max(2, Math.ceil(base.width / 2 ** i));
      this.streak.push(hdrTarget(w, base.height));
      this.streakUp.push(hdrTarget(w, base.height));
    }
    this.accumRT = hdrTarget(width, height);

    this.mComp = mat(COMPOSITE_FRAG, {
      tScene: { value: null },
      tBloom: { value: null },
      tHal: { value: null },
      tStreak: { value: null },
      tOverlay: { value: null },
      uHasOverlay: { value: 0 },
      uRes: { value: new THREE.Vector2(width, height) },
      uScale: { value: this.scale },
      uBloomMix: { value: 0 },
      uBloomNorm: { value: 1 },
      uBloomTint: { value: new THREE.Vector3(1, 1, 1) },
      uHalAmount: { value: 0 },
      uHalNorm: { value: 1 },
      uStreakAmount: { value: 0 },
      uStreakNorm: { value: 1 },
      uStreakTint: { value: new THREE.Vector3(1, 1, 1) },
      uExposure: { value: 0 },
      uSlope: { value: new THREE.Vector3(1, 1, 1) },
      uOffset: { value: new THREE.Vector3(0, 0, 0) },
      uPower: { value: new THREE.Vector3(1, 1, 1) },
      uLookSat: { value: 1 },
      uLift: { value: new THREE.Vector3() },
      uGamma: { value: new THREE.Vector3(1, 1, 1) },
      uGain: { value: new THREE.Vector3(1, 1, 1) },
      uSat: { value: 1 },
      uVignette: { value: 0 },
      uVigK: { value: 1 },
      uCA: { value: 0 },
      uGrain: { value: 0 },
      uGrainSize: { value: 1 },
      uGrainChroma: { value: 0 },
      uFrame: { value: 0 },
      uSeed: { value: 0 },
      uBand: { value: new THREE.Vector2(0, height) },
      uLetterbox: { value: 1 },
      uOverlayClean: { value: 0 },
      uCAWarm: { value: 1 },
      uNegative: { value: 0 },
      uNegPaper: { value: new THREE.Vector3(0.8, 0.74, 0.6) },
      uNegInk: { value: new THREE.Vector3(0.028, 0.028, 0.032) },
      uNegWipe: { value: new THREE.Vector4(0, 0, -1, 0) },
    });
  }

  private fsTriangle(): THREE.BufferGeometry {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
    return g;
  }

  /** Draw one full-screen pass. */
  pass(material: THREE.Material, target: THREE.WebGLRenderTarget | null, clear = true): void {
    this.quad.material = material;
    this.renderer.setRenderTarget(target);
    if (clear) this.renderer.clear(true, false, false);
    this.renderer.render(this.qscene, this.qcam);
  }

  private texel(rt: THREE.WebGLRenderTarget | THREE.Texture, w?: number, h?: number): THREE.Vector2 {
    if (rt instanceof THREE.WebGLRenderTarget) return new THREE.Vector2(1 / rt.width, 1 / rt.height);
    return new THREE.Vector2(1 / (w ?? this.width), 1 / (h ?? this.height));
  }

  render(input: PostInput): void {
    const p = input.params;
    const r = this.renderer;
    const prevAutoClear = r.autoClear;
    r.autoClear = false;
    r.setClearColor(0x000000, 1);

    // --- depth of field (from the world's depth buffer)
    let src: THREE.Texture = input.scene;
    if (p.dof.enabled && input.depth && input.camera) {
      if (!this.dofRT) this.dofRT = hdrTarget(this.width, this.height);
      const u = this.mDof.uniforms;
      const f = p.dof.focalMm / 1000;
      const A = f / p.dof.fStop;
      u.tSrc.value = src;
      u.tDepth.value = input.depth;
      u.uRes.value.set(this.width, this.height);
      u.uNear.value = input.camera.near;
      u.uFar.value = input.camera.far;
      u.uFocus.value = p.dof.focus;
      u.uCocK.value = ((A * f) / Math.max(p.dof.focus - f, 1e-3) / (SUPER35_WIDTH_MM / 1000)) * this.width;
      u.uMaxR.value = p.dof.maxRadius * this.scale;
      u.uMaxRNear.value = (p.dof.maxRadiusNear ?? p.dof.maxRadius) * this.scale;
      u.uStep.value = 0.5 * this.scale;
      this.pass(this.mDof, this.dofRT);
      src = this.dofRT.texture;
    }
    // --- screen layer drawn apart (DOF shots): in focus, but still feeds bloom, halation and streak
    if (input.screen) {
      if (!this.dofRT) this.dofRT = hdrTarget(this.width, this.height);
      if (src !== this.dofRT.texture) {
        this.mAddScreen.uniforms.tSrc.value = src;
        this.pass(this.mAddScreen, this.dofRT);
        src = this.dofRT.texture;
      }
      this.mAddScreen.uniforms.tSrc.value = input.screen;
      this.pass(this.mAddScreen, this.dofRT, false);
    }

    // --- bloom pyramid (down)
    const md = this.mDown.uniforms;
    for (let i = 0; i < this.down.length; i++) {
      const from = i === 0 ? src : this.down[i - 1].texture;
      md.tSrc.value = from;
      md.uTexel.value = i === 0 ? this.texel(src) : this.texel(this.down[i - 1]);
      md.uKaris.value = i === 0 ? 1 : 0;
      md.uThreshold.value = i === 0 ? p.bloom.threshold : 0;
      md.uKnee.value = p.bloom.knee;
      this.pass(this.mDown, this.down[i]);
    }
    // --- bloom pyramid (up), weighted per scale
    const weights = p.bloom.weights;
    const wAt = (k: number): number => weights[Math.min(k, weights.length - 1)] ?? 0;
    const last = this.down.length - 1;
    const mu = this.mUp.uniforms;
    // Coarsest level: copy with weight via up pass against itself (low weight 0).
    mu.tLow.value = this.down[last].texture;
    mu.tCur.value = this.down[last].texture;
    mu.uLowTexel.value = this.texel(this.down[last]);
    mu.uCurWeight.value = wAt(last - this.skip);
    mu.uLowWeight.value = 0;
    this.pass(this.mUp, this.up[last]);
    let wsum = wAt(last - this.skip);
    for (let i = last - 1; i >= this.skip; i--) {
      const w = wAt(i - this.skip);
      wsum += w;
      mu.tLow.value = this.up[i + 1].texture;
      mu.tCur.value = this.down[i].texture;
      mu.uLowTexel.value = this.texel(this.up[i + 1]);
      mu.uCurWeight.value = w;
      mu.uLowWeight.value = 1;
      this.pass(this.mUp, this.up[i]);
    }
    const bloomTex = this.up[this.skip].texture;

    // --- halation: threshold the quarter-res blur, tint red-orange, blur twice more
    const mt = this.mThresh.uniforms;
    mt.tSrc.value = this.down[this.skip + 1].texture;
    mt.uThreshold.value = p.halation.threshold;
    mt.uTint.value.set(...p.halation.tint);
    mt.uDespeckle.value = 0;
    this.pass(this.mThresh, this.hal[0]);
    for (let i = 1; i < this.hal.length; i++) {
      md.tSrc.value = this.hal[i - 1].texture;
      md.uTexel.value = this.texel(this.hal[i - 1]);
      md.uKaris.value = 0;
      md.uThreshold.value = 0;
      // Downsample in place at the next pyramid size.
      this.pass(this.mDown, this.hal[i]);
    }
    const hl = this.hal.length - 1;
    mu.tLow.value = this.hal[hl].texture;
    mu.tCur.value = this.hal[hl].texture;
    mu.uLowTexel.value = this.texel(this.hal[hl]);
    mu.uCurWeight.value = 1;
    mu.uLowWeight.value = 0;
    this.pass(this.mUp, this.halUp[hl]);
    for (let i = hl - 1; i >= 0; i--) {
      mu.tLow.value = this.halUp[i + 1].texture;
      mu.tCur.value = this.hal[i].texture;
      mu.uLowTexel.value = this.texel(this.halUp[i + 1]);
      mu.uCurWeight.value = 1;
      mu.uLowWeight.value = 1;
      this.pass(this.mUp, this.halUp[i]);
    }

    // --- anamorphic streak: threshold, then horizontal-only pyramid
    mt.tSrc.value = this.down[this.skip].texture;
    mt.uThreshold.value = p.streak.threshold;
    mt.uTint.value.set(1, 1, 1);
    mt.uTexel.value.copy(this.texel(this.down[this.skip]));
    mt.uDespeckle.value = p.streak.despeckle ?? 0;
    this.pass(this.mThresh, this.streak[0]);
    const mh = this.mHDown.uniforms;
    for (let i = 1; i < this.streak.length; i++) {
      mh.tSrc.value = this.streak[i - 1].texture;
      mh.uTexel.value = this.texel(this.streak[i - 1]);
      this.pass(this.mHDown, this.streak[i]);
    }
    const sl = this.streak.length - 1;
    const mhu = this.mHUp.uniforms;
    mhu.tLow.value = this.streak[sl].texture;
    mhu.tCur.value = this.streak[sl].texture;
    mhu.uLowTexel.value = this.texel(this.streak[sl]);
    mhu.uFalloff.value = 0;
    this.pass(this.mHUp, this.streakUp[sl]);
    let snorm = 1;
    let acc = 1;
    for (let i = sl - 1; i >= 0; i--) {
      mhu.tLow.value = this.streakUp[i + 1].texture;
      mhu.tCur.value = this.streak[i].texture;
      mhu.uLowTexel.value = this.texel(this.streakUp[i + 1]);
      mhu.uFalloff.value = p.streak.falloff;
      this.pass(this.mHUp, this.streakUp[i]);
      acc *= p.streak.falloff;
      snorm += acc;
    }

    // --- composite to the canvas
    const c = this.mComp.uniforms;
    c.tScene.value = src;
    c.tBloom.value = bloomTex;
    c.tHal.value = this.halUp[0].texture;
    c.tStreak.value = this.streakUp[0].texture;
    c.tOverlay.value = input.overlay;
    c.uHasOverlay.value = input.overlay ? 1 : 0;
    c.uBloomMix.value = p.bloom.mix;
    c.uBloomNorm.value = 1 / Math.max(wsum, 1e-4);
    c.uBloomTint.value.set(...p.bloom.tint);
    c.uHalAmount.value = p.halation.amount;
    c.uHalNorm.value = 1 / 3;
    c.uStreakAmount.value = p.streak.amount;
    c.uStreakNorm.value = 1 / snorm;
    c.uStreakTint.value.set(...p.streak.tint);
    c.uExposure.value = p.exposure;
    c.uSlope.value.set(...p.grade.slope);
    c.uOffset.value.set(...p.grade.offset);
    c.uPower.value.set(...p.grade.power);
    c.uLookSat.value = p.grade.lookSaturation;
    c.uLift.value.set(...p.grade.lift);
    c.uGamma.value.set(...p.grade.gamma);
    c.uGain.value.set(...p.grade.gain);
    c.uSat.value = p.grade.saturation;
    c.uVignette.value = p.vignette.amount;
    c.uVigK.value = p.vignette.softness;
    c.uCA.value = p.chromatic;
    c.uGrain.value = p.grain.amount;
    c.uGrainSize.value = p.grain.size;
    c.uGrainChroma.value = p.grain.chroma;
    c.uFrame.value = Math.floor(input.frame);
    c.uSeed.value = input.seed;
    c.uBand.value.set(input.band.top, input.band.bottom);
    c.uLetterbox.value = p.letterbox ? 1 : 0;
    c.uOverlayClean.value = p.overlayClean;
    c.uCAWarm.value = p.chromaticStyle === 'rgb' ? 0 : 1;
    c.uNegative.value = p.negative.amount;
    c.uNegPaper.value.set(...p.negative.paper);
    c.uNegInk.value.set(...p.negative.ink);
    {
      const w = p.negative.wipe;
      const Hc = this.height;
      c.uNegWipe.value.set(w.center[0] * this.width, w.center[1] * Hc, w.radius < 0 ? -1 : w.radius * Hc, w.soft * Hc);
    }
    this.pass(this.mComp, null);

    r.autoClear = prevAutoClear;
  }

  dispose(): void {
    for (const rt of [...this.down, ...this.up, ...this.hal, ...this.halUp, ...this.streak, ...this.streakUp, this.accumRT]) rt.dispose();
    this.dofRT?.dispose();
    for (const m of [this.mDown, this.mUp, this.mThresh, this.mHDown, this.mHUp, this.mAccum, this.mAddScreen, this.mDof, this.mComp]) m.dispose();
    this.quad.geometry.dispose();
  }
}
