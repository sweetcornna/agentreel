// Screen-space HDR primitives in canvas pixels (y down), drawn into FrameResult.screen.
// They go through the whole post chain, so they bloom, halate and streak like world light.
//   ScreenLine  the persistent line / horizon stroke (thin gaussian core, tapered ends)
//   GlowPoint   a small light source seen from afar: core + in-scattering halo
//   ScreenBar   a solid rectangle that can rotate about a pivot (the cursor)

import * as THREE from 'three';

const unitQuad = (): THREE.BufferGeometry => {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]), 3));
  g.setIndex([0, 1, 2, 0, 2, 3]);
  return g;
};

const additive = (vertexShader: string, fragmentShader: string, uniforms: Record<string, THREE.IUniform>): THREE.ShaderMaterial =>
  new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthTest: false,
    depthWrite: false,
    side: THREE.DoubleSide,
    toneMapped: false,
  });

export class ScreenLine {
  readonly mesh: THREE.Mesh;
  readonly u: {
    uX0: THREE.IUniform<number>;
    uX1: THREE.IUniform<number>;
    uY: THREE.IUniform<number>;
    uSigma: THREE.IUniform<number>;
    uPad: THREE.IUniform<number>;
    uTaper: THREE.IUniform<number>;
    uColor: THREE.IUniform<THREE.Vector3>;
    uIntensity: THREE.IUniform<number>;
    uGlow: THREE.IUniform<number>;
  };

  constructor() {
    this.u = {
      uX0: { value: 0 },
      uX1: { value: 100 },
      uY: { value: 0 },
      uSigma: { value: 0.8 },
      uPad: { value: 12 },
      uTaper: { value: 40 },
      uColor: { value: new THREE.Vector3(1, 0.95, 0.88) },
      uIntensity: { value: 1 },
      uGlow: { value: 0.0 },
    };
    const mat = additive(
      /* glsl */ `
      uniform float uX0, uX1, uY, uPad;
      varying vec2 vP;
      void main() {
        vec2 p = vec2(mix(uX0 - uPad, uX1 + uPad, position.x), uY + (position.y - 0.5) * 2.0 * uPad);
        vP = p;
        gl_Position = projectionMatrix * viewMatrix * vec4(p, 0.0, 1.0);
      }`,
      /* glsl */ `
      uniform float uX0, uX1, uY, uSigma, uTaper, uIntensity, uGlow, uPad;
      uniform vec3 uColor;
      varying vec2 vP;
      void main() {
        float dy = vP.y - uY;
        float core = exp(-0.5 * dy * dy / (uSigma * uSigma));
        float glow = uGlow * exp(-abs(dy) / (uPad * 0.25));
        float ex = min(vP.x - uX0, uX1 - vP.x);
        float taper = smoothstep(-uSigma * 2.0, uTaper, ex);
        gl_FragColor = vec4(uColor * uIntensity * (core + glow) * taper, 1.0);
      }`,
      this.u as unknown as Record<string, THREE.IUniform>,
    );
    this.mesh = new THREE.Mesh(unitQuad(), mat);
    this.mesh.frustumCulled = false;
  }

  set(o: { x0: number; x1: number; y: number; sigma: number; intensity: number; color?: [number, number, number]; taper?: number; glow?: number; pad?: number }): void {
    this.u.uX0.value = o.x0;
    this.u.uX1.value = o.x1;
    this.u.uY.value = o.y;
    this.u.uSigma.value = o.sigma;
    this.u.uIntensity.value = o.intensity;
    if (o.color) this.u.uColor.value.set(...o.color);
    if (o.taper !== undefined) this.u.uTaper.value = o.taper;
    if (o.glow !== undefined) this.u.uGlow.value = o.glow;
    this.u.uPad.value = o.pad ?? Math.max(6 * o.sigma, 8);
    this.mesh.visible = o.intensity > 0 && o.x1 > o.x0;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
  }
}

export interface GlowLook {
  /** Core radius (1/e) in canvas pixels and its linear HDR colour x intensity. */
  coreRadius: number;
  core: [number, number, number];
  /** Halo: radius where it halves, inner and outer colours (linear, already x intensity). */
  haloRadius: number;
  haloInner: [number, number, number];
  haloOuter: [number, number, number];
  /** Extent of the quad in halo radii. */
  extent?: number;
  /** Faint cross star (the cool light's 十): relative strength (default 0 = none). */
  star?: number;
}

export class GlowPoint {
  readonly mesh: THREE.Mesh;
  private readonly u: Record<string, THREE.IUniform>;

  constructor() {
    this.u = {
      uC: { value: new THREE.Vector2() },
      uR: { value: 10 },
      uCoreR: { value: 1 },
      uCore: { value: new THREE.Vector3(1, 1, 1) },
      uHaloR: { value: 4 },
      uHaloIn: { value: new THREE.Vector3(1, 0.7, 0.4) },
      uHaloOut: { value: new THREE.Vector3(0.9, 0.4, 0.1) },
      uGain: { value: 1 },
      uAspect: { value: 1 },
      uStar: { value: 0 },
    };
    const mat = additive(
      /* glsl */ `
      uniform vec2 uC; uniform float uR;
      varying vec2 vD;
      void main() {
        vec2 d = (position.xy - 0.5) * 2.0 * uR;
        vD = d;
        gl_Position = projectionMatrix * viewMatrix * vec4(uC + d, 0.0, 1.0);
      }`,
      /* glsl */ `
      uniform float uCoreR, uHaloR, uGain, uAspect, uR, uStar;
      uniform vec3 uCore, uHaloIn, uHaloOut;
      varying vec2 vD;
      void main() {
        vec2 d = vD * vec2(1.0, uAspect);
        float r = length(d);
        float core = exp(-(r * r) / (uCoreR * uCoreR));
        float q = r / uHaloR;
        float halo = 1.0 / pow(1.0 + q * q, 1.5);
        vec3 hc = mix(uHaloIn, uHaloOut, smoothstep(0.0, 3.0, q));
        float edge = 1.0 - smoothstep(0.7 * uR, uR, r);
        float w = 0.45 * uCoreR;
        float L = 0.9 * uHaloR;
        float star = uStar * (exp(-abs(d.y) / w - abs(d.x) / L) + exp(-abs(d.x) / w - abs(d.y) / L));
        gl_FragColor = vec4((uCore * core + hc * halo + mix(uCore, uHaloIn, 0.5) * star) * uGain * edge, 1.0);
      }`,
      this.u,
    );
    this.mesh = new THREE.Mesh(unitQuad(), mat);
    this.mesh.frustumCulled = false;
  }

  set(x: number, y: number, look: GlowLook, gain = 1, aspect = 1): void {
    this.u.uC.value.set(x, y);
    this.u.uCoreR.value = Math.max(look.coreRadius, 0.35);
    this.u.uCore.value.set(...look.core);
    this.u.uHaloR.value = Math.max(look.haloRadius, 0.5);
    this.u.uHaloIn.value.set(...look.haloInner);
    this.u.uHaloOut.value.set(...look.haloOuter);
    this.u.uR.value = look.haloRadius * (look.extent ?? 14);
    this.u.uGain.value = gain;
    this.u.uAspect.value = aspect;
    this.u.uStar.value = look.star ?? 0;
    this.mesh.visible = gain > 0;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
  }
}

/** Solid rectangle with antialiased edges, rotating about (pivotX, pivotY). */
export class ScreenBar {
  readonly mesh: THREE.Mesh;
  private readonly u: Record<string, THREE.IUniform>;
  constructor() {
    this.u = {
      uPivot: { value: new THREE.Vector2() },
      uSize: { value: new THREE.Vector2(2, 20) },
      uAngle: { value: 0 },
      uColor: { value: new THREE.Vector3(1, 1, 1) },
      uPad: { value: 3 },
    };
    const mat = additive(
      /* glsl */ `
      uniform vec2 uPivot, uSize; uniform float uAngle, uPad;
      varying vec2 vL;
      void main() {
        // Local frame: x across the bar (0..w), y up the bar from the pivot (0..h), screen y is down.
        vec2 l = vec2(mix(-uPad, uSize.x + uPad, position.x), mix(-uPad, uSize.y + uPad, position.y));
        vL = l;
        float c = cos(uAngle), s = sin(uAngle);
        // Bar stands on its pivot (bottom-left corner at angle 0); positive angle tips it to the right.
        vec2 up = vec2(s, -c);
        vec2 across = vec2(c, s);
        vec2 p = uPivot + across * (l.x - uSize.x) + up * l.y;
        gl_Position = projectionMatrix * viewMatrix * vec4(p, 0.0, 1.0);
      }`,
      /* glsl */ `
      uniform vec2 uSize; uniform vec3 uColor;
      varying vec2 vL;
      void main() {
        vec2 a = clamp(vL + 0.5, 0.0, 1.0) * clamp(uSize - vL + 0.5, 0.0, 1.0);
        gl_FragColor = vec4(uColor * a.x * a.y, 1.0);
      }`,
      this.u,
    );
    this.mesh = new THREE.Mesh(unitQuad(), mat);
    this.mesh.frustumCulled = false;
  }
  /** Pivot is the bar's bottom-right corner at rest, so it falls to the right around it. */
  set(pivotX: number, pivotY: number, width: number, height: number, angle: number, color: [number, number, number]): void {
    this.u.uPivot.value.set(pivotX, pivotY);
    this.u.uSize.value.set(width, height);
    this.u.uAngle.value = angle;
    this.u.uColor.value.set(...color);
    this.mesh.visible = color[0] + color[1] + color[2] > 0;
  }
  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
  }
}
