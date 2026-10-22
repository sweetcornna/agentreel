// WorldGlow: a character light (or lamp) drawn in the 3D world instead of the screen layer, so that
// nearer geometry (grass) can hide it and the post DOF blurs it at its own depth. Same look as
// screen.ts GlowPoint (sizes in canvas px, GlowLook), as a camera-facing billboard at a world point.
// A small depth-only disc at the light's depth is drawn as well, so the DOF pass reads the light's
// distance there rather than the far sky behind it.

import * as THREE from 'three';
import type { GlowLook } from './screen';

const VERT = /* glsl */ `
uniform vec3 uCenter;
uniform float uR;        // quad half-size, canvas px
uniform vec2 uRes;
varying vec2 vD;
void main() {
  vec4 c = projectionMatrix * viewMatrix * vec4(uCenter, 1.0);
  vec2 d = (position.xy - 0.5) * 2.0 * uR;
  vD = vec2(d.x, -d.y);
  c.xy += d / uRes * 2.0 * c.w;
  gl_Position = c;
}
`;

const FRAG = /* glsl */ `
uniform float uCoreR, uHaloR, uGain, uR, uStar;
uniform vec3 uCore, uHaloIn, uHaloOut;
varying vec2 vD;
void main() {
  float r = length(vD);
  float core = exp(-(r * r) / (uCoreR * uCoreR));
  float q = r / uHaloR;
  float halo = 1.0 / pow(1.0 + q * q, 1.5);
  vec3 hc = mix(uHaloIn, uHaloOut, smoothstep(0.0, 3.0, q));
  float edge = 1.0 - smoothstep(0.7 * uR, uR, r);
  float w = 0.45 * uCoreR;
  float L = 0.9 * uHaloR;
  float star = uStar * (exp(-abs(vD.y) / w - abs(vD.x) / L) + exp(-abs(vD.x) / w - abs(vD.y) / L));
  gl_FragColor = vec4((uCore * core + hc * halo + mix(uCore, uHaloIn, 0.5) * star) * uGain * edge, 1.0);
}
`;

const DEPTH_FRAG = /* glsl */ `
uniform float uDiscR;
varying vec2 vD;
void main() {
  if (length(vD) > uDiscR) discard;
  gl_FragColor = vec4(0.0);
}
`;

export class WorldGlow {
  readonly mesh: THREE.Mesh;
  readonly depthMesh: THREE.Mesh;
  private readonly u: Record<string, THREE.IUniform>;
  private readonly du: Record<string, THREE.IUniform>;

  constructor(width: number, height: number) {
    const quad = new THREE.BufferGeometry();
    quad.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]), 3));
    quad.setIndex([0, 1, 2, 0, 2, 3]);
    this.u = {
      uCenter: { value: new THREE.Vector3() },
      uR: { value: 10 },
      uRes: { value: new THREE.Vector2(width, height) },
      uCoreR: { value: 1 },
      uHaloR: { value: 4 },
      uGain: { value: 1 },
      uStar: { value: 0 },
      uCore: { value: new THREE.Vector3() },
      uHaloIn: { value: new THREE.Vector3() },
      uHaloOut: { value: new THREE.Vector3() },
    };
    this.mesh = new THREE.Mesh(
      quad,
      new THREE.ShaderMaterial({
        vertexShader: VERT,
        fragmentShader: FRAG,
        uniforms: this.u,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthTest: true,
        depthWrite: false,
        toneMapped: false,
      }),
    );
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 10;
    this.du = { uCenter: this.u.uCenter, uR: { value: 4 }, uRes: this.u.uRes, uDiscR: { value: 3 } };
    this.depthMesh = new THREE.Mesh(
      quad,
      new THREE.ShaderMaterial({
        vertexShader: VERT,
        fragmentShader: DEPTH_FRAG,
        uniforms: this.du,
        colorWrite: false,
        depthWrite: true,
        depthTest: true,
        // Drawn last (transparent pass), so it never hides other additive layers behind the light.
        transparent: true,
      }),
    );
    this.depthMesh.frustumCulled = false;
    this.depthMesh.renderOrder = 1000;
  }

  /** Place at a world point with a GlowLook (canvas px). discR: radius (px) of the depth disc (default 0.6 x halo). */
  set(position: THREE.Vector3, look: GlowLook, gain = 1, discR?: number): void {
    this.u.uCenter.value.copy(position);
    this.u.uCoreR.value = Math.max(look.coreRadius, 0.35);
    this.u.uHaloR.value = Math.max(look.haloRadius, 0.5);
    this.u.uR.value = look.haloRadius * (look.extent ?? 12);
    this.u.uCore.value.set(...look.core);
    this.u.uHaloIn.value.set(...look.haloInner);
    this.u.uHaloOut.value.set(...look.haloOuter);
    this.u.uGain.value = gain;
    this.u.uStar.value = look.star ?? 0;
    const d = discR ?? look.haloRadius * 0.6;
    this.du.uR.value = d + 1;
    this.du.uDiscR.value = d;
    this.mesh.visible = gain > 0;
    this.depthMesh.visible = gain > 0 && d > 0;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
    (this.depthMesh.material as THREE.Material).dispose();
  }
}
