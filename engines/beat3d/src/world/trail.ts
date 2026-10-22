// 光痕: footprints of light on the ground. Each footfall leaves a small soft ellipse of light lying on
// the terrain; it flashes as the foot lands and fades with the half-life given by the persistence
// in effect when it was laid (lib/persistence.ts). A print can be relit and made permanent;
// a relit print can carry two colours (core and rim) so warm and cool visibly mix in it.
// Everything is recomputed from the frame number: no state.
//
// Low cameras (20 cm, 5 cm above the ground) see a flat ellipse as a sliver, so a print can also carry
// a "volume": a glow standing in the grass above it (a camera-facing billboard about the vertical,
// gaussian across, fading upwards), depth-tested so nearer blades hide part of it.

import * as THREE from 'three';
import { trailRemaining } from '../lib/persistence';

export interface Footprint {
  position: THREE.Vector3;
  /** Frame it was laid. */
  frame: number;
  /** Persistence p at the moment it was laid. */
  persistence: number;
  /** Walking direction (radians about +Y, 0 = towards -Z). */
  heading: number;
  color: [number, number, number];
  /** Optional: frame at which it is relit to 100 % and never fades again. */
  relitAt?: number;
  relitColor?: [number, number, number];
  /** Optional rim colour once relit (default = relitColor): the core/rim mix reads as two lights. */
  relitRim?: [number, number, number];
}

export interface TrailOptions {
  /** Height (m) of the standing glow; 0 = off (default). */
  volume?: number;
  /** Brightness of the standing glow relative to the ground ellipse (default 0.35). */
  volumeGain?: number;
}

const VERT = /* glsl */ `
attribute vec3 iPos;
attribute float iHeading;
attribute vec3 iColor;
attribute vec3 iRim;
uniform vec2 uSize;
varying vec2 vQ;
varying vec3 vCol;
varying vec3 vRim;
void main() {
  vQ = position.xy;
  float c = cos(iHeading), s = sin(iHeading);
  vec3 along = vec3(s, 0.0, -c);
  vec3 across = vec3(c, 0.0, s);
  vec3 p = iPos + across * position.x * uSize.x + along * position.y * uSize.y + vec3(0.0, 0.03, 0.0);
  vCol = iColor;
  vRim = iRim;
  gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
}
`;
const FRAG = /* glsl */ `
varying vec2 vQ;
varying vec3 vCol;
varying vec3 vRim;
void main() {
  float r2 = dot(vQ, vQ);
  float g = exp(-4.0 * r2);
  vec3 col = mix(vCol, vRim, smoothstep(0.05, 0.6, r2));
  gl_FragColor = vec4(col * g * (1.0 - smoothstep(0.8, 1.0, r2)), 1.0);
}
`;

const VOL_VERT = /* glsl */ `
attribute vec3 iPos;
attribute vec3 iColor;
attribute vec3 iRim;
uniform vec2 uSize;     // half width (m), height (m)
varying vec2 vQ;
varying vec3 vCol;
varying vec3 vRim;
void main() {
  // Billboard about the vertical axis through the print.
  vec3 toEye = cameraPosition - iPos;
  vec3 side = normalize(vec3(-toEye.z, 0.0, toEye.x) + vec3(1e-5, 0.0, 0.0));
  vQ = vec2(position.x, position.y * 0.5 + 0.5);
  vec3 p = iPos + side * position.x * uSize.x * 1.3 + vec3(0.0, vQ.y * uSize.y, 0.0);
  vCol = iColor;
  vRim = iRim;
  gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
}
`;
const VOL_FRAG = /* glsl */ `
uniform float uGain;
varying vec2 vQ;
varying vec3 vCol;
varying vec3 vRim;
void main() {
  // Both profiles reach exactly zero at the quad's edges (no visible rectangle).
  float across = exp(-5.0 * vQ.x * vQ.x) * (1.0 - vQ.x * vQ.x);
  float up = exp(-3.2 * vQ.y) * (1.0 - vQ.y) * (1.0 - vQ.y) * smoothstep(0.0, 0.06, vQ.y);
  vec3 col = mix(vRim, vCol, across);
  gl_FragColor = vec4(col * across * up * uGain, 1.0);
}
`;

export class FootprintTrail {
  readonly mesh: THREE.Mesh;
  /** Standing glow (only when TrailOptions.volume > 0); add it to the world scene too. */
  readonly volume: THREE.Mesh | null = null;
  private readonly geo: THREE.InstancedBufferGeometry;
  private readonly volGeo: THREE.InstancedBufferGeometry | null = null;
  private readonly pos: Float32Array;
  private readonly heading: Float32Array;
  private readonly color: Float32Array;
  private readonly rim: Float32Array;

  constructor(
    readonly maxPrints: number,
    /** Footprint size in metres (width, length). */
    size: [number, number] = [0.35, 0.5],
    opts: TrailOptions = {},
  ) {
    const base = new THREE.PlaneGeometry(2, 2);
    this.geo = new THREE.InstancedBufferGeometry();
    this.geo.index = base.index;
    this.geo.setAttribute('position', base.getAttribute('position'));
    this.pos = new Float32Array(maxPrints * 3);
    this.heading = new Float32Array(maxPrints);
    this.color = new Float32Array(maxPrints * 3);
    this.rim = new Float32Array(maxPrints * 3);
    const aPos = new THREE.InstancedBufferAttribute(this.pos, 3);
    const aCol = new THREE.InstancedBufferAttribute(this.color, 3);
    const aRim = new THREE.InstancedBufferAttribute(this.rim, 3);
    this.geo.setAttribute('iPos', aPos);
    this.geo.setAttribute('iHeading', new THREE.InstancedBufferAttribute(this.heading, 1));
    this.geo.setAttribute('iColor', aCol);
    this.geo.setAttribute('iRim', aRim);
    this.geo.instanceCount = 0;
    const mat = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: { uSize: { value: new THREE.Vector2(size[0] / 2, size[1] / 2) } },
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: true,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    this.mesh = new THREE.Mesh(this.geo, mat);
    this.mesh.frustumCulled = false;

    if (opts.volume && opts.volume > 0) {
      const vg = new THREE.InstancedBufferGeometry();
      vg.index = base.index;
      vg.setAttribute('position', base.getAttribute('position'));
      vg.setAttribute('iPos', aPos);
      vg.setAttribute('iColor', aCol);
      vg.setAttribute('iRim', aRim);
      vg.instanceCount = 0;
      const vm = new THREE.ShaderMaterial({
        vertexShader: VOL_VERT,
        fragmentShader: VOL_FRAG,
        uniforms: { uSize: { value: new THREE.Vector2(size[0] / 2, opts.volume) }, uGain: { value: opts.volumeGain ?? 0.35 } },
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        depthTest: true,
        side: THREE.DoubleSide,
        toneMapped: false,
      });
      this.volGeo = vg;
      this.volume = new THREE.Mesh(vg, vm);
      this.volume.frustumCulled = false;
    }
  }

  /** Brightness of a print at `frame`: landing flash, then half-life decay (or permanent once relit). */
  static brightness(p: Footprint, frame: number, fps: number): number {
    if (frame < p.frame) return 0;
    if (p.relitAt !== undefined && frame >= p.relitAt) return 1 + 0.6 * Math.exp(-(frame - p.relitAt) / 5);
    const age = (frame - p.frame) / fps;
    const flash = 1 + 0.8 * Math.exp(-(frame - p.frame) / 4);
    return flash * trailRemaining(age, p.persistence);
  }

  /** Colour (core, rim) of a print at `frame`, before brightness. */
  static colors(p: Footprint, frame: number): { core: [number, number, number]; rim: [number, number, number] } {
    if (p.relitAt !== undefined && frame >= p.relitAt) {
      const core = p.relitColor ?? p.color;
      return { core, rim: p.relitRim ?? core };
    }
    return { core: p.color, rim: p.color };
  }

  update(prints: Footprint[], frame: number, fps: number, gain: number): void {
    let n = 0;
    for (const p of prints) {
      if (n >= this.maxPrints) break;
      const b = FootprintTrail.brightness(p, frame, fps) * gain;
      if (b < 1e-4) continue;
      const { core, rim } = FootprintTrail.colors(p, frame);
      this.pos.set([p.position.x, p.position.y, p.position.z], n * 3);
      this.heading[n] = p.heading;
      this.color.set([core[0] * b, core[1] * b, core[2] * b], n * 3);
      this.rim.set([rim[0] * b, rim[1] * b, rim[2] * b], n * 3);
      n++;
    }
    this.geo.instanceCount = n;
    for (const k of ['iPos', 'iHeading', 'iColor', 'iRim']) (this.geo.getAttribute(k) as THREE.InstancedBufferAttribute).needsUpdate = true;
    this.mesh.visible = n > 0;
    if (this.volume && this.volGeo) {
      this.volGeo.instanceCount = n;
      this.volume.visible = n > 0;
    }
  }

  dispose(): void {
    this.geo.dispose();
    (this.mesh.material as THREE.Material).dispose();
    if (this.volume && this.volGeo) {
      this.volGeo.dispose();
      (this.volume.material as THREE.Material).dispose();
    }
  }
}
