// src/lib/aperture.ts 的 GLSL 孪生：镜头光圈（圆、叶形或多边形，运行时由三个 uniform 选）和薄透镜
// 弥散圆。用 ${APERTURE_GLSL} 拼进着色器（有 include guard），材质给 apertureUniforms(spec)；换形状用
// setApertureUniforms()。坐标 y 朝上，单位是弥散圆半径。顺带导出通用的圆角矩形 sdRoundRect()。
// 着色器字符串里的注释保持 ASCII（英文）：WebGL 对着色器源码里的非 ASCII 字符各实现处理不一。

import * as THREE from 'three';
import { compileAperture, defaultAperture, type ApertureSpec } from '../lib/aperture';

export const APERTURE_GLSL = /* glsl */ `
#ifndef B3D_APERTURE
#define B3D_APERTURE
uniform vec4 uApShape;   // kind (0 circle, 1 leaf, 2 polygon), sides, rotation, open
uniform vec4 uApRadii;   // leaf radii (tr, br, tl, bl); polygon: x = corner rounding
uniform vec4 uApNorm;    // k, k2, profile normalisation, extent
// Rounded rectangle, half-size b, radii r = (top-right, bottom-right, top-left, bottom-left), y up.
float sdRoundRect(vec2 p, vec2 b, vec4 r) {
  r.xy = (p.x > 0.0) ? r.xy : r.zw;
  r.x = (p.y > 0.0) ? r.x : r.y;
  vec2 q = abs(p) - b + r.x;
  return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r.x;
}
// Regular polygon of circumradius 1, a vertex on +y, corners rounded by rnd of the apothem.
float sdApPolygon(vec2 p, float n, float rnd) {
  float an = 3.14159265 / n;
  float c = rnd * cos(an);
  float r = 1.0 - c / cos(an);
  float bn = mod(atan(p.x, p.y + 1e-9), 2.0 * an) - an;
  float len = length(p);
  vec2 q = len * vec2(cos(bn), abs(sin(bn)));
  q -= r * vec2(cos(an), sin(an));
  q.y += clamp(-q.y, 0.0, r * sin(an));
  return length(q) * sign(q.x) - c;
}
// Signed distance to the unit-area aperture (negative inside), in CoC radii.
float apertureSd(vec2 p) {
  float k = uApNorm.x > 0.0 ? uApNorm.x : 1.0;
  float k2 = uApNorm.y > 0.0 ? uApNorm.y : 1.0;
  vec2 q = p / k2;
  float cr = cos(-uApShape.z), sr = sin(-uApShape.z);
  q = vec2(cr * q.x - sr * q.y, sr * q.x + cr * q.y);
  int kind = int(uApShape.x + 0.5);
  float a;
  if (kind == 1) a = sdRoundRect(q / k, vec2(1.0), uApRadii) * k;
  else if (kind == 2) a = sdApPolygon(q / k, max(uApShape.y, 3.0), uApRadii.x) * k;
  else a = (length(q / k) - 1.0) * k;
  a = mix(a, length(q) - 1.0, uApShape.w);
  return a * k2;
}
// Brightness across the aperture at depth d (CoC radii) inside the edge; mean 1 over the shape.
float apertureProfile(float d) {
  float rim = exp(-d / 0.1);
  float u = clamp(d / 0.5, 0.0, 1.0);
  float norm = uApNorm.z > 0.0 ? uApNorm.z : 1.0;
  return norm * (0.72 + 0.1 * u + 0.75 * rim + 0.05 * cos(d * 18.0) * u);
}
// Largest radius of the shape, CoC radii.
float apertureExtent() { return uApNorm.w > 0.0 ? uApNorm.w : 1.0; }
// A bokeh of CoC radius R px at offset p (px, y up): profile times an anti-aliased edge of aa px.
float apertureBokeh(vec2 p, float R, float aa) {
  float sd = apertureSd(p / R) * R;
  return apertureProfile(-sd / R) * clamp(0.5 - sd / aa, 0.0, 1.0);
}
// Thin-lens CoC radius in px for view depth z (metres): 0.5 * k * |z - focus| / z, k from lib/lens cocK().
float cocRadiusPx(float z, float k, float focus) {
  return 0.5 * k * abs(z - focus) / max(z, 1e-4);
}
#endif
`;

/** 拼了 APERTURE_GLSL 的材质要的 uniform（默认：片子的光圈）。 */
export function apertureUniforms(spec: ApertureSpec = defaultAperture()): Record<string, THREE.IUniform> {
  const g = compileAperture(spec);
  return {
    uApShape: { value: new THREE.Vector4(...g.shape) },
    uApRadii: { value: new THREE.Vector4(...g.radii) },
    uApNorm: { value: new THREE.Vector4(...g.norm) },
  };
}

/** 把材质的光圈 uniform 换成另一个形状（每帧换也可以：编译结果按值缓存）。 */
export function setApertureUniforms(u: Record<string, THREE.IUniform>, spec: ApertureSpec): void {
  const g = compileAperture(spec);
  (u.uApShape.value as THREE.Vector4).set(...g.shape);
  (u.uApRadii.value as THREE.Vector4).set(...g.radii);
  (u.uApNorm.value as THREE.Vector4).set(...g.norm);
}
