// Virtual cinema lenses. Focal lengths are Super 35 equivalents: the full 16:9 canvas width maps to
// the 24.89 mm Super 35 gate, and the 2.39 letterbox is a crop of that frame.
// Depth of field is the thin-lens model everywhere (bokeh sprites, layer defocus, the post DOF, CPU checks).

import * as THREE from 'three';
import { LINE_Y_FRAC } from './layout';

export const SUPER35_WIDTH_MM = 24.89;

/** What the lens is doing at an instant. */
export interface LensState {
  focalMm: number;
  fStop: number;
  /** Focus distance, metres along the view axis. */
  focus: number;
}

export function fovYDegFromFocal(focalMm: number, canvasAspect: number): number {
  const sensorH = SUPER35_WIDTH_MM / canvasAspect;
  return THREE.MathUtils.radToDeg(2 * Math.atan(sensorH / (2 * focalMm)));
}

/**
 * Perspective camera configured from a focal length. Euler order YXZ (yaw, then pitch, then roll), so
 * aim() works on it; a quaternion set directly ignores the order.
 */
export function makeLensCamera(focalMm: number, width: number, height: number, near = 0.01, far = 500): THREE.PerspectiveCamera {
  const cam = new THREE.PerspectiveCamera(fovYDegFromFocal(focalMm, width / height), width / height, near, far);
  cam.rotation.order = 'YXZ';
  return cam;
}

/** Aim a camera with yaw/pitch/roll (radians). Yaw 0 looks down -Z. */
export function aim(cam: THREE.Camera, yaw: number, pitch: number, roll = 0): void {
  cam.rotation.set(pitch, yaw, roll, 'YXZ');
  cam.updateMatrixWorld(true);
}

/**
 * Upward camera pitch (radians) that puts the true horizon of a level world exactly on the guide line
 * (layout.lineY). For a rectilinear lens with pitch only, the horizon is a straight horizontal line at
 * ndcY = -tan(pitch) / tan(fovY/2); we solve for ndcY = 1 - 2 * yFrac.
 */
export function horizonPitch(fovYDeg: number, yFrac = LINE_Y_FRAC): number {
  const ndcY = 1 - 2 * yFrac;
  return Math.atan(-ndcY * Math.tan(THREE.MathUtils.degToRad(fovYDeg) / 2));
}

/**
 * Circle-of-confusion constant: diameter in canvas px = k * |z - focus| / z for view depth z (m).
 * Pass k to cocRadiusPx() in gl/aperture.glsl.ts.
 */
export function cocK(lens: LensState, canvasWidth: number): number {
  const f = lens.focalMm / 1000;
  const A = f / lens.fStop;
  return ((A * f) / Math.max(lens.focus - f, 1e-4) / (SUPER35_WIDTH_MM / 1000)) * canvasWidth;
}

/** Thin-lens circle of confusion diameter in canvas pixels for a point at `dist` metres. */
export function cocPixels(dist: number, lens: LensState, canvasWidth: number): number {
  return (cocK(lens, canvasWidth) * Math.abs(dist - lens.focus)) / Math.max(dist, 1e-4);
}

/** Canvas pixels per metre at view depth `z` for a camera of vertical fov `fovYDeg` and canvas height `height`. */
export const pxPerMetre = (z: number, fovYDeg: number, height: number): number => height / (2 * Math.tan(THREE.MathUtils.degToRad(fovYDeg) / 2) * z);

/** Project a world point to canvas pixels (y down). Returns z = NDC depth; behind-camera points get z > 1. */
export function projectToCanvas(p: THREE.Vector3, cam: THREE.Camera, width: number, height: number): THREE.Vector3 {
  const v = p.clone().project(cam);
  return new THREE.Vector3((v.x * 0.5 + 0.5) * width, (1 - (v.y * 0.5 + 0.5)) * height, v.z);
}

/** World-space ray through a canvas pixel. */
export function canvasRay(x: number, y: number, cam: THREE.Camera, width: number, height: number): THREE.Ray {
  const ndc = new THREE.Vector3((x / width) * 2 - 1, 1 - (y / height) * 2, 0.5);
  const origin = new THREE.Vector3().setFromMatrixPosition(cam.matrixWorld);
  const dir = ndc.unproject(cam).sub(origin).normalize();
  return new THREE.Ray(origin, dir);
}
