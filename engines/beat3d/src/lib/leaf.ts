// Rounded rectangles with a radius per corner: the geometry of a leaf (two big corners on one diagonal,
// two small on the other), of rounded cards, slabs and rings of light. lib/aperture.ts builds the leaf
// aperture from sdRoundRect(); the haze's ring of light (world/haze.ts) and the dust lit by it use it
// directly; roundRectOutline() gives outlines to extrude (slabs, bevels).
// Coordinates here are y-up (as in GL).

export interface CornerRadii {
  tl: number;
  tr: number;
  br: number;
  bl: number;
}

/** Area of a rounded rectangle of half-size (hx, hy). */
export function roundedRectArea(hx: number, hy: number, r: CornerRadii): number {
  return 4 * hx * hy - (1 - Math.PI / 4) * (r.tl * r.tl + r.tr * r.tr + r.br * r.br + r.bl * r.bl);
}

/** Signed distance to a rounded rectangle centred at 0 with half-size (hx, hy), y up (Quilez). */
export function sdRoundRect(x: number, y: number, hx: number, hy: number, r: CornerRadii): number {
  const rr = x > 0 ? (y > 0 ? r.tr : r.br) : y > 0 ? r.tl : r.bl;
  const qx = Math.abs(x) - hx + rr;
  const qy = Math.abs(y) - hy + rr;
  return Math.min(Math.max(qx, qy), 0) + Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) - rr;
}

/**
 * Outline of a rounded rectangle centred at 0, inset by `inset` (radii shrink with it), as a closed
 * polyline, counter-clockwise from the middle of the right edge. Every corner gets `perCorner` + 1
 * points, so outlines with different insets have the same count and can be stitched into rings.
 */
export function roundRectOutline(hx: number, hy: number, r: CornerRadii, inset: number, perCorner: number): [number, number][] {
  const ix = Math.max(hx - inset, 1e-5);
  const iy = Math.max(hy - inset, 1e-5);
  const out: [number, number][] = [];
  const corners: [number, number, number, number][] = [
    // centre-direction sx, sy, radius, start angle
    [1, 1, r.tr, 0],
    [-1, 1, r.tl, Math.PI / 2],
    [-1, -1, r.bl, Math.PI],
    [1, -1, r.br, (3 * Math.PI) / 2],
  ];
  for (const [sx, sy, rad, a0] of corners) {
    const rc = Math.min(Math.max(rad - inset, 0), ix, iy);
    const cx = sx * (ix - rc);
    const cy = sy * (iy - rc);
    for (let k = 0; k <= perCorner; k++) {
      const a = a0 + (k / perCorner) * (Math.PI / 2);
      out.push([cx + rc * Math.cos(a), cy + rc * Math.sin(a)]);
    }
  }
  return out;
}
