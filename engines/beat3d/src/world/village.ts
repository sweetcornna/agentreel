// A village drawn, not built: ink on paper (墨) or lines of light at dawn (晨), one model for both.
//
// Each house is drawn the way a 白描 draughtsman puts down a farmhouse seen from a little above, with
// the shared brush (world/brush.ts): the roof ridge (the strongest line, its ends lifted and curled
// into a small 鸱吻), the front eave (its ends turned up), the verges running from the eave corners up
// to the ridge ends and on down the back, faint tile rows (瓦垄) down the front slope, the walls as pale
// lines with a firmer line where they meet the ground, the door and the window frame. The brush lifts
// here and there along every line, so the drawing breathes. In ink mode a 淡彩 wash can go under the
// lines (`wash`: the roof slope in pale ink, the front wall warm, the side wall in shade, the door
// dark). A lit window is a small block of colour on the front wall, `setLit(i, 0..1)` per house: a
// warm dab in ink, gold leaf in crisp light (`crisp`: 描金 on lacquer, hard-edged lines, no glow).
//
// The houses also exist as depth-only solids (`solids`), to put into the caller's Occluder so a house
// hides the lines behind it (the far gable, the back eave, the houses behind). For a village seen as
// lights only (at dawn, far away), `lightClusters()` gives a few warm points per house at its windows and door.
//
// `mode: 'ink'` draws on Paper (strokes absorb); 'light' adds light (night or gold). Deterministic
// (seeded by key): the per-house draws (sizes, door, window) come first and in a fixed order, so
// callers that only use the lights (`lightClusters`) see the same houses whatever is drawn.

import * as THREE from 'three';
import type { StageContext } from '../core/types';
import { Rng } from '../lib/rng';
import { fbm2 } from '../lib/noise';
import { LightBrush, type InkLook, type InkMode, type LightPoint, type RGB, type Stroke } from './brush';

/** Window washes per house. */
const WIN_BANDS = 5;

export interface HousePlacement {
  x: number;
  z: number;
  /** Yaw (radians) of the house's front (local +Z), 0 = front faces +Z. */
  yaw: number;
  /** Sizes (m): width along the ridge, depth, wall height; roof pitch (rad). Random if omitted. */
  width?: number;
  depth?: number;
  height?: number;
  pitch?: number;
}

export interface VillageWash {
  /** Absorptions: the front roof slope, the front wall, the side walls (in shade), the door. */
  roof: RGB;
  wall: RGB;
  side: RGB;
  door?: RGB;
}

export interface VillageSpec {
  key: string;
  heightAt: (x: number, z: number) => number;
  houses: HousePlacement[];
  mode?: InkMode;
  /** Line colour (ink: what it absorbs; light: the light). */
  color?: RGB;
  /** Ridge (and 鸱吻) colour, default `color`. */
  ridgeColor?: RGB;
  /** Window colour when lit (same convention), per house or one for all. */
  window?: RGB | ((i: number) => RGB);
  /** Line weight in 1080p px at `refDistance` m (default 2 px at 200 m) and how it scales with distance. */
  weight?: number;
  refDistance?: number;
  /** How often the brush lifts along a line, 0..1 (default 0.35). */
  lift?: number;
  /** Line strength (default 1). */
  gain?: number;
  /** Light mode: 描金, hard-edged lines without glow and gold-leaf windows (default false). */
  crisp?: boolean;
  /** Ink mode: 淡彩 washes under the lines (none if omitted). */
  wash?: VillageWash;
  /**
   * Ink mode, 写意 (the language of modern Jiangnan ink painting: white walls, black roofs, points, lines, planes):
   * each roof a solid block of ink, its ridge and eave a little curved, dry at the edges; the walls
   * left as paper with two or three broken strokes (a gable corner, the foot); door and window one
   * dab of ink each. `tone(position)` thins the ink with distance (air perspective), 0..1.
   * Replaces the 白描 lines, the tiles and the `wash`.
   */
  xieyi?: { roof: RGB; dab: RGB; tone?: (p: THREE.Vector3) => number; eye?: [number, number] };
  /**
   * The thin 写意 roof: the ridge at this multiple of the wall height and the eave and ridge both
   * lifted toward the ends by the same small curve (default 1.27 with `xieyi`, otherwise the pitched
   * roof). Give it to a 白描 village drawn over a `xieyi` one so its lines sit on the ink band.
   */
  roofRise?: number;
  /** Tile rows (瓦垄) down the front slopes, strength 0..1 (default 0: none). */
  tiles?: number;
  /** Elevation (rad) of the view the roof washes are sized for (default 0.12). */
  viewElev?: number;
  /** Ink density of the lines (default 0 = linear; ~3 for a dark, clean 白描 line). */
  density?: number;
  /** Depth bias of the lines, fraction of the distance (default 0.01). */
  depthBias?: number;
}

export interface HouseInfo {
  position: THREE.Vector3;
  yaw: number;
  width: number;
  depth: number;
  height: number;
  /** World height of the ridge. */
  ridgeY: number;
  /** Door (ground level, on the front wall) and window centre (on the front wall). */
  door: THREE.Vector3;
  window: THREE.Vector3;
}

export class Village {
  readonly houses: HouseInfo[] = [];
  /** All the drawing of the houses (ridges, eaves, verges, tiles, walls, doors, window frames). */
  readonly lines: LightBrush;
  /** The windows: WIN_BANDS washes per house (gain 0 until lit). */
  readonly windows: LightBrush;
  /** Ink mode with `wash`: the 淡彩 under the lines; with `xieyi`: the roof blocks. */
  readonly washes: LightBrush | null;
  /** Ink mode with `xieyi`: the door and window dabs, two per house (door, window). */
  readonly dabs: LightBrush | null;
  /** Depth-only solids (positions only), for the caller's Occluder. */
  readonly solids: THREE.BufferGeometry[] = [];
  private readonly lit: number[];

  constructor(ctx: StageContext, spec: VillageSpec) {
    const rng = new Rng(`${spec.key}/village`);
    const rx = new Rng(`${spec.key}/detail`);
    const mode = spec.mode ?? 'ink';
    const crisp = mode === 'light' && !!spec.crisp;
    const col: RGB = spec.color ?? (mode === 'ink' ? [0.93, 0.93, 0.94] : [1, 0.82, 0.55]);
    const ridgeCol: RGB = spec.ridgeColor ?? col;
    const weight = spec.weight ?? 2;
    const ref = spec.refDistance ?? 200;
    const liftK = spec.lift ?? 0.35;
    const g0 = spec.gain ?? 1;
    const tiles = spec.tiles ?? 0;
    const elev = spec.viewElev ?? 0.12;
    const strokes: Stroke[] = [];
    const wins: Stroke[] = [];
    const washes: Stroke[] = [];
    const dabs: Stroke[] = [];
    const xy = mode === 'ink' ? spec.xieyi : undefined;
    let n = 0;
    // A brush that lifts: per-point strength from slow noise along the line.
    const lifts = (pts: THREE.Vector3[], seed: number, k: number): number[] =>
      pts.map((_, j) => 1 - k * THREE.MathUtils.smoothstep(fbm2(j / 4 + seed * 13.1, seed * 3.7, 2), 0.05, 0.3));
    // A line, densified so the lifts and the wobble have points to work on.
    const line = (raw: THREE.Vector3[], w: number, g: number, lift: number, taper: [number, number] = [0.12, 0.18], c: RGB = col, round = false): void => {
      const pts = densify(raw, 10);
      const seed = n++;
      strokes.push({
        points: pts,
        color: c,
        gain: g * g0,
        gains: lifts(pts, seed, lift),
        width: w * weight,
        perspective: 0.6,
        refDistance: ref,
        halo: crisp ? 0.015 : mode === 'ink' ? 0.08 : 0.12,
        haloPx: crisp ? 2 : mode === 'ink' ? 1.5 : 5,
        edge: crisp ? 1 : mode === 'ink' ? 0.7 : 0,
        taper,
        cap: round ? 'round' : 'flat',
        pressure: crisp ? 0.3 : 0.5,
        pressureScale: 3,
        wobble: crisp ? 0.1 : 0.3,
        dry: mode === 'ink' ? 0.15 : 0,
        dryScale: 2,
        seed: `${spec.key}/v/${seed}`,
      });
    };
    const wash = (pts: THREE.Vector3[], c: RGB, widthM: number, g: number, key: string, dry = 0.35, gains?: number[]): void => {
      washes.push({ points: densify(pts, 8), color: c, gain: g, gains: gains ? densifyArr(gains, pts.length, 8) : undefined, widthM, halo: 0, haloPx: 1, edge: 0.45, taper: [0.03, 0.03], pressure: 0.25, pressureScale: 2.5, dry, dryScale: 1.6, wobble: 0.15, seed: key });
    };

    for (const [hi, p] of spec.houses.entries()) {
      const w = p.width ?? 7 + 5 * rng.next();
      const d = p.depth ?? 5 + 1.5 * rng.next();
      const h = p.height ?? 2.6 + 0.8 * rng.next();
      const pitch = p.pitch ?? 0.45 + 0.15 * rng.next();
      const gy = spec.heightAt(p.x, p.z);
      const c = Math.cos(p.yaw);
      const s = Math.sin(p.yaw);
      // Local (x along the ridge, y up, z out of the front) to world.
      const W = (x: number, y: number, z: number): THREE.Vector3 => new THREE.Vector3(p.x + c * x + s * z, gy + y, p.z - s * x + c * z);
      // 写意: a thin roof — its rise about 0.27 of the wall, so the wall stands at least twice the band.
      const rise = spec.roofRise ?? (xy ? 1.27 : 0);
      const ridgeH = rise ? h * rise : h + (d / 2 + 0.4) * Math.tan(pitch);
      const ov = 0.45; // eave overhang
      const hw = w / 2 + ov;
      const ez = d / 2 + ov;
      const up = (x: number, k: number): number => k * Math.pow(Math.min(1.2, Math.abs(x) / hw), 4);
      // The thin roof's curve: ends lifted a little (飞檐), the middle a little hollow.
      const arc = (x: number): number => -0.08 * (1 - Math.min(1, (x / hw) ** 2));
      const lift = (x: number): number => up(x, 0.3) + arc(x);
      const dx = (rng.next() - 0.5) * w * 0.3;
      const wx = dx + (dx > 0 ? -1 : 1) * (1.4 + 0.6 * rng.next());

      const fz = d / 2 + 0.03;
      if (xy) {
        // ---- 写意 (Jiangnan). The roof a thin band of ink, its ends lifted a little (飞檐), firm at
        //      the eave and paler toward the ridge; the walls the paper, stood up by pale corners, the
        //      foot, and a very light wash on the side away from the sunset (the front); a turned
        //      house shows its gable white, only its top edge drawn.
        const tone = clamp01(xy.tone ? xy.tone(W(0, 0, 0)) : 1);
        const proj = (ridgeH - h) * Math.cos(elev) + ez * Math.sin(elev);
        const band = (t: number, wm: number, g: number, dry: number, key: string, x0 = hw, edge = 0.9): void => {
          const pts: THREE.Vector3[] = [];
          for (let i = 0; i <= 12; i++) {
            const x = -x0 + (2 * x0 * i) / 12;
            pts.push(W(x, h + t * (ridgeH - h) + lift(x), ez * (1 - t)));
          }
          washes.push({ points: pts, color: xy.roof, gain: tone * g, widthM: wm, halo: 0, haloPx: 1, edge, taper: [0, 0], cap: 'flat', pressure: 0.1, pressureScale: w * 0.5, dry, dryScale: w * 0.2, wobble: 0.1, seed: `${spec.key}/roof/${hi}/${key}` });
        };
        // One band of ink from the eave up, and a paler soft one on top toward the ridge (tucked in
        // at the ends, so the ends stay one clean lifted cut).
        band(0.33, proj * 0.8, 1, 0.06, 'slope', hw, 0.85);
        // (the pale band only near: past ~150 m it would read as a sheet laid over the roof)
        const [ex0, ez1] = xy.eye ?? [0, 0];
        const near = 1 - THREE.MathUtils.smoothstep(Math.hypot(p.x - ex0, p.z - ez1), 95, 150);
        if (near > 0) band(0.8, proj * 0.42, 0.22 * near, 0.1, 'ridge', hw - 0.3, 0.45);
        band(0, 0.17, 1.2, 0.08, 'eave', hw + 0.12, 0.97);
        // The front wall: a very light wash (the side away from the light), then the two corners,
        // pale and broken but most of their length there, and the foot.
        const wy = (h + 0.14) / 2;
        washes.push({ points: [W(-w / 2 + 0.15, wy, d / 2 + 0.04), W(0, wy, d / 2 + 0.04), W(w / 2 - 0.15, wy, d / 2 + 0.04)], color: xy.roof, gain: 0.08 * tone, widthM: h - 0.3, halo: 0, haloPx: 1, edge: 0.6, taper: [0, 0], cap: 'flat', pressure: 0.2, pressureScale: 2, dry: 0.25, dryScale: 1.5, wobble: 0.1, seed: `${spec.key}/wall/${hi}` });
        const corner = (x: number, z: number, g: number, key: number): void =>
          line([W(x, h - 0.05, z), W(x, h * 0.66, z), W(x, h * 0.33, z), W(x, 0.06, z)], 0.62, g * tone, 0.3 + 0.1 * (key % 2), [0.06, 0.04]);
        corner(-w / 2, d / 2 + 0.03, 0.42, 0);
        corner(w / 2, d / 2 + 0.03, 0.42, 1);
        const fa = -w / 2 + rx.range(0, 0.08) * w;
        const fb = w / 2 - rx.range(0, 0.08) * w;
        line([W(fa, 0.06, d / 2 + 0.03), W((fa + fb) / 2, 0.05, d / 2 + 0.03), W(fb, 0.06, d / 2 + 0.03)], 0.85, 0.6 * tone, liftK * 0.6, [0.06, 0.1]);
        // A turned house: its gable stays white — the back corner, the side foot, and a fine dark line
        // along the gable's top edge only.
        // Which gable faces the eye (x, z), and how squarely.
        const [ex, ez0] = xy.eye ?? [0, 0];
        const tx = ex - p.x;
        const tz = ez0 - p.z;
        const side = (c * tx - s * tz) / Math.hypot(tx, tz);
        if (Math.abs(side) > 0.1) {
          const sv = side > 0 ? 1 : -1;
          corner((sv * w) / 2, -d / 2, 0.36, 2);
          line([W((sv * w) / 2, 0.06, d / 2), W((sv * w) / 2, 0.06, -d / 2)], 0.75, 0.5 * tone, liftK * 0.6, [0.06, 0.12]);
          const gx = sv * hw;
          line([W(gx, h + lift(gx), ez), W(gx, (h + ridgeH) / 2 + lift(gx) * 0.5, ez / 2), W(gx, ridgeH, 0), W(gx, ridgeH - 0.3 * (ridgeH - h), -0.3 * ez)], 0.7, 0.85 * tone, liftK * 0.3, [0.02, 0.35]);
        }
        // Door and window: a square dab of ink each, flat-ended, a little dry at the edges; the door
        // stands on the foot line, the window smaller.
        dabs.push({ points: [W(dx, 0.06, fz), W(dx, 0.95, fz), W(dx, 1.85, fz)], color: xy.dab, gain: tone, widthM: 0.9, halo: 0, haloPx: 1, edge: 0.88, taper: [0, 0], cap: 'flat', pressure: 0.15, pressureScale: 1, dry: 0.22, dryScale: 1.4, wobble: 0.08, seed: `${spec.key}/door/${hi}` });
        dabs.push({ points: [W(wx - 0.28, 1.45, fz), W(wx, 1.45, fz), W(wx + 0.28, 1.45, fz)], color: xy.dab, gain: tone, widthM: 0.56, halo: 0, haloPx: 1, edge: 0.88, taper: [0, 0], cap: 'flat', pressure: 0.15, pressureScale: 1, dry: 0.22, dryScale: 1.2, wobble: 0.08, seed: `${spec.key}/windab/${hi}` });
      } else {
        // ---- the roof. The ridge: the strongest line, lifted at both ends, a small curl (鸱吻) on each.
        const ridge: THREE.Vector3[] = [];
        const eave: THREE.Vector3[] = [];
        // (with `roofRise`, the thin roof: ridge and eave on the ink band's curve)
        const ridgeUp = (x: number): number => (rise ? lift(x) : up(x, 0.4));
        const eaveY = (x: number): number => (rise ? h + lift(x) : h - 0.05 + up(x, 0.35));
        for (let i = 0; i <= 12; i++) {
          const x = -hw - 0.15 + ((2 * hw + 0.3) * i) / 12;
          ridge.push(W(x, ridgeH + ridgeUp(x), 0));
          const xe = -hw + (2 * hw * i) / 12;
          eave.push(W(xe, eaveY(xe), ez));
        }
        line(ridge, 1.2, 1, liftK * 0.35, [0, 0], ridgeCol, true);
        for (const sx of [-1, 1]) {
          const e = sx < 0 ? ridge[0] : ridge[ridge.length - 1];
          const lc = (x: number, y: number): THREE.Vector3 => W(sx * (hw + 0.15) + sx * x, ridgeH + ridgeUp(hw + 0.15) + y, 0);
          const ck = rise ? 0.6 : 1; // (smaller on the thin roof)
          line([e, lc(0.08 * ck, 0.28 * ck), lc(-0.08 * ck, 0.5 * ck), lc(-0.3 * ck, 0.52 * ck)], 0.9, 0.9, 0, [0, 0.45], ridgeCol, true);
        }
        // The front eave, its ends turned up a little less than the ridge's.
        line(eave, 0.85, 0.85, liftK * 0.6, [0.04, 0.04], col, true);
        // Verges: from each eave corner up to the ridge end, the slope a little hollow (凹曲), and on
        // down the back (hidden by the house when that side is away from us).
        for (const sx of [-1, 1]) {
          const re = sx < 0 ? ridge[0] : ridge[ridge.length - 1];
          const fe = sx < 0 ? eave[0] : eave[eave.length - 1];
          const mid = (a: THREE.Vector3, b: THREE.Vector3): THREE.Vector3 => a.clone().lerp(b, 0.5).add(new THREE.Vector3(0, rise ? -0.05 : -0.14, 0));
          line([fe, mid(fe, re), re], 0.75, 0.75, liftK * 0.5, [0.06, 0.02]);
          const be = W(sx * hw, eaveY(hw), -ez);
          line([re, mid(re, be), be], 0.7, 0.6, liftK, [0.02, 0.1]);
        }
        // Tile rows (瓦垄) down the front slope: fine, faint, often lifting.
        if (tiles > 0) {
          const nT = Math.max(4, Math.round((2 * hw - 0.8) / 0.62));
          for (let k = 0; k <= nT; k++) {
            const x = -hw + 0.4 + ((2 * hw - 0.8) * k) / nT;
            const top = W(x, ridgeH + ridgeUp(x) - 0.08, 0.05);
            const bot = W(x, eaveY(x) + 0.03, ez - 0.05);
            const mid = top.clone().lerp(bot, 0.5).add(new THREE.Vector3(0, rise ? -0.04 : -0.12, 0));
            line([top, mid, bot], 0.42, tiles * (0.3 + 0.25 * rx.next()), 0.75, [0.1, 0.06]);
          }
        }

        // ---- the walls: pale corners, a firmer line along the foot, the side's back corner.
        for (const sx of [-1, 1]) {
          line([W((sx * w) / 2, h - 0.1, d / 2), W((sx * w) / 2, h * 0.5, d / 2), W((sx * w) / 2, 0.08, d / 2)], 0.55, 0.42, liftK * 1.4, [0.15, 0.1]);
          line([W((sx * w) / 2, h - 0.1, -d / 2), W((sx * w) / 2, 0.08, -d / 2)], 0.5, 0.32, liftK * 1.6, [0.15, 0.1]);
          line([W((sx * w) / 2, 0.06, d / 2), W((sx * w) / 2, 0.06, -d / 2)], 0.7, 0.55, liftK, [0.1, 0.2]);
        }
        line([W(-w / 2, 0.06, d / 2 + 0.02), W(0, 0.05, d / 2 + 0.02), W(w / 2, 0.06, d / 2 + 0.02)], 0.8, 0.7, liftK * 0.5, [0.05, 0.08]);
        // The door frame and the window frame.
        line([W(dx - 0.5, 0.06, fz), W(dx - 0.5, 2.0, fz), W(dx + 0.5, 2.0, fz), W(dx + 0.5, 0.06, fz)], 0.55, 0.65, liftK * 0.3, [0.03, 0.03]);
        line([W(wx - 0.5, 1.05, fz), W(wx - 0.5, 1.85, fz), W(wx + 0.5, 1.85, fz), W(wx + 0.5, 1.05, fz), W(wx - 0.5, 1.05, fz)], 0.4, 0.45, liftK * 0.5, [0.02, 0.02]);

        // ---- 淡彩 under the lines (ink): the roof slope in rows from the eave up, the walls, the door.
        if (mode === 'ink' && spec.wash) {
          const wc = spec.wash;
          const proj = (ridgeH - h) * Math.cos(elev) + ez * Math.sin(elev);
          const rows = 3;
          for (let r = 0; r < rows; r++) {
            const t = (r + 0.5) / rows;
            const pts: THREE.Vector3[] = [];
            for (let i = 0; i <= 6; i++) {
              const x = -hw + 0.25 + ((2 * hw - 0.5) * i) / 6;
              pts.push(W(x, h + t * (ridgeH - h) + up(x, 0.37) - 0.04, ez * (1 - t)));
            }
            wash(pts, wc.roof, (proj / rows) * 1.55, r === 0 ? 1.15 : r === 1 ? 0.85 : 0.95, `${spec.key}/roofwash/${hi}/${r}`, 0.4);
          }
          const wy = (h - 0.3) / 2;
          wash([W(-w / 2 + 0.05, wy, d / 2 + 0.04), W(w / 2 - 0.05, wy, d / 2 + 0.04)], wc.wall, h - 0.35, 1, `${spec.key}/wallwash/${hi}`, 0.45);
          for (const sx of [-1, 1]) wash([W((sx * w) / 2 + sx * 0.04, wy, d / 2 - 0.05), W((sx * w) / 2 + sx * 0.04, wy, -d / 2 + 0.05)], wc.side, h - 0.35, 1, `${spec.key}/sidewash/${hi}/${sx}`, 0.3);
          if (wc.door) wash([W(dx, 0.08, fz + 0.01), W(dx, 1.92, fz + 0.01)], wc.door, 0.92, 1, `${spec.key}/doorwash/${hi}`, 0.2);
        }
      }

      // ---- the window: a small block of colour (five thin washes stacked; one wide stroke would
      //      come to points). Ink: a warm dab, soft at its edge; crisp light: gold leaf.
      const win = W(wx, 1.45, d / 2 + 0.05);
      const wcol = typeof spec.window === 'function' ? spec.window(hi) : (spec.window ?? (mode === 'ink' ? ([0.02, 0.38, 0.6] as RGB) : ([1.2, 0.75, 0.32] as RGB)));
      for (let r = 0; r < WIN_BANDS; r++) {
        const y = 1.12 + (0.66 * r) / (WIN_BANDS - 1);
        wins.push({
          points: crisp ? [W(wx - 0.44, y, d / 2 + 0.06), W(wx, y, d / 2 + 0.06), W(wx + 0.44, y, d / 2 + 0.06)] : [W(wx - 0.52, y, d / 2 + 0.06), W(wx, y, d / 2 + 0.06), W(wx + 0.52, y, d / 2 + 0.06)],
          color: wcol,
          gain: crisp ? 1 : r === 0 || r === WIN_BANDS - 1 ? 0.9 : 1.4,
          widthM: crisp ? 0.2 : 0.34,
          halo: crisp ? 0 : mode === 'ink' ? 0.45 : 0.7,
          haloPx: crisp ? 1 : mode === 'ink' ? 8 : 9,
          edge: crisp ? 1 : 0,
          taper: crisp ? [0, 0] : [0.04, 0.04],
          pressure: crisp ? 0 : 0.2,
          pressureScale: 0.4,
          wobble: crisp ? 0 : 0.3,
          seed: `${spec.key}/win/${hi}/${r}`,
        });
      }
      this.houses.push({ position: W(0, 0, 0), yaw: p.yaw, width: w, depth: d, height: h, ridgeY: gy + ridgeH, door: W(dx, 0, d / 2), window: win });

      // Solid: the walls as a box and the roof as a prism (slightly inside the lines).
      const pts: number[] = [];
      const quad = (a: THREE.Vector3, b: THREE.Vector3, c2: THREE.Vector3, d2: THREE.Vector3) => pts.push(...a.toArray(), ...b.toArray(), ...c2.toArray(), ...a.toArray(), ...c2.toArray(), ...d2.toArray());
      const tri = (a: THREE.Vector3, b: THREE.Vector3, c2: THREE.Vector3) => pts.push(...a.toArray(), ...b.toArray(), ...c2.toArray());
      const x0 = -w / 2 + 0.1;
      const x1 = w / 2 - 0.1;
      const z0 = -d / 2 + 0.1;
      const z1 = d / 2 - 0.1;
      const hh = h - 0.2;
      quad(W(x0, 0, z1), W(x1, 0, z1), W(x1, hh, z1), W(x0, hh, z1));
      quad(W(x1, 0, z0), W(x0, 0, z0), W(x0, hh, z0), W(x1, hh, z0));
      quad(W(x0, 0, z0), W(x0, 0, z1), W(x0, hh, z1), W(x0, hh, z0));
      quad(W(x1, 0, z1), W(x1, 0, z0), W(x1, hh, z0), W(x1, hh, z1));
      const rh = ridgeH - 0.2;
      quad(W(-hw + 0.3, hh, z1 + ov - 0.2), W(hw - 0.3, hh, z1 + ov - 0.2), W(hw - 0.3, rh, 0), W(-hw + 0.3, rh, 0));
      quad(W(hw - 0.3, hh, z0 - ov + 0.2), W(-hw + 0.3, hh, z0 - ov + 0.2), W(-hw + 0.3, rh, 0), W(hw - 0.3, rh, 0));
      tri(W(x0, hh, z1), W(x0, rh, 0), W(x0, hh, z0));
      tri(W(x1, hh, z0), W(x1, rh, 0), W(x1, hh, z1));
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
      this.solids.push(g);
    }
    this.lines = new LightBrush(ctx, strokes, { mode, renderOrder: 4, depthBias: spec.depthBias ?? 0.01, density: spec.density ?? 0 });
    this.windows = new LightBrush(ctx, wins, { mode, renderOrder: 5, depthBias: 0.02 });
    this.washes = washes.length ? new LightBrush(ctx, washes, { mode, renderOrder: 3, depthBias: 0.012, density: xy ? 6 : 0, grain: xy ? 0.08 : 0.3 }) : null;
    this.dabs = dabs.length ? new LightBrush(ctx, dabs, { mode, renderOrder: 5, depthBias: 0.015, density: 5, grain: 0.1 }) : null;
    this.lit = spec.houses.map(() => 0);
    this.windows.setAll({ gain: 0 });
  }

  /** Window of house i: 0 dark .. 1 fully lit. */
  setLit(i: number, level: number): void {
    this.lit[i] = level;
    for (let r = 0; r < WIN_BANDS; r++) this.windows.set(i * WIN_BANDS + r, { gain: level });
  }

  /** The drawing as a whole: strength of all lines (e.g. fading as the paper darkens). */
  setLineGain(g: number): void {
    this.lines.setAll({ gain: g });
  }

  /** The 淡彩 (or the roof blocks) as a whole (ink mode with `wash` or `xieyi`). */
  setWashGain(g: number): void {
    this.washes?.setAll({ gain: g });
  }

  /** House i's door and window dabs (`xieyi`): 0..1 each. */
  setDabs(i: number, door: number, window: number): void {
    this.dabs?.set(2 * i, { gain: door });
    this.dabs?.set(2 * i + 1, { gain: window });
  }

  update(look: InkLook = {}): void {
    this.lines.update(look);
    this.windows.update(look);
    this.washes?.update(look);
    this.dabs?.update(look);
  }

  get objects(): THREE.Object3D[] {
    return [this.washes?.object, this.dabs?.object, this.lines.object, this.windows.object].filter((o): o is THREE.Mesh => !!o);
  }

  /**
   * A few warm points per house (window, door, a lamp in the yard): the village as lights only.
   * `seed` keys the scatter; colours and gains are a starting point for the caller to set per frame.
   */
  lightClusters(seed: string, color: RGB = [1.2, 0.72, 0.34]): LightPoint[] {
    const rng = new Rng(`${seed}/clusters`);
    const out: LightPoint[] = [];
    for (const h of this.houses) {
      const k = 1 + Math.floor(rng.next() * 4);
      out.push({ position: h.window.toArray() as [number, number, number], color, size: 1.6 + rng.next(), gain: 1.4 + rng.next(), halo: 0.5, haloPx: 8 });
      for (let i = 1; i < k; i++) {
        const p = h.door.clone().add(new THREE.Vector3(rng.range(-3, 3), 0.8 + 1.2 * rng.next(), rng.range(-1, 3)));
        out.push({ position: p.toArray() as [number, number, number], color, size: 1.0 + 0.8 * rng.next(), gain: 0.5 + rng.next(), halo: 0.4, haloPx: 6 });
      }
    }
    return out;
  }

  dispose(): void {
    this.lines.dispose();
    this.windows.dispose();
    this.washes?.dispose();
    this.dabs?.dispose();
    for (const g of this.solids) g.dispose();
  }
}

const clamp01 = (x: number): number => Math.min(1, Math.max(0, x));

/** At least `min` points along a polyline (the lifts, the pressure and the wobble need points). */
function densify(pts: THREE.Vector3[], min: number): THREE.Vector3[] {
  if (pts.length >= min) return pts;
  const seg = pts.length - 1;
  const per = Math.ceil((min - 1) / seg);
  const out: THREE.Vector3[] = [];
  for (let i = 0; i < seg; i++) for (let j = 0; j < per; j++) out.push(pts[i].clone().lerp(pts[i + 1], j / per));
  out.push(pts[seg].clone());
  return out;
}

/** The per-point values of a polyline densified as `densify` does. */
function densifyArr(a: number[], n: number, min: number): number[] {
  if (n >= min) return a;
  const seg = n - 1;
  const per = Math.ceil((min - 1) / seg);
  const out: number[] = [];
  for (let i = 0; i < seg; i++) for (let j = 0; j < per; j++) out.push(a[i] + ((a[i + 1] - a[i]) * j) / per);
  out.push(a[seg]);
  return out;
}
