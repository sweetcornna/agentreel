// S02「光聚成字」与 S03 开头的「字化为光」。
// 3.1（H1）：光丝从镜头后方两侧汇进来，落在字的笔画上，「agentreel」从左到右在四帧里点亮；
// 3.2 副标题「用代码与智能体拍片」晚一拍，用更少、更短的光丝写出来。字是 WebGL 里的光
// （text/lighttext.ts）：按弥散圆取预虚化的字图，参与辉光与景深。
// 镜头推近时，字在离镜头 3 m 到 1.2 m 之间化成光：每个字的笔画放出光丝，往前、往两侧飞出去，
// 落到走廊的弦上（corridor.ts），弦从近到远被它们点亮。

import * as THREE from 'three';
import type { StageContext } from '@beat3d/core/types';
import { LightText, type CharPlacement } from '@beat3d/text/lighttext';
import { ThreadFlow, type ThreadOrigin } from '@beat3d/text/threads';
import { FONT } from '@beat3d/text/fonts';
import type { HazeLight, HazeRing } from '@beat3d/world/haze';
import { Rng } from '@beat3d/lib/rng';
import { clamp, easeOutCubic, smoothstep } from '@beat3d/lib/easing';
import type { LensState } from '@beat3d/lib/lens';
import { scaled, toneOf, type RGB } from '@beat3d/lib/color';
import { CUE, runLift } from './cues';
import { DIM_HUE, PAL } from './palette';
import { STRINGS } from './corridor';
import { WORD_CENTRE } from './lights';

export const TITLE = 'agentreel';
export const SUBTITLE = '用代码与智能体拍片';

/** 字在这段距离里化成光（m）：离镜头 3 m 时开始，1.2 m 时没了。 */
const FADE_NEAR: [number, number] = [1.2, 3.0];

interface Line {
  text: LightText;
  write: ThreadFlow;
  release: ThreadFlow;
  /** 这一行的光在雾里：点亮后有自己的一团光 */
  centre: THREE.Vector3;
  lit: number;
}

export class Title {
  private readonly lines: Line[] = [];
  readonly objects: THREE.Object3D[] = [];

  constructor(ctx: StageContext, key: string, camZ: (f: number) => number) {
    const rng = new Rng(`${key}/title`);
    const specs = [
      { text: TITLE, em: 0.138, y: WORD_CENTRE.y + 0.012, lit: CUE.word, span: 4, threads: 7, out: 8, color: scaled(PAL.warmWhite, 2.3), faces: { latin: [FONT.mono, 400] as [string, number] } },
      { text: SUBTITLE, em: 0.047, y: WORD_CENTRE.y - 0.165, lit: CUE.subtitle, span: 6, threads: 4, out: 5, color: scaled(PAL.amber, 1.55), faces: { cjk: [FONT.sansSC, 400] as [string, number] } },
    ];
    // 字给光的那一刻：离镜头 FADE_NEAR 中段时（按机位路径求）。
    const releaseAt = (() => {
      for (let f = CUE.accel; f < CUE.pass + 8; f += 0.25) if (camZ(f) < 2.4) return f;
      return CUE.pass - 4;
    })();
    for (const sp of specs) {
      const lt = new LightText(ctx, sp.text, sp.faces);
      const chars = Array.from(sp.text);
      const widthEm = chars.reduce((a, ch) => a + lt.advance(ch), 0);
      const placed: CharPlacement[] = [];
      const ins: ThreadOrigin[] = [];
      const outs: ThreadOrigin[] = [];
      let pen = -widthEm / 2;
      const ink = chars.filter((ch) => ch !== ' ').length;
      let n = 0;
      chars.forEach((ch) => {
        const adv = lt.advance(ch);
        const x = (pen + adv / 2) * sp.em;
        pen += adv;
        if (ch === ' ') return;
        const land = sp.lit - 1.5 + (sp.span * n) / Math.max(1, ink - 1);
        n++;
        // 字的 em 框中心：caps 的中线落在这一行的 y 上。
        const position = new THREE.Vector3(x, sp.y - 0.065 * sp.em, 0);
        placed.push({ ch, position, right: new THREE.Vector3(1, 0, 0), up: new THREE.Vector3(0, 1, 0), em: sp.em, ignite: land, color: sp.color });
        const side = Math.sign(x) || 1;
        // 写：从镜头后方、这个字所在的一侧汇进来（像漏斗一样收拢）。
        for (const [ex, ey] of lt.inkPoints(ch, sp.threads, rng)) {
          const rel = land + rng.range(-0.6, 0.6);
          const dur = rng.range(11, 15);
          const gx = x + ex * sp.em;
          const gy = position.y + ey * sp.em;
          const r = rng.range(0.6, 1.4);
          const a = Math.atan2(gy - WORD_CENTRE.y + rng.range(-0.2, 0.2), gx + side * rng.range(0.1, 0.4));
          ins.push({ p: new THREE.Vector3(gx, gy, 0), end: new THREE.Vector3(r * Math.cos(a), WORD_CENTRE.y + 0.55 * r * Math.sin(a), camZ(rel - dur) + rng.range(0.3, 1.2)), rel, dur, amp: rng.range(0.6, 1.1) });
        }
        // 化光：从笔画出发，往前、往这一侧飞，落到同侧的一根弦上。
        const mine = STRINGS.map((s, i) => ({ s, i })).filter((q) => q.s.side === side);
        for (const [ex, ey] of lt.inkPoints(ch, sp.out, rng)) {
          const gx = x + ex * sp.em;
          const gy = position.y + ey * sp.em;
          const rel = releaseAt + rng.range(-1.5, 1.5) - 2 * Math.min(1, Math.abs(x) / 0.5);
          const dur = rng.range(16, 26);
          const target = mine[rng.int(0, mine.length)].s;
          // 落点在它落下那一刻的镜头前方 3–16 m：光丝始终在镜头前面飞，不会从镜头旁边擦过去。
          const land = Math.min(-2, camZ(rel + dur) - rng.range(3, 16));
          outs.push({ p: new THREE.Vector3(gx, gy, 0), end: new THREE.Vector3(target.x, target.y, land), rel, dur, amp: rng.range(0.6, 1.1) });
        }
      });
      lt.place(placed);
      const write = new ThreadFlow(ctx, ins, { mode: 'assemble', segments: 20, trail: 0.3, bend: 0.08, wander: 0.012, wanderScale: 2.0, width: 0.45 }, `${key}/title/${sp.text}/in`);
      const release = new ThreadFlow(ctx, outs, { mode: 'dissolve', segments: 24, trail: 0.3, bend: 0.16, wander: 0.012, wanderScale: 3.0, width: 0.5 }, `${key}/title/${sp.text}/out`);
      this.lines.push({ text: lt, write, release, centre: new THREE.Vector3(0, sp.y, -0.3), lit: sp.lit });
      this.objects.push(lt.mesh, write.mesh, release.mesh);
    }
  }

  /** 摆好帧 f 的字与光丝；返回给雾的光源和 3.1 那一圈光环。 */
  update(f: number, lens: LensState, extinction: number): { lights: HazeLight[]; ring: HazeRing | null } {
    const lights: HazeLight[] = [];
    const show = f >= CUE.word - 20 && f < CUE.pass + 34;
    // 4.4 的上行（乐谱的 run 事件）：每一下字和它在雾里的光都亮一点。
    const lift = runLift(f);
    for (const l of this.lines) {
      l.text.mesh.visible = show;
      l.write.mesh.visible = show;
      l.release.mesh.visible = show;
      if (!show) continue;
      l.text.update(f, lens, 1 + 0.35 * lift, scaled(PAL.warmWhite, 5), { extinction, scatter: 0.35, fadeNear: FADE_NEAR, hotGain: 1.2 });
      l.write.update(f, { lens, gain: 0.6, colFar: scaled(PAL.deep, 1.2), colNear: scaled(PAL.amber, 2.4), extinction, near: [1.0, 3.0] });
      l.release.update(f, { lens, gain: 1.4, colFar: scaled(PAL.amber, 2.6), colNear: scaled(PAL.deep, 1.4), extinction, near: [1.6, 4.0] });
      // 点亮的这一行在雾里有自己的一团光（照亮周围的雾和微尘）；化成光后转给弦。
      const lit = smoothstep(l.lit - 2, l.lit + 3, f) * (1 - smoothstep(CUE.pass - 8, CUE.pass + 2, f));
      if (lit > 0) lights.push({ position: l.centre, color: toneOf(PAL.amber, 0.55 * lit * (1 + 1.2 * lift), DIM_HUE), radius: 0.8 });
    }
    // H1：一圈光从字心穿过雾往外跑（约一拍），圆角取半边长 = 圆环。
    let ring: HazeRing | null = null;
    const d = f - CUE.word;
    if (d >= 0 && d < 18) {
      const u = d / 16;
      const h = 0.08 + 1.4 * easeOutCubic(clamp(u));
      ring = {
        centre: WORD_CENTRE.clone().setZ(-0.05),
        ax: new THREE.Vector3(1, 0, 0),
        ay: new THREE.Vector3(0, 1, 0),
        half: [h, h],
        radii: [h, h, h, h],
        width: 0.004 + 0.03 * u,
        color: scaled(PAL.amber, 1.1 * Math.pow(clamp(1 - u), 1.6) * smoothstep(0, 1.5, d)) as RGB,
      };
    }
    return { lights, ring };
  }

  dispose(): void {
    for (const l of this.lines) {
      l.text.dispose();
      l.write.dispose();
      l.release.dispose();
    }
  }
}
