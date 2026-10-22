// 一镜到底：S01 → S04 是同一个程序（同一台摄影机、同一片雾），中间没有切。
//
// 每一帧：摆机位（rig.ts）→ 各段摆好自己的光（lights / title / corridor / end）并把照亮雾的光源交上来
// → 雾（world/haze.ts）整条视线积分进 HDR 目标 → 所有发光体（光斑、字、光丝、弦、脉冲、微尘）
// 加在上面 → 这条片自己的分级路径（post/gradePath.ts：暗处落在墨色上，弱光走中性灰到琥珀，
// 不在 AgX 的趾部发红）→ 交给引擎的胶片后期。所有东西都是帧号的纯函数。

import * as THREE from 'three';
import type { FrameRequest, FrameResult, ProgramDef, ShotProgram, StageContext } from '@beat3d/core/types';
import { hdrTarget } from '@beat3d/post/FilmPost';
import { FS_VERT } from '@beat3d/post/shaders';
import { GRADE_FRAG, gradeUniforms, setGrade, warmPath } from '@beat3d/post/gradePath';
import { Stage } from '@beat3d/world/stage';
import { ambientFor, MAX_LIGHTS, type HazeLight } from '@beat3d/world/haze';
import type { Plane } from '@beat3d/post/LensBlur';
import { lerp, smoothstep } from '@beat3d/lib/easing';
import { ENERGY } from '@beat3d/lib/curves';
import { CLOCK } from '@beat3d/lib/time';
import { scaled } from '@beat3d/lib/color';
import { blurSamples, CUE, flashes, SHUTTER } from './cues';
import { SHOWS, DIM_HUE } from './palette';
import { shotAt } from './rig';
import { DarkLights } from './lights';
import { Title } from './title';
import { Corridor, SPOT } from './corridor';
import { EndCard, endFade, endFocus } from './end';

const ID = 'take';

/** 雾的样子，按帧插值：密度（1/m）、消光（1/m）、视线的远端（m）、未被照亮的雾显成墨色的比例。 */
interface LookKey {
  at: number;
  density: number;
  extinction: number;
  far: number;
  air: number;
}
const LOOK: LookKey[] = [
  // S01：浓一些的雾，光斑周围有体积。
  { at: 0, density: 0.05, extinction: 0.03, far: 60, air: 0.75 },
  { at: CUE.word, density: 0.045, extinction: 0.03, far: 60, air: 0.75 },
  // S02：字在雾里自己有一团光，未照亮的雾少给一点，暗处才落在墨色上。
  { at: CUE.word + 16, density: 0.045, extinction: 0.03, far: 60, air: 0.55 },
  { at: CUE.pass - 8, density: 0.04, extinction: 0.025, far: 80, air: 0.6 },
  // 穿过字面后进走廊：雾变薄、看得远，弦一直伸进纵深里。
  { at: CUE.pass, density: 0.03, extinction: 0.014, far: 160, air: 0.8 },
  { at: CUE.converge, density: 0.03, extinction: 0.016, far: 120, air: 0.8 },
  // S04：又浓回来一点，让光斑周围有一团光。
  { at: CUE.spot, density: 0.04, extinction: 0.025, far: 60, air: 0.8 },
];
function lookAt(f: number): LookKey {
  let i = 0;
  while (i < LOOK.length - 1 && f >= LOOK[i + 1].at) i++;
  const a = LOOK[i];
  const b = LOOK[Math.min(i + 1, LOOK.length - 1)];
  const t = b === a ? 0 : smoothstep(a.at, b.at, f);
  const m = (k: keyof LookKey) => lerp(a[k], b[k], t);
  return { at: f, density: m('density'), extinction: m('extinction'), far: m('far'), air: m('air') };
}

/** 镜头的对焦：S03 后半从 5 m 按屈光度拉到汇聚点上，7.1 起交给片尾（先在光斑上，再拉到片名上）。 */
function focusFor(f: number, position: THREE.Vector3): number | undefined {
  if (f >= CUE.spot) return endFocus(f);
  if (f < CUE.converge) return undefined;
  const u = smoothstep(CUE.converge, CUE.spot, f);
  return 1 / lerp(1 / 5, 1 / position.distanceTo(SPOT), u);
}

class Take implements ShotProgram {
  private readonly stage: Stage;
  private readonly dark: DarkLights;
  private readonly title: Title;
  private readonly corridor: Corridor;
  private readonly end: EndCard;
  private readonly scene = new THREE.Scene();
  private readonly out: THREE.WebGLRenderTarget;
  private readonly mGrade: THREE.ShaderMaterial;
  private readonly screen = new THREE.Scene();
  // 墨色的范围放宽一点（到 1.5 倍墨色亮度都还是墨色）：被远处的光微微照到的雾仍读成墨，不发棕。
  private readonly path = warmPath(SHOWS.ink, DIM_HUE, { m: [1.4, 1.8, 2.8], grey: [1, 0.82, 0.5] });

  constructor(ctx: StageContext) {
    this.stage = new Stage(ctx, { focalMm: 85, near: 0.02, far: 400 });
    const camZ = (f: number) => shotAt(f).position.z;
    this.dark = new DarkLights(ctx, ID);
    this.title = new Title(ctx, ID, camZ);
    this.corridor = new Corridor(ctx, camZ);
    this.end = new EndCard(ctx, ID);
    this.scene.add(this.dark.points.mesh, this.dark.dustPoints.mesh, ...this.title.objects, this.corridor.lines.mesh, this.corridor.rails.mesh, this.corridor.sparks.mesh, ...this.end.objects);
    this.out = hdrTarget(ctx.width, ctx.height);
    const u = gradeUniforms();
    u.tSrc.value = this.out.texture;
    this.mGrade = new THREE.ShaderMaterial({ vertexShader: FS_VERT, fragmentShader: GRADE_FRAG, uniforms: u, depthTest: false, depthWrite: false, toneMapped: false });
    const quad = new THREE.Mesh(this.stage.fullscreen(), this.mGrade);
    quad.frustumCulled = false;
    this.screen.add(quad);
  }

  private render(f: number, exposure: number): void {
    const st = this.stage;
    const renderer = st.ctx.renderer;
    const pose = shotAt(f);
    const focus = focusFor(f, pose.position);
    const shot = focus === undefined ? pose : shotAt(f, focus);
    st.pose(shot);
    const cam = st.camera;
    const lens = shot.lens;
    const look = lookAt(f);
    const ext = look.extinction;
    const energy = ENERGY.at(f);

    // 各段：在自己的时间里才摆（其余时间藏起来，不花 GPU）。
    const lights: HazeLight[] = [];
    const inDark = f < CUE.word + 8;
    const inTitle = f >= CUE.word - 20 && f < CUE.pass + 34;
    const inCorridor = f >= CUE.pass - 8 && f < CUE.spot + 2;
    const inEnd = f >= CUE.zoom;
    let ring = null;
    let titleLights: HazeLight[] = [];
    if (inTitle) {
      const t = this.title.update(f, lens, ext);
      titleLights = t.lights;
      ring = t.ring;
    } else this.title.update(f, lens, ext);
    this.dark.points.mesh.visible = inDark;
    this.dark.dustPoints.mesh.visible = f < CUE.pass;
    if (f < CUE.pass) lights.push(...this.dark.update(f, lens, cam, ext, energy, titleLights));
    lights.push(...titleLights);
    for (const m of [this.corridor.lines.mesh, this.corridor.rails.mesh, this.corridor.sparks.mesh]) m.visible = inCorridor;
    if (inCorridor) lights.push(...this.corridor.update(f, lens, ext, energy));
    if (inEnd) lights.push(...this.end.update(f, lens, ext, energy));
    else this.end.update(f, lens, ext, energy);

    // 雾：1.3 之前纯黑（没有光，也没有墨色的底）；之后未被照亮的雾显成墨色，S04 跟着淡出。
    const amb = smoothstep(CUE.firstLight - 1, CUE.firstLight + 8, f) * look.air * endFade(f);
    // H1 那一圈光只在字面后面的雾里：这一段用字面把视线分成前后两截。
    const planes: Plane[] = ring && cam.position.z > 0.05 ? [{ point: new THREE.Vector3(0, 0, 0), normal: new THREE.Vector3(0, 0, 1) }] : [];
    const haze = {
      lights: lights.slice(0, MAX_LIGHTS),
      ring,
      planes,
      density: look.density,
      extinction: ext,
      ambient: scaled(ambientFor(SHOWS.ink, ext, look.far), amb),
      time: f / CLOCK.fps,
      far: look.far,
    };
    st.clear(this.out);
    st.haze.render(this.out, cam, { ...haze, side: 'front' }, f);
    if (planes.length) st.haze.render(this.out, cam, { ...haze, side: 'back' }, f);
    renderer.setRenderTarget(this.out);
    renderer.render(this.scene, cam);

    // 这条片的分级路径（按这一帧的曝光量，闪的时候墨色也跟着走）。
    setGrade(this.mGrade.uniforms, this.path, 1, exposure);
  }

  frame({ frame }: FrameRequest): FrameResult {
    const fl = flashes(frame);
    // H1 字落成：曝光一闪；H2 穿过字面：曝光、横向耀斑、色差一起冲一下。
    const exposure = 0.5 * fl.word + 0.55 * fl.pass;
    this.render(frame, exposure);
    const pass = fl.pass;
    const corridor = smoothstep(CUE.pass, CUE.pass + 4, frame) * (1 - smoothstep(CUE.spot - 8, CUE.spot, frame));
    return {
      screen: this.screen,
      post: {
        exposure,
        bloom: { mix: 0.05 + 0.03 * pass + 0.01 * corridor },
        streak: { amount: 0.1 + 0.5 * pass + 0.08 * corridor, threshold: 3.0 - 1.2 * pass },
        chromatic: 0.9 + 4.5 * pass,
        halation: { amount: 0.2 + 0.1 * pass },
        motionBlur: { samples: blurSamples(frame), shutter: SHUTTER },
      },
    };
  }

  dispose(): void {
    this.dark.dispose();
    this.title.dispose();
    this.corridor.dispose();
    this.end.dispose();
    this.stage.dispose();
    this.out.dispose();
    this.mGrade.dispose();
  }
}

export const take: ProgramDef = {
  id: ID,
  shots: ['S01', 'S02', 'S03', 'S04'],
  create: (ctx) => new Take(ctx),
};

