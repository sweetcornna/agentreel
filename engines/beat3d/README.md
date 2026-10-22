# beat3d：三维光影的画面引擎

Bun + TypeScript + three.js 0.186.1 的自写渲染器，不用任何视频框架。无头 Chrome（WebGL2）出帧，ffmpeg 编码。
时间唯一的来源是片子的 `timeline.json`（小节.拍 ↔ 帧），配乐的 `events.json` 在构建时烘进页面包，画面按乐谱事件动。
一帧是帧号的纯函数：可以乱序、并行、按小数帧（运动模糊）渲，结果逐字节一样。

引擎移植自《一拍》与《成路》两部片的画面引擎，合成一份，去掉了片子专属的镜头、品牌与文案：光圈形状、调色、
字、名牌都由片子自己给。示例片是 [`films/demo-light/`](../../films/demo-light/README.md)。两部原片的做法与教训见
[`docs/cases/yipai.md`](../../docs/cases/yipai.md)、[`docs/cases/chenglu.md`](../../docs/cases/chenglu.md)，整套制作流程见
[`docs/METHOD.md`](../../docs/METHOD.md)。

## 1. 一帧怎么出来

```
films/<名>/timeline.json ─ lib/time（小节.拍 ↔ 帧；固定速度，或按乐谱拍点的 "score" 速度）
films/<名>/out/music/events.json、facts.json ─ 构建时烘进页面包（scripts/build.ts，先校验拍点与网格差 ≤ 1 ms）；没有就走网格
films/<名>/scenes/index.ts（默认导出 FilmDef）─ 构建时作为 `@film` 模块打进同一个包
        ▼
src/film.ts（页面入口，window.film）→ core/engine.ts  FilmEngine.renderFrame(n)
   1. shotAt(n) → core/registry.ts 找认领这个镜头的程序（没人认领：占位牌；时间线的镜头之外：纯黑）
      画幅按时间线的 aspect 曲线（lib/layout.ts），例如 2.39 遮幅在某一拍直接跳到 16:9
   2. program.frame({ frame: n ± 快门内的子帧 }) → { world?, screen?, overlay?, post? }
      HDR 目标（RGBA16F）里先画 world、再画 screen 层；运动模糊 = 同一帧在快门内取 N 个子帧平均
   3. overlay：Canvas2D 画的显示空间层（字幕卡、片尾名单、占位标签），在颗粒之前合成
   4. post/FilmPost：深度景深（可选）→ 辉光金字塔 → 晕光 → 变形拉丝 → 合成
      合成 = 色差（暖色单向或 RGB）、曝光、AgX + look（ASC-CDL）、lift/gamma/gain、暗角、
             负片转正片（可选）、overlay、颗粒、三角抖动、遮幅
scripts/render.ts：一个无头 Chrome 开几个页面并行出帧 → readPixels → POST 给 Bun 本地服务
                   → 各自 ffmpeg（libx264 yuv420p BT.709）→ concat 无损拼接 → 混入配乐
```

- **一帧只依赖帧号。** 程序在构造时把能预算的都算好，`frame()` 里不留跨帧状态。
- **单一 WebGL 输出。** 字也在 WebGL 里：参与光效的字（光字、光丝）在 HDR 里；字幕卡在 overlay 层。字体用 FontFace 加载、Canvas2D 栅格化成纹理，没有 DOM 字。
- **尺寸无关。** 长度都按 1080p 像素写，乘 `scale = 高 / 1080`；辉光、晕光、拉丝的金字塔在 4K 下多跳一级。
- **同一支镜头。** 点光斑（`world/bokeh.ts`）、层的虚化（`post/LensBlur.ts`）、光字的预虚化（`text/lighttext.ts`）都走 `lib/aperture.ts` 的同一个光圈、`lib/lens.ts` 的同一个薄透镜弥散圆，失焦的灯、面和字读起来是一支镜头。

## 2. 目录

| 路径 | 内容 |
|---|---|
| `src/film.ts` | 页面入口，`window.film`（`init` / `renderFrame` / `postFrame` / `renderRange`） |
| `src/film-module.d.ts` | `@film` 模块的类型（片子的 `scenes/index.ts`） |
| `src/core/` | `engine.ts` 引擎；`types.ts` 程序契约（FilmDef / ProgramDef / ShotProgram）；`registry.ts` 镜头 → 程序 |
| `src/lib/` | `time` 拍↔帧与乐谱事件、`curves` 曲线（aspect / energy / persistence）、`layout` 遮幅与版式线、`lens` 薄透镜与 CoC、`aperture` 光圈（圆 / 叶形 / 多边形）、`color` 设计色与显示变换反解（`sceneFor`）、`path` 按拍位关键帧的 Hermite 路径、`rng` `noise` `easing`、`facts` 事实表、`contours` `persistence` |
| `src/gl/` | GLSL 片段：`aperture.glsl`（光圈 SDF、亮度分布、CoC）、`noise.glsl`（simplex / fbm / curl / 哈希）、`color.glsl`（AgX、sRGB、调色），都带 include guard |
| `src/post/` | `FilmPost` 后期链、`params` 默认参数与覆盖合并、`shaders`、`LensBlur` 层的光圈虚化（Halton 核、按最多 3 个平面算 CoC）、`gradePath` 场景线性的色度路径（暗处落在墨色上、弱光走中性灰到片子的光色） |
| `src/text/` | `fonts` 字体；`lighttext` 3D 光字；`threads` 3D ThreadFlow（光聚成字 / 字化为光）；`screenThreads` 屏幕空间光丝；`glyphs` GlyphField（打字、字化为光）；`cards` 字幕卡与片尾名单；`featureCard` `featureCard2d` 功能卡 |
| `src/world/` | `stage` 分层舞台（镜头、雾、虚化、back / layer / front 三层合成）、`haze` 体积雾、`dust` 微尘、`bokeh` 点光斑、`lines` 光线、`rails` 弦上的脉冲、`beams` `truss` 圆角方梁与桁架、`glow` `trail` `lights` 光与光痕，以及夜景组件 `sky` `stars` `terrain` `grass` `road` `village` `walker` `brush` `frost` `screen` |
| `src/scenes/placeholder.ts` | 没有程序的镜头画的占位牌 |
| `scripts/` | `build` `serve` `still` `render` `check-determinism` `diff-order` `qa-cuts` `qa_tone.py`；`lib/` 下是 `film`（选片）、`tools`（找 Chrome / ffmpeg、GPU 参数）、`chrome`（CDP 驱动）、`server`、`session`（页面会话、ffmpeg 封装） |
| `public/` | `index.html` 渲染页、`viewer.html` 预览页、`fonts/`（OFL 字体与许可） |

片子目录（`films/<名>/`）：

| 路径 | 内容 |
|---|---|
| `timeline.json` | 时间线（字段见 [`docs/TIMELINE.md`](../../docs/TIMELINE.md)）。beat3d 必须有 `aspect`（全程一个比例也要写一个键）。`framesPerBeat` 直接取用，不复核它是否等于 `fps × 60 / bpm`（paper2d、filmsynth、sampler 会复核），写时间线时自己保证；`docs/TIMELINE.md` 第 11 节的自查片段会查 |
| `scenes/index.ts` | 默认导出 `FilmDef`：镜头程序、整体质感、光圈。没有这个文件，每个镜头都是占位牌 |
| `out/music/` | 配乐交付：`score.wav`、`events.json`（没有就无声、走网格） |
| `facts.json` | 可选：画面上出现的真实文字与数字的出处 |
| `tone.json` | 可选：`qa_tone.py` 的色调目标 |
| `out/` | 静帧、成片、日志、构建产物（不入库） |

## 3. 命令（都在 `engines/beat3d/` 下）

```sh
bun install                                   # 只有 three、typescript 和类型包
bunx tsc --noEmit                             # 类型检查（含 films/*/scenes）
bun scripts/serve.ts [--film <dir>] [--port 5173]
    # 预览页 viewer.html：←/→ 逐帧，shift 按拍，空格播放，?f=<帧号>；每次刷新都重新构建
bun scripts/still.ts [--film <dir>] --at 3.1+2f,6.2 --size 1080|4k|540 [--out <dir>] [--sheet <png> --cols 3]
    # PNG 静帧（默认写到 <film>/out/stills），冷 / 热耗时写进 timings.json；--sheet 顺手拼一张对照表
bun scripts/still.ts --at 6.1 --post '{"motionBlur":{"samples":1},"grain":{"amount":0}}'   # 临时后期覆盖（render 也支持）
bun scripts/render.ts [--film <dir>] --size 1080 [--from 5.1 --to 7.1] [--workers 3] [--no-audio] [--out <mp4> | --out-name <名>]
    # 成片（默认 <film>/out/<名>-1080p.mp4）；有 out/music/score.wav 且没给 --no-audio 就混入；耗时追加到 out/render-log.json
bun run skeleton                              # 整片 540p 时间骨架（全部画占位牌）
bun scripts/check-determinism.ts [--at …] [--size 1080]   # 两个页面、不同顺序渲同一批帧，逐字节比较（默认每个镜头两帧）
bun scripts/diff-order.ts --frame 300         # 两个页面先渲不同的帧再渲同一帧，给出不同之处的方框（查顺序依赖）
bun scripts/qa-cuts.ts --in <mp4>             # 找画面跳变，和时间线对账：镜头边界（含最后一个镜头切黑）、镜头内的蒙太奇切点
                                              # shots[].cuts、硬同步点；不在切点上的跳变退出码 1，没有跳变的切点列为软切
uv run --with numpy python scripts/qa_tone.py <mp4 | png…>   # 色调档位：暗部是不是墨色、弱光与亮光的色相对不对
```

- `--film` 默认是 `../../films/demo-light`；路径相对当前目录。`--timeline`、`--score`、`--events`、`--facts` 可以单独指定。
- 位置写法：`"5.1"`、`"4.3+0.5"`（晚半拍）、`"5.1+3f"`（晚 3 帧）。`--frames 400,401` 直接给帧号。
- 渲染默认 1080p CRF 16 preset slow，4K CRF 14；分段临时文件在 `out/render-tmp/`，拼完即删。
- 每次启动 Chrome 用 `mkdtemp` 的临时 `--user-data-dir`（`b3d-chrome-*`），结束即删，不碰本机自己的 Chrome 配置。
- 这些检查在验收里各管什么、阈值多少，以及和 paper2d、配乐库的检查怎么对应，见 [`docs/QA.md`](../../docs/QA.md) 的「工具对照」。

## 4. 程序契约（`src/core/types.ts`）

片子的 `scenes/index.ts` 默认导出一个 `FilmDef`：

| 字段 | 说明 |
|---|---|
| `programs: ProgramDef[]` | 镜头程序。一个程序可以认领几个连续镜头（不切的长镜头）；同一镜头被认领两次直接报错 |
| `lab?: ProgramDef[]` | 风格试验（`--lab` 时顶替它们认领的镜头，成片里不出现） |
| `post?` / `shotPost?` | 片子的整体质感与逐镜头覆盖 |
| `aperture?` | 片子的光圈（`{ shape: 'circle' }`、`{ shape: 'leaf', radii?, open? }`、`{ shape: 'polygon', sides, roundness? }`），是所有光斑、层虚化、光字的默认值 |
| `overlay?` | 跨镜头的 2D 层（例如跨镜头的字幕） |
| `credits?` | 时间线 `cards` 里片尾名单的数据 |

`ProgramDef = { id, shots, post?, handlesCards?, create(ctx) }`；`create` 可以是 async（栅格化字、预计算），必须确定。
它返回的 `ShotProgram` 有 `frame(req)` 和 `dispose()`。`frame` 收到 `{ frame, shot, layout, subSample }`，返回：

| 字段 | 说明 |
|---|---|
| `world` | `{ scene, camera }`：三维世界，先画进 HDR 目标（带深度，给可选的 `dof` 用） |
| `screen` | 屏幕空间的 HDR 层（画布像素，y 向下，用 `ctx.screenCamera`），画在 world 之上。全屏合成好的画面也放这里（一个全屏三角形） |
| `overlay` | 显示空间的 Canvas2D 回调（字幕、名牌），在显示变换之后、颗粒之前合成 |
| `post` | 这一帧的后期覆盖 |
| `clear` | HDR 清屏色（线性，默认黑） |

后期覆盖的顺序：`DEFAULT_POST` ← `FilmDef.post` ← `FilmDef.shotPost[镜头]` ← `ProgramDef.post` ← `FrameResult.post` ← 命令行 `--post`。
数组整体替换，对象深合并；长度一律是 1080p 像素。参数表见 `src/post/params.ts` 的注释（曝光、辉光、晕光、拉丝、颗粒、暗角、
色差与 `chromaticStyle`、`grade`、`dof`、`motionBlur`、`letterbox`、`negative`）。运动模糊的子帧（`subSample: true`）只用场景，
overlay 和 post 取整数帧那一次的。

## 5. 新增一个镜头

1. 在片子的 `scenes/` 下写一个程序。镜头里的模块从 `@beat3d/*` 引（构建插件把它解析到 `engines/beat3d/src/*`，片子目录不需要 `node_modules`）：

```ts
import * as THREE from 'three';
import type { FrameRequest, FrameResult, ProgramDef, ShotProgram, StageContext } from '@beat3d/core/types';
import { BokehPoints } from '@beat3d/world/bokeh';
import { makeLensCamera } from '@beat3d/lib/lens';
import { CLOCK } from '@beat3d/lib/time';
import { rngFor } from '@beat3d/lib/rng';

const ID = 'lamps';
const START = CLOCK.frameAt('3.1'); // 时刻只从拍位换算，不写死帧号

class Lamps implements ShotProgram {
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly lamps: BokehPoints;
  private readonly base: THREE.Vector3[];

  constructor(ctx: StageContext) {
    const rng = rngFor(ID, 'lamps'); // 随机数按程序 id 与用途播种
    this.camera = makeLensCamera(50, ctx.width, ctx.height);
    this.lamps = new BokehPoints(ctx, 12);
    this.base = Array.from({ length: 12 }, () => new THREE.Vector3(rng.range(-2, 2), rng.range(-1, 1), -rng.range(4, 12)));
    this.scene.add(this.lamps.mesh);
  }

  frame({ frame }: FrameRequest): FrameResult {
    const t = (frame - START) / CLOCK.fps; // 一切从帧号来；frame 可以是小数（运动模糊的子帧）
    this.base.forEach((p, i) => this.lamps.set(i, p.x, p.y + 0.02 * Math.sin(t + i), p.z, 4000, 1500, 400));
    this.lamps.update(this.base.length, { lens: { focalMm: 50, fStop: 1.4, focus: 2 } });
    return { world: { scene: this.scene, camera: this.camera }, post: { exposure: 0.2 } };
  }

  dispose(): void {
    this.lamps.dispose();
  }
}

export const lamps: ProgramDef = { id: ID, shots: ['S02'], create: (ctx) => new Lamps(ctx) };
```

2. 在 `scenes/index.ts` 的 `programs` 里加上它。
3. 看：`bun scripts/serve.ts --film ../../films/<名>`，预览页里跳到 `3.1`；或者 `bun scripts/still.ts --film … --at 3.1,3.3 --size 1080`。
4. 每改一轮：`bunx tsc --noEmit`、`check-determinism`，出片后 `qa-cuts`。

一部片里几个镜头不切地连着（一镜到底），就写成一个程序认领它们，镜头之间的东西在程序里按帧过渡（示例片的 `take.ts` 认领 S01–S04）。
要多层合成（光斑在雾里、玻璃或梁墙作为一层被光圈虚化）就用 `world/stage.ts` 的 `Stage`，自己把三层合成进一张 HDR 目标，再用一个全屏三角形作为 `screen` 交出去。

## 6. 主要组件

| 组件 | 用法要点 |
|---|---|
| `CLOCK`（`lib/time`） | `f(pos)` 整数帧、`frameAt(pos)` 小数帧、`beatPhase(f)`、`label(f)`、`eventFrames(type, from, to)` 乐谱事件（`{ frame, data }`），`source` 为 `grid` / `events` / `beats`。`SHOTS`、`shotAt`、`shotById`、`FILM_FRAMES` |
| `ASPECT` `ENERGY` `PERSISTENCE`（`lib/curves`） | 时间线上的画幅、能量、保留度曲线。`aspect.interp` 写 `constant` 时整段保持、两个相邻键（同一拍）就是一刀跳变 |
| `lib/aperture` | `ApertureSpec`、`compileAperture`（面积与亮度分布在固定网格上数值归一，确定）、`apertureSd` `apertureProfile` 的 CPU 孪生；`open` 让任何形状张开成圆 |
| `lib/lens` | `makeLensCamera(mm, W, H)`（Super 35，画幅宽 24.89 mm 对整块 16:9 画布）、`cocK`、`cocPixels`、`pxPerMetre`、`aim` |
| `lib/color` | `lin(hex)`、`scaled`、`toneOf`；`sceneFor(hex, grade)`：显示出来正好是这个设计色的场景线性值（grade 要和片子 post 的 `grade` 一致） |
| `lib/path` | `Path3` / `Path1`：按拍位给位置和速度的 Hermite 关键帧，机位不走走停停 |
| `BokehPoints` | 每个点按 CoC 画成光圈形状，能量守恒、猫眼、暖色边、雾中消光；`profile: 'hero'` 给少数几个大光斑更细的亮边与环 |
| `Haze` / `ambientFor` | 点光等角采样单散射 + HG 相函数 + fBm 密度 + 墨色环境光 + 均匀消光；最多 8 盏光、可选一圈圆角光环；按最多 3 个平面分前后两段。`ambientFor(墨色, 消光, 远端)` 让没被照亮的雾正好显示成墨色 |
| `Dust` | 被光照到的微尘，位置是时间的解析式；按平面分给前 / 后两组光斑 |
| `LightText` | 3D 光字：每字一个面片，字图集按光圈预虚化几级、按 CoC 取；`ignite` 逐字点亮，`fadeNear` 让字在镜头前给定距离里化成光，`scatter` 让镜头穿过时字向两边让开；`inkPoints` 取笔画上的点给光丝 |
| `ThreadFlow`（3D） / `ScreenThreadFlow`（2D） | 光聚成字（assemble）/ 字化为光（dissolve）：二次贝塞尔 + curl 漂移，全在顶点着色器里按帧号求值；屏幕对齐的高斯带，按 CoC 加宽且能量守恒；`near` 让贴着镜头的段淡掉 |
| `LightLines` / `Rails` | 每帧在 CPU 上摆顶点的光线（按 CoC 加宽）；沿直线逐像素求值的脉冲（头 + 尾） |
| `Stage` | 镜头相机、雾、光圈虚化、三层合成 `out = back·(1 − layer.a) + layer + front` |
| `GradePath`（`post/gradePath`） | `warmPath(墨色, 光色)` + `GRADE_FRAG`：在场景线性里把颜色按亮度走一条路，暗处是墨色，弱光不被 AgX 的趾部显示成品红或红 |
| `lib/facts` | `factText('a.b')`：画面上的真实文字都从 facts.json 取，缺了就显示醒目的 `FACT:a.b` |

## 7. 确定性规则

1. 画面只是帧号的纯函数。程序构造时预算，`frame()` 里不改变任何会影响下一帧的状态（摆 uniform、摆位置不算状态）。
2. 不用 `Math.random`、`Date`、`performance.now()` 影响画面。随机数一律 `rngFor(程序 id, 用途)` / `new Rng('字符串')` 或无状态哈希；GLSL 里用整数 PCG 哈希。雾的分层抖动用整数帧号做哈希。
3. 不做累积模拟。要模拟的就写成解析式（ThreadFlow、微尘），或者预先算好存盘、按帧读取。
4. 时间只写「小节.拍」，经 `CLOCK` 换算；乐谱事件构建时烘进包，渲染期间不读文件。Bun 侧脚本（qa-cuts 等）只用网格，固定速度的片子两者差 ≤ 1 ms（构建时校验）。
5. 运动模糊的子帧也是 `frame(n + δ)`，程序必须接受小数帧。硬切的量（换焦段、换镜头里的世界）按 `Math.round(frame)` 决定，同一帧的子帧不会一半在切前一半在切后。
6. 编译过的光圈按值缓存：每帧换光圈形状可以，但把连续变化的 `open` 量化（示例片取 1/32），否则子帧会不停重算。
7. 自检：`check-determinism`（两个页面、不同顺序、逐字节）每次改完都跑。只保证同一台机器、同一个 Chrome / ANGLE 版本逐位一致；换机器或换 GPU 后端，像素会有微小差异。
8. 成片回归要用同样的 `--workers`：render 按 worker 切段分别编码，切点不同，x264 的段内参考和码率控制就不同，解码后的像素也跟着变。

## 8. 移植与平台

外部工具都不写死路径（`scripts/lib/tools.ts`）：先看环境变量，再看平台的常见安装位置，最后在 PATH 里找。

| 变量 | 说明 |
|---|---|
| `CHROME_PATH` | Chrome / Chromium 可执行文件（120 以上，要有 WebGL2 与 `EXT_color_buffer_float`）。macOS 默认找 `/Applications/Google Chrome.app`，Windows 找 Program Files 与 LocalAppData，Linux 找 `google-chrome` / `chromium` |
| `FFMPEG` / `FFPROBE` | ffmpeg（要带 libx264）/ ffprobe，可以是绝对路径或 PATH 里的名字 |
| `BEAT3D_GPU` | 无头 Chrome 的 GPU 后端预设：`metal`（macOS 默认）、`d3d11`（Windows 默认）、`vulkan`（Linux 默认）、`gl`、`swiftshader`（CPU，哪里都能跑、慢很多倍） |
| `BEAT3D_GPU_FLAGS` | 直接给后端参数（空格分隔），优先于 `BEAT3D_GPU` |

- 实测过的组合：macOS（Apple 芯片）+ Chrome + ANGLE Metal，`--headless=new`。Windows、Linux 的预设按各平台 ANGLE 的常规后端给出，**没有逐一实测**；拿不到 WebGL2 时先试 `BEAT3D_GPU=gl` 或 `swiftshader`。
- `--use-angle=gl` 在 macOS 上拿不到 WebGL2；ANGLE 下 `texture2D(s, uv, bias)` 返回全黑，要 mip 就用 `textureLod`。
- 不同后端、不同 GPU 的浮点细节不同，逐字节确定性只在同一环境内成立（见第 7 节）。
- `qa_tone.py` 要 Python 3.10+ 与 numpy，以及 ffmpeg（`FFMPEG` 或 PATH）。没有装 numpy：`uv run --with numpy python scripts/qa_tone.py …`。
- 字体：`public/fonts/` 带 Inter、JetBrains Mono、Noto Sans SC、EB Garamond（都是 OFL，许可见 `OFL*.txt`）。Noto Serif SC 每个字重约 8 MB，没有随引擎分发：把 `NotoSerifSC-300/400/500.woff` 放进 `public/fonts/` 就会被构建认出来；没有时衬线中文按 CSS 字族列表退到 Noto Sans SC。
- Windows 上路径分隔符、可执行文件后缀由 `tools.ts` 与 `lib/film.ts` 处理；启动失败残留的 Chrome 进程按临时目录名 `b3d-chrome-` 找。

## 9. 已知的坑

- **多重采样目标只在有绘制时才解析**：只清屏不画，纹理里还是上一次的内容。没东西的层用 `Stage.none`。
- **AgX 很吃暗部、压淡高光**：夜景按显示值调，不按线性值直觉调；设计色用 `sceneFor` 反解。默认 look 会把琥珀压成米色，片子可以在 `grade.lookSaturation` 里加一点饱和（`sceneFor` 要传同样的 grade）。
- **调色的 lift 经 sRGB 后约 4 个码值**，加上颗粒「全黑」就不黑：lift 只在场景里有光时才加（`anyLight`），没光的帧是真 0。
- **`ambientFor` 带实测校准**（雾的 fBm 被下限截断后均值约为名义值一半，再加显示的趾部）。墨色偏亮或偏暗时调片子的环境光比例，不改校准。
- **雾的 HG 相角**是 `dot(normalize(p − L), −dir)`，符号反了光晕会朝错的方向。
- **贴着镜头的光丝与字**：弥散圆极大、在屏幕上跑得极快，子帧不够就是一团烟或一串台阶。光丝用 `near` 淡掉，字用 `fadeNear` 化成光。
- **子帧间移动比线宽还大的亮点**会叠成一串点：在一个子帧的快门区间里沿运动摆几颗、各分一份能量（示例片 `corridor.ts` 的火花），或者加子帧。
- **线段的一端在镜头后面**，投影会翻折成一个 V：画之前把它裁到镜头前（`Rails` 的尾巴长度、`ThreadFlow` 两端都要在镜头前）。
- **Hermite 关键帧**紧跟在交接点后面、而下一段切线很大时，曲线会先往反方向走（切线 × 区间 > 3 × 位移差）。
- **GLSL 字符串里的注释保持 ASCII**：WebGL 各实现对着色器源码里的非 ASCII 字符处理不一。
- **本地变量遮住同名 sampler**（例如 `float tText`），GLSL 报 `texture: no matching overloaded function`。
- **CDP 的 `Target.createTarget` 在 headless 下不接受 width/height**：视口用 `Emulation.setDeviceMetricsOverride`，画布像素与视口无关。

## 10. 渲染耗时（示例片，Apple 芯片笔记本，Chrome ANGLE Metal，1080p）

| 项 | 数值 |
|---|---|
| 单帧（`still.ts`，一个页面，热） | S01 60–110 ms，2.4 起光点聚拢用 8 个子帧时约 0.33 s；S02 60–360 ms（4.4 到穿过字面 12 个子帧）；S03 120–310 ms；S04 35–60 ms。SwiftShader（CPU）同样的帧约慢 20 倍 |
| 整片 576 帧，3 个页面，CRF 16 slow，含混音 | 71–72 s 墙钟（0.12 s/帧）；540p 约 20 s |

每次的数字记在 `<film>/out/render-log.json`。
