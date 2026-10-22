# 示例片《光与弦》（demo-light）

beat3d 画面引擎 + filmsynth 配乐的示例片：8 小节画面加 1 小节音乐尾巴，24 fps、90 BPM、每拍 16 帧，共 576 帧（24 s）。
时间线是 [`timeline.json`](timeline.json)（写法见 [`docs/TIMELINE.md`](../../docs/TIMELINE.md)），配乐的工程在 [`score/`](score/)（见那里的 DESIGN.md），画面在 [`scenes/`](scenes/)。
叶形光圈、3D 的光聚成字、画幅一刀跳变这些做法来自《一拍》，见 [`docs/cases/yipai.md`](../../docs/cases/yipai.md)。

一个镜头程序一镜到底拍完 S01–S04，中间没有切：

| 镜头 | 拍位 | 画面 | 演示的引擎能力 |
|---|---|---|---|
| S01 暗处的光 | 1.1–3.1 | 1.3 之前纯黑。雾里六粒暖光随乐谱的 pulse 一明一灭；镜头对焦在 60 m 外，光点全是虚化光斑：先是叶形光圈，2.1 起虹膜在十帧里张开成圆形。雾里的微尘被照亮。2.4 起光点沿弧线飞向画面中心 | `BokehPoints`（hero 光斑、按帧换光圈）、`lib/aperture`（叶形 → 圆形的 `open`）、`Haze` 体积雾、`Dust` 微尘、乐谱事件驱动 |
| S02 光聚成字 | 3.1–5.1 | 3.1（H1）光丝从镜头后方两侧汇进来，「agentreel」逐字点亮，一圈光从字心穿过雾往外跑；3.2 副标题「用代码与智能体拍片」晚一拍写出来。焦点十帧拉到字上。4.4 起加速推近，乐谱的上行（run）让字一下一下更亮 | `LightText`（光字按弥散圆预虚化、参与辉光与景深）、`ThreadFlow` assemble、雾里的 `HazeRing`、拉焦 |
| S03 字化为光 | 5.1–7.1 | 5.1（H2）镜头从两行字之间穿过去，2.39 遮幅一刀跳到 16:9；字在镜头前化成光丝往两侧、往前飞散，落到两侧的弦上，弦从近到远被点亮；乐谱的 pulse 沿弦往前跑，左右轮流。6.1 起每根弦弯成收向纵深一点的曲线 | `ThreadFlow` dissolve、`LightText` 的 `fadeNear` / `scatter`、`LightLines`、`Rails`、画幅跳变、运动模糊（最多 12 个子帧） |
| S04 片尾 | 7.1–9.1 | 汇聚点在 7.1（H3）是一粒很亮的点，焦点拉到镜头前的片名上，它在圆形光圈里张开成一枚光斑，随乐谱 S04 的 pulse 轻轻呼吸。片名与仓库地址淡入；8.4 起淡出，9.1 一刀切黑，配乐的混响尾巴到 10.1 | 拉焦、`BokehPoints` hero、`LightText` 淡入 |

全片共用：AgX 显示变换与胶片后期（辉光、暖色晕光、变形拉丝、只往外溢暖色的色差、暗角、颗粒），以及这部片自己的
调色路径（`post/gradePath.ts`）：暗处落在墨色 #0D0C10 上，弱光走中性灰到琥珀，不被 AgX 的趾部显示成品红。

## 怎么出片

在 `engines/beat3d/` 下（`--film` 默认就是这部片）：

```sh
bun install
bunx tsc --noEmit
bun scripts/still.ts --at 1.4,2.2,3.2,4.3,5.2,6.2,7.3,8.2 --size 1080 --sheet ../../films/demo-light/out/stills/sheet.png --cols 2
bun scripts/render.ts --size 1080                 # → films/demo-light/out/demo-light-1080p.mp4（有配乐就混入）
bun scripts/check-determinism.ts
bun scripts/qa-cuts.ts --in ../../films/demo-light/out/demo-light-1080p.mp4
uv run --with numpy python scripts/qa_tone.py ../../films/demo-light/out/demo-light-1080p.mp4   # 目标在 tone.json
```

每条检查量什么、阈值多少，见 [`docs/QA.md`](../../docs/QA.md)（「工具对照」一节有这部片上实跑的 `qa-cuts` 与 `qa_tone` 输出）。

配乐在 `out/music/`（`score.wav`、`events.json`），由 `music/filmsynth` 按 `score/` 生成，见 [`score/DESIGN.md`](score/DESIGN.md)。
还没有配乐时，画面照样能出：没有 `events.json` 就走拍网格（S01 八分音符、S03 第 5 小节十六分 + 第 6 小节八分、
S04 四分与二分），没有 `score.wav` 就是无声片。构建时会打印烘进了多少个乐谱事件（`N score events baked in`）。

## 画面怎么读乐谱

`scenes/cues.ts` 是这部片全部的时刻，都从时间线的拍位换算，没有写死的帧号。乐谱事件按 `shot` 字段分开用：

| 事件 | 画面 |
|---|---|
| `pulse`，`shot: "S01"`（带 `light` 0..5） | 点亮哪一粒光：`light` 是音高从低到高的序号，光点按屏幕上从左到右排（低音在左）；力度 `bright` 决定亮多少。每一下 3 帧亮起、13 帧回落到余辉 |
| `pulse`，`shot: "S03"`（带 `side` L / R） | 在哪一侧的弦上发一个脉冲，同一侧依次换弦；`bright` 决定脉冲头的亮度 |
| `pulse`，`shot: "S04"` | 光斑的呼吸（2 帧涨起，两拍里落回） |
| `run`（4.4 的上行） | 字和它在雾里的光一下一下更亮 |

H1（3.1 字落成）、H2（5.1 穿过字面）、H3（7.1 落成光斑）、CUT（9.1）按时间线的拍位走；乐谱在这些拍位上有对应的 `chime`、`impact`、`land`、`cut`。

## 文件

| 文件 | 内容 |
|---|---|
| `scenes/index.ts` | `FilmDef`：程序、整体质感（暖色的色差、晕光、拉丝，`lookSaturation`）、默认光圈 |
| `scenes/take.ts` | 一镜到底的程序：每帧摆机位 → 各段交上光源 → 体积雾 → 发光体 → 调色路径 → 交给引擎后期；雾的样子按帧插值；运动模糊子帧数 |
| `scenes/cues.ts` | 时刻、乐谱事件与网格回退、两次硬同步的闪、运动模糊的子帧数 |
| `scenes/rig.ts` | 摄影机：位置是按拍位给速度的 Hermite 关键帧（`Path3`），焦段 85 → 35 → 50 mm，光圈与对焦 |
| `scenes/palette.ts` | 颜色：琥珀、深琥珀、灯色、暖白；`SHOWS.ink` = 显示出来正好是 #0D0C10 的场景线性值 |
| `scenes/lights.ts` | S01 的六粒光与微尘，叶形 → 圆形的光圈 |
| `scenes/title.ts` | S02 的字与写字 / 化光的光丝、H1 的光环 |
| `scenes/corridor.ts` | S03 的弦、脉冲、汇聚 |
| `scenes/end.ts` | S04 的光斑、片名、拉焦与淡出 |
| `tone.json` | `qa_tone.py` 的目标：墨色 (13, 12, 16) ±5，弱光与亮光的色相在琥珀的范围里 |
