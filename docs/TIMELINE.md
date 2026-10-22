# 时间线与拍网格规范

每部片在 `films/<名字>/timeline.json` 里有一份时间线。它是全片唯一的时间来源：画面引擎按它把「小节.拍」换成帧，配乐库按它建拍网格、放同步点、导出 `events.json`，QA 工具按它对账切点。对写镜头和写乐谱的子代理来说它是只读契约，有问题写进报告，由主会话改。

两份样例：

- [`films/demo-light/timeline.json`](../films/demo-light/timeline.json)：beat3d + filmsynth，24 fps、90 BPM、每拍 16 帧，8 小节画面加 1 小节音乐尾巴。
- [`films/demo-paper/timeline.json`](../films/demo-paper/timeline.json)：paper2d + sampler，30 fps、72 BPM、每拍 25 帧，6 小节画面加 1 小节音乐尾巴。

这套写法来自《成路》，《一拍》和阡陌宣传片沿用并补充了几个字段（见第 9 节）。四个库各自怎么读它：beat3d 见 [README](../engines/beat3d/README.md) 第 1、6 节（`lib/time`、`lib/curves`），paper2d 见 [README](../engines/paper2d/README.md)「片目录约定」与组件 `beatGrid`，filmsynth 见 [README](../music/filmsynth/README.md) 第 4.2 节（`filmsynth.film`），sampler 见 [README](../music/sampler/README.md) 第 5 节（`sampler.grid`）。

## 1. 字段

| 字段 | 类型 | 说明 | demo-light | demo-paper |
| --- | --- | --- | --- | --- |
| `version` | 整数 | 时间线格式版本 | 1 | 1 |
| `title` | `{zh, en}` | 片名 | 光与弦 / LIGHT AND STRING | 纸上一线 / ONE LINE ON PAPER |
| `scope` | 文字 | 这份时间线管什么、和其他文件的关系 | 有 | 有 |
| `notation` | 文字 | 位置写法与帧公式（照第 2、3 节写，给读文件的人看） | 有 | 有 |
| `fps` | 整数 | 输出帧率 | 24 | 30 |
| `bpm` | 数 | 速度，全片固定 | 90 | 72 |
| `beatsPerBar` | 整数 | 每小节拍数 | 4 | 4 |
| `framesPerBeat` | 整数 | 每拍帧数，必须等于 `fps × 60 / bpm` | 16 | 25 |
| `framesPerBar` | 整数 | `framesPerBeat × beatsPerBar` | 64 | 100 |
| `bars` | 整数 | 小节数，含音乐尾巴 | 9 | 7 |
| `totalFrames` | 整数 | `bars × framesPerBar`，含尾巴 | 576 | 700 |
| `filmEnd` | 位置 | 画面在哪一拍结束；之后到 `totalFrames` 是尾巴 | `"9.1"` | `"7.1"` |
| `canvas` | `{preview, final}` | 预览与终版的画布像素 | 1920×1080 / 1920×1080 | 同左 |
| `aspect` | `{keys, interp}` | 可视画幅曲线（第 5 节） | 2.39 → 16:9 | 无（全程 16:9） |
| `energy` | `{keys, note}` | 画面与配乐共用的意图曲线 0..1（第 6 节） | 有 | 无 |
| `sections` | 数组 | 段落 `{id, zh, from, to}` | 5 段 | 4 段 |
| `shots` | 数组 | 镜头 `{id, zh, from, to, ...}`（第 7 节） | S01–S04 | P01–P03 |
| `sync` | 数组 | 同步点 `{id, at, hard, picture, music}`（第 8 节） | 5 个 | 无 |
| `cues` | 数组 | 声音提示 `{id, at, kind, note}`（第 8 节） | 无 | 7 个 |
| `loudness` | 对象 | 响度目标（第 8 节） | −16 LUFS / −1 dBTP | 同左 |

网格字段（`fps` 到 `totalFrames`）和 `shots` 是所有工具都要读的；其余字段按需取用：beat3d 读 `aspect` `energy` `persistence` `line` `sync` `cards` `tempo`，paper2d 的 `beatGrid` 读 `sections` `shots` `cues` `filmEnd`，filmsynth 读 `sync` `sections` `energy` 等曲线和 `loudness`，sampler 读 `cues` `shots` `sections`。paper2d、filmsynth、sampler 会检查 `framesPerBeat = fps × 60 / bpm`，对不上直接报错；beat3d 直接用 `framesPerBeat`、不复核，所以写时间线时要自己保证（第 11 节的片段会查）。beat3d 还要求有 `aspect`（全程一个比例也要写一个键）。时间线可以带额外字段，工具不认识的字段一律忽略。

## 2. 位置写法

位置是字符串 `"小节.拍"`，小节和拍都从 1 起：

| 写法 | 含义 | demo-light 里的帧 |
| --- | --- | --- |
| `"1.1"` | 第 1 小节第 1 拍，片头 | 0 |
| `"3.1"` | 第 3 小节第 1 拍 | 128 |
| `"4.4"` | 第 4 小节第 4 拍 | 240 |
| `"4.3+0.5"` | 4.3 之后半拍 | 232 |
| `"5.1+3f"` | 5.1 之后 3 帧 | 259 |
| `"1.3-2.4"` | 一个区间，两端都是拍位（只用于 `sync.at` 与 `cues.at`） | 32 到 112 |

规则：

- `+` 后面不带 `f` 是拍的小数，带 `f` 是帧数。时间线里偏移只往后、只写一段，四个库都能解析。代码里各库多认几种：paper2d 可以连写 `"3.1+0.5+2f"`，sampler 的 `Grid.pos` 认往前的 `"2.3-2f"`，beat3d 和 filmsynth 还接受对象或元组形式的位置（见各自 README）。
- 镜头与段落的 `from` 含、`to` 不含。`to` 可以写到 `bars + 1` 小节的第 1 拍（例如 demo-light 尾巴的 `"10.1"`），表示到总帧数为止。
- 区间 `"a-b"` 用作同步点时，如果 `id` 也写成区间且个数对得上（例如《一拍》的 `{"id": "T1-T6", "at": "1.3-2.4"}`，1.3 到 2.4 正好 6 拍），配乐库按拍展开成 T1…T6 六个点，两端都含；否则这个同步点记在 `a` 上，区间只说明它持续多久（demo-light 的 P1）。用作声音提示时，区间表示「从 a 起到 b 为止的一段」（paper2d 与 sampler 都按含 a、不含 b 处理），具体怎么发声由乐谱解释。
- 镜头代码、乐谱、QA 脚本里都用这套写法，经工具换算；不在代码里写手算的帧号或秒数。

## 3. 帧换算

```
beat(pos)  = (bar − 1) × beatsPerBar + (beat − 1) + 拍的小数
frame(pos) = beat(pos) × framesPerBeat + 帧偏移
秒(pos)    = frame(pos) / fps = beat(pos) × 60 / bpm
```

速度和帧率要选成每拍整数帧，否则拍点会落在两帧之间：

| bpm | fps | 每拍帧数 | 用在 |
| --- | --- | --- | --- |
| 90 | 24 | 16 | 《一拍》、demo-light |
| 72 | 24 | 20 | 《成路》、阡陌宣传片的 24 格网格 |
| 72 | 30 | 25 | 阡陌宣传片 30 fps 成片、demo-paper |
| 120 | 24 | 12 | （可用） |
| 100 | 30 | 18 | （可用） |

拍子越分越细时网格也要整齐：90 BPM、24 fps 下八分音符 8 帧、十六分音符 4 帧；72 BPM、24 fps 下八分 10 帧、十六分 5 帧。《一拍》的进程段靠「四分 → 八分 → 三连 → 十六分」加速，速度不变。三连音落在 16/3 帧上，不在整帧，画面按小数帧取子帧即可。

**24 格网格、30 fps 输出。** 阡陌宣传片的场景按 24 格网格写（一拍 20 格），成片按 30 fps 出：输出第 n 帧显示网格时间 `n × 24 / 30`（垫片里的 `Retimed` / `Freeze` 支持小数帧）。这样旧的帧号全部保留，72 BPM 在 30 fps 下正好每拍 25 帧。报告和交接记录里写帧号时要注明是网格帧号还是输出帧号。

**总长与尾巴。** `bars` 和 `totalFrames` 含尾巴。画面到 `filmEnd` 为止，之后的帧画面是纯黑（demo-light）或停在纸色（demo-paper），配乐的混响尾巴在这段里自然衰减。两部示例片：

| | 画面 | 尾巴 | 总长 |
| --- | --- | --- | --- |
| demo-light | 1.1–9.1，帧 0–511，21.333 s | 9.1–10.1，帧 512–575 | 576 帧，24.000 s |
| demo-paper | 1.1–7.1，帧 0–599，20.000 s | 7.1–8.1，帧 600–699 | 700 帧，23.333 s |

## 4. 画布

`canvas.preview` 是预览和样张的尺寸，`canvas.final` 是母版尺寸。引擎里的长度一律按 1080p 像素写，乘 `scale = 高 / 1080`，所以 4K 和 1080p 的画面只差分辨率。《一拍》的时间线写了 4K 终版，最后按导演本只出了 1080p 母版（4K 渲染时间约 4 倍）。

## 5. 画幅 `aspect`

`keys` 是 `[位置, 宽高比]` 的列表，宽高比是可视画面的比例，画布不变，多出来的部分是上下对称的纯黑遮幅：

```
可视高 = min(画布高, 画布宽 / 宽高比)
遮幅高 = round((画布高 − 可视高) / 2)      # 1920×1080、2.39 时上下各 138 行
```

- **跳变**：在同一位置写两个键（demo-light 的 `["5.1", 2.39], ["5.1", 1.7778]`），或者两个键相差不超过 1 帧且值不同（《一拍》的 `["16.1", 2.39], ["16.1+1f", 1.7778]`），从前一个键的那一帧起就取后一个值。也就是说，画幅在切点的第一帧就打开。
- **渐变**：两个键离得远时按 `interp` 插值。beat3d 认 `constant`（保持前一个键的值直到下一个键）、`linear`、`smooth`、`easeInOutCubic`（默认）等，只看第一个词，后面可以接说明（demo-light 写的是 `"constant；5.1 从 2.39 遮幅直接跳到 16:9……"`）；`aspect.eases` 可以给某个键单独指定它起头那一段的缓动。《成路》在 30.1–30.3 用 easeInOutCubic 缓缓打开到 16:9。
- paper2d 不读 `aspect`，画面比例就是合成的宽高（demo-paper 全程 16:9，时间线里没写）。beat3d 必须写，全程不变时写一个键即可。
- QA 里量画面颜色时要先裁掉遮幅：遮幅是纯黑，不是画面里的暗部（见 [QA.md](QA.md) 的色调检查）。

## 6. 能量 `energy`

`keys` 是 `[位置, 0..1]`，表示这一刻画面密度和配乐密度、响度共用的意图。它不是混音自动化，也不是硬指标：

- 配乐库可以把它读成随时间变化的曲线（filmsynth 的 `timeline_curve(session, timeline, "energy")`，键间线性插值），用来定声部数、力度和响度走向，并用 `s.export_curve` 导出到 `events.json` 的 `curves` 里；`filmsynth.check` 的 `energy_follow` 量 1 秒响度跟这条曲线的相关（demo-light 实测 r = 0.94）。
- 画面可以按它定粒子数、光的强弱、镜头速度（beat3d 的 `ENERGY`，键间用单调插值，和配乐那边的线性插值在键上一致、键间略有不同）。
- 想要一刀落到零，就像《一拍》那样在相邻位置写两个键（`["14.3", 0.9], ["14.3+8f", 0.0]`）。

demo-light 的能量从 1.1 的 0 爬到 5.1 的 0.85（H2 画幅打开），再退到 9.1 的 0。

## 7. 段落与镜头

`sections` 是大段落，配乐的换段、响度表、QA 的分段统计都按它；`shots` 是镜头，引擎按镜头找程序，QA 按镜头边界对账切点。

镜头字段：

| 字段 | 说明 |
| --- | --- |
| `id` | 镜头号，全片唯一（`S01`、`P01`） |
| `zh` | 中文名 |
| `from` / `to` | 起止位置，`from` 含、`to` 不含；镜头之间不重叠，可以留空（空的部分引擎画纯黑或占位） |
| `continuesInto` | 下一个镜头的 `id`：这两个镜头是一镜到底，交界处不该有跳变（《一拍》用过，给人看的标注，工具不读；`qa-cuts` 会把没有跳变的镜头交界列为「软切」） |
| `cuts` | 镜头内的蒙太奇切点 `[{at}]`：`qa-cuts` 把它们也当预期切点 |
| `lens` / `camera` | 焦段、光圈、机位与运动 |
| `content` | 画面内容，写到能照着做的程度：什么时刻出现什么、对着哪个拍点 |
| `owner` | 归哪一路（《一拍》全片时间线用 `P1`、`P2`、`P3`），多路并行时用 |
| `spec` | 细节在哪份文件里（任务书、审片意见） |

一个镜头可以由一个程序画，也可以几个连续镜头由同一个程序画（一镜到底）。镜头程序和登记方式见各引擎的 README。

## 8. 同步点、声音提示与响度

**`sync`**：音画必须对上的点，beat3d + filmsynth 的片子用它。

| 字段 | 说明 |
| --- | --- |
| `id` | `H1`、`H2`……硬同步；`P1`、`T1-T6` 这类软同步 |
| `at` | 位置或区间 |
| `hard` | `true` 是硬同步：画面事件和音乐事件都落在这一帧，自检逐条量偏差（起音在网格 ±10 ms 内，重拍可以晚 0–2 帧）；`false` 是软同步：音乐照常流动，画面去找音乐的事件 |
| `picture` / `music` | 两侧各发生什么 |

硬同步点全片不超过 12 个，放在结构边界上（见 [METHOD.md](METHOD.md) 第 2.4 节）。demo-light 有三个硬同步：H1 3.1 光聚成字配和弦落下，H2 5.1 穿过字面、画幅打开配渐强到顶后的低频轻击，CUT 9.1 切黑配干切。

**`cues`**：给配乐的声音提示，paper2d + sampler 的片子用它。`kind` 是提示的类别（demo-paper 用了 `sfx`、`hit`、`accent`、`phrase`、`cut`），`note` 写清声音是什么。画面用 `beatGrid(timeline).cue(id)` 取起止帧，配乐用 `Grid.cue(id)` 取起止秒。demo-paper 的乐谱把每条提示写成 `events.json` 的一个 marker：`hit` `accent` `cut` 记为 `hard_sync`，`sfx` `phrase` 记为 `soft_sync`，原来的类别留在 `cue_kind`（这是那份乐谱的约定，不是库的规定）。阡陌宣传片的场景代理在各幕的 `cues_<幕>.json` 里列音效时刻（`{frame, kind, note}`），混音时照着放；示例片改为直接写在时间线里。

**`loudness`**：

| 字段 | 说明 |
| --- | --- |
| `integratedLUFS` | 整体响度目标（两部示例片都是 −16） |
| `truePeakDbtp` | 真峰值上限（−1 dBTP） |
| `keys`（可选） | `[位置, 1 秒响度]`：每个点以该位置为中心取 1 秒 K 加权响度，容差 ±3 dB；硬同步点的居中窗会跨过切点，改取「之后 1 秒」 |
| `spanMinDb`（可选） | 动态跨度下限：最响与最静的 1 秒窗之差（《一拍》30 dB，《成路》25 dB） |

《一拍》和《成路》的时间线用的是 `integrated`、`truePeakMax` 这两个名字，意思相同。

## 9. 三部片里出现过的扩展字段

| 字段 | 出处 | 用途 |
| --- | --- | --- |
| `line` | 《成路》 | 贯穿全片的横线：`{"yFrac": 0.588}`。这个高度是 2.39 遮幅可视区（y 0.128–0.872）的黄金分割，画幅打开到 16:9 时线不动 |
| `persistence` | 《成路》 | 「保留度」曲线 0..1，同时驱动脚印光痕的半衰期（h = 0.4 s × 10^(3p)）和拨弦回声的反馈量（fb = 0.15 + 0.45p）。映射公式在画面与配乐两边各写一份，要保持一致 |
| `cards` | 《成路》《一拍》 | 字幕卡与片尾：位置、种类（打字 / 光写 / 电影字幕 / 品牌）、文字来源 |
| `music` | 《成路》《一拍》 | 调式、速度规则、禁用音色清单 |
| `shots[].owner` / `spec` | 《一拍》 | 多路并行时镜头归属与细节出处 |
| `tempo` | beat3d | `"fixed"`（默认，固定速度）或 `"score"`：以配乐 `events.json` 的拍点为钟，乐谱里的渐慢会带着画面一起慢；没有配乐时退回网格 |
| `aspect.eases` | beat3d | 逐键指定画幅曲线的缓动（第 5 节） |
| `shots[].cuts` | beat3d | 镜头内的蒙太奇切点，供 `qa-cuts` 对账（第 7 节） |

## 10. `events.json`：配乐交给画面的拍点与事件

配乐渲染时在 `<片>/out/music/events.json` 导出拍点、同步点和事件。filmsynth 与 sampler 写的是同一套顶层字段（规范以 [filmsynth README](../music/filmsynth/README.md) 第 8 节为准，sampler 见 [README](../music/sampler/README.md) 第 5 节的 `EventLog`）。beat3d 在构建时把它烘进页面包，渲染期间不再读文件；paper2d 不读它，画面和配乐各自按时间线换算，靠配乐的 `measured.sync` 证明落点。

时间一律是从 1.1 起算的秒，`frame = t × fps`（可以是小数）。

| 字段 | 说明 | demo-light（filmsynth） | demo-paper（sampler） |
| --- | --- | --- | --- |
| `version`、`sampleRate`、`fps`、`bpm`、`beatsPerBar`、`bars`、`tempoRamps` | 渲染时用的网格；`tempoRamps` 是渐慢记录，固定速度时为空 | 1、48000、24、90、4、9、`[]` | 1、48000、30、72、4、7、`[]` |
| `end` / `duration` | 画面结束时刻与音频总长（秒） | 24.0 / 24.0 | 20.0 / 23.333333 |
| `beats` | 每拍起点的秒数，下标 0 是 1.1，含曲终那一拍：`bars × beatsPerBar + 1` 个 | 37 个 | 29 个 |
| `barTimes` / `beatTimes` | `[{bar, t, frame}]`（含曲终那一小节的起点）/ `[{bar, beat, t, frame}]` | 10 / 37 | 8 / 29 |
| `markers` | 同步点与设计位置。时间线的同步点是 `kind: hard_sync / soft_sync` | `{name, pos, t, kind, picture}`，区间同步点带 `to`；另有设计位置 `kind: lead`（带 `frames` 领先帧数与 `of`，例如涌起的峰 `swell_peak` 在 H1 前 2 帧） | `{name, pos, t, frame, kind, cue_kind, note}`，区间提示带 `end` 与 `end_frame` |
| `cuts` | 干切 `{start, end, choke, tails}`，`tails: true` 表示干声停、混响尾巴继续响 | 9.1 一刀 | 7.1 一刀，另带 `note` |
| `events` | 给画面读的事件 `{t, frame, type, pos, ...}`，按 `t` 排序；`type` 和其余字段由这部片的乐谱定 | 见下 | 见下 |
| `notes` | 每个音 | `{track, stem, pos, beat, start, end, frame, vel, pitch, midi, hz}` | `{track, stem, pos, start, end, frame, vel, pitch, midi}`，部分带 `pan` `ring_s` `sec` |
| `curves` | 乐谱导出的曲线 | `{name: {hop, t0, values}}`，例如 `energy` 每 0.05 s 一个值 | `{name: [{t, frame, v}]}`，例如 `strings_gain_db`、`pad_gain_db` 的折点 |
| `measured` | 自检从成品音频量出来的结果 | `sync`（每个同步点的实测偏差）、`cut`、`lufs_i`、`true_peak_dbtp` | `sync`、`summary`、`master`、`outputs`、`samples`（打开过的全部采样，用来核对署名） |

两部示例片的事件类型：

- **demo-light**（详见 `films/demo-light/score/DESIGN.md` 第 4 节）：`pulse`（`shot` `side` `light` `div` `vel` `bright` `pitch`，画面按 `shot` 分开用：S01 用 `light` 点亮哪一粒光，S03 用 `side` 决定哪一侧的弦）、`run`（4.4 的上行，`shot` `n` `div` `vel` `pitch`）、`chime`（`line` `pitch`）、`chord`（`chord`）、`motif`（`n` `pitch` `vel`）、`release`、`impact`、`sparkle`（`beats`）、`taut`（`pitch` `to`）、`land`、`cut`（`tails`）。挂在同步点上的事件带 `id`（如 `H1`）。
- **demo-paper**（详见 `films/demo-paper/score/README.md`）：`pencil`（`side` 是卡片的哪一条边，`frames` 一笔几帧，`pan` 起止声像）、`pencil_tick`（`corner`）、`paper`（`source` 用了哪条录音）、`chord`、`run`、`taut`（`stem` `pitches` `what`）、`thread`、`phrase`（`stem` `pitch`）、`chime`、`name`、`repo`、`cut`。用 `place_on_attack` 放下的录音还带对齐记录 `target_s` `attack_sample` `placed_start_sample` `error_samples`（误差 0 样本）。

《一拍》的乐谱另有 `di` / `da` 打字音、`pluck`（带 `string`）、`arrival`（五块屏幕的到达音）等，可以看出 `type` 完全由片子自己约定。

**对齐规则：每一拍与网格差不超过 1 ms。** 第 i 拍应在 `i × 60 / bpm` 秒。beat3d 的 `scripts/build.ts` 构建时逐拍检查，超过 1 ms 就报错退出，不允许画面悄悄和音乐漂开；配乐一侧 `filmsynth.check` 的 `events_check`、`sampler.check.grid_alignment` 也各量一遍。两部示例片实测最大偏差都是 0.0003 ms（秒数保留 6 位小数的舍入）。

画面怎么用（beat3d）：

- 事件的帧位是 `t × fps`（小数帧），画面可以按单个音符动：镜头程序用 `CLOCK.eventFrames(type, from, to)` 取一段里的事件，例如 demo-light 按 `pulse` 让光点一明一灭、让脉冲沿弦跑。
- 事件里的 `t` 是实际起点：人性化过的音符（例如《一拍》的 `motif` 带 5 ms 偏差）不在网格上，给画面卡点的音（脉冲、硬同步）不加人性化，在网格上。
- 没有 `events.json` 时画面按网格走（demo-light 的回退：S01 八分音符、S03 十六分加八分、S04 四分与二分），结果与有乐谱时在 1 ms 内一致，并且无声出片。
- 配乐改了，`events.json` 就变了，画面要重新构建、重渲；不重渲不会报错，只会不同步。

## 11. 自查片段

下面的片段检查网格字段是否自洽、位置能否解析、镜头是否重叠、是否都落在总帧数之内，并打印每个镜头、同步点、声音提示的帧号与秒数。只用 Python 标准库，存成 `timeline_check.py` 就能跑：

```python
import json, re, sys

POS = re.compile(r"^\s*(\d+)\.(\d+)(?:\s*\+\s*([0-9.]+)(f?))?\s*$")

def frame(tl, pos):
    m = POS.match(pos)
    if not m:
        raise SystemExit(f"位置写法不对：{pos!r}")
    bar, beat, off, unit = int(m[1]), int(m[2]), m[3], m[4]
    if not 1 <= beat <= tl["beatsPerBar"]:
        raise SystemExit(f"拍号越界：{pos!r}")
    f = ((bar - 1) * tl["beatsPerBar"] + (beat - 1)) * tl["framesPerBeat"]
    if off:
        f += float(off) if unit == "f" else float(off) * tl["framesPerBeat"]
    return f

tl = json.load(open(sys.argv[1], encoding="utf-8"))
fpb = tl["fps"] * 60 / tl["bpm"]
assert fpb == int(fpb) == tl["framesPerBeat"], "framesPerBeat 应为 fps×60/bpm"
assert tl["framesPerBar"] == tl["framesPerBeat"] * tl["beatsPerBar"]
assert tl["totalFrames"] == tl["bars"] * tl["framesPerBar"]
last = 0
for s in tl["shots"]:
    a, b = frame(tl, s["from"]), frame(tl, s["to"])
    assert last <= a < b <= tl["totalFrames"], s["id"]
    last = b
    print(s["id"], s["from"], s["to"], a, b, round(a / tl["fps"], 3))
for s in tl.get("sync", []) + tl.get("cues", []):
    a = frame(tl, s["at"].split("-")[0])
    print(s.get("id", "-"), s["at"], a, round(a / tl["fps"], 3))
```

对两部示例片跑的结果（镜头每行是 id、起、止、起始帧、终止帧、起始秒；同步点和提示每行是 id、位置、帧、秒）：

```
$ python3 timeline_check.py films/demo-light/timeline.json
S01 1.1 3.1 0 128 0.0
S02 3.1 5.1 128 256 5.333
S03 5.1 7.1 256 384 10.667
S04 7.1 9.1 384 512 16.0
P1 1.3-2.4 32 1.333
H1 3.1 128 5.333
H2 5.1 256 10.667
H3 7.1 384 16.0
CUT 9.1 512 21.333
$ python3 timeline_check.py films/demo-paper/timeline.json
P01 1.1 3.1 0 200 0.0
P02 3.1 5.1 200 400 6.667
P03 5.1 7.1 400 600 13.333
pencil 1.3-2.3 50 1.667
paper 2.3 150 5.0
line 3.1 200 6.667
word1 4.1 300 10.0
word2 4.3-5.1 350 11.667
end 5.1 400 13.333
cut 7.1 600 20.0
```

同一个片段对《一拍》35 小节、2240 帧的全片时间线和《成路》40 小节、3200 帧的时间线也跑通。《成路》的软同步点有几个只写了 `{at, hard, event}`、没有 `id`，所以片段里用 `s.get("id", "-")`；新片的同步点都写 `id`。
