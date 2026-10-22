# paper2d — React 场景按帧号出画

场景用 React 写成「帧号的纯函数」，引擎把它们渲成逐帧确定的画面：

1. **垫片**：自己实现场景用到的那部分 Remotion API（`useCurrentFrame`、`Sequence`、`interpolate`、`spring`…），是一个**兼容子集**，按公开文档描述的行为编写，没有复制或改写 Remotion 的源码（数值部分：`Easing` 与贝塞尔求解来自 React Native / bezier-easing，MIT；`interpolate`、`spring`、`measureSpring`、`random` 是独立实现。来历与许可见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)），也不需要安装 Remotion 或任何同类视频框架；
2. **打包**：`Bun.build` 在内存里把片的入口打成一个页面脚本，裸模块名 `'remotion'`（或 `'paper2d'`）指到垫片；
3. **截帧**：本机 Google Chrome 无头运行，经 CDP 截合成后的整页（DOM、SVG、Canvas、WebGL 一起）；
4. **编码**：多个标签页并行截帧，按帧序直接写进 ffmpeg 的 stdin，出 H.264（不落临时图片目录）。

移植自阡陌宣传片的自写渲染器（那部片的场景当年按 Remotion 的 API 写成，后来改由这套引擎渲染，并逐帧对过 Remotion 渲的母版，见文末「保真」——那是当年那一版垫片的验收；agentreel 里的垫片之后重写了 `interpolate`、`spring`、`measureSpring`、`random`，数值可能与 Remotion 不同；那部片的制作经过见 [`docs/cases/qianmo-promo.md`](../../docs/cases/qianmo-promo.md)）。这里把它与那部片解耦：引擎不含任何一部片的内容，一部片是 `films/<名字>/` 一个目录。Remotion 是 Remotion AG 的产品与商标，本项目与其无关，兼容它的 API 只是为了让按那套写法写的场景能直接用。

## 依赖与安装

- [Bun](https://bun.sh) ≥ 1.3（打包、HTTP 服务、CDP 客户端都用 Bun 自带的）
- Google Chrome 或 Chromium（无头 `--headless=new`）
- ffmpeg 与 ffprobe
- 可选：fontTools（`pyftsubset`，只有做字体子集时用；有 [uv](https://docs.astral.sh/uv/) 时会用 `uvx` 临时装）

```sh
cd engines/paper2d
bun install            # 只有 react、react-dom 和类型；片目录不用单独装
bun run typecheck      # 引擎
bun run test           # 垫片数值部分的自检（bun test）
bun run typecheck:film ../../films/demo-paper   # 一部片
```

外部程序的位置（`lib/tools.ts`）：

| 变量 | 作用 | 不设时 |
| --- | --- | --- |
| `CHROME_PATH` | 浏览器可执行文件 | macOS：`/Applications`、`~/Applications` 下的 Google Chrome / Chromium；Windows：`%PROGRAMFILES%`、`%PROGRAMFILES(X86)%`、`%LOCALAPPDATA%` 下的 `Google\Chrome\Application\chrome.exe`；Linux：`/usr/bin/google-chrome-stable` 等；最后在 PATH 里找 `google-chrome-stable` / `google-chrome` / `chromium` / `chromium-browser` / `chrome` |
| `FFMPEG` / `FFPROBE` | ffmpeg / ffprobe | PATH 里的 `ffmpeg` / `ffprobe` |
| `PAPER2D_GL` | GL 后端（同 `--gl`） | 见下表 |
| `PAPER2D_CHROME_FLAGS` | 额外的 Chrome 参数，空格分隔（同 `--chrome-flags`） | 无。Linux 容器里以 root 跑时通常要 `--no-sandbox` |
| `PYFTSUBSET` | 字体子集工具 | PATH 里的 `pyftsubset`，否则 `uvx --from fonttools[woff] pyftsubset` |

GL 后端（`lib/chrome.ts`）：

| 名字 | 用途 | 默认于 |
| --- | --- | --- |
| `metal` | ANGLE 的 Metal 后端，WebGL2 在 GPU 上（Apple 芯片实测） | macOS |
| `d3d11` | ANGLE 的 Direct3D 11 后端 | Windows |
| `angle` | ANGLE 的平台默认后端（`--use-gl=angle`） | — |
| `vulkan` / `egl` | 有 GPU 的 Linux | — |
| `swangle` | SwiftShader（CPU），没有 GPU 的机器 | Linux |

不同后端、不同机器之间，像素不保证逐位相同。同一台机器、同一后端上：静帧逐位相同；整片里有少数帧会有极小的随机差异（见「确定性与浮点」）。

## 快速开始（示例片）

```sh
cd engines/paper2d
# 静帧：帧号，或按片的 timeline.json 写「小节.拍」
bun still.ts --film ../../films/demo-paper --frames 0,2.3,3.1+12f,4.1 --scale 0.5
# 整片：6 个标签并行，PNG 截帧，x264 crf 15 slow → ../../films/demo-paper/out/DemoPaper.mp4
bun render.ts --film ../../films/demo-paper --hashes ../../films/demo-paper/out/DemoPaper.hashes.tsv
# 配乐混进去（画面原样拷贝，音频 AAC；长度以画面为准）
bun mux.ts --video ../../films/demo-paper/out/DemoPaper.mp4 --audio ../../films/demo-paper/out/music/score.wav
# 相邻帧跳变扫描（按片的时间线标注镜头与小节.拍）
bun scan.ts ../../films/demo-paper/out/DemoPaper.mp4 --film ../../films/demo-paper
```

## 片目录约定

```
films/<名字>/
  src/index.ts      registerRoot(Root)；Root 里放 <Composition id=… component=… durationInFrames fps width height/>
  public/           staticFile('fonts/x.woff2') 指向这里，原地服务、不拷贝
  timeline.json     可选：拍网格（写法见下面的链接）；有它时帧列表能写「小节.拍」，scan 能标注镜头
  tsconfig.json     {"extends": "../../engines/paper2d/tsconfig.film.json", "include": ["src"]}
  out/              输出（不入库）：stills/<组合>/、<组合>.mp4、<组合>.render.json …
```

`timeline.json` 的写法见 [`docs/TIMELINE.md`](../../docs/TIMELINE.md)。画面不读配乐的 `events.json`：画面和配乐各按 `timeline.json` 换算，落点由配乐自检的同步表（`events.json` 的 `measured.sync`）证明。

片的代码：

```tsx
import {AbsoluteFill, Sequence, useCurrentFrame} from 'remotion';      // 或 from 'paper2d'，同一个模块
import {beatGrid, HandPaper, Kinetic, Stroke} from 'paper2d/components';
import timeline from '../timeline.json';
const T = beatGrid(timeline);      // T.at('3.1+4f') → 帧号；T.cue('line').start；T.shot('P02')
```

片目录不装依赖：打包时 `react*` 一律解析到这个引擎的 `node_modules`（保证只有一份 React），类型检查靠
`tsconfig.film.json` 的 `paths` 指过来。`@remotion/*` 打包直接报错（只提供 `'remotion'` 这一个模块）。

## 命令

所有命令出错时只打一行 `error: …` 并以 1 退出（`--verbose` 带堆栈）；会话启动（打包、Chrome、CDP、页面）失败重试 3 次并打印原因（场景文件正被别的编辑器写到一半时打包会失败，过一秒就好）；参数错、组合不存在不重试。共同参数：`--film <片目录>`（或 `--entry <文件> [--public <目录>]`）、`--comp <id>`（片里只有一个组合时可省）、`--props '<json>'`（`getInputProps()`）、`--gl <后端>`、`--chrome-flags '<参数>'`、`--timeout <毫秒>`（delayRender 默认超时）、`--verbose`。

| 命令 | 用途与参数 |
| --- | --- |
| `still.ts` | 单帧 PNG。`--frames 0,150,600-700:20,2.3,3.1+12f,1.1-3.1:25`（位置范围不含终点）；`--out` 目录（默认 `<片>/out/stills/<组合>`）；`--scale 0.5` 出半尺寸；`--grid 24` 把数字当另一张网格（如 24 格）的帧号并四舍五入到输出帧；`--prefix`。文件名 `<前缀或组合>_<帧号四位>.png` |
| `render.ts` | 整段 / 整片 MP4（无声）。`--out`（默认 `<片>/out/<组合>.mp4`）、`--from` / `--to`（含两端，可写小节.拍）、`--tabs 6`、`--crf 15`、`--preset slow`、`--capture png\|jpeg [--quality 100]`、`--scale`、`--keep 帧列表 --keep-dir 目录`（顺手存这些帧的截图）、`--hashes 文件.tsv`（每帧截图的哈希）、`--audio score.wav [--with-audio-out 文件]`（渲完顺手混音）。先写 `<out>.part`，ffmpeg 正常结束才改名；旁边写 `<out 去掉 .mp4>.render.json`（参数与耗时） |
| `mux.ts` | `--video 画面.mp4 --audio 配乐.wav [--out 文件] [--bitrate 320k] [--offset 秒]`。视频流拷贝，音频 AAC-LC；配乐短了补静音、长了截断，都会提示 |
| `pixdiff.ts` | `bun pixdiff.ts 目录A 目录B`：两组同名截图逐像素比（相同 / 不同的通道数、最大差、PSNR）；`--video 文件.mp4 --start 首帧帧号 目录`：截图对视频里的同一帧（按视频自己的矩阵与范围解码） |
| `scan.ts` | `bun scan.ts 视频.mp4 [--film 片目录 \| --timeline 文件] [--start n] [--top 15] [--out 前缀]`：相邻帧差扫描找跳变，出全量 TSV 与前 N 名的对照条（n−1 \| n \| 差 ×4） |
| `fidelity.ts` | `--ref 母版.mp4 --frames …`：同一帧对母版比 Y / YUV / RGB PSNR，低于 `--threshold`（默认 34 dB）的帧自动出对照图（`--images` 全出）；`--mine 目录 --comp id` 用已有截图不开浏览器；`--ref-offset k` 查时间对齐 |
| `tools/subset-font.ts` | `--font 源字体 --out 子集.woff2 --text-file src/copy.ts [--text …] [--no-ascii]`：只留画面用到的字形（`.ts` 文件只取字符串字面量里的字），保留全部 name 记录（版权、许可） |

这些检查在验收里各管什么（确定性比 `--hashes`、跨后端用 `pixdiff`、跳变用 `scan`、换渲染器用 `fidelity`），以及和 beat3d、配乐库的检查怎么对应，见 [`docs/QA.md`](../../docs/QA.md) 的「工具对照」。

## 结构

| 文件 | 作用 |
| --- | --- |
| `shim/remotion.ts` | 片看到的 `'remotion'` / `'paper2d'` 模块：导出下表全部 API |
| `shim/math.ts` | `interpolate`、`Easing`（含 cubic-bezier 求解）、`spring` / `measureSpring`、`random` |
| `shim/math.test.ts` | 上面这些的自检（`bun run test`）：spring 三种阻尼对照独立的 RK4 数值积分、overshootClamping、reverse、delay、durationInFrames、measureSpring 的定义、random 的确定性与分布、interpolate / bezier 的边界 |
| `shim/core.tsx` | 帧与视频配置上下文、`Sequence`、`Freeze`、`AbsoluteFill`、`Img`、`Audio`、`Composition` / `Still` / `Folder`、两个 hook |
| `shim/state.ts` | `delayRender` / `continueRender` / `cancelRender` 句柄表、注册的根组件、页面注入的数据（`window.__PAPER2D__`） |
| `page/entry.tsx` | 页面运行时：导入片的入口（`'paper2d:film'`），暴露 `window.paper2d`（`compositions()` / `select(id)` / `frame(n)`） |
| `lib/build.ts` | `Bun.build` 内存打包（不落盘，多个会话并行互不干扰）与模块别名 |
| `lib/server.ts` | 127.0.0.1 随机端口：页面 HTML、内存里的 bundle、`/public/*` 原地服务片的 `public/` |
| `lib/chrome.ts` | 无依赖 CDP 驱动（Bun 自带 WebSocket），一次性 profile 目录；GL 后端参数表 |
| `lib/tools.ts` | Chrome / ffmpeg / ffprobe 的查找、GL 平台默认、子进程与 ffprobe 小工具 |
| `lib/film.ts` | 片目录解析（入口、public、timeline、out） |
| `lib/session.ts` | 参数与帧列表解析；会话 = 打包 + 服务 + Chrome + N 个标签页（各自选好组合、设好视口），启动失败重试；`captureFrame`；`runCli` |
| `lib/pipeline.ts` | `renderOrdered`：帧按轮转固定分给各标签（第 k 个标签渲 k、k+N、k+2N…，见「确定性与浮点」），重排缓冲按帧序交给写出端，领先不超过窗口（默认 3×标签数）；`H264Writer` |
| `lib/mux.ts` | 画面 + 配乐 |
| `components/` | `'paper2d/components'`：纸面场景的通用画法（见下） |
| `still.ts` `render.ts` `mux.ts` `pixdiff.ts` `scan.ts` `fidelity.ts` `tools/subset-font.ts` | 命令行入口 |
| `THIRD_PARTY_NOTICES.md` | React Native、bezier-easing（MIT 全文）、mulberry32 与 FNV-1a（公有领域）的署名 |

## 垫片的 API 约定（按 Remotion 4 的公开文档描述的行为；数值部分是独立实现，可能与 Remotion 不同）

| API | 约定 |
| --- | --- |
| `useCurrentFrame()` | 时间线帧 T − 所在段的偏移（各级 `from` 之和）；合成根 T 是合成帧。算术按文档描述的模型（各级 `from` 相加，再从 T 里减去），浮点细节见下面 `Freeze` |
| `useVideoConfig()` | `{id,width,height,fps,durationInFrames,defaultProps,props,…}`；在 `Sequence` 内 `durationInFrames`（及给了的 width/height）换成该段的 |
| `Sequence` | `from`（默认 0）、`durationInFrames`（默认无穷，且不超过父段剩余）、`name`（仅标注）、`layout`（`absolute-fill` 默认包一层 AbsoluteFill 且 `flexDirection` 置空即 row；`none` 不包）、`style`/`className`/`width`/`height`。只在 `from ≤ 父帧 < from + 时长` 时渲染子树，小数帧同样适用 |
| `Freeze` | 子树看到所在段内的第 `frame` 帧（可小数，`Retimed` 靠它），里面的 `Sequence` 从它起算；`active` 支持布尔或函数。实现上把 T 设为 `frame + 偏移`，子树算 `(frame + 偏移) − 偏移`——数学上等于 `frame`，浮点上不一定（217.6 + 320 − 320 = 217.60000000000002）。用帧号字符串做种子的东西（`rnd('gx' + frame)`）依赖这个精确值，所以算术不做化简 |
| `AbsoluteFill` | `position:absolute; inset 0; width/height 100%; display:flex; flex-direction:column`，传入 style 覆盖 |
| `interpolate` | 多段线性 + `easing`；`extrapolateLeft/Right`：`extend`（默认）/`clamp`/`identity`（与 React Native 的 Animated 插值同义）/`wrap`（折回整个输入区间）；节点归左边那段；输入区间须严格递增、全是有限数，两边长度一致。独立实现 |
| `Easing` | `linear ease quad cubic poly sin circle exp elastic back bounce bezier in out inOut step0 step1`；公式来自 React Native 的 `Easing.js`；`bezier` 是 React Native 的 `bezier.js`（bezier-easing 的移植）的算法：float32 采样表 + 4 步牛顿 + 二分（MIT，见 `THIRD_PARTY_NOTICES.md`） |
| `spring` | 阻尼振子的解析解（欠阻尼 / 临界 / 过阻尼三式），直接在 t = 帧号 / fps 求值，不逐帧步进，小数帧同样精确；默认 `damping 10, mass 1, stiffness 100`；支持 `from/to/durationInFrames/durationRestThreshold/delay/reverse` 与 `config.overshootClamping`（不越过 `to`）。`durationInFrames` 把弹簧自己的落定时间拉成那么多帧，到点后正好是 `to`；`reverse` 是同一段倒放。独立实现：与 Remotion 的数值可能在末位上不同；过阻尼弹簧按真正的过阻尼解算（比把它当临界阻尼的算法慢），`durationInFrames` / `reverse` 依赖 `measureSpring`，可能差一帧 |
| `measureSpring` | 从第 0 帧起逐帧看，返回此后每一帧都落在 `threshold`（默认 0.005）以内的第一帧；振子的衰减包络一低于阈值就停（之后不可能再出界）。与 Remotion 的结果可能差一帧 |
| `random(seed)` | 字符串种子：UTF-16 字节的 32 位 FNV-1a → mulberry32 取一个数；数字种子：float64 的 8 个字节做 FNV-1a → mulberry32（`0` 与 `-0` 相同，`NaN` 报错）；`null` → `Math.random()`（不确定，场景里别用）。同一个种子给出的数与 Remotion 不同 |
| `delayRender/continueRender/cancelRender` | 句柄表；默认超时 30 s（`timeoutInMilliseconds` 覆盖，`--timeout` 改默认）；超时或 `cancelRender` 让本帧报错退出；`retries` 不实现 |
| `Img` | `<img decoding="sync">`，挂载/换 src 时开句柄，`decode()` 完成才放；加载失败且无 `onError` 时 `cancelRender` |
| `Audio` | 不渲染任何东西（画面渲染不出声，配乐另行合成，用 `mux.ts` 混） |
| `staticFile(p)` | `/public/<逐段 encodeURIComponent 的 p>`；拒绝 http(s)、`./`、`..` |
| `getStaticFiles()` | 片的 `public/` 下全部文件 `{name,src,sizeInBytes,lastModified}`（按名字排序，点文件跳过） |
| `getInputProps()` | `--props` 传入的 JSON（默认 `{}`）；合成组件拿到 `defaultProps` 合并 input props |
| `registerRoot` / `Composition` / `Still` / `Folder` | 根组件只在「登记模式」渲染一次，收集 `{id, component, durationInFrames, fps, width, height, defaultProps}`；校验 id 与正整数尺寸；`lazyComponent` / `calculateMetadata` 不支持 |

未实现：`Series`、`Loop`、`interpolateColors`、`Video`/`OffthreadVideo`、`prefetch`、`useCurrentScale` 等。场景若导入这些名字，打包直接报 “No matching export”，不会静默出错。

## 页面与「帧就绪」

- 页面 CSS：`* { box-sizing: border-box } body { margin: 0 }`。**border-box 不能省**：固定高度又带 padding 的盒子在 content-box 下会长高（原片里一张卡片的第三行因此掉出卡外）。
- DOM 层次：`#paper2d-canvas`（合成宽高，flex）→ 绝对铺满的 portal（flex column）→ 合成自己的 AbsoluteFill → 合成组件。
- 每帧先把合成隐藏（`display:none`）两帧再显示，让 Chrome 丢掉旧的合成层重新建：Chrome 按层保留栅格化状态（缩放/平移的滞后），同一份 DOM 在渲过别的帧之后栅格化结果会不同——原片里移动中的字幕字形差到 92/255、几千个像素，连单标签顺序渲染都和冷渲不同。加了这一步之后，那种大块的差异就没有了（剩下的极小差异见下文「能逐位相同的范围」）。代价：单标签每帧多约 35 ms，并行时几乎看不出。
- `paper2d.frame(n)`：`flushSync` 提交 React（布局 effect 同步执行）→ 循环：等所有 delayRender 句柄放掉 → `document.fonts.ready` → 文档里所有 `<img>` 加载并 `decode()` → 两次 `requestAnimationFrame`（布局 effect 里画的 Canvas / WebGL 已呈现）→ 期间若有新句柄出现就整轮重来。之后驱动用 CDP `Page.captureScreenshot` 截合成后的整页，不读单个 canvas。
- Chrome：`--headless=new`、`--force-color-profile=srgb`（截图按 sRGB）、`--hide-scrollbars`、一次性 `--user-data-dir`。

## 确定性与浮点

- **每一帧只由帧号决定**：场景里不用 `Date`、`performance.now()`、`Math.random()`、`random(null)`，不在组件外存跨帧状态，不读上一帧的画面。随机量用 `random('名字' + 帧号)` 或 `components` 里的 `hash` / `noise1` / `fbm1` / `rnd`，全部是整数运算，跨次运行一致。
- **帧号可以是小数**（`Retimed`、`Freeze` 之下），用它做种子时注意字符串形式：`String(217.60000000000002)` 与 `String(217.6)` 是两个种子。
- **字体**：用 `loadFonts`（FontFace + delayRender）登记，第一帧就用对字体；字体文件放片的 `public/`。没加载成功会在控制台报错并照常出帧（会落到回退字体），所以命令行里看到 `font load failed` 要处理。
- **SVG 滤镜**（`feTurbulence` 等）带固定 `seed`，结果确定；滤镜 id 用 `useId()`，多实例不冲突。
- **Canvas**：`putImageData` 逐字节拷贝；颗粒（`Grain`）是不透明的灰度图块，不经过预乘 alpha 的舍入。
- **截帧格式**：默认 PNG（无损，只做一次 4:2:0 下采样）；`--capture jpeg` 快，留给预览。
- **能逐位相同的范围**（在 macOS 上用示例片实测；除特别注明外，后端都是 metal）：
  - Chrome 的栅格化在一个标签页里多少会带着这个标签之前渲过的帧的痕迹。合成层重建之后，大块的差异没有了，但在 SVG 滤镜笔触正在画的帧（`Stroke` 的 boil、红线）上，仍有约 1–2% 的帧会差 3–23 个像素，每个最多 4/255（PSNR > 99 dB）；加 `--disable-gpu-rasterization` 也去不掉。原引擎也有同样现象。
  - 所以 `render.ts` 按轮转把帧固定分给各标签（第 k 个标签渲 k、k+N、k+2N…），这样每个标签的历史只取决于 `--from/--to/--tabs`。**同样的参数整片重渲，通常逐帧相同；偶有 1–2 帧差几个像素**：本机连续渲三次 700 帧全同，发布前审计同参数渲 4 次，有 1 次第 113、127 帧（铅笔笔触正在画）截图哈希不同，解码后第 113 帧 PSNR 62.87 dB。原引擎用的是共享队列，每个标签渲哪些帧随调度变，两次整片会有 7–10 帧不一样。
  - 单标签顺序渲（`--tabs 1`），以及 `still.ts` 的静帧（跨次运行，正序或倒序），也都逐位相同。
  - `--gl swangle`（SwiftShader）：静帧跨次同样逐位相同；但整片即使轮转分帧，两次之间仍会有个别标签从某一帧起一直和上次不同（0–160 帧两次，16/161 帧不同），每个最多 7/255，PSNR 80–91 dB，肉眼看不出。用 swangle 做整片回归不要比哈希。
  - 换标签数，或者拿 still 和整片比，上面那 1–2% 的帧可能差几个像素。换 GL 后端、换机器，差得更多（抗锯齿、栅格化方式不同）。
  - 所以验收整片时先比 `--hashes`，不同的帧再用 `still.ts` 渲出来、`pixdiff` 看 PSNR（几个像素、PSNR 60 dB 以上就是这种差异）；跨标签数、跨后端、跨机器直接比 PSNR。

## 输出

H.264 High、yuv420p、BT.709 有限范围，primaries/transfer/matrix/range 四个标签都打上（ffmpeg 9 从帧上取标签，所以滤镜链末尾用 `setparams`；只给命令行 `-color_trc` 时 transfer/primaries 会是 unknown）；合成的 fps；无音轨（`--audio` / `mux.ts` 另出带声版）。颜色核过（示例片第 650 帧，平铺纸色）：#F2EBDD(242,235,221) → Y/Cb/Cr 218/121/132，按 BT.709 有限范围解回（`scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=bicubic+accurate_rnd+full_chroma_int`）242,235,220。实测在 4:2:0 上先 crop 再按默认参数转 RGB 解出 240,232,218（差 2–3 个色阶），量颜色时照上面的写法。

## 组件库 `paper2d/components`

纸面场景的通用画法，从原片里抽出、去掉了片子与产品的硬编码（颜色默认取 `palette`，都可以用参数换掉）。

| 组件 / 函数 | 画什么 |
| --- | --- |
| `beatGrid(timeline)` | 拍网格：`at('3.1+0.5')`、`bar(n, beat)`、`pos(frame)`（帧号 → 「小节.拍+n f」）、`range('1.3-2.3')`、`shot(id)` / `section(id)` / `cue(id)`（含 start / end 帧）、`shotAt(frame)`；检查 `framesPerBeat = fps·60/bpm` |
| `Retimed gridFps` / `OnTwos n` | 按别的网格写的画面按合成 fps 取样（原片：24 格网格、30 fps 成片）/ 一拍二 |
| `HandPaper` | 纸：平铺纸色随 `grain` 显影成有纹理的纸（大块斑驳、单向纤维、亮斑、杂点，全是带种子的 SVG 噪声）；铅笔画的方格 / 斜格 / 点阵（`drawn` 画上去、`mix` 换样式） |
| `PaperCard`、`shadow()` | 落在纸上的卡片与大而软的投影（`lift` 控制抬起） |
| `Stroke` 及 `HandCircle` `HandUnderline` `HandArrow` `HandTick` `HandSlash` | 一笔手绘：宽度随笔速与压力起伏、起笔略重、收笔出锋、干笔飞白；材质 `ink` 墨 / `brush` 湿笔 / `pencil` 铅笔 / `gold` 金线 / `cord` 毡笔红线；`progress` 按时间线性给，人手的速度曲线在里面；`boil` 线条轻微抖动 |
| `alive`、`Threaded`、`sag`、`outline`、`Burst`、`blendPaths`、`resampleN`、`slice` | 活的线：沿线传播的细波与「拨一下」的回颤、穿过卡片的前后两层、下垂、绕卡片成描边、落定时弹出的短线（一拍二）、一条线被拉成另一条线 |
| `Kinetic`、`kineticLayout`、`Caption`、`Typed`、`Mono` | 字：逐字说出来（一字一字带弧线落定、逗号处换气、整块缓慢漂移、按来的方向扫走、关键词可被线划出）；`kineticLayout` 预先给出每个字的位置与时间，别的东西（一根线）可以落到词下 |
| `Grain`、`Vignette` | 颗粒（页面内按种子生成的灰度图块，按帧号选块与偏移）与暗角 |
| `lensPath`、`clampLens`、`lensTransform`、`toScreenLens` | 平面镜头：穿过关键帧而不停下的样条路径，不出纸边 |
| `Plate` | 一张 public/ 里的图片铺满画面，带缓慢的推拉 |
| `loadFonts`、`fonts`、`palette` | 字体登记、字体栈、中性默认色板 |
| `clamp lerp ease ramp fadeWindow rnd wobble pulse onTwos` | 动画小工具（`fadeWindow` 在原片里叫 `window`，改名以免遮住全局） |

画法上的经验（来自原片的拉片与审片，概括）：每小节一个主动作，放在强拍；主动作快起步（1–4 帧到峰值速度）、长落定（10–14 帧，大位移 25–40 帧），入场不用 ease-in，落定不回弹；主动作之后留阅读段，只允许缓推与呼吸类微动；同屏同时运动的主元素不超过两个；字幕的可见时长至少「3.2 × 字数 + 16」格（24 格网格）。

## 保真（在原片上做过的验收）

下面是阡陌宣传片当年的记录，用的是那时的垫片。agentreel 里的垫片之后按独立结构重写了 `interpolate`、`spring`、`measureSpring`、`random`（见 `THIRD_PARTY_NOTICES.md`）：`spring` 的欠阻尼与临界阻尼和旧版逐帧对比差在 1e-15 量级、`interpolate` 的 extend / clamp / identity 结果逐位相同，但 `random` 换了哈希，按名字取的随机量（例如 `Grain` 的偏移）整套不同，所以这份记录不能再拿来证明与 Remotion 母版逐帧一致。

渲染器在原片上对 Remotion 当年渲的母版（1920×1080、30 fps、H.264 crf 16、yuvj420p 全范围、BT.601）做过逐帧保真：全片每 7 帧一张（586 帧，覆盖 0.8n 的全部小数相位），亮度 PSNR 最低 41.97 dB、均值 47.28 dB，RGB 最低 38.43 dB；把我们的帧走一遍母版当年的编码管线再比，差异在 0.35 dB 内，剩下的就是编码损失；同一张对母版 n±1 帧只有 22–30 dB，对 n 帧 44–46 dB（时间对齐）。期间查出并修掉的三处：`spring` 小数帧不在精确曲线上（现在的解析解没有这个问题）、页面缺 `border-box`、`Freeze` 的浮点算术。

速度（Apple M5 Pro、1080p、PNG 截帧）：启动约 0.5 s（打包 30 ms、Chrome 250–400 ms、页面 100–200 ms）；单标签每帧 0.12–0.2 s；6 标签并行约 14–30 fps（视场景而定，示例片约 15 fps）；x264 写出端不是瓶颈。同一台机器上用 `--gl swangle`（SwiftShader，纯 CPU）时，示例片 6 标签约 1.6–1.7 fps，和 metal 的画面看不出差别（PSNR 50–57 dB）。

## 已知限制

- 只测过 macOS（Apple 芯片）+ Chrome 155 + ffmpeg 9（GL 后端测过 metal 和 swangle）；Windows / Linux 的路径查找与 GL 参数按文档写好，但没有在本仓库里实测。
- 不出声：`Audio` 是空组件，配乐由 `music/` 的库另行渲染，用 `mux.ts` 混进来。
- 不支持 `Video` / `OffthreadVideo`（需要逐帧解码视频素材的场景要先把素材抽成图片序列，用 `Img`）。
- 截帧走 Chrome 合成器：极端的 CSS 特效（`backdrop-filter`、大半径模糊叠很多层）会让单帧变慢，但不影响确定性。
