# 纸上一线 · ONE LINE ON PAPER（示例片：paper2d + sampler）

一部 23.3 秒的小片，用来演示 `engines/paper2d` 的做法：React 场景按帧号出画、纸面质感、铅笔框成纸、红线穿引与下划线、逐字出现的字幕、确定性颗粒，以及画面与配乐对着同一张拍子表。

- 规格：1920×1080、30 fps、700 帧（6 小节画面 + 1 小节余音）。72 BPM、4/4，一拍 25 帧、一小节 100 帧。
- 时间线：`timeline.json`（契约，画面与配乐都按它写；写法见 [`docs/TIMELINE.md`](../../docs/TIMELINE.md)）。
- 这套做法的来历见 [`docs/cases/qianmo-promo.md`](../../docs/cases/qianmo-promo.md)；下面各条自查在验收里管什么，见 [`docs/QA.md`](../../docs/QA.md) 的「工具对照」。
- 画面：`src/`（本 README 下文）；配乐：`score/`（`music/sampler`），输出 `out/music/score.wav`。

## 镜头

位置写「小节.拍」，括号里是帧号。`src/timing.ts` 里的每个时刻都从 `timeline.json` 换算，不写死帧号。

| 镜头 | 时间 | 画面 |
| --- | --- | --- |
| P01 成纸 | 1.1–3.1（0–200） | 纯纸色，纸纹与纤维 1.1–1.2+20f 随帧号显影，铅笔方格轻轻画上；1.3 / 1.4 / 2.1 / 2.2（50 / 75 / 100 / 125）铅笔一拍一笔画出卡片的上、右、下、左边；2.3（150）「成纸」：框里落下一张有纸纹的卡片，轻抬、落定、长出投影，两个对角弹出几笔铅笔短线（一拍二），铅笔线随后淡去 |
| P02 一线 | 3.1–5.1（200–400） | 3.1 红线从左下甩进来（线头快，线身带着衰减的摆动），绕卡片一圈成了描边，在左下角折回卡内，斜着升上来接成「帧号」的下划线，4.1（300）到位、轻颤；第一句 3.1+10f 起逐字出现；4.3（350）第二句逐字出现，下划线先落到第二行下面、再滑到「拍子表」下（线尾先走，后面的线被拖过去），4.4+0.5（387.5）字落定后贴上来、轻颤；下划线和字形最低点之间留 6–8 px 的纸色，滑动时也不碰字 |
| P03 片尾 | 5.1–7.1（400–600） | 5.1 卡片连同字以横线为轴收起，绕在卡上的线跟着压扁、再被拉直成一条横线，5.2+20f 绷直并回颤；方格换成点阵；5.3 起「agentreel」逐字落下，5.3+18f 打出仓库地址；6.4（575）起淡出 |
| 余音 | 7.1–8.1（600–700） | 7.1 切到纸色（平铺，无纹理、无颗粒），停到配乐的余音结束 |

配乐的提示点（`cues`）与上表对齐：铅笔声落在那四拍上，成纸在 2.3，红线进场 3.1，「帧号」4.1，第二句 4.3–5.1（下划线 4.4+0.5 贴到「拍子表」下），线绷直 5.1，7.1 收住。

## 渲染

先装引擎（只要一次）：

```sh
cd engines/paper2d
bun install
```

以下命令都在 `engines/paper2d` 下执行：

```sh
# 类型检查
bun run typecheck:film ../../films/demo-paper

# 静帧（帧号或小节.拍）→ ../../films/demo-paper/out/stills/DemoPaper/
bun still.ts --film ../../films/demo-paper --frames 1.4+10f,2.3+10f,4.1,4.4+15f,5.4,6.2

# 整片（无声）→ ../../films/demo-paper/out/DemoPaper.mp4，顺手存每帧截图的哈希
# （同一台机器、同样的参数再渲一次，哈希通常逐帧相同；偶有 1–2 帧差几个像素。先比哈希，不同的帧再渲静帧用
#  pixdiff 看 PSNR；见 engines/paper2d/README.md「确定性与浮点」）
bun render.ts --film ../../films/demo-paper --hashes ../../films/demo-paper/out/DemoPaper.hashes.tsv

# 配乐（见 score/ 与 music/sampler/README.md）渲好之后混出带声版 → out/DemoPaper_audio.mp4
bun mux.ts --video ../../films/demo-paper/out/DemoPaper.mp4 --audio ../../films/demo-paper/out/music/score.wav
# 或者渲染时一步做完
bun render.ts --film ../../films/demo-paper --audio ../../films/demo-paper/out/music/score.wav

# 自查：相邻帧跳变。排在前面的应当只有这几处有意的变化：7.1 切到纸色（mad 最大）、5.1 卡片收起的连续动作
# （5.1+几帧，一串相邻的步，spike 约 1，不是跳变）、2.3 成纸的落定（2.3+1f）。名次会随画面细节变；冒出这几处以外的
# 帧就要查
bun scan.ts ../../films/demo-paper/out/DemoPaper.mp4 --film ../../films/demo-paper
```

找不到 Chrome 或 ffmpeg 时设 `CHROME_PATH` / `FFMPEG` / `FFPROBE`（见 `engines/paper2d/README.md`）。

## 文件

| 文件 | 内容 |
| --- | --- |
| `timeline.json` | 拍网格、镜头、配乐提示点、响度目标（契约，不改） |
| `src/index.ts` | `registerRoot` |
| `src/Root.tsx` | 唯一的组合 `DemoPaper`（尺寸、帧率、总帧数取自时间线）；登记字体 |
| `src/timing.ts` | 拍点（`AT`）、卡片与字幕的版面、收起的节奏 |
| `src/copy.ts` | 画面上的全部文字（字体子集按它生成） |
| `src/Film.tsx` | 分层：镜头里的纸面世界 + 画面空间的暗角与颗粒；7.1 之后的纯纸色 |
| `src/Card.tsx` | 铅笔四笔、成纸、收起 |
| `src/Thread.tsx` | 红线的路径与生命：甩进、绕卡、下划线、滑动、压扁拉直、回颤 |
| `src/EndTitle.tsx` | 片尾的名字与仓库地址 |
| `public/fonts/` | 三个 OFL 字体的子集与 `OFL.txt` |
| `score/` | 配乐（另一个单元维护） |

## 字体

画面只用到二十几个汉字和少量拉丁字母，所以 `public/fonts/` 里放的是子集（每个十几 KB），不是整套字体：

| 文件 | 字体 | 用在 |
| --- | --- | --- |
| `NotoSerifSC-500-subset.woff2` | Noto Serif SC Medium（Adobe / Google，OFL-1.1） | 卡片上的两句字 |
| `EBGaramond-500-subset.woff2` | EB Garamond Medium（The EB Garamond Project Authors，OFL-1.1） | 片尾的「agentreel」 |
| `JetBrainsMono-400-subset.woff2` | JetBrains Mono Regular（The JetBrains Mono Project Authors，OFL-1.1） | 仓库地址 |

版权声明与许可全文在 `public/fonts/OFL.txt`。改了 `src/copy.ts` 里的字要重做子集（源字体从各自的官方仓库或 Google Fonts 下载）：

```sh
cd engines/paper2d
bun tools/subset-font.ts --font <NotoSerifSC-Medium.otf> --out ../../films/demo-paper/public/fonts/NotoSerifSC-500-subset.woff2 --text-file ../../films/demo-paper/src/copy.ts
bun tools/subset-font.ts --font <EBGaramond-Medium.ttf> --out ../../films/demo-paper/public/fonts/EBGaramond-500-subset.woff2 --text-file ../../films/demo-paper/src/copy.ts
bun tools/subset-font.ts --font <JetBrainsMono-Regular.ttf> --out ../../films/demo-paper/public/fonts/JetBrainsMono-400-subset.woff2 --text-file ../../films/demo-paper/src/copy.ts
```

## 署名

- 画面：cornna 与 agentreel contributors，MIT（仓库根目录 `LICENSE`）。
- 字体：见上表与 `public/fonts/OFL.txt`。
- 配乐：采样来源与公开时必须带的署名见 `score/` 与 `music/sampler/CREDITS.md`。
