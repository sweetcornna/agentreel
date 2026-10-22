# 《纸上一线》的配乐

`score.py` 照 `../timeline.json` 的拍网格写（72 BPM、4/4、一拍 25 帧），用 `music/sampler` 渲染：
D 大调，23.333 秒（700 帧 = 6 小节画面 + 1 小节余音），整合响度 -16 LUFS，真峰值 -2.1 dBTP。

## 渲染

先按 `music/sampler/README.md` 第 2 节装好环境（venv 在 `music/sampler/.venv`），然后在仓库根目录：

```sh
music/sampler/.venv/bin/python music/sampler/fetch_samples.py --files films/demo-paper/score/samples.txt
music/sampler/.venv/bin/python films/demo-paper/score/score.py --check
```

先 `source music/sampler/.venv/bin/activate` 的话可以直接写 `python`。第一条取 40 个文件、下载约 155 MB，实测 1 分多钟到
约 10 分钟（主要看 GitHub raw 的速度），中断了重跑即可；要频繁重试时设 `GITHUB_TOKEN`。三个 Freesound 文件直接从记录的
预览地址下载、核对 SHA-256，许可以记录为准（会打印一行「未在线复核许可」）；设了 `FREESOUND_API_KEY` 会经官方 API 在线复核。

输出在 `../out/music/`（git 忽略）：

| 文件 | 内容 |
| --- | --- |
| `score.wav` | 48 kHz / 24 bit 立体声，1 120 000 个样本 |
| `stems/*.wav` | 七个分轨（平衡后、加了空间、母带之前）：pencil paper piano harp chime strings pad |
| `events.json` | 拍点、同步点、给画面读的事件、每个音、增益曲线，`measured` 里是自检结果与用到的采样清单（字段见 [`docs/TIMELINE.md`](../../../docs/TIMELINE.md) 第 10 节） |
| `manifest.json` | 产物与源采样的 SHA-256 |

有了带声版画面：`engines/paper2d` 下 `bun mux.ts --video ../../films/demo-paper/out/DemoPaper.mp4 --audio ../../films/demo-paper/out/music/score.wav`。

## 声音对着画面

| 位置（帧） | 画面 | 声音 | 分轨 | 实测起音偏差 |
| --- | --- | --- | --- | --- |
| 1.1 起（0） | 纸纹显影 | pad 从 D 的空五度慢慢起来 | pad | — |
| 1.3 1.4 2.1 2.2（50 75 100 125） | 铅笔一拍一笔画出卡片的上、右、下、左边（一笔 19 帧） | 每笔一段 19 帧长的连续笔声，声像跟着笔走（上边左→右、右边在右、下边右→左、左边在左） | pencil | +2.7 … +4.0 ms |
| 2.3（150） | 成纸 | 速写本翻落的一声 + 钢琴 Dadd9 自下而上分解（每音 22 ms） | paper、piano | +0.2 ms、+5.9 ms |
| 2.3+2f、+4f（152、154） | 两个对角弹出铅笔短线 | 各一下很轻的铅笔声，左上、右下 | pencil | — |
| 3.1（200） | 红线从左下甩进来 | 竖琴 G3→D5 上行（70 ms 一个音，声像从左到中） | harp | +4.3 ms |
| 3.2 3.3 3.4 的后半拍 | 红线绕卡片 | 竖琴 B4、A4、F#4 各一个 | harp | — |
| 4.1（300） | 红线停在「帧号」下 | 手铃 F#5（Bm7 上） | chime | 0.0 ms |
| 4.3–5.1（350–400） | 第二句逐字出现，下划线滑到「拍子表」 | 钢琴 F#4 A4 B4 C#5 八分音符（A 上） | piano | 0.0 ms |
| 5.1（400） | 卡片收起，线被拉直 | 宽和弦：弦乐 D2 A2 F#3 E4 A4 + 钢琴 D2 D3 A3 D5 | strings、piano | 钢琴 +1.4 ms；弦乐弓起得慢，+116 ms |
| 5.2+20f（445） | 线绷直、回颤 | 竖琴 D5 + A5，D3 托底 | harp | +2.3 ms |
| 5.3（450）、5.3+18f（468） | 「agentreel」落下、仓库地址打出 | 小提琴 F#5 进来、钢琴 E5；钢琴 F#5 | strings、piano | — |
| 6.4–7.1（575–600） | 淡出 | 弦乐与 pad 渐弱 | — | — |
| 7.1（600） | 切到纸色 | 所有干声 0.25 s 收住，只剩混响尾巴；8.1 前淡到 0 | — | 切点后 1.8 s 降到 -60 dBFS |

偏差是在干声分轨上量的「听到的起音」（2 ms 包络第一次到 -12 dB 以内）减去目标时刻；画面一帧是 33 ms。
各段整合响度：成纸 -18.2、一线 -15.9、片尾 -14.9、余音 -29.0 LUFS。

## 乐器与采样

| 分轨 | 乐器 | 来源 | 许可 |
| --- | --- | --- | --- |
| piano | Steinway B 三角钢琴（`folk.render('grand_piano')`） | VCSL | CC0 |
| harp | 民谣竖琴（`folk.render('folk_harp')`） | VCSL | CC0 |
| chime | 手铃（`nature.sfx('hand_chime')`） | VCSL | CC0 |
| strings | 弦乐四个声部（`strings.render_strings`） | VSCO 2 CE | CC0 |
| pencil | 铅笔涂写、铅笔书写（`found.stroke`、`found.hit`） | Freesound：waldram、rivernile7 | CC0 |
| paper | 速写本翻页（`found.hit`） | Freesound：OwlStorm | CC0 |
| pad | 合成 pad（`pads.pad`） | 无（合成） | — |
| — | 混响 | 合成的冲激响应 | — |

用到的 40 个采样文件列在 `samples.txt`。**全部是 CC0 或合成的，公开这部片的配乐不要求署名。** 如愿致谢：
"Versilian Community Sample Library" and "VSCO 2: Community Edition" by Versilian Studios (CC0); sounds by waldram,
rivernile7 and OwlStorm on Freesound (CC0)。改了乐谱、用上 Salamander 钢琴或 VocalSet 人声时，就要按
`music/sampler/CREDITS.md` 带署名。

## 确定性

同一份采样、同一套依赖（`music/sampler/requirements.lock`）下，连渲两次 `score.wav`、七个分轨和 `events.json` 的 SHA-256
都相同；只取 `samples.txt` 里 40 个文件渲出来的，与用完整的 VCSL / VSCO / Freesound 采样渲出来的逐样本相同。
macOS arm64、Python 3.12 上的 `score.wav`：

    66464db13c0e9ade98b66c92089e971fc0d3d6107315d9ab1cdac114f5c940b1

## 改乐谱

- 时刻都从时间线来：`G.pos('2.3+4f')`、`G.cue('word1')`。画面上的两个量（一笔 19 帧、卡片左右边的横坐标）抄自
  `../src/timing.ts`，画面改了这里要跟着改。
- 分轨之间的平衡在 `LEVEL`（每个分轨的整合响度），空间在 `SPACE`。改完跑 `--check` 看响度、真峰值和同步表。
- 换了音或乐器，`samples.txt` 要重新生成（要先有完整的采样）：

  ```sh
  AGENTREEL_SAMPLES_LOG=samples-new.txt music/sampler/.venv/bin/python films/demo-paper/score/score.py --no-stems
  ```

  再把这个文件排序后替换 `samples.txt` 里的清单（保留开头的注释）。
