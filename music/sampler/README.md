# sampler · 采样乐器 + 经典 DSP 的配乐库

用 Python 写配乐：真实乐器的录音（CC0 / 公有领域 / CC BY）做采样器，其余用经典信号处理合成，混音、母带、自检都在库里。
`films/demo-paper/score/score.py` 是一部完整的例子（23.3 秒，画面与配乐对着同一张拍子表）。

移植自阡陌宣传片的配乐工具库（制作过程见 [`docs/cases/qianmo-promo.md`](../../docs/cases/qianmo-promo.md)）：乐器模块原样搬过来，只改了采样路径、去掉了
那部片专用的东西，另加了拍网格、分轨与自检三个写谱用的模块、`fetch_samples.py` 和一套 CC0 的三角钢琴。

## 1. 设计

- **采样 + 经典 DSP，没有神经网络或生成模型。** 钢琴、弦乐、竖琴、手铃、人声、铅笔、纸……都是真录音，按实测音高变速移调、
  按力度交叉淡化；合唱、pad、合成乐队、混响冲激响应、各种程序化音效是公式算出来的。我们听不到，所以每件乐器都有客观自检
  （`selfcheck.py`），乐谱渲完也要量（`sampler.check`）。
- **确定。** 所有随机性来自 `seed`（`numpy.random.default_rng` / `random.Random`）：同样的参数、同样的采样文件、
  同样的依赖版本（`requirements.lock`），输出逐样本相同。`films/demo-paper` 连渲两次的 SHA-256 相同，用完整采样和只取
  它用到的 40 个文件渲出来的也相同（见下文「只取用到的文件」）。
- **统一的数组约定。** 48 kHz（`sampler.SR`），float32，形状 `(n, 2)` 的立体声，时间单位秒。乐器输出不归一化，
  首尾样本为 0；响度只在分轨平衡（`bus.to_lufs`）和母带（`mix.master`）两处定。
- **采样不进仓库。** 录音体积大、许可要逐条核对，所以仓库里只有代码、`CREDITS.md` 和下载脚本。`fetch_samples.py`
  从原始出处（GitHub 官方仓库、Wikimedia Commons、Freesound、Zenodo）下载，下载前核对许可，不合格就拒收。
- **时间线是唯一的时间来源。** 乐谱从片目录的 `timeline.json` 读速度、拍号、帧率、提示点（`sampler.grid`），
  导出 `events.json`。顶层字段与 `music/filmsynth` 相同；细节有差异：`markers` 多 `frame`，区间另带 `end` / `end_frame`
  （示例片的乐谱还写了 `cue_kind`、`note`），`curves` 是 `[{t, frame, v}]` 折点（filmsynth 是 `{hop, t0, values}`），`notes` 没有 `beat`、`hz`。
  逐字段的对照见 [`docs/TIMELINE.md`](../../docs/TIMELINE.md) 第 10 节；各项自检在验收里的位置见 [`docs/QA.md`](../../docs/QA.md) 的「工具对照」。

## 2. 安装

要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。在仓库根目录：

```sh
uv venv --python 3.12 music/sampler/.venv
uv pip install --python music/sampler/.venv -r music/sampler/requirements.lock
uv pip install --python music/sampler/.venv --no-deps -e music/sampler
music/sampler/.venv/bin/python -c "import sampler; print(sampler.__version__, sampler.SAMPLES)"
```

- `requirements.lock` 是开发和验证时的确切版本；`pyproject.toml` 只写了下限。换了 numpy / scipy 的版本，数值可能在最后一位
  上不同（听不出来，但 SHA-256 会变）。
- pyworld（人声用的 WORLD 声码器）在没有现成 wheel 的平台上会从源码编译，需要 C++ 编译器。它导入时要 `pkg_resources`，
  所以锁了 setuptools<81。
- ffmpeg 可选：解码 Commons 的野外录音（`fetch_samples.py --only field`）和用 ebur128 复核响度时要。找法：`$FFMPEG`，
  再 PATH，再几个常见安装位置。
- 不装包也能用：脚本把 `music/sampler` 放进 `sys.path` 就能 `import sampler`（`films/demo-paper/score/score.py` 就是这样兜底的）。

## 3. 取采样

下面的命令都在仓库根目录执行，用第 2 节建好的 venv 的解释器（先 `source music/sampler/.venv/bin/activate` 的话，
可以把 `music/sampler/.venv/bin/python` 简写成 `python`）：

```sh
music/sampler/.venv/bin/python music/sampler/fetch_samples.py --list                                       # 包与套
music/sampler/.venv/bin/python music/sampler/fetch_samples.py --only vcsl:grand_piano,vcsl:folk_harp       # 整套
music/sampler/.venv/bin/python music/sampler/fetch_samples.py --only piano --range C3-C6                   # 一段音域
music/sampler/.venv/bin/python music/sampler/fetch_samples.py --files films/demo-paper/score/samples.txt   # 精确清单
music/sampler/.venv/bin/python music/sampler/fetch_samples.py --only vocalset --dry-run                    # 只核对许可
```

**要多久**：按 `films/demo-paper/score/samples.txt` 取 40 个文件（155 MB），实测 1 分多钟到约 10 分钟，
主要看 raw.githubusercontent.com 当时的速度。中断了重跑即可：已经取到、
哈希对得上的文件不会重下。要频繁重试时设 `GITHUB_TOKEN`（GitHub API 未登录每小时只有 60 次，每个 GitHub 包每次运行用 1–2 次）。

采样根是 `$AGENTREEL_SAMPLES`，没设就是 `music/sampler/samples/`（git 忽略）；`--dest` 临时指定。

| 包 | 内容 | 许可 | 下载 / 落盘（全套） |
| --- | --- | --- | --- |
| `piano` | Salamander Grand Piano V3（SFZ，16 个力度层） | CC BY 3.0，**须署名** | 约 750 MB |
| `vcsl` | VCSL：22 套（民谣乐器、Steinway B 三角钢琴、手铃、管钟、锣、木鱼、手鼓类） | CC0 | 1.2 GB / 约 520 MB（转 FLAC） |
| `vsco2` | VSCO 2 CE：弦乐四个声部 + 独奏小提琴，木管、圆号、定音鼓、悬挂钹 | CC0 | 约 290 MB（FLAC） |
| `field` | Commons 野外录音 16 条（溪流、林子、鸟、蟋蟀、鸡、狗） | CC0 / PD / 两条 CC BY 3.0 | 约 160 MB，另存原始下载 |
| `found` | Freesound：铅笔、纸、键盘、响指、敲击、水滴、吱呀声 16 条（不设 `FREESOUND_API_KEY` 时只能取记录了预览地址的 3 条） | CC0 | 8 MB |
| `objects` | Freesound：瓷碗、瓷盘、瓷杯、翅膀 6 条（不设 key 时只能取 1 条） | CC0 | 1 MB |
| `vocalset` | VocalSet 1.1：20 位歌手的长音、音阶、琶音、片段 | CC BY 4.0，**须署名** | 1.5 GB（zip 里按需取成员）/ 约 570 MB |

**许可核对**（每次下载前都做，证据写进 `<采样根>/fetch_manifest.json`）：

- GitHub：读固定 commit 的 `LICENSE` 全文判断（CC0 1.0 / CC BY 3.0 / 4.0），GitHub API 的 spdx_id 作旁证；每个文件核对
  git blob SHA-1 与大小。commit 写死在 `fetch_samples.py` 里，仓库以后改了许可也不影响已核对的版本。
- Commons：用 API 读文件页的 LicenseShortName，必须是 CC0 / Public domain / CC BY x.y，而且与 `fetch_samples.py` 里记录的
  一致；带 SA、NC、ND 的一律不收。
- Freesound：**不抓网页。** 设了 `FREESOUND_API_KEY`（在 https://freesound.org/apiv2/apply 申请）时，用官方 APIv2
  `/apiv2/sounds/<编号>/` 读 `license` 字段，必须是 CC0 且与记录一致（NC / SA / sampling+ 一律拒收），用户名也要对得上，
  预览地址取 API 给的 `preview-hq-ogg`。没设时直接下载 `fetch_samples.py` 里记录的 HQ 预览地址，许可以记录为准
  （CC0，录入时逐条在页面上核对过，见 `CREDITS.md`），并打印「未在线复核许可，设 FREESOUND_API_KEY 可在线复核」；
  没有记录预览地址的条目（`--list` 里带 `*`）跳过，最后以退出码 1 结束。两种方式都核对记录的 SHA-256
  （原始文件要登录才能下，所以取的是预览）。
- Zenodo：读记录 1442513 的 license 字段，必须是 `cc-by-4.0`；zip 成员由 zipfile 核对 CRC-32。

不合格时打印 `REFUSED: ...`，**退出码 2**，不下载任何录音；有文件没取到时退出码 1。网络失败会重试。

**下载后自动导入**：VCSL 的民谣乐器与钢琴转 FLAC，实测音高、起音、电平写 `vcsl/index_folk.json`；VSCO 转 FLAC，实测
音高写 `vsco2/index.json`；Commons 录音解码、截取、重采样到 48 kHz，参数写 `field/manifest.json`（原始下载留在
`field/original/`，可删）；取了鸟鸣录音就重建 `field/bird_calls.json`；VocalSet 建音符单元索引。

**只取用到的文件。** 设 `AGENTREEL_SAMPLES_LOG=<文件>` 跑一遍乐谱，库会把打开过的每个采样文件（相对采样根）记进去；
把这个清单交给 `--files`，取到的就是这部乐谱要的全部录音。`films/demo-paper/score/samples.txt` 就是这样来的：40 个文件、
下载 155 MB，渲出来与完整采样逐样本相同。能做到这一点，是因为 VCSL 民谣乐器的电平校正用的是**整套**的中位数
（`sampler/data/vcsl_ref_levels.json`，导入时写进索引），不随手头有哪些文件变；其余乐器按文件名或实测音高选最近的采样，
选中的文件都在清单里。例外：一个音在多个轮换（rr）采样之间轮流、而乐谱只用到其中一部分时，子集上的轮换顺序可能不同——
渲完和完整采样比一次 SHA-256 就知道。VocalSet 不能按文件取（音符单元索引要整套建），用 `--only vocalset`。

## 4. 乐器参考

各模块的文档字符串是完整说明（`music/sampler/.venv/bin/python -c "import sampler.folk as m; help(m)"`）。下面是入口与要的采样。

| 模块 | 入口 | 是什么 | 采样 |
| --- | --- | --- | --- |
| `sampler.piano` | `render_piano(notes, pedal=(), felt=False, duration=None, *, tuning, una_corda, sympathetic, ...)` | Salamander 钢琴：16 力度层交叉、延音踏板与制音、弦共鸣、弱音（毛毡）与 una corda | `piano` |
| `sampler.folk` | `render(set_id, notes, *, ring_s, damp_s, pan, ...)`、`hit / place_hit(set_id, kind, ...)` | VCSL 有音高的套（`grand_piano`、`folk_harp`、`strumstick`、`dan_tranh`、`marimba`、`balafon`）与手鼓类（`cajon`、`conga`、`shaker_*`……），按实测音高变速，打击乐按听到的起音对齐 | `vcsl` |
| `sampler.strings` | `render_strings(notes, articulation='sus', *, legato, attack_s, release_s, ...)`、`pluck(...)` | VSCO 弦乐：长音（连弓交叉淡化、延音段延长）、拨弦、跳弓、泛音；木管、圆号同一接口（`sec`） | `vsco2` |
| `sampler.voice` | `render_voice(notes, *, gender, tech, vowel, ...)`、`tintinnabuli(...)` | VocalSet 真人声切成音符单元，WORLD 重合成到目标音高，带真实的吸气 | `vocalset` |
| `sampler.choir` | `render_choir(notes, voices=6, vowel='a', ...)`、`choir_pad(...)` | 共振峰合成的合唱 / 独唱（LF 声门 + 时变共振峰），元音渐变 | 无 |
| `sampler.pads` | `pad(notes, *, voices, brightness, attack, release, ...)`、`sub(note, dur)` | 加法合成的弦乐感铺底与超低频 | 无 |
| `sampler.nature` | `ambience(kind, duration, seed)`、`sfx(kind, seed, **kw)`、`distant(x, amount)` | 环境声（录音或合成，`KINDS_AMBIENCE`）、单个音效（`KINDS_SFX`：手铃、管钟、锣、木鱼、纸、水滴、鸡、狗……）、远处化 | `field`、`vcsl`；合成的不要 |
| `sampler.found` | `place(track, name, at, vel, seed, pick=...)`、`hit(...)`、`stroke(name, dur, vel, seed)`、`bed(name, dur)` | 画面里那些东西的真实声音：切成单个事件（按亮暗、长短、响度挑）或取一段连续发声当作一笔 | `found` |
| `sampler.porcelain` | `bowl(note, vel, ...)`、`render_bowls(bowls, ...)` | 瓷碗：录音变速或按实测模态合成，可加水 | `objects` |
| `sampler.birds` | `bird_call(...)`、`transcribe_call(...)`、`render_birds(...)`、`echo_notes(...)` | 鸟鸣单句目录、去噪调音、把鸟鸣转写成钢琴音符 | `field` |
| `sampler.fx` | `reverse_swell`、`granular`、`shimmer`、`tape`、`impact`、`pitch_shift` | 倒放混响涌起、颗粒云、八度反馈混响、磁带质感、幕间落点、相位声码器移调 | `impact` 用 `piano` 与 `vcsl` |
| `sampler.band`、`sampler.synth` | `marimba`、`vibes`、`bass`、`kick`……；`keys`、`pluck`、`riser`、`sidechain`…… | 纯合成的小乐队与电子音色 | 无 |
| `sampler.mix` | `place`、`fade`、`eq`、`compress`、`limit`、`reverb(x, 'room'/'hall'/'valley'/'bloom', mix)`、`master`、`write` | 混音与母带（混响是合成的冲激响应） | 无 |
| `sampler.analysis` | `stats`、`lufs`、`true_peak_db`、`detect_clicks`、`pitch_track`、`plot`、`report` | 客观测量 | 无 |

音符一律写成 `(start_s, dur_s, midi, velocity[, {关键字}])`，velocity 1..127；`sampler.grid.m('F#4')` 把音名换成 MIDI。

## 5. 乐谱的结构

一部片的乐谱放在 `films/<片>/score/score.py`，输出到 `films/<片>/out/music/`。三个写谱用的模块：

- `sampler.grid`：`Grid(timeline.json)` 给出 `pos('2.3+4f')`（小节.拍 → 秒，可带拍或帧的偏移）、`span('1.3-2.3')`、
  `cue('paper')`、`label(t)`；`Human(seed)` 是固定种子的人性化；`EventLog` 收集 `events` / `notes` / `markers` / `cuts` /
  `curves`，写出 `events.json`。时间线的 `framesPerBeat` 与 `fps × 60 / bpm` 不一致时直接报错。
- `sampler.bus`：`to_lufs`（分轨平衡）、`automate` / `gate`（干声上的增益曲线与静音窗口）、`place_on_attack`
  （把片段里实测的起音放在指定时刻，误差 0 样本，返回对齐记录）、`room`、`finish`（淡入淡出 → `mix.master` → 首尾再淡）、
  `write_outputs`（`score.wav`、`stems/`、带 SHA-256 的 `manifest.json`）。
- `sampler.check`：`sync_table`（每个提示点在分轨里的实测起音与偏差）、`grid_alignment`、`tail`（切点后的余音）、
  `jumps`（电平跳变）、`section_loudness`、`ebur128`（ffmpeg 复核）、`summary`。

骨架（`films/demo-paper/score/score.py` 是完整版）：

```python
import numpy as np
from sampler import bus, check, folk, found, mix
from sampler.grid import EventLog, Grid, m

G = Grid('films/<片>/timeline.json')
N = bus.n_samples(G.duration)
log, sync = EventLog(G), []

piano = folk.render('grand_piano', [(G.pos('2.3'), 1.6, m('D3'), 46)], duration=G.duration)
sync.append(('paper', 'piano', G.pos('2.3'), '成纸'))
clip, lead = found.hit('sketchbook_flips_owlstorm', 'dark', 100)
paper, rec = bus.place_on_attack(np.zeros((N, 2), np.float32), clip, G.pos('2.3'), attack=lead)

dry = {'piano': bus.to_lufs(bus.fit(piano, N), -21), 'paper': bus.to_lufs(bus.fit(paper, N), -31)}
wet = {k: bus.fit(mix.reverb(x, 'hall', mix=0.25), N) for k, x in dry.items()}
master, info = bus.finish(sum(wet.values()), N, target_lufs=-16, true_peak_db=-1.2)
bus.write_outputs('films/<片>/out/music', master, wet)
log.write('films/<片>/out/music/events.json', measured=dict(sync=check.sync_table(dry, sync, G.fps)))
```

要点（都是踩过的坑）：

- 平衡用整合响度（`to_lufs`），不在分轨上压限；短促的声音（一声纸响）的整合响度偏低，目标要比长音低几 dB。
- 7.1 这类「收住」放在混响**之前**（`bus.gate` 作用在干声上），房间的尾巴才留得下来。
- 母带的真峰值上限比时间线要求再低 0.2 dB（-1.2 dBTP）：24 bit 量化和别的测量器（ffmpeg）会差一点。
- 重音要对「听到的起音」：钢琴是击槌那一下，纸是最响的那个瞬间，连续的笔声是第一次接近峰值的地方，都不是文件开头。
  `sync_table` 量的是同一个定义（2 ms 包络第一次到窗口峰值的 -12 dB 以内）。
- 弓弦乐器的长音起得慢（实测 100 ms 以上），不要拿它当硬同步点。

## 6. 混音与自检

- 乐谱：`music/sampler/.venv/bin/python films/demo-paper/score/score.py --check` 打印整合响度、真峰值、削波与咔哒候选、电平跳变、各段响度、
  切点后的余音、同步表、网格对齐和 ffmpeg ebur128 的复核；同样的数字写进 `events.json` 的 `measured`。
- 单个文件：`music/sampler/.venv/bin/python -m sampler.check films/demo-paper/out/music/score.wav`。
- 乐器：`music/sampler/.venv/bin/python music/sampler/selfcheck.py <组>`，输出到 `$AGENTREEL_CHECK`（默认 `music/sampler/out/check/`）：

| 组 | 量什么 | 要的采样 |
| --- | --- | --- |
| `piano` `symp` `unacorda` | 音阶与和弦、力度扫描是否单调、踏板逻辑、弦共鸣、una corda | `piano` |
| `folk` | Steinway B 与民谣竖琴：力度扫描、逐音音准、咔哒、分解和弦 | `vcsl` |
| `found` | 事件切分、连续一笔的起音 | `found` |
| `strings` | 长音和弦的接缝、单声部音准、拨弦起音、泛音 | `vsco2` |
| `voice` | 人声音准、HNR、颤音、吸气 | `vocalset` |
| `choir` `choir2` | 元音、共振峰对照表、气声 | 无 |
| `nature` `sfx3` `bowl` `birds` | 环境声接缝、音效、瓷碗模态、鸟鸣目录 | `field`、`vcsl`、`objects` |
| `fx` `pad` `master` | 效果器、pad、母带链（`master` 用一段 20 秒小样） | `piano`、`field`、`vcsl` |

`r2`、`r3` 是两批组合，`all` 是全部。

## 7. 环境变量

| 变量 | 作用 |
| --- | --- |
| `AGENTREEL_SAMPLES` | 采样根（默认 `music/sampler/samples/`）。缺文件时报 `MissingSamples`，消息里写着该跑的 fetch 命令 |
| `AGENTREEL_SAMPLES_LOG` | 把打开过的采样文件逐个追加进这个文件（给 `fetch_samples.py --files`） |
| `AGENTREEL_CHECK` | `selfcheck.py` 的输出目录 |
| `FFMPEG` | ffmpeg 可执行文件（默认在 PATH 里找） |
| `GITHUB_TOKEN` | 可选，放宽 GitHub API 的速率限制（频繁重试 fetch 时设） |
| `FREESOUND_API_KEY` | 可选，Freesound 走官方 APIv2 在线复核许可、取没有记录预览地址的条目 |

## 8. 署名义务

公开用到下列素材的作品（视频、音频、演示），**必须**带上 `CREDITS.md` 里对应的英文署名文本：

- Salamander Grand Piano（`sampler.piano`、`fx.impact` 的钢琴层、`_piano_tuning.py` 的测量数据）：CC BY 3.0。
- `field` 里的 `pine_forest_dobroide`（`nature.ambience('pine_forest')`）与 `cricket_gryllus_campestris`（`nature.ambience('crickets')`、`'night_insects'`）：CC BY 3.0。
- VocalSet（`sampler.voice`）：CC BY 4.0。

VCSL（含 Steinway B）、VSCO 2 CE、Freesound 的 found / objects、Commons 上 CC0 与公有领域的录音不要求署名。
最省事的做法：渲完看 `events.json` 的 `measured.samples`（打开过的全部采样），对照 `CREDITS.md`。
`films/demo-paper` 只用 CC0 与合成的声音，公开时不需要署名。

## 9. 文件

| 路径 | 内容 |
| --- | --- |
| `sampler/` | 库：乐器模块、`mix` / `analysis`、写谱用的 `grid` / `bus` / `check`、`_paths.py`（采样根与访问记录） |
| `sampler/prepare/` | 下载后的导入：`vcsl.py`、`vsco.py`、`field.py`、`vocalset.py`（转 FLAC、实测音高、建索引） |
| `sampler/data/vcsl_ref_levels.json` | VCSL 民谣乐器整套的电平中位数（让只取部分文件时音量不变） |
| `fetch_samples.py` | 下载与许可核对 |
| `selfcheck.py` | 乐器自检 |
| `tools/commons.py` | 在 Commons 上找候选录音、看许可（只列 CC0 / PD / CC BY） |
| `tools/catalog_birds.py`、`tools/measure_piano_tuning.py` | 重建鸟鸣目录、重测钢琴调律（`_piano_tuning.py`） |
| `CREDITS.md` | 每个来源、许可、核对方式与署名文本 |
| `requirements.lock`、`pyproject.toml` | 依赖 |

## 10. 已知局限

- 只在 macOS arm64、Python 3.12 上验证过逐样本复现；别的平台上 numpy / scipy 的浮点实现不同，结果会有末位差异。
- Steinway B 只有全音间隔、3 个力度层的采样（变速最多 1 个半音），极低和极高的几个音实测音高与文件名不符；
  Salamander 更细（每小三度、16 层）但要署名、体积大。
- Freesound 的文件取的是 HQ 预览（有损 Ogg），原始 WAV 要登录；如果 Freesound 重新编码了预览，SHA-256 对不上就不用，要人工更新记录。
  22 条里只有 4 条记录了预览地址（实际下载核对过的），其余要 `FREESOUND_API_KEY`；有 key 时下到的地址会记进
  `fetch_manifest.json`，可以抄回 `fetch_samples.py` 的目录表。
- 合唱是合成的，o / u 在 3–5 kHz 比实测共振峰表暗；要更像真人用 `sampler.voice`。
