# filmsynth：纯代码合成的配乐库

给片子写配乐的 Python 库：乐器、效果器、混音、响度与真峰值、画面要读的事件表，全部是代码。移植自《成路》（MADE BY WALKING）的配乐库，经《一拍》（ONE BEAT）扩展（两部片的做法与教训见 [`docs/cases/chenglu.md`](../../docs/cases/chenglu.md)、[`docs/cases/yipai.md`](../../docs/cases/yipai.md)）；两部片的乐谱不在这里，示例片见仓库的 `films/demo-light/score/`。时间线的写法见 [`docs/TIMELINE.md`](../../docs/TIMELINE.md)，各项自检在验收里的位置见 [`docs/QA.md`](../../docs/QA.md) 的「工具对照」。

## 1. 设计原则

- **纯代码合成，无采样。** 每一个声音都由 `filmsynth/` 里的代码合成：不用录音、采样、音色库、soundfont、外部冲激响应，也不用神经网络或任何生成模型。混响的冲激响应同样由代码生成。所以库里没有任何音频文件，也没有署名义务。
- **确定性。** 每个音符的随机数种子只取决于会话种子、轨名和音符在轨上的序号（`crc32`），合奏里每个演奏者的偏差也由种子决定。同一份谱、同一份代码，渲染结果每次都一样。
- **逐样本可复现。** 各音轨在子进程里并行渲染，但结果与进程数无关：`--workers 1` 和默认的「CPU 数减 1」逐样本相同（示例片实测：混音与 16 个分轨的 24-bit 整数差全为 0，sha256 相同）。前提是同一锁定版本（`requirements.lock.txt`）、同一平台；换 CPU 架构或操作系统后，numba 的 `fastmath` 与 FFT 实现可能让最后几位不同（未实测，见第 12 节）。
- **时间线是唯一的时间来源。** 一部片的速度、拍号、小节数、帧率、同步点、意图曲线都在片目录的 `timeline.json` 里；库从那里读，不写死任何片子的路径或数字。输出的 `events.json` 把每一拍、每个给画面的事件换算成秒和帧，画面引擎读它。
- **可测。** 管线里没人能听，所以凡是说「同步了」「切干净了」「−16 LUFS」的地方都要量出来：`filmsynth.check` 是通用的自检工具（第 9 节）。它是客观指标，**不能代替人耳**。
- 48 kHz，内部 float64，立体声数组形状 `(2, n)`，输出 24-bit WAV。只依赖 numpy、scipy、numba、soundfile、pyloudnorm、matplotlib。

## 2. 安装

需要 Python 3.12（numba 的要求；macOS 可以 `brew install python@3.12`，Linux 用发行版或 pyenv 的 3.12）。在本目录 `music/filmsynth/` 下：

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt     # 版本锁定；最后一行 `-e .` 以可编辑方式安装本库
.venv/bin/python examples/pluck.py                 # 试一下：写出 examples/out/pluck.wav、pluck.png、pluck.qc.json
```

Windows 上把 `.venv/bin/python` 换成 `.venv\Scripts\python`。`requirements.lock.txt` 里的相对路径 `-e .` 相对的是运行 pip 时的当前目录，所以要在本目录下执行。

- 第一次运行时 numba 先编译内核（本机新装的环境里第一个例子约 40 秒，各渲染进程会各编译一次），结果缓存在源码旁的 `__pycache__`，之后直接用。改了带 `@njit` 的文件以后，下一次会重新编译。
- 渲染是多进程（spawn）。**脚本必须有 `if __name__ == "__main__":` 保护**，否则子进程会重复执行整个脚本。
- 只想装库、不要锁定版本：`pip install -e music/filmsynth`（依赖的下限写在 `pyproject.toml`）；逐样本可复现只对锁定版本作保证。

示例片的配乐（在仓库根目录运行）：

```sh
music/filmsynth/.venv/bin/python films/demo-light/score/score.py            # 渲染到 films/demo-light/out/music/ 并自检，约 8 s
music/filmsynth/.venv/bin/python films/demo-light/score/selfcheck.py --determinism   # 两种进程数各渲染一次，逐样本比较
```

## 3. 五分钟上手

```python
from filmsynth import Session, Pluck, Bell, Strings, Reverb, GainCurve
from filmsynth.film import load_timeline, timeline_curve, mark_sync, music_out, loudness_target

FILM = "films/demo-light"                               # 片目录（含 timeline.json）

if __name__ == "__main__":
    tl = load_timeline(FILM)                            # 片目录、timeline.json 路径，或 None（$FILM_DIR / 当前目录往上找）
    s = Session.from_timeline(tl, tail=0.0, seed=1)    # bpm、拍号、小节数、fps 都取自时间线
    energy = timeline_curve(s, tl, "energy")            # t(秒) -> 0..1，画面与配乐共用的意图
    s.export_curve("energy", energy)                    # 写进 events.json 的 curves
    mark_sync(s, tl)                                    # 时间线的同步点写进 events.json 的 markers

    hall = s.bus("hall", Reverb("hall", rt60=2.8))
    pl = s.track("pluck", Pluck(let_ring=False, damp_time=0.12), sends={"hall": -12}, stem="pulses")
    for i in range(12):                                 # P1：1.3 起的八分音符
        nt = pl.at(f"1.3+{0.5 * i}", 0.45, "B3" if i % 2 == 0 else "F#4", 0.5)
        s.event("pulse", nt, side="L" if i % 2 == 0 else "R")   # 给画面的事件：取音符的实际起点
    bell = s.track("bell", Bell(preset="glass"), gain_db=-6, sends={"hall": -6}, stem="bells")
    s.event("chime", bell.at("3.1", 3, "D6", 0.5), id="H1")
    strings = s.track("vc", Strings(section="vc", tasto=0.4), sends={"hall": -10}, stem="strings")
    strings.at("3.1", 8, "D3", [(0, 0.3), (1, 0.6)])     # 力度折线：两小节的渐强

    s.cut("9.1", "10.1", tails=True)                    # 干切：干声在这一帧停，混响尾巴自然衰减
    tg = loudness_target(tl)                            # 时间线的整合响度与真峰值上限
    s.render(music_out(FILM), target_lufs=tg["integrated"], ceiling_dbtp=tg["true_peak"])
    # 写出 <片>/out/music/ 下的 score.wav、stems/*.wav、events.json
```

完整的例子：`examples/`（每件乐器的演示）和 `films/demo-light/score/score.py`（一部片的完整配乐）。

## 4. 会话 API（`filmsynth.session`）

### 4.1 时间与位置

- 位置一律是 1 起算的「小节.拍」：`note(bar, beat, ...)`，`beat` 可以带小数（`2.5` 是第 2 拍的后半拍）。
- 凡是接受位置的地方也接受字符串 `"13.2"`、`"13.2+0.5"`（拍内小数）和元组 `(13, 2)`、`(13, 2, 0.5)`。时间线另有 `"13.2+4f"`（晚 4 帧），用 `filmsynth.film.tl_pos()` 换成拍。
- `s.t(pos)` 把位置换成秒；`s.pos(seconds)` 反过来；`s.beat(pos)` 给 0 起算的拍序号。
- `s.ritardando(from_pos, to_pos, end_bpm)` 做线性渐慢，之后保持 `end_bpm`。必须在写任何音符之前调用。时间线的网格是固定速度的（画面按 `framesPerBeat` 数帧），有渐慢时 `events.json` 的拍就不在网格上了，自检会报出来。
- `s.end_time` 是 `bars` 小节结束的时刻。渲染长度为 `end_time + tail`，最后 `end_fade`（默认 0.5 s）做淡出。时间线把音乐尾巴算进 `bars` 时（如示例片的 9.1–10.1），用 `tail=0`。

### 4.2 从片子的时间线建会话（`filmsynth.film`）

| 函数 | 作用 |
|---|---|
| `find_timeline(where=None)` | 找 `timeline.json`：`where` 是文件、片目录，或 `None`（先看环境变量 `FILM_DIR`，再从当前目录往上找） |
| `load_timeline(where=None)` | 读成 dict（传 dict 原样返回） |
| `film_dir(where)`、`music_out(where)` | 片目录；配乐输出目录 `<片>/out/music`（画面引擎在那里读 `score.wav` 与 `events.json`） |
| `timeline_session(tl, **kw)` / `Session.from_timeline(where, **kw)` | 按时间线的 `bpm`、`beatsPerBar`、`bars`、`fps` 建会话，时间线存在 `session.timeline`；先校验 `framesPerBeat = fps × 60 / bpm` |
| `timeline_curve(s, tl, key)` | 时间线里 `{keys: [[pos, value], ...]}` 形状的曲线（`energy` 等）变成 `Curve` |
| `sync_points(tl, hard=None)`、`hard_sync(tl)` | 时间线的 `sync` 列表；`{"id": "T1-T6", "at": "1.3-2.4"}` 这种区间按拍展开成 T1…T6，其他区间（如 `P1`）保留成一个点并带 `to` |
| `mark_sync(s, tl=None)` | 把同步点写成 markers（`kind` 为 `hard_sync` 或 `soft_sync`） |
| `loudness_target(tl)` | `{integrated, true_peak, keys}`：读 `loudness.integratedLUFS` / `truePeakDbtp`（也认旧名 `integrated` / `truePeakMax`），缺省 −16 / −1 |
| `tl_seconds(pos, tl)`、`tl_frame(pos, tl)`、`tl_beat(pos, tl)`、`film_end(tl)` | 时间线位置换成秒 / 帧 / 拍；画面结束的时刻（`filmEnd`） |

### 4.3 音轨、母线、插件

```python
tr = s.track(name, instrument, gain_db=0, pan=0, width=1, sends={"bus": -10}, inserts=[...],
             stem="strings", humanize_ms=0, humanize_vel=0)
bus = s.bus(name, *effects, gain_db=0, stem="fx_name")
```

- **音轨。** 每个音轨一件乐器。`stem` 相同的音轨在导出时合成同一个分轨。`humanize_ms` 和 `humanize_vel` 给每个音符加播种的随机时间、力度偏差，`events.json` 记的是偏移后的实际起点。给画面卡点的音（脉冲、硬同步）不要加 `humanize_ms`。
- **发送。** `sends` 是推子后发送，单位 dB。母线的输出另成一个分轨（默认 `fx_<名字>`；自检按 `fx_` / `reverb_` 前缀认母线分轨）。想让某组分轨自带混响，就给它单独开一条母线。
- **插件（inserts）。** 任何有 `.process(x, ctx)` 的对象都可以：
  - `FeedbackDelay`、`persistence_delay`；
  - `Swell`（增音箱）、`GainCurve`、`FilterSweep`；
  - `EQ`、`Saturate`、`Compressor`、`Width`、`Haas`、`Gain`、`Limiter`、`DCBlock`。
  母线上也能挂，例如 `s.bus("hall", GainCurve(s.curve([...])), Reverb("hall"))` 是「骑」混响的输入。
- **插件会被 pickle 到子进程。** 不要用 lambda，用 `s.curve(...)` 返回的 `Curve` 对象。不能 pickle 的音轨会退回主进程渲染，结果正确，只是慢。
- **自动化。** `tr.automate("gain_db" | "pan", [(pos, value), ...], interp="linear"|"smooth"|"step")`。
- **侧链。** `tr.duck("boom", depth_db=6)`：「boom」音轨响时压低本音轨。

### 4.4 写音符

```python
tr.note(bar, beat, dur_beats, pitch, vel=0.7, **expr)
tr.at("13.2+0.5", dur_beats, pitch, vel, **expr)
tr.hit(bar, beat, vel, **expr)                      # 无音高
tr.seq(bar, beat, [(dur, pitch, vel, {expr}), ...]) # 连续写，pitch=None 是休止
```

- **pitch。** MIDI 数或音名，如 `"F#3"`、`"Bb2"`、`"A4+5c"`（带音分）。无音高的声音给 `None`。
- **vel。** 0..1；`(v0, v1)` 是贯穿该音的线性渐变；`[(frac, v), ...]` 是折线。弦乐和合唱用它做长渐强，力度同时控制响度和明暗。
- **expr。** 乐器参数（见第 6 节）。拼错会直接报错并列出可用名字。所有乐器另外接受：
  - `pan`：音符级声像偏移；
  - `gain_db`：音符级增益（不改音色，只改电平）；
  - `tag`：写进 `events.json` 的 `notes`；
  - `voice`：给连奏乐器指定声部。

  弦乐和合唱例外：它们按乐句整体渲染，不认音符级的 `pan` 和 `gain_db`。要单独调某几个音的电平，用轨道推子 `tr.automate("gain_db", ...)`，或者另开一条轨。
- **连奏。** 合唱和弦乐把同一声部里首尾相接的音（间隔 < 30 ms）唱成或拉成一条连续的线，带滑音；不相接就是新的乐句。和弦会自动分配到不同声部，用 `voice=` 固定声部走向。一句里除了音高、力度和合唱的 `vowel` 逐音生效，其余参数（`brightness`、`sordino`、`attack`、`release` 等）都取这句第一个音的：想让某个音换参数，就别让它跟前一个音相接。
- **音阶工具。** `filmsynth.penta(degree, octave, tonic="D"|"B")`、`snap_penta(midi)`（D 宫五声 / B 羽）。

### 4.5 硬切、干切、主音量

- `s.cut(start, end, fade_out_ms=4, fade_in_ms=40, choke=True, tails=False)`：
  - 在 `start` 用 4 ms 升余弦把一切压到 −120 dB，`end` 处再打开；
  - `choke=True` 时，`start` 之前开始的一切（包括延迟回声和混响尾巴）永久切断，门再打开时不会漏回来；
  - `tails=True` 是**干切**：所有音轨（连同插件的回声）照样在 `start` 那一帧停，但混响母线既不切也不 choke，切点之前送进混响的声音自然衰减（音乐停在那一帧，房间还在响）。此时同一个分轨不能既有音轨又有母线（会报错）；
  - `start` 之后写的音符照常发声；跨过切点的长音也会被切断，需要的话在切点之后重新起音。
- `s.master_gain([(pos, dB), ...], interp)`：主音量自动化，同时作用于每个分轨（干切时也作用于母线分轨）。
- `s.event(type, at, **data)`：给画面的事件，`at` 是位置，或 `track.note()` / `at()` 返回的音符（取它的实际起点，人性化偏移也算进去）。
- `s.marker(pos, name, **data)`：设计位置（同步点、涌起的峰值……），进 `events.json` 的 `markers`。

### 4.6 渲染与输出

```python
res = s.render(out_dir, mix_name="score.wav", workers=None, target_lufs=-16.0, ceiling_dbtp=-1.0,
               stems=True, events=True, master_inserts=None, end_fade=0.5)
```

1. 各音轨并行渲染（`workers` 默认 CPU 数减 1；`workers=1` 全在主进程，结果相同）。
2. 母线并行处理。
3. 每个分轨依次做：去直流（6 Hz），主音量与切点的门，开头 5 ms 淡入，结尾淡出。
4. 求和后过 `master_inserts`，再做响度归一（pyloudnorm）和前瞻真峰值限幅（4 倍过采样检测）。
5. 分轨乘上同一个归一增益。分轨之和等于限幅前的总混音，限幅是分轨之后唯一的非线性环节。
6. 返回 `res.mix`、`res.stems`、`res.stats`（各音轨、母线耗时，归一增益）、`res.paths`。
7. `target_lufs=None` 时不归一、不限幅（只削到 ±1）。

## 5. 电平约定

- 每件乐器都校准过：力度 0.7 的单个音（或一个乐句），0.4 s 窗的最大响度约 **−24 LUFS**（多数在 −22 到 −30 之间）。
- 小音效更轻，约 −32 到 −46。管风琴全奏比单个音栓响 10 dB 左右，与真琴一致。
- 平衡就用 `gain_db` 调，不必猜各乐器的原始电平。同一乐器跨音域的电平差在 ±3–4 dB 内（拨弦低音区按 K 加权略轻）。
- 校准表由 `examples/calibrate.py` 量出（每件乐器一个力度 0.7 的音，写 `examples/out/calibrate.json`）。本机实测摘录：

| 乐器 | 0.4 s 最大响度 | 乐器 | 0.4 s 最大响度 |
|---|---|---|---|
| Pluck B3 | −25.2 | Strings vn1（14 人） | −19.3 |
| Bell glass / church | −29.6 / −25.9 | Strings va / vc / cb | −24.2 / −24.8 / −28.2 |
| Organ 8′ Gedackt / 全奏加簧管 | −29.4 / −13.0 | Choir 女高 / 女低 / 男高 / 男低（12 人） | −26.2 / −23.3 / −27.1 / −34.9 |
| Xiao D5 | −24.8 | FeltPiano C4 | −24.0 |
| Boom B0 / Timpani B1 | −23.9 / −27.6 | TautString A3 | −22.2 |
| HeartDrum B1 | −27.7 | Wind / AirRush | −23.7 / −24.0 |
| FeltTick / Tick / Tap | −34.9 / −34.3 / −46.4 | Sparkle / Shimmer | −35.1 / −32.6 |

## 6. 乐器参考

### 6.1 推荐用法

- **拨弦 `Pluck`（介于古琴和竖琴之间，温暖、近）。**
  - 动机用 vel 0.5–0.7、`brightness` 0.4–0.6。句尾长音加 `vib_cents` 8–12、`vib_rate` 3–4（吟）；或 20–25、1.8–2.2（猱）。上滑音用 `glide_to`；泛音用 `harmonic=2|3|4`，写的是空弦音高，实际发 n 倍频。
  - 回声用 `persistence_delay`，`time_beats=0.75`、`time_r_beats=1.0`、`mix≈0.4`。
  - **给画面卡点的脉冲**：`let_ring=False`、`damp_time` 0.07–0.12、`attack=0.004`（4 ms 软起音，峰值圆、限幅器不碰它）、固定 `body_seed`（每个音用同一个木质共鸣体，电平均匀；默认 `None` 每个音随机挑一个，电平会跳 ±3 dB）。
  - 同一音高重复弹会自然重叠，想要干净就止音。拨奏（pizz）可用 `Pluck(brightness=0.3, decay=0.3, body=0.6)` 代替。
- **毛毡钢琴 `FeltPiano`。** 温暖、偏暗的击弦音。`let_ring=True` 是踩着踏板；换和弦的地方写 `let_ring=False` 和实际要响的时值（`damp_time` 0.45–0.6 是轻轻放下制音器）。适合和弦落点与安静的动机。
- **箫 `Xiao`。**
  - 写实际音高，音域约 D4–E6；一句之内首尾相接的音（间隔 < 30 ms）共用一口气，`tongue` 是句内的软吐音，0 是连音。
  - 力度逐样本跟着走：`vel=[(0,0.62),(0.5,0.32),(1,0.2)]` 就是一口气里的渐弱。揉音晚起：`vib_delay` 0.5 s 左右开始，`vib_attack` 0.6 s 到满；短音听不到揉音是对的。
  - 滑音用 `glide_from`（起音时从别的音滑来，像半按孔）或 `glide_to`。一口很弱的气：vel 0.06–0.12、`breath` 0.6 以上、`attack` 0.5、`tongue=0`。
- **钟与玻璃 `Bell`。**
  - `glass` 适合高处一声轻钟、卡农；`bianzhong`（编钟）：`strike="center"` 发正鼓音（写的音高），`strike="side"` 发高小三度的侧鼓音；`celesta` 适合琶音；弓擦玻璃 `bowed` 适合高处长音；`church` 的 hum 低八度，作为最后一声钟时注意低频；`lamp` / `LampBell` 是小手铃，调到和弦音上。
  - 钟类很亮，压在全奏下面 10 dB 左右也听得出来（在 4 kHz 以上的频带里它常常和别的声部持平）。
- **管风琴 `Organ`。**
  - 写 8′ 音高。`"32"` 低两个八度：写 B2 发 B0（31 Hz），写 D3 发 D1（37 Hz）。别写得更低。
  - 脚键用 `stops=("32","16")` 单独一轨；手键另一轨，挂 `Swell(curve)` 做增音箱。
  - 全奏：`("8p","8g","4p","2.67","2p","mix","8r")`，再加 `"16r"`、`"16p"` 更厚。所有管风琴轨用同一个 `wind_seed`（默认 11），共用一个风源。
- **合唱 `Choir`（共振峰合成）。**
  - 每声部一轨（soprano/alto/tenor/bass），`distance` 0.5–0.7。远处低吟用 `vowel="mm"` 或 `"oo"`，到达点用 `"ah"`；`vowel=("oo","ah")` 在一个音内渐变。
  - `attack` 1–2 s 做渐入，vel 用 `(v0, v1)` 或折线做渐强。每声部 12 人最自然；8 人更快，16 人更厚但更慢。
- **弦乐 `Strings`。**
  - 五个声部各一轨；和弦在一轨里用 `voice=` 固定声部，换和弦时首尾相接就是连奏（带换把滑音）。
  - pp 进入用 `attack` 1–1.5、`vel=[(0,0.12),(1,0.5)]`。十六分音符织体用 `articulation="spiccato"`；颤弓 `"tremolo"`；加弱音器 `sordino=True`。
  - flautando、弓奏泛音的长音：`tasto` 0.6–1、`brightness` 0.2–0.35、`vib_cents` 4 左右；`tasto=0` 是普通奏法。
  - 渐强到顶「放开」：乐句的最后一个音正好结束在切点上，`release` 0.1 s 左右（示例片 5.1：3 帧内降 10 dB）。
  - 弦乐和合唱按乐句整体渲染，音符级的 `gain_db` 不生效。要让一个长音单独「亮一下再掐断」，给它单开一轨，用推子 `tr.automate("gain_db", …)`。
- **一根被拉紧的弦 `TautString`。** 从低 `slack_cents` 处升到写的音，拍频和亮度一起升；适合「汇聚」「绷紧」到一个落点：写在落点之前，时值到落点为止（`tighten` 0.92 表示在 92% 处到音）。想要绷紧而不变响，力度写成近乎不变的 `(0.46, 0.5)`。
- **低频重拍。**
  - `Boom`：音高写落点（B0、D1、E1），扫频短（`sweep=12`、`sweep_time=0.05`）是软的一击；配 `HeartDrum` 或 `Timpani`（`hardness` 0.3 左右是毛毡槌）给中低频的身体。
  - `HeartDrum`：低而干的「咚」（毛毡槌、大框鼓），`dry` 0.3 圆、0.85 紧而干，音高不变；`dub` 0.2 加一下轻的第二拍。
  - 次低频的持续音用 `Drone(chord=(0,), harmonics=2, brightness=0.1)` 加两道 110 Hz 低通，`tr.duck("boom")` 给重拍让路。
  - `Taiko`、`Footsteps` 是更有节奏感的鼓。
- **转场。**
  - 反向涌起 `ReverseSwell`：它的峰值落在音符末尾**之前**（反转的混响停在自己的预延迟上，最后 8 ms 是淡出）；`space="plate"`、`rt60=2.6` 时峰在写的终点前约 2.2 帧，峰更尖；cathedral 会早约 4.5 帧而且平顶。要让峰落在同步点前 N 帧，终点写在「同步点 − N 帧 + 2.2 帧」，再用自检量一遍。
  - 气流 `AirRush`：`peak=0.85` 表示在音符 85% 处到顶；跟镜头穿过某物时把顶放在同步点后 1–2 帧；`whistle` 调低就只剩空气。
  - `Shimmer`（和弦音上的颗粒闪光）、`Sparkle`（向上散开的闪光，「字化为光」）、`Riser`（落在写的终点上的上升音）。
  - 风雪 `Blizzard`，结尾交给 `s.cut`。`part="noise"` 和 `part="grains"` 把风雪噪声和冰粒拆到两条轨（两条轨的冰粒完全相同）。
- **环境。** `Wind(intensity=…, intensity_end=…, whistle_pitch="B5")`，口哨音调到和弦音上；房间底噪用 `Wind(intensity=0.06, whistle=0, brightness=0.12)` 加 60 Hz 高通。
- **界面声与小音效。**
  - `Tick`：很短的调音咔嗒；`FeltTick`：更软、更闷的一下（毛毡或软木），`ring` 0.1–0.2 s，几乎不送混响；`Tap`：无音高、极轻的指尖敲击（`material` 0 纸 .. 1 木）；`WoodKey`：木键，配 `humanize_ms=15–20` 像打字。
  - `Breath(shape="inhale")` 吸一口气；`shape="pulse"` 是一口口软的「呼」，给音高只染颜色、不出音，可以代替钢片琴做高处的稀疏织体。
  - `FlamePop`、`Puff`、`Crunch`（霜 / 草 / 雪）、`Hiss`（沿路掠过的嘶声）。

### 6.2 参数表

下面每张表都由代码里的 `PARAMS` 生成：`python -m filmsynth.instruments [名字 ...]` 打印同样的 Markdown，`Instrument.describe()` 打印文本版。所有乐器另外接受第 4.4 节的 `pan`、`gain_db`、`tag`、`voice`。

#### `Pluck`

Warm plucked string (guqin/harp hybrid), range B1-B5. Warm and close.

| 参数 | 默认 | 说明 |
|---|---|---|
| `brightness` | `0.5` | 0 dark/flesh .. 1 bright/nail; also scales with velocity |
| `pluck_pos` | `0.18` | pluck position as a fraction of the string (0.05 near bridge .. 0.5 middle) |
| `decay` | `1.0` | multiplies the natural T60 (low notes ~9 s, high ~2.5 s) |
| `stiffness` | `0.35` | 0 ideal string .. 1 stiff (more inharmonic upper partials) |
| `body` | `0.45` | amount of wooden body resonance 0..1 |
| `wood` | `0.5` | body size/colour 0 small-bright .. 1 large-dark |
| `attack` | `0.0` | 0 normal; >0 softens the onset over that many seconds (brushed / rolled) |
| `let_ring` | `True` | True: ring past the written duration; False: damp at the written end |
| `max_ring` | `7.0` | cap on ring length in seconds |
| `damp_time` | `0.12` | T60 (s) after damping |
| `harmonic` | `0` | 0 normal; n>=2 plays the n-th harmonic (fan yin) of the written pitch's string |
| `vib_cents` | `0.0` | yin/nao vibrato depth in cents (yin ~8, nao ~25) |
| `vib_rate` | `2.8` | vibrato rate Hz (yin 3-4, nao 1.5-2.5) |
| `vib_delay` | `0.25` | seconds before vibrato starts |
| `vib_decay` | `1.2` | vibrato depth decay time (s); 0 = constant |
| `glide_to` | `None` | pitch to slide to (shang/xia slide) |
| `glide_from` | `None` | pitch to slide from at the onset |
| `glide_start` | `0.25` | seconds after onset when glide_to begins |
| `glide_time` | `0.35` | glide duration (s) |
| `slide_noise` | `1.0` | finger-slide squeak level during glides |
| `thump` | `0.35` | finger-flesh thump level |
| `polar` | `0.35` | second polarisation level (beating, two-stage decay) |
| `spread` | `0.35` | stereo placement by pitch (low left, high right) |
| `body_seed` | `None` | None: each note gets one of 7 wooden bodies at random (natural variety, +-3 dB); 0..6: always that body (an even pulse) |

#### `Xiao`

End-blown bamboo flute (箫), jet drive + waveguide bore; legato phrases share one breath.

| 参数 | 默认 | 说明 |
|---|---|---|
| `breath` | `0.45` | air heard around the tone 0..1 (missed jet + turbulence) |
| `attack` | `0.09` | breath onset of a phrase (s): 0.05 tongued .. 0.3 a slow 'hu' |
| `release` | `0.16` | breath release at the end of a phrase (s) |
| `tongue` | `0.35` | soft tonguing ('du') of a note inside a phrase 0..1; 0 = slur |
| `portamento` | `0.035` | finger change between slurred notes (s) |
| `vib_rate` | `5.0` | breath vibrato rate (Hz), 4.5-5.5 |
| `vib_depth` | `0.12` | breath vibrato depth (fraction of the breath pressure) |
| `vib_cents` | `6.0` | pitch swing that rides on the breath vibrato (cents) |
| `vib_delay` | `0.5` | seconds into a note before the vibrato starts |
| `vib_attack` | `0.6` | seconds for the vibrato to reach full depth |
| `glide_from` | `None` | pitch to slide up or down from at the note's start (half-holing) |
| `glide_time` | `0.18` | slide time (s) |
| `glide_to` | `None` | pitch to slide to later in the note |
| `glide_start` | `0.5` | seconds after the note's start when glide_to begins |
| `lip` | `0.35` | jet offset at the edge: 0 hollow (odd partials) .. 0.8 fuller (even partials) |
| `edge` | `1.0` | how far the jet swings across the edge per breath (harmonic richness) |
| `bore_t60` | `0.09` | ring time of the bore at the fundamental (s) |
| `bore_dark` | `0.35` | loss of the high partials in the bore 0..0.9 |
| `center` | `0.0` | stereo position |
| `width` | `0.18` | stereo width of the player's sound |
| `humanize` | `1.0` | scales breath drift, pitch drift and timing of the consonants (0 = steady, tests) |

#### `Bell`

Modal bell / glass family (cool colours). Pitch = perceived strike pitch (for bianzhong: centre tone).

| 参数 | 默认 | 说明 |
|---|---|---|
| `preset` | `'glass'` | glass / church / bianzhong / celesta / bowed / lamp |
| `hardness` | `None` | mallet hardness 0 soft .. 1 hard (default from preset); scales high partials |
| `decay` | `1.0` | multiplies all T60s |
| `brightness` | `0.5` | tilts partial amplitudes up (1) or down (0) |
| `strike` | `'center'` | bianzhong only: 'center' (A tone) or 'side' (B tone a minor third up) |
| `beat` | `1.0` | doublet beating amount (0 = none) |
| `spread` | `0.5` | stereo spread of partials |
| `center` | `None` | stereo centre of the partial cloud (default: from pitch) |
| `damp` | `False` | True: stop (choke) at the written end |
| `attack` | `None` | bowed only: onset time in seconds (default 0.8) |
| `release` | `None` | bowed only: fade after the bow stops (default: natural ring) |
| `tremble` | `0.0` | bowed only: slow amplitude waver 0..1 |
| `max_ring` | `12.0` | cap on ring length (s) |

#### `Organ`

Additive pipe organ with registration, chiff, wind, C/C# chest layout.

| 参数 | 默认 | 说明 |
|---|---|---|
| `stops` | `('8p', '8g', '4p')` | registration: tuple of stop keys (see module doc) |
| `chiff` | `0.5` | pipe speech transient amount 0..1 |
| `wind` | `0.5` | wind noise / waver amount 0..1 |
| `attack` | `1.0` | multiplies pipe speech time (larger = slower, softer onset) |
| `release` | `1.0` | multiplies pipe release time |
| `brightness` | `0.5` | voicing tilt 0 dark .. 1 bright |
| `detune` | `1.4` | pipe tuning spread in cents (std) |
| `spread` | `0.65` | stereo width of the pipe layout |
| `tremulant` | `0.0` | tremulant depth 0..1 (5.6 Hz) |
| `wind_seed` | `11` | seed of the shared wind supply (keep equal across organ tracks) |

#### `Choir`

Choir section (soprano / alto / tenor / bass) of 8-16 simulated singers, formant synthesis.

| 参数 | 默认 | 说明 |
|---|---|---|
| `section` | `'alto'` | soprano / alto / tenor / bass |
| `singers` | `12` | number of singers (8-16 recommended) |
| `vowel` | `'oo'` | ah / oh / oo / eh / ee / mm, or (from, to) to morph across the note |
| `attack` | `0.45` | phrase onset time (s) |
| `release` | `0.7` | phrase release time (s) |
| `vib_cents` | `22.0` | mean vibrato depth (cents); each singer 0.4-1.5x |
| `vib_rate` | `5.4` | mean vibrato rate (Hz); singers spread 4.8-6.2 |
| `detune` | `6.0` | per-singer tuning std (cents), clipped to +-12 |
| `timing` | `0.035` | per-singer entry spread std (s) |
| `breath` | `0.3` | aspiration noise 0..1 |
| `portamento` | `0.12` | legato glide time between notes (s) |
| `distance` | `0.5` | 0 close .. 1 far (softer highs, slower edges) |
| `spread` | `0.5` | stereo width of the section |
| `center` | `None` | stereo centre (default by section) |
| `humanize` | `1.0` | scales jitter, drift, shimmer and scoops (0 = machine-steady, for tests only) |

#### `Strings`

Bowed string section; each player simulated (8-14 per section).

| 参数 | 默认 | 说明 |
|---|---|---|
| `section` | `'vn1'` | vn1 / vn2 / va / vc / cb |
| `players` | `None` | players in the section (default 14/12/10/8/8) |
| `articulation` | `'sustain'` | sustain / tremolo / spiccato |
| `attack` | `0.18` | sustain: onset time (s); long pp entries like 0.6-1.5 |
| `release` | `0.4` | sustain: release time (s) |
| `vib_cents` | `None` | vibrato depth (cents), default by section |
| `vib_rate` | `None` | vibrato rate (Hz), default by section |
| `detune` | `4.0` | per-player tuning std (cents) |
| `timing` | `0.025` | per-player timing std (s) |
| `brightness` | `0.5` | overall brightness 0..1 (added to the dynamic brightness) |
| `bow_noise` | `0.5` | friction noise amount 0..1 |
| `portamento` | `0.07` | legato shift time (s); a third of the players slide audibly |
| `trem_rate` | `None` | tremolo strokes/s; None = unmeasured (each player ~11-15/s) |
| `sordino` | `False` | muted strings: darker and softer |
| `tasto` | `0.0` | 0 normal .. 1 over the fingerboard, light and fast (flautando; with a quiet vibrato it also stands for bowed harmonics): fewer upper partials, more air, less vibrato |
| `spread` | `0.25` | stereo spread around the section position |
| `center` | `None` | stereo position (default by seating) |
| `humanize` | `1.0` | scales drift, jitter, attack wobble, shimmer, bow wander (0 = machine-steady, tests) |

#### `Taiko`

Taiko-style drum: membrane modes, strike pitch drop, stick, barrel.

| 参数 | 默认 | 说明 |
|---|---|---|
| `size` | `'nagado'` | odaiko / nagado / shime |
| `tune` | `0.0` | detune in semitones from the size's natural pitch (or give a pitch) |
| `position` | `0.25` | strike position 0 centre .. 0.8 rim (rim = more overtones, less boom) |
| `hardness` | `0.6` | stick hardness 0 soft (felt) .. 1 hard (bachi) |
| `decay` | `1.0` | multiplies T60s |
| `drop` | `1.0` | pitch drop amount after the strike (tension modulation) |
| `center` | `0.0` | stereo position |
| `width` | `0.35` | stereo width of the modes |

#### `Footsteps`

'Footsteps of a crowd': each note is one step taken by a crowd of walkers.

| 参数 | 默认 | 说明 |
|---|---|---|
| `walkers` | `24` | number of walkers |
| `density` | `0.8` | probability that a walker takes this step |
| `timing` | `0.018` | timing spread std (s) |
| `size` | `0.5` | 0 small/light .. 1 big/heavy |
| `softness` | `0.7` | 0 hard sticks .. 1 soft feet/mallets |
| `width` | `0.9` | stereo width of the crowd |
| `earth` | `0.5` | amount of earthy noise (soil, cloth) vs drum tone |

#### `Timpani`

Tuned timpani: air-loaded membrane (near-harmonic (1,1)(2,1)(3,1)... modes), felt mallets, thud of the (0,1) mode, single strokes or rolls (roll=True for the note's duration).

| 参数 | 默认 | 说明 |
|---|---|---|
| `roll` | `False` | roll for the whole written duration |
| `roll_rate` | `11.0` | strokes per second for rolls |
| `hardness` | `0.4` | mallet 0 soft felt .. 1 hard wood |
| `decay` | `1.0` | multiplies T60s |
| `center` | `0.1` | stereo position |
| `damp` | `False` | choke at the written end |

#### `Boom`

Structural low hit: sine down-sweep with a soft transient. Pitch = where the sweep lands (e.g. B0/D1 to sit on the key); the start is `sweep` semitones higher.

| 参数 | 默认 | 说明 |
|---|---|---|
| `sweep` | `24.0` | semitones the sine falls from |
| `sweep_time` | `0.12` | sweep time constant (s) |
| `decay` | `2.2` | amplitude T60 (s) |
| `click` | `0.25` | transient amount |
| `harmonics` | `0.35` | added 2nd/3rd harmonic so small speakers feel the hit |
| `attack` | `0.002` | onset ramp (s); larger for a soft swell into the hit |

#### `HeartDrum`

The heartbeat: a felt mallet on a large frame drum muffled by a hand on the skin. A low, dry 'dum' (pitch = the membrane's fundamental, tuned to the key) with its body in the 150-400 Hz band, so that laptop speakers carry it too, and no ring: every mode is gone within a few tenths of a second.

| 参数 | 默认 | 说明 |
|---|---|---|
| `dry` | `0.4` | 0 round, the fundamental dies in ~0.45 s .. 1 tight, ~0.18 s (same pitch) |
| `felt` | `0.6` | mallet 0 hard felt (more 300-600 Hz knock) .. 1 soft felt (rounder) |
| `body` | `0.6` | level of the frame and air-cavity resonances (150-400 Hz, very short) |
| `position` | `0.25` | strike position 0 centre .. 0.6 (more of the asymmetric modes, less boom) |
| `dub` | `0.0` | level of a soft second beat, 0 = a single beat |
| `dub_time` | `0.3` | time of the second beat after the first (s) |
| `drop` | `0.05` | pitch drop after the strike (fraction of the pitch, relaxes in ~40 ms) |
| `center` | `0.0` | stereo position |
| `width` | `0.15` | stereo width of the modes |

#### `Gong`

Gong family. presets: 'tamtam' (unpitched, blooming wash), 'chau' (big Chinese gong with a pitched centre that sags slightly), 'opera_rise' (small gong whose pitch rises after the hit). Pitch sets the centre tone for 'chau'/'opera_rise' and the overall register for 'tamtam'.

| 参数 | 默认 | 说明 |
|---|---|---|
| `preset` | `'tamtam'` | tamtam / chau / opera_rise |
| `modes` | `220` | number of modes |
| `bloom` | `1.0` | how slowly the upper modes swell after the hit (0 = none) |
| `decay` | `1.0` | multiplies T60s |
| `brightness` | `0.5` | 0 dark .. 1 bright |
| `mallet` | `0.5` | 0 very soft (barely touching) .. 1 hard |
| `damp` | `False` | choke at the written end |
| `width` | `0.8` | stereo width |
| `max_ring` | `16.0` | cap on ring length (s) |

#### `CymbalSwell`

Suspended cymbal rolled with soft mallets: crescendo over the written duration, then rings. vel may be a ramp (v0, v1); default ramps from near silence to vel.

| 参数 | 默认 | 说明 |
|---|---|---|
| `modes` | `160` | resonator count per channel |
| `roll_rate` | `14.0` | mallet strokes per second |
| `ring` | `3.5` | ring time after the swell (s) |
| `brightness` | `0.5` | 0 dark .. 1 bright |
| `size` | `0.5` | 0 small/high .. 1 large/low |
| `choke` | `False` | stop dead at the written end (for a cut) |
| `curve` | `2.2` | crescendo shape exponent |

#### `LampBell`

Small tuned handbell (Bell with preset='lamp'). Tune it to a chord tone.

| 参数 | 默认 | 说明 |
|---|---|---|
| `preset` | `'glass'` | glass / church / bianzhong / celesta / bowed / lamp |
| `hardness` | `None` | mallet hardness 0 soft .. 1 hard (default from preset); scales high partials |
| `decay` | `1.0` | multiplies all T60s |
| `brightness` | `0.5` | tilts partial amplitudes up (1) or down (0) |
| `strike` | `'center'` | bianzhong only: 'center' (A tone) or 'side' (B tone a minor third up) |
| `beat` | `1.0` | doublet beating amount (0 = none) |
| `spread` | `0.5` | stereo spread of partials |
| `center` | `None` | stereo centre of the partial cloud (default: from pitch) |
| `damp` | `False` | True: stop (choke) at the written end |
| `attack` | `None` | bowed only: onset time in seconds (default 0.8) |
| `release` | `None` | bowed only: fade after the bow stops (default: natural ring) |
| `tremble` | `0.0` | bowed only: slow amplitude waver 0..1 |
| `max_ring` | `12.0` | cap on ring length (s) |

#### `Wind`

Night wind: two decorrelated filtered noises with gusts and rare whistles.

| 参数 | 默认 | 说明 |
|---|---|---|
| `intensity` | `0.5` | 0 still air .. 1 strong wind (level, brightness, gust depth) |
| `gustiness` | `0.5` | gust depth and rate |
| `whistle` | `0.25` | amount of whistling resonances (0 none) |
| `whistle_pitch` | `None` | tune whistles to this pitch (and its octave/fifth); None = free |
| `brightness` | `0.5` | 0 dull rumble .. 1 airy hiss |
| `fade_in` | `1.5` | s |
| `fade_out` | `1.5` | s |
| `intensity_end` | `None` | if set, intensity ramps to this value across the note |

#### `Blizzard`

Storm: broadband gusting noise + ice grains hitting (granular clicks, panned).

| 参数 | 默认 | 说明 |
|---|---|---|
| `intensity` | `0.7` | 0..1 |
| `intensity_end` | `None` | ramp target across the note |
| `grains` | `600.0` | ice grains per second at full intensity |
| `grain_level` | `0.5` | ice grain level |
| `part` | `'all'` | all / noise / grains: render only the storm noise or only the ice grains (the grains are the same either way, so the two can sit on separate tracks) |
| `fade_in` | `1.0` | s |
| `fade_out` | `0.02` | s (storms are usually cut, not faded) |

#### `AirRush`

Huge air rush through clouds: a noise band sweeping upward, a whistle with Doppler shift and a pan sweep, sub rumble underneath. Peaks at `peak` (fraction of the note) then falls away.

| 参数 | 默认 | 说明 |
|---|---|---|
| `f_start` | `180.0` | band centre at the start (Hz) |
| `f_end` | `5000.0` | band centre at the peak (Hz) |
| `peak` | `0.85` | where the rush peaks (fraction of the note); after it, a fast decay |
| `pan_from` | `-0.8` | pan at the start |
| `pan_to` | `0.8` | pan at the end |
| `doppler` | `0.5` | whistle Doppler amount |
| `sub` | `0.5` | sub rumble level |
| `tail` | `0.6` | seconds after the written end |
| `whistle` | `1.0` | level of the narrow Doppler whistle inside the rush (0 = air only) |

#### `Shimmer`

Granular glitter of high partials on chord tones. pitch = lowest tone; chord = intervals.

| 参数 | 默认 | 说明 |
|---|---|---|
| `chord` | `(0, 4, 7, 9, 12, 16, 19, 24)` | semitone offsets from the pitch (default D gong-ish) |
| `density` | `18.0` | grains per second |
| `grain` | `0.09` | mean grain length (s) |
| `octaves` | `2` | spread grains up this many octaves |
| `rise` | `0.0` | 0 static; >0 grains drift upward over the note (glyphs to light) |
| `fade_in` | `0.6` | s |
| `fade_out` | `1.2` | s |
| `width` | `1.0` | stereo width |

#### `Drone`

Sustained beating drone: several slightly detuned partials per chord tone, slow filter drift.

| 参数 | 默认 | 说明 |
|---|---|---|
| `chord` | `(0, 7, 12)` | semitone offsets from the pitch |
| `beat_hz` | `0.25` | typical beating rate between detuned copies |
| `harmonics` | `5` | harmonics per tone |
| `brightness` | `0.4` | 0 dark .. 1 bright |
| `attack` | `2.0` | s |
| `release` | `2.5` | s |
| `motion` | `0.5` | slow timbral motion 0..1 |

#### `Riser`

Rising noise + tone that lands exactly at the written end (pitch = landing tone).

| 参数 | 默认 | 说明 |
|---|---|---|
| `span` | `24.0` | semitones the tone rises over the note |
| `noise` | `0.6` | noise amount |
| `tone` | `0.4` | tone amount |
| `curve` | `2.5` | crescendo exponent |
| `tail` | `0.15` | decay after the end (s) |

#### `ReverseSwell`

Reverse-reverb inhale: a chord (or noise) is reverberated, reversed, and placed so its peak lands exactly at the written end. Use source='chord' with pitch + chord, or source='noise'.

| 参数 | 默认 | 说明 |
|---|---|---|
| `source` | `'chord'` | chord / noise |
| `chord` | `(0, 7, 12, 16)` | semitone offsets from the pitch |
| `space` | `'cathedral'` | reverb preset used for the swell |
| `rt60` | `4.0` | reverb time (s) |
| `brightness` | `0.6` | 0 dark .. 1 bright |
| `cut` | `True` | True: end abruptly at the peak (classic); False: let the dry hit ring 0.3 s |

#### `Tick`

Very short tuned click. pitch = the click's tone (e.g. B6 warm, F#7 cool).

| 参数 | 默认 | 说明 |
|---|---|---|
| `decay` | `0.018` | tone decay time constant (s) |
| `click` | `0.35` | broadband edge amount |
| `body` | `0.3` | low wooden body under the tone |
| `center` | `0.0` | stereo position |

#### `FeltTick`

Soft cork / felt 'da': a dull noise transient into one tuned resonance with a short ring.

| 参数 | 默认 | 说明 |
|---|---|---|
| `ring` | `0.2` | T60 of the tuned resonance (s) |
| `hardness` | `0.3` | 0 felt .. 1 hard cork: brightness, level and shortness of the transient |
| `overtone` | `0.3` | level of the wooden overtone (2.76 x pitch, decays 3x faster) |
| `thump` | `0.2` | low body thump (160 Hz, 8 ms) |
| `center` | `0.0` | stereo position |

#### `WoodKey`

Wooden key press. pitch optional: tunes the strongest resonance (keeps typing in the key).

| 参数 | 默认 | 说明 |
|---|---|---|
| `size` | `0.5` | 0 small/high .. 1 large/low |
| `release_click` | `0.5` | level of the key-up click |
| `hold` | `0.09` | seconds before the key-up click |
| `center` | `0.0` | stereo position |
| `spread` | `0.2` | random pan spread per key |

#### `Tap`

A fingertip on paper or wood: pitchless, very short and dry. A few milliseconds of shaped noise and a damped knock below 1 kHz that is over before it can ring (no tuned resonance, no 'ding').

| 参数 | 默认 | 说明 |
|---|---|---|
| `material` | `0.5` | 0 paper (a soft papery brush) .. 1 wood (a duller knock) |
| `tone` | `1600.0` | centre of the tap's colour (Hz) |
| `length` | `0.012` | length of the contact noise (s) |
| `center` | `0.0` | stereo position |
| `spread` | `0.15` | random pan spread per tap |

#### `Breath`

Breath. shape='inhale': air drawn in, rising in level and colour, stopping at the written end (like a breath before a word); 'pulse': a short 'hu' (breath pulses of a crowd); 'swell': in and out over the note. With a pitch, the air is coloured by a soft resonance there (and an octave above), like air across the mouth of a pipe: a hint of pitch, no tone. Without a pitch the colour is set by `formant`.

| 参数 | 默认 | 说明 |
|---|---|---|
| `shape` | `'pulse'` | inhale / pulse / swell |
| `attack` | `0.015` | pulse / swell: onset time (s) |
| `decay` | `0.08` | pulse: decay time constant (s) |
| `formant` | `1100.0` | centre of the breath colour (Hz) when no pitch is given |
| `rise` | `7.0` | inhale: semitones the colour rises across the note |
| `resonance` | `0.5` | 0 plain air .. 1 strongly coloured by the pitch (still no tone) |
| `brightness` | `0.4` | 0 dark (mouth) .. 1 airy (teeth, hiss) |
| `center` | `0.0` | stereo position |
| `width` | `0.3` | stereo width |

#### `Sparkle`

Rising granular glitter over the written duration; pitch = lowest grain pitch.

| 参数 | 默认 | 说明 |
|---|---|---|
| `density` | `60.0` | grains per second |
| `span` | `24.0` | semitones the grain cloud rises across the note |
| `scale` | `(0, 2, 4, 7, 9)` | pitch classes offsets for grains (pentatonic by default) |
| `width` | `0.9` | stereo width |

#### `FlamePop`

Soft 'puff' of a flame catching: low breathy burst, quick downward filter, a few crackles.

| 参数 | 默认 | 说明 |
|---|---|---|
| `size` | `0.5` | 0 small .. 1 large |
| `crackle` | `0.4` | crackle amount |
| `center` | `0.0` | pan |

#### `Puff`

Flame going out: a short falling breath.

| 参数 | 默认 | 说明 |
|---|---|---|
| `length` | `0.25` | s |
| `center` | `0.0` | pan |

#### `Crunch`

Step on frost / grass / snow: dense micro-cracks shaped by the material.

| 参数 | 默认 | 说明 |
|---|---|---|
| `material` | `'frost'` | frost / grass / snow |
| `length` | `0.16` | s |
| `center` | `0.0` | pan |

#### `Hiss`

Moving hiss for light travelling along a road: pans from -> to with a Doppler-like filter.

| 参数 | 默认 | 说明 |
|---|---|---|
| `pan_from` | `-0.8` | start pan |
| `pan_to` | `0.8` | end pan |
| `centre_hz` | `4000.0` | band centre |
| `doppler` | `0.5` | filter sweep amount as it passes |

#### `FeltPiano`

Felt piano: a felt-muted hammer on unison strings (modal synthesis). Warm and dark, with a soft onset, a long two-stage decay (prompt sound, then aftersound) and the soft 'thock' of the felt mechanism. Its partials are harmonic (slightly stretched), so it reads as a struck string, not as a bell. pitch = the string; C2-C6.

| 参数 | 默认 | 说明 |
|---|---|---|
| `felt` | `0.7` | felt mute 0 bare hammer .. 1 thick felt: darker, softer onset, more mechanism noise |
| `decay` | `1.0` | multiplies the natural T60s (C4: aftersound ~14 s, prompt sound ~4 s) |
| `strings` | `3` | unison strings 1..3 |
| `detune` | `0.7` | unison mistuning std (cents): the slow beating of a real unison |
| `hammer_pos` | `0.12` | strike point as a fraction of the string (1/8 is the classic) |
| `inharm` | `1.0` | scales the string stiffness (stretch of the upper partials) |
| `body` | `0.45` | soundboard resonance level |
| `noise` | `0.5` | mechanism noise: felt on the string, hammer shank, key bed |
| `let_ring` | `True` | True: pedal down, the note rings on; False: the damper falls at the written end |
| `damp_time` | `0.18` | T60 (s) once the damper is down |
| `max_ring` | `12.0` | cap on ring length (s) |
| `spread` | `0.3` | stereo placement by pitch (low left, high right, as the player hears it) |
| `center` | `0.0` | stereo centre |

#### `TautString`

One long bowed string of light being drawn taut (a sound-design voice): starts flat and slack, tightens to the written pitch, its hum beating faster and getting brighter as the tension rises.

| 参数 | 默认 | 说明 |
|---|---|---|
| `slack_cents` | `45.0` | how far below the written pitch the string starts (cents) |
| `tighten` | `0.92` | fraction of the note at which it reaches the written pitch |
| `beat_from` | `0.25` | beat rate between the two copies at the start (Hz) |
| `beat_to` | `3.5` | beat rate once taut (Hz) |
| `bright_from` | `0.15` | brightness at the start 0..1 |
| `bright_to` | `0.7` | brightness once taut 0..1 |
| `beta` | `0.13` | bow position as a fraction of the string: 0.1 bright and peaky .. 0.3 hollow and round |
| `copy2` | `1.0` | level of the second, beating copy (lower = shallower beats, rounder peaks) |
| `bow` | `0.35` | bow friction noise 0..1 |
| `flutter` | `0.25` | amplitude flutter at full tension 0..1 |
| `attack` | `1.2` | onset time (s): the string materialises out of the dark |
| `release` | `0.25` | release time (s) |
| `center` | `0.0` | stereo position |
| `width` | `0.25` | stereo distance between the two copies |


## 7. DSP 参考（`filmsynth.dsp`）

| 模块 | 内容 |
|---|---|
| `filters` | `rbj/biquad/eq`（RBJ cookbook：lowpass, highpass, bandpass, notch, peak, lowshelf, highshelf, allpass），`butter`，`svf`（Zavalishin TPT，可逐样本变截止），`onepole_lp`，`modal_bank`（并联谐振器），`fir_from_modes`，`dc_block` |
| `reverb` | `make_ir(kind, rt60, predelay, …)`：room / chamber / hall / cathedral / plate，代码合成（预延迟 + 早期反射 + 频率相关衰减的 L/R 去相关噪声尾 + 回声密度渐增）；`Reverb`（母线用，只出湿声，含输入高低切）；`convolve`（FFT 卷积，带交叉馈送）；`reverb(x, kind, wet, dry)` |
| `delay` | `FeedbackDelay`（立体声 / 乒乓、环路低通 + 高通 + 软饱和、反馈与低通可随时间变化、慢速 wow）；`persistence_delay`（一条 0..1 的「保留度」曲线 p 驱动回声：反馈 `0.15 + 0.45·p`，环路低通 `1800 + 3200·p` Hz，p 低时更暗、更快消失；`lp_scale` 让回声整体更暗）；`vardelay`（三次 Hermite 变延迟） |
| `dynamics` | `saturate`（4× 过采样 tanh / tape / soft）；`Compressor`（RMS、软拐点、侧链）；`duck`；`limit`（前瞻、4× 过采样峰值检测）；`true_peak_db`；`lufs`；`normalize_lufs`；`master_chain` |
| `stereo` | `pan`（等功率，可逐样本）；`width`（M/S）；`haas`；`correlation`；`side_ratio_db` |
| `automation` | `Swell`（增音箱：增益 + 低通）、`GainCurve`（dB 曲线）、`FilterSweep` |
| `inserts` | `EQ`、`Saturate`、`Width`、`Haas`、`Gain`、`Limiter`、`DCBlock` |
| `modal` | `render_modes`（加法模态核：每个分音各自衰减、绽开、音高下坠、声像）、`doublets` |

## 8. `events.json` 规范（画面怎么读它）

`Session.render()` 在输出目录写 `events.json`，这是配乐交给画面的**唯一时间来源**。时间一律是从 1.1 起算的秒，`frame = t × fps`（可以有小数）。

```text
version, sampleRate (48000), fps, bpm, beatsPerBar, bars, tempoRamps, end, duration
beats:      [t0, t1, ...]                     # 每拍起点（秒），下标 0 = 1.1，共 bars × beatsPerBar + 1 个（含曲终）
barTimes:   [{bar, t, frame}]                 # 含 bars + 1 小节的起点（= 曲终）
beatTimes:  [{bar, beat, t, frame}]
markers:    [{name, pos, t, ...}]             # s.marker()；mark_sync() 写时间线的同步点（kind: hard_sync | soft_sync）
cuts:       [{start, end, choke, tails}]      # s.cut()
events:     [{t, frame, type, pos, ...}]      # s.event()，按 t 排序；type 和其余字段由片子的乐谱约定
notes:      [{track, stem, pos, beat, start, end, frame, vel, pitch, midi, hz, tag?}]
curves:     {name: {hop, t0, values}}         # s.export_curve(name, curve, hop=0.05)
measured:   {...}                             # 可选：片子的自检从成品音频量出来的同步表等
```

- `beat` 是名义网格位置（0 起算），`start` 是带人性化偏移的实际起点。合奏乐器每个演奏者另有 ±几十毫秒的个人偏差，不进 `events.json`。
- 事件 `pos` 是名义位置；`t` 是实际时刻（事件挂在音符上时取音符的实际起点）。
- `markers` 的实际字段：`mark_sync()` 写 `{name, pos, t, kind, picture}`（`picture` 取自时间线的 `sync`），区间同步点另带 `to`；`s.marker()` 写的设计位置可以带任意字段，例如示例片涌起的峰是 `kind: "lead"`，带 `frames`（相对同步点的帧数）与 `of`（哪个同步点）。
- `music/sampler` 写的 `events.json` 顶层字段与这里相同，`markers`、`curves`、`notes` 的细节不同，两边逐字段的对照见 [`docs/TIMELINE.md`](../../docs/TIMELINE.md) 第 10 节。

**画面这一侧的约定**（`engines/beat3d` 按这个读）：

1. 构建时读 `<片>/out/music/events.json`，校验 `beats[i]` 与时间线网格 `i × 60 / bpm` 的差 ≤ 1 ms，超了就报错——画面绝不能悄悄和音乐错开。
2. 只用 `beats` 和 `events`：`events` 里每个 `{t, type}` 变成帧 `t × fps`，镜头程序按 `type`（和片子约定的字段，如 `shot`、`side`、`vel`、`bright`）取用。
3. 文件不存在时画面走拍网格（与有配乐时到毫秒是同一件事），并且无声出片；配乐出来后重新构建即可。
4. `markers`、`notes`、`curves` 是给人和自检看的，画面可以不读。

一部片的事件类型写在它的乐谱设计里，例如示例片的 `pulse`（`shot side light div vel bright pitch`）、`chime`、`chord`、`motif`、`run`、`release`、`impact`、`sparkle`、`taut`、`land`、`cut`，见 `films/demo-light/score/DESIGN.md` 第 4 节。

## 9. 自检工具

### 9.1 `filmsynth.check`：对渲染结果的通用检查

全部从文件读（片的 `timeline.json` 加 `score.wav`、`stems/*.wav`、`events.json`），所以任何片子都能用；片子自己的自检只加它特有的判断（示例：`films/demo-light/score/selfcheck.py`）。

| 函数 | 量什么 |
|---|---|
| `load(out_dir)` | 读成 `Render(dir, mix, stems, events)` |
| `levels(mix, stems, onsets)` | 整合响度、真峰值、采样峰值、削波样本数、疑似咔嗒（起音处不算）、直流、左右相关；有分轨时还有限幅器的最大压缩量 |
| `loudness(mix, tl)` | 整合响度对时间线目标（±0.5 LU）、真峰值对上限；时间线有 `loudness.keys` 时逐点比 1 s 响度（±3 dB，硬同步点上的点按其后 1 s 判）；每段（`sections`）的 1 s 响度与整合响度 |
| `ffmpeg_loudness(path)` | 同一文件让 ffmpeg 的 `ebur128=peak=true` 再量一遍（独立于 pyloudnorm 与库内真峰值表的实现）；ffmpeg 路径：环境变量 `FFMPEG`，否则 PATH，否则常见安装位置 |
| `events_check(ev, tl, duration)` | 事件表校验：fps / bpm / 拍号与时间线一致；每拍、每小节在网格上（≤ 1 ms）；`frame = t × fps`；事件有 `t type pos`、按时间排序、在文件内；markers 的 `t` 与 `pos` 一致；时间线的每个同步点都有 marker；渲染长度等于 `totalFrames` |
| `on_grid(times, tl, division)` | 一串时刻是否在（细分后的）网格上 |
| `sync_check(r, tl, spec)` | 每个同步点上音乐落在哪：`onset`（某分轨的第一个毫秒达到局部峰值 −20 dB，适合同步前静默的层）、`rise` / `drop`（±窗口内最大的 50 ms 阶跃，适合一直在响的层）、`bands`（sub / low / mid / high 四个频带各自最大的 50 ms 阶跃，取变化最大的那个频带）；可限频带、K 加权；给出毫秒和帧的偏差与判定。没写 spec 的点用混音的 `bands`（切点上用 `drop`），硬同步点 ±1 帧算过——这只说明「音乐在这一帧变了」，片子自己的自检应当指明每个同步点由哪一层造成 |
| `dry_cut_check(mix, stems, t)` | 干切之后：干声分轨是否 < −120 dBFS、混响分轨还在响多少、混音掉了多少 dB、尾巴多久降到 −60 / −70 LUFS |
| `harmony(ev, stems)` | 和声层每拍中点在响的音级（从 `notes` 算）、换和弦的位置与间隔 |
| `chord_fit(ev, chart, stems)` | 每个音对一张和弦表：写的时值覆盖的每一拍，音级是否属于当时的和弦；列出不合的音 |
| `energy_follow(mix, tl)` | 1 s 响度对时间线的意图曲线 `energy`：Pearson / Spearman 相关、直线拟合 `L = a + b·energy`、残差 |
| `stem_levels(stems, tl)`、`audibility(r, pick)` | 每个分轨每小节的响度；画面跟的事件在其后 250 ms 内比其余混音高多少（全频带、300–3000 Hz、只比干声） |
| `figures(r, tl, out)` | `report/loudness.png`（1 s 响度、时间线的 energy 或 loudness 目标、同步点）、`report/spectrogram.png` |
| `compare(dir_a, dir_b)` | 两次渲染逐样本比较（混音与每个分轨按 24-bit 整数、sha256、`events.json` 去掉 `measured`） |
| `run(film)` / `write_report(...)` | 以上通用项一次跑完，写 `selfcheck.json` 与 `CHECK.md` |

命令行：

```sh
.venv/bin/python -m filmsynth.check run --film ../../films/demo-light      # 通用检查，退出码 1 = 有不通过的项
.venv/bin/python -m filmsynth.check compare DIR_A DIR_B                     # 退出码 0 = 逐样本相同
```

`run` 的参数：`--film`（片目录或 `timeline.json`；不给时先看 `$FILM_DIR`，再从当前目录往上找）、`--out`（渲染目录，默认 `<片>/out/music`）、`--no-ffmpeg`（跳过 ffmpeg 的 ebur128 复核）、`--no-write`（只打印汇总，不写 `selfcheck.json` 与 `CHECK.md`，适合在别人的产物上只读地查一遍）。

确定性的标准做法：同一份谱用不同的 `--workers` 各渲染一次（或隔天再渲染一次），`compare` 两个目录。

### 9.2 `filmsynth.analysis`：信号级的工具

- `qc(x, onsets)`：峰值、真峰值、积分响度、1 s 响度最大最小值、峰值因数、直流、左右相关、高频比例、咔嗒检测、首尾样本；
- `alias_ratio`（稳态音的混叠）、`spectrogram_png`、`loudness_curve`（1 s K 加权）、`plot_loudness(x, path, target_keys, s.t)`、`detect_clicks`、`k_weight`。

## 10. 例子（`examples/`）

移植自原库的乐器演示，各写一个 WAV、一张频谱图和一份 QC（`examples/out/`，不入库；环境变量 `FILMSYNTH_OUT` 可改目录）：

| 文件 | 内容 | 本机耗时 |
|---|---|---|
| `pluck.py` | B 羽上的问答动机、吟猱、上滑、泛音、低音 B1、止音，以及保留度从「忘了」到「记得」的回声 | 约 3 s |
| `choir.py` | 远处的四部合唱（每部 12 人）：mm 哼唱打开到 oo、ah，女高连奏一句，落在 D 宫和弦 | 约 10 s |
| `textures.py` | 转场工具箱：调到 B 的夜风、渐强的风雪被一刀切断、静默、反向涌起接气流、D 宫闪光、拍频长音、上升音 | 约 4 s |
| `calibrate.py` | 每件乐器单音的电平校准、速度、首尾、直流、咔嗒，振荡器与成品乐器的混叠（`calibrate.json`） | 约 10 s |

```sh
.venv/bin/python examples/pluck.py
```

## 11. 性能

- 本机（Apple 芯片，15 核）：示例片 24 s 的配乐，32 个音轨、3 条母线，默认进程数约 6 s，4 个进程约 7.5 s，1 个进程约 18 s，三者逐样本相同。
- 合唱最慢，12 人一声部约 0.2× 实时（单核）；弦乐 0.1–0.2×；其余乐器都在 0.04× 以下。
- 总耗时由最慢的单轨决定。长的合唱或弦乐声部可以拆成两轨（例如前后两段）来并行。
- 内存：每轨一份全长的立体声 float64（每分钟约 46 MB），主进程会同时持有各分轨。

## 12. 已知短板与边界

- **合唱。** 共振峰合成没有辅音，也不能唱词；适合远处铺底和元音长音，近距离独唱会露出合成感；高音区（女高 A5 以上）的「ah」靠 F1 追随基频，音色会变窄。
- **弦乐。** 没有真实的换把和换弦过程模型（只有滑音和弓的变化）；独奏暴露时不够真实，适合声部铺底和织体；没有拨奏（pizz）。
- **拨弦。** 弦与弦之间没有共鸣耦合；泛音是把激励频谱里非 n 倍的成分去掉得到的近似。
- **管风琴。** 风压只是共用的微小起伏，没有大和弦时的风压下沉；簧管是通用的 Trompete 频谱；32′ 的基频只有大音箱和耳机听得到。
- **钟与金属。** 模态是线性的，只有锣模拟了能量往高处「绽开」；镲片渐强的质感靠密集谐振器加噪声近似，近听偏「洗」。
- **鼓。** 太鼓没有双面皮耦合；`Footsteps` 是统一的软鼓加土声，不分地面；`HeartDrum` 的手掌闷音是固定的阻尼。
- **箫。** 射流驱动的波导是简化模型，没有真实的指法和叉口；超吹靠写高音；很快的经过句会露出合成感。
- **音效。** 都是设计出来的近似，音色偏干净。
- **自检的边界。** 都是客观指标（混叠、咔嗒、直流、电平、相关度、响度、时间），**不能代替人耳**；每版都要有人戴耳机听。咔嗒检测会报「有意的瞬态」：按键抬起、冰粒、碎响；单独渲染的 `TautString` 在两份拷贝拍频的峰上（两个 Helmholtz 波形的拐角对齐）也会被报出来，混进音乐里就不报了。
- **逐样本可复现的边界。** 实测的是：同一台机器、同一锁定版本、不同进程数、不同次运行，结果逐位相同。不同 CPU 架构或操作系统之间没有实测：numba 的 `fastmath`、SIMD 与 FFT 实现可能让最后几位不同（听不出来，但 sha256 会变）。要跨机器核对，比较 `compare` 的 `max_abs_diff` 而不是哈希。

## 13. 目录

```
music/filmsynth/
  pyproject.toml            包定义（Python ≥ 3.12）
  requirements.lock.txt     锁定版本（最后一行 -e .）
  filmsynth/
    core.py                 采样率、音高、种子、包络
    session.py              Session：速度、音轨、母线、标记、曲线、切点、并行渲染、events.json
    film.py                 片子的时间线：找到它、网格、曲线、同步点、响度目标
    check.py                渲染结果的通用自检（第 9 节），python -m filmsynth.check
    analysis.py             信号级的 QC 与图
    dsp/                    滤波、混响、延迟、动态、立体声、自动化、插件、模态
    instruments/            乐器（python -m filmsynth.instruments 打印参数表）
  examples/                 乐器演示与校准（输出到 examples/out/，不入库）
```

## 14. 来历与相对原库的改动

库的代码来自《成路》的配乐库，经《一拍》扩展（两份逐文件对比过：除下面几处外字节相同，《一拍》的版本是超集）。用同一段测试会话（31 件乐器、默认参数、混响母线、保留度回声、硬切、主音量）分别在《成路》原库、《一拍》的库和本库渲染，三者的 float64 混音逐位相同、32 个分轨的 24-bit 整数全同。

- 《一拍》加的（保留）：`FeltPiano`；`TautString`；`Pluck.body_seed`、`AirRush.whistle`（默认值下与原来逐样本相同）；`s.cut(..., tails=True)` 干切；`s.event()`；`events.json` 的 `beats`、`events`、`cuts`。
- 本库改的：
  - `film.py` 不再写死任何片子的路径与同步点：从任意片目录找 `timeline.json`（参数、`FILM_DIR`、当前目录往上），新增 `find_timeline`、`film_dir`、`music_out`、`check_grid`、`tl_beat`、`tl_seconds`、`tl_frame`、`film_end`、`loudness_target`；不能展开的区间同步点带 `to`。原来写死的 12 个同步点与 `spec/timeline.json` 路径删去。
  - `Session.from_timeline()`，会话记住 `session.timeline`。
  - 新增 `check.py`（第 9 节，从两部片的自检脚本里抽出通用部分并改成读文件）、`python -m filmsynth.instruments`。
  - 注释和文档字符串里提到原片剧情、段落号的地方改成中性说法；代码行为不变。
