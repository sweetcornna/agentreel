# CREDITS · 采样的来源、许可与署名

本仓库**不含任何录音**。`fetch_samples.py` 按下表从原始出处下载到采样根（`$AGENTREEL_SAMPLES`，默认
`music/sampler/samples/`，git 忽略），下载前核对许可，每个文件的来源、许可证据和哈希写进
`<采样根>/fetch_manifest.json`。

只收 **CC0 / 公有领域 / CC-BY**。CC BY-SA、NC、ND 一律不收；许可读出来和下表记录的不一致时 `fetch_samples.py`
拒收（退出码 2）。**用到 CC-BY 素材的作品，公开时必须带上该条目下的「署名」文本**（片尾字幕、视频简介或随附的
CREDITS 都可以）；CC0 与公有领域不要求署名。

| 包（`--only`） | 内容 | 许可 | 公开时要署名 |
| --- | --- | --- | --- |
| `piano` | Salamander Grand Piano V3（Yamaha C5） | CC BY 3.0 | **要**（第 1 节） |
| `vcsl` | VCSL：打击乐、民谣乐器、Steinway B 三角钢琴 | CC0 1.0 | 不要 |
| `vsco2` | VSCO 2 CE：弦乐、木管、圆号、定音鼓、悬挂钹 | CC0 1.0 | 不要 |
| `field` | Wikimedia Commons 的野外录音 16 条 | CC0 / 公有领域 / CC BY 3.0 | 只有 `pine_forest_dobroide`、`cricket_gryllus_campestris` **要**（第 4 节） |
| `found` | Freesound：铅笔、纸、键盘、响指、敲击、水滴、吱呀声 | CC0 1.0 | 不要 |
| `objects` | Freesound：瓷碗、瓷盘、瓷杯、翅膀 | CC0 1.0 | 不要 |
| `vocalset` | VocalSet 1.1 真人声 | CC BY 4.0 | **要**（第 7 节） |

以下不是外部素材，由 `sampler/` 里的经典 DSP 代码生成：合唱（共振峰合成）、pad、合成乐队与合成器、各种合成音效
（纸、风、水滴、昆虫……）、混响冲激响应。本库不含也不调用任何神经网络或训练得到的模型。

## 1. 钢琴：Salamander Grand Piano V3（CC BY 3.0，须署名）

| 项 | 内容 |
| --- | --- |
| 作者 | Alexander Holm（录音，Yamaha C5，48 kHz / 24 bit）；SFZ 重建 kinwie；Retuned 版 Markus Fiedler |
| 许可 | CC BY 3.0（仓库 `LICENSE` 全文为 "Attribution 3.0 Unported"；`fetch_samples.py` 读该 commit 的 LICENSE 全文核对，GitHub API 的 spdx_id 作旁证） |
| 来源 | https://github.com/sfzinstruments/SalamanderGrandPiano ，commit `3382bf9496bba2486f5ab0de55a264d1dfc38404`，逐文件下载并核对 git blob SHA-1 |
| 原始出处 | https://archive.org/details/SalamanderGrandPianoV3 |
| 采样根 | `salamander/`（约 750 MB；`--range` 可以只取一段音域） |
| 改动 | 不改原文件；渲染时移调、力度层交叉、滤波（felt 模式）、弦共鸣 |

**署名**："Salamander Grand Piano V3" by Alexander Holm, licensed under CC BY 3.0 (https://creativecommons.org/licenses/by/3.0/); SFZ mapping from sfzinstruments/SalamanderGrandPiano.

`sampler/_piano_tuning.py` 是对这套采样的测量结果（`tools/measure_piano_tuning.py` 生成），属于派生数据，沿用上面的署名。

## 2. VCSL（Versilian Community Sample Library，CC0）

| 项 | 内容 |
| --- | --- |
| 作者 | Versilian Studios LLC（Sam Gossner）及贡献者 |
| 许可 | CC0 1.0（仓库 `LICENSE` 全文；GitHub API 的 spdx_id 为 `CC0-1.0`） |
| 来源 | https://github.com/sgossner/VCSL ，commit `c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e`，逐文件下载并核对 git blob SHA-1 |
| 采样根 | `vcsl/`；民谣乐器与钢琴转成无损 FLAC（采样率、位深不变），实测音高、起音、电平写进 `vcsl/index_folk.json`；打击乐保留 WAV |

| 套（`--only vcsl:套`） | VCSL 源文件夹 | 文件数 |
| --- | --- | --- |
| woodblock、hand_chimes、tubular_bells、gong、slit_drum | Idiophones/Struck Idiophones/ 下的 Woodblock、Hand Chimes、Tubular Bells 1、Gong 1、Slit Drum | 10、19、18、7、6 |
| strumstick | Chordophones/Composite Chordophones/Strumstick/Finger | 57 |
| folk_harp | Chordophones/Composite Chordophones/Folk Harp | 58 |
| dan_tranh、dan_tranh_gliss | Chordophones/Zithers/Dan Tranh/Normal、Gliss | 48、12 |
| grand_piano | Chordophones/Zithers/Grand Piano, Steinway B/Sus（近距离话筒，踏板踩下，全音间隔，3 个力度层） | 126 |
| marimba、balafon | Idiophones/Struck Idiophones/Marimba、Balafon/Soft Mallet | 30、18 |
| cajon、claps、cabasa、tambourine、hihat、shaker_small、shaker_large | Idiophones/Struck Idiophones/ 下的同名文件夹 | 18、10、6、7、15、16、6 |
| frame_drum、bongos、conga | Membranophones/Struck Membranophones/ 下的同名文件夹 | 18、28、34 |

grand_piano 套是 agentreel 新加的（2026-10-21 下载 126 个 WAV 共 770 MB，逐个核对 SHA-1）：文件名就是实际发声音高
（实测，不像民谣乐器那样低一个八度）；C3–C6 的实测音准中位 2.4 音分、最大 11 音分，最低的 A#0、C1 偏 -10 到 -49 音分；
F#1、G#1（各 3 层）和 F#7 一层共 7 个文件实测音高与文件名不符（标为 pitch_suspect），G#6 以上另有 8 个文件测不出音高，
这些都按文件名用。

CC0 不要求署名；如愿致谢："Versilian Community Sample Library" by Versilian Studios (CC0)。

## 3. VSCO 2 Community Edition（CC0）

| 项 | 内容 |
| --- | --- |
| 作者 | Versilian Studios：录音 Sam Gossner、Simon Dalzell；切片 Elan Hickler / Soundemote（据仓库 README） |
| 许可 | CC0 1.0（仓库 `LICENSE`；GitHub API 的 spdx_id 为 `CC0-1.0`） |
| 来源 | https://github.com/sgossner/VSCO-2-CE ，commit `440300901dfe9275fd84e0b7763af1f8443ae62e`，逐文件下载并核对 git blob SHA-1 |
| 取用范围 | Violin / Viola / Cello Section 的 sustain（vib）、pizz、spic；Solo Contrabass 的 SusVib、SusNV、Pizz、Spic；Solo Violin 的 Arco Vib、Pizz；Flute、Oboe、Clarinet 的长音与断奏，Bassoon 断奏，F Horn 长音；Timpani 单击与滚奏；VSCO 1 Percussion 的悬挂钹 |
| 采样根 | `vsco2/<声部>/<奏法>/*.flac`（无损，位深不变）、`vsco2/perc/timp/`、`vsco2/perc/cym/`；`vsco2/index.json` 含实测音高与稳定段音准 |
| 改动 | 变速移调、力度层交叉、延音段交叉延长、连弓交叉淡化；泛音音色由延音采样经 WORLD 改造（不是录音） |

CC0 不要求署名；如愿致谢："VSCO 2: Community Edition" by Versilian Studios (CC0)。

## 4. 野外录音：Wikimedia Commons

下载前用 Commons API 读每个文件页面的 LicenseShortName，与下表一致才下载（并核对文件大小）。`sampler/prepare/field.py`
处理成 48 kHz FLAC，每条的处理参数写进 `field/manifest.json`：ffmpeg 解码 → 截取 → 左右相同时并成单声道 →（个别）三次样条
修复削波 → 重采样到 48 kHz → 20 Hz 零相位高通 →（个别）额外高通、去咔哒 → 首尾 5 ms 淡入淡出 → 峰值 ≤ -0.2 dBFS → 24 bit FLAC。

| 采样根里的文件 | Commons 文件 | 作者 | 许可 |
| --- | --- | --- | --- |
| `stream_molln_bachlauf` | File:2024-07-26 Molln (Oberösterreich) Bachlauf plätschert (Krumme Steyerling bei Piesslingerstraße).wav | DrTrumpet | CC0 |
| `rivulet_molln_rinnsal` | File:2024-07-26 Molln (Oberösterreich) Rinnsal plätschert bei der Wiese im Wasserschutzgebiet Bräugrabenstraße.wav | DrTrumpet | CC0 |
| `creek_forest_lawn` | File:Forest lawn creek.ogg | Dsw4 | 公有领域 |
| `river_shallow_stony` | File:Shallow small river with stony riverbed.ogg | stephan | 公有领域 |
| `forest_ambience_nille` | File:20090610 0 ambience.ogg | nille | 公有领域 |
| `pine_forest_dobroide` | File:Dobroide - 20060824.forest03 (cc-by) (freesound).wav | dobroide | **CC BY 3.0** |
| `birds_fontainebleau` | File:Birds forest.ogg | Barracuda1983（页面标注作者为 "assumed"） | 公有领域 |
| `birds_garden_sparse` | File:Birds singing in garden.ogg | ezwa | 公有领域 |
| `birds_evening_waidachswald` | File:Waidachswald Oberschefflenz Abend 20250616 2005.ogg | Evillan | CC0 |
| `cricket_gryllus_campestris` | File:BioAcoustica 991 1 Gryllus campestris 142r1.flac | Natural History Museum, London（经 BioAcoustica） | **CC BY 3.0** |
| `rooster_chardonnet` | File:Coq qui chante (DenisChardonnet).wav | DenisChardonnet | CC0 |
| `rooster_alys_medium` | File:Medium rooster crowing.ogg | alys | 公有领域 |
| `rooster_alys_small` | File:Small rooster crowing.ogg | alys | 公有领域 |
| `dawn_chorus_dorset` | File:Dawnchorus-uk.ogg（取 10:00–17:00，额外 300 Hz 高通、去咔哒） | Silas S. Brown | 公有领域 |
| `dog_bark_edo` | File:Ladrido perro.ogg（源文件 38 个削波样本，三次样条修复） | Edo.pt2 | CC0 |
| `dog_bark_george` | File:George vuf 1996.ogg | Broadbeer | 公有领域 |

文件页面都在 `https://commons.wikimedia.org/wiki/<Commons 文件名，空格换成下划线>`。

**署名（pine_forest_dobroide）**："20060824.forest03" by dobroide (https://freesound.org/people/dobroide/sounds/22384/), via Wikimedia Commons, licensed CC BY 3.0 (https://creativecommons.org/licenses/by/3.0/). Changes: resampled to 48 kHz, 20 Hz high-pass, cut into random segments and re-joined with crossfades, loudness-normalised.

**署名（cricket_gryllus_campestris）**："Gryllus campestris, BioAcoustica 991-1 (142r1)", recorded 1975-06-11, (c) Natural History Museum (London), via BioAcoustica (http://bio.acousti.ca/node/11117) and Wikimedia Commons, licensed CC BY 3.0 (https://creativecommons.org/licenses/by/3.0/). Changes: first 44.5 s removed, resampled to 48 kHz, re-segmented, pitch-shifted +-6 %, filtered, layered as four individuals.

`field/bird_calls.json`（鸟鸣单句目录）是对上面四条鸟鸣录音（均为公有领域 / CC0）的分析结果，许可沿用源文件。

看过但没有采用的候选：`File:Dog.ogg`（没有吠叫）、`File:Bark (avk).wav` 与 `File:Dog in Antefasy.wav`（其实是读单词的发音录音）、
PLOS ONE 的狗低吼刺激音（重合成的）、`Upbeat huamei bird calling.oga`（4–10 kHz 有持续噪声带）、
`Reinsamba - evening in the forest`（2068 个削波样本）、`Country night noise.ogg`（带宽只有 7 kHz）；
所有 CC BY-SA / NC / ND 文件在检索阶段就排除了，其中包括 xeno-canto 上的大多数鸟鸣。

## 5. 画面里那些东西的声音：Freesound（`found`，全部 CC0）

下表每一条都在录入时（2026-09-27 / 09-29）逐条在 Freesound 页面上读到 "Creative Commons 0"。`fetch_samples.py` 不抓网页：
设了 `FREESOUND_API_KEY` 时用官方 APIv2（`/apiv2/sounds/<编号>/`）读 license 字段在线复核（必须是 CC0，否则拒收），预览地址
也取自 API；没设时直接下载目录里记录的 HQ 预览地址，许可以这里的记录为准，并打印「未在线复核许可」。两种方式都核对记录的
SHA-256（预览是 Ogg Vorbis；原始 WAV 要登录才能下），逐字节存进 `found/`，来源与复核方式记在 `found/manifest.json`。
目前记录了预览地址的是 `pencil_writing_rivernile7`、`scribble_long_waldram`、`sketchbook_flips_owlstorm`（和第 6 节的
`cup_ceramic_that_finn`），其余条目要有 API key 才能取。
`sampler/found.py` 载入时把录音切成单个事件（单独听会咔哒的不选），或取一段连续发声当作一笔（`found.stroke`）。

| 文件 | Freesound 标题 | 作者 | 页面 |
| --- | --- | --- | --- |
| `pencil_writing_rivernile7` | Writing On Paper.wav | rivernile7 | https://freesound.org/people/rivernile7/sounds/234016/ |
| `pencil_writing_fthgurdy` | Writing with pencil 2 | fthgurdy | https://freesound.org/people/fthgurdy/sounds/376705/ |
| `scribble_long_waldram` | Scribble_long_strokes.wav | waldram | https://freesound.org/people/waldram/sounds/257517/ |
| `paper_rustle_keweldog` | rustling paper.wav | keweldog | https://freesound.org/people/keweldog/sounds/181774/ |
| `sketchbook_flips_owlstorm` | Paper Sketchbook Page Flips 1 | OwlStorm | https://freesound.org/people/OwlStorm/sounds/320148/ |
| `page_turn_davidbain` | Page Turn | davidbain | https://freesound.org/people/davidbain/sounds/136778/ |
| `keyboard_typing_trollarch2` | Keyboard Typing | Trollarch2 | https://freesound.org/people/Trollarch2/sounds/331656/ |
| `keyboard_typing_florianreichelt` | Keyboard Typing | florianreichelt | https://freesound.org/people/florianreichelt/sounds/448209/ |
| `finger_snaps_7146007` | Fingers Snapping Finger Snap | deleted_user_7146007 | https://freesound.org/people/deleted_user_7146007/sounds/383652/ |
| `finger_snaps_johnsonbrand` | Finger Snaps | JohnsonBrandEditing | https://freesound.org/people/JohnsonBrandEditing/sounds/369549/ |
| `wood_knock_mrguff` | Wood Knock | Mrguff | https://freesound.org/people/Mrguff/sounds/369710/ |
| `knuckle_hits_batchku` | Hit_knuckle-15_009.aif | batchku | https://freesound.org/people/batchku/sounds/10542/ |
| `water_drops_reiyamanor` | Water Drops.wav | ReiyaManor | https://freesound.org/people/ReiyaManor/sounds/351623/ |
| `water_drop_paespedro` | Single Water Drop | paespedro | https://freesound.org/people/paespedro/sounds/174718/ |
| `creak_slow_harveyism` | Slow Creaking Stereo.wav | harveyism | https://freesound.org/people/harveyism/sounds/130364/ |
| `rope_creaks_6polnic` | hamp rope creaks | 6polnic | https://freesound.org/people/6polnic/sounds/231438/ |

`finger_snaps_johnsonbrand`、`rope_creaks_6polnic` 单独听有咔哒，`knuckle_hits_batchku` 很轻，用之前先听一下。CC0 不要求署名。

## 6. 瓷器与翅膀：Freesound（`objects`，全部 CC0）

取法同第 5 节，存进 `objects/`，来源与 SHA-256 记在 `objects/manifest.json`。

| 文件 | Freesound 标题 | 作者 | 页面 | 用在 |
| --- | --- | --- | --- | --- |
| `bowl_ceramic_3dblanco` | Ceramic Bowl Taps and Scrapes | 3D_Blanco | https://freesound.org/people/3D_Blanco/sounds/848463/ | `porcelain` 的 bowl：4 下干净的敲击，基频 821 Hz，分音 2.36×、4.18× |
| `plate_porcelain_chilsville` | Plate ding.m4a | Chilsville | https://freesound.org/people/Chilsville/sounds/591575/ | `porcelain` 的 plate：571 Hz，2.12×、3.35× |
| `cup_ceramic_that_finn` | Light ceramic cling | that_Finn | https://freesound.org/people/that_Finn/sounds/706536/ | `porcelain` 的 cup：2667 Hz（最高的八度） |
| `wings_pigeon_kinoton` | Pigeon, Flies Away, Flapping Wings | Kinoton | https://freesound.org/people/Kinoton/sounds/689998/ | `nature.sfx('flutter')` 主音源 |
| `wings_bird_21100495` | bird wings flapping.wav | 21100495 | https://freesound.org/people/21100495/sounds/655322/ | `nature.sfx('flutter')` |
| `wings_pigeon_gerardcatala` | Beating wings of a pigeon | gerardcatala | https://freesound.org/people/gerardcatala/sounds/153277/ | `nature.sfx('flutter')` |

`porcelain` 的模态合成参数（频率比、电平、衰减时间）是对 bowl 录音的实测。CC0 不要求署名。

## 7. 真人声：VocalSet 1.1（CC BY 4.0，须署名）

| 项 | 内容 |
| --- | --- |
| 作者 | Julia Wilkins, Prem Seetharaman, Alison Wahl, Bryan Pardo |
| 许可 | CC BY 4.0（`fetch_samples.py` 读 Zenodo API 记录 1442513 的 license 字段，必须是 `cc-by-4.0`） |
| 来源 | https://doi.org/10.5281/zenodo.1442513 ，文件 `VocalSet11.zip`（HTTP range 请求只取需要的成员，不下载整包） |
| 取用范围 | 20 位歌手（目录 female1–9、male1–11）的 long_tones（straight / pp / forte / messa）、scales 与 arpeggios（straight / breathy / vibrato / slow_piano）、excerpts（straight / vibrato）；长音、音阶、琶音都有 a e i o u 五个元音 |
| 采样根 | `vocalset/<歌手>/<语境>/<技法>/*.flac`（无损转 16 bit FLAC）；派生数据 `f0.npz`（Harvest 基频）、`index.json`（音符单元与吸气片段） |
| 改动 | 渲染时切成音符单元，用 WORLD 分析再合成，改音高（多数小于 1 个半音）、拼接、交叉淡化、改电平 |

**署名**："VocalSet: A Singing Voice Dataset" by Julia Wilkins, Prem Seetharaman, Alison Wahl and Bryan Pardo (ISMIR 2018), https://doi.org/10.5281/zenodo.1442513, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Changes: cut into notes, resynthesised with the WORLD vocoder, re-pitched, spliced and mixed.

论文：J. Wilkins, P. Seetharaman, A. Wahl, B. Pardo, "VocalSet: A Singing Voice Dataset", Proc. 19th ISMIR, 2018。
（Zenodo 简介写 "9 male, 11 female"，数据里的目录是 female1–9、male1–11，以目录为准。）

## 8. 方法与数据引用（不是素材，列出以示来源）

- 元音共振峰频率 / 带宽：Csound Manual, Appendix "Formant Values"（soprano / alto 表）。
- LF 声门模型：Fant, Liljencrants & Lin (1985)；Rd 参数化：Fant (1995)。
- 气泡 / 水声：Minnaert (1933)；K. van den Doel, "Physically based models for liquid sounds" (2005)。
- 滤波器：R. Bristow-Johnson, "Audio EQ Cookbook"。响度：ITU-R BS.1770（pyloudnorm 实现）。
- 相位声码器 identity phase locking：J. Laroche & M. Dolson (1999)。
- 弦的非谐性 f_n = n f0 sqrt(1 + B n^2)：H. Fletcher (1964)。
- WORLD 声码器：M. Morise, F. Yokomori, K. Ozawa, IEICE Trans. Inf. & Syst. E99-D(7), 2016；CheapTrick (2015)、D4C (2016)、Harvest (2017)。
- HNR 自检的自相关法：P. Boersma (1993)。

## 9. 软件

numpy、scipy、soundfile（libsndfile）、pyloudnorm 用于运行，matplotlib 只给自检画图；ffmpeg 只用来解码 Commons
的音频和复核响度（`$FFMPEG` 或 PATH）。**pyworld**（MIT）封装的 **WORLD**（M. Morise，修改版 BSD 许可，
https://github.com/mmorise/World ）是经典信号处理声码器（基频估计 + 频谱包络 + 非周期成分），不含神经网络。
pyworld 0.3.x 导入时要 `pkg_resources`，所以依赖里有 setuptools<81。压缩、限幅、混响、移调、母带链都是本库自己实现的。
