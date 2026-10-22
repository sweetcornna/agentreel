# 文档

agentreel 的做法：一个负责人加一组智能体，从一句话做到多规格交付的短片。这里的文档写的是流程、时间线约定和验收方法；具体命令见各引擎与配乐库目录里的 README。

## 阅读顺序

1. [METHOD.md](METHOD.md)：制作方式。角色、流程、版本轮次、只改该改的帧、交接、反馈怎么落地、常见坑、新片的最小清单。
2. [TIMELINE.md](TIMELINE.md)：`timeline.json` 的字段、位置写法与帧换算、画幅与能量曲线、镜头与同步点、配乐导出的 `events.json`。用 `films/demo-light/` 和 `films/demo-paper/` 两条示例时间线讲。
3. [QA.md](QA.md)：每道闸量什么、阈值多少、为什么：确定性、切点对账、相邻帧跳变、逐帧 PSNR、色调、响度与真峰值、同步、字幕可读时长、光敏、交付检查，以及客观检查管不到的部分。

## 引擎、配乐库与示例片

命令、参数、平台注意事项都在各自的 README 里，这里的文档不重复：

| 目录 | 是什么 | 示例片 |
| --- | --- | --- |
| [engines/beat3d](../engines/beat3d/README.md) | 三维光影画面引擎（three.js，无头 Chrome 出帧） | [films/demo-light](../films/demo-light/README.md) |
| [engines/paper2d](../engines/paper2d/README.md) | 纸面二维画面引擎（React 场景，Remotion 风格 API 的垫片） | [films/demo-paper](../films/demo-paper/README.md) |
| [music/filmsynth](../music/filmsynth/README.md) | 纯代码合成的配乐库 | demo-light 的配乐：[score/DESIGN.md](../films/demo-light/score/DESIGN.md) |
| [music/sampler](../music/sampler/README.md) | 采样乐器 + 经典 DSP 的配乐库（采样另行下载） | demo-paper 的配乐：[score/README.md](../films/demo-paper/score/README.md) |

几个库共用的环境变量（每个变量的细节见对应 README）：

| 变量 | 谁读 | 作用 |
| --- | --- | --- |
| `CHROME_PATH` | beat3d、paper2d | Chrome / Chromium 可执行文件 |
| `FFMPEG` / `FFPROBE` | beat3d、paper2d（两者都读）；filmsynth、sampler（只读 `FFMPEG`） | ffmpeg / ffprobe |
| `BEAT3D_GPU`、`BEAT3D_GPU_FLAGS` | beat3d | 无头 Chrome 的 GPU 后端预设 / 直接给的后端参数 |
| `PAPER2D_GL`、`PAPER2D_CHROME_FLAGS` | paper2d | GL 后端（同 `--gl`）/ 额外的 Chrome 参数 |
| `FILM_DIR` | filmsynth | 没给 `--film` 时去哪里找片目录（beat3d、paper2d 用 `--film`） |
| `AGENTREEL_SAMPLES`、`AGENTREEL_SAMPLES_LOG`、`AGENTREEL_CHECK` | sampler | 采样根目录 / 记录打开过的采样 / 乐器自检的输出目录 |

## 模板

[templates/](templates/) 下五份，抄过去填：

| 文件 | 谁写 | 什么时候 |
| --- | --- | --- |
| [FILM_BIBLE.md](templates/FILM_BIBLE.md) | 主会话 | 开工，一句话之后 |
| [TASK.md](templates/TASK.md) | 主会话 | 每派一路 |
| [REPORT.md](templates/REPORT.md) | 子代理 | 每交一次 |
| [REVIEW.md](templates/REVIEW.md) | 主会话 | 每次打回或小改 |
| [HANDOFF.md](templates/HANDOFF.md) | 主会话 | 每一轮结束、换会话、代理中断时；幕间交接帧规格也写在这里 |

## 案例

三部用这套方法做出来的片子。文中只写方法、数字和取舍，不含各片的产品内容。

| 片子 | 长度 | 画面 | 配乐 | 对应的库 |
| --- | --- | --- | --- | --- |
| [《一拍》ONE BEAT](cases/yipai.md) | 93.3 s，24 fps | 三维光影，自写 WebGL 渲染器 | 全部代码合成 | `engines/beat3d`、`music/filmsynth` |
| [《成路》MADE BY WALKING](cases/chenglu.md) | 133.3 s，24 fps（做到 1080p 动态分镜） | 三维，七章七种笔法 | 全部代码合成 | `engines/beat3d`、`music/filmsynth` |
| [阡陌宣传片](cases/qianmo-promo.md) | R1–R20 加 U90–U90j 共三十轮，最后一版 89.967 s，30 fps | 纸面二维动画，React 场景 | 真实采样乐器 | `engines/paper2d`、`music/sampler` |

[images/](images/) 里是案例用到的静帧，每部三张，都是从各片的成片或动态分镜里按帧号抽出、缩到 1280 宽的 JPEG。
