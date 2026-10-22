"""采样放在哪里，缺了怎么提示。

采样根：设了 `$AGENTREEL_SAMPLES` 就用它，否则是本包旁边的 `music/sampler/samples/`（git 忽略）。
根下的布局（fetch_samples.py 写出）：

    salamander/  vcsl/  vsco2/  field/  found/  objects/  vocalset/

AGENTREEL_SAMPLES_LOG=<文件>：库打开的每个采样文件都以「相对采样根的路径」追加进 <文件>（每个进程每个文件一次）。
带着它把乐谱跑一遍，再 `fetch_samples.py --files <文件>`，取到的就是这部乐谱用到的全部录音、不多不少。
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]          # music/sampler/
DEFAULT_SAMPLES = PACKAGE_ROOT / 'samples'
SAMPLES = Path(os.environ.get('AGENTREEL_SAMPLES') or DEFAULT_SAMPLES).expanduser().resolve()

# 子目录 -> 由 fetch_samples.py 的哪个包提供
PACKS = {'salamander': 'piano', 'vcsl': 'vcsl', 'vsco2': 'vsco2', 'field': 'field', 'found': 'found',
         'objects': 'objects', 'vocalset': 'vocalset'}

_logged: set[str] = set()


def rel(path) -> str:
    """相对采样根的路径（POSIX 分隔符）；不在根下时原样返回。不跟随符号链接（根里的子目录可以是链接）。"""
    p = Path(os.path.abspath(path))
    try:
        return p.relative_to(SAMPLES).as_posix()
    except ValueError:
        pass
    try:
        return Path(path).resolve().relative_to(SAMPLES).as_posix()
    except ValueError:
        return str(path)


def log_access(path) -> None:
    """记下本进程打开过的一个采样文件（accessed() 可取），设了 AGENTREEL_SAMPLES_LOG 时同时追加进那个文件。"""
    r = rel(path)
    if r in _logged:
        return
    _logged.add(r)
    log = os.environ.get('AGENTREEL_SAMPLES_LOG')
    if log:
        with open(log, 'a', encoding='utf-8') as f:
            f.write(r + '\n')


def accessed() -> list[str]:
    """本进程到目前为止打开过的采样文件（相对采样根，排序）。"""
    return sorted(_logged)


class MissingSamples(FileNotFoundError):
    pass


def need(path, what: str = '') -> Path:
    """文件在就返回 `path`；不在就抛 MissingSamples，并说明该用 fetch_samples.py 的哪个包去取。"""
    p = Path(path)
    if p.exists():
        return p
    sub = rel(p).split('/', 1)[0]
    pack = PACKS.get(sub, sub)
    raise MissingSamples(
        f'采样缺失：{p}（{what or sub}）。采样根是 {SAMPLES}（环境变量 AGENTREEL_SAMPLES）。'
        f'先运行 `music/sampler/.venv/bin/python music/sampler/fetch_samples.py --only {pack}`（或用 --files 只取需要的文件），'
        f'见 music/sampler/README.md。')
