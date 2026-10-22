#!/usr/bin/env python3
"""取样：按 CREDITS.md 记录的来源把采样下载到采样根目录，下载前核对许可，下载后校验并导入。

    python music/sampler/fetch_samples.py --list
    python music/sampler/fetch_samples.py --only piano,vcsl            # 整包
    python music/sampler/fetch_samples.py --only vcsl:folk_harp,vsco2:vc/sus --range D2-A5
    python music/sampler/fetch_samples.py --files films/demo-paper/score/samples.txt
    python music/sampler/fetch_samples.py --only field --dry-run       # 只核对许可、列出文件

采样根目录：--dest，否则 $AGENTREEL_SAMPLES，否则 music/sampler/samples/（已被 git 忽略）。

只收 CC0 / 公有领域 / CC-BY，来源与核对方式：
  piano     GitHub sfzinstruments/SalamanderGrandPiano（固定 commit）  读该 commit 的 LICENSE 全文
  vcsl      GitHub sgossner/VCSL（固定 commit）                        LICENSE 全文 + GitHub license API
  vsco2     GitHub sgossner/VSCO-2-CE（固定 commit）                   同上
  field     Wikimedia Commons                                          API extmetadata 的 LicenseShortName
  found     Freesound（HQ 预览）                                       设了 FREESOUND_API_KEY：官方 APIv2 的 license 字段；
  objects   Freesound（HQ 预览）                                       没设：以本文件的记录为准（不抓网页）
  vocalset  Zenodo 记录 1442513（zip 用 HTTP Range 只取成员）          records API 的 metadata.license.id
许可读出来不在白名单、或与本文件记录的不一致，就拒收（退出码 2），一个字节也不写进采样根。
完整性：GitHub 文件逐个核对 git blob SHA-1；Freesound 预览核对 SHA-256；Commons 核对大小；zip 成员由 zipfile 核对 CRC-32。

选择方式（可以组合）：
  --only 包[:套],...   整包或整套；--list 列出全部包与套
  --range 低-高        有音高的套只取文件名音高（换算成实际发声音高）落在范围内的文件，两头各放宽 2 个半音
  --files 清单         精确文件清单：每行一个相对采样根的路径（# 开头为注释）。用 AGENTREEL_SAMPLES_LOG
                       跑一遍乐谱就能得到它（见 sampler/_paths.py）
下载完成后自动导入：VCSL 民乐套转 FLAC 并建 index_folk.json，VSCO 转 FLAC 并建 index.json，
Commons 录音解码、重采样并写 manifest（需要 ffmpeg：$FFMPEG 或 PATH），鸟鸣目录，VocalSet 音符单元索引。
每个文件的来源、许可证据和哈希记在 <采样根>/fetch_manifest.json。
退出码：0 完成；1 有文件没取到（例如 Freesound 条目没有记录预览地址、又没设 FREESOUND_API_KEY）；2 许可拒收。
环境变量：GITHUB_TOKEN（可选，放宽 GitHub API 速率限制）、FREESOUND_API_KEY（可选，Freesound 在线复核许可）。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = 'agentreel-sampler/0.1 (sample fetcher; licence check before download)'
ALLOWED = ('CC0-1.0', 'PD', 'CC-BY-3.0', 'CC-BY-4.0')

# ----------------------------------------------------------------------------- 目录（与 CREDITS.md 一致）
GITHUB = {
    'piano': dict(repo='sfzinstruments/SalamanderGrandPiano', commit='3382bf9496bba2486f5ab0de55a264d1dfc38404',
                  license='CC-BY-3.0', dest='salamander',
                  attribution='"Salamander Grand Piano V3" by Alexander Holm, licensed under CC BY 3.0 '
                              '(https://creativecommons.org/licenses/by/3.0/); SFZ mapping from '
                              'sfzinstruments/SalamanderGrandPiano.'),
    'vcsl': dict(repo='sgossner/VCSL', commit='c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e', license='CC0-1.0',
                 dest='vcsl', attribution=None),
    'vsco2': dict(repo='sgossner/VSCO-2-CE', commit='440300901dfe9275fd84e0b7763af1f8443ae62e', license='CC0-1.0',
                  dest='vsco2', attribution=None),
}
# Commons：库内名 -> (文件页标题, 记录的许可)
COMMONS = {
    'stream_molln_bachlauf': ('File:2024-07-26 Molln (Oberösterreich) Bachlauf plätschert (Krumme Steyerling bei '
                              'Piesslingerstraße).wav', 'CC0'),
    'rivulet_molln_rinnsal': ('File:2024-07-26 Molln (Oberösterreich) Rinnsal plätschert bei der Wiese im '
                              'Wasserschutzgebiet Bräugrabenstraße.wav', 'CC0'),
    'creek_forest_lawn': ('File:Forest lawn creek.ogg', 'Public domain'),
    'river_shallow_stony': ('File:Shallow small river with stony riverbed.ogg', 'Public domain'),
    'forest_ambience_nille': ('File:20090610 0 ambience.ogg', 'Public domain'),
    'pine_forest_dobroide': ('File:Dobroide - 20060824.forest03 (cc-by) (freesound).wav', 'CC BY 3.0'),
    'birds_fontainebleau': ('File:Birds forest.ogg', 'Public domain'),
    'birds_garden_sparse': ('File:Birds singing in garden.ogg', 'Public domain'),
    'birds_evening_waidachswald': ('File:Waidachswald Oberschefflenz Abend 20250616 2005.ogg', 'CC0'),
    'cricket_gryllus_campestris': ('File:BioAcoustica 991 1 Gryllus campestris 142r1.flac', 'CC BY 3.0'),
    'rooster_chardonnet': ('File:Coq qui chante (DenisChardonnet).wav', 'CC0'),
    'rooster_alys_medium': ('File:Medium rooster crowing.ogg', 'Public domain'),
    'rooster_alys_small': ('File:Small rooster crowing.ogg', 'Public domain'),
    'dawn_chorus_dorset': ('File:Dawnchorus-uk.ogg', 'Public domain'),
    'dog_bark_edo': ('File:Ladrido perro.ogg', 'CC0'),
    'dog_bark_george': ('File:George vuf 1996.ogg', 'Public domain'),
}
# Commons 页面上的署名写法（CC BY 的条目必须带，见 CREDITS.md）
COMMONS_AUTHOR_FIX = {
    'birds_fontainebleau': 'Barracuda1983 (PD-self; the page marks the author as "assumed")',
    'cricket_gryllus_campestris': 'Natural History Museum, London (copyright holder, via BioAcoustica)',
}
# Freesound：库内名 -> (声音编号, 用户名, 标题, HQ 预览的 SHA-256, HQ 预览地址或 None)
# 不抓 Freesound 的网页。设了 FREESOUND_API_KEY 时用官方 APIv2（/apiv2/sounds/<编号>/）在线读 license 字段核对、取预览地址；
# 没设时直接下载这里记录的预览地址，许可以 FREESOUND_LICENSE 的记录为准（2026-09-27/29 逐条在页面上核对过，见 CREDITS.md）。
# 记录了地址的是 2026-10-21 实际下载过、SHA-256 对得上的几条；其余为 None，只能经 API 取。两种方式都核对 SHA-256。
FREESOUND_LICENSE = 'CC0-1.0'
FREESOUND_API = 'https://freesound.org/apiv2/sounds/{sid}/?fields=id,username,name,license,previews'
FOUND = {
    'pencil_writing_rivernile7': (234016, 'rivernile7', 'Writing On Paper.wav',
                                  'b7a7731f581e4ab891354889a5bdff161033005079d442ee46135f29e2543839',
                                  'https://cdn.freesound.org/previews/234/234016_3735801-hq.ogg'),
    'pencil_writing_fthgurdy': (376705, 'fthgurdy', 'Writing with pencil 2',
                                'de5ac6292509b7c8fb63a2af57a051eae1b9c05cfc4820a01679f50622e53b9d', None),
    'scribble_long_waldram': (257517, 'waldram', 'Scribble_long_strokes.wav',
                              '6140fee0e70214b6540bbd269c070cbf8471ff0e6e93210c435cf61adffe7d5f',
                              'https://cdn.freesound.org/previews/257/257517_114036-hq.ogg'),
    'paper_rustle_keweldog': (181774, 'keweldog', 'rustling paper.wav',
                              '01910462a10f65b3c30252b19934ee6d515ecb774a21f61b54f59dd2937f4c63', None),
    'sketchbook_flips_owlstorm': (320148, 'OwlStorm', 'Paper Sketchbook Page Flips 1',
                                  '4595224dd8d172202b8f2075f56c465e640d9ee135be559c4282b4ad3d45274f',
                                  'https://cdn.freesound.org/previews/320/320148_140737-hq.ogg'),
    'page_turn_davidbain': (136778, 'davidbain', 'Page Turn',
                            '64f34f24618a5849a2e190ae9ff5b8979d2686b7fb29f926c7ae2f9e4f452fee', None),
    'keyboard_typing_trollarch2': (331656, 'Trollarch2', 'Keyboard Typing',
                                   'fd51262088908e95cedcc20471ee18a80cca003d3c417acb48c40006ec39b131', None),
    'keyboard_typing_florianreichelt': (448209, 'florianreichelt', 'Keyboard Typing',
                                        '3c8f565cd478b7be0e30e710c5ba9202361d9391a2bedbc865d9189fabe3ad8e', None),
    'finger_snaps_7146007': (383652, 'deleted_user_7146007', 'Fingers Snapping Finger Snap',
                             '6aeb18a5038a21da4ab4f78ac44e2f2410bd41322c884d97f3690596824b46cc', None),
    'finger_snaps_johnsonbrand': (369549, 'JohnsonBrandEditing', 'Finger Snaps',
                                  'fe6865f26c945919183027ce1896b5adff74e92f00e82f26f35712d03f8984d7', None),
    'wood_knock_mrguff': (369710, 'Mrguff', 'Wood Knock',
                          'c3ae566b236e1d4afc9a7ae70d05191b7f17ea37ecfe2ad6a75ed09e7f820f3a', None),
    'knuckle_hits_batchku': (10542, 'batchku', 'Hit_knuckle-15_009.aif',
                             'dca0e6d0ca56394a994e697056782cde7c2e0363044f4d4d5e70c68934e01d92', None),
    'water_drops_reiyamanor': (351623, 'ReiyaManor', 'Water Drops.wav',
                               '904668d2377b39b602eb3004b3e383851744d50a0d6884a4e0229bdc7f5a015d', None),
    'water_drop_paespedro': (174718, 'paespedro', 'Single Water Drop',
                             '236ce9ac3b8375b97aca1564caa7f54f6329f20aa789e17b9ed3bc2cfbeac8fa', None),
    'creak_slow_harveyism': (130364, 'harveyism', 'Slow Creaking Stereo.wav',
                             'dab940c1122b5912ca4e93384a6e21634c9d882a0542c4c3e7437f556841279c', None),
    'rope_creaks_6polnic': (231438, '6polnic', 'hamp rope creaks',
                            '8e860cabd88011fb2a144702c56aee0a4d85003e4efea2d59ad75730eb232c87', None),
}
OBJECTS = {
    'bowl_ceramic_3dblanco': (848463, '3D_Blanco', 'Ceramic Bowl Taps and Scrapes',
                              '670c3862dc89c0997107b5d3ea37c4f3db43603e3403fc74a3103c60c0cf37a2', None),
    'plate_porcelain_chilsville': (591575, 'Chilsville', 'Plate ding.m4a',
                                   '0104357a61ba907253e2df288aea5bf872fa3caeec30d9928149af2fa9d15b62', None),
    'cup_ceramic_that_finn': (706536, 'that_Finn', 'Light ceramic cling',
                              '20c2ae0faaa0bbcac9fe04cbe4a933cbd28818a8212f8e7fa12c4ebb36f6315d',
                              'https://cdn.freesound.org/previews/706/706536_15331068-hq.ogg'),
    'wings_pigeon_kinoton': (689998, 'Kinoton', 'Pigeon, Flies Away, Flapping Wings',
                             'af995ba91aadc7c3fc1bfe38e16542ff43364b0d116f6853a459277d869a3ef5', None),
    'wings_bird_21100495': (655322, '21100495', 'bird wings flapping.wav',
                            '9c20a61f9a28c8b0e8b1c4e056fbfad26cdaa7d4513c7ce4d4feb5282a7fe50e', None),
    'wings_pigeon_gerardcatala': (153277, 'gerardcatala', 'Beating wings of a pigeon',
                                  '28cc2a1c2b93f6b3a5ffadb0186a25b753874c6e4f2678256c0c2cf9edc94e86', None),
}
ZENODO = dict(record=1442513, file='VocalSet11.zip', license='cc-by-4.0', dest='vocalset',
              attribution='"VocalSet: A Singing Voice Dataset" by Julia Wilkins, Prem Seetharaman, Alison Wahl and '
                          'Bryan Pardo (ISMIR 2018), https://doi.org/10.5281/zenodo.1442513, licensed under CC BY 4.0 '
                          '(https://creativecommons.org/licenses/by/4.0/). Changes: cut into notes, resynthesised '
                          'with the WORLD vocoder, re-pitched, spliced and mixed.')
PACKS = ('piano', 'vcsl', 'vsco2', 'field', 'found', 'objects', 'vocalset')


# ----------------------------------------------------------------------------- 网络
def _open(url, headers=None, timeout=120, method=None):
    req = urllib.request.Request(url, headers={'User-Agent': UA, **(headers or {})}, method=method)
    tok = os.environ.get('GITHUB_TOKEN')
    if tok and url.startswith('https://api.github.com/'):
        req.add_header('Authorization', f'Bearer {tok}')
    return urllib.request.urlopen(req, timeout=timeout)


def get(url, headers=None, timeout=120, tries=5):
    for k in range(tries):
        try:
            with _open(url, headers, timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404, 410) or k == tries - 1:
                raise
            time.sleep(3 * (k + 1))           # 429 / 5xx：退让后重试
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def get_json(url):
    return json.loads(get(url))


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


class Rejected(RuntimeError):
    """许可不合格或与目录记录不一致：拒收。"""


# ----------------------------------------------------------------------------- 许可识别
def license_from_text(text: str) -> str | None:
    """CC 法律文本 -> SPDX 风格的标识；不认识或带 SA/NC/ND 时返回 None。"""
    head = text[:4000]
    if re.search(r'\b(ShareAlike|NonCommercial|NoDerivs|NoDerivatives)\b', head):
        return None
    if 'CC0 1.0 Universal' in head:
        return 'CC0-1.0'
    if re.search(r'Attribution 3\.0 (Unported|International)', head):
        return 'CC-BY-3.0'
    if re.search(r'Attribution 4\.0 International', head):
        return 'CC-BY-4.0'
    return None


def license_from_commons(short: str | None) -> str | None:
    """Commons 的 LicenseShortName -> 标识。规则与原 tools/commons.py 相同：只收 CC0 / PD / CC BY x.y。"""
    if not short:
        return None
    s = short.strip()
    if re.search(r'SA|NC|ND', s):
        return None
    if re.match(r'^CC0', s, re.I):
        return 'CC0-1.0'
    if re.match(r'^(Public domain|PD)', s, re.I):
        return 'PD'
    m = re.match(r'^CC BY (\d(?:\.\d)?)( [a-z]{2})?$', s, re.I)
    if m:
        return f'CC-BY-{m.group(1)}' if '.' in m.group(1) else f'CC-BY-{m.group(1)}.0'
    return None


def license_from_cc_url(url: str | None) -> str | None:
    """Freesound APIv2 的 license 字段（creativecommons.org 的许可 URL）-> 标识；NC / SA / ND / sampling+ 返回 None。"""
    m = re.search(r'creativecommons\.org/(licenses/[a-z+-]+/[\d.]+|publicdomain/zero/[\d.]+)', url or '')
    if not m:
        return None
    x = m.group(1)
    if x.startswith('publicdomain/zero/'):
        return 'CC0-1.0'
    if re.search(r'-nc|-nd|-sa|sampling', x):
        return None
    m = re.fullmatch(r'licenses/by/(\d)\.(\d)', x)
    return f'CC-BY-{m.group(1)}.{m.group(2)}' if m else None


# ----------------------------------------------------------------------------- 文件选择
_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def note_midi(name: str) -> int:
    m = re.fullmatch(r'([A-Ga-g])([#b]?)(-?\d)', name.strip())
    if not m:
        raise ValueError(f'bad note name {name!r} (e.g. D2, F#4, Bb3)')
    acc = {'#': 1, 'b': -1, '': 0}[m.group(2)]
    return _PC[m.group(1).upper()] + acc + 12 * (int(m.group(3)) + 1)


def parse_range(s):
    if not s:
        return None
    lo, hi = s.split('-', 1) if not s.startswith('-') else (s, '')
    return note_midi(lo) - 2, note_midi(hi) + 2


def _in_range(midi, rng):
    return rng is None or midi is None or rng[0] <= midi <= rng[1]


def vsco_sets():
    from sampler.prepare import vsco
    out = {}
    for folder, sec, art in vsco.FOLDERS + vsco.WINDS:
        out[f'{sec}/{art}'] = folder
    for folder, sec, art in vsco.PERC:
        out[f'{sec}/{art}' if art else sec] = folder
    return out


def vsco_groups():
    from sampler.prepare import vsco
    g = {'strings': [f'{s}/{a}' for _, s, a in vsco.FOLDERS], 'winds': [f'{s}/{a}' for _, s, a in vsco.WINDS],
         'perc': ['timp/hit', 'timp/roll', 'cym']}
    for _, s, a in vsco.FOLDERS + vsco.WINDS:
        g.setdefault(s, []).append(f'{s}/{a}')
    return g


def select_github(pack, tree, sets, rng):
    """GitHub 包里要下载的 [(仓库路径, git sha, 大小, 本地相对路径, 套)]。"""
    out = []
    if pack == 'piano':
        for e in tree:
            p = e['path']
            if p in ('.gitattributes',):
                continue
            # 只有音符采样（A0v1.flac …）和弦共鸣采样（harmLA0.flac …）按音高过滤；rel*、pedal* 总是带上
            m = re.fullmatch(r'Samples/(?:harm(?:L|S|V3))?([A-G])(#?)(\d)(?:v\d+)?\.flac', p)
            midi = (_PC[m.group(1)] + (1 if m.group(2) else 0) + 12 * (int(m.group(3)) + 1)) if m else None
            if _in_range(midi, rng):
                out.append((p, e['sha'], e['size'], f'salamander/{p}', 'piano'))
        return out
    if pack == 'vcsl':
        from sampler.prepare import vcsl
        want = sets or list(vcsl.SETS)
        for sid in want:
            folder, vfolder, kind, pitched, off = vcsl.SETS[sid]
            for e in tree:
                p = e['path']
                if p.rsplit('/', 1)[0] != vfolder or not p.endswith('.wav'):
                    continue
                name = p.rsplit('/', 1)[1]
                nm = vcsl.name_midi(name) if pitched else None
                if not _in_range(None if nm is None else nm + (off or 0), rng):
                    continue
                local = f'vcsl/{folder}/{name}' if kind == 'wav' else f'vcsl/{folder}/{name[:-4]}.flac'
                out.append((p, e['sha'], e['size'], local, sid))
        for e in tree:                          # 许可与说明随包带上
            if e['path'] in ('LICENSE', 'README.md'):
                out.append((e['path'], e['sha'], e['size'], f"vcsl/{e['path']}", 'meta'))
        return out
    if pack == 'vsco2':
        all_sets = vsco_sets()
        groups = vsco_groups()
        want = []
        for s in sets or ['strings', 'winds', 'perc']:
            want += groups.get(s, [s])
        for s in want:
            if s not in all_sets:
                raise SystemExit(f'unknown vsco2 set {s!r}; one of {sorted(all_sets) + sorted(groups)}')
            folder = all_sets[s]
            sec = s.split('/')[0]
            for e in tree:
                p = e['path']
                if p.rsplit('/', 1)[0] != folder or not p.endswith('.wav'):
                    continue
                name = p.rsplit('/', 1)[1]
                from sampler.prepare import vsco
                if sec not in ('timp', 'cym'):
                    nm = vsco.name_midi(name[:-4])
                    if not _in_range(None if nm is None else nm + vsco.NAME_OFFSET.get(sec, 0), rng):
                        continue
                out.append((p, e['sha'], e['size'], f'vsco2/{vsco.local_path(p)}', s))
        for e in tree:
            if e['path'] in ('LICENSE', 'README.md'):
                out.append((e['path'], e['sha'], e['size'], f"vsco2/{e['path']}", 'meta'))
        return out
    raise ValueError(pack)


def source_of_local(rel: str):
    """采样根下的相对路径 -> (包, 套或名字, 仓库路径或 None)。供 --files 用。"""
    rel = rel.strip().lstrip('./')
    top, _, rest = rel.partition('/')
    if top == 'salamander':
        return 'piano', 'piano', rest
    if top == 'vcsl':
        from sampler.prepare import vcsl
        folder, _, name = rest.rpartition('/')
        for sid, (f, vfolder, kind, _, _) in vcsl.SETS.items():
            if f == folder:
                return 'vcsl', sid, f'{vfolder}/{Path(name).stem}.wav'
        if rest in ('LICENSE', 'README.md'):
            return 'vcsl', 'meta', rest
    if top == 'vsco2':
        parts = rest.split('/')
        stem = Path(parts[-1]).stem
        if parts[0] == 'perc':
            sec = parts[1]
            if sec == 'timp':
                folder = 'Percussion/Timpani/Rolls' if 'roll' in stem.lower() else 'Percussion/Timpani'
                return 'vsco2', 'timp/roll' if 'roll' in stem.lower() else 'timp/hit', f'{folder}/{stem}.wav'
            return 'vsco2', 'cym', f'VSCO 1 Percussion/varMetal/Cymbals/susp/{stem}.wav'
        sets = vsco_sets()
        key = f'{parts[0]}/{parts[1]}'
        if key in sets:
            return 'vsco2', key, f'{sets[key]}/{stem}.wav'
    if top == 'field' and rest.endswith('.flac') and Path(rest).stem in COMMONS:
        return 'field', Path(rest).stem, None
    if top == 'found' and Path(rest).stem in FOUND:
        return 'found', Path(rest).stem, None
    if top == 'objects' and Path(rest).stem in OBJECTS:
        return 'objects', Path(rest).stem, None
    if top == 'vocalset':
        raise SystemExit('vocalset 不能按文件取（音符单元索引要整套建）：用 --only vocalset')
    raise SystemExit(f'--files: 不认识的路径 {rel!r}')


# ----------------------------------------------------------------------------- 记录
class Manifest:
    def __init__(self, root: Path):
        self.path = root / 'fetch_manifest.json'
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {'files': {}, 'licenses': {}}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.d, ensure_ascii=False, indent=1, sort_keys=True))


# ----------------------------------------------------------------------------- GitHub
def github_license(pack, root, staging, log):
    cfg = GITHUB[pack]
    repo, commit = cfg['repo'], cfg['commit']
    tree_cache = staging / f"{repo.replace('/', '_')}@{commit}.tree.json"
    if tree_cache.exists():
        tree = json.loads(tree_cache.read_text())
    else:
        d = get_json(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1')
        if d.get('truncated'):
            raise SystemExit(f'{repo}: git tree truncated')
        tree = [e for e in d['tree'] if e['type'] == 'blob']
        tree_cache.parent.mkdir(parents=True, exist_ok=True)
        tree_cache.write_text(json.dumps(tree))
    lic = next((e for e in tree if e['path'] == 'LICENSE'), None)
    if lic is None:
        raise Rejected(f'{repo}@{commit[:8]}: no LICENSE file')
    data = get(f'https://raw.githubusercontent.com/{repo}/{commit}/LICENSE')
    if git_blob_sha1(data) != lic['sha']:
        raise SystemExit(f'{repo}: LICENSE blob hash mismatch')
    found = license_from_text(data.decode('utf-8', 'replace'))
    api = None
    try:
        api = get_json(f'https://api.github.com/repos/{repo}/license').get('license', {}).get('spdx_id')
    except Exception as e:                  # API 只做旁证；以固定 commit 的 LICENSE 全文为准
        log(f'  (GitHub license API unavailable: {e})')
    log(f'  {pack}: {repo}@{commit[:10]} LICENSE text -> {found}; GitHub API spdx_id -> {api}')
    if found not in ALLOWED or found != cfg['license']:
        raise Rejected(f'{pack}: licence {found!r} (expected {cfg["license"]}) - refused')
    if api not in (None, 'NOASSERTION', found):
        raise Rejected(f'{pack}: GitHub reports {api!r} but LICENSE reads {found!r} - refused')
    evidence = dict(source=f'https://github.com/{repo}/tree/{commit}', license=found,
                    license_file_sha1=lic['sha'], github_api_spdx=api,
                    checked=time.strftime('%Y-%m-%d'), attribution=cfg['attribution'])
    return tree, data, evidence


def download_github(pack, items, root, staging, man, jobs, log):
    cfg = GITHUB[pack]
    repo, commit = cfg['repo'], cfg['commit']

    def one(it):
        path, sha, size, local, sid = it
        dst = root / local
        rec = man.d['files'].get(local)
        if dst.exists() and rec and rec.get('git_sha1') == sha:
            return local, 'have', 0
        url = f'https://raw.githubusercontent.com/{repo}/{commit}/' + urllib.parse.quote(path)
        data = get(url, timeout=300)
        if len(data) != size or git_blob_sha1(data) != sha:
            raise SystemExit(f'{path}: size / git blob hash mismatch')
        if pack == 'piano' or sid == 'meta':
            target = dst
        elif pack == 'vcsl':                    # 暂存区按套分目录，导入时复制或转 FLAC
            target = staging / 'vcsl' / sid / path.rsplit('/', 1)[1]
        else:                                   # vsco2：暂存区保留仓库路径，导入时据此定位
            target = staging / 'vsco2' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + '.part')
        tmp.write_bytes(data)
        tmp.replace(target)
        man.d['files'][local] = dict(pack=pack, set=sid, source=f'{repo}@{commit}:{path}', git_sha1=sha,
                                     bytes=size, license=cfg['license'])
        return local, 'got', size

    got = 0
    with ThreadPoolExecutor(max(1, jobs)) as ex:
        for k, (local, st, n) in enumerate(ex.map(one, items), 1):
            got += n
            if st == 'got' and (k % 10 == 0 or k == len(items)):
                log(f'  {pack}: {k}/{len(items)}  {got / 1e6:.1f} MB')
    return got


# ----------------------------------------------------------------------------- Commons
def commons_info(titles):
    out = {}
    for i in range(0, len(titles), 40):
        q = urllib.parse.urlencode({'action': 'query', 'titles': '|'.join(titles[i:i + 40]), 'prop': 'imageinfo',
                                    'iiprop': 'url|size|mime|extmetadata', 'format': 'json'})
        d = get_json('https://commons.wikimedia.org/w/api.php?' + q)
        norm = {n['to']: n['from'] for n in d['query'].get('normalized', [])}
        for p in d['query']['pages'].values():
            ii = (p.get('imageinfo') or [{}])[0]
            em = ii.get('extmetadata', {})
            strip = lambda h: re.sub(r'<[^>]+>', '', h or '').strip()          # noqa: E731
            out[norm.get(p.get('title'), p.get('title'))] = dict(
                title=p.get('title'), license=em.get('LicenseShortName', {}).get('value'),
                license_url=em.get('LicenseUrl', {}).get('value'), artist=strip(em.get('Artist', {}).get('value')),
                credit=strip(em.get('Credit', {}).get('value'))[:120],
                desc=strip(em.get('ImageDescription', {}).get('value'))[:160],
                size=ii.get('size'), url=ii.get('url'), page=ii.get('descriptionurl'))
    return out


def fetch_commons(names, root, staging, man, dry, log):
    from sampler.prepare import field
    info = commons_info([COMMONS[n][0] for n in names])
    todo = []
    for n in names:
        title, expected = COMMONS[n]
        m = info.get(title)
        if not m or not m.get('url'):
            raise SystemExit(f'Commons: {title} not found')
        lic = license_from_commons(m['license'])
        log(f'  field:{n}: LicenseShortName {m["license"]!r} -> {lic}')
        if lic is None or lic != license_from_commons(expected):
            raise Rejected(f'field:{n}: licence {m["license"]!r} (recorded {expected!r}) - refused')
        if n in COMMONS_AUTHOR_FIX:
            m['artist'] = COMMONS_AUTHOR_FIX[n]
        todo.append((n, m, lic))
    if dry:
        return todo
    for n, m, lic in todo:
        local = f'field/{n}.flac'
        if (root / local).exists() and man.d['files'].get(local, {}).get('bytes') == m['size']:
            continue
        data = get(m['url'], timeout=600)
        if len(data) != m['size']:
            raise SystemExit(f'{n}: size {len(data)} != {m["size"]}')
        src = staging / 'field' / m['title'].split(':', 1)[1].replace('/', '_')
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(data)
        field.process(n, src, m)
        man.d['files'][local] = dict(pack='field', set=n, source=m['page'], download_url=m['url'], bytes=m['size'],
                                     sha256=hashlib.sha256(data).hexdigest(), license=lic,
                                     license_short_name=m['license'], author=m['artist'],
                                     checked=time.strftime('%Y-%m-%d'))
        man.save()
    return todo


# ----------------------------------------------------------------------------- Freesound
def fetch_freesound(pack, names, root, man, dry, log):
    """返回 (取到或已有的, 没取到的)。没设 FREESOUND_API_KEY 时只用记录的预览地址，没有记录的跳过。"""
    table = FOUND if pack == 'found' else OBJECTS
    out_dir = root / pack
    mpath = out_dir / 'manifest.json'
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    key = os.environ.get('FREESOUND_API_KEY')
    if FREESOUND_LICENSE not in ALLOWED:
        raise Rejected(f'{pack}: recorded licence {FREESOUND_LICENSE!r} - refused')
    if not key:
        log(f'  {pack}: 未在线复核许可，以记录为准（{FREESOUND_LICENSE}）；设 FREESOUND_API_KEY 可在线复核')
    done, missing = [], []
    for n in names:
        sid, user, title, sha, url = table[n]
        local = f'{pack}/{n}.ogg'
        if (root / local).exists() and hashlib.sha256((root / local).read_bytes()).hexdigest() == sha:
            done.append(n)
            continue
        page = f'https://freesound.org/people/{user}/sounds/{sid}/'          # 只记进 manifest，不去抓
        lic, how = FREESOUND_LICENSE, 'recorded in fetch_samples.py (not re-checked online)'
        if key:
            info = json.loads(get(FREESOUND_API.format(sid=sid), headers={'Authorization': f'Token {key}'}))
            lic = license_from_cc_url(info.get('license'))
            log(f'  {pack}:{n}: APIv2 license {info.get("license")!r} -> {lic}')
            if lic not in ALLOWED or lic != FREESOUND_LICENSE:
                raise Rejected(f'{pack}:{n}: licence {info.get("license")!r} (recorded {FREESOUND_LICENSE}) - refused')
            if info.get('username') != user:
                raise SystemExit(f'{pack}:{n}: APIv2 username {info.get("username")!r} != recorded {user!r}')
            url = (info.get('previews') or {}).get('preview-hq-ogg') or url
            how = f'Freesound APIv2 license field, {time.strftime("%Y-%m-%d")}'
        if not url:
            log(f'  {pack}:{n}: 没有记录预览地址，跳过（设 FREESOUND_API_KEY 经官方 API 取）')
            missing.append(n)
            continue
        if dry:
            log(f'  {pack}:{n}: {url}')
            done.append(n)
            continue
        try:
            data = get(url)
        except urllib.error.HTTPError as e:
            log(f'  {pack}:{n}: {url} -> HTTP {e.code}; skipped')
            missing.append(n)
            continue
        got = hashlib.sha256(data).hexdigest()
        if got != sha:
            raise SystemExit(f'{pack}:{n}: preview SHA-256 {got[:12]} != recorded {sha[:12]} (Freesound re-encoded '
                             f'it?); not used')
        out_dir.mkdir(parents=True, exist_ok=True)
        (root / local).write_bytes(data)
        import soundfile as sf
        info = sf.info(root / local)
        manifest[n] = dict(file=f'{n}.ogg', title=title, author=user, page=page, download_url=url,
                           license='CC0 1.0' if lic == 'CC0-1.0' else lic,
                           license_url='https://creativecommons.org/publicdomain/zero/1.0/' if lic == 'CC0-1.0'
                           else None, license_check=how, sr=info.samplerate,
                           channels=info.channels, duration_s=round(info.duration, 3), sha256=sha,
                           processing='none on disk (byte copy of the HQ preview); cut at load time')
        man.d['files'][local] = dict(pack=pack, set=n, source=page, preview=url, sha256=sha, license=lic,
                                     license_check=how, checked=time.strftime('%Y-%m-%d'))
        mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        man.save()
        done.append(n)
    return done, missing


# ----------------------------------------------------------------------------- Zenodo（VocalSet）
class RangeFile(io.RawIOBase):
    """只读、可 seek 的 HTTP 文件：每次 read 发一个 Range 请求（外面套 BufferedReader）。"""

    def __init__(self, url):
        with _open(url, method='HEAD', timeout=60) as r:
            self.size = int(r.headers['Content-Length'])
            self.url = r.geturl()
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        data = get(self.url, headers={'Range': f'bytes={self.pos}-{end}'}, timeout=300)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_vocalset(root, staging, man, dry, jobs, log):
    rec = get_json(f'https://zenodo.org/api/records/{ZENODO["record"]}')
    lic = (rec.get('metadata', {}).get('license') or {}).get('id')
    log(f'  vocalset: Zenodo {ZENODO["record"]} metadata.license.id -> {lic!r}')
    if lic != ZENODO['license']:
        raise Rejected(f'vocalset: licence {lic!r} (expected {ZENODO["license"]}) - refused')
    f = next(x for x in rec['files'] if x['key'] == ZENODO['file'])
    from sampler.prepare import vocalset
    z = zipfile.ZipFile(io.BufferedReader(RangeFile(f['links']['self']), buffer_size=1 << 20))
    members = []
    for n in z.namelist():
        p = n.split('/')
        if p[0] == 'FULL' and len(p) == 5 and n.endswith('.wav') and (p[2], p[3]) in vocalset.SUBSET:
            members.append(n)
    total = sum(z.getinfo(n).file_size for n in members)
    log(f'  vocalset: {len(members)} members, {total / 1e6:.0f} MB (of a {f["size"] / 1e9:.2f} GB zip)')
    if dry:
        return members
    for k, n in enumerate(members, 1):
        dst = staging / 'vocalset' / n
        if dst.exists() and dst.stat().st_size == z.getinfo(n).file_size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(z.read(n))            # zipfile 校验 CRC-32
        if k % 50 == 0:
            log(f'  vocalset: {k}/{len(members)}')
    log('  vocalset: building the note index (WORLD harvest on every file; takes a while)')
    vocalset.build(staging / 'vocalset', jobs=jobs)
    man.d['licenses']['vocalset'] = dict(source=f'https://doi.org/10.5281/zenodo.{ZENODO["record"]}',
                                         license='CC-BY-4.0', zenodo_license_id=lic, members=len(members),
                                         checked=time.strftime('%Y-%m-%d'), attribution=ZENODO['attribution'])
    return members


# ----------------------------------------------------------------------------- 主流程
def parse_only(spec):
    sel = {}
    for tok in filter(None, (t.strip() for t in (spec or '').split(','))):
        pack, _, sub = tok.partition(':')
        if pack not in PACKS:
            raise SystemExit(f'unknown pack {pack!r}; one of {PACKS}')
        sel.setdefault(pack, [])
        if sub:
            sel[pack].append(sub)
    return sel


def list_catalog():
    from sampler.prepare import vcsl
    print('包 / 套（--only 包 或 包:套）')
    print(f"  piano     Salamander Grand Piano V3, {GITHUB['piano']['license']}（须署名）, ~750 MB")
    print(f"  vcsl      VCSL, {GITHUB['vcsl']['license']}: " + ', '.join(vcsl.SETS))
    print(f"  vsco2     VSCO 2 CE, {GITHUB['vsco2']['license']}: " + ', '.join(sorted(vsco_groups())) + ', '
          + ', '.join(sorted(vsco_sets())))
    print('  field     Wikimedia Commons: ' + ', '.join(f'{k} ({v[1]})' for k, v in COMMONS.items()))
    star = lambda t: ', '.join(k + ('' if v[4] else '*') for k, v in t.items())          # noqa: E731
    print('  found     Freesound CC0: ' + star(FOUND))
    print('  objects   Freesound CC0: ' + star(OBJECTS))
    print('            * 没有记录预览地址：要设 FREESOUND_API_KEY 经官方 API 取')
    print(f"  vocalset  VocalSet 1.1, CC BY 4.0（须署名）, Zenodo {ZENODO['record']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--only', help='包[:套],...')
    ap.add_argument('--range', help='音高范围，如 D2-A5（实际发声音高）')
    ap.add_argument('--files', help='精确文件清单（相对采样根的路径，每行一个）')
    ap.add_argument('--dest', help='采样根目录（默认 $AGENTREEL_SAMPLES 或 music/sampler/samples）')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--dry-run', action='store_true', help='只核对许可并列出文件，不下载')
    ap.add_argument('--keep-staging', action='store_true')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args(argv)
    if a.dest:
        os.environ['AGENTREEL_SAMPLES'] = str(Path(a.dest).expanduser().resolve())
    sys.path.insert(0, str(HERE))
    from sampler import _paths
    root = _paths.SAMPLES
    if a.list:
        list_catalog()
        return 0
    sel = parse_only(a.only)
    files = {}
    if a.files:
        for line in Path(a.files).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):        # 只有行首的 # 是注释：音名里有 #（A#3）
                continue
            pack, sid, src = source_of_local(line)
            files.setdefault(pack, {})[line] = (sid, src)
            sel.setdefault(pack, None)
    if not sel:
        ap.error('nothing selected: use --only, --files or --list')
    rng = parse_range(a.range)
    staging = root / '.staging'
    root.mkdir(parents=True, exist_ok=True)
    man = Manifest(root)
    t0 = time.time()
    incomplete = []

    def log(msg):
        print(f'[{time.time() - t0:6.1f}s] {msg}', flush=True)

    log(f'sample root: {root}')
    try:
        for pack in PACKS:
            if pack not in sel:
                continue
            sets = sel[pack] or []
            log(f'{pack}: licence check')
            if pack in GITHUB:
                tree, lic_bytes, evidence = github_license(pack, root, staging, log)
                if pack in files:
                    want = {src: local for local, (sid, src) in files[pack].items()}
                    tmap = {e['path']: e for e in tree}
                    items = []
                    for src, local in want.items():
                        if src not in tmap:
                            raise SystemExit(f'{src}: not in {GITHUB[pack]["repo"]}@{GITHUB[pack]["commit"][:8]}')
                        e = tmap[src]
                        items.append((src, e['sha'], e['size'], local, files[pack][local][0]))
                    for e in tree:                 # 许可文件总是一起取
                        if e['path'] in ('LICENSE', 'README.md') and f"{GITHUB[pack]['dest']}/{e['path']}" not in want.values():
                            items.append((e['path'], e['sha'], e['size'], f"{GITHUB[pack]['dest']}/{e['path']}", 'meta'))
                else:
                    items = select_github(pack, tree, sets, rng)
                mb = sum(i[2] for i in items) / 1e6
                log(f'{pack}: {len(items)} files, {mb:.1f} MB')
                if a.dry_run:
                    for i in items[:12]:
                        print('   ', i[0], f'{i[2] / 1e6:.1f} MB')
                    if len(items) > 12:
                        print(f'    ... {len(items) - 12} more')
                    continue
                man.d['licenses'][pack] = evidence
                download_github(pack, items, root, staging, man, a.jobs, log)
                man.save()
                if pack == 'piano':
                    (root / 'salamander').mkdir(parents=True, exist_ok=True)
                if pack == 'vcsl':
                    from sampler.prepare import vcsl
                    touched = vcsl.import_wavs(staging / 'vcsl', keep=a.keep_staging)
                    if touched or not (root / 'vcsl' / 'index_folk.json').exists():
                        log('vcsl: measuring the folk / piano sets -> index_folk.json')
                        vcsl.build_index(verbose=False)
                if pack == 'vsco2':
                    from sampler.prepare import vsco
                    n = vsco.import_wavs(staging / 'vsco2', keep=a.keep_staging)
                    if n or not (root / 'vsco2' / 'index.json').exists():
                        log('vsco2: measuring -> index.json')
                        vsco.build_index(verbose=False)
            elif pack == 'field':
                names = [x for x, _ in files['field'].values()] if 'field' in files else (sets or list(COMMONS))
                fetch_commons(names, root, staging, man, a.dry_run, log)
                if not a.dry_run:
                    from sampler import birds
                    if any((root / 'field' / f'{s}.flac').exists() for s in birds.SOURCES.values()):
                        log('field: bird-call catalogue')
                        birds.write_catalog()
            elif pack in ('found', 'objects'):
                table = FOUND if pack == 'found' else OBJECTS
                names = [x for x, _ in files[pack].values()] if pack in files else (sets or list(table))
                for n in names:
                    if n not in table:
                        raise SystemExit(f'unknown {pack} item {n!r}; one of {list(table)}')
                _, miss = fetch_freesound(pack, names, root, man, a.dry_run, log)
                incomplete += [f'{pack}/{x}.ogg' for x in miss]
                if not a.dry_run:
                    online = bool(os.environ.get('FREESOUND_API_KEY'))
                    man.d['licenses'][pack] = dict(source='https://freesound.org (HQ previews)', license=FREESOUND_LICENSE,
                                                   check='Freesound APIv2 license field' if online
                                                   else 'recorded in fetch_samples.py (not re-checked online)',
                                                   checked=time.strftime('%Y-%m-%d'))
            elif pack == 'vocalset':
                fetch_vocalset(root, staging, man, a.dry_run, a.jobs, log)
            if not a.dry_run:
                man.save()
    except Rejected as e:
        print(f'REFUSED: {e}', file=sys.stderr)
        return 2
    if not a.keep_staging and staging.exists() and not a.dry_run:
        for p in sorted(staging.rglob('*'), reverse=True):
            if p.is_file() and not p.name.endswith('.tree.json'):
                p.unlink()
            elif p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
    if incomplete:
        log(f'没取到 {len(incomplete)} 个文件：' + ', '.join(incomplete))
        return 1
    log('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
