"""Minimal SFZ v2 parser (the subset the Salamander Grand Piano V3 SFZ actually uses).

Handles, in file order:
  * ``//`` comments,
  * ``#define $NAME value`` and ``#include "file"`` (sequential: macros are expanded
    at the moment a line is read, so the same include can be re-expanded with
    different macro values -- Salamander relies on this for its 16 velocity groups),
  * headers ``<control> <global> <master> <group> <region> <curve>`` with the usual
    inheritance (a new global resets master+group, a new master resets group),
  * opcode values that contain spaces (value runs to the next ``opcode=`` / header /
    directive or the end of the line).

Returns plain dicts; interpretation of opcodes lives in ``piano.py``.
"""
from __future__ import annotations

import re
from pathlib import Path

_TOKEN = re.compile(
    r'#define[ \t]+(?P<dname>\$\w+)[ \t]+(?P<dval>\S+)'
    r'|#include[ \t]+"(?P<inc>[^"]+)"'
    r'|<(?P<hdr>\w+)>'
    r'|(?P<op>[A-Za-z0-9_$]+)='
)
_MACRO = re.compile(r'\$\w+')
_NOTE = re.compile(r'^([A-Ga-g])([#b]?)(-?\d+)$')
_SEMI = {'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11}


def note_number(v) -> int:
    """SFZ key value -> MIDI number. Accepts ints or names like C4 / c#4 / Eb3 (C4 = 60)."""
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip()
    try:
        return int(s)
    except ValueError:
        pass
    m = _NOTE.match(s)
    if not m:
        raise ValueError(f'bad key value {v!r}')
    n = _SEMI[m.group(1).lower()] + {'#': 1, 'b': -1, '': 0}[m.group(2)]
    return n + 12 * (int(m.group(3)) + 1)


class SFZ:
    def __init__(self, path):
        self.path = Path(path)
        self.root = self.path.parent
        self.defines: dict[str, str] = {}
        self.control: dict[str, str] = {}
        self.regions: list[dict[str, str]] = []
        self.curves: list[dict[str, str]] = []
        self._sec = {'global': {}, 'master': {}, 'group': {}}
        self._cur: dict | None = None      # dict opcodes currently go into
        self._region: dict | None = None
        self._scan(self.path)
        self._flush()

    # -- scanning -----------------------------------------------------------------
    def _expand(self, s: str) -> str:
        for _ in range(4):  # nested macros are not used here, but be safe
            new = _MACRO.sub(lambda m: self.defines.get(m.group(0), m.group(0)), s)
            if new == s:
                break
            s = new
        return s

    def _scan(self, path: Path):
        text = path.read_text(encoding='utf-8', errors='replace')
        text = re.sub(r'//[^\n]*', '', text)
        matches = list(_TOKEN.finditer(text))
        for i, m in enumerate(matches):
            if m.group('dname'):
                self.defines[m.group('dname')] = m.group('dval')
            elif m.group('inc'):
                self._scan(self.root / self._expand(m.group('inc')))
            elif m.group('hdr'):
                self._header(m.group('hdr').lower())
            else:
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                nl = text.find('\n', m.end())
                if nl != -1:
                    end = min(end, nl)
                val = self._expand(text[m.end():end].strip())
                self._opcode(self._expand(m.group('op')), val)

    def _flush(self):
        if self._region is not None:
            merged = {}
            for k in ('global', 'master', 'group'):
                merged.update(self._sec[k])
            merged.update(self._region)
            self.regions.append(merged)
            self._region = None

    def _header(self, h: str):
        self._flush()
        if h == 'control':
            self.control = {}
            self._cur = self.control
        elif h == 'global':
            self._sec = {'global': {}, 'master': {}, 'group': {}}
            self._cur = self._sec['global']
        elif h == 'master':
            self._sec['master'] = {}
            self._sec['group'] = {}
            self._cur = self._sec['master']
        elif h == 'group':
            self._sec['group'] = {}
            self._cur = self._sec['group']
        elif h == 'region':
            self._region = {}
            self._cur = self._region
        elif h == 'curve':
            self.curves.append({})
            self._cur = self.curves[-1]
        else:  # unknown header (e.g. <effect>): swallow its opcodes
            self._cur = {}

    def _opcode(self, k: str, v: str):
        if self._cur is None:
            self._cur = {}
        self._cur[k] = v

    # -- helpers ------------------------------------------------------------------
    def cc_defaults(self) -> dict[int, float]:
        """Normalised (0..1) default CC values from set_ccN / set_hdccN."""
        out = {}
        for k, v in self.control.items():
            m = re.match(r'set_(hd)?cc(\d+)$', k)
            if m:
                out[int(m.group(2))] = float(v) if m.group(1) else float(v) / 127.0
        return out


def parse(path) -> SFZ:
    return SFZ(path)
