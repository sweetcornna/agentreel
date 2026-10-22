"""agentreel sampler：代码写配乐用的采样乐器 + 经典 DSP（48 kHz，float32 立体声数组 (n, 2)，时间单位秒）。
移植自一部示例片的配乐工具库。

不用任何神经网络或生成模型：采样钢琴（Salamander，CC BY 3.0；或 VCSL 的 Steinway B，CC0）、共振峰合成的合唱、
加法合成的弦乐 pad、野外录音（CC0 / 公有领域 / CC BY，见 CREDITS.md）与程序化合成音效、音效与质感（sampler.fx）、
混音与母带（sampler.mix）。
真录音：人声（sampler.voice：VocalSet，CC BY 4.0，WORLD 声码器）、弦乐与木管（sampler.strings：VSCO 2 CE，CC0）、
民谣乐器、三角钢琴与手鼓类（sampler.folk：VCSL，CC0）、画面里那些东西的声音（sampler.found：Freesound，CC0）、
瓷碗（sampler.porcelain，CC0）、鸟鸣与它的钢琴转写（sampler.birds）。
写谱用：sampler.grid（从一部片的 timeline.json 读拍网格、写 events.json）、sampler.bus（分轨、按起音对齐、母带、
manifest）、sampler.check（同步、切点、响度自检）。

采样不在包里：见 fetch_samples.py 与环境变量 AGENTREEL_SAMPLES（sampler._paths）。
"""
from ._dsp import SR  # noqa: F401

__version__ = '0.1.0'

__all__ = ['SR', 'render_piano', 'render_choir', 'choir_pad', 'ambience', 'sfx', 'distant',
           'KINDS_AMBIENCE', 'KINDS_SFX', 'place', 'place_end', 'reverb', 'eq', 'lowpass',
           'highpass', 'compress', 'limit', 'fade', 'normalize_lufs', 'master', 'write',
           'pad', 'sub', 'reverse_swell', 'granular', 'shimmer', 'tape', 'impact', 'pitch_shift',
           'IMPACT_KINDS', 'stats', 'report', 'plot',
           'render_voice', 'tintinnabuli', 'tintinnabuli_pair', 'render_strings', 'pluck',
           'bowl', 'render_bowls', 'BOWL_SCALE', 'bird_calls', 'bird_call', 'transcribe_call',
           'render_birds', 'echo_notes', 'SAMPLES']

_MODULES = ('mix', 'analysis', 'piano', 'choir', 'pads', 'fx', 'nature', 'voice', 'strings', 'porcelain',
            'birds', 'folk', 'found', 'band', 'synth', 'grid', 'bus', 'check', 'sfz')


def __getattr__(name):  # lazy imports keep `import sampler` cheap
    if name in _MODULES:
        import importlib
        return importlib.import_module(f'.{name}', __name__)
    if name == 'SAMPLES':
        from ._paths import SAMPLES
        return SAMPLES
    if name == 'render_piano':
        from .piano import render_piano
        return render_piano
    if name in ('render_choir', 'choir_pad'):
        from . import choir
        return getattr(choir, name)
    if name in ('pad', 'sub'):
        from . import pads
        return getattr(pads, name)
    if name in ('reverse_swell', 'granular', 'shimmer', 'tape', 'impact', 'pitch_shift', 'IMPACT_KINDS'):
        from . import fx
        return getattr(fx, name)
    if name in ('ambience', 'sfx', 'distant', 'KINDS_AMBIENCE', 'KINDS_SFX'):
        from . import nature
        return getattr(nature, name)
    if name in ('place', 'place_end', 'reverb', 'eq', 'lowpass', 'highpass', 'compress', 'limit',
                'fade', 'normalize_lufs', 'master', 'write', 'pan', 'gain', 'mixdown', 'new_track'):
        from . import mix
        return getattr(mix, name)
    if name in ('render_voice', 'tintinnabuli', 'tintinnabuli_pair'):
        from . import voice
        return getattr(voice, name)
    if name in ('render_strings', 'pluck'):
        from . import strings
        return getattr(strings, name)
    if name in ('bowl', 'render_bowls', 'BOWL_SCALE', 'bowl_scale'):
        from . import porcelain
        return getattr(porcelain, name)
    if name in ('bird_calls', 'bird_call', 'transcribe_call', 'render_birds', 'echo_notes',
                'resolve_calls', 'pick_calls'):
        from . import birds
        return getattr(birds, name)
    if name in ('stats', 'report', 'plot'):
        from . import analysis
        return getattr(analysis, name)
    raise AttributeError(name)
