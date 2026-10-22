"""每件乐器的客观自检（我们听不到，所以量）。移植自一部示例片的 music/selfcheck.py，分组不变，另加 folk / found。

    python music/sampler/selfcheck.py piano|choir|nature|symp|choir2|fx|pad|master|r2|
                                      voice|strings|bowl|birds|sfx3|unacorda|r3|folk|found|all
    （r2 = symp choir2 fx pad master 五组，r3 = voice strings bowl birds sfx3 unacorda 六组）

输出到 $AGENTREEL_CHECK（默认 music/sampler/out/check/，git 忽略）：<名字>.png（频谱图 + 长时频谱 + 电平）、
<名字>.wav（短渲染，24 bit），数字追加进 stats.json。要用到哪些采样见 README.md 的乐器表；缺了会提示 fetch 命令。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sampler.analysis import report  # noqa: E402
from sampler._dsp import SR  # noqa: E402
from sampler._paths import SAMPLES  # noqa: E402

CHECK = Path(os.environ.get('AGENTREEL_CHECK') or ROOT / 'out' / 'check')
STATS = CHECK / 'stats.json'


def _write(name, x):
    import soundfile as sf
    sf.write(CHECK / f'{name}.wav', np.asarray(x, dtype=np.float32), SR, subtype='PCM_24')


def _timed(fn, *a, **k):
    t = time.time()
    y = fn(*a, **k)
    return y, time.time() - t


def check_piano():
    from sampler.piano import render_piano
    up = [60, 62, 64, 65, 67, 69, 71, 72]
    scale = [(i * 0.45, 0.42, m, 64 + 3 * i) for i, m in enumerate(up + up[::-1][1:])]
    # I - vi - IV - V - I with legato (syncopated) pedalling, keys released early
    chords = [[48, 55, 60, 64, 67], [45, 52, 57, 60, 64], [41, 48, 53, 57, 60],
              [43, 50, 55, 59, 62], [36, 48, 55, 60, 64, 72]]
    prog, ped = [], []
    for i, ch in enumerate(chords):
        t0 = i * 2.2
        for j, m in enumerate(ch):
            prog.append((t0 + 0.012 * j, 0.9, m, 58 + 4 * j - 3 * (i % 2)))
        ped.append((t0 + 0.12, t0 + 2.2 + 0.02 if i < len(chords) - 1 else t0 + 5.0))
    out = {}
    for felt in (False, True):
        tag = 'felt' if felt else 'normal'
        x, dt = _timed(render_piano, scale, felt=felt)
        out[f'piano_scale_{tag}'] = report(f'piano_scale_{tag}', x, CHECK, STATS,
                                           extra={'render_s': round(dt, 2)})
        _write(f'piano_scale_{tag}', x)
        x, dt = _timed(render_piano, prog, pedal=ped, felt=felt)
        out[f'piano_chords_pedal_{tag}'] = report(f'piano_chords_pedal_{tag}', x, CHECK, STATS,
                                                  extra={'render_s': round(dt, 2)})
        _write(f'piano_chords_pedal_{tag}', x)

    # velocity sweep: loudness must rise smoothly (no layer steps) -- compare with hard switching
    vels = np.arange(8, 128, 1)
    res = {}
    for xf in (0.0, 1.0):
        levels = []
        for v in vels:
            x = render_piano([(0.0, 0.6, 60, int(v))], string_res=0, hammer_noise=0, pedal_noise=0,
                             velocity_xfade=xf, duration=0.6)
            levels.append(20 * np.log10(np.sqrt((x[:int(0.3 * SR)].astype(np.float64) ** 2).mean())))
        levels = np.array(levels)
        steps = np.diff(levels)
        res[xf] = (levels, steps)
        print(f'velocity sweep C4 xfade={xf}: range {levels.min():.1f}..{levels.max():.1f} dB RMS(0-300ms), '
              f'max step between adjacent velocities {steps.max():+.2f} dB, min step {steps.min():+.2f} dB, '
              f'non-monotonic steps {(steps < -0.05).sum()}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    for xf, (lv, _) in res.items():
        ax.plot(vels, lv, label=f'velocity_xfade={xf}')
    ax.set_xlabel('MIDI velocity')
    ax.set_ylabel('RMS first 300 ms (dBFS)')
    ax.set_title('Piano C4 velocity response (layer switching vs crossfade)')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHECK / 'piano_velocity_sweep.png', dpi=90)
    plt.close(fig)
    d = json.loads(STATS.read_text()) if STATS.exists() else {}
    d['piano_velocity_sweep'] = {str(k): {'max_step_db': round(float(v[1].max()), 2),
                                          'min_step_db': round(float(v[1].min()), 2),
                                          'range_db': [round(float(v[0].min()), 1), round(float(v[0].max()), 1)]}
                                 for k, v in res.items()}

    # pedal logic: key released at 0.5 s, pedal up at 2.0 s -> level must hold until 2.0 then fall
    x = render_piano([(0.0, 0.5, 60, 80)], pedal=[(0.1, 2.0)], string_res=0, hammer_noise=0,
                     pedal_noise=0, duration=3.5)
    y = render_piano([(0.0, 0.5, 60, 80)], string_res=0, hammer_noise=0, pedal_noise=0, duration=3.5)

    def lvl(z, t):
        a = int(t * SR)
        return round(float(20 * np.log10(np.sqrt((z[a:a + 2400].astype(np.float64) ** 2).mean()) + 1e-12)), 1)
    pl = {t: (lvl(x, t), lvl(y, t)) for t in (0.4, 0.7, 1.5, 1.95, 2.1, 2.5, 3.0)}
    print('pedal test (RMS dB at t: with pedal / without pedal):', pl)
    d['piano_pedal_test'] = {str(k): v for k, v in pl.items()}
    STATS.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    return out


def _save_stats(key, val):
    d = json.loads(STATS.read_text()) if STATS.exists() else {}
    d[key] = val
    STATS.write_text(json.dumps(d, indent=1, ensure_ascii=False))


def _expression_metrics(x, notes, a4=440.0):
    """Pitch-track a solo render and measure scoop, intonation, vibrato and final fall."""
    from sampler.analysis import pitch_track
    t, f0, _ = pitch_track(x, fmin=120, fmax=1200)
    rows = []
    for i, (st, du, m, *_r) in enumerate(notes):
        tgt = a4 * 2 ** ((m - 69) / 12)
        c = 1200 * np.log2(f0 / tgt)
        on = (t >= st) & (t < st + 0.12)
        prev_end = notes[i - 1][0] + notes[i - 1][1] if i else -1
        legato = i > 0 and st - prev_end < 0.08
        steady = (t >= st + max(1.0, 0.3 * du)) & (t < st + du - 0.15)
        row = {'midi': m, 'legato_from_prev': bool(legato)}
        valid = on & np.isfinite(c) & (np.abs(c) < 300)      # drop octave errors
        if valid.any() and not legato:
            k0 = np.flatnonzero(valid)[0]
            row['onset_first_tracked'] = {'t_ms': round(float(t[k0] - st) * 1000, 0),
                                          'cents': round(float(c[k0]), 1)}
            row['onset_min_cents'] = round(float(c[valid].min()), 1)
            # time until the pitch stays within 15 cents of target (ignoring vibrato build-up)
            win = (t >= st) & (t < st + 0.4)
            ok = np.flatnonzero(np.abs(c[win]) < 15)
            row['scoop_settle_ms'] = round(float(t[win][ok[0]] - st) * 1000, 0) if len(ok) else None
        if steady.sum() > 40:
            cs = c[steady]
            cs = cs[np.isfinite(cs)]
            row['median_cents'] = round(float(np.median(cs)), 1)
            d = cs - np.median(cs)
            n = len(d)
            spec = np.abs(np.fft.rfft(d * np.hanning(n), 1 << 14))
            fr = np.fft.rfftfreq(1 << 14, 0.005)
            band = (fr > 3) & (fr < 9)
            row['vibrato_hz'] = round(float(fr[band][np.argmax(spec[band])]), 2)
            row['vibrato_extent_cents'] = round(float(np.percentile(np.abs(d), 95)), 1)
        rows.append(row)
    st, du, m = notes[-1][0], notes[-1][1], notes[-1][2]
    tgt = a4 * 2 ** ((m - 69) / 12)
    c = 1200 * np.log2(f0 / tgt)
    end = (t > st + du) & (t < st + du + 0.3)
    fall = float(np.nanmedian(c[end])) if np.isfinite(c[end]).any() else None
    return rows, (round(fall, 1) if fall is not None else None)


def check_choir():
    from sampler.choir import render_choir, VOWELS
    # slow melody between D4 and A5 (legato pairs + separate phrases)
    mel = [(0.4, 2.4, 62, 64, 'a'), (2.8, 1.6, 66, 70, None), (4.4, 2.8, 69, 78, None),
           (7.6, 1.4, 71, 72, 'o'), (9.0, 3.0, 69, 66, None),
           (12.6, 1.8, 74, 76, 'a'), (14.4, 4.2, 81, 70, None)]
    out = {}
    for solo in (True, False):
        tag = 'solo' if solo else 'choir6'
        x, dt = _timed(render_choir, mel, solo=solo)
        s = report(f'choir_melody_{tag}', x, CHECK, STATS, extra={'render_s': round(dt, 2)})
        out[tag] = s
        _write(f'choir_melody_{tag}', x)
        if solo:
            # pitch metrics on the same seed with breath=0: identical f0 contour, but the
            # tracker is not confused by aspiration noise at the soft onsets
            rows, fall = _expression_metrics(render_choir(mel, solo=True, breath=0.0), mel)
            print('solo expression (pitch-tracked, breath=0 render of the same seed):')
            for r in rows:
                print('   ', r)
            print('    phrase-end fall (median cents, 0-300 ms after last note end):', fall)
            _save_stats('choir_solo_expression', {'notes': rows, 'final_fall_cents': fall})
    # breath objectively: same seed with breath=0 gives the identical voiced part, so the
    # difference is exactly the aspiration/air noise -> harmonics-to-noise ratio
    note = [(0.2, 3.0, 69, 72, 'a')]
    v0 = render_choir(note, solo=True, breath=0.0).astype(np.float64)
    hnr = {}
    for b in (0.15, 0.3):
        vb = render_choir(note, solo=True, breath=b / 1.6).astype(np.float64)   # solo x1.6
        nz = vb - v0
        seg = slice(int(1.0 * SR), int(3.0 * SR))
        from scipy import signal as _sg
        fv, pv = _sg.welch(v0[seg].mean(1), SR, nperseg=4096)
        fn, pn = _sg.welch(nz[seg].mean(1), SR, nperseg=4096)
        band = (fv >= 2000) & (fv < 8000)
        hnr[str(b)] = {'hnr_total_db': round(float(10 * np.log10(pv.sum() / pn.sum())), 1),
                       'hnr_2k_8k_db': round(float(10 * np.log10(pv[band].sum() / pn[band].sum())), 1)}
    print('breath (solo, sustained A4 "a"): harmonics-to-noise', hnr)
    _save_stats('choir_breath_hnr', hnr)
    # D major triad, three parts, 6 singers each
    chord = [(0.3, 8.0, 62, 72, 'a'), (0.3, 8.0, 66, 70, 'a'), (0.3, 8.0, 69, 74, 'a')]
    x, dt = _timed(render_choir, chord, voices=6)
    report('choir_Dmajor_3parts', x, CHECK, STATS, extra={'render_s': round(dt, 2)})
    _write('choir_Dmajor_3parts', x)
    # vowels and morphs: a o u m, then hum opening to ah
    vw = [(0.3, 2.0, 67, 72, 'a'), (2.5, 2.0, 67, 72, 'o'), (4.7, 2.0, 67, 72, 'u'),
          (6.9, 2.0, 67, 72, 'm'), (9.2, 4.0, 67, 72, 'm>a')]
    x, dt = _timed(render_choir, vw, voices=6, legato=False)
    report('choir_vowels_a_o_u_m_morph', x, CHECK, STATS, extra={'render_s': round(dt, 2)})
    _write('choir_vowels_a_o_u_m_morph', x)
    # formant-region harmonic levels vs the Csound soprano table (steady A4, no vibrato)
    csound = {'a': [0, -6, -32, -20, -50], 'e': [0, -20, -15, -40, -56], 'i': [0, -12, -26, -26, -44],
              'o': [0, -11, -22, -22, -50], 'u': [0, -16, -35, -40, -60]}
    tab = {}
    for v in 'aeiou':
        x = render_choir([(0.2, 3.0, 69, 80, v)], solo=True, breath=0.0, vibrato_depth=(0, 0),
                         jitter=0, shimmer=0, presence_cut_db=0).mean(axis=1)
        seg = x[int(1.2 * SR):int(2.8 * SR)].astype(np.float64) * np.hanning(int(1.6 * SR))
        X = 20 * np.log10(np.abs(np.fft.rfft(seg, 1 << 20)) + 1e-12)
        f = np.fft.rfftfreq(1 << 20, 1 / SR)
        h = np.array([X[(f > k * 440 * 0.985) & (f < k * 440 * 1.015)].max() for k in range(1, 40)])
        F = VOWELS['soprano'][v][0]
        lv = np.array([max(h[k - 1] for k in range(1, 40) if abs(k * 440 - Fi) < 0.5 * 440 + 0.06 * Fi)
                       for Fi in F])
        lv = lv - max(lv[0], h.max())
        tab[v] = {'synth_db': [round(float(a), 1) for a in lv], 'csound_db': csound[v],
                  'mean_diff_F3_F5_db': round(float(np.mean(lv[2:] - np.array(csound[v][2:]))), 1)}
        print(f'formant levels {v}: synth {tab[v]["synth_db"]}  csound {csound[v]}  '
              f'mean diff F3-F5 {tab[v]["mean_diff_F3_F5_db"]:+.1f} dB')
    _save_stats('choir_formant_levels_vs_csound', tab)
    return out


def check_nature():
    from sampler.nature import ambience, sfx, distant, KINDS_AMBIENCE, KINDS_SFX, _REC_AMB, _field
    from sampler.analysis import stats, detect_clicks, plot
    wav_keep = {'stream', 'forest_wind', 'birds_morning', 'crickets', 'stream_synth',
                'bamboo_wind_synth', 'birds_synth', 'night_insects'}
    for k in KINDS_AMBIENCE:
        x, dt = _timed(ambience, k, 30.0, seed=3)
        extra = {'render_s': round(dt, 2), 'source': KINDS_AMBIENCE[k]}
        if k in _REC_AMB:
            srcs = [_field(f) for f in _REC_AMB[k][0] if _field(f) is not None]
            n_src = sum(len(s_) for s_ in srcs) / SR
            c_src = sum(len(detect_clicks(s_ if s_.shape[1] == 2 else np.repeat(s_, 2, 1))) for s_ in srcs)
            extra['source_click_candidates_per_s'] = round(c_src / n_src, 3)
        s_ = report(f'amb_{k}', x, CHECK, STATS, extra=extra)
        print(f'    clicks/s in render {s_["click_candidates"] / 30:.3f}  vs source '
              f'{extra.get("source_click_candidates_per_s", "n/a")}   render {dt:.2f}s for 30 s')
        if k in wav_keep:
            _write(f'amb_{k}', x[:int(20 * SR)])
    # loop seams: tile a loop=True render twice and inspect the joint
    seam = {}
    for k in ('stream', 'forest_wind', 'birds_morning', 'stream_synth', 'wind_synth'):
        y = ambience(k, 12.0, seed=5, loop=True).astype(np.float64)
        tiled = np.concatenate([y, y])
        j = len(y)
        d_seam = float(np.abs(tiled[j] - tiled[j - 1]).max())
        d_typ = float(np.median(np.abs(np.diff(y, axis=0))))
        d_p99 = float(np.percentile(np.abs(np.diff(y, axis=0)), 99.9))
        near = [t for t in detect_clicks(tiled.astype(np.float32)) if abs(t - j / SR) < 0.05]
        w = int(0.2 * SR)
        r1 = 20 * np.log10(np.sqrt((tiled[j - w:j] ** 2).mean()))
        r2 = 20 * np.log10(np.sqrt((tiled[j:j + w] ** 2).mean()))
        seam[k] = {'seam_step_over_p99.9_step': round(d_seam / d_p99, 3),
                   'seam_step_over_median_step': round(d_seam / max(d_typ, 1e-12), 2),
                   'clicks_within_50ms_of_seam': len(near), 'rms_200ms_before_after_db': [round(r1, 1), round(r2, 1)]}
        print(f'loop seam {k:14}: step/p99.9 {seam[k]["seam_step_over_p99.9_step"]}, '
              f'clicks near seam {len(near)}, RMS before/after {r1:.1f}/{r2:.1f} dB')
    _save_stats('nature_loop_seams', seam)
    # one-shots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import signal as _sg
    kinds = list(KINDS_SFX)
    fig, axs = plt.subplots((len(kinds) + 2) // 3, 3, figsize=(16, 2.6 * ((len(kinds) + 2) // 3)))
    for ax, k in zip(axs.flat, kinds):
        x, dt = _timed(sfx, k, seed=0)
        s_ = report(f'sfx_{k}', x, None, STATS, extra={'render_s': round(dt, 3), 'source': KINDS_SFX[k]})
        _write(f'sfx_{k}', x)
        m = x.mean(axis=1)
        f, t, S = _sg.spectrogram(m, SR, nperseg=1024, noverlap=768)
        ax.pcolormesh(t, f / 1000, 10 * np.log10(S + 1e-14), vmin=10 * np.log10(S.max() + 1e-14) - 80,
                      shading='auto', cmap='magma')
        ax.set_ylim(0, 16)
        ax.set_title(f'{k}  peak {s_["peak_dbfs"]} dBFS, clicks {s_["click_candidates"]}', fontsize=9)
    fig.tight_layout()
    fig.savefig(CHECK / 'sfx_contact_sheet.png', dpi=70)
    plt.close(fig)
    # distance processing
    base = sfx('rooster', seed=0)
    s0 = report('sfx_rooster_near', base, None, STATS)
    dist = {}
    for a in (0.3, 0.6, 0.85, 1.0):
        y = distant(base, a)
        s_ = report(f'rooster_distant_{a}', y, CHECK if a == 0.85 else None, STATS)
        f, pw = _sg.welch(y.mean(axis=1), SR, nperseg=4096)
        cen = float((f * pw).sum() / pw.sum())
        hf = float(10 * np.log10(pw[f >= 4000].sum() / pw.sum()))
        dist[str(a)] = {'lufs': s_['lufs'], 'lufs_drop': round(s_['lufs'] - s0['lufs'], 1),
                        'spectral_centroid_hz': round(cen), 'energy_above_4k_db': round(hf, 1),
                        'lr_corr': s_['lr_corr']}
        if a == 0.85:
            _write('rooster_distant_0.85', y)
    f, pw = _sg.welch(base.mean(axis=1), SR, nperseg=4096)
    dist['0'] = {'lufs': s0['lufs'], 'spectral_centroid_hz': round(float((f * pw).sum() / pw.sum())),
                 'energy_above_4k_db': round(float(10 * np.log10(pw[f >= 4000].sum() / pw.sum())), 1)}
    print('distant(rooster):', dist)
    _save_stats('nature_distant_rooster', dist)
    y = distant(sfx('dog', seed=0, barks=2), 0.85)
    report('dog_distant_0.85', y, CHECK, STATS)
    _write('dog_distant_0.85', y)


# =============================================================================== round 2
def _env_db(x, win_s):
    from scipy import ndimage
    p = (np.asarray(x, dtype=np.float64) ** 2)
    p = p.mean(axis=1) if p.ndim == 2 else p
    return 10 * np.log10(np.maximum(ndimage.uniform_filter1d(p, max(1, int(win_s * SR)), mode='constant'), 1e-20))


def _chroma_fraction(y, chord_pcs, lo=200.0, hi=5000.0, tol_cents=35.0):
    """Share of 200-5000 Hz power within +-tol of the chord's pitch classes (uniform
    spectrum baseline: n_pcs * 2 * tol / 1200)."""
    from scipy import signal as sg
    m = np.asarray(y, dtype=np.float64).mean(axis=1)
    f, p = sg.welch(m, SR, nperseg=1 << 15)
    band = (f >= lo) & (f <= hi)
    pc = np.mod(69 + 12 * np.log2(f[band] / 440.0), 12)
    d = np.min(np.abs((pc[:, None] - np.array(chord_pcs)[None, :] + 6) % 12 - 6), axis=1)
    return float(p[band][d * 100 <= tol_cents].sum() / p[band].sum())


def _partials_db(y, freqs, t0, t1):
    m = np.asarray(y, dtype=np.float64)[int(t0 * SR):int(t1 * SR)].mean(axis=1)
    X = np.abs(np.fft.rfft(m * np.hanning(len(m)))) ** 2
    f = np.fft.rfftfreq(len(m), 1 / SR)
    return [10 * np.log10(X[np.abs(f - fr) < 8].sum() + 1e-30) for fr in freqs]


def _pedal_chords():
    chords = [[48, 55, 60, 64, 67], [45, 52, 57, 60, 64], [41, 48, 53, 57, 60],
              [43, 50, 55, 59, 62], [36, 48, 55, 60, 64, 72]]
    prog, ped = [], []
    for i, ch in enumerate(chords):
        t0 = i * 2.2
        for j, m in enumerate(ch):
            prog.append((t0 + 0.012 * j, 0.9, m, 58 + 4 * j - 3 * (i % 2)))
        ped.append((t0 + 0.12, t0 + 2.2 + 0.02 if i < len(chords) - 1 else t0 + 5.0))
    return prog, ped


def check_symp():
    """Piano pedal sympathetic resonance."""
    from sampler.piano import render_piano
    prog, ped = _pedal_chords()
    e = lambda z: 10 * np.log10((np.asarray(z, dtype=np.float64) ** 2).sum())   # noqa: E731
    res = {}
    for felt in (False, True):
        tag = 'felt' if felt else 'normal'
        dry = render_piano(prog, pedal=ped, felt=felt, sympathetic=0.0, duration=16)
        x, dt = _timed(render_piano, prog, pedal=ped, felt=felt, duration=16)      # default 0.35
        r = x - dry
        from sampler.analysis import detect_clicks
        res[tag] = {'resonance_re_dry_db': round(e(r) - e(dry), 1),
                    'resonance_only_click_candidates_x10': len(detect_clicks(r * 10)),
                    'render_s': round(dt, 2)}
        report(f'piano_symp_chords_{tag}', x, CHECK, STATS, extra=res[tag])
        _write(f'piano_symp_chords_{tag}', x)
        if not felt:
            report('piano_symp_resonance_only', r * 10, CHECK, None)
            _write('piano_symp_resonance_only_x10', r * 10)
    x1 = render_piano(prog, pedal=ped, sympathetic=1.0, duration=16)
    dry_n = render_piano(prog, pedal=ped, sympathetic=0.0, duration=16)
    res['amount_1.0_re_dry_db'] = round(e(x1 - dry_n) - e(dry_n), 1)
    a = render_piano(prog, sympathetic=0.35, duration=12)
    b = render_piano(prog, sympathetic=0.0, duration=12)
    res['no_pedal_identical'] = bool(np.array_equal(a, b))
    # single C4 with pedal: where does the resonance sit?
    d = render_piano([(0.0, 3.0, 60, 70)], pedal=[(0.0, 3.0)], sympathetic=0.0, duration=4)
    w = render_piano([(0.0, 3.0, 60, 70)], pedal=[(0.0, 3.0)], sympathetic=1.0, duration=4)
    r = (w - d)[int(0.5 * SR):int(3.0 * SR)].mean(axis=1).astype(np.float64)
    X = np.abs(np.fft.rfft(r * np.hanning(len(r)), 1 << 19))
    f = np.fft.rfftfreq(1 << 19, 1 / SR)
    from scipy.signal import find_peaks
    pk, _ = find_peaks(X, distance=40)
    top = pk[np.argsort(X[pk])[::-1][:8]]
    res['single_C4_resonance_peaks'] = [{'hz': round(float(f[j]), 1),
                                         'midi': round(float(69 + 12 * np.log2(f[j] / 440)), 2),
                                         'db': round(float(20 * np.log10(X[j] / X[top[0]])), 1)} for j in top]
    res['single_C4_resonance_re_dry_db_at_0.35'] = round(
        e(render_piano([(0.0, 3.0, 60, 70)], pedal=[(0.0, 3.0)], duration=4) - d) - e(d), 1)
    print('sympathetic:', json.dumps(res, ensure_ascii=False))
    _save_stats('piano_sympathetic', res)


def check_choir2():
    from sampler.choir import render_choir, choir_pad
    from scipy import signal as sg
    out = {}
    mel = [(0.4, 2.4, 62, 64, 'a'), (2.8, 1.6, 66, 70, None), (4.4, 2.8, 69, 78, None),
           (7.6, 1.4, 71, 72, 'o'), (9.0, 3.0, 69, 66, None), (12.6, 1.8, 74, 76, 'a'), (14.4, 4.2, 81, 70, None)]
    chord = [(0.3, 8.0, 62, 72, 'a'), (0.3, 8.0, 66, 70, 'a'), (0.3, 8.0, 69, 74, 'a')]
    vw = [(0.3, 2.0, 67, 72, 'a'), (2.5, 2.0, 67, 72, 'o'), (4.7, 2.0, 67, 72, 'u'),
          (6.9, 2.0, 67, 72, 'm'), (9.2, 4.0, 67, 72, 'm>a')]
    for spec in ('v1', 'v2'):
        row = {}
        for name, notes, kw in (('solo', mel, dict(solo=True)), ('choir6', mel, {}), ('Dmajor', chord, {}),
                                ('vowels', vw, dict(legato=False))):
            x, dt = _timed(render_choir, notes, spectrum=spec, **kw)
            s = report(f'choir2_{name}_{spec}', x, CHECK if spec == 'v2' else None, None)
            row[name] = {'harsh_2k5_4k_db': s['harsh_2k5_4k_rel_total_db'], 'lufs': s['lufs'],
                         'band_5k_10k_db': s['bands_db']['5000-10000'],
                         'band_10k_20k_db': s['bands_db']['10000-20000'],
                         'peak_dbfs': s['peak_dbfs'], 'clicks': s['click_candidates'],
                         'clipped': s['clipped_samples'], 'render_s': round(dt, 2)}
            if spec == 'v2' and name in ('vowels', 'choir6'):
                _write(f'choir2_{name}_v2', x)
        # harmonic bands with breath 0 (solo A4, 1.0-2.5 s)
        hb = {}
        for v in 'aiou':
            y = render_choir([(0.0, 3.0, 69, 80, v)], solo=True, breath=0.0, seed=1, spectrum=spec)
            seg = y[int(1.0 * SR):int(2.5 * SR), 0].astype(np.float64)
            X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
            f = np.fft.rfftfreq(len(seg), 1 / SR)
            hb[v] = {f'{lo // 1000 if lo % 1000 == 0 else lo / 1000}-{hi // 1000}k':
                     round(float(10 * np.log10(X[(f >= lo) & (f < hi)].sum() / X.sum() + 1e-30)), 1)
                     for lo, hi in ((2500, 4000), (5000, 8000), (8000, 12000))}
        row['harmonic_bands_breath0_re_total_db'] = hb
        # formant levels vs Csound table
        csound = {'a': [0, -6, -32, -20, -50], 'e': [0, -20, -15, -40, -56], 'i': [0, -12, -26, -26, -44],
                  'o': [0, -11, -22, -22, -50], 'u': [0, -16, -35, -40, -60]}
        from sampler.choir import VOWELS
        tab = {}
        for v in 'aeiou':
            x = render_choir([(0.2, 3.0, 69, 80, v)], solo=True, breath=0.0, vibrato_depth=(0, 0),
                             jitter=0, shimmer=0, presence_cut_db=0, spectrum=spec).mean(axis=1)
            seg = x[int(1.2 * SR):int(2.8 * SR)].astype(np.float64) * np.hanning(int(1.6 * SR))
            X = 20 * np.log10(np.abs(np.fft.rfft(seg, 1 << 20)) + 1e-12)
            f = np.fft.rfftfreq(1 << 20, 1 / SR)
            h = np.array([X[(f > k * 440 * 0.985) & (f < k * 440 * 1.015)].max() for k in range(1, 40)])
            lv = np.array([max(h[k - 1] for k in range(1, 40) if abs(k * 440 - Fi) < 0.5 * 440 + 0.06 * Fi)
                           for Fi in VOWELS['soprano'][v][0]])
            lv = lv - max(lv[0], h.max())
            tab[v] = round(float(np.mean(lv[2:] - np.array(csound[v][2:]))), 1)
        row['formant_F3_F5_mean_diff_vs_csound_db'] = tab
        # breath HNR (solo A4 'a')
        note = [(0.2, 3.0, 69, 72, 'a')]
        v0 = render_choir(note, solo=True, breath=0.0, spectrum=spec).astype(np.float64)
        hn = {}
        for b in (0.15, 0.3):
            nz = render_choir(note, solo=True, breath=b / 1.6, spectrum=spec).astype(np.float64) - v0
            sl = slice(int(1.0 * SR), int(3.0 * SR))
            fv, pv = sg.welch(v0[sl].mean(1), SR, nperseg=4096)
            _, pn = sg.welch(nz[sl].mean(1), SR, nperseg=4096)
            bd = (fv >= 2000) & (fv < 8000)
            hn[str(b)] = {'hnr_total_db': round(float(10 * np.log10(pv.sum() / pn.sum())), 1),
                          'hnr_2k_8k_db': round(float(10 * np.log10(pv[bd].sum() / pn[bd].sum())), 1)}
        row['breath_hnr'] = hn
        out[spec] = row
        print(f'choir spectrum {spec}:', json.dumps(row, ensure_ascii=False))
    # pad style
    pn = [(0.5, 9.0, 62, 56, 'u>a'), (0.5, 9.0, 66, 52, 'u>a'), (0.5, 9.0, 69, 54, 'u>a'), (0.5, 9.0, 57, 50, 'o')]
    x, dt = _timed(choir_pad, pn)
    s = report('choir2_pad_style', x, CHECK, STATS, extra={'render_s': round(dt, 2)})
    _write('choir2_pad_style', x)
    z = render_choir(pn, voices=8)
    s0 = report('choir2_pad_notes_default_style', z, None, None)
    e = _env_db(x, 0.05)
    on = int(0.5 * SR)
    steps = np.diff(e[on:on + int(1.0 * SR)][::int(0.01 * SR)])
    out['pad_style'] = {'lr_corr': s['lr_corr'], 'lr_corr_default_style': s0['lr_corr'],
                        'harsh_2k5_4k_db': s['harsh_2k5_4k_rel_total_db'],
                        'harsh_default_style_db': s0['harsh_2k5_4k_rel_total_db'],
                        'max_10ms_env_step_first_1s_db': round(float(np.abs(steps).max()), 2),
                        'clicks': s['click_candidates'], 'render_s': round(dt, 2)}
    print('choir pad style:', out['pad_style'])
    _save_stats('choir2', out)


def check_fx():
    from sampler.fx import reverse_swell, granular, shimmer, tape, impact, pitch_shift, IMPACT_KINDS
    from sampler.mix import reverb
    from sampler.piano import render_piano
    from sampler.choir import render_choir
    from scipy import signal as sg
    res = {}
    # ---- pitch shifter on a sine
    t = np.arange(2 * SR) / SR
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    ps = {}
    for st in (12, 7, -12):
        y = pitch_shift(sine, st)[:, 0].astype(np.float64)
        seg = y[int(0.5 * SR):int(1.5 * SR)] * np.hanning(SR)
        X = np.abs(np.fft.rfft(seg, 1 << 21))
        f = np.fft.rfftfreq(1 << 21, 1 / SR)
        k = int(np.argmax(X))
        ps[str(st)] = {'peak_hz': round(float(f[k]), 3),
                       'err_cents_vs_12tet': round(float(1200 * np.log2(f[k] / (440 * 2 ** (st / 12)))), 2),
                       'max_spur_dbc': round(float(20 * np.log10(X[np.abs(f - f[k]) > 30].max() / X[k])), 1),
                       'level_change_db': round(float(20 * np.log10(np.sqrt((y[SR // 2:3 * SR // 2] ** 2).mean()) / np.sqrt(0.125))), 2)}
    res['pitch_shift_sine_440'] = ps
    print('pitch_shift:', ps)
    # ---- reverse swell
    chord = render_piano([(0.0, 3.0, m, 60) for m in (50, 57, 62, 66, 69)], pedal=[(0.0, 3.0)], duration=4.0)
    rs = {}
    for L in (1.0, 2.5):
        sw, dt = _timed(reverse_swell, chord, L)
        e100 = _env_db(sw, 0.1)
        e20 = _env_db(sw, 0.02)
        a, b = int(0.3 * len(sw)), len(sw) - int(0.05 * SR)
        tt = np.arange(a, b) / SR
        fit = np.polyfit(tt, e100[a:b], 1)
        last = e20[len(sw) - int(0.1 * SR):len(sw) - int(0.015 * SR)]
        s = report(f'fx_reverse_swell_{L}s', sw, CHECK, STATS, extra={'render_s': round(dt, 2)})
        rs[str(L)] = {'samples': len(sw), 'exact_length': len(sw) == int(round(L * SR)),
                      'first_last_sample': [s['first_sample'], s['last_sample']],
                      'rise_30pct_to_end_db': round(float(e100[b] - e100[a]), 1),
                      'slope_db_per_s': round(float(fit[0]), 1),
                      'fit_residual_sd_db': round(float((e100[a:b] - np.polyval(fit, tt)).std()), 2),
                      'last_100ms_min_re_clip_max_db': round(float(last.min() - e20.max()), 1),
                      'env20ms_max_before_end_ms_seeds_0_1_2': [
                          round(1000 * (int(round(L * SR)) - int(np.argmax(_env_db(reverse_swell(chord, L, seed=sd), 0.02)))) / SR, 1)
                          for sd in (0, 1, 2)],
                      'clicks': s['click_candidates'], 'clipped': s['clipped_samples']}
        _write(f'fx_reverse_swell_{L}s', sw)
    # swell followed by the chord it leads into (listening example)
    from sampler.mix import place, place_end, new_track
    ex = place_end(new_track(2.5), reverse_swell(chord, 2.5), 2.5, gain_db=-4)
    ex = place(ex, chord, 2.5 - 0.012)                      # hammer lands at 2.5 s
    _write('fx_reverse_swell_into_chord', ex)
    res['reverse_swell'] = rs
    print('reverse_swell:', rs)
    # ---- granular
    gr = {}
    pcs = [2, 6, 9]
    from sampler.nature import ambience, _field
    from sampler.analysis import detect_clicks
    for tag, kw in (('unpitched', {}), ('chord_quantised', dict(chord=[62, 66, 69])),
                    ('chord_tonal0.6', dict(chord=[62, 66, 69], tonal=0.6)),
                    ('chord_tonal0.8_oct', dict(chord=[62, 66, 69], tonal=0.8, pitch=[0, 12]))):
        y, dt = _timed(granular, 'stream', 20.0, seed=1, **kw)
        s = report(f'fx_granular_{tag}', y, CHECK, STATS, extra={'render_s': round(dt, 2)})
        st = _env_db(y, 1.0)[int(1 * SR):int(19 * SR)][::SR // 10]
        gr[tag] = {'chroma_fraction_D_F#_A': round(_chroma_fraction(y, pcs), 3),
                   'level_1s_sd_db': round(float(st.std()), 2), 'lr_corr': s['lr_corr'],
                   'clicks_per_s': round(s['click_candidates'] / 20, 3), 'clipped': s['clipped_samples'],
                   'harsh_2k5_4k_db': s['harsh_2k5_4k_rel_total_db'], 'render_s': round(dt, 2)}
        _write(f'fx_granular_{tag}', y)
    src = _field('stream_molln_bachlauf')
    gr['baseline_uniform_chroma_fraction'] = round(3 * 70 / 1200, 3)
    gr['source_stream_clicks_per_s'] = round(len(detect_clicks(np.repeat(src, 2, 1))) / (len(src) / SR), 3)
    res['granular'] = gr
    print('granular:', gr)
    # ---- shimmer
    sh = {}
    hi = render_piano([(0.1, 2.0, m, 50) for m in (86, 90, 93)] + [(1.6, 2.0, 98, 44)], pedal=[(0.0, 3.5)], duration=4.0)
    ch = render_choir([(0.2, 4.0, 69, 60, 'a'), (0.2, 4.0, 74, 60, 'a')], voices=6)
    for name, x in (('piano_high', hi), ('choir', ch)):
        y, dt = _timed(shimmer, x, mix=0.35)
        s = report(f'fx_shimmer_{name}', y, CHECK, STATS, extra={'render_s': round(dt, 2)})
        b = report(f'fx_shimmer_{name}_plain_bloom_ref', reverb(x, 'bloom', mix=0.35), None, None)
        d = report(f'fx_shimmer_{name}_dry_ref', x, None, None)
        sh[name] = {'harsh_2k5_4k_db': s['harsh_2k5_4k_rel_total_db'],
                    'harsh_plain_reverb_db': b['harsh_2k5_4k_rel_total_db'],
                    'harsh_dry_db': d['harsh_2k5_4k_rel_total_db'],
                    'clicks': s['click_candidates'], 'clipped': s['clipped_samples'],
                    'lr_corr': s['lr_corr'], 'render_s': round(dt, 2)}
        _write(f'fx_shimmer_{name}', y)
    tb = np.arange(int(0.5 * SR)) / SR
    burst = np.zeros(SR, np.float32)
    burst[:len(tb)] = 0.3 * np.sin(2 * np.pi * 440 * tb) * np.hanning(len(tb))
    for label, y in (('plain_bloom', reverb(burst, 'bloom', mix=1.0)), ('shimmer', shimmer(burst, mix=1.0))):
        lv = _partials_db(y, (440, 880, 1760, 3520), 1.0, 3.0)
        sh[f'sine_burst_tail_1_3s_{label}_880_1760_3520_re_440_db'] = [round(v - lv[0], 1) for v in lv[1:]]
    res['shimmer'] = sh
    print('shimmer:', sh)
    # ---- tape
    tp = {}
    x1k = 0.5 * np.sin(2 * np.pi * 1000 * t)
    for d in (0, 3, 6, 9, 12):
        y = tape(x1k, drive=d, wow=0, flutter=0)
        m = y[:, 0].astype(np.float64)[SR // 2:SR // 2 + SR]
        X = np.abs(np.fft.rfft(m * np.blackman(len(m)), 1 << 20))
        f = np.fft.rfftfreq(1 << 20, 1 / SR)
        lv = lambda fr: X[(f > fr - 5) & (f < fr + 5)].max()   # noqa: E731
        tp[f'drive_{d}dB_H2_H3_H5_dBc_at_-6dBFS_1k'] = [round(float(20 * np.log10(lv(k * 1000) / lv(1000) + 1e-15)), 1) for k in (2, 3, 5)]
        tp[f'drive_{d}dB_fundamental_change_db'] = round(float(20 * np.log10(lv(1000) / (0.25 * np.blackman(SR).sum()))), 2)
    x7 = 0.7 * np.sin(2 * np.pi * 7000 * t)
    y = tape(x7, drive=12, wow=0, flutter=0)
    m = y[:, 0].astype(np.float64)[SR // 2:SR // 2 + SR]
    X = 20 * np.log10(np.abs(np.fft.rfft(m * np.blackman(len(m)), 1 << 20)) + 1e-12)
    f = np.fft.rfftfreq(1 << 20, 1 / SR)
    X -= X.max()
    harm = np.zeros_like(f, bool)
    for k in range(1, 4):
        harm |= np.abs(f - 7000 * k) < 30
    tp['aliasing_7k_-3dBFS_drive12_max_nonharmonic_dbc'] = round(float(X[(~harm) & (f > 50)].max()), 1)
    x3 = 0.5 * np.sin(2 * np.pi * 3000 * np.arange(20 * SR) / SR)
    for w, fl in ((0.05, 0.0), (0.0, 0.02), (0.05, 0.02)):
        y = tape(x3, drive=0, wow=w, flutter=fl)[:, 0].astype(np.float64)
        a = sg.hilbert(y)
        inst = np.diff(np.unwrap(np.angle(a))) * SR / (2 * np.pi)
        inst = sg.sosfiltfilt(sg.butter(2, 40 / (SR / 2), output='sos'), inst)
        dev = ((inst - 3000) / 3000 * 100)[2 * SR:-2 * SR]
        tp[f'wow{w}_flutter{fl}_measured_peak_speed_dev_pct'] = round(float(np.percentile(np.abs(dev), 99.9)), 4)
    z = tape(np.zeros(5 * SR), hiss=-66)
    tp['hiss_-66_measured_rms_dbfs'] = round(float(20 * np.log10(np.sqrt((z.astype(np.float64) ** 2).mean()))), 2)
    tp['deterministic'] = bool(np.array_equal(tape(x1k, seed=3, hiss=-70), tape(x1k, seed=3, hiss=-70)))
    scale = [(i * 0.45, 0.42, m, 64 + 3 * i) for i, m in enumerate([60, 62, 64, 65, 67, 69, 71, 72])]
    pn = render_piano(scale, duration=4.5)
    y, dt = _timed(tape, pn, drive=6, wow=0.05, flutter=0.02, hiss=-68)
    s = report('fx_tape_piano', y, CHECK, STATS, extra={'render_s': round(dt, 2)})
    tp['piano_example'] = {'lufs_in': report('fx_tape_piano_in', pn, None, None)['lufs'], 'lufs_out': s['lufs'],
                           'clicks': s['click_candidates'], 'clipped': s['clipped_samples'], 'render_s': round(dt, 2)}
    _write('fx_tape_piano', y)
    res['tape'] = tp
    print('tape:', tp)
    # ---- impact
    im = {}
    for k in IMPACT_KINDS:
        y, dt = _timed(impact, k)
        s = report(f'fx_impact_{k}', y, CHECK, STATS, extra={'render_s': round(dt, 2)})
        m = y.mean(axis=1).astype(np.float64)
        f, p = sg.welch(m, SR, nperseg=8192)
        e = _env_db(y, 0.02)
        ipk = int(np.argmax(e))
        sch = np.cumsum((m[ipk:] ** 2)[::-1])[::-1]
        sdb = 10 * np.log10(sch / sch[0] + 1e-30)
        i5, i35 = int(np.argmax(sdb < -5)), int(np.argmax(sdb < -35))
        im[k] = {'centroid_hz': round(float((f * p).sum() / p.sum())), 'harsh_2k5_4k_db': s['harsh_2k5_4k_rel_total_db'],
                 'reach_-6dB_of_env_peak_ms': round(1000 * int(np.argmax(e > e.max() - 6)) / SR, 1),
                 'env_peak_ms': round(1000 * ipk / SR), 'schroeder_T30_from_peak_s': round(2 * (i35 - i5) / SR, 2) if i35 > i5 else None,
                 'lufs': s['lufs'], 'clicks': s['click_candidates'], 'clipped': s['clipped_samples'],
                 'render_s': round(dt, 2)}
        _write(f'fx_impact_{k}', y)
    res['impact'] = im
    print('impact:', im)
    _save_stats('fx', res)


def check_pad():
    from sampler.pads import pad, sub
    from sampler.analysis import pitch_track
    from scipy import signal as sg
    res = {}
    notes = [(0.0, 10.0, 'D2', 76), (0.0, 10.0, 'D3', 80), (0.0, 10.0, 'A3', 72), (0.0, 10.0, 'F#4', 70)]
    x, dt = _timed(pad, notes, seed=2)
    s = report('pad_Dmajor', x, CHECK, STATS, extra={'render_s': round(dt, 2)})
    _write('pad_Dmajor', x)
    m = x.mean(axis=1).astype(np.float64)
    cen = []
    for i in range(2, 10):
        f, p = sg.welch(m[i * SR:(i + 1) * SR], SR, nperseg=4096)
        cen.append(float((f * p).sum() / p.sum()))
    cen = np.array(cen)
    y = pad([(0.0, 8.0, 'A3', 80)], seed=3)
    tt, f0, _ = pitch_track(y, fmin=150, fmax=300)
    sel = (tt > 2) & (tt < 7.5) & np.isfinite(f0)
    c = 1200 * np.log2(f0[sel] / 220.0)
    lv = []
    for mm in ('D2', 'A3', 'F#4', 'D5'):
        z = pad([(0.0, 8.0, mm, 80)], seed=3)[3 * SR:7 * SR].astype(np.float64)
        lv.append(round(float(20 * np.log10(np.sqrt((z ** 2).mean()))), 1))
    res['pad'] = {'centroid_per_s_hz': [round(v) for v in cen],
                  'centroid_range_pct': round(float(100 * (cen.max() - cen.min()) / cen.mean()), 1),
                  'A3_pitch_median_cents': round(float(np.median(c)), 2),
                  'A3_pitch_p5_p95_cents': [round(float(np.percentile(c, 5)), 1), round(float(np.percentile(c, 95)), 1)],
                  'single_note_vel80_rms_dbfs_D2_A3_F#4_D5': lv,
                  'lr_corr': s['lr_corr'], 'clicks': s['click_candidates'], 'clipped': s['clipped_samples'],
                  'render_s_10s_4notes': round(dt, 2)}
    z, dt = _timed(sub, 'D1', 6.0)
    s = report('sub_D1', z, CHECK, STATS)
    X = np.abs(np.fft.rfft(z[SR:5 * SR, 0].astype(np.float64) * np.blackman(4 * SR)))
    f = np.fft.rfftfreq(4 * SR, 1 / SR)
    f1 = 440 * 2 ** ((26 - 69) / 12)
    h1 = X[np.abs(f - f1) < 1].max()
    res['sub'] = {'H2_H3_dBc': [round(float(20 * np.log10(X[np.abs(f - k * f1) < 1].max() / h1)), 1) for k in (2, 3)],
                  'first_last_sample': [s['first_sample'], s['last_sample']], 'clicks': s['click_candidates'],
                  'peak_dbfs': s['peak_dbfs'], 'dur_s': s['dur_s']}
    _write('sub_D1', z)
    print('pad/sub:', res)
    _save_stats('pad_sub', res)


def _demo_premaster(dur=20.0):
    """母带自检用的 20 秒小样（未过总线处理的分轨之和）：溪流 + 鸟的底、弱音钢琴和弦、合唱长音、远处一声鸡鸣。
    要用到 field 包的录音和 Salamander 钢琴。"""
    from sampler import mix
    from sampler.choir import render_choir
    from sampler.nature import ambience, distant, sfx
    from sampler.piano import render_piano
    brook = ambience('stream', dur, seed=2)
    birds = ambience('birds_morning', dur, seed=4)
    bed = mix.place(None, brook, 0.0, gain_db=-4.0)
    bed = mix.place(bed, birds, 0.0, gain_db=-7.0)
    bed = mix.highpass(bed, 90.0)
    bed = mix.reverb(bed, 'room', mix=0.12, tail=False)
    bed = mix.fade(bed, in_s=2.5, out_s=3.0)
    # D  Bm/D  G  A(sus)  D，连音踏板
    chords = [(0.8, [50, 57, 62, 66]), (4.8, [47, 54, 62, 66]), (8.8, [43, 50, 59, 62]),
              (12.4, [45, 52, 57, 61, 64]), (15.6, [38, 50, 57, 62, 66])]
    notes, pedal = [], []
    for i, (t, ch) in enumerate(chords):
        for j, m in enumerate(ch):
            notes.append((t + 0.03 * j, 3.2, m, 44 + 5 * (j == len(ch) - 1) - 2 * (0 < j < len(ch) - 1)))
        nxt = chords[i + 1][0] if i + 1 < len(chords) else dur
        pedal.append((t + 0.1, nxt + 0.03))
    piano = mix.reverb(render_piano(notes, pedal=pedal, felt=True, duration=dur + 2), 'hall', mix=0.28)
    choir = render_choir([(6.0, 10.5, 62, 58, 'm>a'), (6.0, 10.5, 66, 54, 'm>a'), (6.0, 10.5, 69, 60, 'm>a')],
                         voices=6, breath=0.15, width=0.8, attack=1.6, release=1.2)
    choir = mix.reverb(choir, 'valley', mix=0.45)
    rooster = distant(sfx('rooster', seed=0), amount=0.85)
    out = mix.new_track(dur)
    out = mix.place(out, bed, 0.0, gain_db=0.0)
    out = mix.place(out, piano, 0.0, gain_db=-1.0)
    out = mix.place(out, choir, 0.0, gain_db=-5.0)
    out = mix.place(out, rooster, 10.6, gain_db=-4.0, pan=-0.35)
    return mix.fade(out[:int(dur * mix.SR)], in_s=0.05, out_s=2.5)


def check_master():
    from sampler import mix
    res = {}
    pre = _demo_premaster()
    res['premaster'] = {'side_over_mid_below_120Hz_db': round(mix._lf_side_ratio_db(pre, 120.0), 1),
                        'side_over_mid_below_80Hz_db': round(mix._lf_side_ratio_db(pre, 80.0), 1)}
    for name, fn in (('old_chain', lambda z: mix.normalize_lufs(mix.compress(z, threshold_db=-20, ratio=1.8, attack_ms=30, release_ms=300), -16.0, -1.0)),
                     ('master', lambda z: mix.master(z, report=True))):
        out, dt = _timed(fn, pre)
        info = None
        if isinstance(out, tuple):
            out, info = out
        s = report(f'master_demo_{name}', out, CHECK if name == 'master' else None, STATS)
        res[name] = {'lufs': s['lufs'], 'true_peak_dbtp': s['true_peak_dbtp'], 'clipped': s['clipped_samples'],
                     'lr_corr': s['lr_corr'], 'harsh_2k5_4k_db': s['harsh_2k5_4k_rel_total_db'],
                     'side_over_mid_below_120Hz_db': round(mix._lf_side_ratio_db(out, 120.0), 1),
                     'side_over_mid_below_80Hz_db': round(mix._lf_side_ratio_db(out, 80.0), 1),
                     'render_s': round(dt, 2)}
        if info:
            res[name]['info'] = info
            _write('master_demo', out)
    t = np.arange(20 * SR) / SR
    rng = np.random.default_rng(0)
    midp = 0.2 * np.sin(2 * np.pi * 110 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.2 * t)) + 0.05 * rng.standard_normal(len(t))
    side = 0.15 * np.sin(2 * np.pi * 55 * t + 0.3) + 0.04 * rng.standard_normal(len(t))
    x = np.stack([midp + side, midp - side], 1).astype(np.float32)
    y, info = mix.master(x, report=True)
    res['wide_bass_test'] = info
    print('master:', json.dumps(res, ensure_ascii=False))
    _save_stats('master', res)


# ============================================================================= round 3
SPB3 = 60 / 72
# a call phrase and its continuation: breathy / straight / vibrato, a fall at the end
CALL3 = [(0.30, 0.78, 69, 44, {'vowel': 'a', 'tech': 'breathy'}), (1.05, 0.36, 71, 48, {'vowel': 'a', 'tech': 'breathy'}),
         (1.38, 1.20, 74, 54, {'vowel': 'a', 'tech': 'straight'}), (2.56, 0.52, 71, 38, {'vowel': 'a', 'tech': 'breathy'}),
         (4.00, 0.80, 69, 44, {'vowel': 'a'}), (4.95, 0.38, 71, 48, {'vowel': 'a'}),
         (5.40, 1.55, 74, 52, {'vowel': 'a', 'tech': 'vibrato'}), (7.30, 0.95, 69, 42, {'vowel': 'a'}),
         (8.30, 0.70, 66, 38, {'vowel': 'a'}), (9.10, 2.60, 62, 34, {'vowel': 'a', 'tech': 'breathy', 'fall': -150})]


def _note_pitch_stats(y, notes, fmin=70.0, fmax=1000.0):
    """Per note, on its middle (35 %..end-80 ms): median cents vs target, and the natural
    movement (std of cents after removing a straight line)."""
    from sampler.analysis import pitch_track
    tt, f0, _ = pitch_track(y, fmin=fmin, fmax=fmax)
    med, mov = [], []
    for n in notes:
        st, du, m = n[0], n[1], n[2]
        sel = (tt > st + 0.35 * du) & (tt < st + du - 0.08) & np.isfinite(f0)
        if sel.sum() < 6:
            continue
        c = 1200 * np.log2(f0[sel] / (440 * 2 ** ((m - 69) / 12)))
        med.append(float(np.median(c)))
        k = np.arange(len(c))
        mov.append(float(np.std(c - np.polyval(np.polyfit(k, c, 1), k))))
    med = np.abs(np.array(med))
    return {'notes_measured': int(len(med)), 'median_abs_cents': round(float(np.median(med)), 1),
            'max_abs_cents': round(float(med.max()), 1),
            'within_note_movement_cents_median': round(float(np.median(mov)), 1)}


def _hnr_db(y, notes, win_s=0.04):
    """Harmonics-to-noise ratio (Boersma 1993 idea: normalised autocorrelation r at the
    known period, HNR = 10 log10(r / (1 - r))), median over 40 ms frames in the middle of
    each note. Real singing is typically ~15-25 dB."""
    x = y.mean(axis=1).astype(np.float64) if y.ndim == 2 else y.astype(np.float64)
    w = int(win_s * SR)
    win = np.hanning(w)
    ac_w = np.correlate(win, win, 'full')[w - 1:]
    vals = []
    for n in notes:
        st, du, m = n[0], n[1], n[2]
        T = SR / (440 * 2 ** ((m - 69) / 12))
        for c in np.arange(st + 0.35 * du, st + du - 0.08 - win_s, 0.02):
            a = int(c * SR)
            fr = x[a:a + w]
            if len(fr) < w or np.abs(fr).max() < 1e-4:
                continue
            fr = (fr - fr.mean()) * win
            r = np.correlate(fr, fr, 'full')[w - 1:] / (ac_w + 1e-12)
            r = r / (r[0] + 1e-20)
            lo, hi = int(T * 0.95), int(T * 1.05) + 1
            if hi >= w // 2:
                continue
            rr = float(np.clip(r[lo:hi].max(), 1e-4, 0.99999))
            vals.append(10 * np.log10(rr / (1 - rr)))
    return round(float(np.median(vals)), 1) if vals else None


def _raw_unit_reference(units, y, notes):
    """Per note: the RAW VocalSet recording of the unit the render used (its stable part) vs.
    the middle of the rendered note - HNR and 2.5-4 kHz level (re. total), and their
    differences (render - raw)."""
    import soundfile as sf
    from sampler._dsp import resample_sr
    from sampler.analysis import stats
    rows = []
    for u, n in zip(units, notes):
        x, sr = sf.read(SAMPLES / 'vocalset' / u['file'], dtype='float32')
        seg = resample_sr(x[int(u['t0'] * sr):int(u['t1'] * sr)], sr, SR)
        st, du = n[0], n[1]
        mid = y[int((st + 0.35 * du) * SR):int((st + du - 0.08) * SR)]
        if len(seg) < int(0.3 * SR) or len(mid) < int(0.2 * SR):
            continue
        raw = np.stack([seg, seg], 1)
        rows.append((_hnr_db(raw, [(0.0, len(seg) / SR, round(u['midi']))]),
                     _hnr_db(y, [n[:3]]),
                     stats(raw)['harsh_2k5_4k_rel_total_db'], stats(mid)['harsh_2k5_4k_rel_total_db']))
    r = np.array([[np.nan if v is None else v for v in row] for row in rows], float)
    return {'notes_compared': len(rows),
            'raw_hnr_db_median': round(float(np.nanmedian(r[:, 0])), 1),
            'render_hnr_db_median': round(float(np.nanmedian(r[:, 1])), 1),
            'hnr_diff_db_median': round(float(np.nanmedian(r[:, 1] - r[:, 0])), 1),
            'raw_2k5_4k_db_median': round(float(np.nanmedian(r[:, 2])), 1),
            'harsh_diff_db_median': round(float(np.nanmedian(r[:, 3] - r[:, 2])), 1),
            'harsh_diff_db_max_abs': round(float(np.nanmax(np.abs(r[:, 3] - r[:, 2]))), 1)}


def check_voice():
    from sampler.voice import render_voice, tintinnabuli_pair
    from sampler.choir import render_choir
    from sampler.analysis import stats
    out = {}
    plain = [(st, du, m, v) for st, du, m, v, _ in CALL3]
    old_notes = [(st, du, m, v, 'a') for st, du, m, v in plain]
    rows = {}
    for name, fn in (('real_voice', lambda: render_voice(CALL3, gender='female', seed=1, return_info=True)),
                     ('real_voice_presence-4', lambda: render_voice(CALL3, gender='female', seed=1, presence_db=-4,
                                                                    return_info=True)),
                     ('old_formant_solo', lambda: (render_choir(old_notes, solo=True, seed=1), None))):
        (y, info), dt = _timed(fn)
        s = report(f'r3_voice_call_{name}', y, CHECK if 'presence' not in name else None, STATS,
                   extra={'render_s': round(dt, 2)})
        if 'presence' not in name:
            _write(f'r3_voice_call_{name}', y)
        row = {'render_s': round(dt, 2), 'lufs': s['lufs'], 'clicks': s['click_candidates'],
               'clipped': s['clipped_samples'], 'harsh_2k5_4k_rel_total_db': s['harsh_2k5_4k_rel_total_db'],
               'harsh_2k5_4k_rel_200_2k_db': s['harsh_2k5_4k_rel_200_2k_db'],
               'hnr_db_median': _hnr_db(y, plain), **_note_pitch_stats(y, plain)}
        if info:
            ph = [p for L in info['lines'] for p in L['phrases']]
            row['singers'] = sorted({p['singer'] for p in ph})
            row['onsets'] = [p['onset'] for p in ph]
            row['releases'] = [p['release'] for p in ph]
            row['real_transitions_used'] = int(sum(sum(bool(c) for c in p['contiguous']) for p in ph))
            sh = [abs(v) for p in ph for v in p['shifts_semitones']]
            row['pitch_shift_semitones_max'] = round(float(max(sh)), 2)
            row['pitch_shift_semitones_median'] = round(float(np.median(sh)), 2)
            if name == 'real_voice':
                row['vs_raw_recordings_of_the_units_used'] = _raw_unit_reference(
                    [u for p in ph for u in p['units']], y, plain)
        rows[name] = row
    out['call_same_notes'] = rows
    print('voice call:', json.dumps(rows, ensure_ascii=False))
    # techniques on one held note (A4, 3 s): HNR and vibrato
    tech = {}
    from sampler.analysis import pitch_track
    for tq in ('straight', 'breathy', 'vibrato'):
        n = [(0.3, 3.0, 69, 60, {'vowel': 'a', 'tech': tq})]
        y = render_voice(n, gender='female', seed=3)
        tt, f0, _ = pitch_track(y, fmin=300, fmax=600)
        sel = (tt > 1.0) & (tt < 3.0) & np.isfinite(f0)
        c = 1200 * np.log2(f0[sel] / 440.0)
        c = c - np.polyval(np.polyfit(tt[sel], c, 1), tt[sel])
        sp = np.abs(np.fft.rfft(c * np.hanning(len(c)), 4096))
        fr = np.fft.rfftfreq(4096, 0.005)
        b = (fr > 3) & (fr < 9)
        tech[tq] = {'hnr_db': _hnr_db(y, [(0.3, 3.0, 69)]), 'pitch_extent_p95_cents': round(float(np.percentile(np.abs(c), 95)), 1),
                    'vibrato_rate_hz': round(float(fr[b][np.argmax(sp[b])]), 2)}
    out['techniques_A4'] = tech
    print('voice techniques:', tech)
    # male line (tenor/baritone range)
    ml = [(0.3 + i * 0.9, 0.95, m, 50, {'vowel': 'o'}) for i, m in enumerate([50, 52, 54, 57, 54, 52, 50])]
    y = render_voice(ml, gender='male', seed=4)
    st = stats(y)
    out['male_line'] = {**_note_pitch_stats(y, [n[:3] for n in ml], 70, 400), 'clicks': st['click_candidates'],
                        'hnr_db_median': _hnr_db(y, ml)}
    _write('r3_voice_male_line', y)
    print('voice male:', out['male_line'])
    # tintinnabuli choir, 3 different singers per line, against the old formant choir (6 voices)
    mel, b = [], 0.0
    for p, d in ((66, 1.0), (66, 3.0), (64, 2.0), (66, 2.0), (69, 2.0), (71, 1.8)):
        mel.append((0.5 + b * SPB3, SPB3 * d * 1.04, p, 36, {'vowel': 'a', 'tech': 'breathy'}))
        b += d
    pair = tintinnabuli_pair(mel, 'D', 1)
    (y, info), dt = _timed(render_voice, pair, voices=3, seed=2, return_info=True)
    s = report('r3_voice_tintinnabuli_choir', y, CHECK, STATS, extra={'render_s': round(dt, 2)})
    _write('r3_voice_tintinnabuli_choir', y)
    old = render_choir([(st, du, m, v, 'a', kw['part']) for st, du, m, v, kw in pair], voices=6, seed=2)
    so = report('r3_voice_tintinnabuli_old_formant_choir6', old, None, None)
    _write('r3_voice_tintinnabuli_old_formant_choir6', old)
    t_ok = all(((n[2] % 12) in (2, 6, 9)) for n in pair if n[4]['part'] == 'T')
    out['tintinnabuli_choir'] = {
        'render_s': round(dt, 2), 'singers_per_line': {L['part']: L['singers'] for L in info['lines']},
        'all_singers_distinct': len({s_ for L in info['lines'] for s_ in L['singers']}) == sum(len(L['singers']) for L in info['lines']),
        't_voice_all_in_D_triad': bool(t_ok), 'lufs': s['lufs'], 'lr_corr': s['lr_corr'], 'clicks': s['click_candidates'],
        'harsh_2k5_4k_rel_total_db': s['harsh_2k5_4k_rel_total_db'],
        'old_formant_choir6': {'lr_corr': so['lr_corr'], 'clicks': so['click_candidates'],
                               'harsh_2k5_4k_rel_total_db': so['harsh_2k5_4k_rel_total_db'], 'lufs': so['lufs']}}
    print('voice choir:', json.dumps(out['tintinnabuli_choir'], ensure_ascii=False))
    # hum 'm' and vowel morph
    y = render_voice([(0.3, 2.2, 62, 40, {'vowel': 'm'}), (2.6, 2.6, 66, 44, {'vowel': 'm>a'})], gender='female', seed=5)
    s = report('r3_voice_hum_morph', y, CHECK, STATS)
    _write('r3_voice_hum_morph', y)
    out['hum_morph'] = {'clicks': s['click_candidates'], 'lufs': s['lufs'], 'bands_db': s['bands_db']}
    _save_stats('r3_voice', out)


def _join_dips(y, joins, win=0.01):
    """Largest level dip (dB) in +-60 ms around each legato join vs. the 150 ms before it."""
    e = _env_db(y, win)
    dips = []
    for t in joins:
        a, b = int((t - 0.2) * SR), int((t - 0.05) * SR)
        c, d = int((t - 0.06) * SR), int((t + 0.06) * SR)
        if a < 0 or d > len(e):
            continue
        dips.append(float(np.median(e[a:b]) - e[c:d].min()))
    return [round(float(np.median(dips)), 1), round(float(max(dips)), 1)] if dips else None


def check_strings():
    from sampler.strings import render_strings, pluck
    from sampler.analysis import stats
    out = {}
    # four sections, three sustained chords, one bow each (legato): cb vc va vn
    notes, joins = [], []
    for b_, ch in enumerate([(38, 45, 54, 62), (35, 42, 50, 62), (31, 38, 47, 66)]):
        for p, sec in zip(ch, ('cb', 'vc', 'va', 'vn')):
            notes.append((0.5 + b_ * 4 * SPB3, SPB3 * 4.02, p, 34, {'sec': sec, 'art': 'sus'}))
        if b_:
            joins.append(0.5 + b_ * 4 * SPB3)
    (y, dt) = _timed(render_strings, notes, seed=1)
    s = report('r3_strings_sustain_chords', y, CHECK, STATS, extra={'render_s': round(dt, 2)})
    _write('r3_strings_sustain_chords', y)
    out['sustain_chords'] = {'render_s': round(dt, 2), 'lufs': s['lufs'], 'clicks': s['click_candidates'],
                             'lr_corr': s['lr_corr'], 'dip_at_bar_joins_db_median_max': _join_dips(y, joins)}
    # single section lines: pitch accuracy and join dips
    lines = {'vc': [50, 52, 54, 57, 59, 57, 54, 50], 'vn': [74, 76, 78, 81, 83, 81, 78, 74],
             'va': [62, 64, 66, 69, 66, 64, 62, 57], 'cb': [38, 40, 42, 45, 42, 38, 35, 38]}
    for sec, ps in lines.items():
        n = [(0.3 + i * SPB3, SPB3 * 1.02, p, 44, {'sec': sec, 'art': 'sus'}) for i, p in enumerate(ps)]
        y = render_strings(n, seed=2)
        fmin, fmax = (30, 200) if sec == 'cb' else (60, 1400)
        st = stats(y)
        out[f'line_{sec}'] = {**_note_pitch_stats(y, n, fmin, fmax), 'clicks': st['click_candidates'],
                              'dip_at_joins_db_median_max': _join_dips(y, [0.3 + i * SPB3 for i in range(1, len(ps))]),
                              'dip_mid_note_db_median_max': _join_dips(y, [0.3 + (i + 0.5) * SPB3 for i in range(1, len(ps) - 1)])}
        if sec == 'vc':
            _write('r3_strings_line_vc', y)
            report('r3_strings_line_vc', y, CHECK, STATS)
    # pizzicato: onset timing and pitch
    pz = [(0.3 + i * 0.5 * SPB3, 0.3, p, 60, {'sec': 'vc' if i % 2 else 'cb', 'art': 'pizz'})
          for i, p in enumerate([38, 50, 45, 50, 38, 54, 45, 50])]
    y = render_strings(pz, seed=3)
    s = report('r3_strings_pizz', y, CHECK, STATS)
    _write('r3_strings_pizz', y)
    from scipy.ndimage import maximum_filter1d
    errs = []
    for i, n in enumerate(pz):     # each note alone: attack = first moment within 15 dB of its peak
        z = render_strings([(0.3,) + tuple(n[1:])], seed=3 + i)
        e = maximum_filter1d(np.abs(z).max(axis=1), int(0.002 * SR))
        errs.append((int(np.argmax(e >= e.max() * 10 ** (-15 / 20))) / SR - 0.3) * 1000)
    out['pizz'] = {'attack_error_ms': [round(float(v), 1) for v in errs], 'clicks': s['click_candidates'],
                   'peak_dbfs': s['peak_dbfs']}
    # harmonic
    h = render_strings([(0.0, 3.0, 74, 40, {'sec': 'vc', 'art': 'harm'})])
    S = np.abs(np.fft.rfft(h[SR:2 * SR, 0] * np.hanning(SR)))
    f = np.fft.rfftfreq(SR, 1 / SR)
    part = {}
    for k in (1, 2, 3):
        i = np.argmin(np.abs(f - 587.33 * k))
        part[k] = round(float(20 * np.log10(S[i - 3:i + 4].max() / S.max())), 1)
    out['harmonic_D5'] = {**_note_pitch_stats(h, [(0.0, 3.0, 74)], 300, 900), 'partials_db': part}
    p = pluck('D3', 70)
    out['pluck'] = {'peak_dbfs': stats(p)['peak_dbfs'], 'dur_s': round(len(p) / SR, 2)}
    print('strings:', json.dumps(out, ensure_ascii=False))
    _save_stats('r3_strings', out)


def check_bowl():
    from sampler.porcelain import bowl, render_bowls, BOWL_SCALE, _fundamental
    from sampler.analysis import stats
    out = {}
    for model in ('rec', 'modal'):
        tun, t60, cen = [], [], {}
        for m in BOWL_SCALE:
            y = bowl(m, 100, 0.5, model=model, seed=1).mean(axis=1).astype(np.float64)
            ft = 440 * 2 ** ((m - 69) / 12)
            seg = y[int(0.05 * SR):int(1.5 * SR)]          # 1.45 s window: resolves the doublets
            S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 20))
            fr_ = np.fft.rfftfreq(1 << 20, 1 / SR)
            bd = (fr_ > ft * 0.9) & (fr_ < ft * 1.1)
            f = fr_[np.flatnonzero(bd)[np.argmax(S[bd])]]
            tun.append(1200 * np.log2(f / ft))
            from scipy import signal as sg
            z = sg.sosfiltfilt(sg.butter(2, [ft * 0.97, ft * 1.03], 'bandpass', fs=SR, output='sos'), y)
            e = 20 * np.log10(np.abs(sg.hilbert(z)) + 1e-9)
            i0 = int(np.argmax(e[:int(0.1 * SR)]))
            tt = np.arange(len(e) - i0) / SR
            ok = (e[i0:] > e[i0] - 35) & (tt > 0.05)
            k = np.polyfit(tt[ok], e[i0:][ok], 1)[0] if ok.sum() > 100 else 0
            t60.append(round(-60 / k, 2) if k < 0 else None)
        p2 = {}
        for hd in (0.0, 0.5, 1.0):
            vals, v2 = [], []
            for m in BOWL_SCALE:
                yy = bowl(m, 100, hd, model=model, seed=2).mean(axis=1).astype(np.float64)
                y = yy[:int(0.05 * SR)]
                S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
                fr = np.fft.rfftfreq(len(y), 1 / SR)
                vals.append((S * fr).sum() / S.sum())
                # the second mode (x2.36) re. the fundamental over the first 0.2 s
                z = yy[:int(0.2 * SR)]
                Z = np.abs(np.fft.rfft(z * np.hanning(len(z)), 1 << 16))
                fz = np.fft.rfftfreq(1 << 16, 1 / SR)
                ft = 440 * 2 ** ((m - 69) / 12)
                a1 = Z[(fz > ft * 0.97) & (fz < ft * 1.03)].max()
                a2 = Z[(fz > ft * 2.36 * 0.96) & (fz < ft * 2.36 * 1.04)].max()
                v2.append(20 * np.log10(a2 / a1))
            cen[str(hd)] = round(float(np.median(vals)))
            p2[str(hd)] = round(float(np.median(v2)), 1)
        out[model] = {'tuning_cents_max_abs': round(float(np.max(np.abs(tun))), 2),
                      'fundamental_T60_s_D5_to_B6': t60, 'attack_50ms_centroid_hz_by_hardness': cen,
                      'mode2_re_fundamental_db_by_hardness': p2}
    # water glide: pitch after the glide vs request (FFT peak on the last 0.3 s of the bend)
    y = bowl(81, 100, 0.5, water=-80, water_time=0.6).mean(axis=1).astype(np.float64)
    seg = y[int(0.8 * SR):int(1.2 * SR)]
    f = _fundamental(np.concatenate([np.zeros(int(0.05 * SR)), seg]), SR, 700, 950)
    out['water_bend_-80c_measured_cents'] = round(float(1200 * np.log2(f / 880.0)), 1)
    # a 3+2+2 figure through the set
    evs, t = [], 0.3
    for i, m in enumerate([81, 83, 86, 88, 86, 83, 81, 78, 76, 74, 76, 78, 81, 86, 90, 93, 95]):
        evs.append((t, m, 60 + (20 if i % 3 == 0 else 0), 0.35 + (0.4 if i % 3 == 0 else 0)))
        t += SPB3 / 2 * (1.5 if i % 3 == 0 else 1.0)
    y = render_bowls(evs, duration=t + 3)
    s = report('r3_bowls_figure', y, CHECK, STATS)
    _write('r3_bowls_figure', y)
    out['figure'] = {'clicks': s['click_candidates'], 'peak_dbfs': s['peak_dbfs'], 'lufs': s['lufs'],
                     'first_last_sample': [s['first_sample'], s['last_sample']]}
    print('bowl:', json.dumps(out, ensure_ascii=False))
    _save_stats('r3_bowl', out)


def check_birds():
    from sampler import birds
    from sampler.analysis import stats
    out = {}
    allr = birds.bird_calls(clear=False, sort=None)
    clear = birds.bird_calls()
    out['catalogue'] = {'calls': len(allr), 'clear': len(clear),
                        'clear_by_source': {s_: sum(r['source'] == s_ for r in clear) for s_ in birds.SOURCES.values()},
                        'clear_dur_s_min_med_max': [round(min(r['dur'] for r in clear), 2),
                                                    round(float(np.median([r['dur'] for r in clear])), 2),
                                                    round(max(r['dur'] for r in clear), 2)],
                        'clear_snr_db_median': round(float(np.median([r['snr_db'] for r in clear])), 1),
                        'clear_isolation_db_median': round(float(np.median([r['isolation_db'] for r in clear])), 1)}
    # denoise and tuning on the 12 best
    rows = []
    for r in clear[:12]:
        raw, _ = birds.bird_call(r['id'], denoise=False, tune=None, return_info=True)
        cln, info = birds.bird_call(r['id'], return_info=True)

        def snr(x):   # dynamic range inside the cut: 95th vs 10th percentile of 20 ms levels
            m_ = x.mean(axis=1).astype(np.float64)
            k = int(0.02 * SR)
            e = 10 * np.log10((m_[:len(m_) // k * k].reshape(-1, k) ** 2).mean(axis=1) + 1e-20)
            return float(np.percentile(e, 95) - np.percentile(e, 10))
        # the shift actually applied: the strongest spectral peak (Welch PSD, parabolic
        # interpolation) of the cleaned untuned clip vs. the cleaned tuned clip
        def peak_hz(x):
            from scipy import signal as sg
            f_, p_ = sg.welch(x.mean(axis=1).astype(np.float64), SR, nperseg=4096, noverlap=3072)
            band_ = (f_ > 1000) & (f_ < 12000)
            k = int(np.flatnonzero(band_)[np.argmax(p_[band_])])
            y0, y1, y2 = np.log(p_[k - 1:k + 2] + 1e-30)
            return (k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) * (f_[1] - f_[0])
        applied = 1200 * np.log2(peak_hz(cln) / peak_hz(birds.bird_call(r['id'], tune=None)))
        rows.append({'id': r['id'], 'dur': r['dur'], 'range_raw_db': round(snr(raw), 1),
                     'range_clean_db': round(snr(cln), 1), 'shift_semitones': info['shift_semitones'],
                     'measured_shift_error_cents': round(float(applied - info['shift_semitones'] * 100), 0),
                     'clicks': stats(cln)['click_candidates']})
    out['best12'] = rows
    # transcription rules
    tr_rows = []
    for r in clear[:12]:
        for st_, low in ((2.0, 84), (3.0, 79)):
            tr = birds.transcribe_call(r['id'], register=low, stretch=st_)
            if not tr:
                continue
            span = max(t + d for t, d, _, _ in tr)
            tr_rows.append({'id': r['id'], 'stretch': st_, 'notes': len(tr),
                            'all_in_scale': all(p % 12 in birds.SCALES['D'] for _, _, p, _ in tr),
                            'min_note': min(p for *_, p, _ in tr), 'register_ok': min(p for *_, p, _ in tr) >= low,
                            'span_over_call_dur': round(span / r['dur'], 2)})
    out['transcription'] = {'cases': len(tr_rows), 'all_in_scale': all(x['all_in_scale'] for x in tr_rows),
                            'register_ok': all(x['register_ok'] for x in tr_rows),
                            'notes_median': float(np.median([x['notes'] for x in tr_rows])),
                            'span_over_call_dur_median': float(np.median([x['span_over_call_dur'] for x in tr_rows]))}
    # the dialogue: bird, then the piano learns it over a resonance chord
    from sampler.piano import render_piano
    from sampler import mix
    BIRDS = [(0.4, None, -8, -0.3)]
    ECHO = [(2.0, None, 2.0, 84, 40, [62, 69, 74])]
    b = birds.render_birds(BIRDS, duration=7, echo=ECHO)
    notes, ped = birds.echo_notes(ECHO, BIRDS)
    p = render_piano(notes, ped, duration=7)
    y = b + p
    s = report('r3_birds_dialogue', y, CHECK, STATS)
    _write('r3_birds_dialogue', y)
    out['dialogue'] = {'call': birds.resolve_calls(BIRDS, ECHO)[0][0], 'piano_notes': len(notes), 'clicks': s['click_candidates']}
    print('birds:', json.dumps(out, ensure_ascii=False))
    _save_stats('r3_birds', out)


def check_sfx3():
    from sampler import nature
    from sampler.analysis import stats
    out = {}
    for kind, n in (('chirp', 4), ('flutter', 8), ('pluck', 2), ('bowl', 2), ('rooster', 3), ('dog', 2)):
        rows = []
        for i in range(n):
            x = nature.sfx(kind, seed=i)
            s = stats(x)
            rows.append({'seed': i, 'dur_s': s['dur_s'], 'peak_dbfs': s['peak_dbfs'], 'lufs': s['lufs'],
                         'clicks': s['click_candidates'], 'edges': [s['first_sample'], s['last_sample']]})
            if i == 0:
                _write(f'r3_sfx_{kind}', x)
        out[kind] = rows
    print('sfx r3:', json.dumps(out, ensure_ascii=False))
    _save_stats('r3_sfx', out)


def check_unacorda():
    from sampler.piano import render_piano
    from sampler.analysis import stats
    notes = [(0.1, 1.5, 62, 50), (0.1, 1.5, 69, 44), (0.9, 1.2, 74, 60), (1.8, 2.0, 78, 70), (2.6, 1.0, 81, 40),
             (3.2, 1.5, 57, 30)]
    ped = [(0.05, 3.0)]
    a = render_piano(notes, ped, duration=6, seed=3)
    out = {'identical_when_0': bool(np.array_equal(a, render_piano(notes, ped, duration=6, seed=3, una_corda=0.0)))}
    sa = stats(a)
    for uc in (0.5, 1.0):
        c = render_piano(notes, ped, duration=6, seed=3, una_corda=uc)
        sc = stats(c)
        out[str(uc)] = {'lufs_delta': round(sc['lufs'] - sa['lufs'], 2), 'clicks': sc['click_candidates'],
                        'band_delta_db': {k: round(sc['bands_db'][k] - sa['bands_db'][k], 1) for k in sa['bands_db']}}
        if uc == 1.0:
            _write('r3_piano_una_corda_1.0', c)
    _write('r3_piano_una_corda_0', a)
    print('una corda:', out)
    _save_stats('r3_unacorda', out)


# ============================================================================= agentreel：folk（VCSL 拨弦与钢琴）、found
def _note_cents(y, t0, t1, target, fmin=50.0, fmax=2000.0):
    """t0..t1 秒之间的中位音高相对 target（MIDI）的音分；测不到时 None。"""
    from sampler.analysis import pitch_track
    seg = y[int(t0 * SR):int(t1 * SR)]
    _, f0, _ = pitch_track(seg, fmin=fmin, fmax=fmax, win_s=0.05)
    f0 = f0[np.isfinite(f0)]
    if len(f0) < 5:
        return None
    c = 1200 * np.log2(np.median(f0) / (440.0 * 2 ** ((target - 69) / 12)))
    return round(float(c - 1200 * np.round(c / 1200)), 1)    # 八度误判不算音不准，单独看 octave_jump


def check_folk():
    """grand_piano（Steinway B）与 folk_harp：力度扫描（电平随力度单调、相邻力度没有台阶）、逐音音准、
    单音的咔哒与首尾、分解和弦的频谱图。"""
    from sampler import folk
    from sampler.analysis import stats
    out = {}
    for set_id, lo, hi, probe in (('grand_piano', 36, 96, 60), ('folk_harp', 43, 88, 67)):
        vels = np.arange(10, 128, 2)
        lv = []
        for v in vels:
            y = folk.render(set_id, [(0.05, 0.6, probe, int(v))], duration=0.6, ring_s=0.6)
            lv.append(20 * np.log10(np.sqrt((y[:int(0.3 * SR)].astype(np.float64) ** 2).mean()) + 1e-12))
        lv = np.array(lv)
        steps = np.diff(lv)
        rows, worst = [], 0.0
        for m in range(lo, hi + 1, 2):
            y = folk.render(set_id, [(0.05, 1.0, m, 80)], duration=1.4, ring_s=1.0)
            c = _note_cents(y, 0.25, 0.9, m)
            s = stats(y)
            rows.append({'midi': m, 'cents': c, 'clicks': s['click_candidates'], 'peak_dbfs': s['peak_dbfs']})
            if c is not None:
                worst = max(worst, abs(c))
        measured = [r['cents'] for r in rows if r['cents'] is not None]
        out[set_id] = {'velocity_sweep_db': [round(float(lv.min()), 1), round(float(lv.max()), 1)],
                       'max_step_db': round(float(steps.max()), 2), 'non_monotonic_steps': int((steps < -0.05).sum()),
                       'notes': len(rows), 'pitch_measured': len(measured),
                       'cents_median_abs': round(float(np.median(np.abs(measured))), 1) if measured else None,
                       'cents_worst_abs': round(worst, 1), 'clicks': int(sum(r['clicks'] for r in rows)),
                       'per_note': rows}
        print(f"{set_id}: velocity {out[set_id]['velocity_sweep_db']} dB, max step {out[set_id]['max_step_db']} dB, "
              f"non-monotonic {out[set_id]['non_monotonic_steps']}; pitch |cents| median "
              f"{out[set_id]['cents_median_abs']} worst {out[set_id]['cents_worst_abs']} over "
              f"{len(measured)}/{len(rows)} notes; clicks {out[set_id]['clicks']}")
        chord = [(0.1 + 0.03 * j, 2.5, mm, 60 + 3 * j) for j, mm in enumerate((50, 57, 62, 66, 69, 74))]
        y = folk.render(set_id, chord, duration=4.0)
        report(f'folk_{set_id}_chord', y, CHECK, STATS)
        _write(f'folk_{set_id}_chord', y)
    _save_stats('folk', out)


def check_found():
    """found：事件切分（每段录音的事件数、单独听会咔哒而被排除的）与 stroke（连续一笔）的起音和长度。"""
    from sampler import found, check
    from sampler.analysis import stats
    out = {}
    for name in ('scribble_long_waldram', 'pencil_writing_rivernile7', 'sketchbook_flips_owlstorm'):
        ev = found.events(name)
        out[name] = {'events': len(ev), 'clean': len(found._clean(name)),
                     'dur_median_s': round(float(np.median([e['dur'] for e in ev])), 3)}
    rows = []
    for seed in range(4):
        y = found.stroke('scribble_long_waldram', 19 / 30, 96, seed)
        tr = np.zeros((SR, 2), np.float32)
        tr[4800:4800 + len(y)] += y
        on = check.measure_onset(tr, 0.1)
        s = stats(tr)
        rows.append({'seed': seed, 'samples': len(y), 'onset_ms': round((on - 0.1) * 1000, 1),
                     'clicks': s['click_candidates'], 'peak_dbfs': s['peak_dbfs']})
        if seed == 0:
            _write('found_stroke_scribble', tr)
    out['stroke'] = rows
    print('found:', json.dumps(out, ensure_ascii=False))
    _save_stats('found', out)


if __name__ == '__main__':
    CHECK.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('piano', 'all'):
        check_piano()
    if what in ('choir', 'all'):
        check_choir()
    if what in ('nature', 'all'):
        check_nature()
    # round 2
    if what in ('symp', 'r2', 'all'):
        check_symp()
    if what in ('choir2', 'r2', 'all'):
        check_choir2()
    if what in ('fx', 'r2', 'all'):
        check_fx()
    if what in ('pad', 'r2', 'all'):
        check_pad()
    if what in ('master', 'r2', 'all'):
        check_master()
    # round 3
    if what in ('voice', 'r3', 'all'):
        check_voice()
    if what in ('strings', 'r3', 'all'):
        check_strings()
    if what in ('bowl', 'r3', 'all'):
        check_bowl()
    if what in ('birds', 'r3', 'all'):
        check_birds()
    if what in ('sfx3', 'r3', 'all'):
        check_sfx3()
    if what in ('unacorda', 'r3', 'all'):
        check_unacorda()
    # agentreel
    if what in ('folk', 'all'):
        check_folk()
    if what in ('found', 'all'):
        check_found()
