"""Objective self-checks: we cannot listen, so every render gets numbers and a picture.

stats(x)  -> dict with peak / true peak / RMS / LUFS / clipping / DC / edge samples /
             click candidates / L-R correlation / band energies (incl. 2.5-4 kHz)
plot(x, path, title) -> spectrogram + long-term spectrum + level envelope PNG
report(name, x, png_dir) -> both, printed as one line, appended to check/stats.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import signal

from ._dsp import SR, a2db, stereo

BANDS = [(20, 100), (100, 500), (500, 2000), (2000, 5000), (5000, 10000), (10000, 20000)]
HARSH = (2500, 4000)


def band_energy_db(x: np.ndarray, bands=BANDS, sr=SR):
    """Band energy relative to total (dB), from a Welch PSD of the mid signal."""
    x = stereo(x)
    mid = x.mean(axis=1).astype(np.float64)
    if len(mid) < 4096:
        mid = np.pad(mid, (0, 4096 - len(mid)))
    f, p = signal.welch(mid, sr, nperseg=8192 if len(mid) >= 8192 else 4096)
    tot = p[(f >= 20) & (f <= 20000)].sum() + 1e-30
    out = {}
    for lo, hi in bands:
        e = p[(f >= lo) & (f < hi)].sum()
        out[f'{lo}-{hi}'] = float(10 * np.log10(e / tot + 1e-30))
    return out


def detect_clicks(x: np.ndarray, sr=SR, ratio=12.0, floor_db=-60.0):
    """Discontinuity detector: |2nd difference| spikes vs. its local (20 ms) RMS.

    Real attacks (piano hammer, woodblock) spread over several samples and usually
    stay below the threshold; a sample-level step or truncation does not.
    Returns list of times (s) of clustered events.
    """
    x = stereo(x).astype(np.float64)
    d2 = np.abs(np.diff(x, n=2, axis=0)).max(axis=1)
    if len(d2) < 10:
        return []
    win = max(3, int(0.02 * sr))
    k = np.ones(win) / win
    local = np.sqrt(np.convolve(d2 ** 2, k, mode='same')) + 1e-12
    thr = 10 ** (floor_db / 20)
    hits = np.flatnonzero((d2 > ratio * local) & (d2 > thr))
    times = []
    last = -1e9
    for h in hits:
        if h - last > 0.005 * sr:
            times.append(round((h + 1) / sr, 4))
        last = h
    return times


def lufs(x: np.ndarray, sr=SR):
    import pyloudnorm as pyln
    x = stereo(x).astype(np.float64)
    if len(x) < int(0.5 * sr):
        x = np.pad(x, ((0, int(0.5 * sr) - len(x)), (0, 0)))
    return float(pyln.Meter(sr).integrated_loudness(x))


def true_peak_db(x: np.ndarray):
    x = stereo(x).astype(np.float64)
    if len(x) == 0:
        return -np.inf
    up = signal.resample_poly(x, 4, 1, axis=0)
    return float(a2db(np.abs(up).max()))


def stats(x: np.ndarray, sr=SR) -> dict:
    x = stereo(x)
    xd = x.astype(np.float64)
    pk = np.abs(xd).max() if len(xd) else 0.0
    rms = np.sqrt((xd ** 2).mean()) if len(xd) else 0.0
    l, r = xd[:, 0], xd[:, 1]
    corr = float(np.corrcoef(l, r)[0, 1]) if l.std() > 0 and r.std() > 0 else 1.0
    clicks = detect_clicks(x, sr)
    be = band_energy_db(x, sr=sr)
    harsh = band_energy_db(x, bands=[HARSH, (200, 2000)], sr=sr)
    return {
        'dur_s': round(len(x) / sr, 3),
        'peak_dbfs': round(float(a2db(pk)), 2),
        'true_peak_dbtp': round(true_peak_db(x), 2),
        'rms_dbfs': round(float(a2db(rms)), 2),
        'lufs': round(lufs(x, sr), 2),
        'clipped_samples': int((np.abs(xd) >= 0.999).sum()),
        'dc_l': float(f'{l.mean():.2e}'), 'dc_r': float(f'{r.mean():.2e}'),
        'dc_dbfs': round(float(a2db(max(abs(l.mean()), abs(r.mean())))), 1),
        'first_sample': float(np.abs(xd[0]).max()) if len(xd) else 0.0,
        'last_sample': float(np.abs(xd[-1]).max()) if len(xd) else 0.0,
        'edge_10ms_dbfs': [round(float(a2db(np.abs(xd[:int(0.01 * sr)]).max())), 1),
                           round(float(a2db(np.abs(xd[-int(0.01 * sr):]).max())), 1)],
        'click_candidates': len(clicks),
        'click_times': clicks[:12],
        'lr_corr': round(corr, 3),
        'bands_db': {k: round(v, 1) for k, v in be.items()},
        'harsh_2k5_4k_rel_total_db': round(harsh[f'{HARSH[0]}-{HARSH[1]}'], 1),
        'harsh_2k5_4k_rel_200_2k_db': round(harsh[f'{HARSH[0]}-{HARSH[1]}'] - harsh['200-2000'], 1),
    }


def plot(x: np.ndarray, path, title='', sr=SR, fmax=16000):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = stereo(x).astype(np.float64)
    mid = x.mean(axis=1)
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1.3], height_ratios=[3, 1, 0.01])
    ax = fig.add_subplot(gs[0, 0])
    nper = 4096
    f, t, S = signal.spectrogram(mid, sr, nperseg=nper, noverlap=nper * 3 // 4,
                                 window='hann', scaling='spectrum')
    Sdb = 10 * np.log10(S + 1e-14)
    vmax = Sdb.max() if Sdb.size else 0
    m = f <= fmax
    ax.pcolormesh(t, f[m], Sdb[m], vmin=vmax - 90, vmax=vmax, shading='auto', cmap='magma')
    ax.set_yscale('symlog', linthresh=500)
    ax.set_ylim(30, fmax)
    ax.axhspan(HARSH[0], HARSH[1], color='cyan', alpha=0.08)
    ax.set_ylabel('Hz')
    ax.set_title(title)

    ax2 = fig.add_subplot(gs[0, 1])
    fw, pw = signal.welch(mid if len(mid) >= 8192 else np.pad(mid, (0, 8192 - len(mid))), sr,
                          nperseg=8192)
    pdb = 10 * np.log10(pw + 1e-20)
    ax2.semilogx(fw[1:], pdb[1:], lw=0.8)
    ax2.axvspan(HARSH[0], HARSH[1], color='cyan', alpha=0.2, label='2.5-4 kHz')
    ax2.set_xlim(30, 20000)
    ax2.set_ylim(pdb[1:].max() - 100, pdb[1:].max() + 5)
    ax2.set_title('long-term spectrum (dB)')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, :])
    hop = max(1, int(0.01 * sr))
    n = len(x) // hop
    if n > 0:
        blk = np.abs(x[:n * hop]).reshape(n, hop, 2).max(axis=1)
        tt = np.arange(n) * hop / sr
        ax3.plot(tt, a2db(blk[:, 0]), lw=0.6, label='L peak')
        ax3.plot(tt, a2db(blk[:, 1]), lw=0.6, label='R peak', alpha=0.7)
    ax3.set_ylim(-90, 3)
    ax3.axhline(0, color='r', lw=0.5)
    ax3.set_xlim(0, len(x) / sr)
    ax3.set_ylabel('dBFS')
    ax3.set_xlabel('s')
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def report(name: str, x: np.ndarray, png_dir=None, json_path=None, extra=None) -> dict:
    s = stats(x)
    if extra:
        s.update(extra)
    if png_dir is not None:
        plot(x, Path(png_dir) / f'{name}.png', title=name)
    print(f"{name:28} peak {s['peak_dbfs']:6.1f} dBFS  TP {s['true_peak_dbtp']:6.1f}  "
          f"RMS {s['rms_dbfs']:6.1f}  LUFS {s['lufs']:6.1f}  clip {s['clipped_samples']}  "
          f"DC {s['dc_dbfs']:6.1f} dB  edges {s['first_sample']:.1e}/{s['last_sample']:.1e}  "
          f"clicks {s['click_candidates']}  LRcorr {s['lr_corr']:+.2f}  "
          f"2.5-4k {s['harsh_2k5_4k_rel_total_db']:+.1f} dB(tot) "
          f"{s['harsh_2k5_4k_rel_200_2k_db']:+.1f} dB(vs 200-2k)")
    if json_path is not None:
        p = Path(json_path)
        d = json.loads(p.read_text()) if p.exists() else {}
        d[name] = s
        p.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    return s


def pitch_track(x: np.ndarray, fmin=100.0, fmax=1400.0, hop_s=0.005, win_s=0.025, sr=SR):
    """McLeod NSDF pitch tracker (deterministic). Returns (times_s, f0_hz, clarity);
    f0 is NaN where clarity < 0.6 (unvoiced / silence)."""
    x = stereo(x).mean(axis=1).astype(np.float64)
    w = int(win_s * sr)
    hop = int(hop_s * sr)
    n = 1 + max(0, (len(x) - w) // hop)
    idx = np.arange(w)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx]
    fr = fr - fr.mean(axis=1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * w)))
    F = np.fft.rfft(fr, nfft, axis=1)
    r = np.fft.irfft(np.abs(F) ** 2, nfft, axis=1)[:, :w]
    sq = fr ** 2
    cs = np.cumsum(sq, axis=1)
    csr = np.cumsum(sq[:, ::-1], axis=1)[:, ::-1]      # csr[:, k] = sum_{j>=k} x_j^2
    m = cs[:, ::-1] + csr                               # sum_{j<w-tau} x_j^2 + sum_{j>=tau} x_j^2
    nsdf = 2 * r / np.maximum(m, 1e-20)
    lo, hi = int(sr / fmax), min(w - 2, int(sr / fmin))
    f0 = np.full(n, np.nan)
    clar = np.zeros(n)
    for i in range(n):
        d = nsdf[i, :hi + 1]
        neg = np.flatnonzero(d[1:] < 0)
        if not len(neg):
            continue
        z = max(neg[0] + 1, lo)
        seg = d[z:hi]
        if not len(seg):
            continue
        pk = seg.max()
        if pk < 0.6:
            clar[i] = pk
            continue
        k = z + int(np.argmax(seg >= 0.9 * pk))
        # walk to the local maximum
        while k + 1 < hi and d[k + 1] > d[k]:
            k += 1
        a, b, c = d[k - 1], d[k], d[k + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
        f0[i] = sr / (k + off)
        clar[i] = b
    t = (np.arange(n) * hop + w / 2) / sr
    return t, f0, clar
