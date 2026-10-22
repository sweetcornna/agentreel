"""Objective checks and pictures, because nobody here can listen.

qc(x)                 -> dict of level, loudness, DC, stereo, HF, click and edge metrics
alias_ratio(x, f0)    -> energy away from the harmonic series of a steady test tone (dB)
spectrogram_png(...)  -> log-frequency spectrogram with waveform envelope
loudness_curve(x)     -> 1 s K-weighted loudness (approx LUFS) series, hop 0.1 s
plot_loudness(...)    -> loudness curve against target keys (e.g. timeline.json 'loudness')
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .core import SR, stereo
from .dsp.dynamics import lufs, peak_db, true_peak_db
from .dsp.stereo import correlation, side_ratio_db

# BS.1770 K-weighting at 48 kHz
_KB1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
_KA1 = [1.0, -1.69065929318241, 0.73248077421585]
_KB2 = [1.0, -2.0, 1.0]
_KA2 = [1.0, -1.99004745483398, 0.99007225036621]


def k_weight(x):
    return signal.lfilter(_KB2, _KA2, signal.lfilter(_KB1, _KA1, x, axis=-1), axis=-1)


def loudness_curve(x, win: float = 1.0, hop: float = 0.1, sr: int = SR):
    """Return (times_centre, loudness) of K-weighted mean square over `win` s windows (approx LUFS)."""
    x = stereo(x)
    k = k_weight(x)
    p = np.sum(k ** 2, axis=0)
    n = int(win * sr)
    h = int(hop * sr)
    c = np.concatenate([[0.0], np.cumsum(p)])
    starts = np.arange(0, max(1, len(p) - n + 1), h)
    ms = (c[starts + n] - c[starts]) / n
    L = -0.691 + 10 * np.log10(ms + 1e-20)
    return (starts + n / 2) / sr, L


def crest_db(x):
    x = stereo(x)
    rms = np.sqrt(np.mean(x ** 2))
    return float(20 * np.log10(np.max(np.abs(x)) / (rms + 1e-20)))


def hf_ratios(x, sr: int = SR):
    """Energy above 8 k, 12 k and 16 k relative to total (dB) and spectral centroid (Hz)."""
    m = np.mean(stereo(x), axis=0)
    f, P = signal.welch(m, sr, nperseg=8192)
    tot = np.sum(P) + 1e-30
    r = {f"hf_{int(fc / 1000)}k_db": float(10 * np.log10(np.sum(P[f >= fc]) / tot + 1e-20)) for fc in (8000, 12000, 16000)}
    r["centroid_hz"] = float(np.sum(f * P) / tot)
    return r


def detect_clicks(x, onsets=None, sr: int = SR, thresh_db: float = 24.0, floor_db: float = -75.0,
                  guard=(0.004, 0.03)):
    """Derivative-spike detector. A 1 ms frame of high-passed (>5 kHz) first difference whose energy
    jumps `thresh_db` above the median of its 60 ms neighbourhood is a click unless it lies within
    `guard` (before, after) seconds of an expected onset. Returns a list of times (s)."""
    x = stereo(x)
    d = np.diff(x, axis=-1, prepend=x[:, :1])
    hp = signal.sosfilt(signal.butter(4, 5000, "highpass", fs=sr, output="sos"), d, axis=-1)
    e = np.sum(hp ** 2, axis=0)
    fl = int(0.001 * sr)
    nf = len(e) // fl
    if nf < 10:
        return []
    fe = e[: nf * fl].reshape(nf, fl).mean(axis=1) + 1e-30
    fdb = 10 * np.log10(fe)
    from scipy.ndimage import median_filter
    med = median_filter(fdb, size=61, mode="nearest")
    # absolute floor relative to the signal's own derivative level
    ref = 10 * np.log10(np.mean(fe) + 1e-30)
    cand = np.nonzero((fdb - med > thresh_db) & (fdb > ref + floor_db + 60))[0]
    times = []
    on = np.array(sorted(onsets)) if onsets is not None and len(onsets) else np.zeros(0)
    last = -1.0
    for i in cand:
        t = i * fl / sr
        if len(on):
            j = np.searchsorted(on, t)
            near = False
            for jj in (j - 1, j):
                if 0 <= jj < len(on) and -guard[0] <= t - on[jj] <= guard[1]:
                    near = True
            if near:
                continue
        if t - last > 0.01:
            times.append(round(t, 4))
        last = t
    return times


def edges(x, sr: int = SR):
    """Absolute first/last sample values and the step size at the very end (dBFS)."""
    x = stereo(x)
    return {"first_abs_db": float(20 * np.log10(np.max(np.abs(x[:, 0])) + 1e-12)),
            "last_abs_db": float(20 * np.log10(np.max(np.abs(x[:, -1])) + 1e-12))}


def dc_db(x):
    x = stereo(x)
    return float(20 * np.log10(np.max(np.abs(np.mean(x, axis=1))) + 1e-12))


def qc(x, onsets=None, sr: int = SR):
    """Full objective report for a rendered signal."""
    x = stereo(x)
    t, L = loudness_curve(x)
    Lv = L[L > -70]
    r = {
        "duration_s": round(x.shape[1] / sr, 3),
        "peak_dbfs": round(peak_db(x), 2),
        "true_peak_dbtp": round(true_peak_db(x), 2),
        "lufs_i": round(lufs(x), 2),
        "loud_1s_max": round(float(Lv.max()), 2) if len(Lv) else None,
        "loud_1s_min": round(float(Lv.min()), 2) if len(Lv) else None,
        "crest_db": round(crest_db(x), 2),
        "dc_db": round(dc_db(x), 1),
        "corr_lr": round(correlation(x), 3),
        "side_mid_db": round(side_ratio_db(x), 1),
    }
    r.update({k: round(v, 1) for k, v in hf_ratios(x).items()})
    r.update({k: round(v, 1) for k, v in edges(x).items()})
    cl = detect_clicks(x, onsets)
    r["clicks"] = len(cl)
    r["click_times"] = cl[:10]
    return r


def alias_ratio(x, f0: float, sr: int = SR, t0: float = 0.2, dur: float = 1.0, tol_hz: float = None,
                partials=None):
    """For a steady tone: energy NOT within tol of k*f0 (or of the given partial list), relative to
    energy near partials, measured above 0.5*f0 (dB). Lower is cleaner; < -50 dB is inaudible."""
    m = np.mean(stereo(x), axis=0)[int(t0 * sr): int((t0 + dur) * sr)]
    nfft = 1 << int(np.ceil(np.log2(len(m) * 4)))
    X = np.abs(np.fft.rfft(m * signal.windows.blackmanharris(len(m)), nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / sr)
    tol = tol_hz if tol_hz is not None else max(4.0, 0.012 * f0)
    if partials is None:
        partials = f0 * np.arange(1, int(sr / 2 / f0) + 1)
    near = np.zeros_like(f, dtype=bool)
    for p in partials:
        near |= np.abs(f - p) < tol * max(1.0, np.sqrt(p / f0) * 0.6)
    band = f > 0.5 * f0
    a = np.sum(X[band & near]) + 1e-30
    b = np.sum(X[band & ~near]) + 1e-30
    return float(10 * np.log10(b / a))


def spectrogram_png(x, path, title: str = "", sr: int = SR, fmin: float = 30.0, fmax: float = 24000.0,
                    dyn: float = 100.0, nfft: int = 4096):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = stereo(x)
    m = np.mean(x, axis=0)
    f, t, S = signal.spectrogram(m, sr, nperseg=nfft, noverlap=nfft * 3 // 4, window="hann", mode="psd")
    S = 10 * np.log10(S + 1e-20)
    S -= S.max()
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), gridspec_kw={"height_ratios": [1, 4]}, sharex=True)
    tt, L = loudness_curve(x, 0.4, 0.05)
    axes[0].plot(tt, L, lw=0.8, color="#333")
    axes[0].set_ylabel("LUFS(0.4s)")
    axes[0].set_ylim(max(-80, np.nanmax(L) - 60), np.nanmax(L) + 3)
    axes[0].grid(alpha=0.3)
    axes[0].set_title(title)
    sel = (f >= fmin) & (f <= fmax)
    axes[1].pcolormesh(t, f[sel], S[sel], shading="auto", vmin=-dyn, vmax=0, cmap="magma")
    axes[1].set_yscale("log")
    axes[1].set_ylim(fmin, fmax)
    axes[1].set_ylabel("Hz")
    axes[1].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_loudness(x, path, target_keys=None, to_seconds=None, tol_db: float = 3.0, title: str = ""):
    """1 s loudness curve; optional target keys [(pos, lufs), ...] mapped with to_seconds(pos)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, L = loudness_curve(x)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, L, lw=1.0, label="1 s loudness")
    if target_keys:
        ts = [to_seconds(k[0]) if to_seconds else float(k[0]) for k in target_keys]
        vs = [k[1] for k in target_keys]
        ax.plot(ts, vs, "o-", color="tab:orange", lw=0.8, ms=3, label="target")
        ax.fill_between(ts, np.array(vs) - tol_db, np.array(vs) + tol_db, color="tab:orange", alpha=0.15)
    ax.set_ylim(-70, 0)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("s")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
