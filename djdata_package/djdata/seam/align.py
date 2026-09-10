"""Stretch-free alignment of the two records to the mix window.

Each record keeps its original audio. A mapping orig_t = o + (mix_t - m) * rate says which span of
the original plays in each mix beat. Nothing is time-stretched on our side, because a phase vocoder
on our side never matches the DJ's engine and the mismatch leaked 5 to 10 dB between the records.

Steps (measured 2026-09-09/10 on rendered seams with a coarse error of ±1 bar and ±0.1 % rate):
  1. beat period from the mix's own onset autocorrelation (a 1 % tempo error is 13 ms per bar)
  2. onset-envelope correlation, source envelope time-scaled by the rate, ±1 bar
  3. residual scan: shift the source in 5 ms steps over ±0.6 beat at two 16-beat regions where the
     record plays alone, fit offset + rate, then refine at 1 ms (recovers a half-beat onset lock)
  4. bar check: -2..+2 bars, each re-refined ±25 ms, lowest alone-region residual wins
Result: alignment within 9 ms in 14 of 14 runs.
"""

import numpy as np
import librosa
from scipy.signal import correlate

SR, NFFT, HOP, SLOTS = 44_100, 2048, 256, 8


def load_mono(path, offset=0.0, duration=None) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=True, offset=offset, duration=duration)
    return y.astype(np.float32)


def power_spec(y: np.ndarray) -> np.ndarray:
    return np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP)) ** 2


def onset_env(S: np.ndarray) -> np.ndarray:
    return librosa.onset.onset_strength(S=librosa.power_to_db(S), sr=SR, hop_length=HOP)


def slots(S: np.ndarray, t0: float, t1: float, n: int = SLOTS) -> np.ndarray:
    """Mean power per sub-slot between t0 and t1 seconds of the signal S belongs to: (bins, n)."""
    fr = np.round(np.linspace(t0, t1, n + 1) * SR / HOP).astype(int)
    out = np.zeros((S.shape[0], n))
    for j in range(n):
        a, b = fr[j], max(fr[j + 1], fr[j] + 1)
        if a >= 0 and b <= S.shape[1]:
            out[:, j] = S[:, a:b].mean(axis=1)
    return out


def band_indices(bands: list) -> dict:
    freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
    return {i: np.where((freqs >= f0) & (freqs < f1))[0] for i, (f0, f1) in enumerate(bands)}


def mix_beat_period(oe_mix: np.ndarray, bpm_guess: float) -> float:
    fps = SR / HOP
    a = oe_mix[: int(min(len(oe_mix), 60 * fps))]
    ac = librosa.autocorrelate(a - a.mean())
    lag = np.arange(len(ac)) / fps
    g = 60.0 / bpm_guess
    idx = np.where((lag > g * 0.96) & (lag < g * 1.04))[0]
    k = idx[int(np.argmax(ac[idx]))]
    if 0 < k < len(ac) - 1:
        y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
        k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    return float(k / fps)


class Aligner:
    """Holds the window spectrogram and both source spectrograms; produces one mapping per record."""

    def __init__(self, SM, SS: dict, oe: dict, beat: float, idx: dict):
        self.SM, self.SS, self.oe, self.beat, self.idx = SM, SS, oe, beat, idx

    def step1(self, tag, anchor_m, anchor_o, rate, side, win=30.0) -> float:
        """Onset correlation. Returns the corrected window time at which orig time anchor_o plays."""
        fps, bar = SR / HOP, 4 * self.beat
        if side == "before":
            m0 = anchor_m - win - bar
            o_t = np.arange(anchor_o - win * rate, anchor_o, rate / fps)
        else:
            m0 = anchor_m - bar
            o_t = np.arange(anchor_o, anchor_o + win * rate, rate / fps)
        lo = max(int(m0 * fps), 0)
        es = self.oe["M"][lo: int((m0 + win + 2 * bar) * fps)]
        er = np.interp(o_t * fps, np.arange(len(self.oe[tag])), self.oe[tag])
        if len(es) <= len(er):
            return anchor_m
        es, er = es - es.mean(), er - er.mean()
        k = int(np.argmax(correlate(es, er, mode="valid")))
        return lo / fps + k / fps + (win if side == "before" else 0.0)

    def resid1(self, tag, anchor_m, anchor_o, rate, region_m0, n_beats) -> float:
        """Single-record fit residual over n_beats from window time region_m0."""
        S = self.SS[tag]
        tot = 0.0
        for k in range(n_beats):
            mt0 = region_m0 + k * self.beat
            mt1 = mt0 + self.beat
            o0 = anchor_o + (mt0 - anchor_m) * rate
            o1 = anchor_o + (mt1 - anchor_m) * rate
            M, X = slots(self.SM, mt0, mt1), slots(S, o0, o1)
            for idx in self.idx.values():
                m, x = M[idx].ravel(), X[idx].ravel()
                g = max(np.dot(x, m) / (np.dot(x, x) + 1e-12), 0)
                r = m - g * x
                tot += np.dot(r, r) / (np.dot(m, m) + 1e-12)
        return tot

    def step2(self, tag, anchor_m, anchor_o, rate, regions) -> tuple[float, float]:
        span, step = 0.6 * self.beat, 0.005
        for _ in range(2):
            shifts = np.arange(-span, span + 1e-9, step)
            best = []
            for r0 in regions:
                res = [self.resid1(tag, anchor_m, anchor_o + s, rate, r0, 16) for s in shifts]
                best.append(shifts[int(np.argmin(res))])
            d = (best[1] - best[0]) / (regions[1] - regions[0])
            anchor_o = anchor_o + best[1] + (anchor_m - regions[1]) * d
            rate += d
            span, step = 0.010, 0.001
        return anchor_o, rate

    def bar_check(self, tag, o, m, r, side) -> tuple[float, int]:
        bar = 4 * self.beat
        region = m - 34 * self.beat - 1.0 if side == "before" else m + 2 * self.beat + 1.0
        best = None
        for nb in (-2, -1, 0, 1, 2):
            base = o + nb * bar * r
            shifts = np.arange(-0.025, 0.0251, 0.001)
            res = [self.resid1(tag, m, base + sh, r, region, 32) for sh in shifts]
            j = int(np.argmin(res))
            if best is None or res[j] < best[0]:
                best = (res[j], nb, base + shifts[j])
        return best[2], best[1]

    def align(self, tag, side, anchor_m, anchor_o, rate) -> dict:
        m1 = self.step1(tag, anchor_m, anchor_o, rate, side)
        regions = ([m1 - 40 * self.beat - 1.0, m1 - 6 * self.beat - 1.0] if side == "before"
                   else [m1 + 2 * self.beat + 1.0, m1 + 36 * self.beat + 1.0])
        o2, r2 = self.step2(tag, m1, anchor_o, rate, regions)
        o3, nb = self.bar_check(tag, o2, m1, r2, side)
        return {"o": o3, "m": m1, "rate": r2, "bar_fix": nb, "onset_shift_s": m1 - anchor_m}
