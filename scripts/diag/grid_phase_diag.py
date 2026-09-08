import json
import logging
import sys

import librosa
import numpy as np

logging.disable(logging.WARNING)
sys.path.insert(0, ".")
from scipy.signal import correlate

from src.audio import audio_mixer as am
from src.audio.audio_mixer import ONSET_HOP, SR, _to_mono

plan = json.load(open("data/external/ear_test_2026-09-07/techno/plan.json"))["tracks"]
a, b = plan[2], plan[3]  # pair 2
pa, pb = [f"data/external/test_tracks/techno/{t['track_id']}.m4a" for t in (a, b)]
target = 127.5
A = am._prepare_track(pa, target, 16, 128, a["energy_target01"], 8)
B = am._prepare_track(pb, target, 16, 128, b["energy_target01"], 8)
bar = 4 * 60 / target


def env_band(y, lo, hi):
    S = np.abs(librosa.stft(_to_mono(y), n_fft=1024, hop_length=ONSET_HOP))
    f = librosa.fft_frequencies(sr=SR, n_fft=1024)
    return np.maximum(np.diff(S[(f >= lo) & (f < hi)].sum(0), prepend=0), 0)


def pulse_vs_grid(y, beat_times, t0, n_bars):
    """offset (ms) of each band's pulse relative to the record's own beat grid over [t0, t0+n_bars bars]."""
    seg = y[int(t0 * SR) : int((t0 + n_bars * bar) * SR)]
    beats = beat_times[(beat_times >= t0) & (beat_times < t0 + n_bars * bar)] - t0
    train = np.zeros(int(len(seg) / ONSET_HOP) + 2)
    train[np.clip((beats * SR / ONSET_HOP).astype(int), 0, len(train) - 1)] = 1.0
    out = {}
    for lo, hi, lab in (
        (30, 130, "kick"),
        (130, 1000, "lowmid"),
        (1000, 5000, "mid"),
        (5000, 12000, "hats"),
        (20, 16000, "full"),
    ):
        e = env_band(seg, lo, hi)
        n = min(len(e), len(train))
        e, tr = e[:n], train[:n]
        e = (e - e.mean()) / (e.std() + 1e-9)
        tr = (tr - tr.mean()) / (tr.std() + 1e-9)
        ml = int(0.5 * (60 / target) * SR / ONSET_HOP)
        c = correlate(e, tr, "full") / n
        ctr = n - 1
        w = c[ctr - ml : ctr + ml + 1]
        k = int(np.argmax(w))
        out[lab] = f"{(k - ml) * ONSET_HOP / SR * 1000:+5.0f}ms({w.max():.2f})"
    return out


print(
    f"A cue_out {A['cue_out_bar']} drop_bar {A['drop_bar']}; B cue_in {B['cue_in_bar']} drop_bar {B['drop_bar']}"
)
print("Pulse offset relative to the record's OWN grid (positive = pulse after the grid beat):")
for name, T, bar0 in (
    ("A tail (bars after cue-out)", A, A["cue_out_bar"]),
    ("A body (8 bars before cue-out)", A, A["cue_out_bar"] - 8),
    ("B head (from cue-in)", B, B["cue_in_bar"]),
    ("B body (8 bars later)", B, B["cue_in_bar"] + 8),
):
    print(f"  {name:32s}", pulse_vs_grid(T["audio"], T["beat_times"], T["bars"][bar0], 8))
