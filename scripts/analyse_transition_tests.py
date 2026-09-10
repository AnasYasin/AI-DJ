"""Measure what each rendered transition actually did to the audio."""

import json
from pathlib import Path

import librosa
import numpy as np

from src.audio.audio_mixer import SR

OUT = Path("data/external/transition_tests")
reports = json.loads((OUT / "report.json").read_text())


def band_rms_db(y, lo, hi, win=0.25):
    """Per-window RMS of one frequency band, in dB."""
    n = int(win * SR)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=n))
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    band = S[(f >= lo) & (f < hi)]
    return 20 * np.log10(np.sqrt((band**2).mean(axis=0)) + 1e-9), win


def mmss(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


print(
    f"{'render':<8}{'type':>7}{'bars':>6}{'seam before':>14}{'after':>10}{'LUFS':>8}{'peak':>7}{'len':>7}"
)
print("-" * 67)
for name, r in reports.items():
    tr = r["transitions"][0]
    print(
        f"{name:<8}{tr['type']:>7}{tr['bars']:>6}{tr['seam_offset_before_ms']:>12.1f}ms"
        f"{tr['seam_offset_ms']:>8.1f}ms{r['lufs']:>8.1f}{r['peak']:>7.2f}{r['duration_s'] / 60:>6.1f}m"
    )

print("\n\nLOW BAND (30-180 Hz) THROUGH EACH TRANSITION, dB relative to the track body")
print("Each column is one eighth of the overlap; 'post' is just after the seam.\n")
print(f"{'render':<8}" + "".join(f"{i / 8:>7.2f}" for i in range(8)) + f"{'post':>8}")
print("-" * 74)
for name, r in reports.items():
    y, _ = librosa.load(r["out"], sr=SR, mono=True)
    tr = r["transitions"][0]
    m, s = tr["at"].split(":")
    start = int(m) * 60 + int(s)
    m, s = tr["end"].split(":")
    end = int(m) * 60 + int(s)
    db, win = band_rms_db(y, 30, 180)
    ref = np.median(db[int(10 / win) : int((start - 5) / win)])  # A's own body
    i0, i1 = int(start / win), int(end / win)
    eighths = [
        db[i0 + int((i1 - i0) * k / 8) : i0 + int((i1 - i0) * (k + 1) / 8)].mean()
        for k in range(8)
    ]
    post = db[i1 + 1 : i1 + int(8 / win)].mean()
    print(f"{name:<8}" + "".join(f"{e - ref:>7.1f}" for e in eighths) + f"{post - ref:>8.1f}")
