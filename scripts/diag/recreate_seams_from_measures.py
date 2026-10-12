"""Ear test of the seam measurements (Anas's design, 2026-10-12): rebuild each picked seam from the measured moves alone
and play it in the LEFT channel, the original seam excerpt in the RIGHT, on one clock.

Left = record A and record B at their measured time zero and speed, each split into the reader's three bands (20-200,
200-3000, 3000-14000 Hz, the butterworth of bands.py); A's band fades out over one bar ending at the band's measured
exit, B's band fades in over one bar from its measured entry; nothing else. A band the reader never heard enter stays
silent, one it never heard leave stays on. Right = the clip cut from the mix window. If left sounds like right, the
measurements hold the transition; what differs is the band or the moment that is wrong.

    conda run -n aidj python scripts/diag/recreate_seams_from_measures.py data/external/ear_test/types_2026-10-13
reads picks.csv there (columns corpus, n, move, dj, seam_id, window_file, placement and heard moments), cuts the plain clip
NN_<move>_<dj>.mp3 from the corpus's window file and writes recreated/NN_..._LR.mp3 beside it. Band moments come from
the corpus's measures.csv.
"""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, "djdata_package")
from djdata.seam.bands import band_filter
from djdata.seam.fingerprint import SR, at_mix_speed, load
from scipy.signal import sosfiltfilt

D = Path(sys.argv[1] if len(sys.argv) > 1 else "data/external/ear_test/types_2026-10-13")
OUT = D / "recreated"
OUT.mkdir(exist_ok=True)
picks = pd.read_csv(D / "picks.csv")
meas = {
    c: pd.read_csv(f"data/djdata/{c}/out/measures.csv", low_memory=False).set_index("seam_id")
    for c in picks.corpus.unique()
}
BANDS_COLS = ["low_out_s", "low_in_s", "mid_out_s", "mid_in_s", "high_out_s", "high_in_s"]


def envelope(n, t_out, t_in, bar_s, start):
    """Gain over n samples on the window clock from `start`: 1 then a one-bar fade to 0 ending at t_out (outgoing), or
    0 then a one-bar fade to 1 from t_in (incoming). None = never heard leave (stays 1) / never heard enter (stays 0)."""
    t = start + np.arange(n) / SR
    if t_in is None:  # outgoing record
        if t_out is None or np.isnan(t_out):
            return np.ones(n, np.float32)
        return np.clip((t_out - t) / bar_s, 0, 1).astype(np.float32)
    if t_in is None or np.isnan(t_in):
        return np.zeros(n, np.float32)
    return np.clip((t - t_in) / bar_s, 0, 1).astype(np.float32)


lines = []
for _, r in picks.iterrows():
    t0 = float(r.window_t0)
    bar = float(r.bar_s)
    c = r.corpus
    m = meas[c].loc[r.seam_id]
    lo, hi = (
        min(r.a_last_heard_s, r.b_first_heard_s) - 25,
        max(r.a_last_heard_s, r.b_first_heard_s) + 25,
    )
    start = max(0.0, lo - t0)
    dur = min(hi - lo, 150.0, float(r.window_s) - start)
    name = f"{int(r.n):02d}_{r.move}_{str(r.dj).replace(' ', '').replace('..', '').replace('+', '').replace(',', '')}"
    wpath = Path(f"data/djdata/{c}/windows") / r.window_file
    right = load(str(wpath), offset=start, duration=dur)
    n = len(right)
    plain = D / f"{name}.mp3"
    sf.write(D / "tmp.wav", right, SR)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(D / "tmp.wav"),
            "-b:a",
            "160k",
            str(plain),
        ],
        check=True,
    )
    left = np.zeros(n, np.float32)
    for side, tid, tz, rate in (
        ("A", r.track_a, r.a_time_zero_s, r.a_rate),
        ("B", r.track_b, r.b_time_zero_s, r.b_rate),
    ):
        path = next(Path(f"data/djdata/{c}/tracks").glob(f"{tid}.*"))
        rec = at_mix_speed(load(str(path)), float(rate))
        first = int((start + t0 - float(tz)) * SR)
        piece = np.zeros(n, np.float32)
        a0, a1 = max(first, 0), min(first + n, len(rec))
        if a1 > a0:
            piece[a0 - first : a1 - first] = rec[a0:a1]
        for band in ("low", "mid", "high"):
            x = sosfiltfilt(band_filter(band), piece).astype(np.float32)
            env = (
                envelope(n, m[f"{band}_out_s"], None, bar, start)
                if side == "A"
                else envelope(n, None, m[f"{band}_in_s"], bar, start)
            )
            left += x * env
    rl, rr = np.sqrt(np.mean(left**2)) + 1e-9, np.sqrt(np.mean(right**2)) + 1e-9
    left = left * (rr / rl)
    peak = max(np.abs(left).max(), np.abs(right).max(), 1e-9)
    stereo = np.stack([left, right], axis=1) / peak * 0.95
    sf.write(D / "tmp.wav", stereo, SR)
    mp3 = OUT / f"{name}_LR.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(D / "tmp.wav"),
            "-b:a",
            "192k",
            str(mp3),
        ],
        check=True,
    )
    rel = {k: (None if pd.isna(m[k]) else round(float(m[k]) - start)) for k in BANDS_COLS}
    swap = (
        None if pd.isna(m.get("bass_swap_s", np.nan)) else round(float(m["bass_swap_s"]) - start)
    )
    lines.append(
        f"{name}  {c}, {r.genre}; bass swap read at +{swap} s; B in (high, mid, low) +{rel['high_in_s']}, +{rel['mid_in_s']}, +{rel['low_in_s']}; A out +{rel['high_out_s']}, +{rel['mid_out_s']}, +{rel['low_out_s']}; {dur:.0f} s"
    )
    print(lines[-1], flush=True)
(D / "tmp.wav").unlink(missing_ok=True)
(D / "README.txt").write_text(
    "LEFT = the seam rebuilt from the measurements alone (records placed at the measured time zero and speed, each band\n"
    "faded out at its measured exit / in at its measured entry over one bar). RIGHT = the original seam excerpt. Same clock.\n"
    "If left sounds like right, the measurements hold the transition. Moments below are seconds into the clip.\n\n"
    + "\n".join(lines)
    + "\n"
)
print("done", len(lines))
