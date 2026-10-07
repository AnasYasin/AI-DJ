"""How good are the old dataset's features? Measured on the Raveform tracks that have both a 30 s
iTunes preview on the laptop and full-track values in track_features.parquet (2026-10-07).

For each such track the OLD method runs on the preview, exactly as build_features.py did for the
28,460 old rows: DeepRhythm BPM, essentia edma key, pyloudnorm LUFS, librosa RMS, onset, centroid,
MFCC. The mixer's kick-autocorrelation tempo (djdata tempo.py) also runs on the preview, to see
whether it beats DeepRhythm on 30 s. Each value is scored against the full-track table: BPM against
the checked tempo, key against the full-track key, the rest as preview-against-whole-track error.

    python scripts/features/old_features_check.py --workers 3
    -> data/interim/old_features_check_2026-10-07.csv and a summary on stdout
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "djdata_package")

PREVIEWS = Path("data/interim/seam_previews/audio")
FULL = Path("data/djdata/raveform/out/track_features.parquet")
OUT = Path("data/interim/old_features_check_2026-10-07.csv")

_extractor = None


def one(args):
    tid, path = args
    global _extractor
    from djdata.seam import tempo

    from src.features.build_features import LibrosaExtractor

    if _extractor is None:
        logging.disable(logging.WARNING)
        _extractor = LibrosaExtractor()
    feats = _extractor.extract(path) or {}
    kick = tempo.bpm_of(path)
    return {"track_id": tid, "kick_bpm": kick, **{f"prev_{k}": v for k, v in feats.items()}}


def within(a, b, pct):
    return np.abs(a / b - 1) <= pct


def main(workers: int, limit: int | None):
    full = pd.read_parquet(FULL).set_index("track_id")
    jobs = [(p.stem, str(p)) for p in sorted(PREVIEWS.iterdir()) if p.stem in full.index][:limit]
    print(f"{len(jobs)} tracks with a preview and a full-track row", flush=True)
    pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
    rows = []
    for i, r in enumerate(pool.map(one, jobs, chunksize=4), 1):
        rows.append(r)
        if i % 500 == 0:
            print(i, flush=True)
    d = pd.DataFrame(rows).set_index("track_id").join(full, how="inner")
    d.to_csv(OUT)

    has = d.bpm.notna()
    print("\nBPM against the checked full-track tempo, n =", int(has.sum()))
    for col, name in (
        ("prev_bpm", "DeepRhythm on the preview (the old table's method)"),
        ("kick_bpm", "kick autocorrelation on the preview"),
    ):
        ok = d[col].notna() & has
        r = d.loc[ok, col] / d.loc[ok, "bpm"]
        print(
            f"  {name}: within 2% {100 * within(r, 1, 0.02).mean():.1f}%  half {100 * within(r, 0.5, 0.02).mean():.1f}%  double {100 * within(r, 2, 0.02).mean():.1f}%  no reading {int((~d[col].notna() & has).sum())}"
        )
    k = d.prev_key.notna()
    same = (d.loc[k, "prev_key"] == d.loc[k, "key"]).mean()
    root = (d.loc[k, "prev_key"].str.rstrip("m") == d.loc[k, "key"].str.rstrip("m")).mean()
    print(
        f"\nkey, preview against full track, n = {int(k.sum())}: same key {100 * same:.1f}%, same root {100 * root:.1f}%"
    )
    print("\npreview against whole track, median absolute error and rank correlation")
    for col in (
        "loudness_lufs",
        "energy_mean",
        "onset_strength",
        "spectral_centroid",
        "mfcc_0",
        "mfcc_1",
    ):
        a, b = d[f"prev_{col}"], d[col]
        print(
            f"  {col:18s} median |diff| {float((a - b).abs().median()):8.3f}   spearman {float(a.corr(b, method='spearman')):.3f}   full-track median {float(b.median()):.3f}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(a.workers, a.limit)
