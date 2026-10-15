"""The old table against the new method on the same records: 200 old-dataset tracks fetched in full and
verified by fingerprint against their preview (data/interim/old_sample_full/, 2026-10-07).

Step 1 (run first): track features on the fetched files, the production code
    python -m src.features.track_features --tracks old_sample=data/interim/old_sample_full/tracks \
        --out data/interim/old_sample_full/track_features.parquet \
        --embeddings data/interim/old_sample_full/embeddings --workers 3
Step 2: this script. Measures the mixer's tempo on each fetched file (tempo.py::bpm_of), joins the old
table's row for the same track_id, and scores old against new: BPM within 2 %, key agreement, and the
error of LUFS, energy, onset, centroid and MFCCs. Writes old_sample_compare.csv and prints the summary.
"""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "djdata_package")

ROOT = Path("data/interim/old_sample_full")
OLD = Path("data/processed/features.parquet")
FEATURES = ["loudness_lufs", "energy_mean", "onset_strength", "spectral_centroid"] + [
    f"mfcc_{i}" for i in range(13)
]


def kick_bpm(args):
    tid, path = args
    from djdata.seam import tempo

    return tid, tempo.bpm_of(path)


def main():
    new = pd.read_parquet(ROOT / "track_features.parquet").set_index("track_id")
    files = {p.stem: str(p) for p in (ROOT / "tracks").iterdir() if p.stem in new.index}
    pool = ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn"))
    new["bpm"] = pd.Series(dict(pool.map(kick_bpm, files.items())))
    old = pd.read_parquet(OLD).set_index("track_id")
    old = old.loc[old.index.intersection(new.index)]
    d = new.join(old, lsuffix="_new", rsuffix="_old", how="inner")
    sample = pd.read_csv(ROOT / "sample.csv").set_index("track_id")
    d["genre"] = sample.genre
    d.drop(columns=[c for c in d.columns if c.startswith("embedding")]).to_csv(
        ROOT / "old_sample_compare.csv"
    )

    print(f"tracks fetched, verified and compared: {len(d)}")
    ok = d.bpm_new.notna() & d.bpm_old.notna()
    r = d.loc[ok, "bpm_old"] / d.loc[ok, "bpm_new"]
    w = lambda k: int((np.abs(r / k - 1) <= 0.02).sum())  # noqa: E731
    print(
        f"\nBPM, old table (DeepRhythm on the preview) against the mixer's tempo on the full track, n = {int(ok.sum())}:"
        f"\n  within 2% {w(1)} ({100 * w(1) / ok.sum():.1f}%)   half {w(0.5)}   double {w(2)}   other {int(ok.sum()) - w(1) - w(0.5) - w(2)}"
        f"\n  no new tempo (no periodic kick) {int(d.bpm_new.isna().sum())}"
    )
    for g, gd in d[ok].groupby("genre"):
        rg = gd.bpm_old / gd.bpm_new
        print(
            f"    {g:14s} n {len(gd):3d}  within 2% {100 * (np.abs(rg - 1) <= 0.02).mean():.0f}%"
        )
    same = (d.key_old == d.key_new).mean()
    root = (d.key_old.str.rstrip("m") == d.key_new.str.rstrip("m")).mean()
    print(
        f"\nkey, old (preview) against new (full track): same key {100 * same:.1f}%, same root {100 * root:.1f}%"
    )
    print(
        "\nold (preview) against new (full track), median |diff|, rank correlation, full-track median:"
    )
    for col in FEATURES:
        a, b = d[f"{col}_old"], d[f"{col}_new"]
        print(
            f"  {col:18s} {float((a - b).abs().median()):9.3f}   spearman {float(a.corr(b, method='spearman')):.3f}   {float(b.median()):9.3f}"
        )


if __name__ == "__main__":
    main()
