"""Keep a fetched old-data record as held full audio, so no later mix looks for it again.

    register(path, track_id)

copies the file to data/djdata/old/tracks/<track_id>.<ext>, measures the full-track features and
the whole-track effnet mean the way `src.features.track_features` does for the measured sources
(vectors kept as data/djdata/old/embeddings/<track_id>.npy), measures the BPM with the tempo stage's
instrument when the djdata package is importable, and appends one row to
data/djdata/old/out/track_features.parquet. The next `djdata export-dataset` turns that row into
`audio = full` in the dataset (`djdata.dataset.old_tables`).

Anas, 2026-10-13: every record fetched by link or by search joins the full-audio pool with entries
everywhere, and the rest of the old records get fetched the same way once the models are judged.
"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path("data/djdata/old")
TRACKS, EMBEDDINGS, TABLE = (
    ROOT / "tracks",
    ROOT / "embeddings",
    ROOT / "out" / "track_features.parquet",
)


def register(path: str | Path, track_id: str, root: Path = ROOT) -> dict:
    """Copy, measure, append. Returns the row written. A record already registered is left alone."""
    path = Path(path)
    tracks, embeddings, table = (
        root / "tracks",
        root / "embeddings",
        root / "out" / "track_features.parquet",
    )
    for d in (tracks, embeddings, table.parent):
        d.mkdir(parents=True, exist_ok=True)
    if table.exists():
        have = pd.read_parquet(table, columns=["track_id"])
        if track_id in set(have["track_id"]):
            return {"track_id": track_id, "status": "already registered"}
    dest = tracks / f"{track_id}{path.suffix}"
    if not dest.exists():
        shutil.copy2(path, dest)

    from src.features.track_features import one_track  # essentia loads here, not at import

    row = one_track(
        {
            "track_id": track_id,
            "corpus": "old",
            "path": str(dest),
            "embeddings_dir": str(embeddings),
        }
    )
    row["bpm"], row["bpm_source"] = _bpm(dest)
    existing = pd.read_parquet(table) if table.exists() else pd.DataFrame()
    pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_parquet(table, index=False)
    log.info(
        "registered %s as full audio (%.1f min, %s BPM)", track_id, row["minutes"], row["bpm"]
    )
    return row


def _bpm(path: Path) -> tuple[float | None, str | None]:
    try:
        from djdata.seam.tempo import bpm_of  # the measured sources' instrument
    except ImportError:
        return None, None
    bpm = bpm_of(path)
    return (float(bpm), "measured") if bpm else (None, None)
