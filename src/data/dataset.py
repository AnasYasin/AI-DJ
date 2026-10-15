"""Read the dataset the models train on: `data/dataset/<date>/` built by `djdata export-dataset`.

    from src.data.dataset import load_tracks, load_plays, load_seams
    tracks = load_tracks()                       # the newest date, every source
    tracks = load_tracks(sources=["raveform", "djs"], audio="full")
    plays  = load_plays()                        # plays["split"] is train or validation

Nothing else in `src/` reads `data/processed` or `data/djdata` for training or planning. A new source
is a rerun of the export, never an edit here.
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

DATASET_DIR = Path("data/dataset")
VALIDATION = "validation"  # the value of the `split` column on plays and seams


def dj_key(name: str) -> str:
    """One DJ, one key: letters and digits only, & and + spelled as the 1001 slugs spell them
    (Aly & Fila -> alyandfila). The `dj_key` column of plays is this."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower().replace("&", "and").replace("+", "plus"))


def latest(root: Path = DATASET_DIR) -> Path:
    dates = sorted(
        p for p in Path(root).iterdir() if p.is_dir() and (p / "tracks.parquet").exists()
    )
    if not dates:
        raise FileNotFoundError(
            f"no dataset under {root}; run `djdata export-dataset --date <date>`"
        )
    return dates[-1]


def _load(
    name: str, date: str | None, root: Path, sources: list[str] | None, columns: list[str] | None
) -> pd.DataFrame:
    folder = Path(root) / date if date else latest(root)
    df = pd.read_parquet(folder / f"{name}.parquet", columns=columns)
    if sources is not None:
        df = df[df["source"].isin(sources)]
    return df.reset_index(drop=True)


def load_tracks(
    date: str | None = None,
    sources: list[str] | None = None,
    audio: str | None = None,
    columns: list[str] | None = None,
    root: Path = DATASET_DIR,
) -> pd.DataFrame:
    df = _load("tracks", date, root, sources, columns)
    if audio is not None:
        df = df[df["audio"] == audio].reset_index(drop=True)
    return df


def load_plays(
    date: str | None = None,
    sources: list[str] | None = None,
    columns: list[str] | None = None,
    root: Path = DATASET_DIR,
) -> pd.DataFrame:
    return _load("plays", date, root, sources, columns)


def load_seams(
    date: str | None = None,
    sources: list[str] | None = None,
    columns: list[str] | None = None,
    root: Path = DATASET_DIR,
) -> pd.DataFrame:
    return _load("seams", date, root, sources, columns)


def load_layers(
    date: str | None = None, sources: list[str] | None = None, root: Path = DATASET_DIR
) -> pd.DataFrame:
    return _load("layers", date, root, sources, None)
