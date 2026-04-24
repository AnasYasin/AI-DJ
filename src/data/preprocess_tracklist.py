#!/usr/bin/env python3
"""Clean raw tracklist.csv → data/processed/tracklist_clean.csv.

Steps:
  1. Drop mixes with <4 or >30 tracks
  2. Drop simultaneous rows
  3. Cap each genre at MAX_MIXES_PER_GENRE mixes (random, fixed seed)
  4. Drop rows with null track_id / track_name / artist_name
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

INPUT_PATH = Path("data/interim/tracklist.csv")
OUTPUT_PATH = Path("data/processed/tracklist_clean.csv")
MIN_TRACKS = 4
MAX_TRACKS = 30
MAX_MIXES_PER_GENRE = 500
SEED = 42


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info("Loaded %d rows, %d mixes from %s", len(df), df["mix_id"].nunique(), path)
    return df


def _drop_bad_mixes(df: pd.DataFrame) -> pd.DataFrame:
    mix_sizes = df.groupby("mix_id")["track_id"].transform("count")
    mask = (mix_sizes >= MIN_TRACKS) & (mix_sizes <= MAX_TRACKS)
    dropped = df["mix_id"].nunique() - df.loc[mask, "mix_id"].nunique()
    log.info(
        "Mix size filter [%d–%d]: dropped %d mixes", MIN_TRACKS, MAX_TRACKS, dropped
    )
    return df.loc[mask].copy()


def _drop_simultaneous(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["play_type"] != "simultaneous"].copy()
    log.info("Dropped %d simultaneous rows", before - len(df))
    return df


def _drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["track_id", "track_name", "artist_name"]).copy()
    log.info("Dropped %d rows with null track_id/track_name/artist_name", before - len(df))
    return df


def _cap_genres(df: pd.DataFrame) -> pd.DataFrame:
    rng = pd.core.common.maybe_make_list  # just use pandas sample below
    kept_mixes = []
    for genre, gdf in df.groupby("genre"):
        mix_ids = gdf["mix_id"].unique()
        if len(mix_ids) <= MAX_MIXES_PER_GENRE:
            kept_mixes.extend(mix_ids)
            log.info("  %-20s %d mixes (under cap, kept all)", genre, len(mix_ids))
        else:
            sampled = (
                pd.Series(mix_ids)
                .sample(n=MAX_MIXES_PER_GENRE, random_state=SEED)
                .tolist()
            )
            kept_mixes.extend(sampled)
            log.info(
                "  %-20s %d → %d mixes (capped)", genre, len(mix_ids), MAX_MIXES_PER_GENRE
            )
    before_mixes = df["mix_id"].nunique()
    df = df[df["mix_id"].isin(kept_mixes)].copy()
    log.info(
        "Genre cap (%d max): %d → %d mixes",
        MAX_MIXES_PER_GENRE,
        before_mixes,
        df["mix_id"].nunique(),
    )
    return df


def _summary(df: pd.DataFrame) -> None:
    log.info("=== Final dataset ===")
    log.info("  Rows          : %d", len(df))
    log.info("  Mixes         : %d", df["mix_id"].nunique())
    log.info("  Unique tracks : %d", df["track_id"].nunique())
    log.info("  DJs           : %d", df["dj_name"].nunique())
    log.info("  Genres        : %d", df["genre"].nunique())
    genre_stats = (
        df.groupby("genre")
        .agg(mixes=("mix_id", "nunique"), tracks=("track_id", "nunique"))
        .sort_values("mixes", ascending=False)
    )
    log.info("\n%s", genre_stats.to_string())


def main() -> None:
    df = _load(INPUT_PATH)
    df = _drop_bad_mixes(df)
    df = _drop_simultaneous(df)
    df = _drop_nulls(df)
    df = _cap_genres(df)
    _summary(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved → %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
