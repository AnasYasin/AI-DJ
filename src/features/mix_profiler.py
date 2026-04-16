"""
Phase 9 — Mix profiler.

Reads tracklist.csv + features.parquet, computes an energy curve shape
for each mix, and writes data/processed/mix_metadata.csv.

Energy curve shapes
───────────────────
  escalating  — monotonically rising energy   (Spearman r > 0.6)
  chill-down  — monotonically falling energy  (Spearman r < -0.6)
  peak-drop   — rises then falls              (peak in middle 40% of mix)
  wave        — oscillating                   (≥ 2 direction reversals)
  plateau     — flat                          (energy range < 0.15)

Output columns: mix_id, dj_name, energy_curve_shape

Run:
  conda activate djtest
  python src/features/mix_profiler.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
TRACKLIST_PATH = Path("data/interim/tracklist.csv")
FEATURES_PATH = Path("data/processed/features.parquet")
MIX_METADATA_PATH = Path("data/processed/mix_metadata.csv")

# ── Thresholds ────────────────────────────────────────────────────────────────
SPEARMAN_RISING = 0.6  # r > this  → escalating
SPEARMAN_FALLING = -0.6  # r < this  → chill-down
PLATEAU_MAX_RANGE = 0.15  # max(energy) - min(energy) < this → plateau
MIN_TRACKS = 3  # skip mixes with fewer than this many tracks with features


# ── Core algorithm ────────────────────────────────────────────────────────────


def _time_gap(t_a: float, t_b: float) -> float:
    """Minutes between two starting_time values with hour rollover (0-59 min)."""
    if t_b >= t_a:
        return t_b - t_a
    return (60.0 - t_a) + t_b


def _sort_by_time(group: pd.DataFrame) -> pd.DataFrame:
    """
    Sort mix tracks by starting_time.
    If all starting_times are NaN (1001tracklists sets with no timestamps),
    fall back to original row order (position in CSV = play order).
    """
    if group["starting_time"].isna().all():
        return group.reset_index(drop=True)
    return group.sort_values("starting_time").reset_index(drop=True)


def _energy_curve_shape(energies: list[float]) -> str:
    """
    Classify the energy trajectory of a mix given an ordered list of
    per-track energy_mean values.

    Priority: plateau → escalating → chill-down → peak-drop → wave
    """
    if len(energies) < MIN_TRACKS:
        return "unknown"

    e = np.array(energies, dtype=float)
    n = len(e)
    x = np.arange(n)

    # ── plateau: very little energy variation ────────────────────────────────
    if e.max() - e.min() < PLATEAU_MAX_RANGE:
        return "plateau"

    # ── Spearman correlation against position index ──────────────────────────
    r, _ = spearmanr(x, e)

    if r > SPEARMAN_RISING:
        return "escalating"
    if r < SPEARMAN_FALLING:
        return "chill-down"

    # ── peak-drop: energy peaks in middle 40% of the mix ────────────────────
    peak_idx = int(np.argmax(e))
    mid_lo = int(0.30 * n)
    mid_hi = int(0.70 * n)
    if mid_lo <= peak_idx <= mid_hi:
        return "peak-drop"

    # ── wave: ≥ 2 direction reversals ────────────────────────────────────────
    diffs = np.diff(e)
    signs = np.sign(diffs[diffs != 0])  # ignore flat steps
    reversals = int(np.sum(np.diff(signs) != 0))
    if reversals >= 2:
        return "wave"

    # Catch-all (weak monotone or irregular)
    return "plateau"


# ── Main ──────────────────────────────────────────────────────────────────────


def compute_mix_metadata(
    tracklist_path: Path = TRACKLIST_PATH,
    features_path: Path = FEATURES_PATH,
) -> pd.DataFrame:
    """
    Build one row per mix_id with its energy curve shape.

    Args:
        tracklist_path: path to tracklist.csv
        features_path:  path to features.parquet

    Returns:
        DataFrame with columns: mix_id, dj_name, energy_curve_shape
    """
    tracklist = pd.read_csv(tracklist_path)
    features = pd.read_parquet(features_path, columns=["track_id", "energy_mean"])

    feat_idx = features.set_index("track_id")["energy_mean"]

    # Only sequential tracks form the energy arc — simultaneous overlays skew it
    if "play_type" in tracklist.columns:
        seq = tracklist[tracklist["play_type"] == "sequential"].copy()
    else:
        seq = tracklist.copy()

    records = []
    skipped = 0

    for mix_id, group in seq.groupby("mix_id"):
        dj_name = group["dj_name"].iloc[0] if "dj_name" in group.columns else ""
        group = _sort_by_time(group)

        # Keep only tracks that have feature data
        track_ids = group["track_id"].tolist() if "track_id" in group.columns else []
        energies = [feat_idx[tid] for tid in track_ids if tid in feat_idx.index]

        if len(energies) < MIN_TRACKS:
            skipped += 1
            continue

        shape = _energy_curve_shape(energies)
        records.append(
            {
                "mix_id": mix_id,
                "dj_name": dj_name,
                "energy_curve_shape": shape,
            }
        )

    df = pd.DataFrame(records, columns=["mix_id", "dj_name", "energy_curve_shape"])

    if skipped:
        log.info("Skipped %d mixes with < %d tracks in features.parquet", skipped, MIN_TRACKS)

    shape_dist = df["energy_curve_shape"].value_counts().to_string()
    log.info("Computed %d mix profiles:\n%s", len(df), shape_dist)

    return df


def main() -> None:
    if not TRACKLIST_PATH.exists():
        log.error("tracklist.csv not found at %s — run scraper first.", TRACKLIST_PATH)
        return
    if not FEATURES_PATH.exists():
        log.error("features.parquet not found at %s — run build_features.py first.", FEATURES_PATH)
        return

    log.info("Loading %s", TRACKLIST_PATH)
    log.info("Loading %s", FEATURES_PATH)

    df = compute_mix_metadata()

    if df.empty:
        log.warning("No mix profiles produced — check features.parquet coverage.")
        return

    MIX_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MIX_METADATA_PATH, index=False)
    log.info("Wrote %d rows → %s", len(df), MIX_METADATA_PATH)


if __name__ == "__main__":
    main()
