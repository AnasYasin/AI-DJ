"""
Merge embeddings.parquet + librosa_features.parquet → features.parquet.

Run after build_features.py (discogs-only + librosa-only) completes on both machines:
    python src/data/preprocess_features.py
"""

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

EMBEDDINGS_PATH = Path("data/processed/embeddings.parquet")
LIBROSA_FEATURES_PATH = Path("data/processed/librosa_features.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")


def _merge() -> pd.DataFrame:
    for path in (EMBEDDINGS_PATH, LIBROSA_FEATURES_PATH):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run build_features.py first")

    emb = pd.read_parquet(EMBEDDINGS_PATH)
    feats = pd.read_parquet(LIBROSA_FEATURES_PATH)
    log.info("embeddings: %d tracks | librosa_features: %d tracks", len(emb), len(feats))

    merged = emb.merge(feats, on="track_id", how="inner")
    lost = len(emb) - len(merged)
    if lost:
        log.warning("  %d tracks in embeddings have no librosa features", lost)

    return merged


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    # placeholder — cleaning steps to be defined
    return df


def run() -> pd.DataFrame:
    df = _merge()
    df = _clean(df)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PATH, index=False)
    log.info("Wrote %d tracks → %s", len(df), FEATURES_PATH)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
