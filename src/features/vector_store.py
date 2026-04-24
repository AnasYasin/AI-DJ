"""
ChromaDB vector store population — Phase 3.

Reads features.parquet (produced by build_features.py) and inserts every track
into a ChromaDB persistent collection named "tracks".

Each entry stores:
  id        — track_id (12-char MD5 hex, stable across runs)
  embedding — MERT 768-dim vector (used for cosine nearest-neighbor search)
  metadata  — librosa features + genre + dj_name (joined from tracklist_clean.csv)
  document  — "Artist - Track Name" (human-readable label)

genre and dj_name are required for:
  - Training: within-genre semi-hard negative mining (query_hard_negatives genre filter)
  - Inference: track_selector filters ChromaDB by genre+bpm to build candidate pool

Idempotent: tracks already in the collection are skipped on re-runs.
Batched:    inserts in chunks of BATCH_SIZE (ChromaDB recommends ≤ 5000 per call).

Output: data/processed/chromadb/  (ChromaDB persistent store)
"""

import logging
from pathlib import Path
import time

import chromadb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FEATURES_PATH = Path("data/processed/features.parquet")
TRACKLIST_PATH = Path("data/processed/tracklist_clean.csv")
CHROMA_PATH = Path("data/processed/chromadb")
COLLECTION_NAME = "tracks"
BATCH_SIZE = 500

# Librosa feature columns stored as metadata (everything except track_id and embedding)
_LIBROSA_COLS = [
    "bpm",
    "key",
    "loudness_lufs",
    "energy_mean",
    "energy_std",
    "spectral_centroid",
    "onset_strength",
    *[f"mfcc_{i}" for i in range(13)],
]
# Additional metadata joined from tracklist_clean.csv
_EXTRA_COLS = ["genre", "dj_name", "artist_name", "track_name"]


# ── Client factory ─────────────────────────────────────────────────────────────


def get_client(path: str | Path = CHROMA_PATH) -> chromadb.PersistentClient:
    """
    Return a ChromaDB PersistentClient rooted at `path`.
    Data survives process restarts — same as SQLite but for vectors.
    Pass path=None or use EphemeralClient() in tests to avoid touching disk.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def get_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """
    Get or create the 'tracks' collection with cosine distance.
    get_or_create is idempotent — safe to call on every run.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── Population ─────────────────────────────────────────────────────────────────


def _load_track_metadata(tracklist_path: str | Path = TRACKLIST_PATH) -> pd.DataFrame:
    """
    Build a track_id → (genre, dj_name, artist_name, track_name) mapping from tracklist_clean.csv.

    Tracks appear in multiple mixes — take the first occurrence of genre and dj_name per track_id
    (consistent across runs since tracklist_clean.csv is deterministically ordered).
    """
    tl = pd.read_csv(
        tracklist_path, usecols=["track_id", "genre", "dj_name", "artist_name", "track_name"]
    )
    tl = tl.dropna(subset=["track_id"])
    # First occurrence wins — deterministic since CSV order is stable
    return tl.drop_duplicates(subset="track_id", keep="first").set_index("track_id")


def populate(
    features_path: str | Path = FEATURES_PATH,
    chroma_path: str | Path = CHROMA_PATH,
    tracklist_path: str | Path = TRACKLIST_PATH,
) -> int:
    """
    Read features.parquet and upsert all tracks into ChromaDB.

    Returns the total number of tracks now in the collection.

    Steps:
      1. Load features.parquet + join genre/dj_name from tracklist_clean.csv
      2. Connect to (or create) the persistent ChromaDB store
      3. Identify tracks not yet in the collection
      4. Insert in batches of BATCH_SIZE, logging progress every batch
    """
    df = pd.read_parquet(features_path)
    log.info("Loaded %d tracks from %s", len(df), features_path)

    # Join genre, dj_name, artist_name, track_name from tracklist
    track_meta = _load_track_metadata(tracklist_path)
    before = len(df)
    overlap = [c for c in track_meta.columns if c in df.columns]
    if overlap:
        df = df.drop(columns=overlap)
    df = df.join(track_meta, on="track_id", how="left")
    missing_genre = df["genre"].isna().sum()
    if missing_genre:
        log.warning("%d tracks have no genre in tracklist — filling 'unknown'", missing_genre)
        df["genre"] = df["genre"].fillna("unknown")
        df["dj_name"] = df["dj_name"].fillna("unknown")
        df["artist_name"] = df["artist_name"].fillna("")
        df["track_name"] = df["track_name"].fillna("")
    log.info("Joined tracklist metadata: %d/%d tracks have genre", before - missing_genre, before)

    client = get_client(chroma_path)
    collection = get_collection(client)

    # ── Skip already-inserted tracks (idempotency) ─────────────────────────
    existing_count = collection.count()
    if existing_count > 0:
        existing_ids = set(collection.get(include=[])["ids"])
        before = len(df)
        df = df[~df["track_id"].isin(existing_ids)]
        log.info(
            "Collection already has %d tracks. Skipping %d duplicates. Inserting %d new.",
            existing_count,
            before - len(df),
            len(df),
        )
    else:
        log.info("Collection is empty. Inserting all %d tracks.", len(df))

    if df.empty:
        log.info("Nothing to insert.")
        return collection.count()

    if "embedding" not in df.columns:
        raise ValueError("features.parquet missing 'embedding' column")

    # ── Batch insert ───────────────────────────────────────────────────────
    total = len(df)
    inserted = 0
    t_start = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        chunk = df.iloc[batch_start : batch_start + BATCH_SIZE]

        ids = chunk["track_id"].tolist()
        embeddings = [e.tolist() if hasattr(e, "tolist") else list(e) for e in chunk["embedding"]]
        metadatas = _build_metadatas(chunk)

        has_labels = "artist_name" in chunk.columns and "track_name" in chunk.columns
        documents = (
            (chunk["artist_name"] + " - " + chunk["track_name"]).tolist() if has_labels else None
        )

        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

        inserted += len(chunk)
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        elapsed = time.time() - t_start
        rate = inserted / elapsed if elapsed > 0 else 0
        log.info(
            "Batch %d/%d — inserted %d/%d tracks (%.1f tracks/s)",
            batch_num,
            total_batches,
            inserted,
            total,
            rate,
        )

    final_count = collection.count()
    log.info(
        "Done. Collection '%s' now has %d tracks. (%.1fs total)",
        COLLECTION_NAME,
        final_count,
        time.time() - t_start,
    )
    return final_count


def _build_metadatas(chunk: pd.DataFrame) -> list[dict]:
    """
    Build the metadata list for a DataFrame chunk.

    ChromaDB metadata values must be str, int, or float — no None, no numpy types.
    - float columns: cast to Python float, replace NaN with 0.0
    - string columns (key, genre, dj_name): cast to str, fallback to "unknown"
    """
    _STR_COLS = {"key", "genre", "dj_name"}
    all_cols = _LIBROSA_COLS + ["genre", "dj_name"]
    metas = []
    for _, row in chunk.iterrows():
        meta = {}
        for col in all_cols:
            if col not in chunk.columns:
                continue
            val = row[col]
            if col in _STR_COLS:
                meta[col] = str(val) if (val is not None and str(val) != "nan") else "unknown"
            else:
                v = float(val)
                meta[col] = 0.0 if (v != v) else v  # NaN check: NaN != NaN
        metas.append(meta)
    return metas


# ── Semi-hard negative mining defaults ────────────────────────────────────────
#
# Cosine distance range [0, 2]:
#   0.00–0.10  "almost identical" — likely false negatives: tracks from another
#              mix that genuinely mix well with the anchor. Treating them as
#              negatives would actively hurt training. EXCLUDED via min_distance.
#   0.10–0.60  "semi-hard zone" — similar enough to be a real training challenge,
#              different enough to probably not be an unlabelled positive pair.
#              These are the useful hard negatives.
#   0.60–2.00  "easy" — trivially distinguishable (often different genre/tempo).
#              Model learns nothing from these. EXCLUDED via max_distance.
#
# These defaults were chosen based on within-genre electronic music.
# Tune via min_distance / max_distance args if you add very different genres.

HARD_NEG_MIN_DISTANCE = 0.10  # below this → likely false negative, skip
HARD_NEG_MAX_DISTANCE = 0.60  # above this → too easy, skip


# ── Nearest-neighbour query helper (used by training loop in Phase 5) ──────────


def query_hard_negatives(
    collection: chromadb.Collection,
    embedding: list[float] | np.ndarray,
    n_results: int = 20,
    exclude_ids: list[str] | None = None,
    genre: str | None = None,
    min_distance: float | None = None,
    max_distance: float | None = None,
) -> dict:
    """
    Find `n_results` semi-hard negative tracks for contrastive training.

    Args:
        collection:   ChromaDB collection to search.
        embedding:    768-dim MERT vector of the anchor track.
        n_results:    How many hard negatives to return.
        exclude_ids:  Track IDs to unconditionally exclude.
                      In Phase 5 this MUST include:
                        - the anchor track_id itself
                        - ALL track_ids that are consecutive to the anchor in
                          any mix (the known positive pairs).
                      Failing to exclude positives leaks them into the negative
                      set and corrupts the NT-Xent loss.
        genre:        If set, restrict search to this genre (within-genre negatives).
                      Enforced via ChromaDB where filter: {"genre": {"$eq": genre}}.
                      Cross-genre hard negatives are rejected — fuzzy genre
                      boundaries risk false negatives that poison the loss.
        min_distance: Cosine distance lower bound (default: HARD_NEG_MIN_DISTANCE).
                      Tracks closer than this are treated as potential false
                      negatives — they may be genuine mix partners that just
                      lack a label. Skipped to avoid poisoning the loss.
        max_distance: Cosine distance upper bound (default: HARD_NEG_MAX_DISTANCE).
                      Tracks farther than this are trivially easy negatives with
                      no training signal. Skipped for efficiency.

    Returns:
        Dict with keys ids, distances, metadatas, documents (each a list-of-lists,
        matching chromadb.Collection.query() output format).
    """
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()

    lo = min_distance if min_distance is not None else HARD_NEG_MIN_DISTANCE
    hi = max_distance if max_distance is not None else HARD_NEG_MAX_DISTANCE
    exclude_set = set(exclude_ids or [])

    # Over-fetch: we'll lose some candidates to distance filtering and exclusions.
    # Fetch 4× what we need so the window still yields n_results after filtering.
    fetch_n = (n_results + len(exclude_set) + 1) * 4
    where = {"genre": {"$eq": genre}} if genre else None
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(fetch_n, collection.count()),
        where=where,
        include=["metadatas", "distances", "documents"],
    )

    ids = results["ids"][0]
    distances = results["distances"][0]
    metas = results["metadatas"][0]
    docs = results["documents"][0]

    # Apply: exclude known positives, then apply distance window
    filtered = [
        (i, d, m, doc)
        for i, d, m, doc in zip(ids, distances, metas, docs)
        if i not in exclude_set and lo <= d <= hi
    ][:n_results]

    if not filtered:
        return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

    f_ids, f_dist, f_metas, f_docs = zip(*filtered)
    return {
        "ids": [list(f_ids)],
        "distances": [list(f_dist)],
        "metadatas": [list(f_metas)],
        "documents": [list(f_docs)],
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _fmt = "%(asctime)s %(levelname)s %(message)s"
    _log_path = Path("logs/vector_store.log")
    _log_path.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=_fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(_log_path, encoding="utf-8"),
        ],
    )
    log.info("Logging to %s", _log_path)
    populate()
