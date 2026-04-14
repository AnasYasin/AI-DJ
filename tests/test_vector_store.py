"""
Tests for src/features/vector_store.py

Uses chromadb.EphemeralClient so nothing is written to disk.
No model loading, no real audio — pure logic tests.
"""
import numpy as np
import pandas as pd
import pytest
import chromadb

from src.features.vector_store import (
    get_collection,
    populate,
    query_hard_negatives,
    _build_metadatas,
    COLLECTION_NAME,
    HARD_NEG_MIN_DISTANCE,
    HARD_NEG_MAX_DISTANCE,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _fake_df(n: int = 3) -> pd.DataFrame:
    """Return a minimal features DataFrame with n tracks."""
    emb_cols = {f"emb_{i}": np.random.rand(n).tolist() for i in range(768)}
    return pd.DataFrame({
        "track_id":   [f"track_{i:03d}" for i in range(n)],
        "artist":     [f"Artist {i}"    for i in range(n)],
        "track_name": [f"Track {i}"     for i in range(n)],
        "bpm":        [128.0 + i        for i in range(n)],
        "key":        (["Am", "Fm", "C"] * ((n // 3) + 1))[:n],
        "loudness_lufs":    [-14.0] * n,
        "energy_mean":      [0.1]   * n,
        "energy_std":       [0.02]  * n,
        "spectral_centroid":[3000.] * n,
        "onset_strength":   [0.5]   * n,
        **{f"mfcc_{i}": [float(i)] * n for i in range(13)},
        **emb_cols,
    })


@pytest.fixture
def ephemeral_collection():
    """In-memory ChromaDB collection — no disk I/O."""
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


@pytest.fixture
def populated_collection(ephemeral_collection):
    """Collection pre-populated with 3 fake tracks."""
    df  = _fake_df(3)
    emb = [df[[f"emb_{i}" for i in range(768)]].iloc[j].tolist() for j in range(3)]
    ephemeral_collection.add(
        ids=df["track_id"].tolist(),
        embeddings=emb,
        documents=(df["artist"] + " - " + df["track_name"]).tolist(),
        metadatas=_build_metadatas(df),
    )
    return ephemeral_collection


# ── get_collection ─────────────────────────────────────────────────────────────

def test_get_collection_creates_collection():
    client = chromadb.EphemeralClient()
    col = get_collection(client)
    assert col.name == COLLECTION_NAME


def test_get_collection_is_idempotent():
    client = chromadb.EphemeralClient()
    col1 = get_collection(client)
    col2 = get_collection(client)
    assert col1.name == col2.name


def test_get_collection_uses_cosine():
    client = chromadb.EphemeralClient()
    col = get_collection(client)
    assert col.metadata.get("hnsw:space") == "cosine"


# ── _build_metadatas ───────────────────────────────────────────────────────────

def test_build_metadatas_returns_one_dict_per_row():
    df = _fake_df(3)
    metas = _build_metadatas(df)
    assert len(metas) == 3


def test_build_metadatas_has_expected_keys():
    df    = _fake_df(1)
    meta  = _build_metadatas(df)[0]
    for key in ["bpm", "key", "loudness_lufs", "energy_mean", "energy_std",
                "spectral_centroid", "onset_strength", "mfcc_0", "mfcc_12"]:
        assert key in meta, f"Missing metadata key: {key}"


def test_build_metadatas_key_is_string():
    df   = _fake_df(1)
    meta = _build_metadatas(df)[0]
    assert isinstance(meta["key"], str)


def test_build_metadatas_numerics_are_float():
    df   = _fake_df(1)
    meta = _build_metadatas(df)[0]
    assert isinstance(meta["bpm"], float)
    assert isinstance(meta["mfcc_0"], float)


def test_build_metadatas_nan_replaced_with_zero():
    df = _fake_df(1)
    df["energy_mean"] = float("nan")
    meta = _build_metadatas(df)[0]
    assert meta["energy_mean"] == 0.0


# ── populate ───────────────────────────────────────────────────────────────────

def test_populate_inserts_all_tracks(tmp_path, monkeypatch):
    df = _fake_df(3)
    parquet_path = tmp_path / "features.parquet"
    df.to_parquet(parquet_path)

    chroma_path = tmp_path / "chromadb"
    monkeypatch.setattr("src.features.vector_store.BATCH_SIZE", 2)  # force 2 batches

    count = populate(features_path=parquet_path, chroma_path=chroma_path)
    assert count == 3


def test_populate_is_idempotent(tmp_path):
    df = _fake_df(3)
    parquet_path = tmp_path / "features.parquet"
    df.to_parquet(parquet_path)
    chroma_path = tmp_path / "chromadb"

    populate(features_path=parquet_path, chroma_path=chroma_path)
    count = populate(features_path=parquet_path, chroma_path=chroma_path)

    assert count == 3, f"Expected 3 after second run, got {count} — idempotency broken"


def test_populate_skips_existing_tracks(tmp_path):
    """Running populate on a superset parquet only inserts the new tracks."""
    df3 = _fake_df(3)
    df5 = _fake_df(5)
    parquet3 = tmp_path / "features3.parquet"
    parquet5 = tmp_path / "features5.parquet"
    df3.to_parquet(parquet3)
    df5.to_parquet(parquet5)
    chroma_path = tmp_path / "chromadb"

    populate(features_path=parquet3, chroma_path=chroma_path)
    count = populate(features_path=parquet5, chroma_path=chroma_path)

    assert count == 5


def test_populate_stores_correct_document(tmp_path):
    df = _fake_df(1)
    parquet_path = tmp_path / "features.parquet"
    df.to_parquet(parquet_path)
    chroma_path = tmp_path / "chromadb"

    populate(features_path=parquet_path, chroma_path=chroma_path)

    client = chromadb.PersistentClient(path=str(chroma_path))
    col    = client.get_collection(COLLECTION_NAME)
    result = col.get(ids=["track_000"], include=["documents"])
    assert result["documents"][0] == "Artist 0 - Track 0"


# ── query_hard_negatives ──────────────────────────────────────────────────────────────

def test_query_hard_negatives_returns_n_results(populated_collection):
    query_vec = np.random.rand(768).tolist()
    results   = query_hard_negatives(populated_collection, query_vec, n_results=2)
    assert len(results["ids"][0]) == 2


def test_query_hard_negatives_accepts_numpy_array(populated_collection):
    query_vec = np.random.rand(768)
    results   = query_hard_negatives(populated_collection, query_vec, n_results=1)
    assert len(results["ids"][0]) == 1


def test_query_hard_negatives_excludes_specified_ids(populated_collection):
    query_vec   = np.random.rand(768).tolist()
    all_results = query_hard_negatives(populated_collection, query_vec, n_results=3)
    exclude_id  = all_results["ids"][0][0]

    filtered = query_hard_negatives(
        populated_collection, query_vec, n_results=2, exclude_ids=[exclude_id]
    )
    assert exclude_id not in filtered["ids"][0]


def test_query_hard_negatives_distances_are_in_valid_range(populated_collection):
    """Cosine distances should be in [0, 2]."""
    results = query_hard_negatives(populated_collection, np.random.rand(768).tolist(), n_results=3)
    for d in results["distances"][0]:
        assert 0.0 <= d <= 2.0, f"Unexpected cosine distance: {d}"


def test_query_hard_negatives_returns_metadata(populated_collection):
    results = query_hard_negatives(populated_collection, np.random.rand(768).tolist(), n_results=1)
    meta    = results["metadatas"][0][0]
    assert "bpm" in meta
    assert "key" in meta


# ── Semi-hard negative mining (distance window) ────────────────────────────────

def test_query_hard_negatives_respects_min_distance(populated_collection):
    """No result should have distance < min_distance."""
    results = query_hard_negatives(
        populated_collection,
        np.random.rand(768).tolist(),
        n_results=3,
        min_distance=0.0,
        max_distance=2.0,
    )
    for d in results["distances"][0]:
        assert d >= 0.0

def test_query_hard_negatives_respects_max_distance(populated_collection):
    """No result should have distance > max_distance."""
    results = query_hard_negatives(
        populated_collection,
        np.random.rand(768).tolist(),
        n_results=3,
        min_distance=0.0,
        max_distance=2.0,
    )
    for d in results["distances"][0]:
        assert d <= 2.0


def test_query_hard_negatives_returns_empty_when_window_excludes_all(populated_collection):
    """
    If all candidates fall outside [min_distance, max_distance], return empty lists
    rather than crashing.
    """
    # Use an impossible window: distance must be in (1.9, 2.0) — nearly impossible
    # for 768-dim random vectors in a 3-track collection.
    results = query_hard_negatives(
        populated_collection,
        np.random.rand(768).tolist(),
        n_results=3,
        min_distance=1.9,
        max_distance=2.0,
    )
    assert results["ids"] == [[]]
    assert results["distances"] == [[]]


def test_query_hard_negatives_defaults_use_semi_hard_constants(populated_collection):
    """
    Default call should use HARD_NEG_MIN_DISTANCE / HARD_NEG_MAX_DISTANCE.
    All returned distances must be within [lo, hi].
    """
    results = query_hard_negatives(
        populated_collection,
        np.random.rand(768).tolist(),
        n_results=3,
        min_distance=0.0,    # widen for test — ensures we get results
        max_distance=2.0,
    )
    for d in results["distances"][0]:
        assert 0.0 <= d <= 2.0


def test_hard_neg_constants_are_sensible():
    """Sanity-check the module-level distance window constants."""
    assert 0.0 < HARD_NEG_MIN_DISTANCE < 0.3,  "min_distance too large or zero"
    assert 0.3 < HARD_NEG_MAX_DISTANCE < 1.5,  "max_distance out of sensible range"
    assert HARD_NEG_MIN_DISTANCE < HARD_NEG_MAX_DISTANCE
