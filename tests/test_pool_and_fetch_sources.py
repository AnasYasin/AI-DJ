"""The pool's audio and dj filters, the pool-own energy floor, and the fetcher's held and link paths."""

import numpy as np
import pandas as pd

from src.data import track_fetcher as tf
from src.models import predict_model as pm


def _tracks(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "track_id": [f"t{i:03d}" for i in range(n)],
            "artist": [f"artist {i}" for i in range(n)],
            "title": [f"title {i}" for i in range(n)],
            "genre": "techno",
            "source": ["raveform" if i % 2 else "old" for i in range(n)],
            "source_ref": [f"yt{i}" if i % 2 else f"t{i:03d}" for i in range(n)],
            "audio": ["full" if i % 2 else "preview" for i in range(n)],
            "bpm": 130.0 + rng.integers(0, 6, n),
            "key": "Am",
            "energy_mean": np.linspace(0.1, 0.5, n),
            "loudness_lufs": -10.0,
            "onset_strength": 1.5,
            "embedding": [np.zeros(1280, np.float32)] * n,
        }
    )


def _patch(monkeypatch, tracks, plays=None):
    monkeypatch.setattr(pm, "load_tracks", lambda **k: tracks.copy())
    monkeypatch.setattr(
        pm,
        "load_plays",
        lambda **k: plays if plays is not None else pd.DataFrame(columns=["dj_key", "track_id"]),
    )
    monkeypatch.setattr(
        "src.models.edge_scorer.project_tracks", lambda df: np.zeros((len(df), 128), np.float32)
    )


def test_audio_full_keeps_only_held_records(monkeypatch):
    _patch(monkeypatch, _tracks())
    pool = pm._load_pool("techno", (128, 138), 4, None, None, None, audio="full")
    assert (pool["f"]["audio"] == "full").all() and len(pool["f"]) == 20


def test_dj_filter_keeps_only_that_djs_records(monkeypatch):
    plays = pd.DataFrame(
        {"dj_key": ["solomun"] * 25 + ["other"] * 5, "track_id": [f"t{i:03d}" for i in range(30)]}
    )
    _patch(monkeypatch, _tracks(), plays)
    pool = pm._load_pool("techno", (128, 138), 4, None, None, None, dj="Solomun")
    assert set(pool["f"]["track_id"]) <= {f"t{i:03d}" for i in range(25)} and len(pool["f"]) == 25


def test_energy_floor_is_a_percentile_of_the_pool(monkeypatch):
    t = _tracks()
    t.loc[t.index[:20], "genre"] = "trance"  # loud trance rows must not raise the techno floor
    t.loc[t.index[:20], "energy_mean"] = 5.0
    _patch(monkeypatch, t)
    pool = pm._load_pool("techno", (128, 138), 4, None, None, 50)
    assert 8 <= len(pool["f"]) <= 11  # half of the 20 techno rows, not none of them


def test_plan_entries_carry_the_source(monkeypatch):
    _patch(monkeypatch, _tracks())
    pool = pm._load_pool("techno", (128, 138), 4, None, None, None)
    entry = pm._plan_entry(pool["A"], pool["meta"], 1, 1, 0.3, 0.5)
    assert (
        entry["source"] == "raveform" and entry["source_ref"] == "yt1" and entry["audio"] == "full"
    )


def test_fetch_copies_a_held_file_without_searching(monkeypatch, tmp_path):
    held = tmp_path / "held"
    held.mkdir()
    (held / "yt1.m4a").write_bytes(b"audio")
    monkeypatch.setitem(tf.HELD, "raveform", (held, "s3://nowhere/"))
    monkeypatch.setattr(
        tf, "_search", lambda *a, **k: (_ for _ in ()).throw(AssertionError("searched"))
    )
    res = tf.fetch_track(
        "a", "b", "t001", tmp_path / "out", None, source="raveform", source_ref="yt1", audio="full"
    )
    assert res["status"] == "ok" and (tmp_path / "out" / "t001.m4a").read_bytes() == b"audio"


def test_fetch_uses_the_known_link_then_registers(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setitem(tf.HELD, "raveform", (tmp_path / "none", "s3://nowhere/"))
    monkeypatch.setattr(
        tf, "_search", lambda *a, **k: (_ for _ in ()).throw(AssertionError("searched"))
    )

    def fake_download(url, dest):
        seen["url"] = url
        dest = dest.with_suffix(".m4a")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest

    monkeypatch.setattr(tf, "_download", fake_download)
    monkeypatch.setattr(tf, "_decoded_seconds", lambda p: 400.0)
    monkeypatch.setattr(tf, "_register", lambda path, tid: seen.setdefault("registered", tid))
    res = tf.fetch_track(
        "a",
        "b",
        "t001",
        tmp_path / "out",
        None,
        source="raveform",
        source_ref="yt1",
        audio="preview",
    )
    assert res["status"] == "ok" and seen["url"].endswith("v=yt1") and seen["registered"] == "t001"
