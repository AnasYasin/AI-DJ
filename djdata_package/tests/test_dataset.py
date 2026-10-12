"""The export's id scheme and the old data's pairing, on tiny frames with a known answer."""

import numpy as np
import pandas as pd

from djdata import dataset


def test_ids_follow_the_1001_scheme():
    assert dataset.track_id("Charlotte de Witte & Theo Nasa", "We Are One") == "a5fd33e17392"
    assert dataset.split_title("[05] Santé & Sidney Charles - Butt Head [Sci Tec]") == (
        "Santé & Sidney Charles",
        "Butt Head",
    )
    assert dataset.split_title("no separator here") == ("", "no separator here")
    assert dataset.record_key("Solarstone vs. Sirocco", "Destination") == dataset.record_key(
        "Solarstone vs Sirocco", "Destination"
    )
    assert dataset.dj_key("Aly & Fila") == "alyandfila" == dataset.dj_key("Alyandfila")
    assert 0 <= dataset.fold_of("Solomun") < dataset.N_FOLDS
    assert dataset.genre_of("Trance Progressive Trance Psytrance") == "trance"
    assert dataset.genre_of("drum and base") == "drum and bass"


def test_old_tables_pair_in_play_order_and_keep_overlays(tmp_path):
    tl = pd.DataFrame(
        {
            "mix_id": ["m1"] * 3,
            "mix_title": "DJ A @ Club",
            "dj_name": "Dja",
            "genre": "techno",
            "url": "http://x/m1",
            "track_id": ["t1", "t2", "t3"],
            "starting_time": [10.0, 0.0, 5.0],  # listed out of order; the times decide
            "track_name": ["one", "two", "three"],
            "artist_name": ["a", "b", "c"],
            "play_type": "sequential",
            "overlay_parent": np.nan,
        }
    )
    raw = pd.concat(
        [
            tl,
            pd.DataFrame(
                [
                    {
                        "mix_id": "m1",
                        "track_id": "t9",
                        "play_type": "simultaneous",
                        "overlay_parent": "t2",
                    }
                ]
            ),
        ]
    )
    feats = pd.DataFrame(
        {
            "track_id": ["t1", "t2", "t3", "zz"],
            "embedding": [np.zeros(4, np.float32)] * 4,
            "bpm": 130.0,
            "key": "C",
        }
    )
    tl.to_csv(tmp_path / "clean.csv", index=False)
    raw.to_csv(tmp_path / "raw.csv", index=False)
    feats.to_parquet(tmp_path / "f.parquet", index=False)
    tracks, plays, seams = dataset.old_tables(
        {
            "plays": tmp_path / "clean.csv",
            "raw": tmp_path / "raw.csv",
            "features": tmp_path / "f.parquet",
        }
    )
    assert plays.sort_values("position")["track_id"].tolist() == ["t2", "t3", "t1"]
    pairs = seams[seams["kind"] == "seam"][["track_a", "track_b"]].values.tolist()
    assert pairs == [["t2", "t3"], ["t3", "t1"]]
    ov = seams[seams["kind"] == "overlay"].iloc[0]
    assert (ov["track_a"], ov["track_b"]) == ("t2", "t9")
    assert set(tracks["track_id"]) == {"t1", "t2", "t3"} and (tracks["audio"] == "preview").all()
