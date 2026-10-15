"""track_features on two synthetic tracks in two corpora, the effnet model stubbed."""

import shutil

import numpy as np
import pandas as pd

from src.features import track_features as stf
from src.features.build_features import VALID_KEYS


def _fake_embed(y16):
    n = max(1, len(y16) // stf.SR_EMBED)  # one patch per second, like essentia's default hop
    return np.random.default_rng(0).standard_normal((n, 1280)).astype(np.float16)


def test_two_corpora_end_to_end_and_resume(tmp_path, tmp_audio_file, monkeypatch):
    monkeypatch.setattr(stf, "embed_track", _fake_embed)
    dirs = {}
    for corpus, tid in (("rave", "vidA"), ("djs", "t1001")):
        d = tmp_path / corpus / "tracks"
        d.mkdir(parents=True)
        shutil.copy(tmp_audio_file, d / f"{tid}.wav")
        (d / "notes.txt").write_text("ignored")
        dirs[corpus] = d
    tempos = {"rave": tmp_path / "rave_tempos.csv"}
    pd.DataFrame({"track_id": ["vidA"], "bpm": [127.5], "seconds": [4.0]}).to_csv(
        tempos["rave"], index=False
    )
    out, emb = tmp_path / "out.parquet", tmp_path / "embeddings"

    table = stf.run(dirs, tempos, out, emb, workers=1)

    assert len(table) == 2 and set(table.corpus) == {"rave", "djs"}
    row = table.set_index("track_id").loc["vidA"]
    assert row.bpm == 127.5 and row.bpm_source == "measured"
    assert pd.isna(table.set_index("track_id").loc["t1001"].bpm)
    assert row.key in VALID_KEYS and np.isfinite(row.loudness_lufs)
    assert len(row.embedding) == 1280 and row.n_vectors == 5  # the fixture is 5 s long
    saved = np.load(emb / "vidA.npy")
    assert saved.shape == (5, 1280) and saved.dtype == np.float16
    assert np.allclose(saved.astype(np.float32).mean(axis=0), np.asarray(row.embedding), atol=1e-3)
    for col in ("onset_strength", "energy_mean", "spectral_centroid", "mfcc_12", "minutes"):
        assert col in table.columns

    # a second run finds everything done and changes nothing
    again = stf.run(dirs, tempos, out, emb, workers=1)
    assert len(again) == 2 and list(again.track_id) == list(table.track_id)
