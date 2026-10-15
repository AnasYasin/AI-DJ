"""The pair model's data path on a synthetic pool: features, training rows, judging, inference."""

import numpy as np
import pandas as pd
import torch

from src.models import pair_model as pm
from src.models.track_arrays import TrackArrays


def _pool(n=60, seed=0):
    rng = np.random.default_rng(seed)
    keys = ["Am", "C", "Em", "G", "Bm", "D"]
    return pd.DataFrame(
        {
            "track_id": [f"t{i:03d}" for i in range(n)],
            "artist": [f"artist {i % 20}" for i in range(n)],
            "genre": ["techno"] * n,
            "bpm": rng.uniform(128, 134, n),
            "key": [keys[i % len(keys)] for i in range(n)],
            "energy_mean": rng.uniform(0.2, 0.5, n),
            "energy_std": rng.uniform(0.01, 0.1, n),
            "loudness_lufs": rng.uniform(-12, -6, n),
            "onset_strength": rng.uniform(1.1, 2.5, n),
            "spectral_centroid": rng.uniform(1000, 3000, n),
            "release_year": np.nan,
            "listeners": rng.integers(0, 100000, n).astype(float),
            "embedding": list(rng.normal(size=(n, 1280)).astype(np.float32)),
            **{c: rng.normal(size=n) for c in pm.MFCC},
        }
    )


def _plays(n_mix=6, per_mix=8):
    rows = []
    for m in range(n_mix):
        for p in range(per_mix):
            rows.append(
                {
                    "mix_id": f"m{m}",
                    "position": p,
                    "track_id": f"t{(m * per_mix + p) % 60:03d}",
                    "split": "validation" if m == 0 else "train",
                    "dj_key": f"dj{m}",
                    "source": "raveform",
                }
            )
    return pd.DataFrame(rows)


def test_train_rows_judge_and_score(tmp_path):
    T = _pool()
    A = TrackArrays(T)
    plays = _plays()
    pairs = pm.consecutive(plays, A.idx, T["genre"].to_numpy())
    assert len(pairs) == 6 * 7
    plays_by_mix = {
        m: list(g.sort_values("position")["track_id"]) for m, g in plays.groupby("mix_id")
    }
    feas = pm.Feasible(A, np.random.default_rng(0))
    train = pm.Rows(pairs[pairs["split"] != "validation"], feas, A.idx, plays_by_mix, pm.N_SAMPLED)
    val = pm.Rows(pairs[pairs["split"] == "validation"], feas, A.idx, plays_by_mix, 20)
    assert train.before.shape == (35, pm.CTX_LEN) and (train.before[:, -1] == train.i).all()
    assert (val.neg != val.j[:, None]).all(), "the true record is never among its own negatives"

    gap = pm.gap_features(A, train.i, train.j)
    assert gap.shape == (35, 31)
    stats = pm.fit_stats(A, gap)
    X = torch.from_numpy(pm.record_inputs(A, stats))
    assert X.shape == (60, 1280 + 8 + 13 + 4) and torch.isfinite(X).all()

    model, ep, dv = pm.train_one(
        0,
        X,
        A,
        train,
        val,
        stats,
        epochs=1,
        patience=1,
        hparams=dict(hidden=32, d_z=8, dropout=0.1),
    )
    metrics = pm.rank_eval([model], X, A, val, stats, {"all": np.ones(val.n, bool)})
    assert 0 <= metrics["hit@10"] <= 1 and metrics["median rank"] >= 1

    path = tmp_path / "pair.pt"
    torch.save(
        {
            "state_dicts": [model.state_dict()],
            "stats": stats,
            "hparams": dict(hidden=32, d_z=8, dropout=0.1),
        },
        path,
    )
    scorer = pm.PairScorer(A, path)
    cand = np.array([5, 6, 7])
    s = scorer.score(last=2, before=[0, 1, 2], cand=cand)
    assert s.shape == (3,) and np.isfinite(s).all()
    # inference equals the judging path: same logit for the same (last, set so far, candidate)
    model.eval()
    gap = torch.from_numpy(pm._z(pm.gap_features(A, np.full(3, 2), cand), stats["gap"]))
    with torch.no_grad():
        z = model.encoder(X)
        ref = model.score(
            z[2].expand(3, -1), z[cand], gap, z[[0, 1, 2]].mean(0, keepdim=True).expand(3, -1)
        )
    assert np.allclose(s, ref.numpy(), atol=1e-5)
    # an opening slot: "next after candidate" equals scoring the pair the normal way round
    sb = scorer.score_before(nxt=7, cand=np.array([3, 4]))
    assert np.allclose(
        sb,
        [scorer.score(3, [3], np.array([7]))[0], scorer.score(4, [4], np.array([7]))[0]],
        atol=1e-5,
    )
