"""
Pairwise transition edge scorer (GBM) — the beam-search edge weight.

Gradient-boosted trees on pair features:
  |log bpm ratio|, Camelot wheel distance, |Δenergy|, |Δloudness|, |Δonset|,
  raw-embedding cosine, Model-A projection cosine.

Trained on consecutive pairs (positives) vs random same-genre pairs
(negatives) from TRAIN mixes only (split_mixes.csv). Val AUC ≈ 0.69 —
the strongest pairwise signal we have (scalars alone 0.66, encoder 0.66).

Train + persist:  python -m src.models.edge_scorer
Use:              scorer = EdgeScorer.load(); scorer.score(feats)
"""

import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.train_model import _CAMELOT

log = logging.getLogger(__name__)

FEATURES_PATH = Path("data/processed/features.parquet")
TRACKLIST_PATH = Path("data/processed/tracklist_clean.csv")
SPLIT_PATH = Path("data/processed/split_mixes.csv")
MODEL_PATH = Path("models/edge_gbm.pkl")

FEATURE_NAMES = ["bpm_lr", "cam_dist", "d_energy", "d_loud", "d_onset", "cos_raw", "cos_proj"]


def camelot_arrays(keys: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Key strings → (wheel position 1-12, mode 0/1) arrays. Unknown key → pos 0."""
    pos = keys.map(_CAMELOT).fillna(0).to_numpy(dtype=np.float32)
    mode = (~keys.astype(str).str.endswith("m")).to_numpy(dtype=np.float32)
    return pos, mode


def camelot_distance(pa, ma, pb, mb) -> np.ndarray:
    """Wheel distance + 0.5 mode penalty; unknown keys (pos 0) → neutral 2.5."""
    d = np.abs(pa - pb)
    d = np.minimum(d, 12 - d) + 0.5 * (ma != mb)
    return np.where((pa == 0) | (pb == 0), 2.5, d)


class TrackArrays:
    """Column arrays for a track pool — shared by trainer and planner."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.tid = self.df["track_id"].to_numpy()
        self.idx = {t: i for i, t in enumerate(self.tid)}
        self.bpm = self.df["bpm"].to_numpy(dtype=np.float32)
        self.energy = self.df["energy_mean"].to_numpy(dtype=np.float32)
        self.loud = self.df["loudness_lufs"].to_numpy(dtype=np.float32)
        self.onset = self.df["onset_strength"].to_numpy(dtype=np.float32)
        self.cam_pos, self.cam_mode = camelot_arrays(self.df["key"])
        raw = np.stack([np.asarray(e, dtype=np.float32) for e in self.df["embedding"]])
        self.raw = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)
        proj = np.stack([np.asarray(e, dtype=np.float32) for e in self.df["embedding_proj"]])
        self.proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8)


def pair_features(A: TrackArrays, i: int, cand: np.ndarray) -> np.ndarray:
    """Feature matrix for track i → each candidate index in `cand`. (n, 7)"""
    return np.column_stack(
        [
            np.abs(np.log(A.bpm[i] / A.bpm[cand])),
            camelot_distance(A.cam_pos[i], A.cam_mode[i], A.cam_pos[cand], A.cam_mode[cand]),
            np.abs(A.energy[i] - A.energy[cand]),
            np.abs(A.loud[i] - A.loud[cand]),
            np.abs(A.onset[i] - A.onset[cand]),
            A.raw[cand] @ A.raw[i],
            A.proj[cand] @ A.proj[i],
        ]
    ).astype(np.float32)


class EdgeScorer:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "EdgeScorer":
        with open(path, "rb") as fh:
            return cls(pickle.load(fh))

    def score(self, feats: np.ndarray) -> np.ndarray:
        """(n, 7) pair features → P(good transition) per row."""
        return self.model.predict_proba(feats)[:, 1]


def train() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    f = pd.read_parquet(FEATURES_PATH).drop_duplicates("track_id")
    A = TrackArrays(f)
    tc = pd.read_csv(TRACKLIST_PATH).sort_values(["mix_id", "starting_time"])
    split = pd.read_csv(SPLIT_PATH).set_index("mix_id")["split"]
    artist = tc.drop_duplicates("track_id").set_index("track_id")["artist_name"].to_dict()

    rng = np.random.default_rng(42)
    X, y = {"train": [], "val": []}, {"train": [], "val": []}
    pool_by = {"train": defaultdict(list), "val": defaultdict(list)}
    for mid, grp in tc.groupby("mix_id"):
        s = split.get(mid)
        if s in ("train", "val"):
            g = grp["genre"].iloc[0]
            pool_by[s][g] += [t for t in grp.track_id if t in A.idx]
    for mid, grp in tc.groupby("mix_id"):
        s = split.get(mid)
        if s not in ("train", "val"):
            continue
        g = grp["genre"].iloc[0]
        pool = pool_by[s][g]
        tt = [t for t in grp.track_id if t in A.idx]
        for a, b in zip(tt, tt[1:]):
            if a == b or artist.get(a) == artist.get(b):
                continue
            X[s].append(pair_features(A, A.idx[a], np.array([A.idx[b]]))[0])
            y[s].append(1)
            while True:
                i, j = rng.integers(0, len(pool), 2)
                if pool[i] != pool[j]:
                    break
            X[s].append(pair_features(A, A.idx[pool[i]], np.array([A.idx[pool[j]]]))[0])
            y[s].append(0)

    Xtr, ytr = np.array(X["train"]), np.array(y["train"])
    Xv, yv = np.array(X["val"]), np.array(y["val"])
    log.info("pairs — train %d, val %d", len(ytr), len(yv))
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0)
    model.fit(Xtr, ytr)
    auc = roc_auc_score(yv, model.predict_proba(Xv)[:, 1])
    log.info("EdgeScorer val AUC: %.3f", auc)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(model, fh)
    log.info("saved → %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train()
