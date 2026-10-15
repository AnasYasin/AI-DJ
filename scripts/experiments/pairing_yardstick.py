"""Which kind of model can learn what DJs play next, on the validation DJs, beside the baselines.

Two tasks, both inside the planner's feasible set (same genre, BPM within 5 %, another artist):
  pair    a chosen pair (two consecutive plays by a DJ) against a random feasible pair: AUC
  rank    the true next record among 99 random feasible candidates: hit at 10, MRR

Scored on the validation split (whole DJs the models never saw), on all plays and on the measured
seams alone. Writes a CSV beside the stdout table.

    PYTHONPATH=. python scripts/experiments/pairing_yardstick.py --out reports/pairing_yardstick.csv
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.data.dataset import VALIDATION, load_plays, load_tracks
from src.features.compatibility import compatibility
from src.models.track_arrays import TrackArrays, camelot_distance

log = logging.getLogger("yardstick")
RNG = np.random.default_rng(0)
BPM_TOL = 0.05
N_RANK = 99


# ── data ───────────────────────────────────────────────────────────────────────


def consecutive(plays: pd.DataFrame, idx: dict) -> pd.DataFrame:
    rows = []
    for mix, g in plays.sort_values(["mix_id", "position"]).groupby("mix_id", sort=False):
        t = [x for x in g["track_id"] if x in idx]
        rows += [
            (mix, g["split"].iloc[0], g["genre"].iloc[0], g["source"].iloc[0], a, b)
            for a, b in zip(t, t[1:])
            if a != b
        ]
    return pd.DataFrame(rows, columns=["mix_id", "split", "genre", "source", "a", "b"])


class Feasible:
    """Random candidates inside the planner's rules, per genre, by a BPM window."""

    def __init__(self, A: TrackArrays, genre: np.ndarray, artist: np.ndarray):
        self.A, self.genre, self.artist = A, genre, artist
        self.by_genre = {}
        for g in np.unique(genre):
            rows = np.flatnonzero(genre == g)
            order = rows[np.argsort(A.bpm[rows])]
            self.by_genre[g] = (order, A.bpm[order])

    def sample(self, i: int, k: int, exclude: set) -> np.ndarray:
        order, bpm = self.by_genre[self.genre[i]]
        lo, hi = np.searchsorted(bpm, A_bpm_window(self.A.bpm[i]))
        cand = order[lo:hi]
        cand = cand[
            (cand != i) & ~np.isin(cand, list(exclude)) & (self.artist[cand] != self.artist[i])
        ]
        if len(cand) == 0:
            return np.array([], dtype=int)
        return RNG.choice(cand, min(k, len(cand)), replace=len(cand) < k)


def A_bpm_window(bpm: float) -> tuple[float, float]:
    return bpm * np.exp(-BPM_TOL), bpm * np.exp(BPM_TOL)


# ── features ───────────────────────────────────────────────────────────────────


# no play_count or dj_count: a validation DJ's records are mostly theirs alone (92 % of records have one
# DJ), so record popularity cannot transfer to unseen DJs and inverted in run 2 (AUC 0.24). Artist
# listeners (Last.fm) and the release year are external and stay.
SCALARS = [
    "bpm",
    "energy_mean",
    "loudness_lufs",
    "onset_strength",
    "spectral_centroid",
    "energy_std",
    "release_year",
    "listeners",
]


def rich_features(A: TrackArrays, T: pd.DataFrame, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """The old pair features (minus the Model A cosine) plus each record's own scalars and their gaps."""
    base = _pair_features_fast(A, i, j)
    S = T[SCALARS].to_numpy(dtype=np.float32)
    return np.column_stack(
        [
            base,
            S[i],
            S[j],
            np.abs(S[i] - S[j]),
            np.log1p(np.nan_to_num(S[i, 7])) - np.log1p(np.nan_to_num(S[j, 7])),
        ]
    )


def _pair_features_fast(A: TrackArrays, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.abs(np.log(A.bpm[i] / A.bpm[j])),
            camelot_distance(A.cam_pos[i], A.cam_mode[i], A.cam_pos[j], A.cam_mode[j]),
            np.abs(A.energy[i] - A.energy[j]),
            np.abs(A.loud[i] - A.loud[j]),
            np.abs(A.onset[i] - A.onset[j]),
            np.einsum("ij,ij->i", A.raw[i], A.raw[j]),
        ]
    ).astype(np.float32)


# ── scorers: each maps (i, j, context) -> score, higher = more likely the next record ──


def make_scorers(
    A: TrackArrays,
    T: pd.DataFrame,
    train_pairs: pd.DataFrame,
    feas: Feasible,
    idx: dict,
    device: str = "cpu",
) -> dict:
    pop = np.log1p(T["play_count"].to_numpy(dtype=np.float32))
    scorers = {
        "random": lambda i, j, ctx: RNG.random(len(j)),
        "bpm closeness": lambda i, j, ctx: -np.abs(np.log(A.bpm[i] / A.bpm[j])),
        "key closeness": lambda i, j, ctx: (
            -camelot_distance(A.cam_pos[i], A.cam_mode[i], A.cam_pos[j], A.cam_mode[j])
        ),
        "popularity": lambda i, j, ctx: pop[j],
        "compat formula": lambda i, j, ctx: compatibility(
            camelot_distance(A.cam_pos[i], A.cam_mode[i], A.cam_pos[j], A.cam_mode[j]),
            np.abs(A.energy[i] - A.energy[j]),
            np.abs(A.loud[i] - A.loud[j]),
            np.abs(np.log(A.bpm[i] / A.bpm[j])) * 100,
        ),
        "raw effnet cosine": lambda i, j, ctx: np.einsum("ij,ij->i", A.raw[i], A.raw[j]),
        "crate context (mean of set so far, raw cosine)": lambda i, j, ctx: np.einsum(
            "ij,ij->i", ctx, A.raw[j]
        ),
    }

    # training pairs with feasible negatives, 2 per positive
    ti = train_pairs["a"].map(idx).to_numpy()
    tj = train_pairs["b"].map(idx).to_numpy()
    neg_i, neg_j = [], []
    for a, b in zip(ti, tj):
        c = feas.sample(int(a), 2, {int(b)})
        neg_i += [a] * len(c)
        neg_j += list(c)
    Xi = np.concatenate([ti, np.array(neg_i)])
    Xj = np.concatenate([tj, np.array(neg_j)])
    y = np.concatenate([np.ones(len(ti)), np.zeros(len(neg_i))])
    log.info("training pairs: %d positives, %d feasible negatives", len(ti), len(neg_i))

    t0 = time.time()
    gbm7 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0).fit(
        _pair_features_fast(A, Xi, Xj), y
    )
    log.info("GBM 6 features trained in %.0fs", time.time() - t0)
    scorers["GBM, 6 pair features (old GBM minus Model A)"] = lambda i, j, ctx: gbm7.predict_proba(
        _pair_features_fast(A, i, j)
    )[:, 1]

    t0 = time.time()
    Xr = rich_features(A, T, Xi, Xj)
    gbm_rich = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, random_state=0
    ).fit(Xr, y)
    log.info("GBM rich features (%d) trained in %.0fs", Xr.shape[1], time.time() - t0)
    scorers["GBM, rich pair features"] = lambda i, j, ctx: gbm_rich.predict_proba(
        rich_features(A, T, i, j)
    )[:, 1]

    scorers["pair-encoder MLP (ReLU), raw effnet + scalars"] = train_pair_encoder(
        A, T, Xi, Xj, y, device
    )
    return scorers


def train_pair_encoder(A: TrackArrays, T: pd.DataFrame, Xi, Xj, y, device: str, epochs: int = 8):
    """A small ReLU record encoder on each record, dot product, trained on the same pairs and negatives."""
    import torch
    import torch.nn as nn

    S = T[SCALARS].to_numpy(dtype=np.float32)
    S = (S - np.nanmean(S, 0)) / (np.nanstd(S, 0) + 1e-6)
    S = np.nan_to_num(S)
    X = torch.from_numpy(np.concatenate([A.raw, S], axis=1).astype(np.float32)).to(device)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    X = (X - mu) / sd
    encoder = nn.Sequential(
        nn.Linear(X.shape[1], 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 128)
    ).to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    ii, jj, yy = (
        torch.from_numpy(Xi),
        torch.from_numpy(Xj),
        torch.from_numpy(y.astype(np.float32)).to(device),
    )
    n = len(ii)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n)
        encoder.train()
        tot = 0.0
        for s in range(0, n, 1024):
            b = perm[s : s + 1024]
            zi, zj = encoder(X[ii[b]]), encoder(X[jj[b]])
            logit = (zi * zj).sum(1) / 8.0
            loss = nn.functional.binary_cross_entropy_with_logits(logit, yy[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        log.info("pair-encoder epoch %d loss %.4f (%.0fs)", ep + 1, tot / n, time.time() - t0)
    encoder.eval()
    with torch.no_grad():
        Z = encoder(X).cpu().numpy()
    return lambda i, j, ctx: np.einsum("ij,ij->i", Z[i], Z[j])


# ── evaluation ─────────────────────────────────────────────────────────────────


def evaluate(
    scorers: dict,
    pairs: pd.DataFrame,
    A: TrackArrays,
    feas: Feasible,
    idx: dict,
    plays_by_mix: dict,
) -> pd.DataFrame:
    """AUC on pair vs feasible random, hit@10 and MRR with 99 feasible candidates, per scorer."""
    pi = pairs["a"].map(idx).to_numpy()
    pj = pairs["b"].map(idx).to_numpy()
    neg = [feas.sample(int(a), N_RANK, {int(b)}) for a, b in zip(pi, pj)]
    keep = np.array([len(c) >= 10 for c in neg])
    pi, pj, neg = pi[keep], pj[keep], [c for c, k in zip(neg, keep) if k]
    ctx_rows = []
    for (mix, a), i in zip(pairs.loc[keep, ["mix_id", "a"]].itertuples(index=False), pi):
        before = plays_by_mix[mix]
        before = before[: before.index(a) + 1] if a in before else [a]
        ctx_rows.append(A.raw[[idx[t] for t in before]].mean(0))
    ctx = np.vstack(ctx_rows)
    ctx /= np.linalg.norm(ctx, axis=1, keepdims=True) + 1e-8

    out = []
    for name, f in scorers.items():
        auc_scores = np.concatenate([f(pi, pj, ctx), f(pi, np.array([c[0] for c in neg]), ctx)])
        auc = roc_auc_score(np.r_[np.ones(len(pi)), np.zeros(len(pi))], auc_scores)
        ranks = []
        for k, (i, j, c) in enumerate(zip(pi, pj, neg)):
            cand = np.r_[j, c]
            s = f(np.full(len(cand), i), cand, np.repeat(ctx[k : k + 1], len(cand), axis=0))
            ranks.append(int((s[1:] > s[0]).sum()) + 1)
        r = np.array(ranks)
        out.append(
            {
                "model": name,
                "pairs": len(pi),
                "AUC": round(auc, 3),
                "hit@10": round((r <= 10).mean(), 3),
                "MRR": round((1 / r).mean(), 3),
                "median rank": int(np.median(r)),
            }
        )
        log.info(
            "%-48s AUC %.3f  hit@10 %.3f  MRR %.3f", name, auc, (r <= 10).mean(), (1 / r).mean()
        )
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/pairing_yardstick.csv")
    ap.add_argument("--max-train", type=int, default=60000)
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", force=True
    )  # mlflow configures logging on import

    T = load_tracks().dropna(subset=["bpm"]).reset_index(drop=True)
    T["artist"] = T["artist"].fillna("").str.lower()
    plays = load_plays()
    # popularity from TRAINING plays only: the export's play_count includes the validation play itself,
    # so the true next record always has one and a random candidate often has none (first run: a leak)
    train_plays = plays[plays["split"] != VALIDATION]
    T["play_count"] = (
        T["track_id"].map(train_plays.groupby("track_id").size()).fillna(0).astype(int)
    )
    T["dj_count"] = (
        T["track_id"]
        .map(train_plays.groupby("track_id")["dj_key"].nunique())
        .fillna(0)
        .astype(int)
    )
    A = TrackArrays(T)
    idx = A.idx
    genre = T["genre"].to_numpy()
    artist = T["artist"].to_numpy()
    feas = Feasible(A, genre, artist)
    pairs = consecutive(plays, idx)
    pairs = pairs[
        pairs["a"].map(lambda t: genre[idx[t]]) == pairs["b"].map(lambda t: genre[idx[t]])
    ]  # same-genre pairs only, the planner's rule
    train = pairs[pairs["split"] != VALIDATION]
    val = pairs[pairs["split"] == VALIDATION]
    if len(train) > args.max_train:
        train = train.sample(args.max_train, random_state=0)
    log.info("pairs: train %d, validation %d", len(train), len(val))
    plays_by_mix = {  # only records with features, so every context row has a vector
        m: [x for x in g.sort_values("position")["track_id"] if x in idx]
        for m, g in plays.groupby("mix_id")
    }

    scorers = make_scorers(A, T, train, feas, idx)
    measured = set(
        load_plays(sources=["raveform", "djs"])["mix_id"]
    )  # the measured sources' validation pairs
    tables = []
    for label, sub in (
        ("all validation pairs", val),
        ("measured sources only", val[val["mix_id"].isin(measured)]),
    ):
        t = evaluate(scorers, sub, A, feas, idx, plays_by_mix)
        t.insert(0, "set", label)
        tables.append(t)
        print(f"\n== {label}\n{t.to_string(index=False)}")
    pd.concat(tables).to_csv(args.out, index=False)
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
