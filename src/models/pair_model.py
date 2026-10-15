"""
The pair model: the planner's one learned score for "record B after record A, given the set so far".

Shape (chosen on the validation DJs, PAIRING_EXPERIMENTS.md):

    record encoder   FC 1,305 → 512, batch normalisation, ReLU, dropout, FC 512 → 128; one per record
    context vector   mean of the encoded records in the set so far (the last CTX_LEN)
    pair head        FC 799 → 256, ReLU, dropout, FC 256 → 1 logit, over
                     [zA, zB, zA ⊙ zB, |zA − zB|, 31 gap features, c, c ⊙ zB]

Per-record input: unit effnet mean (1,280, scaled by 8), 8 scalars z-scored, 13 MFCCs z-scored, key as
sin/cos on the Camelot wheel, mode and an unknown-key flag. Gap features: the 6 old pair features
(no Model A cosine), both records' scalars, their absolute gaps, the log-listeners difference.

Trained with a softmax cross-entropy over the true next record and N_NEG random feasible candidates
(same genre, BPM within 5 %, another artist); early-stopped on 10 % of the training DJs; SEEDS
initialisations averaged at inference. The validation DJs are only ever scored.

Train:  python -m src.models.pair_model                      (about 3 h on the laptop for 5 seeds)
Use:    scorer = PairScorer(A); scorer.score(last, before, cand)
"""

from __future__ import annotations

import argparse
from datetime import date
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn

from src.data.dataset import VALIDATION, load_plays, load_tracks
from src.models.track_arrays import TrackArrays, camelot_distance

log = logging.getLogger(__name__)

MODEL_PATH = Path("models/pair_model.pt")
CTX_LEN = 16
BPM_TOL = 0.05
N_NEG = 16  # candidates in the softmax beside the true next record
N_SAMPLED = 32  # feasible negatives sampled per training positive, rotated through per epoch
N_RANK = 99  # candidates beside the true record when judging on the validation DJs
HIDDEN, D_Z, DROPOUT = 512, 128, 0.5
# no play_count or dj_count: 92 % of records have one DJ, so record popularity inverts on unseen DJs
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
MFCC = [f"mfcc_{k}" for k in range(13)]
LISTENERS = SCALARS.index("listeners")


# ── features ───────────────────────────────────────────────────────────────────


def _scalars(A: TrackArrays) -> np.ndarray:
    S = A.df[SCALARS].to_numpy(dtype=np.float32)
    S[:, LISTENERS] = np.log1p(np.nan_to_num(S[:, LISTENERS]))
    return S


def fit_stats(A: TrackArrays, gap: np.ndarray) -> dict:
    """Means and stds for the per-record inputs (over the records) and the gap features (over the
    training pairs). Saved with the model; a pool is normalised with these, never its own."""
    S, M = _scalars(A), A.df[MFCC].to_numpy(dtype=np.float32)
    with np.errstate(all="ignore"):  # release_year is 1 % present: its gap column can be all nan
        return {
            "scalar": (np.nanmean(S, 0), np.nanstd(S, 0) + 1e-6),
            "mfcc": (np.nanmean(M, 0), np.nanstd(M, 0) + 1e-6),
            "gap": (
                np.nan_to_num(np.nanmean(gap, 0)),
                np.nan_to_num(np.nanstd(gap, 0), nan=1.0) + 1e-6,
            ),
        }


def _z(X: np.ndarray, stats: tuple) -> np.ndarray:
    return np.nan_to_num((X - stats[0]) / stats[1]).astype(np.float32)


def record_inputs(A: TrackArrays, stats: dict) -> np.ndarray:
    """(N, 1305) per-record input matrix."""
    ang = A.cam_pos * (2 * np.pi / 12)
    key = np.column_stack([np.sin(ang), np.cos(ang), A.cam_mode, A.cam_pos == 0]).astype(
        np.float32
    )
    return np.concatenate(
        [
            A.raw * 8.0,  # unit effnet scaled so its block is not swamped by the scalars
            _z(_scalars(A), stats["scalar"]),
            _z(A.df[MFCC].to_numpy(dtype=np.float32), stats["mfcc"]),
            key,
        ],
        axis=1,
    )


def gap_features(A: TrackArrays, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """(n, 31) raw gap features for record i → record j."""
    S = A.df[SCALARS].to_numpy(dtype=np.float32)
    return np.column_stack(
        [
            np.abs(np.log(A.bpm[i] / A.bpm[j])),
            camelot_distance(A.cam_pos[i], A.cam_mode[i], A.cam_pos[j], A.cam_mode[j]),
            np.abs(A.energy[i] - A.energy[j]),
            np.abs(A.loud[i] - A.loud[j]),
            np.abs(A.onset[i] - A.onset[j]),
            np.einsum("ij,ij->i", A.raw[i], A.raw[j]),
            S[i],
            S[j],
            np.abs(S[i] - S[j]),
            np.log1p(np.nan_to_num(S[i, LISTENERS])) - np.log1p(np.nan_to_num(S[j, LISTENERS])),
        ]
    ).astype(np.float32)


# ── the model ──────────────────────────────────────────────────────────────────


class PairModel(nn.Module):
    def __init__(self, d_in: int, d_gap: int, hidden=HIDDEN, d_z=D_Z, dropout=DROPOUT):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_z),
        )
        self.head = nn.Sequential(
            nn.Linear(6 * d_z + d_gap, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    @staticmethod
    def context(zc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean of the encoded records in the set so far. zc (B, L, d_z), mask True where real."""
        return (zc * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

    def score(self, zi, zj, gap, c) -> torch.Tensor:
        return self.head(torch.cat([zi, zj, zi * zj, (zi - zj).abs(), gap, c, c * zj], 1)).squeeze(
            1
        )

    def forward(self, xi, xj, gap, xc, mask) -> torch.Tensor:
        B, L, D = xc.shape
        zc = self.encoder(xc.reshape(B * L, D)).reshape(B, L, -1)
        return self.score(self.encoder(xi), self.encoder(xj), gap, self.context(zc, mask))


# ── inference ──────────────────────────────────────────────────────────────────


class PairScorer:
    """The saved model over one pool. Record vectors are computed once per seed; the head runs per
    candidate set."""

    def __init__(self, A: TrackArrays, path: str | Path = MODEL_PATH):
        ckpt = torch.load(path, weights_only=False)
        self.A, self.stats = A, ckpt["stats"]
        X = torch.from_numpy(record_inputs(A, self.stats))
        self.models, self.Z = [], []
        with torch.no_grad():
            for sd in ckpt["state_dicts"]:
                m = PairModel(X.shape[1], len(self.stats["gap"][0]), **ckpt["hparams"])
                m.load_state_dict(sd)
                m.eval()
                self.models.append(m)
                self.Z.append(m.encoder(X))

    @torch.no_grad()
    def score(self, last: int, before: list[int], cand: np.ndarray) -> np.ndarray:
        """Logit for each candidate after `last`, given `before` (the set so far, `last` included).
        Mean over the seeds."""
        n = len(cand)
        gap = torch.from_numpy(_z(gap_features(self.A, np.full(n, last), cand), self.stats["gap"]))
        ctx = torch.tensor(before[-CTX_LEN:] or [last])
        out = torch.zeros(n)
        for m, Z in zip(self.models, self.Z):
            c = Z[ctx].mean(0, keepdim=True).expand(n, -1)
            out += m.score(Z[last].expand(n, -1), Z[cand], gap, c)
        return (out / len(self.models)).numpy()

    @torch.no_grad()
    def score_before(self, nxt: int, cand: np.ndarray) -> np.ndarray:
        """Logit for `nxt` after each candidate, the candidate alone as the set so far: for an
        opening slot, which has no previous record."""
        n = len(cand)
        gap = torch.from_numpy(_z(gap_features(self.A, cand, np.full(n, nxt)), self.stats["gap"]))
        out = torch.zeros(n)
        for m, Z in zip(self.models, self.Z):
            out += m.score(Z[cand], Z[nxt].expand(n, -1), gap, Z[cand])
        return (out / len(self.models)).numpy()


# ── training data ──────────────────────────────────────────────────────────────


def consecutive(plays: pd.DataFrame, idx: dict, genre: np.ndarray) -> pd.DataFrame:
    """Consecutive plays in one mix, both records with features and of the same genre."""
    rows = []
    for mix, g in plays.sort_values(["mix_id", "position"]).groupby("mix_id", sort=False):
        t = [x for x in g["track_id"] if x in idx]
        rows += [
            (mix, g["split"].iloc[0], g["dj_key"].iloc[0], g["source"].iloc[0], a, b)
            for a, b in zip(t, t[1:])
            if a != b and genre[idx[a]] == genre[idx[b]]
        ]
    return pd.DataFrame(rows, columns=["mix_id", "split", "dj", "source", "a", "b"])


class Feasible:
    """Random candidates inside the planner's rules: same genre, BPM within BPM_TOL, another artist."""

    def __init__(self, A: TrackArrays, rng: np.random.Generator):
        self.A, self.rng = A, rng
        self.genre = A.df["genre"].to_numpy()
        self.artist = A.df["artist"].fillna("").str.lower().to_numpy()
        self.by_genre = {}
        for g in np.unique(self.genre):
            rows = np.flatnonzero(self.genre == g)
            order = rows[np.argsort(A.bpm[rows])]
            self.by_genre[g] = (order, A.bpm[order])

    def sample(self, i: int, k: int, exclude: int) -> np.ndarray:
        order, bpm = self.by_genre[self.genre[i]]
        lo, hi = np.searchsorted(
            bpm, (self.A.bpm[i] * np.exp(-BPM_TOL), self.A.bpm[i] * np.exp(BPM_TOL))
        )
        cand = order[lo:hi]
        cand = cand[(cand != i) & (cand != exclude) & (self.artist[cand] != self.artist[i])]
        if len(cand) == 0:
            return cand
        return self.rng.choice(cand, min(k, len(cand)), replace=len(cand) < k)


class Rows:
    """One split's positives, their sampled feasible negatives and their set-so-far context."""

    def __init__(self, pairs: pd.DataFrame, feas: Feasible, idx: dict, plays_by_mix: dict, k: int):
        self.i = pairs["a"].map(idx).to_numpy()
        self.j = pairs["b"].map(idx).to_numpy()
        self.n = len(self.i)
        self.neg = np.full((self.n, k), -1, dtype=np.int64)
        for r, (a, b) in enumerate(zip(self.i, self.j)):
            c = feas.sample(int(a), k, int(b))
            self.neg[r, : len(c)] = c
        self.before = np.full((self.n, CTX_LEN), -1, dtype=np.int64)
        for r, (mix, a) in enumerate(pairs[["mix_id", "a"]].itertuples(index=False)):
            seq = plays_by_mix[mix]
            sub = [idx[t] for t in seq[: seq.index(a) + 1]][-CTX_LEN:] if a in seq else [idx[a]]
            self.before[r, -len(sub) :] = sub

    def groups(self, n_neg: int, epoch: int):
        """Rows for the listwise loss: per positive with all `n_neg` negatives present, the positive
        then its negatives, consecutive. Negatives rotate through the sampled columns per epoch."""
        cols = [(epoch * n_neg + c) % self.neg.shape[1] for c in range(n_neg)]
        neg = self.neg[:, cols]
        keep = np.flatnonzero((neg >= 0).all(1))
        j = np.column_stack([self.j[keep], neg[keep]]).ravel()
        return np.repeat(self.i[keep], n_neg + 1), j, np.repeat(keep, n_neg + 1), len(keep)

    def pairs(self, col: int = 0):
        """One positive and one negative per row, for AUC."""
        ok = self.neg[:, col] >= 0
        i = np.r_[self.i, self.i[ok]]
        j = np.r_[self.j, self.neg[ok, col]]
        src = np.r_[np.arange(self.n), np.flatnonzero(ok)]
        return i, j, src, np.r_[np.ones(self.n), np.zeros(ok.sum())]


def tensors(X, A, rows: Rows, stats, i, j, src):
    gap = torch.from_numpy(_z(gap_features(A, i, j), stats["gap"]))
    bef = rows.before[src]
    return X[i], X[j], gap, X[np.where(bef >= 0, bef, 0)], torch.from_numpy(bef >= 0)


@torch.no_grad()
def auc_of(model, X, A, rows: Rows, stats, rng, n_max=8000) -> float:
    model.eval()
    i, j, src, y = rows.pairs()
    if len(i) > n_max:
        keep = rng.choice(len(i), n_max, replace=False)
        i, j, src, y = i[keep], j[keep], src[keep], y[keep]
    s = np.concatenate(
        [
            model(
                *tensors(X, A, rows, stats, i[b : b + 4096], j[b : b + 4096], src[b : b + 4096])
            ).numpy()
            for b in range(0, len(i), 4096)
        ]
    )
    return roc_auc_score(y, s)


@torch.no_grad()
def rank_eval(models: list[PairModel], X, A, val: Rows, stats, groups: dict) -> dict:
    """hit@10, MRR and median rank among the N_RANK feasible candidates, AUC against one of them;
    hit@10 per named subset of the validation rows."""
    for m in models:
        m.eval()
    Z = [m.encoder(X) for m in models]
    ranks = np.full(val.n, np.nan)
    for r in range(val.n):
        cand = val.neg[r][val.neg[r] >= 0]
        if len(cand) < 10:
            continue
        cand = np.r_[val.j[r], cand]
        n = len(cand)
        gap = torch.from_numpy(_z(gap_features(A, np.full(n, val.i[r]), cand), stats["gap"]))
        bef = val.before[r]
        s = np.zeros(n, dtype=np.float32)
        for m, z in zip(models, Z):
            c = z[bef[bef >= 0]].mean(0, keepdim=True).expand(n, -1)
            s += m.score(z[val.i[r]].expand(n, -1), z[cand], gap, c).numpy()
        ranks[r] = int((s[1:] > s[0]).sum()) + 1
    i, j, src, y = val.pairs()
    s = np.zeros(len(i), dtype=np.float32)
    for m in models:
        for b in range(0, len(i), 4096):
            s[b : b + 4096] += m(
                *tensors(X, A, val, stats, i[b : b + 4096], j[b : b + 4096], src[b : b + 4096])
            ).numpy()
    ok = ~np.isnan(ranks)
    out = {
        "val AUC": roc_auc_score(y, s),
        "hit@10": (ranks[ok] <= 10).mean(),
        "MRR": (1 / ranks[ok]).mean(),
        "median rank": float(np.median(ranks[ok])),
    }
    for name, mask in groups.items():
        out[f"hit@10 {name}"] = (ranks[ok & mask] <= 10).mean()
    return {k: round(float(v), 3) for k, v in out.items()}


def train_one(seed, X, A, train: Rows, dev: Rows, stats, epochs, patience, hparams) -> tuple:
    """One initialisation, early-stopped on dev AUC. Returns the model, its best epoch and dev AUC."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = PairModel(X.shape[1], len(stats["gap"][0]), **hparams)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    g = N_NEG + 1
    best, best_state, best_ep, bad, t0 = -1.0, None, 0, 0, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        i, j, src, n_groups = train.groups(N_NEG, ep - 1)
        order = rng.permutation(n_groups)
        bs = max(1024 // g, 8)
        tot = 0.0
        for s in range(0, n_groups, bs):
            b = (order[s : s + bs][:, None] * g + np.arange(g)).ravel()
            logit = model(*tensors(X, A, train, stats, i[b], j[b], src[b])).reshape(-1, g)
            loss = nn.functional.cross_entropy(logit, torch.zeros(len(logit), dtype=torch.long))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(logit)
        tr, dv = auc_of(model, X, A, train, stats, rng), auc_of(model, X, A, dev, stats, rng)
        log.info(
            "seed %d ep %d loss %.4f train AUC %.3f dev AUC %.3f (%.0fs)",
            seed,
            ep,
            tot / n_groups,
            tr,
            dv,
            time.time() - t0,
        )
        if dv > best:
            best, best_ep, bad = dv, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_ep, best


def train(
    seeds=5, epochs=20, patience=3, out: Path = MODEL_PATH, hparams: dict | None = None
) -> dict:
    hparams = hparams or dict(hidden=HIDDEN, d_z=D_Z, dropout=DROPOUT)
    rng = np.random.default_rng(0)
    T = load_tracks().dropna(subset=["bpm"]).reset_index(drop=True)
    A = TrackArrays(T)
    genre = T["genre"].to_numpy()
    plays = load_plays()
    pairs = consecutive(plays, A.idx, genre)
    train_djs = pairs.loc[pairs["split"] != VALIDATION, "dj"].unique()
    dev_djs = set(rng.choice(train_djs, max(3, len(train_djs) // 10), replace=False))
    is_val = pairs["split"] == VALIDATION
    val_p, dev_p = pairs[is_val], pairs[~is_val & pairs["dj"].isin(dev_djs)]
    train_p = pairs[~is_val & ~pairs["dj"].isin(dev_djs)]
    log.info(
        "pairs: train %d (%d DJs), dev %d (%d DJs), validation %d (%d DJs)",
        len(train_p),
        len(train_djs) - len(dev_djs),
        len(dev_p),
        len(dev_djs),
        len(val_p),
        val_p["dj"].nunique(),
    )
    plays_by_mix = {
        m: [x for x in g.sort_values("position")["track_id"] if x in A.idx]
        for m, g in plays.groupby("mix_id")
    }
    feas = Feasible(A, rng)
    tr_rows = Rows(train_p, feas, A.idx, plays_by_mix, N_SAMPLED)
    dv_rows = Rows(dev_p, feas, A.idx, plays_by_mix, 2)
    va_rows = Rows(val_p, feas, A.idx, plays_by_mix, N_RANK)
    stats = fit_stats(A, gap_features(A, tr_rows.i, tr_rows.j))
    X = torch.from_numpy(record_inputs(A, stats))

    models, best = [], []
    for seed in range(seeds):
        m, ep, dv = train_one(seed, X, A, tr_rows, dv_rows, stats, epochs, patience, hparams)
        models.append(m)
        best.append((ep, dv))
        log.info("seed %d: best epoch %d, dev AUC %.3f", seed, ep, dv)
    src = val_p["source"].to_numpy()
    metrics = rank_eval(
        models, X, A, va_rows, stats, {"old": src == "old", "measured": src != "old"}
    )
    metrics["dev AUC"] = round(float(np.mean([d for _, d in best])), 3)
    metrics["best epochs"] = [e for e, _ in best]
    log.info("validation: %s", metrics)
    out.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dicts": [m.state_dict() for m in models],
            "stats": stats,
            "hparams": hparams,
            "metrics": metrics,
            "train": {
                "pairs": len(train_p),
                "n_neg": N_NEG,
                "seeds": seeds,
                "date": str(date.today()),
            },
        },
        out,
    )
    log.info("saved %s", out)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    torch.set_num_threads(6)
    print(train(args.seeds, args.epochs, args.patience, args.out))
