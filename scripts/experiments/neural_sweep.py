"""Neural models for the next-record question: Model A shapes (pair) and Model B shapes (context),
swept on size, inputs, loss, regularisation and extra signals, each judged for over- or
underfitting, on the yardstick's validation DJs.

Same data and same feasible negatives as pairing_yardstick.py (same genre, BPM within 5 %, another
artist). Training DJs are split once more: 10 % of them are a dev set for early stopping, so the
validation DJs are never looked at while training. Per config the CSV gets train AUC, dev AUC, the
best epoch, and validation AUC / hit@10 / MRR (also per source), plus a fit note (train minus dev).

    PYTHONPATH=. python scripts/experiments/neural_sweep.py --out reports/neural_sweep_<date>.csv
    PYTHONPATH=. python scripts/experiments/neural_sweep.py --only "pair M rich" "mean ctx M rich"
    PYTHONPATH=. python scripts/experiments/neural_sweep.py --sweep 2      # the second sweep's configs
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
from scripts.experiments.pairing_yardstick import (
    N_RANK,
    RNG,
    SCALARS,
    Feasible,
    consecutive,
    rich_features,
)
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn

from src.data.dataset import VALIDATION, load_plays, load_tracks
from src.models.track_arrays import TrackArrays

log = logging.getLogger("sweep")
CTX_LEN = 16
MFCC = [f"mfcc_{k}" for k in range(13)]
GENRES = [
    "afro house",
    "drum and bass",
    "house",
    "melodic house",
    "tech house",
    "techno",
    "trance",
]
N_SAMPLED_NEG = (
    32  # feasible negatives sampled per training positive; a config uses 2 to 16 of them
)


# ── per-record inputs ──────────────────────────────────────────────────────────


def _z(S: np.ndarray, fit: np.ndarray | None = None) -> np.ndarray:
    fit = S if fit is None else fit
    S = (S - np.nanmean(fit, 0)) / (np.nanstd(fit, 0) + 1e-6)
    return np.nan_to_num(S).astype(np.float32)


def record_inputs(T: pd.DataFrame, A: TrackArrays, kind: str) -> np.ndarray:
    """`base`: effnet + the 8 yardstick scalars. `rich`: plus the 13 MFCCs and the key as
    sin/cos on the Camelot wheel, mode and an unknown-key flag."""
    S = T[SCALARS].to_numpy(dtype=np.float32)
    S[:, SCALARS.index("listeners")] = np.log1p(np.nan_to_num(S[:, SCALARS.index("listeners")]))
    parts = [A.raw * 8.0, _z(S)]  # unit effnet scaled so its block is not swamped by the scalars
    if kind == "rich":
        ang = A.cam_pos * (2 * np.pi / 12)
        key = np.column_stack([np.sin(ang), np.cos(ang), A.cam_mode, (A.cam_pos == 0)])
        parts += [_z(T[MFCC].to_numpy(dtype=np.float32)), key.astype(np.float32)]
    return np.concatenate(parts, axis=1)


# ── models ─────────────────────────────────────────────────────────────────────


def mlp(d_in: int, hidden: int, d_out: int, dropout: float, bn: bool = True) -> nn.Sequential:
    layers = [nn.Linear(d_in, hidden)]
    if bn:
        layers.append(nn.BatchNorm1d(hidden))
    layers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, d_out)]
    return nn.Sequential(*layers)


class PairModel(nn.Module):
    """One record encoder per record. `head`:
    dot    - dot product of the two record vectors (Model A's shape)
    pair   - an MLP over [zi, zj, zi*zj, |zi-zj|, gap features] (a neural edge scorer)
    gap    - the MLP over the gap features alone (the GBM's inputs, no record encoder)
    `ctx`: none | mean | gru | tfm | attn, a context vector from the records before, fed to the
    head; `attn` is one attention step with the candidate as the query over the set so far.
    `in_drop`: dropout on the input features. `signals`: position in the set and a genre embedding
    into the head."""

    def __init__(
        self,
        d_in,
        d_gap,
        hidden=512,
        d_z=128,
        dropout=0.2,
        head="dot",
        ctx="none",
        in_drop=0.0,
        signals=False,
    ):
        super().__init__()
        self.head, self.ctx, self.in_drop, self.signals = head, ctx, in_drop, signals
        self.encoder = mlp(d_in, hidden, d_z, dropout) if head != "gap" else None
        if ctx == "gru":
            self.seq = nn.GRU(d_z, d_z, batch_first=True)
        elif ctx == "tfm":
            layer = nn.TransformerEncoderLayer(
                d_z, 4, d_z * 2, dropout, batch_first=True, norm_first=True
            )
            self.seq = nn.TransformerEncoder(layer, 1)
            self.pos = nn.Parameter(torch.zeros(CTX_LEN, d_z))
        elif ctx == "attn":
            self.q, self.k, self.v = (nn.Linear(d_z, d_z) for _ in range(3))
        if signals:
            self.genre = nn.Embedding(len(GENRES), 8)
        if head == "dot":
            self.scale = nn.Parameter(torch.tensor(8.0))
        else:
            d = d_gap if head == "gap" else 4 * d_z + d_gap + (2 * d_z if ctx != "none" else 0)
            d += 9 if signals else 0
            self.out = mlp(d, hidden // 2, 1, dropout, bn=False)

    def context(self, zc, mask, zj=None):
        """zc (B, L, d_z) record vectors of the records before, mask True where real; zj the
        candidate vectors, used by `attn` only."""
        if self.ctx == "mean":
            return (zc * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        if self.ctx == "gru":
            out, _ = self.seq(zc)
            last = mask.sum(1).clamp(min=1) - 1
            return out[torch.arange(len(zc)), last]
        if self.ctx == "attn":
            att = torch.einsum("bd,bld->bl", self.q(zj), self.k(zc)) / zc.shape[-1] ** 0.5
            att = att.masked_fill(~mask, -1e9).softmax(1)
            return torch.einsum("bl,bld->bd", att, self.v(zc))
        out = self.seq(zc + self.pos, src_key_padding_mask=~mask)
        return (out * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

    def score(self, zi, zj, gap, c=None, sig=None):
        """The head over record vectors (and the context vector when the model has one)."""
        if self.head == "gap":
            return self.out(gap).squeeze(1)
        if self.head == "dot":
            zi, zj = nn.functional.normalize(zi, dim=1), nn.functional.normalize(zj, dim=1)
            return (zi * zj).sum(1) * self.scale
        feats = [zi, zj, zi * zj, (zi - zj).abs(), gap]
        if c is not None:
            feats += [c, c * zj]
        if self.signals:
            pos, genre = sig
            feats += [pos[:, None], self.genre(genre)]
        return self.out(torch.cat(feats, 1)).squeeze(1)

    def encode(self, x):
        return self.encoder(nn.functional.dropout(x, self.in_drop, self.training))

    def forward(self, xi, xj, gap, xc=None, mask=None, sig=None):
        if self.head == "gap":
            return self.score(None, None, gap)
        zi, zj, c = self.encode(xi), self.encode(xj), None
        if self.ctx != "none":
            B, L, D = xc.shape
            c = self.context(self.encode(xc.reshape(B * L, D)).reshape(B, L, -1), mask, zj)
        return self.score(zi, zj, gap, c, sig)


# ── data for training ──────────────────────────────────────────────────────────


def before_matrix(pairs, plays_by_mix, idx):
    """(n, CTX_LEN) record indices before and including `a`, -1 padded on the left, and the
    position of `a` in its mix as a fraction."""
    out = np.full((len(pairs), CTX_LEN), -1, dtype=np.int64)
    pos = np.zeros(len(pairs), dtype=np.float32)
    for r, (mix, a) in enumerate(pairs[["mix_id", "a"]].itertuples(index=False)):
        seq = plays_by_mix[mix]
        k = seq.index(a) if a in seq else 0
        pos[r] = k / max(len(seq) - 1, 1)
        sub = [idx[t] for t in seq[: k + 1]][-CTX_LEN:] or [idx[a]]
        out[r, -len(sub) :] = sub
    return out, pos


def negatives(pairs, feas, idx, k):
    """(n, k) feasible negatives per pair, -1 where the genre window had too few."""
    pi, pj = pairs["a"].map(idx).to_numpy(), pairs["b"].map(idx).to_numpy()
    out = np.full((len(pairs), k), -1, dtype=np.int64)
    for r, (a, b) in enumerate(zip(pi, pj)):
        c = feas.sample(int(a), k, {int(b)})
        out[r, : len(c)] = c
    return out


class Rows:
    """Rows for one split: positive pairs, their sampled negatives (with their cosine to A, for hard
    negatives), their context, position, genre and sample weight."""

    def __init__(self, pairs, feas, idx, A, plays_by_mix, k_neg):
        self.i = pairs["a"].map(idx).to_numpy()
        self.j = pairs["b"].map(idx).to_numpy()
        self.neg = negatives(pairs, feas, idx, k_neg)
        ok = self.neg >= 0
        self.sim = np.where(
            ok, np.einsum("nd,nkd->nk", A.raw[self.i], A.raw[np.where(ok, self.neg, 0)]), -9.0
        )
        self.before, self.pos = before_matrix(pairs, plays_by_mix, idx)
        self.genre = pairs["genre"].map(GENRES.index).fillna(0).to_numpy(dtype=np.int64)
        self.weight = pairs["weight"].to_numpy(dtype=np.float32) if "weight" in pairs else None
        self.n = len(self.i)
        self.groups = {}  # name -> boolean mask over the rows, for a per-subset breakdown

    def epoch_rows(self, n_neg, epoch, hard=False):
        """Positives plus `n_neg` negatives each. Random: rotate through the sampled columns per
        epoch. Hard: half the closest to A by raw cosine, half rotating random."""
        k = self.neg.shape[1]
        if hard:
            h = n_neg // 2
            hard_cols = np.argsort(-self.sim, axis=1)[:, :h]
            rot = [(epoch * (n_neg - h) + c) % k for c in range(n_neg - h)]
            cols = np.concatenate([hard_cols, np.broadcast_to(rot, (self.n, len(rot)))], axis=1)
            neg = np.take_along_axis(self.neg, cols, axis=1)
        else:
            cols = [(epoch * n_neg + c) % k for c in range(n_neg)]
            neg = self.neg[:, cols]
        ok = neg >= 0
        rows = np.repeat(np.arange(self.n), n_neg)[ok.ravel()]
        i = np.r_[self.i, self.i[rows]]
        j = np.r_[self.j, neg.ravel()[ok.ravel()]]
        src = np.r_[np.arange(self.n), rows]
        y = np.r_[np.ones(self.n), np.zeros(len(rows))].astype(np.float32)
        return i, j, src, y

    def list_rows(self, n_neg, epoch, hard=False):
        """Groups of (positive, n_neg negatives) for the listwise loss: only positives with all
        negatives present. Returns flat i, j, src and the group count; rows of one group are
        consecutive, the positive first."""
        i, j, src, _ = self.epoch_rows(n_neg, epoch, hard)
        neg_src, neg_j = src[self.n :], j[self.n :]
        keep = np.flatnonzero(np.bincount(neg_src, minlength=self.n) == n_neg)
        neg_j = neg_j[np.isin(neg_src, keep)].reshape(len(keep), n_neg)
        flat_j = np.column_stack([self.j[keep], neg_j]).ravel()
        flat_i = np.repeat(self.i[keep], n_neg + 1)
        flat_src = np.repeat(keep, n_neg + 1)
        return flat_i, flat_j, flat_src, len(keep)


# ── train and judge one config ─────────────────────────────────────────────────


def tensors(model, X, A, T, rows, i, j, src, gap_stats):
    gap = (rich_features(A, T, i, j) - gap_stats[0]) / gap_stats[1]
    gap = torch.from_numpy(np.nan_to_num(gap).astype(np.float32))
    xc = mask = sig = None
    if model.ctx != "none":
        bef = rows.before[src]
        mask = torch.from_numpy(bef >= 0)
        xc = X[torch.from_numpy(np.where(bef >= 0, bef, 0))]
    if model.signals:
        sig = (torch.from_numpy(rows.pos[src]), torch.from_numpy(rows.genre[src]))
    return X[torch.from_numpy(i)], X[torch.from_numpy(j)], gap, xc, mask, sig


@torch.no_grad()
def scores(models, X, A, T, rows, i, j, src, gap_stats) -> np.ndarray:
    """Mean logit over the models (one, or an ensemble of seeds)."""
    out = np.zeros(len(i), dtype=np.float32)
    for m in models:
        m.eval()
        for s in range(0, len(i), 4096):
            b = slice(s, s + 4096)
            out[b] += m(*tensors(m, X, A, T, rows, i[b], j[b], src[b], gap_stats)).numpy()
    return out / len(models)


def auc_of(models, X, A, T, rows, gap_stats, n_max=8000) -> float:
    i, j, src, y = rows.epoch_rows(1, 0)
    if len(i) > n_max:
        keep = RNG.choice(len(i), n_max, replace=False)
        i, j, src, y = i[keep], j[keep], src[keep], y[keep]
    return roc_auc_score(y, scores(models, X, A, T, rows, i, j, src, gap_stats))


@torch.no_grad()
def rank_eval(models, X, A, T, val: Rows, gap_stats) -> dict:
    """hit@10 and MRR among the 99 feasible candidates, AUC against the first one. Record vectors
    are computed once for every record; the head runs per candidate."""
    Z = []
    for m in models:
        m.eval()
        Z.append(
            torch.cat([m.encoder(X[s : s + 4096]) for s in range(0, len(X), 4096)])
            if m.encoder is not None
            else None
        )
    ranks = np.full(val.n, np.nan)
    for r in range(val.n):
        cand = val.neg[r][val.neg[r] >= 0]
        if len(cand) < 10:
            continue
        cand = np.r_[val.j[r], cand]
        n = len(cand)
        gap = (rich_features(A, T, np.full(n, val.i[r]), cand) - gap_stats[0]) / gap_stats[1]
        gap = torch.from_numpy(np.nan_to_num(gap).astype(np.float32))
        s = np.zeros(n, dtype=np.float32)
        for m, z in zip(models, Z):
            zi = zj = c = sig = None
            if z is not None:
                zi, zj = z[val.i[r]].expand(n, -1), z[cand]
            if m.ctx != "none":
                bef = val.before[r : r + 1]
                mask = torch.from_numpy(bef >= 0)
                zc = z[torch.from_numpy(np.where(bef >= 0, bef, 0))]
                if m.ctx == "attn":
                    c = m.context(zc.expand(n, -1, -1), mask.expand(n, -1), zj)
                else:
                    c = m.context(zc, mask).expand(n, -1)
            if m.signals:
                sig = (
                    torch.full((n,), float(val.pos[r])),
                    torch.full((n,), int(val.genre[r]), dtype=torch.long),
                )
            s += m.score(zi, zj, gap, c, sig).numpy()
        ranks[r] = int((s[1:] > s[0]).sum()) + 1
    r = ranks[~np.isnan(ranks)]
    i, j, src, y = val.epoch_rows(1, 0)
    auc = roc_auc_score(y, scores(models, X, A, T, val, i, j, src, gap_stats))
    out = {
        "val AUC": auc,
        "hit@10": (r <= 10).mean(),
        "MRR": (1 / r).mean(),
        "median rank": int(np.median(r)),
    }
    for name, m in val.groups.items():  # hit@10 per validation subset (by source)
        rr = ranks[m & ~np.isnan(ranks)]
        out[f"hit@10 {name}"] = (rr <= 10).mean() if len(rr) else np.nan
    return out


def train_one(name, cfg, X, A, T, train, dev, gap_stats, epochs, patience, seed):
    """One model, early-stopped on dev AUC. Training options (popped from cfg):
    loss bce|listwise, n_neg, hard, wd, smooth, weighted."""
    cfg = dict(cfg)
    loss_kind = cfg.pop("loss", "bce")
    n_neg = cfg.pop("n_neg", 2)
    hard = cfg.pop("hard", False)
    wd = cfg.pop("wd", 1e-4)
    smooth = cfg.pop("smooth", 0.0)
    weighted = cfg.pop("weighted", False)
    torch.manual_seed(seed)
    model = PairModel(X.shape[1], gap_stats[0].shape[0], **cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    w_all = (
        train.weight if (weighted and train.weight is not None) else np.ones(train.n, np.float32)
    )
    best, best_state, best_row, bad, t0, history = -1.0, None, None, 0, time.time(), []
    for ep in range(epochs):
        model.train()
        if loss_kind == "listwise":
            i, j, src, n_groups = train.list_rows(n_neg, ep, hard)
            g = n_neg + 1
            order = RNG.permutation(n_groups)
            bs = max(1024 // g, 8)
        else:
            i, j, src, y = train.epoch_rows(n_neg, ep, hard)
            yt = torch.from_numpy(y * (1 - smooth) + smooth / 2)
            order = RNG.permutation(len(i))
            bs = 1024
        tot, cnt = 0.0, 0
        for s in range(0, len(order), bs):
            b = order[s : s + bs]
            if loss_kind == "listwise":
                rows_b = (b[:, None] * g + np.arange(g)).ravel()
                logit = model(
                    *tensors(model, X, A, T, train, i[rows_b], j[rows_b], src[rows_b], gap_stats)
                )
                logit = logit.reshape(len(b), g)
                w = torch.from_numpy(w_all[src[rows_b][::g]])
                loss = nn.functional.cross_entropy(
                    logit,
                    torch.zeros(len(b), dtype=torch.long),
                    reduction="none",
                    label_smoothing=smooth,
                )
            else:
                logit = model(*tensors(model, X, A, T, train, i[b], j[b], src[b], gap_stats))
                w = torch.from_numpy(w_all[src[b]])
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logit, yt[b], reduction="none"
                )
            loss = (loss * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            cnt += len(b)
        tr = auc_of([model], X, A, T, train, gap_stats)
        dv = auc_of([model], X, A, T, dev, gap_stats)
        history.append((ep + 1, tot / cnt, tr, dv))
        log.info(
            "%s seed %d ep %d loss %.4f train AUC %.3f dev AUC %.3f (%.0fs)",
            name,
            seed,
            ep + 1,
            tot / cnt,
            tr,
            dv,
            time.time() - t0,
        )
        if dv > best:
            best, bad = dv, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_row = history[-1]
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_row, len(history)


def run_config(name, cfg, X, A, T, train, dev, val, gap_stats, epochs, patience):
    t0 = time.time()
    seeds = list(range(cfg.pop("seeds", 1)))
    models, rows, n_ep = [], [], 0
    for seed in seeds:
        m, best_row, ran = train_one(
            name, cfg, X, A, T, train, dev, gap_stats, epochs, patience, seed
        )
        models.append(m)
        rows.append(best_row)
        n_ep += ran
    n_params = sum(p.numel() for p in models[0].parameters())
    tr = float(np.mean([r[2] for r in rows]))
    dv = float(np.mean([r[3] for r in rows]))
    ev = rank_eval(models, X, A, T, val, gap_stats)
    fit = "overfits" if tr - dv > 0.06 else ("underfits" if tr < 0.75 else "ok")
    return {
        "config": name,
        "params": n_params,
        "seeds": len(seeds),
        "best epoch": rows[0][0],
        "epochs run": n_ep,
        "train AUC": round(tr, 3),
        "dev AUC": round(dv, 3),
        "fit": fit,
        **{k: round(float(v), 3) for k, v in ev.items()},
        "seconds": int(time.time() - t0),
    }


# ── the sweeps ─────────────────────────────────────────────────────────────────

M = dict(
    head="pair", hidden=512, d_z=128, dropout=0.3, inputs="rich", ctx="mean"
)  # the best of sweep 1

CONFIGS = {
    # Model A shapes: two record encoders, dot product
    "dot S base": dict(head="dot", hidden=256, d_z=64, dropout=0.2),
    "dot M base": dict(head="dot", hidden=512, d_z=128, dropout=0.2),
    "dot L base": dict(head="dot", hidden=1024, d_z=256, dropout=0.3),
    "dot M rich": dict(head="dot", hidden=512, d_z=128, dropout=0.2, inputs="rich"),
    # neural edge scorer: record encoders plus a pair head over the gaps
    "pair S rich": dict(head="pair", hidden=256, d_z=64, dropout=0.2, inputs="rich"),
    "pair M rich": dict(head="pair", hidden=512, d_z=128, dropout=0.3, inputs="rich"),
    "pair L rich": dict(head="pair", hidden=1024, d_z=256, dropout=0.4, inputs="rich"),
    "gap MLP": dict(head="gap", hidden=256, dropout=0.2),
    # Model B shapes: the pair head with a context vector from the records before
    "mean ctx M rich": M,
    "gru ctx M rich": dict(M, ctx="gru"),
    "tfm ctx M rich": dict(M, ctx="tfm"),
    "tfm ctx L rich": dict(M, hidden=1024, d_z=256, dropout=0.4, ctx="tfm"),
}

SWEEP2 = {
    # 1. train on the ranking: softmax over the positive and n negatives; random or hard negatives
    "listwise 8 random": dict(M, loss="listwise", n_neg=8),
    "listwise 16 random": dict(M, loss="listwise", n_neg=16),
    "listwise 16 hard": dict(M, loss="listwise", n_neg=16, hard=True),
    "bce 8 hard": dict(M, n_neg=8, hard=True),
    # 2. regularisation on the best model
    "reg dropout .5": dict(M, dropout=0.5),
    "reg wd 1e-2": dict(M, wd=1e-2),
    "reg input dropout .1": dict(M, in_drop=0.1),
    "reg smooth .1": dict(M, smooth=0.1),
    "reg all": dict(M, dropout=0.5, wd=1e-2, in_drop=0.1, smooth=0.1),
    # 3. five seeds averaged
    "5 seeds": dict(M, seeds=5),
    # 4. the candidate queries the set
    "attn ctx": dict(M, ctx="attn"),
    # 5. signals not fed in yet
    "signals position + genre": dict(M, signals=True),
    "seam hold weight": dict(M, weighted=True),
}


def hold_weight(pairs, plays):
    """Measured sources: the incoming record's play length relative to the DJ's median, clipped
    to [0.5, 2]. Old source (no play spans): 1."""
    p = plays.dropna(subset=["played_from_s", "played_to_s"]).copy()
    p["hold"] = p["played_to_s"] - p["played_from_s"]
    p = p[p["hold"] > 0]
    p["rel"] = p["hold"] / p.groupby("dj_key")["hold"].transform("median")
    rel = p.drop_duplicates(["mix_id", "track_id"]).set_index(["mix_id", "track_id"])["rel"]
    key = pd.MultiIndex.from_arrays([pairs["mix_id"], pairs["b"]])
    return rel.reindex(key).to_numpy().clip(0.5, 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/neural_sweep.csv")
    ap.add_argument("--only", nargs="*")
    ap.add_argument(
        "--sweep", type=int, default=1, help="1: the shapes; 2: loss, regularisation, signals"
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=60000)
    ap.add_argument(
        "--train-sources",
        nargs="*",
        help="train and dev pairs from these sources only (all by default)",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    torch.set_num_threads(6)

    T = load_tracks().dropna(subset=["bpm"]).reset_index(drop=True)
    T["artist"] = T["artist"].fillna("").str.lower()
    plays = load_plays()
    A = TrackArrays(T)  # no Model A here
    idx = A.idx
    genre, artist = T["genre"].to_numpy(), T["artist"].to_numpy()
    feas = Feasible(A, genre, artist)
    pairs = consecutive(plays, idx)
    pairs = pairs[
        pairs["a"].map(lambda t: genre[idx[t]]) == pairs["b"].map(lambda t: genre[idx[t]])
    ]
    dj_of_mix = plays.drop_duplicates("mix_id").set_index("mix_id")["dj_key"]
    pairs["dj"] = pairs["mix_id"].map(dj_of_mix)
    pairs["weight"] = np.nan_to_num(hold_weight(pairs, plays), nan=1.0)
    train_djs = pairs.loc[pairs["split"] != VALIDATION, "dj"].unique()
    dev_djs = set(RNG.choice(train_djs, max(3, len(train_djs) // 10), replace=False))
    val_p = pairs[pairs["split"] == VALIDATION]
    dev_p = pairs[(pairs["split"] != VALIDATION) & pairs["dj"].isin(dev_djs)]
    train_p = pairs[(pairs["split"] != VALIDATION) & ~pairs["dj"].isin(dev_djs)]
    if args.train_sources:
        dev_p = dev_p[dev_p["source"].isin(args.train_sources)]
        train_p = train_p[train_p["source"].isin(args.train_sources)]
    if len(train_p) > args.max_train:
        train_p = train_p.sample(args.max_train, random_state=0)
    log.info(
        "pairs: train %d (%d DJs), dev %d (%d DJs), validation %d; seam-weighted training pairs %d",
        len(train_p),
        len(train_djs) - len(dev_djs),
        len(dev_p),
        len(dev_djs),
        len(val_p),
        int((train_p["weight"] != 1.0).sum()),
    )
    plays_by_mix = {
        m: [x for x in g.sort_values("position")["track_id"] if x in idx]
        for m, g in plays.groupby("mix_id")
    }
    t0 = time.time()
    train = Rows(train_p, feas, idx, A, plays_by_mix, N_SAMPLED_NEG)
    dev = Rows(dev_p, feas, idx, A, plays_by_mix, 2)
    val = Rows(val_p, feas, idx, A, plays_by_mix, N_RANK)
    src = val_p["source"].to_numpy()
    val.groups = {"old": src == "old", "measured": src != "old"}
    log.info("negatives and contexts built in %.0fs", time.time() - t0)
    g = rich_features(A, T, train.i, train.j)
    with np.errstate(all="ignore"):  # release_year is 1 % present: its gap column is all nan
        gap_stats = (
            np.nan_to_num(np.nanmean(g, 0)),
            np.nan_to_num(np.nanstd(g, 0), nan=1.0) + 1e-6,
        )

    inputs = {k: torch.from_numpy(record_inputs(T, A, k)) for k in ("base", "rich")}
    configs = CONFIGS if args.sweep == 1 else SWEEP2
    results = []
    for name, cfg in configs.items():
        if args.only and name not in args.only:
            continue
        cfg = dict(cfg)
        X = inputs[cfg.pop("inputs", "base")]
        row = run_config(
            name, cfg, X, A, T, train, dev, val, gap_stats, args.epochs, args.patience
        )
        results.append(row)
        log.info("RESULT %s", row)
        pd.DataFrame(results).to_csv(args.out, index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
