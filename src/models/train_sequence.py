"""
Model B — sequence model for next-track prediction (two-tower context tower).

Causal transformer over the tracks played so far in a mix; at every position it
predicts the Model-A embedding of the track the DJ actually played next
(InfoNCE against in-batch + same-genre pool negatives). Genre conditioning via
a learned prefix token.

Token = Model-A projection (128) + [position fraction, bpm_norm, Δbpm, Δenergy].
Targets = frozen Model-A projections (embedding_proj in features.parquet).

Eval: held-out mixes (split_mixes.csv), recall@k of the true next track among
all embedded tracks of the same genre, vs. the context-free Model-A baseline
(query = last track's embedding).

Run:
  conda activate aidj
  python -m src.models.train_sequence
"""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

FEATURES_PATH = Path("data/processed/features.parquet")
TRACKLIST_PATH = Path("data/processed/tracklist_clean.csv")
SPLIT_PATH = Path("data/processed/split_mixes.csv")
MODELS_DIR = Path("models")
SEQ_MODEL_PATH = MODELS_DIR / "sequence_model.pt"

HPARAMS = {
    "d_model": 192,
    "n_layers": 3,
    "n_heads": 4,
    "ff_dim": 384,
    "dropout": 0.2,
    "max_len": 24,  # tokens per training window (avg mix ≈ 16 tracks)
    "min_len": 4,
    "temperature": 0.07,
    "pool_negatives": 256,  # random same-genre targets added per batch
    "batch_mixes": 16,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "epochs": 100,
    "seed": 42,
}

TOKEN_EXTRA = 4  # pos_frac, bpm_norm, d_bpm, d_energy


# ── Data ───────────────────────────────────────────────────────────────────────


def load_sequences():
    """Per-mix track sequences with Model-A embeddings and scalar deltas."""
    f = pd.read_parquet(FEATURES_PATH).drop_duplicates("track_id").set_index("track_id")
    if "embedding_proj" not in f.columns:
        raise RuntimeError("features.parquet has no embedding_proj — train Model A first")
    proj = {t: np.asarray(v, dtype=np.float32) for t, v in f["embedding_proj"].items()}
    bpm = f["bpm"].to_dict()
    energy = f["energy_mean"].to_dict()

    tc = pd.read_csv(TRACKLIST_PATH).sort_values(["mix_id", "starting_time"])
    split = pd.read_csv(SPLIT_PATH).set_index("mix_id")["split"]

    seqs = {"train": [], "val": []}
    for mix_id, grp in tc.groupby("mix_id"):
        which = split.get(mix_id)
        if which not in ("train", "val"):
            continue
        tids = [t for t in dict.fromkeys(grp["track_id"]) if t in proj]
        if len(tids) < HPARAMS["min_len"]:
            continue
        seqs[which].append({"genre": grp["genre"].iloc[0], "tids": tids})

    genres = sorted({s["genre"] for s in seqs["train"]})
    log.info(
        "Sequences — train: %d, val: %d | genres: %s",
        len(seqs["train"]),
        len(seqs["val"]),
        genres,
    )
    return seqs, proj, bpm, energy, genres


def seq_tensors(tids, genre_pool_unused, proj, bpm, energy):
    """Token features (L, 132) and target embeddings (L, 128): target[i] = emb(tids[i+1])."""
    L = len(tids)
    emb = np.stack([proj[t] for t in tids])  # (L, 128)
    b = np.array([bpm[t] for t in tids], dtype=np.float32)
    e = np.array([energy[t] for t in tids], dtype=np.float32)
    d_bpm = np.r_[0.0, np.diff(b)] / 20.0
    d_energy = np.r_[0.0, np.diff(e)] * 5.0
    pos = np.arange(L, dtype=np.float32) / max(L - 1, 1)
    extra = np.stack([pos, b / 200.0, d_bpm, d_energy], axis=1).astype(np.float32)
    tokens = np.concatenate([emb, extra], axis=1)  # (L, 132)
    return tokens[:-1], emb[1:]  # predict next: token i → target emb of track i+1


class MixBatcher:
    """Genre-homogeneous batches of mixes (so negatives are within-genre)."""

    def __init__(self, seqs, batch_mixes, seed):
        self.by_genre = defaultdict(list)
        for s in seqs:
            self.by_genre[s["genre"]].append(s)
        self.batch_mixes = batch_mixes
        self.rng = np.random.default_rng(seed)

    def epoch_batches(self):
        batches = []
        for genre, mixes in self.by_genre.items():
            order = self.rng.permutation(len(mixes))
            for k in range(0, len(order), self.batch_mixes):
                chunk = [mixes[i] for i in order[k : k + self.batch_mixes]]
                if len(chunk) >= 2:
                    batches.append((genre, chunk))
        self.rng.shuffle(batches)
        return batches


# ── Model ──────────────────────────────────────────────────────────────────────


class SequenceModel(nn.Module):
    def __init__(self, genres, hp):
        super().__init__()
        d = hp["d_model"]
        self.input_proj = nn.Linear(128 + TOKEN_EXTRA, d)
        self.genre_emb = nn.Embedding(len(genres), d)
        self.genre_to_idx = {g: i for i, g in enumerate(genres)}
        self.pos_emb = nn.Embedding(hp["max_len"] + 1, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=hp["n_heads"],
            dim_feedforward=hp["ff_dim"],
            dropout=hp["dropout"],
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=hp["n_layers"])
        self.head = nn.Linear(d, 128)

    def forward(self, tokens, genre_idx, pad_mask):
        """
        tokens:   (B, L, 132)  pad_mask: (B, L) True = padding
        returns:  (B, L, 128) unit-norm predicted next-track embeddings
        """
        B, L, _ = tokens.shape
        x = self.input_proj(tokens)
        g = self.genre_emb(genre_idx).unsqueeze(1)  # (B, 1, d)
        x = torch.cat([g, x], dim=1)  # genre prefix token
        x = x + self.pos_emb(torch.arange(L + 1, device=x.device)).unsqueeze(0)
        causal = nn.Transformer.generate_square_subsequent_mask(L + 1, device=x.device)
        pad = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad_mask], dim=1
        )
        h = self.encoder(x, mask=causal, src_key_padding_mask=pad)
        return F.normalize(self.head(h[:, 1:]), dim=-1)  # drop genre token


# ── Loss ───────────────────────────────────────────────────────────────────────


def info_nce(preds, targets, tids, pool_embs, tau):
    """
    preds/targets: (N, 128) valid positions, unit-norm. tids: list[str] of the
    N target track ids (to mask duplicate-track false negatives).
    pool_embs: (M, 128) extra same-genre negatives.
    """
    logits_batch = preds @ targets.T / tau  # (N, N)
    tid_arr = np.array(tids)
    dup = torch.from_numpy((tid_arr[None, :] == tid_arr[:, None])).to(preds.device)
    eye = torch.eye(len(tids), dtype=torch.bool, device=preds.device)
    logits_batch = logits_batch.masked_fill(dup & ~eye, float("-inf"))
    logits = torch.cat([logits_batch, preds @ pool_embs.T / tau], dim=1)
    return F.cross_entropy(logits, torch.arange(len(tids), device=preds.device))


# ── Eval: next-track retrieval on held-out mixes ───────────────────────────────


def evaluate(model, seqs_val, proj, bpm, energy, genre_tracks, device, min_context=3):
    """
    For every val position with ≥min_context tracks of context, rank the true
    next track among ALL embedded tracks of that genre (minus already-played).
    Baseline: context-free Model A — query = last played track's embedding.
    Returns dict of recall@k and MRR for model and baseline.
    """
    model.eval()
    genre_mat = {
        g: (np.stack([proj[t] for t in ts]), {t: i for i, t in enumerate(ts)})
        for g, ts in genre_tracks.items()
    }
    stats = {"model": [], "baseline": []}
    with torch.no_grad():
        for s in seqs_val:
            tids, g = s["tids"], s["genre"]
            mat, t2i = genre_mat[g]
            tokens, _ = seq_tensors(tids, None, proj, bpm, energy)
            L = tokens.shape[0]
            if L > HPARAMS["max_len"]:
                tokens = tokens[-HPARAMS["max_len"] :]
            tk = torch.from_numpy(tokens).unsqueeze(0).to(device)
            gi = torch.tensor([model.genre_to_idx[g]], device=device)
            pm = torch.zeros(1, tk.shape[1], dtype=torch.bool, device=device)
            preds = model(tk, gi, pm)[0].cpu().numpy()  # (L, 128)
            off = len(tids) - 1 - preds.shape[0]  # crop offset

            for i in range(preds.shape[0]):
                ctx_end = off + i  # context = tids[:ctx_end+1]
                if ctx_end + 1 < min_context:
                    continue
                true_next = tids[ctx_end + 1]
                if true_next not in t2i:  # track's canonical genre differs (multi-genre track)
                    continue
                played = set(tids[: ctx_end + 1])
                for name, q in (("model", preds[i]), ("baseline", proj[tids[ctx_end]])):
                    sims = mat @ q
                    for p in played:
                        if p in t2i:
                            sims[t2i[p]] = -2.0
                    rank = int((sims > sims[t2i[true_next]]).sum()) + 1
                    stats[name].append(rank)

    out = {}
    for name, ranks in stats.items():
        r = np.array(ranks)
        out[name] = {
            "n": len(r),
            "recall@1": float((r <= 1).mean()),
            "recall@10": float((r <= 10).mean()),
            "recall@50": float((r <= 50).mean()),
            "recall@100": float((r <= 100).mean()),
            "MRR": float((1 / r).mean()),
            "median_rank": float(np.median(r)),
        }
    return out


# ── Training ───────────────────────────────────────────────────────────────────


def collate(chunk, proj, bpm, energy, max_len, rng):
    """Batch of mixes → padded tokens, target embs, target tids, pad mask."""
    toks, tgts, tids = [], [], []
    for s in chunk:
        t = s["tids"]
        if len(t) - 1 > max_len:  # random crop window
            start = rng.integers(0, len(t) - 1 - max_len + 1)
            t = t[start : start + max_len + 1]
        tok, tgt = seq_tensors(t, None, proj, bpm, energy)
        toks.append(tok)
        tgts.append(tgt)
        tids.append(t[1:])
    L = max(x.shape[0] for x in toks)
    B = len(toks)
    tokens = np.zeros((B, L, 128 + TOKEN_EXTRA), dtype=np.float32)
    pad = np.ones((B, L), dtype=bool)
    for i, x in enumerate(toks):
        tokens[i, : x.shape[0]] = x
        pad[i, : x.shape[0]] = False
    return tokens, pad, tgts, tids


def train():
    hp = HPARAMS
    torch.manual_seed(hp["seed"])
    rng = np.random.default_rng(hp["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    seqs, proj, bpm, energy, genres = load_sequences()
    genre_tracks = defaultdict(list)
    tc = pd.read_csv(TRACKLIST_PATH)
    for t, g in tc.drop_duplicates("track_id")[["track_id", "genre"]].itertuples(index=False):
        if t in proj:
            genre_tracks[g].append(t)

    model = SequenceModel(genres, hp).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("SequenceModel: %.1fk params", n_params / 1e3)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"])
    batcher = MixBatcher(seqs["train"], hp["batch_mixes"], hp["seed"])

    pool_by_genre = {
        g: np.stack([proj[t] for t in ts]) for g, ts in genre_tracks.items()
    }

    best_r10 = 0.0
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        losses = []
        for genre, chunk in batcher.epoch_batches():
            tokens, pad, tgts, tids = collate(chunk, proj, bpm, energy, hp["max_len"], rng)
            tk = torch.from_numpy(tokens).to(device)
            pm = torch.from_numpy(pad).to(device)
            gi = torch.tensor([model.genre_to_idx[genre]] * len(chunk), device=device)
            preds = model(tk, gi, pm)  # (B, L, 128)

            flat_preds, flat_tgts, flat_tids = [], [], []
            for i, (tgt, tid) in enumerate(zip(tgts, tids)):
                n = tgt.shape[0]
                flat_preds.append(preds[i, :n])
                flat_tgts.append(torch.from_numpy(tgt).to(device))
                flat_tids += list(tid)
            fp = torch.cat(flat_preds)
            ft = F.normalize(torch.cat(flat_tgts), dim=1)

            pool = pool_by_genre[genre]
            sel = rng.choice(len(pool), min(hp["pool_negatives"], len(pool)), replace=False)
            pool_t = F.normalize(torch.from_numpy(pool[sel]).to(device), dim=1)

            loss = info_nce(fp, ft, flat_tids, pool_t, hp["temperature"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        if epoch % 5 == 0 or epoch == 1:
            ev = evaluate(model, seqs["val"], proj, bpm, energy, genre_tracks, device)
            m = ev["model"]
            log.info(
                "epoch %d/%d  loss=%.3f | val r@10=%.3f r@50=%.3f MRR=%.4f med=%d (n=%d)",
                epoch,
                hp["epochs"],
                float(np.mean(losses)),
                m["recall@10"],
                m["recall@50"],
                m["MRR"],
                int(m["median_rank"]),
                m["n"],
            )
            if m["recall@10"] > best_r10:
                best_r10 = m["recall@10"]
                torch.save(
                    {"state_dict": model.state_dict(), "genres": genres, "hparams": hp},
                    SEQ_MODEL_PATH,
                )
                log.info("  ↑ new best r@10=%.3f → %s", best_r10, SEQ_MODEL_PATH)
        else:
            log.info("epoch %d/%d  loss=%.3f", epoch, hp["epochs"], float(np.mean(losses)))

    # final report with baseline comparison, using best checkpoint
    ckpt = torch.load(SEQ_MODEL_PATH, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    ev = evaluate(model, seqs["val"], proj, bpm, energy, genre_tracks, device)
    for name in ("model", "baseline"):
        e = ev[name]
        log.info(
            "FINAL %-8s r@1=%.3f r@10=%.3f r@50=%.3f r@100=%.3f MRR=%.4f median=%d",
            name,
            e["recall@1"],
            e["recall@10"],
            e["recall@50"],
            e["recall@100"],
            e["MRR"],
            int(e["median_rank"]),
        )
    return ev


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/train_sequence.log")],
    )
    train()
