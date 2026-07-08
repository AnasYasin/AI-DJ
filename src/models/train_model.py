"""
Phase 5 — Contrastive encoder training.

ContrastiveEncoder  MLP(1287→256→128)
Input:   1280-dim discogs-effnet embedding + 7 normalised librosa features:
           bpm_norm (bpm/200), key_sin, key_cos, key_mode (Camelot circular
           encoding), energy_mean, onset_norm (onset/5), lufs_norm
Loss:    NT-Xent (temperature-scaled cross-entropy)
Signal:  consecutive tracks in a DJ mix = positive pairs
Negatives: in-batch (genre-blocked batches, known positives masked)
           + semi-hard negatives mined from ChromaDB (genre-filtered)
Split:   mix-level, from data/processed/split_mixes.csv (15% val per genre)
Output:  128-dim "mixability" embedding on the unit hypersphere
Saved:   models/contrastive_encoder.pt

Logs to MLflow (experiment registry) AND W&B (live dashboard, disabled if not
logged in).

Run:
  python src/models/train_model.py
"""

import logging
from collections import defaultdict
from pathlib import Path
import time

import chromadb
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
import wandb

from src.features.vector_store import (
    CHROMA_PATH,
    COLLECTION_NAME,
    get_client,
    get_collection,
)

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

FEATURES_PATH = Path("data/processed/features.parquet")
TRACKLIST_PATH = Path("data/processed/tracklist_clean.csv")
SPLIT_PATH = Path("data/processed/split_mixes.csv")
MODELS_DIR = Path("models")
ENCODER_PATH = MODELS_DIR / "contrastive_encoder.pt"

# ── Hyperparameters ────────────────────────────────────────────────────────────

HPARAMS = {
    # Contrastive encoder
    "encoder_input_dim": 1287,  # 1280 discogs-effnet + 7 librosa (bpm, key×3, energy, onset, lufs)
    "encoder_hidden_dim": 256,
    "encoder_output_dim": 128,
    "temperature": 0.07,  # NT-Xent temperature (lower = sharper contrast)
    "encoder_lr": 3e-4,
    "encoder_epochs": 50,
    "encoder_batch_size": 128,
    "hard_neg_per_anchor": 5,
    "hard_neg_refresh_epochs": 5,  # re-mine ChromaDB negatives every N epochs
    # Semi-hard band calibrated on the fixed embeddings (2026-07-07):
    # consecutive-pair cosine dist p10 = 0.147, within-genre random p60 = 0.413.
    "hard_neg_min_dist": 0.15,
    "hard_neg_max_dist": 0.40,
    # Shared
    "weight_decay": 1e-4,
    "random_seed": 42,
}

# Camelot wheel positions for key circular encoding (1–12).
_CAMELOT = {
    "C": 8,
    "Cm": 5,
    "C#": 3,
    "C#m": 12,
    "D": 10,
    "Dm": 7,
    "D#": 5,
    "D#m": 2,
    "E": 12,
    "Em": 9,
    "F": 7,
    "Fm": 4,
    "F#": 2,
    "F#m": 11,
    "G": 9,
    "Gm": 6,
    "G#": 4,
    "G#m": 1,
    "A": 11,
    "Am": 8,
    "A#": 6,
    "A#m": 3,
    "B": 1,
    "Bm": 10,
}


# ── Encoder input preparation ──────────────────────────────────────────────────

# bpm_norm(1) + key_sin(1) + key_cos(1) + key_mode(1) + energy_mean(1)
# + onset_norm(1) + lufs_norm(1) = 7
ENCODER_EXTRA_DIM = 7


def _prepare_encoder_inputs(features: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    """
    Build the augmented encoder input matrix: discogs-effnet[1280] + librosa[7] = [1287].

    Key encoding uses circular (sin/cos) Camelot-wheel position so the model
    sees C (position 8) and B (position 1) as close on the wheel, not far apart
    as raw integers would suggest.

    Normalisation:
      bpm       / 200         → [0, 1] for typical 80–180 BPM range
      onset     / 5.0         → [0, 1] for librosa's raw onset scale
      lufs      (x+40)/40     → [0, 1] mapping −40…0 LUFS

    Expensive (~10s for 28k tracks) — call ONCE and pass the result around.

    Returns:
        matrix     (N, 1287) float32 array, row order matches features rows
        tid_to_idx track_id → row index in matrix
    """
    emb = np.stack([np.asarray(e, dtype=np.float32) for e in features["embedding"]])

    bpm_norm = features["bpm"].to_numpy(dtype=np.float32) / 200.0
    keys = features["key"].astype(str)
    camelot = keys.map(_CAMELOT).fillna(1).to_numpy(dtype=np.float32)
    angle = 2 * np.pi * camelot / 12
    key_sin = np.sin(angle).astype(np.float32)
    key_cos = np.cos(angle).astype(np.float32)
    key_mode = (~keys.str.endswith("m")).to_numpy(dtype=np.float32)  # minor=0, major=1
    energy = features["energy_mean"].to_numpy(dtype=np.float32)
    onset_norm = np.minimum(features["onset_strength"].to_numpy(dtype=np.float32) / 5.0, 1.0)
    lufs_norm = np.clip(
        (features["loudness_lufs"].to_numpy(dtype=np.float32) + 40.0) / 40.0, 0.0, 1.0
    )

    extra = np.stack(
        [bpm_norm, key_sin, key_cos, key_mode, energy, onset_norm, lufs_norm], axis=1
    ).astype(np.float32)
    matrix = np.concatenate([emb, extra], axis=1)
    tid_to_idx = {tid: idx for idx, tid in enumerate(features["track_id"])}
    return matrix, tid_to_idx


# ── Pair building ──────────────────────────────────────────────────────────────


def build_consecutive_pairs(
    tracklist_csv: str | Path,
    features: pd.DataFrame,
    split_csv: str | Path = SPLIT_PATH,
) -> tuple[list[tuple], list[tuple]]:
    """
    Build (track_id_A, track_id_B, genre) triples from consecutive tracks in each mix.

    Split is read from split_mixes.csv (mix-level, stratified per genre) so
    train/val is stable across runs and no mix leaks across the boundary.
    """
    df = pd.read_csv(tracklist_csv)
    split = pd.read_csv(split_csv).set_index("mix_id")["split"]
    feat_ids = set(features["track_id"])

    pairs = {"train": [], "val": []}
    for mix_id, group in df.groupby("mix_id"):
        which = split.get(mix_id)
        if which not in ("train", "val"):
            continue
        if "play_type" in group.columns:
            group = group[group["play_type"] == "sequential"]
        if not group["starting_time"].isna().any():
            group = group.sort_values("starting_time")

        genre = group["genre"].iloc[0]
        tids = group["track_id"].tolist()
        pairs[which] += [
            (a, b, genre)
            for a, b in zip(tids, tids[1:])
            if a != b and a in feat_ids and b in feat_ids
        ]

    log.info("Pairs — train: %d, val: %d", len(pairs["train"]), len(pairs["val"]))
    return pairs["train"], pairs["val"]


def build_positive_index(pairs: list[tuple]) -> dict[str, set[str]]:
    """
    Build a lookup: track_id → set of track_ids that it positively pairs with.
    Used to exclude known positives from hard negatives AND to mask false
    negatives inside in-batch similarity.
    """
    index: dict[str, set] = {}
    for a, b, _genre in pairs:
        index.setdefault(a, set()).add(b)
        index.setdefault(b, set()).add(a)
    return index


# ── Dataset ────────────────────────────────────────────────────────────────────


class ContrastiveDataset(Dataset):
    """
    Returns (anchor_emb, positive_emb, hard_neg_embs, tid_a, tid_b).

    Hard negatives are pre-mined from ChromaDB (genre-filtered, semi-hard
    distance band) every `hard_neg_refresh_epochs` epochs.
    """

    def __init__(
        self,
        pairs: list[tuple],
        input_matrix: np.ndarray,
        tid_to_idx: dict[str, int],
        genre_of: dict[str, str],
        positive_index: dict[str, set],
        collection: chromadb.Collection,
        hparams: dict,
        query_matrix: np.ndarray | None = None,
    ):
        self.pairs = pairs
        self.input_matrix = input_matrix
        # raw effnet vectors for ChromaDB queries (index stores raw embeddings;
        # input_matrix may be z-scored and would mismatch)
        self.query_matrix = query_matrix if query_matrix is not None else input_matrix[:, :1280]
        self.tid_to_idx = tid_to_idx
        self.genre_of = genre_of
        self.pos_index = positive_index
        self.collection = collection
        self.n_hard = hparams["hard_neg_per_anchor"]
        self.min_dist = hparams["hard_neg_min_dist"]
        self.max_dist = hparams["hard_neg_max_dist"]
        self.hard_neg_cache: dict[str, np.ndarray] = {}

    _MINE_BATCH = 256  # anchors per ChromaDB query call
    _FETCH_N = 48  # over-fetch: exclusions + distance-band filtering eat candidates

    def mine_hard_negatives(self) -> None:
        """
        Query ChromaDB for genre-filtered semi-hard negatives, batched per genre.
        One query call per _MINE_BATCH anchors — per-anchor calls took ~85ms each
        (~45min for 31k anchors on chromadb 0.5.3); batching brings it to ~1min.
        """
        by_genre: dict[str, list[str]] = defaultdict(list)
        for a, _, g in self.pairs:
            if a in self.tid_to_idx:
                by_genre[g].append(a)
        n_total = sum(len(set(v)) for v in by_genre.values())
        log.info("Mining hard negatives for %d unique anchors...", n_total)
        t0 = time.time()
        found = 0
        enc_dim = self.input_matrix.shape[1]

        for genre, anchors in by_genre.items():
            anchors = list(dict.fromkeys(anchors))
            for start in range(0, len(anchors), self._MINE_BATCH):
                chunk = anchors[start : start + self._MINE_BATCH]
                embs = [self.query_matrix[self.tid_to_idx[t]].tolist() for t in chunk]
                results = self.collection.query(
                    query_embeddings=embs,
                    n_results=min(self._FETCH_N, self.collection.count()),
                    where={"genre": {"$eq": genre}},
                    include=["distances"],
                )
                for tid, ids, dists in zip(chunk, results["ids"], results["distances"]):
                    exclude = {tid} | self.pos_index.get(tid, set())
                    neg_ids = [
                        i
                        for i, d in zip(ids, dists)
                        if i not in exclude
                        and self.min_dist <= d <= self.max_dist
                        and i in self.tid_to_idx
                    ][: self.n_hard]
                    if not neg_ids:
                        continue
                    neg_embs = np.stack([self.input_matrix[self.tid_to_idx[n]] for n in neg_ids])
                    if len(neg_embs) < self.n_hard:
                        pad = np.zeros((self.n_hard - len(neg_embs), enc_dim), dtype=np.float32)
                        neg_embs = np.vstack([neg_embs, pad])
                    self.hard_neg_cache[tid] = neg_embs[: self.n_hard]
                    found += 1
        log.info(
            "Hard negative mining done: %d/%d anchors got negatives (%.1fs)",
            found,
            n_total,
            time.time() - t0,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        tid_a, tid_b, _genre = self.pairs[idx]
        anchor = torch.from_numpy(self.input_matrix[self.tid_to_idx[tid_a]])
        positive = torch.from_numpy(self.input_matrix[self.tid_to_idx[tid_b]])
        enc_dim = self.input_matrix.shape[1]
        hard_negs = torch.from_numpy(
            self.hard_neg_cache.get(tid_a, np.zeros((self.n_hard, enc_dim), dtype=np.float32))
        )
        return anchor, positive, hard_negs, tid_a, tid_b


class GenreBatchSampler(Sampler[list[int]]):
    """
    Yields batches whose pairs all share one genre, so in-batch negatives are
    within-genre (hard) instead of cross-genre (trivial). Batch order is
    shuffled across genres each epoch.
    """

    def __init__(self, pair_genres: list[str], batch_size: int, seed: int = 0):
        self.by_genre: dict[str, list[int]] = defaultdict(list)
        for i, g in enumerate(pair_genres):
            self.by_genre[g].append(i)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for idxs in self.by_genre.values():
            idxs = rng.permutation(idxs)
            for k in range(0, len(idxs), self.batch_size):
                b = idxs[k : k + self.batch_size]
                if len(b) >= 2:  # BatchNorm needs ≥2 rows
                    batches.append([int(i) for i in b])
        order = rng.permutation(len(batches))
        for i in order:
            yield batches[i]

    def __len__(self) -> int:
        return sum(
            (len(v) + self.batch_size - 1) // self.batch_size for v in self.by_genre.values()
        )


def build_false_negative_mask(
    tids_a: list[str], tids_b: list[str], pos_index: dict[str, set]
) -> torch.Tensor:
    """
    (B, B) bool mask, True where anchor j must NOT be used as a negative for
    anchor i: same track, or a known positive of i. Consecutive pairs overlap
    ((t1,t2) and (t2,t3) coexist in a batch), so without this mask the loss
    pushes apart genuine positive pairs.
    """
    B = len(tids_a)
    mask = torch.zeros(B, B, dtype=torch.bool)
    for i in range(B):
        pos = pos_index.get(tids_a[i], set())
        for j in range(B):
            if i == j:
                continue
            if tids_a[j] == tids_a[i] or tids_a[j] == tids_b[i] or tids_a[j] in pos:
                mask[i, j] = True
    return mask


# ── Models ─────────────────────────────────────────────────────────────────────


class ContrastiveEncoder(nn.Module):
    """
    Projection head: 1287-dim inputs → 128-dim unit hypersphere.

    Two branches so the 7 scalar features (BPM/key/energy — individually as
    predictive as the whole raw embedding) are not drowned by the 1280 effnet
    dims: effnet → hidden_dim, scalars → 64, fused → output.

    Inputs are expected z-scored (see train(): stats saved to encoder_norm.npz —
    inference MUST apply the same normalisation).
    discogs-effnet backbone stays frozen. Only this head is trained.
    """

    SCALAR_DIM = ENCODER_EXTRA_DIM

    def __init__(self, input_dim: int = 1287, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        effnet_dim = input_dim - self.SCALAR_DIM
        self.effnet_branch = nn.Sequential(
            nn.Linear(effnet_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.scalar_branch = nn.Sequential(
            nn.Linear(self.SCALAR_DIM, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Linear(hidden_dim + 64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eff = self.effnet_branch(x[:, : -self.SCALAR_DIM])
        sca = self.scalar_branch(x[:, -self.SCALAR_DIM :])
        return F.normalize(self.fuse(torch.cat([eff, sca], dim=1)), dim=1)


# ── NT-Xent Loss ───────────────────────────────────────────────────────────────


class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy (SimCLR loss) with hard negatives.

    For a batch of B (anchor, positive) pairs, the loss for anchor i is:
      -log( exp(sim(a_i, p_i) / τ) /
            [exp(sim(a_i, p_i) / τ)
             + Σ_{j≠i, not masked} exp(sim(a_i, a_j) / τ)   ← in-batch negatives
             + Σ_k valid           exp(sim(a_i, h_ik) / τ)] ) ← hard negatives

    neg_mask marks in-batch entries that are known positives (false negatives).
    hard_valid marks zero-padded hard-negative rows to exclude.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchors: torch.Tensor,  # (B, D) — L2 normalised
        positives: torch.Tensor,  # (B, D) — L2 normalised
        hard_negs: torch.Tensor | None = None,  # (B, K, D) — L2 normalised
        neg_mask: torch.Tensor | None = None,  # (B, B) bool, True = exclude
        hard_valid: torch.Tensor | None = None,  # (B, K) bool, True = keep
    ) -> torch.Tensor:
        B = anchors.size(0)

        pos_sim = (anchors * positives).sum(dim=1) / self.temperature

        batch_sim = torch.mm(anchors, anchors.T) / self.temperature
        mask = torch.eye(B, device=anchors.device).bool()
        if neg_mask is not None:
            mask = mask | neg_mask.to(anchors.device)
        batch_sim = batch_sim.masked_fill(mask, float("-inf"))

        logits = torch.cat([pos_sim.unsqueeze(1), batch_sim], dim=1)  # (B, 1+B)

        if hard_negs is not None and hard_negs.size(1) > 0:
            hard_sim = torch.bmm(anchors.unsqueeze(1), hard_negs.transpose(1, 2))
            hard_sim = hard_sim.squeeze(1) / self.temperature  # (B, K)
            if hard_valid is not None:
                hard_sim = hard_sim.masked_fill(~hard_valid.to(anchors.device), float("-inf"))
            logits = torch.cat([logits, hard_sim], dim=1)  # (B, 1+B+K)

        targets = torch.zeros(B, dtype=torch.long, device=anchors.device)
        return F.cross_entropy(logits, targets)


# ── Metrics ────────────────────────────────────────────────────────────────────


def compute_contrastive_metrics(
    encoder: ContrastiveEncoder,
    val_pairs: list[tuple],
    input_matrix: np.ndarray,
    tid_to_idx: dict[str, int],
    device: torch.device,
) -> dict:
    """
    Evaluate on validation pairs (fully vectorised).

    Metrics:
      alignment       -mean||z_a - z_p||²  (higher = closer positives)
      uniformity      log mean exp(-2||z_i - z_j||²) (lower = more spread)
      top1/5/10_retrieval  % where positive is in top-K nearest neighbours
      mean_pos_sim    mean cosine similarity of positive pairs
      mean_neg_sim    mean cosine similarity of random non-pair tracks
      emb_variance    mean per-dim variance (collapse early warning)
    """
    encoder.eval()
    val_ids = list({tid for a, b, _ in val_pairs for tid in (a, b) if tid in tid_to_idx})
    if not val_ids:
        return {}
    id2i = {t: i for i, t in enumerate(val_ids)}

    inputs = torch.from_numpy(
        np.stack([input_matrix[tid_to_idx[t]] for t in val_ids])
    ).to(device)
    with torch.no_grad():
        proj = encoder(inputs)  # (N, 128)

    vp = [(a, b) for a, b, _ in val_pairs if a in id2i and b in id2i]
    if not vp:
        return {}
    ai = torch.tensor([id2i[a] for a, _ in vp], device=device)
    bi = torch.tensor([id2i[b] for _, b in vp], device=device)
    za, zp = proj[ai], proj[bi]

    alignment = -((za - zp) ** 2).sum(dim=1).mean().item()
    sample = proj[: min(1000, len(proj))]
    sq_dists = torch.cdist(sample, sample, p=2) ** 2
    uniformity = torch.log(torch.exp(-2 * sq_dists).mean()).item()

    pos_sim = (za * zp).sum(dim=1)  # (P,)
    P = len(vp)
    ranks = torch.empty(P, device=device)
    neg_sims = torch.empty(P, device=device)
    rand_j = torch.randint(0, len(val_ids), (P,), device=device)
    for start in range(0, P, 2048):  # chunk to bound memory
        end = min(start + 2048, P)
        sims = za[start:end] @ proj.T  # (chunk, N)
        sims[torch.arange(end - start, device=device), ai[start:end]] = -2.0  # exclude self
        ranks[start:end] = (sims > pos_sim[start:end].unsqueeze(1)).sum(dim=1) + 1
        neg_sims[start:end] = sims[torch.arange(end - start, device=device), rand_j[start:end]]

    return {
        "val/alignment": alignment,
        "val/uniformity": uniformity,
        "val/top1_retrieval": (ranks <= 1).float().mean().item(),
        "val/top5_retrieval": (ranks <= 5).float().mean().item(),
        "val/top10_retrieval": (ranks <= 10).float().mean().item(),
        "val/mean_pos_sim": pos_sim.mean().item(),
        "val/mean_neg_sim": neg_sims.mean().item(),
        "val/emb_variance": proj.var(dim=0).mean().item(),
    }


# ── Training loop ──────────────────────────────────────────────────────────────


def train_contrastive(
    encoder: ContrastiveEncoder,
    criterion: NTXentLoss,
    train_ds: ContrastiveDataset,
    val_pairs: list[tuple],
    input_matrix: np.ndarray,
    tid_to_idx: dict[str, int],
    device: torch.device,
    hparams: dict,
) -> ContrastiveEncoder:
    """
    Each epoch:
      1. Refresh hard negatives every hard_neg_refresh_epochs epochs
      2. Genre-blocked batches; known positives masked out of in-batch negatives
      3. NT-Xent loss with in-batch + hard negatives
      4. Eval + save best model by val/top1_retrieval
    """
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=hparams["encoder_lr"],
        weight_decay=hparams["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hparams["encoder_epochs"]
    )

    sampler = GenreBatchSampler(
        [g for _, _, g in train_ds.pairs],
        hparams["encoder_batch_size"],
        seed=hparams["random_seed"],
    )
    loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    best_top1 = 0.0
    MODELS_DIR.mkdir(exist_ok=True)

    for epoch in range(1, hparams["encoder_epochs"] + 1):
        if (epoch - 1) % hparams["hard_neg_refresh_epochs"] == 0:
            train_ds.mine_hard_negatives()
        sampler.set_epoch(epoch)

        encoder.train()
        epoch_loss = 0.0
        grad_norms = []

        for anchors, positives, hard_negs, tids_a, tids_b in loader:
            anchors = anchors.to(device)
            positives = positives.to(device)
            hard_valid = hard_negs.abs().sum(dim=2) > 0  # (B, K) — zero rows are padding
            hard_negs = hard_negs.to(device)
            neg_mask = build_false_negative_mask(list(tids_a), list(tids_b), train_ds.pos_index)

            z_a = encoder(anchors)
            z_p = encoder(positives)
            B, K, D = hard_negs.shape
            z_h = encoder(hard_negs.view(B * K, D)).view(B, K, -1)

            loss = criterion(z_a, z_p, z_h, neg_mask=neg_mask, hard_valid=hard_valid)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            grad_norms.append(grad_norm.item())
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                }
            )

        scheduler.step()

        avg_loss = epoch_loss / max(len(loader), 1)
        avg_grad = float(np.mean(grad_norms))

        mlflow.log_metric("train/epoch_loss", avg_loss, step=epoch)
        mlflow.log_metric("train/grad_norm", avg_grad, step=epoch)

        val_metrics = compute_contrastive_metrics(
            encoder, val_pairs, input_matrix, tid_to_idx, device
        )
        for k, v in val_metrics.items():
            mlflow.log_metric(k, v, step=epoch)
        wandb.log({"epoch": epoch, **val_metrics})

        top1 = val_metrics.get("val/top1_retrieval", 0.0)
        log.info(
            "[Encoder] epoch %d/%d  loss=%.4f  top1=%.3f  top10=%.3f  alignment=%.4f  uniformity=%.4f",
            epoch,
            hparams["encoder_epochs"],
            avg_loss,
            top1,
            val_metrics.get("val/top10_retrieval", 0),
            val_metrics.get("val/alignment", 0),
            val_metrics.get("val/uniformity", 0),
        )

        if top1 > best_top1:
            best_top1 = top1
            torch.save(encoder.state_dict(), ENCODER_PATH)
            log.info("  ↑ New best top-1 retrieval: %.3f → saved %s", best_top1, ENCODER_PATH)
            mlflow.pytorch.log_model(encoder, artifact_path="encoder_best")

    log.info("Encoder training done. Best top-1: %.3f", best_top1)
    return encoder


# ── Main entry point ───────────────────────────────────────────────────────────


def train(
    tracklist_csv: str | Path = TRACKLIST_PATH,
    hparams: dict | None = None,
) -> None:
    """
    Full training pipeline: contrastive encoder.

    mlflow: logs to ./mlruns/ — view with `mlflow ui --port 5000`
    wandb:  logs to wandb.ai — run `wandb login` once; falls back to disabled.
    """
    if hparams is None:
        hparams = HPARAMS

    torch.manual_seed(hparams["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    features = pd.read_parquet(FEATURES_PATH).drop_duplicates("track_id").reset_index(drop=True)
    log.info("Features loaded: %d tracks, %d columns", len(features), len(features.columns))

    log.info("Preparing encoder inputs (once)...")
    input_matrix, tid_to_idx = _prepare_encoder_inputs(features)

    # ChromaDB stores raw effnet vectors — mining queries must stay in that space
    raw_effnet = input_matrix[:, :1280].copy()

    # z-score all input dims — otherwise the 1280 effnet dims (varied scale)
    # drown the 7 [0,1] scalars. Stats saved for inference-time normalisation.
    norm_mu = input_matrix.mean(axis=0)
    norm_sd = input_matrix.std(axis=0) + 1e-6
    input_matrix = (input_matrix - norm_mu) / norm_sd
    MODELS_DIR.mkdir(exist_ok=True)
    np.savez(MODELS_DIR / "encoder_norm.npz", mu=norm_mu, sd=norm_sd)
    log.info("Input z-scoring applied; stats saved to %s", MODELS_DIR / "encoder_norm.npz")

    train_pairs, val_pairs = build_consecutive_pairs(tracklist_csv, features)
    positive_index = build_positive_index(train_pairs + val_pairs)

    genre_of = (
        pd.read_csv(tracklist_csv, usecols=["track_id", "genre"])
        .drop_duplicates("track_id")
        .set_index("track_id")["genre"]
        .to_dict()
    )

    client = get_client(CHROMA_PATH)
    collection = get_collection(client)
    n_indexed = collection.count()
    log.info("ChromaDB collection '%s': %d tracks", COLLECTION_NAME, n_indexed)
    if n_indexed < len(features) * 0.9:
        raise RuntimeError(
            f"ChromaDB has {n_indexed} tracks but features has {len(features)} — "
            "rebuild the index first: rm -rf data/processed/chromadb && "
            "python src/features/vector_store.py"
        )

    mlflow.set_experiment("ai-dj-training")
    try:
        wandb.init(
            project="ai-dj",
            name=f"run-{int(time.time())}",
            config=hparams,
            tags=["contrastive"],
        )
    except Exception as e:  # not logged in / offline — don't block training
        log.warning("wandb init failed (%s) — continuing with wandb disabled", e)
        wandb.init(mode="disabled")

    with mlflow.start_run():
        mlflow.log_params(hparams)

        log.info("=== Training contrastive encoder ===")
        encoder = ContrastiveEncoder(
            input_dim=hparams["encoder_input_dim"],
            hidden_dim=hparams["encoder_hidden_dim"],
            output_dim=hparams["encoder_output_dim"],
        ).to(device)

        criterion = NTXentLoss(temperature=hparams["temperature"])

        train_ds = ContrastiveDataset(
            pairs=train_pairs,
            input_matrix=input_matrix,
            tid_to_idx=tid_to_idx,
            genre_of=genre_of,
            positive_index=positive_index,
            collection=collection,
            hparams=hparams,
            query_matrix=raw_effnet,
        )

        wandb.watch(encoder, log="gradients", log_freq=100)
        encoder = train_contrastive(
            encoder, criterion, train_ds, val_pairs, input_matrix, tid_to_idx, device, hparams
        )

        # Project all embeddings and persist to features.parquet.
        # Use the BEST checkpoint, not the final epoch — long runs overfit
        # (150-epoch run: final model val AUC 0.655 vs best-epoch 0.667).
        log.info("Projecting all embeddings with best checkpoint...")
        encoder.load_state_dict(torch.load(ENCODER_PATH, weights_only=True))
        encoder.eval()
        all_raw = torch.from_numpy(input_matrix).to(device)
        with torch.no_grad():
            all_proj = encoder(all_raw).cpu().numpy()  # (N, 128)

        features = features.assign(embedding_proj=[row.tolist() for row in all_proj])
        features.to_parquet(FEATURES_PATH, index=False)
        log.info("Projected embeddings written to %s", FEATURES_PATH)

        mlflow.log_artifact(str(ENCODER_PATH))

    wandb.finish()
    log.info("Training complete. Models saved to %s/", MODELS_DIR)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _fmt = "%(asctime)s %(levelname)s %(message)s"
    _log_path = Path("logs/train_model.log")
    _log_path.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=_fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(_log_path, encoding="utf-8"),
        ],
    )
    log.info("Logging to %s", _log_path)
    train()
