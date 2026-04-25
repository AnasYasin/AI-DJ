"""
Phase 5 — Contrastive encoder training.

ContrastiveEncoder  MLP(1287→256→128)
Input:   1280-dim discogs-effnet embedding + 7 normalised librosa features:
           bpm_norm (bpm/200), key_sin, key_cos, key_mode (Camelot circular
           encoding), energy_mean, onset_norm (onset/5), lufs_norm
Loss:    NT-Xent (temperature-scaled cross-entropy)
Signal:  consecutive tracks in a DJ mix = positive pairs
Negatives: in-batch + semi-hard negatives mined from ChromaDB
           (ChromaDB queries use raw 1280-dim discogs-effnet embeddings)
Output:  128-dim "mixability" embedding on the unit hypersphere
Saved:   models/contrastive_encoder.pt

Logs to MLflow (experiment registry) AND W&B (live dashboard).

MLflow setup:
  mlflow ui --port 5000          → http://localhost:5000
  Or: docker-compose up mlflow

W&B setup (one-time):
  wandb login                    → paste API key from wandb.ai/settings

Run:
  conda activate djtest
  python src/models/train_model.py
"""

import hashlib
import logging
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
from torch.utils.data import DataLoader, Dataset
import wandb

from src.features.vector_store import (
    CHROMA_PATH,
    COLLECTION_NAME,
    HARD_NEG_MAX_DISTANCE,
    HARD_NEG_MIN_DISTANCE,
    get_client,
    get_collection,
    query_hard_negatives,
)

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

FEATURES_PATH = Path("data/processed/features.parquet")
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
    "hard_neg_min_dist": HARD_NEG_MIN_DISTANCE,
    "hard_neg_max_dist": HARD_NEG_MAX_DISTANCE,
    # Shared
    "weight_decay": 1e-4,
    "val_split": 0.2,
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

    Returns:
        matrix     (N, 1287) float32 array, row order matches features rows
        tid_to_idx track_id → row index in matrix
    """
    n = len(features)
    matrix = np.zeros((n, 1280 + ENCODER_EXTRA_DIM), dtype=np.float32)
    tid_to_idx: dict[str, int] = {}

    for idx, (_, row) in enumerate(features.iterrows()):
        tid = row["track_id"]
        tid_to_idx[tid] = idx

        emb = np.asarray(row["embedding"], dtype=np.float32)

        bpm_norm = float(row["bpm"]) / 200.0
        camelot = _CAMELOT.get(str(row["key"]), 1)
        angle = 2 * np.pi * camelot / 12
        key_sin = float(np.sin(angle))
        key_cos = float(np.cos(angle))
        key_mode = 0.0 if str(row["key"]).endswith("m") else 1.0  # minor=0, major=1
        energy = float(row["energy_mean"])
        onset_norm = min(float(row["onset_strength"]) / 5.0, 1.0)
        lufs_norm = float(np.clip((float(row["loudness_lufs"]) + 40.0) / 40.0, 0.0, 1.0))

        extra = np.array(
            [bpm_norm, key_sin, key_cos, key_mode, energy, onset_norm, lufs_norm], dtype=np.float32
        )
        matrix[idx] = np.concatenate([emb, extra])

    return matrix, tid_to_idx


# ── Pair building ──────────────────────────────────────────────────────────────


def _track_id(artist: str, track: str) -> str:
    """Same deterministic ID as preview_fetcher.py."""
    return hashlib.md5(f"{artist}|{track}".lower().encode()).hexdigest()[:12]


def build_consecutive_pairs(
    mix_csvs: list[Path],
    features: pd.DataFrame,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple], list[tuple]]:
    """
    Read mix CSVs (each has mix_id, artist_name, track_name, starting_time).
    Return (train_pairs, val_pairs) where each pair = (track_id_A, track_id_B).

    Splitting is done at the MIX level (not pair level) to prevent data leakage —
    no mix appears in both train and val.
    """
    all_pairs: dict[str, list[tuple]] = {}
    feat_ids = set(features["track_id"])

    for csv_path in mix_csvs:
        df = pd.read_csv(csv_path)

        for mix_id, group in df.groupby("mix_id"):
            if "play_type" in group.columns:
                group = group[group["play_type"] == "sequential"]

            if group["starting_time"].isna().any():
                group = group.reset_index(drop=True)
            else:
                group = group.sort_values("starting_time").reset_index(drop=True)

            all_tids = [_track_id(r["artist_name"], r["track_name"]) for _, r in group.iterrows()]

            pairs = [
                (all_tids[i], all_tids[i + 1])
                for i in range(len(all_tids) - 1)
                if all_tids[i] in feat_ids and all_tids[i + 1] in feat_ids
            ]
            if pairs:
                all_pairs[str(mix_id)] = pairs

    mix_ids = list(all_pairs.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(mix_ids)
    split = int(len(mix_ids) * (1 - val_split))
    train_mixes, val_mixes = mix_ids[:split], mix_ids[split:]

    train_pairs = [p for mid in train_mixes for p in all_pairs[mid]]
    val_pairs = [p for mid in val_mixes for p in all_pairs[mid]]

    log.info(
        "Pairs — train: %d (%d mixes), val: %d (%d mixes)",
        len(train_pairs),
        len(train_mixes),
        len(val_pairs),
        len(val_mixes),
    )
    return train_pairs, val_pairs


def build_positive_index(pairs: list[tuple]) -> dict[str, set[str]]:
    """
    Build a lookup: track_id → set of track_ids that it positively pairs with.
    Used in hard negative mining to exclude all known positives.
    """
    index: dict[str, set] = {}
    for a, b in pairs:
        index.setdefault(a, set()).add(b)
        index.setdefault(b, set()).add(a)
    return index


# ── Dataset ────────────────────────────────────────────────────────────────────


class ContrastiveDataset(Dataset):
    """
    Returns (anchor_emb, positive_emb, hard_neg_embs) tensors.

    Hard negatives are pre-mined at the start of each epoch by querying ChromaDB.
    Pre-mining is faster than querying per-batch and produces stable negatives
    within an epoch.
    """

    def __init__(
        self,
        pairs: list[tuple],
        features: pd.DataFrame,
        positive_index: dict[str, set],
        collection: chromadb.Collection,
        hard_neg_per_anchor: int = 5,
    ):
        self.pairs = pairs
        self.pos_index = positive_index
        self.collection = collection
        self.n_hard = hard_neg_per_anchor

        # Augmented inputs for encoder training (discogs-effnet + librosa features)
        self.input_matrix, self.tid_to_idx = _prepare_encoder_inputs(features)
        # Raw 1280-dim discogs-effnet embeddings for ChromaDB ANN queries
        raw = features.set_index("track_id")["embedding"]
        self.raw_emb: dict[str, np.ndarray] = {
            tid: np.asarray(raw.loc[tid], dtype=np.float32) for tid in raw.index
        }
        self.hard_neg_cache: dict[str, np.ndarray] = {}

    def mine_hard_negatives(self) -> None:
        """
        Query ChromaDB for hard negatives for every unique anchor.
        Call this at the start of each epoch to refresh the cache.
        """
        unique_anchors = set(a for a, _ in self.pairs)
        log.info("Mining hard negatives for %d unique anchors...", len(unique_anchors))
        t0 = time.time()
        found = 0
        enc_dim = self.input_matrix.shape[1]
        for tid in unique_anchors:
            if tid not in self.raw_emb:
                continue
            exclude = {tid} | self.pos_index.get(tid, set())
            results = query_hard_negatives(
                self.collection,
                self.raw_emb[tid],
                n_results=self.n_hard,
                exclude_ids=list(exclude),
            )
            neg_ids = results["ids"][0]
            if neg_ids:
                neg_embs = np.stack(
                    [
                        self.input_matrix[self.tid_to_idx[nid]]
                        for nid in neg_ids
                        if nid in self.tid_to_idx
                    ]
                )
                if len(neg_embs) < self.n_hard:
                    pad = np.zeros((self.n_hard - len(neg_embs), enc_dim), dtype=np.float32)
                    neg_embs = np.vstack([neg_embs, pad])
                self.hard_neg_cache[tid] = neg_embs[: self.n_hard]
                found += 1
        log.info(
            "Hard negative mining done: %d/%d anchors got negatives (%.1fs)",
            found,
            len(unique_anchors),
            time.time() - t0,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        tid_a, tid_b = self.pairs[idx]
        if tid_a not in self.tid_to_idx or tid_b not in self.tid_to_idx:
            enc_dim = self.input_matrix.shape[1]
            z = torch.zeros(enc_dim)
            return z, z, torch.zeros(self.n_hard, enc_dim)

        anchor = torch.tensor(self.input_matrix[self.tid_to_idx[tid_a]])
        positive = torch.tensor(self.input_matrix[self.tid_to_idx[tid_b]])

        enc_dim = self.input_matrix.shape[1]
        hard_neg_arr = self.hard_neg_cache.get(
            tid_a, np.zeros((self.n_hard, enc_dim), dtype=np.float32)
        )
        hard_negs = torch.tensor(hard_neg_arr)

        return anchor, positive, hard_negs


# ── Models ─────────────────────────────────────────────────────────────────────


class ContrastiveEncoder(nn.Module):
    """
    Projection head: 1280-dim discogs-effnet embeddings → 128-dim unit hypersphere.

    Architecture: Linear → BN → ReLU → Linear → L2-normalise
    BatchNorm before ReLU stabilises training when input scale varies across tracks.
    L2 normalisation forces all embeddings onto the unit sphere — required for
    NT-Xent loss (cosine similarity = dot product on unit sphere).

    discogs-effnet backbone stays frozen. Only this head is trained (~400K params).
    """

    def __init__(self, input_dim: int = 1280, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)  # → unit hypersphere


# ── NT-Xent Loss ───────────────────────────────────────────────────────────────


class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy (SimCLR loss) with hard negatives.

    For a batch of B (anchor, positive) pairs, the loss for anchor i is:
      -log( exp(sim(a_i, p_i) / τ) /
            [exp(sim(a_i, p_i) / τ)
             + Σ_{j≠i} exp(sim(a_i, a_j) / τ)     ← in-batch negatives
             + Σ_k     exp(sim(a_i, h_ik) / τ)] )  ← hard negatives

    All vectors must be L2-normalised so sim = dot product.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchors: torch.Tensor,  # (B, D) — L2 normalised
        positives: torch.Tensor,  # (B, D) — L2 normalised
        hard_negs: torch.Tensor | None = None,  # (B, K, D) — L2 normalised
    ) -> torch.Tensor:
        B = anchors.size(0)

        pos_sim = (anchors * positives).sum(dim=1) / self.temperature

        batch_sim = torch.mm(anchors, anchors.T) / self.temperature
        mask = torch.eye(B, device=anchors.device).bool()
        batch_sim = batch_sim.masked_fill(mask, float("-inf"))

        logits = torch.cat([pos_sim.unsqueeze(1), batch_sim], dim=1)  # (B, 1+B)

        if hard_negs is not None and hard_negs.size(1) > 0:
            hard_negs = F.normalize(hard_negs, dim=2)
            hard_sim = torch.bmm(anchors.unsqueeze(1), hard_negs.transpose(1, 2))
            hard_sim = hard_sim.squeeze(1) / self.temperature  # (B, K)
            logits = torch.cat([logits, hard_sim], dim=1)  # (B, 1+B+K)

        targets = torch.zeros(B, dtype=torch.long, device=anchors.device)
        return F.cross_entropy(logits, targets)


# ── Metrics ────────────────────────────────────────────────────────────────────


def compute_contrastive_metrics(
    encoder: ContrastiveEncoder,
    val_pairs: list[tuple],
    features: pd.DataFrame,
    device: torch.device,
) -> dict:
    """
    Evaluate contrastive encoder quality on validation pairs.

    Metrics:
      alignment       -mean||z_a - z_p||²  (higher = closer positives)
      uniformity      log mean exp(-2||z_i - z_j||²) (lower = more spread)
      top1/5/10_retrieval  % where positive is in top-K nearest neighbours
      mean_pos_sim    mean cosine similarity of positive pairs
      mean_neg_sim    mean cosine similarity of random non-pair tracks
      emb_variance    mean per-dim variance (collapse early warning)
    """
    encoder.eval()
    input_matrix, tid_to_idx = _prepare_encoder_inputs(features)

    val_ids = list({tid for pair in val_pairs for tid in pair if tid in tid_to_idx})
    if not val_ids:
        return {}

    val_inputs = np.stack([input_matrix[tid_to_idx[tid]] for tid in val_ids])
    raw_embs = torch.tensor(val_inputs, dtype=torch.float32).to(device)
    with torch.no_grad():
        proj = encoder(raw_embs)  # (N, 128)

    proj_idx = {tid: proj[i] for i, tid in enumerate(val_ids)}

    valid_pairs = [(a, b) for a, b in val_pairs if a in proj_idx and b in proj_idx]
    if not valid_pairs:
        return {}

    za = torch.stack([proj_idx[a] for a, _ in valid_pairs])
    zp = torch.stack([proj_idx[b] for _, b in valid_pairs])
    alignment = -((za - zp) ** 2).sum(dim=1).mean().item()

    sample = proj[: min(1000, len(proj))]
    sq_dists = torch.cdist(sample, sample, p=2) ** 2
    uniformity = torch.log(torch.exp(-2 * sq_dists).mean()).item()

    all_proj = torch.stack([proj_idx[tid] for tid in val_ids])
    top1 = top5 = top10 = 0
    pos_sims = []
    neg_sims_list = []

    for tid_a, tid_b in valid_pairs:
        if tid_a not in proj_idx or tid_b not in proj_idx:
            continue
        q = proj_idx[tid_a]
        sims = torch.mv(all_proj, q)
        self_idx = val_ids.index(tid_a)
        sims[self_idx] = -2.0
        ranked = sims.argsort(descending=True).tolist()
        pos_rank = val_ids.index(tid_b)
        pos_sims.append(sims[pos_rank].item())
        rank = ranked.index(pos_rank) + 1
        if rank <= 1:
            top1 += 1
        if rank <= 5:
            top5 += 1
        if rank <= 10:
            top10 += 1
        rand_idx = np.random.randint(0, len(val_ids))
        if val_ids[rand_idx] not in {tid_a, tid_b}:
            neg_sims_list.append(sims[rand_idx].item())

    n = len(valid_pairs)
    return {
        "val/alignment": alignment,
        "val/uniformity": uniformity,
        "val/top1_retrieval": top1 / n,
        "val/top5_retrieval": top5 / n,
        "val/top10_retrieval": top10 / n,
        "val/mean_pos_sim": float(np.mean(pos_sims)),
        "val/mean_neg_sim": float(np.mean(neg_sims_list)) if neg_sims_list else 0.0,
        "val/emb_variance": proj.var(dim=0).mean().item(),
    }


# ── Training loop ──────────────────────────────────────────────────────────────


def train_contrastive(
    encoder: ContrastiveEncoder,
    criterion: NTXentLoss,
    train_ds: ContrastiveDataset,
    val_pairs: list[tuple],
    features: pd.DataFrame,
    device: torch.device,
    hparams: dict,
) -> ContrastiveEncoder:
    """
    Train the contrastive encoder for encoder_epochs epochs.

    Each epoch:
      1. Mine hard negatives from ChromaDB (refreshes the cache)
      2. Forward pass: encode anchor + positive + hard negatives
      3. NT-Xent loss with in-batch + hard negatives
      4. Log train loss to MLflow + W&B every batch
      5. Evaluate val metrics every epoch
      6. Save best model by val/top1_retrieval
    """
    optimizer = torch.optim.AdamW(
        encoder.parameters(),
        lr=hparams["encoder_lr"],
        weight_decay=hparams["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hparams["encoder_epochs"]
    )

    loader = DataLoader(
        train_ds,
        batch_size=hparams["encoder_batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    best_top1 = 0.0
    MODELS_DIR.mkdir(exist_ok=True)

    for epoch in range(1, hparams["encoder_epochs"] + 1):
        train_ds.mine_hard_negatives()

        encoder.train()
        epoch_loss = 0.0
        grad_norms = []

        for _batch_idx, (anchors, positives, hard_negs) in enumerate(loader):
            anchors = anchors.to(device)
            positives = positives.to(device)
            hard_negs = hard_negs.to(device)

            z_a = encoder(anchors)
            z_p = encoder(positives)
            B, K, D = hard_negs.shape
            z_h = encoder(hard_negs.view(B * K, D)).view(B, K, -1)

            loss = criterion(z_a, z_p, z_h)

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

        val_metrics = compute_contrastive_metrics(encoder, val_pairs, features, device)
        for k, v in val_metrics.items():
            mlflow.log_metric(k, v, step=epoch)
        wandb.log({"epoch": epoch, **val_metrics})

        top1 = val_metrics.get("val/top1_retrieval", 0.0)
        log.info(
            "[Encoder] epoch %d/%d  loss=%.4f  top1=%.3f  alignment=%.4f  uniformity=%.4f",
            epoch,
            hparams["encoder_epochs"],
            avg_loss,
            top1,
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
    mix_csvs: list[str | Path] | None = None,
    hparams: dict | None = None,
) -> None:
    """
    Full training pipeline: contrastive encoder.

    mlflow: logs to ./mlruns/ — view with `mlflow ui --port 5000`
    wandb:  logs to wandb.ai  — run `wandb login` once before calling this

    mix_csvs: list of CSV paths with mix tracklists. Defaults to romanFlugel.csv.
              Each CSV must have columns: mix_id, artist_name, track_name, starting_time.
    """
    if mix_csvs is None:
        mix_csvs = [Path("data/interim/romanFlugel.csv")]
    if hparams is None:
        hparams = HPARAMS

    torch.manual_seed(hparams["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    features = pd.read_parquet(FEATURES_PATH)
    log.info("Features loaded: %d tracks, %d columns", len(features), len(features.columns))

    train_pairs, val_pairs = build_consecutive_pairs(
        [Path(p) for p in mix_csvs],
        features,
        val_split=hparams["val_split"],
        seed=hparams["random_seed"],
    )
    positive_index = build_positive_index(train_pairs + val_pairs)

    client = get_client(CHROMA_PATH)
    collection = get_collection(client)
    log.info("ChromaDB collection '%s': %d tracks", COLLECTION_NAME, collection.count())

    mlflow.set_experiment("ai-dj-training")
    wandb.init(
        project="ai-dj",
        name=f"run-{int(time.time())}",
        config=hparams,
        tags=["contrastive"],
    )

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
            features=features,
            positive_index=positive_index,
            collection=collection,
            hard_neg_per_anchor=hparams["hard_neg_per_anchor"],
        )

        wandb.watch(encoder, log="gradients", log_freq=100)
        encoder = train_contrastive(
            encoder, criterion, train_ds, val_pairs, features, device, hparams
        )

        # Project all embeddings and persist to features.parquet
        log.info("Projecting all embeddings with trained encoder...")
        encoder.eval()
        input_matrix, _ = _prepare_encoder_inputs(features)
        all_raw = torch.tensor(input_matrix, dtype=torch.float32).to(device)
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
