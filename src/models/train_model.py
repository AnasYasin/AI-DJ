"""
Phase 5 — Model training: contrastive encoder + transition classifier.

Two models trained sequentially:

  1. ContrastiveEncoder  MLP(775→256→128)
     Input:   768-dim MERT embedding + 7 normalised librosa features:
                bpm_norm (bpm/200), key_sin, key_cos, key_mode (Camelot circular
                encoding), energy_mean, onset_norm (onset/5), lufs_norm
     Loss:    NT-Xent (temperature-scaled cross-entropy)
     Signal:  consecutive tracks in a DJ mix = positive pairs
     Negatives: in-batch + semi-hard negatives mined from ChromaDB
                (ChromaDB queries still use raw 768-dim MERT only)
     Output:  128-dim "mixability" embedding on the unit hypersphere
     Saved:   models/contrastive_encoder.pt

  2. TransitionClassifier  MLP(260→128→64→6)
     Input:   concat(emb_A[128], emb_B[128], bpm_ratio, energy_delta,
                     harmonic_dist, time_gap)  → 260 dims
     Classes: slam / melt / blend / rise / fade / wave
     Labels:  data/processed/transition_labels.csv  (from Phase 4)
     Saved:   models/transition_classifier.pt

Both models log to MLflow (experiment registry) AND W&B (live dashboard).

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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

FEATURES_PATH    = Path("data/processed/features.parquet")
LABELS_PATH      = Path("data/processed/transition_labels.csv")
MODELS_DIR       = Path("models")
ENCODER_PATH     = MODELS_DIR / "contrastive_encoder.pt"
CLASSIFIER_PATH  = MODELS_DIR / "transition_classifier.pt"

# ── Hyperparameters ────────────────────────────────────────────────────────────
# Centralised here so MLflow/W&B can log them all at once.

HPARAMS = {
    # Contrastive encoder
    "encoder_input_dim":   775,   # 768 MERT + 7 librosa (bpm, key×3, energy, onset, lufs)
    "encoder_hidden_dim":  256,
    "encoder_output_dim":  128,
    "temperature":         0.07,    # NT-Xent temperature (lower = sharper contrast)
    "encoder_lr":          3e-4,
    "encoder_epochs":      50,
    "encoder_batch_size":  128,
    "hard_neg_per_anchor": 5,       # how many hard negatives to fetch per anchor
    "hard_neg_min_dist":   HARD_NEG_MIN_DISTANCE,
    "hard_neg_max_dist":   HARD_NEG_MAX_DISTANCE,
    # Transition classifier
    "cls_lr":              1e-3,
    "cls_epochs":          30,
    "cls_batch_size":      64,
    "cls_dropout":         0.3,
    # Shared
    "weight_decay":        1e-4,
    "val_split":           0.2,     # fraction of mixes held out for validation
    "random_seed":         42,
}

TRANSITION_CLASSES = ["slam", "melt", "blend", "rise", "fade", "wave"]

# Transition feature normalisation constants
HARM_DIST_MAX         = 6.0   # max Camelot wheel distance; divides harm_dist → [0, 1]
TIME_GAP_NORM_DEFAULT = 0.5   # fallback when time_gap_norm is absent from labels

# Camelot wheel distances for harmonic compatibility scoring.
# Each key maps to its Camelot number (1-12). Same number or ±1 = compatible.
_CAMELOT = {
    "C": 8, "Cm": 5, "C#": 3, "C#m": 12,
    "D": 10, "Dm": 7, "D#": 5, "D#m": 2,
    "E": 12, "Em": 9, "F": 7, "Fm": 4,
    "F#": 2, "F#m": 11, "G": 9, "Gm": 6,
    "G#": 4, "G#m": 1, "A": 11, "Am": 8,
    "A#": 6, "A#m": 3, "B": 1, "Bm": 10,
}


def _harmonic_dist(key_a: str, key_b: str) -> float:
    """
    Camelot wheel distance between two keys: 0 (same) to 6 (maximally incompatible).
    Wraps around the 12-position wheel.
    Unknown keys default to 6 (worst-case).
    """
    ca = _CAMELOT.get(key_a, -1)
    cb = _CAMELOT.get(key_b, -1)
    if ca < 0 or cb < 0:
        return 6.0
    diff = abs(ca - cb)
    return float(min(diff, 12 - diff))


# ── Encoder input preparation ──────────────────────────────────────────────────

# Number of librosa features appended to the MERT embedding before the encoder.
# bpm_norm(1) + key_sin(1) + key_cos(1) + key_mode(1) + energy_mean(1)
# + onset_norm(1) + lufs_norm(1) = 7
ENCODER_EXTRA_DIM = 7


def _prepare_encoder_inputs(features: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    """
    Build the augmented encoder input matrix: MERT[768] + librosa[7] = [775].

    The 7 extra features make BPM and key compatibility explicit in the
    embedding space — critical for EDM where many tracks share similar timbral
    MERT embeddings but differ significantly in tempo and key.

    Key encoding uses circular (sin/cos) Camelot-wheel position so the model
    sees C (position 8) and B (position 1) as close on the wheel, not far apart
    as raw integers would suggest.

    Normalisation:
      bpm       / 200         → [0, 1] for typical 80–180 BPM range
      onset     / 5.0         → [0, 1] for librosa's raw onset scale
      lufs      (x+40)/40     → [0, 1] mapping −40…0 LUFS

    Returns:
        matrix     (N, 775) float32 array, row order matches features rows
        tid_to_idx track_id → row index in matrix
    """
    n = len(features)
    matrix = np.zeros((n, 768 + ENCODER_EXTRA_DIM), dtype=np.float32)
    tid_to_idx: dict[str, int] = {}

    for idx, (_, row) in enumerate(features.iterrows()):
        tid = row["track_id"]
        tid_to_idx[tid] = idx

        emb = np.asarray(row["embedding"], dtype=np.float32)

        bpm_norm  = float(row["bpm"]) / 200.0
        camelot   = _CAMELOT.get(str(row["key"]), 1)
        angle     = 2 * np.pi * camelot / 12
        key_sin   = float(np.sin(angle))
        key_cos   = float(np.cos(angle))
        key_mode  = 0.0 if str(row["key"]).endswith("m") else 1.0   # minor=0, major=1
        energy    = float(row["energy_mean"])
        onset_norm = min(float(row["onset_strength"]) / 5.0, 1.0)
        lufs_norm  = float(np.clip((float(row["loudness_lufs"]) + 40.0) / 40.0, 0.0, 1.0))

        extra = np.array([bpm_norm, key_sin, key_cos, key_mode, energy, onset_norm, lufs_norm],
                         dtype=np.float32)
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

    Pairs from the same mix are positives for contrastive training.
    Also builds a lookup: track_id → list of all its positive partner IDs (used
    for hard negative exclusion — we must not treat known positives as negatives).
    """
    all_pairs: dict[str, list[tuple]] = {}  # mix_id → list of (tid_a, tid_b)
    feat_ids = set(features["track_id"])

    for csv_path in mix_csvs:
        df = pd.read_csv(csv_path)

        for mix_id, group in df.groupby("mix_id"):
            # Keep only sequential tracks — simultaneous ("w/") tracks are
            # overlays, not transitions. They stay in tracklist.csv for the
            # inference engine but must not corrupt the pair sequence here.
            if "play_type" in group.columns:
                group = group[group["play_type"] == "sequential"]

            # Sort by starting_time when all values are real (MixesDB real minutes).
            # When any are NaN (1001tracklists missing timestamps), trust row order —
            # the scraper preserves track numbering order so row order = play order.
            if group["starting_time"].isna().any():
                group = group.reset_index(drop=True)
            else:
                group = group.sort_values("starting_time").reset_index(drop=True)
            tids = []
            for _, row in group.iterrows():
                tid = _track_id(row["artist_name"], row["track_name"])
                if tid in feat_ids:
                    tids.append(tid)

            # Build consecutive pairs
            pairs = [(tids[i], tids[i + 1]) for i in range(len(tids) - 1)]
            if pairs:
                all_pairs[str(mix_id)] = pairs

    mix_ids = list(all_pairs.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(mix_ids)
    split = int(len(mix_ids) * (1 - val_split))
    train_mixes, val_mixes = mix_ids[:split], mix_ids[split:]

    train_pairs = [p for mid in train_mixes for p in all_pairs[mid]]
    val_pairs   = [p for mid in val_mixes   for p in all_pairs[mid]]

    log.info(
        "Pairs — train: %d (%d mixes), val: %d (%d mixes)",
        len(train_pairs), len(train_mixes),
        len(val_pairs),   len(val_mixes),
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


# ── Datasets ───────────────────────────────────────────────────────────────────

class ContrastiveDataset(Dataset):
    """
    Returns (anchor_emb, positive_emb, hard_neg_embs) tensors.

    Hard negatives are pre-mined at the start of each epoch by querying ChromaDB.
    Pre-mining is faster than querying per-batch and produces stable negatives
    within an epoch.

    hard_neg_embs: (hard_neg_per_anchor, 768) tensor. May be zeros if ChromaDB
    yields no candidates in the distance window (rare but handled gracefully).
    """

    def __init__(
        self,
        pairs: list[tuple],
        features: pd.DataFrame,
        positive_index: dict[str, set],
        collection: chromadb.Collection,
        hard_neg_per_anchor: int = 5,
    ):
        self.pairs        = pairs
        self.pos_index    = positive_index
        self.collection   = collection
        self.n_hard       = hard_neg_per_anchor

        # Augmented inputs for encoder training (MERT + librosa features)
        self.input_matrix, self.tid_to_idx = _prepare_encoder_inputs(features)
        # Raw 768-dim MERT embeddings kept separately for ChromaDB ANN queries
        raw = features.set_index("track_id")["embedding"]
        self.raw_emb: dict[str, np.ndarray] = {
            tid: np.asarray(raw.loc[tid], dtype=np.float32) for tid in raw.index
        }
        self.hard_neg_cache: dict[str, np.ndarray] = {}

    def mine_hard_negatives(self) -> None:
        """
        Query ChromaDB for hard negatives for every unique anchor.
        Call this at the start of each epoch to refresh the cache.

        exclude_ids = anchor itself + all known positive partners.
        Candidates in [HARD_NEG_MIN_DISTANCE, HARD_NEG_MAX_DISTANCE] are kept.
        """
        unique_anchors = set(a for a, _ in self.pairs)
        log.info("Mining hard negatives for %d unique anchors...", len(unique_anchors))
        t0 = time.time()
        found = 0
        enc_dim = self.input_matrix.shape[1]
        for tid in unique_anchors:
            if tid not in self.raw_emb:
                continue
            # ChromaDB query uses raw 768-dim MERT — the index was built on those
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
                    [self.input_matrix[self.tid_to_idx[nid]]
                     for nid in neg_ids if nid in self.tid_to_idx]
                )
                # Pad or trim to exactly n_hard rows
                if len(neg_embs) < self.n_hard:
                    pad = np.zeros((self.n_hard - len(neg_embs), enc_dim), dtype=np.float32)
                    neg_embs = np.vstack([neg_embs, pad])
                self.hard_neg_cache[tid] = neg_embs[:self.n_hard]
                found += 1
        log.info(
            "Hard negative mining done: %d/%d anchors got negatives (%.1fs)",
            found, len(unique_anchors), time.time() - t0,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        tid_a, tid_b = self.pairs[idx]
        # Skip pairs where either track is missing from features
        if tid_a not in self.tid_to_idx or tid_b not in self.tid_to_idx:
            # Return zeros — collate_fn will drop these
            enc_dim = self.input_matrix.shape[1]
            z = torch.zeros(enc_dim)
            return z, z, torch.zeros(self.n_hard, enc_dim)

        anchor   = torch.tensor(self.input_matrix[self.tid_to_idx[tid_a]])
        positive = torch.tensor(self.input_matrix[self.tid_to_idx[tid_b]])

        enc_dim = self.input_matrix.shape[1]
        hard_neg_arr = self.hard_neg_cache.get(tid_a, np.zeros((self.n_hard, enc_dim), dtype=np.float32))
        hard_negs = torch.tensor(hard_neg_arr)

        return anchor, positive, hard_negs


class TransitionDataset(Dataset):
    """
    Returns (feature_vec, label_idx) for transition classification.

    feature_vec = concat(emb_A[128], emb_B[128], delta_features[4]) = 260-dim
    delta_features = [bpm_ratio, energy_delta, harmonic_dist, time_gap_norm]

    Requires:
      - transition_labels.csv: from_track_id, to_track_id, label, confidence
      - features.parquet: must contain 128-dim projected embeddings (emb_proj_*)
        produced by the contrastive encoder. Run this AFTER encoder training.
    """

    def __init__(self, labels_df: pd.DataFrame, features: pd.DataFrame, label_encoder: LabelEncoder):
        self.label_enc = label_encoder
        feat_idx = features.set_index("track_id")
        rows = []
        for _, row in labels_df.iterrows():
            tid_a, tid_b = row["from_track_id"], row["to_track_id"]
            if tid_a not in feat_idx.index or tid_b not in feat_idx.index:
                continue

            fa = feat_idx.loc[tid_a]
            fb = feat_idx.loc[tid_b]

            emb_a = np.array(fa["embedding_proj"], dtype=np.float32)
            emb_b = np.array(fb["embedding_proj"], dtype=np.float32)

            bpm_ratio    = float(fb["bpm"]) / max(float(fa["bpm"]), 1.0)
            energy_delta = float(fb["energy_mean"]) - float(fa["energy_mean"])
            harm_dist    = _harmonic_dist(str(fa["key"]), str(fb["key"])) / HARM_DIST_MAX
            _tg = row.get("time_gap_norm", TIME_GAP_NORM_DEFAULT)
            time_gap = TIME_GAP_NORM_DEFAULT if pd.isna(_tg) else float(_tg)

            delta = np.array([bpm_ratio, energy_delta, harm_dist, time_gap], dtype=np.float32)
            vec   = np.concatenate([emb_a, emb_b, delta])  # (260,)
            label = label_encoder.transform([row["label"]])[0]
            rows.append((vec, label))

        self.data = rows

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vec, label = self.data[idx]
        return torch.tensor(vec), torch.tensor(label, dtype=torch.long)


# ── Models ─────────────────────────────────────────────────────────────────────

class ContrastiveEncoder(nn.Module):
    """
    Projection head that maps 768-dim MERT embeddings → 128-dim unit hypersphere.

    Architecture: Linear → BN → ReLU → Linear → L2-normalise
    BatchNorm before ReLU stabilises training when input scale varies across tracks.
    L2 normalisation forces all embeddings onto the unit sphere — required for
    NT-Xent loss (which uses cosine similarity = dot product on unit sphere).

    The MERT backbone stays frozen. Only this head is trained (~200K params).
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)  # → unit hypersphere


class TransitionClassifier(nn.Module):
    """
    6-class MLP classifying transition type between two consecutive tracks.

    Input:  260-dim (concat of 128-dim projected embeddings A + B + 4 delta features)
    Hidden: 128 → 64 with BatchNorm + Dropout for regularisation
    Output: 6 logits (slam / melt / blend / rise / fade / wave)
    """

    def __init__(self, input_dim: int = 260, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, len(TRANSITION_CLASSES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── NT-Xent Loss ───────────────────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy (SimCLR loss) with hard negatives.

    For a batch of B (anchor, positive) pairs, the loss for anchor i is:
      -log( exp(sim(a_i, p_i) / τ) /
            [exp(sim(a_i, p_i) / τ)
             + Σ_{j≠i} exp(sim(a_i, a_j) / τ)     ← in-batch negatives
             + Σ_k     exp(sim(a_i, h_ik) / τ)] )  ← hard negatives

    Lower temperature τ → sharper distribution → harder training signal.
    Too low (< 0.05) → numerical instability. Typical range: 0.05–0.2.

    All vectors must be L2-normalised (encoder does this) so sim = dot product.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchors:   torch.Tensor,   # (B, D) — L2 normalised
        positives: torch.Tensor,   # (B, D) — L2 normalised
        hard_negs: torch.Tensor | None = None,  # (B, K, D) — L2 normalised
    ) -> torch.Tensor:
        B = anchors.size(0)

        # Positive similarity: (B,)
        pos_sim = (anchors * positives).sum(dim=1) / self.temperature

        # In-batch negative similarities: (B, B)
        # sim(a_i, a_j) for all j — diagonal is self-similarity (excluded)
        batch_sim = torch.mm(anchors, anchors.T) / self.temperature
        # Mask diagonal (self-similarity) with -inf so it's zeroed by softmax
        mask = torch.eye(B, device=anchors.device).bool()
        batch_sim = batch_sim.masked_fill(mask, float("-inf"))

        # Concatenate: positives | in-batch negatives
        # logits[:, 0] = positive sim, logits[:, 1:] = negative sims
        logits = torch.cat([pos_sim.unsqueeze(1), batch_sim], dim=1)  # (B, 1+B)

        if hard_negs is not None and hard_negs.size(1) > 0:
            # hard_negs: (B, K, D). Normalise (may have been padded with zeros).
            hard_negs = F.normalize(hard_negs, dim=2)
            # sim(a_i, h_ik): (B, K) via batched dot product
            hard_sim = torch.bmm(anchors.unsqueeze(1), hard_negs.transpose(1, 2))
            hard_sim = hard_sim.squeeze(1) / self.temperature  # (B, K)
            logits = torch.cat([logits, hard_sim], dim=1)      # (B, 1+B+K)

        # Target: position 0 is always the positive pair
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

    Metrics logged to both MLflow and W&B:

      alignment       Wang & Isola (2020): -mean||z_a - z_p||²  (higher = closer positives)
      uniformity      log mean exp(-2||z_i - z_j||²) (lower = more spread, no collapse)
      top1_retrieval  % of val anchors whose positive is their #1 nearest neighbor
      top5_retrieval  % where positive is in top 5
      top10_retrieval % where positive is in top 10
      mean_pos_sim    mean cosine similarity of positive pairs (higher = better)
      mean_neg_sim    mean cosine similarity of random non-pair tracks (should be low)
      emb_variance    mean per-dim variance of projected embeddings (collapse = near 0)

    Why these metrics matter:
      - alignment + uniformity: theoretical guarantees from contrastive learning theory.
        Good models should be both aligned (positives close) AND uniform (no mode collapse).
      - top-K retrieval: practical DJ use case — if you query with track A, does track B
        (the one that actually followed it in a real mix) come up first?
      - emb_variance: mode collapse early warning. If variance → 0, all tracks map to
        same point and the model is useless.
    """
    encoder.eval()
    input_matrix, tid_to_idx = _prepare_encoder_inputs(features)

    # Project all val-involved tracks
    val_ids = list({tid for pair in val_pairs for tid in pair if tid in tid_to_idx})
    if not val_ids:
        return {}

    val_inputs = np.stack([input_matrix[tid_to_idx[tid]] for tid in val_ids])
    raw_embs = torch.tensor(val_inputs, dtype=torch.float32).to(device)
    with torch.no_grad():
        proj = encoder(raw_embs)  # (N, 128)

    proj_idx = {tid: proj[i] for i, tid in enumerate(val_ids)}

    # ── Alignment ──────────────────────────────────────────────────────────────
    valid_pairs = [(a, b) for a, b in val_pairs if a in proj_idx and b in proj_idx]
    if not valid_pairs:
        return {}

    za = torch.stack([proj_idx[a] for a, _ in valid_pairs])  # (P, 128)
    zp = torch.stack([proj_idx[b] for _, b in valid_pairs])  # (P, 128)
    alignment = -((za - zp) ** 2).sum(dim=1).mean().item()

    # ── Uniformity ─────────────────────────────────────────────────────────────
    # Sample up to 1000 embeddings to keep it fast
    sample = proj[:min(1000, len(proj))]
    sq_dists = torch.cdist(sample, sample, p=2) ** 2
    uniformity = torch.log(torch.exp(-2 * sq_dists).mean()).item()

    # ── Top-K retrieval ────────────────────────────────────────────────────────
    # For each anchor, rank all val tracks by cosine similarity, check positive rank.
    all_proj = torch.stack([proj_idx[tid] for tid in val_ids])  # (N, 128)
    top1 = top5 = top10 = 0
    pos_sims = []
    neg_sims_list = []

    for tid_a, tid_b in valid_pairs:
        if tid_a not in proj_idx or tid_b not in proj_idx:
            continue
        q = proj_idx[tid_a]                              # (128,)
        sims = torch.mv(all_proj, q)                     # (N,) cosine sim (unit sphere)
        # Exclude self (tid_a)
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

        # Random non-pair similarity
        rand_idx = np.random.randint(0, len(val_ids))
        if val_ids[rand_idx] not in {tid_a, tid_b}:
            neg_sims_list.append(sims[rand_idx].item())

    n = len(valid_pairs)
    return {
        "val/alignment":        alignment,
        "val/uniformity":       uniformity,
        "val/top1_retrieval":   top1  / n,
        "val/top5_retrieval":   top5  / n,
        "val/top10_retrieval":  top10 / n,
        "val/mean_pos_sim":     float(np.mean(pos_sims)),
        "val/mean_neg_sim":     float(np.mean(neg_sims_list)) if neg_sims_list else 0.0,
        "val/emb_variance":     proj.var(dim=0).mean().item(),
    }


def compute_classifier_metrics(
    model: TransitionClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate transition classifier. All metrics logged to MLflow + W&B.

    Metrics:
      val/cls_accuracy     overall accuracy
      val/macro_f1         macro F1 — weights all 6 classes equally (catches imbalanced classes)
      val/weighted_f1      weighted F1 — weighted by support (matches real-world frequency)
      val/per_class_f1     dict of F1 per class (slam, melt, blend, rise, fade, wave)
      val/confusion_matrix logged as W&B table for interactive inspection
      val/mean_confidence  avg softmax prob on the predicted class (calibration proxy)

    Why macro_f1 matters: if the dataset has 80% "melt" transitions and 5% "slam",
    a model that always predicts "melt" gets 80% accuracy but 0% macro F1 on "slam".
    Macro F1 catches this — it's the honest metric for imbalanced multiclass problems.
    """
    model.eval()
    all_preds, all_labels, all_confs = [], [], []

    with torch.no_grad():
        for vecs, labels in loader:
            vecs, labels = vecs.to(device), labels.to(device)
            logits = model(vecs)
            probs  = F.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)
            confs  = probs.max(dim=1).values

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())

    report = classification_report(
        all_labels, all_preds,
        target_names=TRANSITION_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    per_class_f1 = {cls: report[cls]["f1-score"] for cls in TRANSITION_CLASSES if cls in report}

    cm = confusion_matrix(all_labels, all_preds)
    # Log confusion matrix as W&B Table — each row is a true class, columns are predicted
    cm_table = wandb.Table(
        columns=["true \\ pred"] + TRANSITION_CLASSES,
        data=[[TRANSITION_CLASSES[i]] + cm[i].tolist() for i in range(len(cm))],
    )

    metrics = {
        "val/cls_accuracy":    accuracy_score(all_labels, all_preds),
        "val/macro_f1":        f1_score(all_labels, all_preds, average="macro",    zero_division=0),
        "val/weighted_f1":     f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "val/mean_confidence": float(np.mean(all_confs)),
        "val/confusion_matrix": cm_table,
        **{f"val/f1_{cls}": v for cls, v in per_class_f1.items()},
    }
    return metrics


# ── Training loops ─────────────────────────────────────────────────────────────

def train_contrastive(
    encoder:    ContrastiveEncoder,
    criterion:  NTXentLoss,
    train_ds:   ContrastiveDataset,
    val_pairs:  list[tuple],
    features:   pd.DataFrame,
    device:     torch.device,
    hparams:    dict,
) -> ContrastiveEncoder:
    """
    Train the contrastive encoder for `encoder_epochs` epochs.

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
        # Refresh hard negatives at the start of each epoch
        train_ds.mine_hard_negatives()

        encoder.train()
        epoch_loss = 0.0
        grad_norms = []

        for batch_idx, (anchors, positives, hard_negs) in enumerate(loader):
            anchors   = anchors.to(device)
            positives = positives.to(device)
            hard_negs = hard_negs.to(device)

            z_a = encoder(anchors)
            z_p = encoder(positives)
            # Project hard negatives — flatten batch dim, encode, reshape
            B, K, D = hard_negs.shape
            z_h = encoder(hard_negs.view(B * K, D)).view(B, K, -1)

            loss = criterion(z_a, z_p, z_h)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping — prevents exploding gradients with hard negatives
            grad_norm = nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            grad_norms.append(grad_norm.item())

            # W&B: log every batch for smooth loss curves
            wandb.log({
                "train/loss":      loss.item(),
                "train/grad_norm": grad_norm.item(),
                "train/lr":        scheduler.get_last_lr()[0],
            })

        scheduler.step()

        avg_loss = epoch_loss / max(len(loader), 1)
        avg_grad = float(np.mean(grad_norms))

        # MLflow: log epoch-level summary
        mlflow.log_metric("train/epoch_loss", avg_loss,  step=epoch)
        mlflow.log_metric("train/grad_norm",  avg_grad,  step=epoch)

        # Validation metrics every epoch
        val_metrics = compute_contrastive_metrics(encoder, val_pairs, features, device)
        for k, v in val_metrics.items():
            mlflow.log_metric(k, v, step=epoch)
        wandb.log({"epoch": epoch, **val_metrics})

        top1 = val_metrics.get("val/top1_retrieval", 0.0)
        log.info(
            "[Encoder] epoch %d/%d  loss=%.4f  top1=%.3f  alignment=%.4f  uniformity=%.4f",
            epoch, hparams["encoder_epochs"], avg_loss, top1,
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


def train_classifier(
    classifier: TransitionClassifier,
    train_ds:   TransitionDataset,
    val_ds:     TransitionDataset,
    device:     torch.device,
    hparams:    dict,
) -> TransitionClassifier:
    """
    Train the 6-class transition classifier.

    Uses cross-entropy with class weights to handle label imbalance —
    rare transition types (slam, wave) get upweighted so the model doesn't
    ignore them in favour of the majority class (melt/blend).
    """
    train_loader = DataLoader(train_ds, batch_size=hparams["cls_batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=hparams["cls_batch_size"])

    # Compute class weights from training set label distribution
    label_counts = np.bincount([lbl for _, lbl in train_ds.data], minlength=len(TRANSITION_CLASSES))
    weights = 1.0 / (label_counts + 1e-6)
    weights /= weights.sum()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=hparams["cls_lr"],
        weight_decay=hparams["weight_decay"],
    )

    best_macro_f1 = 0.0

    for epoch in range(1, hparams["cls_epochs"] + 1):
        classifier.train()
        epoch_loss = 0.0

        for vecs, labels in train_loader:
            vecs, labels = vecs.to(device), labels.to(device)
            logits = classifier(vecs)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            wandb.log({"train/cls_loss": loss.item()})

        avg_loss = epoch_loss / max(len(train_loader), 1)
        mlflow.log_metric("train/cls_epoch_loss", avg_loss, step=epoch)

        val_metrics = compute_classifier_metrics(classifier, val_loader, device)
        macro_f1    = val_metrics.get("val/macro_f1", 0.0)

        # W&B: log all metrics; confusion matrix is a Table (rendered interactively)
        wandb.log({"epoch": epoch, **val_metrics})
        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v, step=epoch)

        log.info(
            "[Classifier] epoch %d/%d  loss=%.4f  acc=%.3f  macro_f1=%.3f",
            epoch, hparams["cls_epochs"], avg_loss,
            val_metrics.get("val/cls_accuracy", 0),
            macro_f1,
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(classifier.state_dict(), CLASSIFIER_PATH)
            log.info("  ↑ New best macro F1: %.3f → saved %s", best_macro_f1, CLASSIFIER_PATH)
            mlflow.pytorch.log_model(classifier, artifact_path="classifier_best")

    log.info("Classifier training done. Best macro F1: %.3f", best_macro_f1)
    return classifier


# ── Main entry point ───────────────────────────────────────────────────────────

def train(
    mix_csvs:  list[str | Path] | None = None,
    hparams:   dict | None = None,
) -> None:
    """
    Full training pipeline: encoder → classifier.

    mlflow:  logs to ./mlruns/  — view with `mlflow ui --port 5000`
    wandb:   logs to wandb.ai   — run `wandb login` once before calling this

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

    # ── Load features ──────────────────────────────────────────────────────────
    features = pd.read_parquet(FEATURES_PATH)
    log.info("Features loaded: %d tracks, %d columns", len(features), len(features.columns))

    # ── Build consecutive pairs ────────────────────────────────────────────────
    train_pairs, val_pairs = build_consecutive_pairs(
        [Path(p) for p in mix_csvs],
        features,
        val_split=hparams["val_split"],
        seed=hparams["random_seed"],
    )
    positive_index = build_positive_index(train_pairs + val_pairs)

    # ── ChromaDB ───────────────────────────────────────────────────────────────
    client     = get_client(CHROMA_PATH)
    collection = get_collection(client)
    log.info("ChromaDB collection '%s': %d tracks", COLLECTION_NAME, collection.count())

    # ── MLflow + W&B: start a single run covering both models ─────────────────
    mlflow.set_experiment("ai-dj-training")

    # wandb.init groups all epochs under one run on the W&B dashboard.
    # config=hparams stores all hyperparameters — visible in the run summary.
    wandb.init(
        project="ai-dj",
        name=f"run-{int(time.time())}",
        config=hparams,
        tags=["contrastive", "transition-classifier"],
    )

    with mlflow.start_run():
        # Log all hyperparameters to MLflow in one call
        mlflow.log_params(hparams)

        # ── Phase A: Contrastive encoder ───────────────────────────────────────
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

        # W&B model watcher: logs gradient histograms every 100 batches
        wandb.watch(encoder, log="gradients", log_freq=100)

        encoder = train_contrastive(
            encoder, criterion, train_ds, val_pairs, features, device, hparams
        )

        # ── Phase B: Project all embeddings and add to features ────────────────
        # The classifier needs 128-dim projected embeddings, not raw 768-dim MERT.
        log.info("Projecting all embeddings with trained encoder...")
        encoder.eval()
        input_matrix, _ = _prepare_encoder_inputs(features)
        all_raw = torch.tensor(input_matrix, dtype=torch.float32).to(device)
        with torch.no_grad():
            all_proj = encoder(all_raw).cpu().numpy()  # (N, 128)

        features = features.assign(embedding_proj=[row.tolist() for row in all_proj])
        features.to_parquet(FEATURES_PATH, index=False)
        log.info("Projected embeddings written to %s", FEATURES_PATH)

        # ── Phase C: Transition classifier (requires Phase 4 labels) ──────────
        if not LABELS_PATH.exists():
            log.warning(
                "Transition labels not found at %s — skipping classifier training.\n"
                "Run Phase 4 (transition_labeler.py) first, then re-run this script.",
                LABELS_PATH,
            )
        else:
            log.info("=== Training transition classifier ===")
            labels_df = pd.read_csv(LABELS_PATH)

            le = LabelEncoder()
            le.fit(TRANSITION_CLASSES)

            train_lbl, val_lbl = train_test_split(
                labels_df,
                test_size=hparams["val_split"],
                stratify=labels_df["label"],
                random_state=hparams["random_seed"],
            )

            cls_train_ds = TransitionDataset(train_lbl, features, le)
            cls_val_ds   = TransitionDataset(val_lbl,   features, le)

            classifier = TransitionClassifier(
                input_dim=260,
                dropout=hparams["cls_dropout"],
            ).to(device)
            wandb.watch(classifier, log="gradients", log_freq=100)

            train_classifier(classifier, cls_train_ds, cls_val_ds, device, hparams)

        mlflow.log_artifact(str(ENCODER_PATH))
        if CLASSIFIER_PATH.exists():
            mlflow.log_artifact(str(CLASSIFIER_PATH))

    wandb.finish()
    log.info("Training complete. Models saved to %s/", MODELS_DIR)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _fmt      = "%(asctime)s %(levelname)s %(message)s"
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
