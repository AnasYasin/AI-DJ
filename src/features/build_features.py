"""
Feature extraction pipeline.

For each track in preview_manifest.csv, extracts:
  1. MERT embedding (768-dim) — m-a-p/MERT-v1-95M
  2. Librosa features — bpm, key, loudness_lufs, energy_mean/std,
                        spectral_centroid, onset_strength, mfcc_0..12

Modes (--mode flag):
  both         — MERT + librosa → features.parquet  (default)
  mert-only    — MERT only      → embeddings.parquet  (run on GPU instance)
  librosa-only — librosa only   → features.parquet  (run locally, reads embeddings.parquet)

All modes are resumable: re-running skips already-processed tracks.
"""

import argparse
import logging
from pathlib import Path
import signal
import time

import librosa
from mutagen import File as MutaFile
import numpy as np
import pandas as pd
import pyloudnorm as pyln
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

log = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/raw/preview_manifest.csv")
FEATURES_PATH = Path("data/processed/features.parquet")
EMBEDDINGS_PATH = Path("data/processed/embeddings.parquet")
MERT_MODEL_NAME = "m-a-p/MERT-v1-95M"
LOAD_TIMEOUT_S = 30
# Conservative for T4 16GB: model ~380MB + batch-4 attention ~1.5GB → ~14GB headroom.
# Falls back to 1 automatically on OOM — do not raise this without testing.
MERT_BATCH_SIZE = 4


def _load_audio(path: str, sr: int) -> tuple:
    """Load audio with a hard timeout — librosa can hang on corrupt MP3 headers."""
    def _handler(signum, frame):
        raise TimeoutError(f"librosa.load timed out after {LOAD_TIMEOUT_S}s: {path}")
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(LOAD_TIMEOUT_S)
    try:
        return librosa.load(path, sr=sr, mono=True)
    finally:
        signal.alarm(0)


_CHROMA_TO_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _detect_key(chroma_mean: np.ndarray) -> str:
    best_score, best_key = -np.inf, "C"
    for root in range(12):
        major_profile = np.roll(_KS_MAJOR, root)
        minor_profile = np.roll(_KS_MINOR, root)
        score_major = np.corrcoef(chroma_mean, major_profile)[0, 1]
        score_minor = np.corrcoef(chroma_mean, minor_profile)[0, 1]
        if score_major > best_score:
            best_score = score_major
            best_key = _CHROMA_TO_NOTE[root]
        if score_minor > best_score:
            best_score = score_minor
            best_key = _CHROMA_TO_NOTE[root] + "m"
    return best_key


# ── MERT Embedder ──────────────────────────────────────────────────────────────


class MERTEmbedder:
    SAMPLE_RATE = 24_000

    def __init__(self, model_name: str = MERT_MODEL_NAME):
        log.info("Loading MERT model: %s", model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Using device: %s", self.device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        log.info("MERT model loaded on %s", self.device)

    def _forward(self, audios: list[np.ndarray]) -> list[np.ndarray]:
        """One batched GPU forward pass. Returns one 768-dim array per audio."""
        inputs = self.processor(
            audios,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # (batch, 768)
        del inputs, outputs
        return [embs[i] for i in range(len(audios))]

    def embed_batch(self, paths: list[str]) -> list[np.ndarray | None]:
        """
        Embed a list of audio files in one GPU forward pass.
        Falls back to one-by-one if GPU OOM — never crashes the pipeline.
        Returns one embedding per path (None if that track failed).
        """
        results: list[np.ndarray | None] = [None] * len(paths)

        # Load audio sequentially — SIGALRM timeout only works single-threaded
        loaded: list[tuple[int, np.ndarray]] = []
        for idx, path in enumerate(paths):
            try:
                audio, _ = _load_audio(path, self.SAMPLE_RATE)
                loaded.append((idx, audio))
            except Exception as e:
                log.warning("  Audio load failed %s: %s", path, e)

        if not loaded:
            return results

        indices, audios = zip(*loaded)

        try:
            embs = self._forward(list(audios))
            for orig_idx, emb in zip(indices, embs):
                results[orig_idx] = emb

        except torch.cuda.OutOfMemoryError:
            log.warning("  GPU OOM on batch=%d — falling back one-by-one", len(audios))
            torch.cuda.empty_cache()
            for orig_idx, audio in zip(indices, audios):
                try:
                    results[orig_idx] = self._forward([audio])[0]
                except Exception as e:
                    log.warning("  Fallback embed failed: %s", e)
                    torch.cuda.empty_cache()

        return results

    def embed(self, audio_path: str) -> np.ndarray | None:
        return self.embed_batch([audio_path])[0]


# ── Librosa Feature Extractor ──────────────────────────────────────────────────


class LibrosaExtractor:
    SAMPLE_RATE = 22_050

    def extract(self, audio_path: str) -> dict | None:
        try:
            y, sr = _load_audio(audio_path, self.SAMPLE_RATE)

            bpm, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120.0)
            bpm = float(bpm)
            while bpm < 80:
                bpm *= 2
            while bpm > 180:
                bpm /= 2

            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key = _detect_key(chroma.mean(axis=1))

            y_stereo = np.stack([y, y], axis=1)
            meter = pyln.Meter(sr)
            lufs = meter.integrated_loudness(y_stereo)
            if not np.isfinite(lufs):
                lufs = -70.0

            rms = librosa.feature.rms(y=y)[0]
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            return {
                "bpm": float(bpm),
                "key": key,
                "loudness_lufs": float(lufs),
                "energy_mean": float(rms.mean()),
                "energy_std": float(rms.std()),
                "spectral_centroid": float(centroid.mean()),
                "onset_strength": float(onset_env.mean()),
                **{f"mfcc_{i}": float(v) for i, v in enumerate(mfcc.mean(axis=1))},
            }

        except Exception as e:
            log.warning("librosa extraction failed for %s: %s", audio_path, e)
            return None


# ── Main pipeline ──────────────────────────────────────────────────────────────


def _do_checkpoint(
    existing: pd.DataFrame,
    rows: list[dict],
    out_path: Path,
    i: int,
    total: int,
    t_start: float,
    skipped: int,
    use_gpu: bool,
) -> pd.DataFrame:
    elapsed = time.time() - t_start
    rate = i / elapsed if elapsed > 0 else 1
    remaining = (total - i) / rate
    hrs, rem_s = divmod(int(remaining), 3600)
    mins = rem_s // 60
    log.info(
        "--- checkpoint %d/%d | %.2f s/track | ETA %dh %02dm | skipped %d ---",
        i, total, elapsed / max(i, 1), hrs, mins, skipped,
    )
    if rows:
        batch_df = pd.DataFrame(rows)
        existing = (
            pd.concat([existing, batch_df], ignore_index=True)
            if not existing.empty
            else batch_df
        )
        existing.to_parquet(out_path, index=False)
        rows.clear()
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("  Checkpoint saved → %s (%d tracks total)", out_path, len(existing))
    return existing


def build_features(manifest_path: str = str(MANIFEST_PATH), mode: str = "both") -> pd.DataFrame:
    """
    mode="both"         — MERT + librosa → features.parquet  (default)
    mode="mert-only"    — MERT only      → embeddings.parquet
    mode="librosa-only" — librosa only   → features.parquet  (reads embeddings.parquet)
    All modes resumable: re-running skips already-processed track_ids.
    """
    assert mode in ("both", "mert-only", "librosa-only"), f"Unknown mode: {mode}"
    out_path = EMBEDDINGS_PATH if mode == "mert-only" else FEATURES_PATH
    log.info("Mode: %s  →  %s", mode, out_path)

    manifest = pd.read_csv(manifest_path)

    if mode == "librosa-only":
        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                f"embeddings.parquet not found at {EMBEDDINGS_PATH} — run --mode mert-only first"
            )
        embeddings_df = pd.read_parquet(EMBEDDINGS_PATH)
        to_process = embeddings_df.merge(
            manifest[["track_id", "local_path", "artist", "track_name"]],
            on="track_id",
            how="inner",
        )
        log.info("Embeddings loaded: %d tracks", len(embeddings_df))
    else:
        to_process = manifest[manifest["source"] != "not_found"].copy()
        log.info("Manifest: %d tracks total, %d have audio", len(manifest), len(to_process))

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        done = set(existing["track_id"]) if "track_id" in existing.columns else set()
        to_process = to_process[~to_process["track_id"].isin(done)]
        log.info("Already processed: %d  Remaining: %d", len(done), len(to_process))
    else:
        existing = pd.DataFrame()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if to_process.empty:
        log.info("Nothing new to process.")
        return existing

    embedder = MERTEmbedder() if mode != "librosa-only" else None
    extractor = LibrosaExtractor() if mode != "mert-only" else None

    all_rows = to_process.to_dict("records")
    total = len(all_rows)
    skipped = 0
    CHECKPOINT_EVERY = max(10, total // 100)
    rows: list[dict] = []
    t_start = time.time()
    i = 0
    last_ckpt_i = 0

    # ── librosa-only: CPU-only, no batching ───────────────────────────────────
    if mode == "librosa-only":
        for row in all_rows:
            i += 1
            log.info("[%d/%d] %s - %s", i, total, row["artist"], row["track_name"])
            librosa_feats = extractor.extract(row["local_path"])
            if librosa_feats is None:
                log.warning("  Skipping — librosa extraction failed")
                skipped += 1
            else:
                rows.append({
                    "track_id": row["track_id"],
                    "embedding": row["embedding"],
                    **librosa_feats,
                })

            if i - last_ckpt_i >= CHECKPOINT_EVERY or i >= total:
                existing = _do_checkpoint(existing, rows, out_path, i, total, t_start, skipped, use_gpu=False)
                last_ckpt_i = i

    # ── mert-only / both: GPU batched MERT ────────────────────────────────────
    else:
        for batch_start in range(0, total, MERT_BATCH_SIZE):
            batch = all_rows[batch_start : batch_start + MERT_BATCH_SIZE]

            valid: list[dict] = []
            for row in batch:
                i += 1
                path = row["local_path"]
                try:
                    mf = MutaFile(path)
                    if mf is None or mf.info.length <= 0:
                        raise ValueError("empty or unreadable")
                    valid.append(row)
                except Exception as e:
                    log.warning("[%d/%d] Skipping corrupt %s: %s", i, total, path, e)
                    skipped += 1

            if valid:
                emb_vecs = embedder.embed_batch([r["local_path"] for r in valid])

                for j, (row, emb_vec) in enumerate(zip(valid, emb_vecs)):
                    log.info(
                        "[%d/%d] %s - %s",
                        batch_start + j + 1,
                        total,
                        row["artist"],
                        row["track_name"],
                    )
                    if emb_vec is None:
                        log.warning("  Skipping — MERT embed failed")
                        skipped += 1
                        continue

                    if mode == "mert-only":
                        rows.append({
                            "track_id": row["track_id"],
                            "embedding": emb_vec.tolist(),
                        })
                    else:
                        librosa_feats = extractor.extract(row["local_path"])
                        if librosa_feats is None:
                            log.warning("  Skipping — librosa extraction failed")
                            skipped += 1
                            continue
                        rows.append({
                            "track_id": row["track_id"],
                            "embedding": emb_vec.tolist(),
                            **librosa_feats,
                        })

            if i - last_ckpt_i >= CHECKPOINT_EVERY or i >= total:
                existing = _do_checkpoint(existing, rows, out_path, i, total, t_start, skipped, use_gpu=True)
                last_ckpt_i = i

    elapsed_total = time.time() - t_start
    log.info(
        "Done. %d tracks in %s (%.0f min, %d skipped)",
        len(existing), out_path, elapsed_total / 60, skipped,
    )
    return existing


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MERT and/or librosa features.")
    parser.add_argument(
        "--mode",
        choices=["both", "mert-only", "librosa-only"],
        default="both",
        help=(
            "both (default): MERT+librosa → features.parquet | "
            "mert-only: MERT → embeddings.parquet (run on GPU instance) | "
            "librosa-only: librosa → features.parquet (run locally, reads embeddings.parquet)"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(MANIFEST_PATH),
        help="Path to preview manifest CSV (default: data/raw/preview_manifest.csv)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process the first N tracks (for quick smoke tests)",
    )
    args = parser.parse_args()

    _fmt = "%(asctime)s %(levelname)s %(message)s"
    _log_path = Path("logs/build_features.log")
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

    manifest_path = args.manifest
    if args.sample:
        import tempfile
        df = pd.read_csv(args.manifest)
        df = df[df["source"] != "not_found"].head(args.sample)
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        df.to_csv(tmp.name, index=False)
        manifest_path = tmp.name
        log.info("Sample mode: using %d tracks from %s", len(df), manifest_path)

    build_features(manifest_path=manifest_path, mode=args.mode)
