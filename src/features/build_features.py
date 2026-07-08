"""
Feature extraction pipeline.

For each track in preview_manifest.csv, extracts:
  1. discogs-effnet embedding (1280-dim) — essentia.upf.edu/models
  2. Audio features — bpm (DeepRhythm CNN), key (Essentia KeyExtractor),
                     loudness_lufs, energy_mean/std, spectral_centroid,
                     onset_strength, mfcc_0..12 (librosa)

Modes (--mode flag):
  both          — discogs-effnet + features → features.parquet  (default)
  discogs-only  — discogs-effnet only       → embeddings.parquet  (run on GPU instance)
  librosa-only  — features only             → librosa_features.parquet  (any machine with audio files)

All modes are resumable: re-running skips already-processed tracks.
Merge embeddings + librosa features: src/data/preprocess_features.py
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import multiprocessing
import os
from pathlib import Path
import signal
import time

from deeprhythm import DeepRhythmPredictor
import essentia.standard as es
import librosa
from mutagen import File as MutaFile
import numpy as np
import pandas as pd
import pyloudnorm as pyln

log = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/raw/preview_manifest.csv")
EMBEDDINGS_PATH = Path("data/processed/embeddings.parquet")
DISCOGS_MODEL_PATH = Path("models/essentia/discogs-effnet-bs64-1.pb")
LIBROSA_FEATURES_PATH = Path("data/processed/librosa_features.parquet")
LOAD_TIMEOUT_S = 30

# Essentia may return flat key names on some platforms — normalise to sharps
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

# All valid key strings produced by Essentia KeyExtractor after normalisation
VALID_KEYS = {
    n + m
    for n in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for m in ["", "m"]
}


def _load_audio(path: str, sr: int) -> tuple:
    """Load audio with a hard timeout — librosa can hang on corrupt M4A headers."""

    def _handler(signum, frame):
        raise TimeoutError(f"librosa.load timed out after {LOAD_TIMEOUT_S}s: {path}")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(LOAD_TIMEOUT_S)
    try:
        return librosa.load(path, sr=sr, mono=True)
    finally:
        signal.alarm(0)


def _essentia_key(y: np.ndarray, sr: int) -> str:
    """Detect key using Essentia KeyExtractor — more accurate than Krumhansl-Schmuckler."""
    key_note, scale, _ = es.KeyExtractor(sampleRate=float(sr), profileType="edma")(
        y.astype(np.float32)
    )
    key_note = _FLAT_TO_SHARP.get(key_note, key_note)
    return key_note + ("m" if scale == "minor" else "")


# ── discogs-effnet Embedder ────────────────────────────────────────────────────


class DiscogsEmbedder:
    """
    Wraps the Essentia discogs-effnet TensorFlow model.
    GPU is used automatically by TensorFlow when CUDA is available — no extra config needed.
    Output: 1280-dim mean-pooled embedding per track.
    """

    SAMPLE_RATE = 16_000
    EMBEDDING_DIM = 1280

    def __init__(self, model_path: Path = DISCOGS_MODEL_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"discogs-effnet model not found at {model_path}.\n"
                "Download: wget -O models/essentia/discogs-effnet-bs64-1.pb "
                "https://essentia.upf.edu/models/feature-extractors/"
                "discogs-effnet/discogs-effnet-bs64-1.pb"
            )
        log.info("Loading discogs-effnet: %s", model_path)
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            log.info("TensorFlow device: %s", "GPU" if gpus else "CPU")
        except Exception:
            pass
        self._model = es.TensorflowPredictEffnetDiscogs(
            graphFilename=str(model_path),
            output="PartitionedCall:1",
        )
        log.info("discogs-effnet loaded (dim=%d)", self.EMBEDDING_DIM)

    def embed(self, audio_path: str) -> np.ndarray | None:
        """Embed one audio file → (1280,) float32. Returns None on failure."""
        try:
            y, _ = _load_audio(audio_path, self.SAMPLE_RATE)
            frames = self._model(y.astype(np.float32))  # (N_frames, 1280)
            return np.array(frames, dtype=np.float32).mean(axis=0)
        except Exception as e:
            log.warning("  discogs embed failed %s: %s", audio_path, e)
            return None


# ── Batched GPU Embedder ───────────────────────────────────────────────────────


class DiscogsEmbedderGPU:
    """
    Batched GPU discogs-effnet embedder.
    Loads the frozen .pb directly with TensorFlow — bypasses essentia's one-at-a-time
    TF wrapper so the GPU is actually saturated.
    Stacks mel patches from batch_tracks audio files into one GPU call.
    """

    SAMPLE_RATE = 16_000
    N_FFT = 512
    HOP_LENGTH = 256
    N_MELS = 96
    PATCH_FRAMES = 128
    MODEL_BATCH = 64  # fixed batch size baked into the .pb graph
    EMBEDDING_DIM = 1280
    INPUT_TENSOR = "serving_default_melspectrogram:0"
    OUTPUT_TENSOR = "PartitionedCall:1"

    def __init__(self, model_path: Path = DISCOGS_MODEL_PATH, batch_tracks: int = 64, loader_workers: int = 8):
        import tensorflow as tf

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"discogs-effnet model not found at {model_path}.\n"
                "Download: wget -O models/essentia/discogs-effnet-bs64-1.pb "
                "https://essentia.upf.edu/models/feature-extractors/"
                "discogs-effnet/discogs-effnet-bs64-1.pb"
            )
        self.batch_tracks = batch_tracks
        self.loader_workers = loader_workers

        graph_def = tf.compat.v1.GraphDef()
        with open(str(model_path), "rb") as f:
            graph_def.ParseFromString(f.read())

        self._graph = tf.compat.v1.Graph()
        with self._graph.as_default():
            tf.compat.v1.import_graph_def(graph_def, name="")

        cfg = tf.compat.v1.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self._sess = tf.compat.v1.Session(graph=self._graph, config=cfg)

        try:
            self._input = self._graph.get_tensor_by_name(self.INPUT_TENSOR)
            self._output = self._graph.get_tensor_by_name(self.OUTPUT_TENSOR)
        except KeyError:
            placeholders = [
                op.name for op in self._graph.get_operations() if op.type == "Placeholder"
            ]
            log.error("Tensor not found. Placeholders in graph: %s", placeholders[:10])
            raise

        gpus = tf.config.list_physical_devices("GPU")
        log.info(
            "DiscogsEmbedderGPU loaded (device=%s, batch_tracks=%d)",
            "GPU" if gpus else "CPU",
            batch_tracks,
        )

    def _mel_patches(self, y: np.ndarray) -> np.ndarray | None:
        """Float32 mono 16 kHz → (N_patches, 128, 96) float32 — model input format.

        Mel frames MUST come from essentia's TensorflowInputMusiCNN — the same
        frontend TensorflowPredictEffnetDiscogs applies internally (512/256 STFT,
        96 mel bands, log10(1 + 10000·x) compression). A librosa reimplementation
        (plain log10, different filterbank) fed the network out-of-distribution
        input and produced garbage embeddings that still looked shape-valid.
        """
        mel_input = es.TensorflowInputMusiCNN()  # fresh instance: not thread-safe
        frames = [
            mel_input(frame)
            for frame in es.FrameGenerator(
                y.astype(np.float32),
                frameSize=self.N_FFT,
                hopSize=self.HOP_LENGTH,
                startFromZero=True,
            )
        ]
        n = len(frames) // self.PATCH_FRAMES
        if n == 0:
            return None
        mel = np.array(frames[: n * self.PATCH_FRAMES], dtype=np.float32)  # (T, 96)
        return mel.reshape(n, self.PATCH_FRAMES, self.N_MELS)

    def _load_patches(self, args: tuple[int, str]) -> tuple[int, np.ndarray | None]:
        i, path = args
        try:
            # SIGALRM is not thread-safe; MutaFile pre-validates files so hangs are unlikely
            y, _ = librosa.load(path, sr=self.SAMPLE_RATE, mono=True)
            return i, self._mel_patches(y)
        except Exception as e:
            log.warning("  audio load failed %s: %s", path, e)
            return i, None

    def embed_batch(self, audio_paths: list[str]) -> list[np.ndarray | None]:
        """List of paths → list of (1280,) float32 arrays (None on failure)."""
        all_patches: list[np.ndarray] = []
        patch_counts: list[int] = []
        valid_idx: list[int] = []

        with ThreadPoolExecutor(max_workers=self.loader_workers) as pool:
            patch_results = sorted(
                pool.map(self._load_patches, enumerate(audio_paths)),
                key=lambda x: x[0],
            )

        for i, patches in patch_results:
            if patches is not None:
                all_patches.append(patches)
                patch_counts.append(len(patches))
                valid_idx.append(i)

        results: list[np.ndarray | None] = [None] * len(audio_paths)
        if not all_patches:
            return results

        X = np.concatenate(all_patches, axis=0)  # (total_patches, 128, 96)

        # model has fixed batch size MODEL_BATCH=64 — feed in chunks, pad last if needed
        all_embeddings = []
        for start in range(0, len(X), self.MODEL_BATCH):
            chunk = X[start : start + self.MODEL_BATCH]
            if len(chunk) < self.MODEL_BATCH:
                pad = np.zeros((self.MODEL_BATCH - len(chunk), *chunk.shape[1:]), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)
            out = self._sess.run(self._output, {self._input: chunk})  # (64, 1280)
            all_embeddings.append(out)

        embeddings = np.concatenate(all_embeddings, axis=0)[: len(X)]  # trim padding

        idx = 0
        for i, count in zip(valid_idx, patch_counts):
            results[i] = embeddings[idx : idx + count].mean(axis=0).astype(np.float32)
            idx += count

        return results


# ── Audio Feature Extractor ────────────────────────────────────────────────────


class LibrosaExtractor:
    """
    Extracts per-track audio features.
    BPM: DeepRhythm CNN — more accurate than rule-based estimators for EDM.
    Key: Essentia KeyExtractor.
    Everything else: librosa.
    """

    SAMPLE_RATE = 22_050

    def __init__(self):
        self._dr = DeepRhythmPredictor()

    def extract(self, audio_path: str) -> dict | None:
        try:
            y, sr = _load_audio(audio_path, self.SAMPLE_RATE)

            bpm = self._dr.predict_from_audio(y, sr)
            if bpm is None:
                raise ValueError("DeepRhythm could not detect BPM")
            bpm = float(bpm)
            key = _essentia_key(y, sr)

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
                "bpm": bpm,
                "key": key,
                "loudness_lufs": float(lufs),
                "energy_mean": float(rms.mean()),
                "energy_std": float(rms.std()),
                "spectral_centroid": float(centroid.mean()),
                "onset_strength": float(onset_env.mean()),
                **{f"mfcc_{i}": float(v) for i, v in enumerate(mfcc.mean(axis=1))},
            }

        except Exception as e:
            log.warning("Feature extraction failed for %s: %s", audio_path, e)
            return None


# ── Parallel worker (module-level so ProcessPoolExecutor can pickle it) ────────

_worker_extractor: "LibrosaExtractor | None" = None


def _librosa_worker(args: tuple) -> tuple[str, dict | None]:
    """Extract librosa features for one track. Returns (track_id, feats_or_None).
    DeepRhythm model is loaded once per worker process, not once per track."""
    global _worker_extractor
    if _worker_extractor is None:
        _worker_extractor = LibrosaExtractor()
    track_id, local_path = args
    return track_id, _worker_extractor.extract(local_path)


# ── Checkpoint helper ──────────────────────────────────────────────────────────


def _do_checkpoint(
    existing: pd.DataFrame,
    rows: list[dict],
    out_path: Path,
    i: int,
    total: int,
    t_start: float,
    skipped: int,
) -> pd.DataFrame:
    elapsed = time.time() - t_start
    rate = i / elapsed if elapsed > 0 else 1
    remaining = (total - i) / rate
    hrs, rem_s = divmod(int(remaining), 3600)
    mins = rem_s // 60
    log.info(
        "--- checkpoint %d/%d | %.2f s/track | ETA %dh %02dm | skipped %d ---",
        i,
        total,
        elapsed / max(i, 1),
        hrs,
        mins,
        skipped,
    )
    if rows:
        batch_df = pd.DataFrame(rows)
        existing = (
            pd.concat([existing, batch_df], ignore_index=True) if not existing.empty else batch_df
        )
        existing.to_parquet(out_path, index=False)
        rows.clear()
    log.info("  Checkpoint saved → %s (%d tracks total)", out_path, len(existing))
    return existing


# ── Main pipeline ──────────────────────────────────────────────────────────────


def build_features(
    manifest_path: str = str(MANIFEST_PATH),
    mode: str = "both",
    workers: int = 1,
    batch_tracks: int = 64,
    loader_workers: int = 8,
) -> pd.DataFrame:
    """
    mode="both"          — discogs-effnet + features → features.parquet  (default)
    mode="discogs-only"  — discogs-effnet only       → embeddings.parquet
    mode="librosa-only"  — features only             → librosa_features.parquet (no embeddings needed)
    workers              — parallel processes for librosa extraction (default 1 = sequential)
                           discogs-only always runs sequentially.
    Extract modes resumable: re-running skips already-processed track_ids.
    """
    assert mode in ("both", "discogs-only", "librosa-only"), f"Unknown mode: {mode}"
    out_path = (
        EMBEDDINGS_PATH
        if mode == "discogs-only"
        else LIBROSA_FEATURES_PATH
        if mode == "librosa-only"
        else FEATURES_PATH
    )
    use_parallel = workers > 1 and mode != "discogs-only"
    log.info("Mode: %s  →  %s  (workers=%d)", mode, out_path, workers)

    manifest = pd.read_csv(manifest_path)

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

    if mode == "librosa-only":
        embedder = None
    elif mode == "discogs-only":
        embedder = DiscogsEmbedderGPU(batch_tracks=batch_tracks, loader_workers=loader_workers)
    else:
        embedder = DiscogsEmbedder()

    all_rows = to_process.to_dict("records")
    total = len(all_rows)
    skipped = 0
    CHECKPOINT_EVERY = max(10, total // 100)
    rows: list[dict] = []
    t_start = time.time()
    last_ckpt_i = 0

    # ── librosa-only parallel ──────────────────────────────────────────────────
    if mode == "librosa-only" and use_parallel:
        log.info("Parallel librosa extraction: %d workers", workers)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            future_to_row = {
                pool.submit(_librosa_worker, (row["track_id"], row["local_path"])): row
                for row in all_rows
            }
            for i, future in enumerate(as_completed(future_to_row), start=1):
                row = future_to_row[future]
                try:
                    track_id, feats = future.result()
                except Exception as e:
                    log.warning("  worker error %s: %s", row.get("local_path"), e)
                    skipped += 1
                else:
                    if feats is None:
                        log.warning(
                            "  Skipping — feature extraction failed: %s", row.get("track_id")
                        )
                        skipped += 1
                    else:
                        rows.append({"track_id": track_id, **feats})
                if i - last_ckpt_i >= CHECKPOINT_EVERY or i >= total:
                    existing = _do_checkpoint(existing, rows, out_path, i, total, t_start, skipped)
                    last_ckpt_i = i

    # ── both parallel: discogs sequential (GPU) → librosa parallel (CPU) ───────
    elif mode == "both" and use_parallel:
        log.info("Two-pass parallel: discogs sequential → librosa %d workers", workers)

        # Pass 1: collect all discogs embeddings sequentially
        embed_rows: list[dict] = []
        for i, row in enumerate(all_rows, start=1):
            log.info(
                "[%d/%d discogs] %s - %s",
                i,
                total,
                row.get("artist", ""),
                row.get("track_name", ""),
            )
            path = row["local_path"]
            try:
                mf = MutaFile(path)
                if mf is None or mf.info.length <= 0:
                    raise ValueError("empty or unreadable")
            except Exception as e:
                log.warning("  Skipping corrupt %s: %s", path, e)
                skipped += 1
                continue
            emb = embedder.embed(path)
            if emb is None:
                log.warning("  Skipping — discogs embed failed")
                skipped += 1
            else:
                embed_rows.append({**row, "embedding": emb.tolist()})

        log.info(
            "Discogs done: %d embeddings (%.0f min). Starting parallel librosa...",
            len(embed_rows),
            (time.time() - t_start) / 60,
        )

        # Pass 2: parallel librosa on embedded tracks
        t_librosa = time.time()
        total_emb = len(embed_rows)
        last_ckpt_i = 0
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            future_to_row = {
                pool.submit(_librosa_worker, (row["track_id"], row["local_path"])): row
                for row in embed_rows
            }
            for i, future in enumerate(as_completed(future_to_row), start=1):
                row = future_to_row[future]
                try:
                    track_id, feats = future.result()
                except Exception as e:
                    log.warning("  librosa worker error %s: %s", row.get("local_path"), e)
                    skipped += 1
                else:
                    if feats is None:
                        skipped += 1
                    else:
                        rows.append({"track_id": track_id, "embedding": row["embedding"], **feats})
                if i - last_ckpt_i >= CHECKPOINT_EVERY or i >= total_emb:
                    existing = _do_checkpoint(
                        existing, rows, out_path, i, total_emb, t_librosa, skipped
                    )
                    last_ckpt_i = i

    # ── discogs-only: batched GPU inference ───────────────────────────────────
    elif mode == "discogs-only":
        log.info("Batched GPU discogs extraction (batch_tracks=%d)", embedder.batch_tracks)
        bt = embedder.batch_tracks
        for batch_start in range(0, total, bt):
            batch = all_rows[batch_start : batch_start + bt]
            i = batch_start + len(batch)

            valid: list[tuple[dict, str]] = []
            for r in batch:
                path = r["local_path"]
                try:
                    mf = MutaFile(path)
                    if mf is None or mf.info.length <= 0:
                        raise ValueError("empty")
                    valid.append((r, path))
                except Exception as e:
                    log.warning("  Skipping corrupt %s: %s", path, e)
                    skipped += 1

            if valid:
                embs = embedder.embed_batch([p for _, p in valid])
                for (r, path), emb in zip(valid, embs):
                    if emb is None:
                        log.warning("  discogs embed failed: %s", path)
                        skipped += 1
                    else:
                        rows.append({"track_id": r["track_id"], "embedding": emb.tolist()})

            log.info("[%d/%d] batch done", i, total)
            if i - last_ckpt_i >= CHECKPOINT_EVERY or i >= total:
                existing = _do_checkpoint(existing, rows, out_path, i, total, t_start, skipped)
                last_ckpt_i = i

    # ── sequential path (both mode, or workers=1) ─────────────────────────────
    else:
        extractor = LibrosaExtractor() if mode != "discogs-only" else None

        for i, row in enumerate(all_rows, start=1):
            log.info("[%d/%d] %s - %s", i, total, row.get("artist", ""), row.get("track_name", ""))
            path = row["local_path"]

            if mode != "librosa-only":
                try:
                    mf = MutaFile(path)
                    if mf is None or mf.info.length <= 0:
                        raise ValueError("empty or unreadable")
                except Exception as e:
                    log.warning("  Skipping corrupt %s: %s", path, e)
                    skipped += 1
                    if i - last_ckpt_i >= CHECKPOINT_EVERY or i >= total:
                        existing = _do_checkpoint(
                            existing, rows, out_path, i, total, t_start, skipped
                        )
                        last_ckpt_i = i
                    continue

            if mode == "librosa-only":
                feats = extractor.extract(path)
                if feats is None:
                    log.warning("  Skipping — feature extraction failed")
                    skipped += 1
                else:
                    rows.append({"track_id": row["track_id"], **feats})

            elif mode == "discogs-only":
                emb = embedder.embed(path)
                if emb is None:
                    log.warning("  Skipping — discogs embed failed")
                    skipped += 1
                else:
                    rows.append({"track_id": row["track_id"], "embedding": emb.tolist()})

            else:  # both, sequential
                emb = embedder.embed(path)
                if emb is None:
                    log.warning("  Skipping — discogs embed failed")
                    skipped += 1
                else:
                    feats = extractor.extract(path)
                    if feats is None:
                        log.warning("  Skipping — feature extraction failed")
                        skipped += 1
                    else:
                        rows.append(
                            {"track_id": row["track_id"], "embedding": emb.tolist(), **feats}
                        )

            if i - last_ckpt_i >= CHECKPOINT_EVERY or i >= total:
                existing = _do_checkpoint(existing, rows, out_path, i, total, t_start, skipped)
                last_ckpt_i = i

    elapsed_total = time.time() - t_start
    log.info(
        "Done. %d tracks in %s (%.0f min, %d skipped)",
        len(existing),
        out_path,
        elapsed_total / 60,
        skipped,
    )
    return existing


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract discogs-effnet embeddings and features.")
    parser.add_argument(
        "--mode",
        choices=["both", "discogs-only", "librosa-only"],
        default="both",
        help=(
            "both (default): discogs-effnet + features → features.parquet | "
            "discogs-only: discogs-effnet → embeddings.parquet (GPU instance) | "
            "librosa-only: features → librosa_features.parquet (any machine with audio)"
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
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help=(
            "Parallel worker processes for librosa extraction "
            "(default: all CPU cores). Ignored for discogs-only and merge modes."
        ),
    )
    parser.add_argument(
        "--batch-tracks",
        type=int,
        default=64,
        help="Tracks per GPU batch for discogs-only mode (default: 64).",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=8,
        help="Parallel threads for audio loading within each batch (default: 8).",
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

    build_features(
        manifest_path=manifest_path,
        mode=args.mode,
        workers=args.workers,
        batch_tracks=args.batch_tracks,
        loader_workers=args.loader_workers,
    )
