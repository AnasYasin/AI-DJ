"""
Feature extraction pipeline.

For each MP3 in preview_manifest.csv, extracts two sets of features:

  1. MERT embedding (768-dim)
     Model: m-a-p/MERT-v1-95M
     What it captures: deep music-specific representations trained via masked
     acoustic modelling on a large music corpus. Unlike CLAP (general audio),
     MERT is trained exclusively on music and captures sub-genre timbral
     differences, harmonic structure, and rhythmic patterns within a genre.
     Output: mean-pooled last hidden state → 768-dim vector per track.

  2. Librosa features
     Numeric properties directly computed from the audio signal:
       bpm              — tempo in beats per minute
       key              — detected root note + major/minor (Krumhansl-Schmuckler)
       loudness_lufs    — integrated loudness (ITU-R BS.1770, same as Spotify)
       energy_mean/std  — RMS amplitude mean and dynamics
       spectral_centroid — frequency "brightness" (high = bright/harsh, low = warm)
       onset_strength   — how punchy note attacks are (danceability proxy)
       mfcc_0..12       — timbre fingerprint (13 Mel-Frequency Cepstral Coefficients)

Output: data/processed/features.parquet
  Columns: track_id, embedding (768-dim list), bpm, key, loudness_lufs,
           energy_mean, energy_std, spectral_centroid, onset_strength, mfcc_0..12

Idempotent: re-running appends only new tracks, skips already-processed ones.
"""
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

MANIFEST_PATH   = Path("data/raw/preview_manifest.csv")
FEATURES_PATH   = Path("data/processed/features.parquet")
MERT_MODEL_NAME = "m-a-p/MERT-v1-95M"
LOAD_TIMEOUT_S  = 30   # max seconds to spend loading a single audio file


def _load_audio(path: str, sr: int) -> tuple:
    """
    Load audio with a hard timeout. Raises TimeoutError if librosa.load
    hangs on a corrupt file (e.g. illegal MP3 headers causing infinite resync).
    """
    def _handler(signum, frame):
        raise TimeoutError(f"librosa.load timed out after {LOAD_TIMEOUT_S}s: {path}")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(LOAD_TIMEOUT_S)
    try:
        return librosa.load(path, sr=sr, mono=True)
    finally:
        signal.alarm(0)   # cancel alarm

# Maps chroma index (0=C, 1=C#, ..., 11=B) to note name
_CHROMA_TO_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles (major and minor) — 12 values each,
# starting from C. These encode how characteristic each pitch class is for
# that key, derived from music-theoretic and listener perception studies.
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                       2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                       2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _detect_key(chroma_mean: np.ndarray) -> str:
    """
    Krumhansl-Schmuckler algorithm: correlate the 12-element mean chroma
    vector against all 24 key profiles (12 major + 12 minor) and return
    the best-matching key as e.g. 'Am' or 'D'.

    Each profile is circularly shifted to represent every root note.
    The key with the highest Pearson correlation wins.
    """
    best_score = -np.inf
    best_key = "C"
    for root in range(12):
        # Rotate profile so root note aligns with position 0 (= C slot)
        major_profile = np.roll(_KS_MAJOR, root)
        minor_profile = np.roll(_KS_MINOR, root)
        score_major = np.corrcoef(chroma_mean, major_profile)[0, 1]
        score_minor = np.corrcoef(chroma_mean, minor_profile)[0, 1]
        if score_major > best_score:
            best_score = score_major
            best_key = _CHROMA_TO_NOTE[root]          # e.g. "D"
        if score_minor > best_score:
            best_score = score_minor
            best_key = _CHROMA_TO_NOTE[root] + "m"    # e.g. "Dm"
    return best_key


# ── MERT Embedder ──────────────────────────────────────────────────────────────

class MERTEmbedder:
    """
    Wraps the MERT (Music undERstanding model with large-scale self-supervised
    Training) from HuggingFace. Load once, call embed() per track.

    Why MERT over CLAP:
      - CLAP: general audio-language model, poor sub-genre resolution — tested
        and found zero correlation with actual DJ mix order in this dataset
      - MERT: trained exclusively on music via masked acoustic modelling,
        captures fine-grained timbral/harmonic differences within a genre

    Embedding: mean-pool the last hidden state across time → 768-dim vector.
    """

    # MERT processor specifies 24kHz
    SAMPLE_RATE = 24_000

    def __init__(self, model_name: str = MERT_MODEL_NAME):
        log.info("Loading MERT model: %s", model_name)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        log.info("MERT model loaded.")

    def embed(self, audio_path: str) -> np.ndarray | None:
        """
        Load audio file, resample to 24kHz, return 768-dim embedding.

        Steps:
          1. librosa.load resamples to 24kHz mono float array
          2. Wav2Vec2FeatureExtractor normalises and pads to model input
          3. model() runs the transformer encoder, outputs hidden states
          4. Mean-pool last_hidden_state over time: (1, T, 768) → (768,)

        Returns None if the file cannot be read (logs a warning).
        """
        try:
            audio, _ = _load_audio(audio_path, self.SAMPLE_RATE)
            inputs = self.processor(
                audio, sampling_rate=self.SAMPLE_RATE, return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            # last_hidden_state: (1, time_steps, 768) → mean over time → (768,)
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        except Exception as e:
            log.warning("MERT embed failed for %s: %s", audio_path, e)
            return None


# ── Librosa Feature Extractor ──────────────────────────────────────────────────

class LibrosaExtractor:
    """
    Extracts numeric audio features using librosa.

    Uses 22050Hz (librosa default) — lower than CLAP's 48kHz but sufficient
    for tempo/key/energy analysis and much faster to process.
    """

    SAMPLE_RATE = 22_050

    def extract(self, audio_path: str) -> dict | None:
        """
        Load audio and compute all features. Returns a flat dict of floats/strings.
        Returns None if the file cannot be read.

        Feature explanations:
          bpm              tempo detection via beat tracking
          key              dominant chroma bin → root note name
          loudness_lufs    integrated loudness (LUFS) — ITU-R BS.1770 standard,
                           perceptually weighted. This is what Spotify/YouTube use
                           for normalisation. More accurate than RMS for perceived
                           loudness. Negative values: -14 LUFS = Spotify target,
                           -9 LUFS = loud master, -23 LUFS = very quiet.
          energy_mean      mean RMS amplitude — raw signal energy (kept alongside
                           LUFS as a fast proxy used in delta features)
          energy_std       std of RMS → how dynamic the track is
          spectral_centroid mean frequency of energy distribution → brightness
          onset_strength   mean onset envelope → how percussive/punchy
          mfcc_0..12       13 MFCCs summarise the timbre (texture) of the sound
        """
        try:
            y, sr = _load_audio(audio_path, self.SAMPLE_RATE)

            # ── Tempo ──────────────────────────────────────────────────────────
            # start_bpm=120 biases the tracker toward DJ tempos (avoids half-time
            # detection on tracks with strong half-note pulses).
            # After tracking, snap into 80–180 BPM: double if too slow (half-time
            # detection), halve if too fast (double-time detection).
            bpm, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120.0)
            bpm = float(bpm)
            while bpm < 80:
                bpm *= 2
            while bpm > 180:
                bpm /= 2

            # ── Key ────────────────────────────────────────────────────────────
            # Krumhansl-Schmuckler: correlate mean chroma against 24 key profiles.
            # Returns e.g. "Am" or "D" (major has no suffix, minor has "m").
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)        # (12, n_frames)
            key = _detect_key(chroma.mean(axis=1))

            # ── Loudness (LUFS) ─────────────────────────────────────────────────
            # pyloudnorm implements ITU-R BS.1770-4. Needs stereo input (N, 2).
            # We load mono then duplicate the channel — result is identical to
            # true mono LUFS measurement (BS.1770 sums channels before K-weighting).
            y_stereo = np.stack([y, y], axis=1)                     # (n_samples, 2)
            meter = pyln.Meter(sr)
            lufs = meter.integrated_loudness(y_stereo)
            # pyln returns -inf for silence; clamp to a floor of -70 LUFS
            if not np.isfinite(lufs):
                lufs = -70.0

            # ── Energy ─────────────────────────────────────────────────────────
            # rms returns (1, n_frames); [0] gives the 1-D array
            rms = librosa.feature.rms(y=y)[0]                       # (n_frames,)

            # ── Brightness ─────────────────────────────────────────────────────
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

            # ── Danceability proxy ─────────────────────────────────────────────
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)

            # ── Timbre (MFCC) ──────────────────────────────────────────────────
            # mfcc returns (n_mfcc, n_frames); mean over time → (13,)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)     # (13, n_frames)
            mfcc_means = mfcc.mean(axis=1)                           # (13,)

            return {
                "bpm": float(bpm),
                "key": key,
                "loudness_lufs": float(lufs),
                "energy_mean": float(rms.mean()),
                "energy_std": float(rms.std()),
                "spectral_centroid": float(centroid.mean()),
                "onset_strength": float(onset_env.mean()),
                **{f"mfcc_{i}": float(v) for i, v in enumerate(mfcc_means)},
            }

        except Exception as e:
            log.warning("librosa extraction failed for %s: %s", audio_path, e)
            return None


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_features(manifest_path: str = str(MANIFEST_PATH)) -> pd.DataFrame:
    """
    Main entry point. Reads the preview manifest, extracts features for every
    track that has a local audio file, appends to features.parquet.

    Skips tracks already in features.parquet (idempotent — safe to re-run
    after adding more previews without reprocessing existing ones).

    Returns the full features DataFrame.
    """
    manifest = pd.read_csv(manifest_path)

    # Only process tracks we actually have audio for
    to_process = manifest[manifest["source"] != "not_found"].copy()
    log.info("Manifest: %d tracks total, %d have audio", len(manifest), len(to_process))

    # Load existing results and skip already-processed tracks
    if FEATURES_PATH.exists():
        existing = pd.read_parquet(FEATURES_PATH)
        done = set(existing["track_id"]) if "track_id" in existing.columns else set()
        to_process = to_process[~to_process["track_id"].isin(done)]
        log.info("Already processed: %d. Remaining: %d", len(done), len(to_process))
    else:
        existing = pd.DataFrame()
        FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    if to_process.empty:
        log.info("Nothing new to process.")
        return existing

    # Load models once — expensive, do not reload inside the loop
    embedder = MERTEmbedder()
    extractor = LibrosaExtractor()

    rows = []
    total = len(to_process)
    skipped = 0
    # Checkpoint every ~1% of total — gives ~100 saves regardless of dataset size.
    # Floor of 10 so small test runs still get at least one mid-run save.
    CHECKPOINT_EVERY = max(10, total // 100)

    t_start = time.time()

    for i, (_, row) in enumerate(to_process.iterrows(), 1):
        path = row["local_path"]
        log.info("[%d/%d] %s - %s", i, total, row["artist"], row["track_name"])

        # Fast corruption check before librosa (which can hang on bad files)
        try:
            mf = MutaFile(path)
            if mf is None or mf.info.length <= 0:
                raise ValueError("empty or unreadable")
        except Exception as e:
            log.warning("  Skipping corrupt file %s: %s", path, e)
            skipped += 1
            continue

        emb_vec = embedder.embed(path)
        librosa_feats = extractor.extract(path)

        # Skip this track if either extractor failed
        if emb_vec is None or librosa_feats is None:
            log.warning("  Skipping — extraction failed")
            skipped += 1
            continue

        rows.append({
            "track_id":  row["track_id"],
            "embedding": emb_vec.tolist(),   # 768-dim MERT vector
            **librosa_feats,
        })

        # ── ETA + checkpoint every CHECKPOINT_EVERY tracks ─────────────────
        if i % CHECKPOINT_EVERY == 0 or i == total:
            elapsed = time.time() - t_start
            rate = i / elapsed                          # tracks/sec
            remaining = (total - i) / rate if rate > 0 else 0
            hrs, mins = divmod(int(remaining), 3600)
            mins //= 60
            log.info(
                "--- checkpoint %d/%d | %.2f sec/track | ETA %dh %02dm | skipped %d ---",
                i, total, elapsed / i, hrs, mins, skipped,
            )
            # Flush to disk — if the process dies, we restart from here
            new_df = pd.DataFrame(rows)
            all_so_far = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
            all_so_far.to_parquet(FEATURES_PATH, index=False)
            log.info("  Checkpoint saved → %s (%d tracks total)", FEATURES_PATH, len(all_so_far))

    new_df = pd.DataFrame(rows)
    all_features = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df

    all_features.to_parquet(FEATURES_PATH, index=False)
    n_lib = len(all_features.columns) - 2  # track_id + embedding
    elapsed_total = time.time() - t_start
    log.info("Saved %d tracks → %s  (%.0f min total, %d skipped)",
             len(all_features), FEATURES_PATH, elapsed_total / 60, skipped)
    log.info("Columns: %d  (embedding[768] + %d librosa)", len(all_features.columns), n_lib)
    return all_features


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    _fmt = "%(asctime)s %(levelname)s %(message)s"
    _log_path = Path("logs/build_features.log")
    _log_path.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=_fmt,
        handlers=[
            logging.StreamHandler(),                          # stdout (live)
            logging.FileHandler(_log_path, encoding="utf-8"), # persistent log
        ],
    )
    log.info("Logging to %s", _log_path)
    build_features()
