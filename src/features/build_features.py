"""
Feature extraction pipeline.

For each MP3 in preview_manifest.csv, extracts two sets of features:

  1. CLAP embedding (512-dim)
     Model: laion/clap-htsat-unfused
     What it captures: the semantic "meaning" of the music — genre, mood, energy,
     texture — in a single vector. Trained on 500K (audio, text) pairs so it
     understands what music sounds like AND what words describe it.

  2. Librosa features
     Numeric properties directly computed from the audio signal:
       bpm              — tempo in beats per minute
       key              — detected root note (C, C#, D, ... B)
       energy_mean/std  — loudness level and dynamics
       spectral_centroid — frequency "brightness" (high = bright/harsh, low = warm)
       onset_strength   — how punchy note attacks are (danceability proxy)
       mfcc_0..12       — timbre fingerprint (13 Mel-Frequency Cepstral Coefficients)

Output: data/processed/features.parquet
  Columns: track_id, artist, track_name, clap_0..511, bpm, key,
           energy_mean, energy_std, spectral_centroid, onset_strength, mfcc_0..12

Idempotent: re-running appends only new tracks, skips already-processed ones.
"""
import logging
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import ClapModel, ClapProcessor

log = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/raw/preview_manifest.csv")
FEATURES_PATH = Path("data/processed/features.parquet")
CLAP_MODEL_NAME = "laion/clap-htsat-unfused"

# Maps chroma index (0=C, 1=C#, ..., 11=B) to note name
_CHROMA_TO_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ── CLAP Embedder ──────────────────────────────────────────────────────────────

class CLAPEmbedder:
    """
    Wraps the CLAP HuggingFace model.
    Load once, call embed() for each track — do not reload per track.

    Why CLAP over other models:
      - VGGish: 128-dim, trained on YouTube sounds (dogs, cars) — not music-semantic
      - Open-L3: good audio embeddings but no language branch
      - CLAP: 512-dim, trained on music+text pairs, understands genre/mood/energy
    """

    # CLAP requires audio at 48kHz — different from librosa's default 22050Hz
    SAMPLE_RATE = 48_000

    def __init__(self, model_name: str = CLAP_MODEL_NAME):
        log.info("Loading CLAP model: %s (this takes ~10s and ~2GB RAM)", model_name)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name)
        self.model.eval()  # disable dropout, use running batch-norm stats
        log.info("CLAP model loaded.")

    def embed(self, audio_path: str) -> np.ndarray | None:
        """
        Load audio file, resample to 48kHz, return 512-dim embedding.

        Steps:
          1. librosa.load resamples any MP3/M4A/WAV to 48kHz mono float array
          2. ClapProcessor normalises the array and converts to model input tensors
          3. model.get_audio_features() runs the audio encoder (no text encoder)
          4. squeeze() removes the batch dimension: (1, 512) → (512,)

        Returns None if the file cannot be read (logs a warning).
        """
        try:
            # Load at CLAP's required sample rate
            audio, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)

            inputs = self.processor(
                audios=audio,
                sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt",
            )

            with torch.no_grad():  # no gradient tracking needed — we are not training CLAP
                embedding = self.model.get_audio_features(**inputs)

            return embedding.squeeze().numpy()  # shape: (512,)

        except Exception as e:
            log.warning("CLAP embed failed for %s: %s", audio_path, e)
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
          energy_mean      mean RMS amplitude → overall loudness
          energy_std       std of RMS → how dynamic the track is
          spectral_centroid mean frequency of energy distribution → brightness
          onset_strength   mean onset envelope → how percussive/punchy
          mfcc_0..12       13 MFCCs summarise the timbre (texture) of the sound
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)

            # ── Tempo ──────────────────────────────────────────────────────────
            # beat_track returns (bpm_float, beat_frame_indices)
            bpm, _ = librosa.beat.beat_track(y=y, sr=sr)

            # ── Key ────────────────────────────────────────────────────────────
            # chroma_cqt: 12 × n_frames matrix, each row = energy for one semitone
            # Mean over time → 12-element vector; argmax = most prominent note
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)        # (12, n_frames)
            key_idx = int(np.argmax(chroma.mean(axis=1)))           # 0-11
            key = _CHROMA_TO_NOTE[key_idx]

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
        done = set(existing["track_id"])
        to_process = to_process[~to_process["track_id"].isin(done)]
        log.info("Already processed: %d. Remaining: %d", len(done), len(to_process))
    else:
        existing = pd.DataFrame()
        FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)

    if to_process.empty:
        log.info("Nothing new to process.")
        return existing

    # Load models once — expensive, do not reload inside the loop
    embedder = CLAPEmbedder()
    extractor = LibrosaExtractor()

    rows = []
    total = len(to_process)

    for i, (_, row) in enumerate(to_process.iterrows(), 1):
        path = row["local_path"]
        log.info("[%d/%d] %s - %s", i, total, row["artist"], row["track_name"])

        clap_vec = embedder.embed(path)
        librosa_feats = extractor.extract(path)

        # Skip this track if either extractor failed
        if clap_vec is None or librosa_feats is None:
            log.warning("  Skipping — extraction failed")
            continue

        rows.append({
            "track_id":   row["track_id"],
            "artist":     row["artist"],
            "track_name": row["track_name"],
            # CLAP embedding: 512 columns named clap_0 … clap_511
            **{f"clap_{j}": float(v) for j, v in enumerate(clap_vec)},
            # Librosa features: bpm, key, energy_*, spectral_centroid, onset_strength, mfcc_0..12
            **librosa_feats,
        })

    new_df = pd.DataFrame(rows)
    all_features = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df

    all_features.to_parquet(FEATURES_PATH, index=False)
    log.info("Saved %d tracks → %s", len(all_features), FEATURES_PATH)
    log.info("Columns: %d  (512 CLAP + %d librosa)", len(all_features.columns),
             len(all_features.columns) - 514)  # 514 = track_id + artist + track_name + 511 clap cols
    return all_features


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_features()
