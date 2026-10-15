"""Track features and embeddings for a corpus with full tracks (Raveform, the DJ corpus).

One row per track: the whole-track discogs-effnet mean, the librosa features with the same names and
scales as `features.parquet` (raw onset, see CLAUDE.md), key through `normalise_key()`, and the BPM
joined from the corpus's `tempos.csv` (the mixer's kick autocorrelation; DeepRhythm is not run).
Every effnet vector is kept as `<embeddings_dir>/<track_id>.npy`, float16, one 1,280-vector per second
of audio, so any span can be pooled later without running the model again. Plan and numbers:
EMBEDDINGS.md; the decisions on old against new data: DATASET_STATE.md.

    python -m src.features.track_features \
        --tracks raveform=data/djdata/raveform/tracks --tempos raveform=data/djdata/raveform/out/tempos.csv \
        --out data/djdata/raveform/out/track_features.parquet --embeddings data/djdata/raveform/embeddings \
        --workers 6 [--only ids.txt]

One run per corpus, one table per corpus, like the other tables under `data/djdata/<corpus>/out/`.

Resumable: a track already in the output parquet is skipped.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
import multiprocessing
from pathlib import Path
import time

import librosa
import numpy as np
import pandas as pd
import pyloudnorm as pyln

from src.features.build_features import DISCOGS_MODEL_PATH, _essentia_key

log = logging.getLogger(__name__)

SR_EMBED, SR_FEAT = 16_000, 22_050
CHECKPOINT_EVERY = 50
AUDIO_SUFFIXES = (".m4a", ".mp3", ".webm", ".opus", ".wav", ".flac")

_model = None


def embed_track(y16: np.ndarray) -> np.ndarray:
    """(n_vectors, 1280) float16 from 16 kHz mono. The model loads once per process."""
    global _model
    if _model is None:
        import essentia.standard as es

        _model = es.TensorflowPredictEffnetDiscogs(
            graphFilename=str(DISCOGS_MODEL_PATH), output="PartitionedCall:1"
        )
    return np.asarray(_model(y16.astype(np.float32)), dtype=np.float16)


def librosa_features(y22: np.ndarray) -> dict:
    """Same columns and scales as the old table (`LibrosaExtractor.extract`), without the BPM."""
    lufs = float(pyln.Meter(SR_FEAT).integrated_loudness(np.stack([y22, y22], axis=1)))
    rms = librosa.feature.rms(y=y22)[0]
    mfcc = librosa.feature.mfcc(y=y22, sr=SR_FEAT, n_mfcc=13).mean(axis=1)
    return {
        "key": _essentia_key(y22, SR_FEAT),
        "loudness_lufs": lufs if np.isfinite(lufs) else -70.0,
        "energy_mean": float(rms.mean()),
        "energy_std": float(rms.std()),
        "spectral_centroid": float(librosa.feature.spectral_centroid(y=y22, sr=SR_FEAT)[0].mean()),
        "onset_strength": float(librosa.onset.onset_strength(y=y22, sr=SR_FEAT).mean()),
        **{f"mfcc_{i}": float(v) for i, v in enumerate(mfcc)},
    }


def one_track(job: dict) -> dict:
    """job: {track_id, corpus, path, embeddings_dir}. Decodes once, embeds, saves the vectors, measures."""
    y16, _ = librosa.load(job["path"], sr=SR_EMBED, mono=True)
    vectors = embed_track(y16)
    np.save(Path(job["embeddings_dir"]) / f"{job['track_id']}.npy", vectors)
    y22 = librosa.resample(y16, orig_sr=SR_EMBED, target_sr=SR_FEAT)
    return {
        "track_id": job["track_id"],
        "corpus": job["corpus"],
        "minutes": round(len(y16) / SR_EMBED / 60, 2),
        "n_vectors": int(vectors.shape[0]),
        "embedding": vectors.astype(np.float32).mean(axis=0).tolist(),
        **librosa_features(y22),
    }


def manifest(tracks_dirs: dict[str, Path]) -> pd.DataFrame:
    """One row per audio file: track_id (the file stem), corpus, path."""
    rows = [
        {"track_id": p.stem, "corpus": corpus, "path": str(p)}
        for corpus, d in tracks_dirs.items()
        for p in sorted(Path(d).iterdir())
        if p.suffix in AUDIO_SUFFIXES
    ]
    return pd.DataFrame(rows, columns=["track_id", "corpus", "path"])


def join_bpm(table: pd.DataFrame, tempos: dict[str, Path]) -> pd.DataFrame:
    """bpm from each corpus's tempos.csv; bpm_source says where it came from, None when absent."""
    parts = []
    for corpus, path in tempos.items():
        t = pd.read_csv(path)[["track_id", "bpm"]]
        t["corpus"] = corpus
        parts.append(t)
    bpm = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["track_id", "corpus", "bpm"])
    )
    out = table.drop(columns=[c for c in ("bpm", "bpm_source") if c in table]).merge(
        bpm, on=["track_id", "corpus"], how="left"
    )
    out["bpm_source"] = np.where(out["bpm"].notna(), "measured", None)
    return out


def run(
    tracks_dirs: dict[str, Path],
    tempos: dict[str, Path],
    out: Path,
    embeddings_dir: Path,
    workers: int = 1,
    only: set[str] | None = None,
) -> pd.DataFrame:
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    man = manifest(tracks_dirs)
    if only is not None:
        man = man[man.track_id.isin(only)]
    existing = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    done = set(existing["track_id"]) if len(existing) else set()
    todo = man[~man.track_id.isin(done)]
    log.info(
        "%d tracks in manifest, %d done, %d to do, %d workers",
        len(man),
        len(done),
        len(todo),
        workers,
    )
    jobs = [{**r, "embeddings_dir": str(embeddings_dir)} for r in todo.to_dict("records")]

    rows, t0 = [], time.time()

    def checkpoint():
        nonlocal existing, rows
        if rows:
            existing = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            existing.to_parquet(out, index=False)
            rows = []

    results = (
        map(one_track, jobs)
        if workers == 1
        # spawn, not fork: the parent has imported essentia's TensorFlow runtime, and forked
        # children deadlock on their first model call (both runs hung for 15 min, 2026-10-07)
        else ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")).map(
            one_track, jobs, chunksize=2
        )
    )
    for i, row in enumerate(results, 1):
        rows.append(row)
        if i % CHECKPOINT_EVERY == 0:
            checkpoint()
            log.info("%d / %d  %.1f s per track", i, len(jobs), (time.time() - t0) / i)
    checkpoint()
    if len(existing):
        existing = join_bpm(existing, tempos)
        existing.to_parquet(out, index=False)
    log.info("done: %d rows in %s, %.0f s", len(existing), out, time.time() - t0)
    return existing


def _pairs(values: list[str]) -> dict[str, Path]:
    return {v.split("=", 1)[0]: Path(v.split("=", 1)[1]) for v in values or []}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tracks", action="append", required=True, help="corpus=dir, repeatable")
    ap.add_argument("--tempos", action="append", default=[], help="corpus=tempos.csv, repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only", default=None, help="file with one track_id per line")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    only = set(Path(a.only).read_text().split()) if a.only else None
    run(_pairs(a.tracks), _pairs(a.tempos), Path(a.out), Path(a.embeddings), a.workers, only)
