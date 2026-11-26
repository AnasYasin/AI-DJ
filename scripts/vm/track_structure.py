"""
Level curves and sections for every held track, one file per track, resumable.

    PYTHONPATH=. python scripts/vm/track_structure.py --workers 90 [--limit 20]

Per track under data/djdata/<source>/tracks/<ref>.*:
  curves/<ref>.npz     loudness, lows (< 150 Hz), mids, highs (> 2 kHz) in dB and onset strength on a
                       0.25 s grid (`curves` in scripts/experiments/seam_situation.py)
  segments/<ref>.json  beats, bars, downbeats (Beat This!), phrase offset, sections and per-bar energy
                       (`_segment_uncached` in src/data/audio_segmenter.py, what the mixer uses)
and one row per track in data/djdata/<source>/out/track_structure_log.csv (status, seconds, where the
downbeats came from). A track with both files is skipped, so rerunning continues.
"""

import argparse
import csv
import json
from multiprocessing import Pool
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one core per worker; the pool is the parallelism

import numpy as np  # noqa: E402

ROOT = Path("data/djdata")
SOURCES = ("raveform", "djs")


def one(job):
    src, path = job
    ref, t0 = path.stem, time.time()
    curve_file = ROOT / src / "curves" / f"{ref}.npz"
    segment_file = ROOT / src / "segments" / f"{ref}.json"
    downbeats = ""
    try:
        if not curve_file.exists():
            from scripts.experiments.seam_situation import curves

            tmp = curve_file.with_name(f"{ref}.tmp.npz")
            np.savez(tmp, **curves(path))
            tmp.rename(curve_file)
        if not segment_file.exists():
            import torch

            torch.set_num_threads(1)
            from src.data.audio_segmenter import _segment_uncached

            seg = _segment_uncached(path)
            downbeats = seg.get("downbeat_source", "")
            tmp = segment_file.with_name(f"{ref}.tmp.json")
            tmp.write_text(json.dumps(seg))
            tmp.rename(segment_file)
        return src, ref, "ok", round(time.time() - t0, 1), downbeats
    except Exception as e:  # a bad file must not stop the run
        return src, ref, f"failed: {type(e).__name__}: {e}", round(time.time() - t0, 1), downbeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, help="first N tracks per source, for a timing test")
    args = ap.parse_args()
    jobs = []
    for src in SOURCES:
        for d in ("curves", "segments", "out"):
            (ROOT / src / d).mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in (ROOT / src / "tracks").iterdir() if p.is_file())
        todo = [
            p
            for p in files
            if not (
                (ROOT / src / "curves" / f"{p.stem}.npz").exists()
                and (ROOT / src / "segments" / f"{p.stem}.json").exists()
            )
        ]
        print(f"{src}: {len(files)} tracks, {len(todo)} to do", flush=True)
        jobs += [(src, p) for p in todo[: args.limit]]
    logs = {
        src: csv.writer(open(ROOT / src / "out" / "track_structure_log.csv", "a", newline=""))
        for src in SOURCES
    }
    # Load both models once, alone, before the pool. The segmenter falls back silently when a model is
    # missing, and workers downloading one checkpoint at the same time corrupt it (2 of 8 on 2026-11-13).
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from deeprhythm import DeepRhythmPredictor; DeepRhythmPredictor();"
            "from beat_this.inference import Audio2Beats;"
            "Audio2Beats(checkpoint_path='final0', device='cpu', dbn=False)",
        ],
        check=True,
    )
    t0, failed = time.time(), 0
    # a worker is replaced after 25 tracks, so memory a track leaves behind cannot pile up
    with Pool(args.workers, maxtasksperchild=25) as pool:
        for n, (src, ref, status, seconds, downbeats) in enumerate(
            pool.imap_unordered(one, jobs), 1
        ):
            logs[src].writerow([ref, status, seconds, downbeats])
            failed += status != "ok"
            if n % 100 == 0 or n == len(jobs):
                print(f"{n} / {len(jobs)}  failed {failed}  {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
