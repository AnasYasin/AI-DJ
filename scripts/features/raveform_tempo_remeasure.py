"""Re-measure tempo with the Raveform annotation BPM as the hint, for the records that missed it.
Same instrument as tempos.csv (audio_mixer._measure_tempo), different hint. Writes one row per record."""

from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path
import sys
import time

import librosa

sys.path.insert(0, ".")
from src.audio import audio_mixer as mixer

TRACKS = Path("data/djdata/raveform/tracks")


def one(row):
    tid, hint = row["track_id"], float(row["ann_bpm"])
    hits = list(TRACKS.glob(f"{tid}.*"))
    if not hits:
        return {"track_id": tid, "ann_bpm": hint, "remeasured_bpm": None, "note": "no audio"}
    y, _ = librosa.load(str(hits[0]), sr=mixer.SR, mono=True)
    m = mixer._measure_tempo(y, hint)
    return {
        "track_id": tid,
        "ann_bpm": hint,
        "remeasured_bpm": None if m is None else round(float(m), 3),
        "note": "",
    }


if __name__ == "__main__":
    rows = list(csv.DictReader(open(sys.argv[1])))
    out = Path(sys.argv[2])
    t0 = time.time()
    with open(out, "w", newline="") as f, ProcessPoolExecutor(16) as pool:
        w = csv.DictWriter(f, fieldnames=["track_id", "ann_bpm", "remeasured_bpm", "note"])
        w.writeheader()
        for i, r in enumerate(pool.map(one, rows, chunksize=4), 1):
            w.writerow(r)
            f.flush()
            if i % 100 == 0:
                print(i, round(time.time() - t0), "s", flush=True)
    print("done", len(rows), round(time.time() - t0), "s")
