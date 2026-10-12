"""The loudness curve of every record on the laptop, for the transition cards in the DJ data report.

Writes data/interim/card_music.parquet: one row per record with its id, corpus, BPM, crest (how peaky
it is: 95th percentile over the median of the 20 ms RMS curve) and the curve itself (float16, 20 ms
steps, normalised to its own loudest moment). The report picks, per transition type, two records and
the sections that suit the type from this file, so a build runs in seconds and never decodes audio.

    conda activate aidj && python notebooks/card_music.py      # about 5 minutes for 220 records
"""

from pathlib import Path

import librosa
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/interim/card_music.parquet"
SR, STEP_S = 11025, 0.02


def curve(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    hop = int(SR * STEP_S)
    n = len(y) // hop
    rms = np.sqrt(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1))
    return (rms / (rms.max() + 1e-9)).astype(np.float16)


def main():
    rows = []
    for corpus in ("raveform", "djs"):
        feats = pd.read_parquet(ROOT / f"data/djdata/{corpus}/out/track_features.parquet", columns=["track_id", "bpm"])
        bpm = feats.drop_duplicates("track_id").set_index("track_id").bpm
        for path in sorted((ROOT / f"data/djdata/{corpus}/tracks").iterdir()):
            if path.suffix == ".part":
                continue
            track_id = path.stem
            if track_id not in bpm.index or pd.isna(bpm[track_id]):
                continue
            c = curve(path)
            med = float(np.median(c.astype(np.float32)))
            rows.append(
                {
                    "track_id": track_id,
                    "corpus": corpus,
                    "bpm": float(bpm[track_id]),
                    "seconds": round(len(c) * STEP_S, 1),
                    "crest": round(float(np.percentile(c.astype(np.float32), 95)) / max(med, 1e-6), 2),
                    "curve": c,
                }
            )
            print(f"{len(rows):4d} {corpus:9s} {track_id} {rows[-1]['seconds']:6.0f} s crest {rows[-1]['crest']}", flush=True)
    pd.DataFrame(rows).to_parquet(OUT)
    print("wrote", OUT, len(rows), "records")


if __name__ == "__main__":
    main()
