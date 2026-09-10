"""Score key detectors against published keys (data/interim/key_ground_truth.csv).

Detectors: the catalog key (essentia on a 30 s preview), essentia KeyExtractor on the
full track with each profile, and, with --cnn, madmom's CNN key model (GiantSteps).
A reading counts as right when it equals the published key or its relative key
(Camelot distance <= 0.5); when two sources disagree, matching either counts.

    python scripts/diag/key_detector_compare.py [--cnn]
"""

import argparse
import logging
from pathlib import Path
import sys
import time

import librosa
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
logging.disable(logging.WARNING)
from src.audio.audio_mixer import SR, _to_mono, load_audio  # noqa: E402
from src.audio.key_shift import camelot_distance  # noqa: E402
from src.features.transition_labeler import normalise_key  # noqa: E402

PROFILES = ("edma", "edmm", "bgate", "braw", "temperley", "krumhansl")
CNN_NAMES = [f"{n} major" for n in "A A# B C C# D D# E F F# G G#".split()] + [
    f"{n} minor" for n in "A A# B C C# D D# E F F# G G#".split()
]


def essentia_keys(y22: np.ndarray) -> dict[str, str]:
    import essentia.standard as es

    out = {}
    for profile in PROFILES:
        note, scale, _ = es.KeyExtractor(sampleRate=22050.0, profileType=profile)(y22)
        out[f"essentia_{profile}"] = normalise_key(note, scale)
    return out


def cnn_key(path: str) -> str:
    from madmom.features.key import CNNKeyRecognitionProcessor

    probs = CNNKeyRecognitionProcessor()(path)
    note, scale = CNN_NAMES[int(np.argmax(probs))].split()
    return normalise_key(note, scale)


def right(reading: str, truth: str, second: str) -> bool | None:
    if truth.endswith("?"):  # mode unknown: compare the wheel position of both modes
        note = truth[:-1]
        return any(camelot_distance(reading, k) <= 0.5 for k in (note, note + "m"))
    return any(camelot_distance(reading, k) <= 0.5 for k in (truth, second) if isinstance(k, str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", action="store_true", help="also run madmom's CNN key model")
    args = ap.parse_args()
    truth = pd.read_csv("data/interim/key_ground_truth.csv")
    catalog = (
        pd.read_parquet("data/processed/features.parquet", columns=["track_id", "key"])
        .drop_duplicates("track_id")
        .set_index("track_id")["key"]
    )
    paths = {
        p.stem: str(p) for p in Path("data/external").rglob("*.*") if p.suffix in (".m4a", ".webm")
    }
    rows, seconds = [], {"catalog": 0.0, "essentia": 0.0, "cnn": 0.0}
    for r in truth.itertuples():
        y = load_audio(paths[r.track_id])
        t0 = time.time()
        y22 = librosa.resample(_to_mono(y), orig_sr=SR, target_sr=22050).astype(np.float32)
        readings = {"catalog": catalog[r.track_id], **essentia_keys(y22)}
        seconds["essentia"] += time.time() - t0
        if args.cnn:
            t0 = time.time()
            readings["cnn"] = cnn_key(paths[r.track_id])
            seconds["cnn"] += time.time() - t0
        rows.append({"name": r.name, "published": r.published_key, **readings})
        marks = " ".join(
            f"{k.replace('essentia_', '')}={v}{'✓' if right(v, r.published_key, r.second_key) else '✗'}"
            for k, v in readings.items()
        )
        print(f"{r.name[:38]:38s} published {r.published_key:4s} | {marks}", flush=True)
    df = pd.DataFrame(rows)
    print("\nAccuracy (exact or relative key), n =", len(df))
    for col in [c for c in df.columns if c not in ("name", "published")]:
        ok = [right(v, t, s) for v, t, s in zip(df[col], truth.published_key, truth.second_key)]
        print(f"  {col:20s} {sum(ok):2d}/{len(ok)}  ({100 * sum(ok) / len(ok):.0f} %)")
    print(
        f"\nTime per track: essentia (6 profiles) {seconds['essentia'] / len(df):.1f} s, cnn {seconds['cnn'] / len(df):.1f} s"
    )
    df.to_csv("data/interim/key_detector_compare.csv", index=False)


if __name__ == "__main__":
    main()
