"""Sweep KEY_WEIGHT per genre and report the planned clash share against REAL_CLASH_SHARE.

    python scripts/diag/key_weight_sweep.py            # all genres, weights below
Prints one line per (genre, weight) and the best weight per genre. Writes nothing.
"""

import logging
import sys

import numpy as np

sys.path.insert(0, ".")
logging.disable(logging.WARNING)
from src.models import key_rules  # noqa: E402
from src.models.predict_model import plan_mix  # noqa: E402

RANGES = {
    "melodic house": (118, 126),
    "trance": (128, 140),
    "techno": (128, 140),
    "tech house": (122, 130),
    "drum and base": (84, 88),
    "afro house": (118, 125),
}
WEIGHTS = [-0.8, -0.4, -0.2, 0.0, 0.2, 0.4, 0.8]
CURVES = ("build", "peak", "wave")
N_TRACKS = 10


def clash_share(genre: str, weight: float) -> tuple[float, int]:
    key_rules.KEY_WEIGHT[genre] = weight
    dist = []
    for curve in CURVES:
        plan = plan_mix(genre, RANGES[genre], n_tracks=N_TRACKS, curve=curve)
        dist += [t["cam_dist"] for t in plan["tracks"][1:]]
    return float(np.mean(key_rules.is_clash(dist))), len(dist)


def main():
    best = {}
    for genre in RANGES:
        target = key_rules.REAL_CLASH_SHARE[genre]
        rows = []
        for w in WEIGHTS:
            share, n = clash_share(genre, w)
            rows.append((abs(share - target), w, share))
            print(
                f"{genre:13s} weight {w:+.1f}  clash share {share * 100:4.0f}%  (real {target * 100:.0f}%, n={n})",
                flush=True,
            )
        _, w, share = min(rows)
        best[genre] = (w, share)
        print(
            f"  -> best for {genre}: weight {w:+.1f} gives {share * 100:.0f}% vs real {target * 100:.0f}%",
            flush=True,
        )
    print("BEST", {g: w for g, (w, _) in best.items()})


if __name__ == "__main__":
    main()
