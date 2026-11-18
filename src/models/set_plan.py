"""
The plan for a set from a style group: how many transitions of each type, roughly how each is done,
and for each incoming track a rough play time and cue-in.

    python -m src.models.set_plan "techno blend" 6 [--dj adambeyer] [--seed 0]

Reads `group_types.parquet` (or `dj_types.parquet` for a named DJ), written by `src.data.profiles`.
No real seam is copied: a seam was made for its two tracks. The plan carries counts and ranges, the
planner picks tracks that fit, and the mixer renders each type with its own recipe and its rules as
the check (`seam_plan` in `src.audio.audio_mixer.render_mix`).
"""

import argparse
import json

import numpy as np
import pandas as pd

from src.data.dataset import load_dj_types, load_group_types

MIN_ROW_SEAMS = 15  # a type backed by fewer seams in the group is left out of the plan
RANGES = ("overlap_bars", "enters_bar", "play_min")  # drawn between their 25th and 75th percentile


def type_counts(shares: pd.Series, n_seams: int) -> dict[str, int]:
    """Whole seams per type in the shares' proportions (largest remainder), so 10 % of 7 is 0 or 1."""
    exact = shares / shares.sum() * n_seams
    counts = np.floor(exact).astype(int)
    extra = (exact - counts).sort_values(ascending=False).index[: n_seams - counts.sum()]
    counts[extra] += 1
    return counts[counts > 0].to_dict()


def set_plan(group: str, n_tracks: int, dj: str | None = None, seed: int = 0) -> list[dict]:
    """One dict per seam, in set order: type, overlap_bars, bass_free_bars, enters_bar, play_min.

    The order of types is random: consecutive seams in real sets are independent of each other
    (TRANSITION_EXPERIMENTS.md, Test 3).
    """
    rows = load_group_types()
    rows = rows[rows["group"] == group]
    if dj:  # the DJ's own rows where they are deep enough, the group's for the other types
        own = load_dj_types()
        own = own[
            (own["group"] == group) & (own["dj_key"] == dj) & (own["seams"] >= MIN_ROW_SEAMS)
        ]
        rows = pd.concat([own, rows[~rows["type"].isin(own["type"])].assign(share=0.0)])
    if rows.empty:
        raise ValueError(f"no group {group!r}; see load_groups()")
    rows = rows[(rows["seams"] >= MIN_ROW_SEAMS) | (rows["share"] == 0)].set_index("type")
    rng = np.random.default_rng(seed)
    counts = type_counts(rows["share"], n_tracks - 1)
    order = rng.permutation([t for t, n in counts.items() for _ in range(n)])
    plan = []
    for kind in order:
        row = rows.loc[kind]
        seam = {"type": str(kind), "bass_free_bars": round(float(row["bass_free_bars"]), 1)}
        for name in RANGES:
            seam[name] = round(float(rng.uniform(row[f"{name}_low"], row[f"{name}_high"])), 1)
        plan.append(seam)
    return plan


def demo():
    shares = pd.Series({"centre swap": 0.54, "tension": 0.18, "end swap": 0.12, "cut": 0.07})
    # 7 seams: 4.15, 1.38, 0.92 and 0.54 exact, so the two spare seams go to end swap and cut
    assert type_counts(shares, 7) == {"centre swap": 4, "tension": 1, "end swap": 1, "cut": 1}
    assert sum(type_counts(shares, 5).values()) == 5


if __name__ == "__main__":
    demo()
    p = argparse.ArgumentParser()
    p.add_argument("group")
    p.add_argument("n_tracks", type=int)
    p.add_argument("--dj", default=None, help="dj_key, to plan from that DJ's own rows")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print(json.dumps(set_plan(a.group, a.n_tracks, a.dj, a.seed), indent=1))
