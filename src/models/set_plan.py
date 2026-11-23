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

from src.data.dataset import dj_key, load_dj_types, load_group_types

MIN_ROW_SEAMS = 15  # a type backed by fewer seams in the group is left out of the plan


def type_counts(shares: pd.Series, n_seams: int) -> dict[str, int]:
    """Whole seams per type in the shares' proportions (largest remainder), so 10 % of 7 is 0 or 1."""
    exact = shares / shares.sum() * n_seams
    counts = np.floor(exact).astype(int)
    extra = (exact - counts).sort_values(ascending=False).index[: n_seams - counts.sum()]
    counts[extra] += 1
    return counts[counts > 0].to_dict()


def style_of(
    genre: str | None, group: str | None, dj: str | None
) -> tuple[str | None, str | None]:
    """(group, DJ key) a request mixes in the style of. A named DJ gives their own group in the genre
    and their own rows; a DJ nobody profiled gives (the group asked for, None)."""
    rows = load_dj_types()
    rows = rows[rows["dj_key"] == dj_key(dj)] if dj else rows.iloc[:0]
    if group:
        rows = rows[rows["group"] == group]
    elif genre:
        rows = rows[rows["group"].str.startswith(genre + " ")]
    if rows.empty:
        return group, None
    return rows.groupby("group")["seams"].sum().idxmax(), dj_key(dj)


def set_plan(group: str, n_tracks: int, dj: str | None = None, seed: int = 0) -> list[dict]:
    """The set's budget of seams, one dict each: `type`, `bass_free_bars`, `entry_section` (the
    section the group's incoming tracks are most often in when first heard), `play_min` drawn from
    the group's middle half, and the middle half itself for the overlap and for the bar the incoming
    track starts at (`overlap_low`, `overlap_high`, `enters_low`, `enters_high`).

    The plan advises: the planner gives each seam to the join it suits (`join_fit`), and the mixer
    picks the exact bar and length inside the ranges from the two tracks. The list order is random
    and means nothing until the planner has placed the seams.
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
        seam["entry_section"] = str(row["entry_section"])
        seam["play_min"] = round(float(rng.uniform(row["play_min_low"], row["play_min_high"])), 1)
        for name, column in (("overlap", "overlap_bars"), ("enters", "enters_bar")):
            seam[f"{name}_low"] = round(float(row[f"{column}_low"]), 1)
            seam[f"{name}_high"] = round(float(row[f"{column}_high"]), 1)
        plan.append(seam)
    return plan


def join_fit(
    structure: pd.DataFrame, seam: dict, tempo_gap_pct: np.ndarray, last_grid: float
) -> np.ndarray:
    """0 to 1 per candidate track: how well this seam suits the join from the last track to it.
    Half is `slot_fit`, half is the tempo gap between the two tracks (0 to 5 %): a cut scores
    higher the further apart they are, every other type the closer they are.

    Measured on 3,646 real seams (2026-11-23): the tempo gap is the one pair feature that differs
    by type. Cuts sit on a median gap of 1.63 %, centre swaps on 0.97 % and late swaps on 0.86 %,
    and inside a group a longer overlap goes with a smaller gap (rank correlation -0.16). Key
    distance and energy step do not differ by type, so they are not used. The signal is weak; the
    larger gain is that long blends land on pairs that drift less.

    A cut is 0 when either bar grid is too unsure for the mixer to cut on (`last_grid`, and the
    candidate's own `downbeat_confidence`).
    """
    close = 1 - np.clip(np.asarray(tempo_gap_pct) / 5.0, 0, 1)
    slot = slot_fit(structure, seam)
    if seam["type"] != "cut":
        return 0.5 * slot + 0.5 * close
    sure = (structure["downbeat_confidence"].fillna(1.0) >= 0.5).to_numpy() & (last_grid >= 0.5)
    return np.where(sure, 0.5 * slot + 0.5 * (1 - close), 0.0)


def slot_fit(structure: pd.DataFrame, seam: dict) -> np.ndarray:
    """0 to 1 per track of `structure`: can it carry the slot this seam opens. Half for room (it is
    long enough to start at the low end of the style's start range and play the planned time), half
    for the place (the range reaches into its intro when the group enters tracks in their intro, past
    its first drop when the group enters them in their body). A track with no segment file gets 0.5.

    ponytail: two yes-or-no checks from the section labels, and "drop" means the loud body on many
    techno tracks. Use the level curves at the planned bar if the ear says tracks start in the wrong place.
    """
    play_bars = seam["play_min"] * 60 / structure["bar_s"]
    room = seam["enters_low"] + play_bars <= structure["n_bars"]
    if seam["entry_section"] == "intro":
        place = structure["intro_bars"] >= seam["enters_low"]
    else:
        place = structure["first_drop_bar"] <= seam["enters_high"]
    fit = 0.5 * room + 0.5 * place
    return fit.where(structure["n_bars"].notna(), 0.5).to_numpy(dtype=float)


def demo():
    tracks = pd.DataFrame(
        {
            "n_bars": [200, 100, None],
            "downbeat_confidence": [0.9, 0.2, None],
            "bar_s": 2.0,
            "intro_bars": [16, 0, None],
            "first_drop_bar": [32, 0, None],
        }
    )
    seam = {"type": "tension", "enters_low": 8, "enters_high": 8, "play_min": 5.0}
    seam["entry_section"] = "intro"  # 150 bars from bar 8
    assert slot_fit(tracks, seam).tolist() == [1.0, 0.0, 0.5]
    assert slot_fit(tracks, seam | {"entry_section": "drop"}).tolist() == [0.5, 0.5, 0.5]
    gap = np.array([0.0, 0.0, 5.0])  # the first two tracks at the same tempo, the third 5 % off
    assert join_fit(tracks, seam, gap, last_grid=0.9).tolist() == [
        1.0,
        0.5,
        0.25,
    ]  # a blend: close
    cut = seam | {"type": "cut"}
    assert join_fit(tracks, cut, gap, last_grid=0.9).tolist() == [
        0.5,
        0.0,
        0.75,
    ]  # a cut: far apart
    assert join_fit(tracks, cut, gap, last_grid=0.2).tolist() == [
        0.0,
        0.0,
        0.0,
    ]  # unsure grid before
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
