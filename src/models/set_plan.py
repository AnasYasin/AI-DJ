"""
The plan for a set from a style group: how many transitions of each type, roughly how each is done,
and for each incoming track a rough play time and cue-in.

    python -m src.models.set_plan --genre techno --style blend cut --n 8
    python -m src.models.set_plan --dj "fred again" --n 6

Reads `group_types.parquet` (or `dj_types.parquet` for a named DJ), written by `src.data.profiles`.
No real seam is copied: a seam was made for its two tracks. The plan carries counts and ranges, the
planner picks tracks that fit, and the mixer renders each type with its own recipe and its rules as
the check (`seam_plan` in `src.audio.audio_mixer.render_mix`).
"""

import argparse
import json

import numpy as np
import pandas as pd

from src.data.dataset import dj_key, load_dj_types, load_group_types, load_groups

MIN_ROW_SEAMS = 15  # a type backed by fewer seams in the group is left out of the plan
MIN_OWN_SEAMS = 3  # a named DJ's own overlap length is used from this many seams of a type
CUT_CONTRAST_SHARE = (
    0.25  # of a set's cuts, placed where the two tracks differ. Untuned, for the ear
)


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


def pick_styles(
    genre: str | None,
    djs: list[str] | None = None,
    words: list[str] | None = None,
    groups: list[str] | None = None,
    bpm: tuple[float, float] | None = None,
) -> tuple[str | None, list[dict]]:
    """(genre, styles) a request is mixed like. A style is {"group", "dj", "weight"}: a style group,
    the DJ key when one DJ's own rows are meant, and its share of the set's seams (they sum to 1).

    `djs` are DJ names; each profiled one gives their own style in the genre (any genre when none is
    given), an unprofiled one gives nothing. `words` name groups of the genre by their word ("blend",
    "cut", "fast"); "auto" takes every group of the genre by its seams, less the further its sets'
    tempo range lies from `bpm`. `groups` are full group names. Nothing found gives no styles, and
    the caller plans the old way. One genre per set: a techno group and a house group sit 10 BPM
    apart, which the planner cannot bridge.
    """
    cards = load_groups().set_index("group")
    styles = [{"group": g, "dj": None, "weight": 1.0} for g in groups or []]
    for dj in djs or []:
        group, key = style_of(genre, None, dj)
        if key:
            genre = genre or cards.loc[group, "genre"]
            styles.append({"group": group, "dj": key, "weight": 1.0})
    for word in words or []:
        own = cards[cards["genre"] == genre]
        if word == "auto":
            weight = own["seams"].astype(float)
            if (
                bpm
            ):  # the part of the asked tempo range the group's sets cover, never under a tenth
                cover = np.minimum(own["bpm_high"], bpm[1]) - np.maximum(own["bpm_low"], bpm[0])
                weight = weight * np.clip(cover / max(bpm[1] - bpm[0], 1e-9), 0.1, 1)
            styles += [{"group": g, "dj": None, "weight": float(w)} for g, w in weight.items()]
        elif f"{genre} {word}" in own.index:
            styles.append({"group": f"{genre} {word}", "dj": None, "weight": 1.0})
        else:
            have = ", ".join(g.removeprefix(f"{genre} ") for g in own.index)
            raise ValueError(f"no {genre} style {word!r}; there is: {have}")
    if styles:
        genres = set(cards.loc[[s["group"] for s in styles], "genre"])
        if len(genres) > 1:
            raise ValueError(f"one genre per set, asked for {sorted(genres)}")
        genre = genres.pop()
    total = sum(s["weight"] for s in styles)
    return genre, [s | {"weight": round(s["weight"] / total, 3)} for s in styles]


def set_plan(styles: list[dict], n_tracks: int, seed: int = 0) -> list[dict]:
    """The set's budget of seams, one dict each: `group` (the style it is drawn from), `type`,
    `bass_free_bars`, `entry_section` (the section the style's incoming tracks are most often in when
    first heard), `play_min` drawn from the style's middle half, and the middle half itself for the
    overlap and for the bar the incoming track starts at (`overlap_low`, `overlap_high`,
    `enters_low`, `enters_high`).

    With several styles each seam comes whole from one of them, by weight: 70 % of one style and
    30 % of another gives about 7 seams in 10 from the first. Their numbers are never averaged, since
    an average is a style nobody plays.

    The plan advises: the planner gives each seam to the join it suits (`join_fit`), and the mixer
    picks the exact bar and length inside the ranges from the two tracks. The list order is random
    and means nothing until the planner has placed the seams.
    """
    every, own = load_group_types(), load_dj_types()
    parts = []
    for style in styles:
        rows = every[every["group"] == style["group"]]
        if rows.empty:
            raise ValueError(f"no group {style['group']!r}; see load_groups()")
        if style["dj"]:
            # The DJ's own share of each type, from all their seams. The recipe of a type comes from
            # their own row where it rests on MIN_ROW_SEAMS seams, from the group's row otherwise
            # (Fred again has 31 typed seams and no type reaches 15).
            mine = own[(own["group"] == style["group"]) & (own["dj_key"] == style["dj"])]
            if mine["seams"].sum() >= MIN_ROW_SEAMS:
                deep = mine[mine["seams"] >= MIN_ROW_SEAMS]
                rest = rows[~rows["type"].isin(deep["type"])].copy()
                rest["share"] = rest["type"].map(mine.set_index("type")["share"]).fillna(0.0)
                # How long a DJ blends is their own even on few seams: Fred again's five centre
                # swaps overlap 62 to 97 bars where his group's overlap 23 to 35 (Anas, 2026-11-25:
                # "add one long blend too"). Thin, so only the overlap and the bars left are taken.
                few = mine[mine["seams"] >= MIN_OWN_SEAMS].set_index("type")
                lengths = ["overlap_bars_low", "overlap_bars", "overlap_bars_high", "bars_left"]
                for column in lengths:
                    theirs = rest["type"].map(few[column])
                    rest[column] = theirs.where(theirs.notna(), rest[column])
                rows = pd.concat([deep, rest])
        ranges = ["play_min_low", "play_min_high", "enters_bar_low", "overlap_bars_low"]
        rows = rows[rows["seams"] >= MIN_ROW_SEAMS].dropna(subset=ranges)
        if not rows["share"].sum():
            raise ValueError(
                f"style {style} has no transition type backed by {MIN_ROW_SEAMS} seams"
            )
        parts.append(rows.assign(share=rows["share"] / rows["share"].sum() * style["weight"]))
    rows = pd.concat(parts, ignore_index=True)
    rng = np.random.default_rng(seed)
    counts = type_counts(rows["share"], n_tracks - 1)
    plan = []
    for k in rng.permutation([k for k, n in counts.items() for _ in range(n)]):
        row = rows.loc[k]
        seam = {"group": row["group"], "type": row["type"]}
        seam["bass_free_bars"] = round(float(row["bass_free_bars"]), 1)
        seam["entry_section"] = str(row["entry_section"])
        seam["bars_left"] = (
            None if pd.isna(row["bars_left"]) else round(float(row["bars_left"]), 1)
        )
        if row["type"] == "cut":
            # Most cuts go where the two tracks match, some where they differ. Placed on the pairs
            # furthest apart in tempo, as real cuts lean, they landed on the least similar tracks:
            # "lyrics to cut and diff vibe" (Anas, 2026-11-25). The contrast cut keeps its room.
            seam["cut_on"] = "contrast" if rng.random() < CUT_CONTRAST_SHARE else "match"
        seam["play_min"] = round(float(rng.uniform(row["play_min_low"], row["play_min_high"])), 1)
        for name, column in (("overlap", "overlap_bars"), ("enters", "enters_bar")):
            seam[f"{name}_low"] = round(float(row[f"{column}_low"]), 1)
            seam[f"{name}_high"] = round(float(row[f"{column}_high"]), 1)
        plan.append(seam)
    return plan


def join_fit(
    structure: pd.DataFrame,
    seam: dict,
    tempo_gap_pct: np.ndarray,
    last_grid: float,
    energy_gap: np.ndarray | float = 0.0,
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
    candidate's own `downbeat_confidence`). A cut planned as "match" (`cut_on`) wants the two tracks
    alike in tempo and loudness (`energy_gap`, 0 to 1), since the mixer cannot pick the moment by ear
    the way a DJ does; a "contrast" cut keeps the rule the data gave.
    """
    close = 1 - np.clip(np.asarray(tempo_gap_pct) / 5.0, 0, 1)
    slot = slot_fit(structure, seam)
    if seam["type"] != "cut":
        return 0.5 * slot + 0.5 * close
    sure = (structure["downbeat_confidence"].fillna(1.0) >= 0.5).to_numpy() & (last_grid >= 0.5)
    alike = 0.5 * close + 0.5 * (1 - np.clip(np.asarray(energy_gap), 0, 1))
    pair = alike if seam.get("cut_on") == "match" else 1 - close
    return np.where(sure, 0.5 * slot + 0.5 * pair, 0.0)


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
    match = cut | {"cut_on": "match"}  # alike in tempo and loudness scores highest
    assert join_fit(tracks, match, gap, 0.9, np.array([0.0, 0.0, 1.0])).tolist() == [
        1.0,
        0.0,
        0.25,
    ]
    shares = pd.Series({"centre swap": 0.54, "tension": 0.18, "end swap": 0.12, "cut": 0.07})
    # 7 seams: 4.15, 1.38, 0.92 and 0.54 exact, so the two spare seams go to end swap and cut
    assert type_counts(shares, 7) == {"centre swap": 4, "tension": 1, "end swap": 1, "cut": 1}
    assert sum(type_counts(shares, 5).values()) == 5


if __name__ == "__main__":
    demo()
    p = argparse.ArgumentParser()
    p.add_argument("--genre", default=None)
    p.add_argument("--dj", nargs="*", default=[], help="DJ names, each mixed in their own style")
    p.add_argument(
        "--style", nargs="*", default=[], help='"auto" or the words of the genre\'s groups'
    )
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    genre, styles = pick_styles(a.genre, a.dj, a.style)
    print(genre, json.dumps(styles))
    print(json.dumps(set_plan(styles, a.n, a.seed), indent=1))
