"""
Profiles built from the dataset: what kind of set each mix is, and which DJs share a playing style.

    python -m src.data.profiles        -> data/dataset/<newest date>/mixes.parquet    load_mixes()
                                                                     groups.parquet   load_groups()
                                                                     group_types.parquet   load_group_types()
                                                                     dj_types.parquet      load_dj_types()

Both are computed from plays, tracks and seams alone, so new DJ data needs only `djdata export-dataset`
and this command again.

mixes.parquet, one row per mix, every source:

  energy, loudness, brightness, density   mean over the mix's tracks of each track's percentile inside its
                                          genre and audio kind (preview or full), 0 to 1
  bpm                                     median track BPM
  energy_rise                             energy of the last third minus the first third
  shape                                   `_energy_curve_shape` of the tracks' energy in play order
  setting                                 festival | club | radio, from words in the mix title, else empty
  set_kind                                warm-up | middle | peak-time: the third of its genre's mixes that
                                          the mix's energy falls in. Empty under MIN_TRACKS tracks with features
  change_min                              median minutes from one track's entry to the next (measured mixes)
  seams                                   measured seams in the mix
  group                                   the style group of the mix's DJ in this genre, empty for a DJ
                                          under the genre's FLOOR of measured seams

groups.parquet, one row per group. A unit is one DJ in one genre with the genre's FLOOR of measured seams. Units of a
genre are clustered on STYLE (z-scored inside the genre, Ward linkage, the number of groups with the best
silhouette). A DJ who sits nearer another group than their own (silhouette under 0) is left out of every
group, unless the DJ is deep (DEEP seams) or hand-picked (seams in the `djs` source): that DJ becomes a
group alone. The row holds the group's DJs, its seam count,
the mean of each STYLE column over its DJs, and what a prompt can be matched against: `mixes` and the 10th and
90th percentile of BPM and energy over the group's mixes (`bpm_low`, `bpm_high`, `energy_low`, `energy_high`).
A group is named by its genre and one word from WORDS: its most used transition type, or, when another group
of the genre has that word, the STYLE column where it stands furthest above the genre's other groups. A type
word needs NAMED of the group's seams to be of that type, so the name never claims a rare move. A group
with no word left is named by its first DJ.
A group's full distributions are its seams: `load_seams()` joined to `load_mixes()` on mix_id.

group_types.parquet, one row per group and transition type: how often the group uses the type, how it is
done, and what the incoming track is like in that place. Every number is over the group's seams of that
type; `_low` and `_high` are the 25th and 75th percentile, the plain name the median. A DJ counts by the
square root of their seams in the group, in the share and in every percentile, so the deepest DJ leads the
group without owning it.
  share, seams                     the type's weighted share in the group, and the seams behind the row
  overlap_bars                     how long both tracks are heard together
  bass_free_bars                   bars with no bass from either track
  new_bass_at, old_bass_out        where in the overlap (0 start, 1 end) the incoming track's bass enters
                                   and the outgoing track's bass leaves
  sweep                            share of the seams with a filter sweep on either track
  enters_bar                       the bar of the incoming track where it is first heard (entries read
                                   more than 2 bars before the track's start are left out)
  entry_level_db, entry_section    the incoming track over its first 8 s heard, against its own loud
                                   level, and the section it is most often in
  bars_left                        bars of the outgoing track after it is last heard
  play_min                         minutes the incoming track is heard, to its own exit at the next seam
  energy_step                      incoming minus outgoing track energy (`energy_mean`, the mixer's scale)

dj_types.parquet, the same columns per group, DJ and transition type: what one DJ of the group does, for a
request that names the DJ.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_samples, silhouette_score

from src.data.dataset import latest, load_plays, load_seams, load_tracks
from src.features.mix_profiler import _energy_curve_shape

MIN_TRACKS = 8  # tracks with features a mix needs before its level means anything
LEVELS = {
    "energy": "energy_mean",
    "loudness": "loudness_lufs",
    "brightness": "spectral_centroid",
    "density": "onset_strength",
}
SETTINGS = {  # first match wins
    "radio": r"radio|podcast|episode|broadcast|essential mix",
    "festival": r"festival|stage|tomorrowland|awakenings|ultra|edc|creamfields",
    "club": r"club|boiler room|cercle|warehouse|ibiza",
}
KINDS = ["warm-up", "middle", "peak-time"]
# Measured seams a DJ needs in a genre to be profiled. Chosen 2026-11-18 from floors 10, 12, 15, 20 and 25:
# per genre the floor that keeps the most DJs while the groups stay within 0.05 of their tightest (mean
# silhouette), checked by whether a DJ's held-out mixes land in their own group.
#   genre           floor  DJs kept  groups  tightness  held-out   note
#   house           10     15        2       0.22       0.97       Anas: more house DJs. Tightest is 15 (6 DJs, 0.32)
#   drum and bass   10     19        2       0.17       0.97       tightest floor and the most DJs
#   tech house      12     34        2       0.26       0.97       10 splits into 4 loose groups (0.15, 0.89)
#   trance          12     12        4       0.24       0.87       10 adds 3 DJs and loosens to 0.21
#   techno          15     13        3       0.25       0.93       same DJs at 10 and 12; 20 keeps 10 DJs (0.28)
#   afro house, melodic house: one DJ each, the floor changes nothing
# Rerun the comparison when DJ data is added: more deep DJs let a floor rise.
FLOOR = {"house": 10, "drum and bass": 10, "tech house": 12, "trance": 12}
FLOOR_DEFAULT = 15
DEEP = 50  # a DJ with this many seams in a genre is never left out of the groups
NAMED = 0.3  # share of a group's seams a transition type needs before the group is named by it
TYPES = ["centre swap", "cut", "tension", "end swap", "late swap"]
STYLE = {  # what a group is built on: one number per DJ and genre, from that DJ's measured seams
    **{
        t: lambda s, t=t: (s["signature"] == t).sum() / s["signature"].isin(TYPES).sum()
        for t in TYPES
    },
    "overlap_bars": lambda s: s["overlap_bars"].median(),
    "enters_first_8_bars": lambda s: (s["b_in_bar"] < 8).mean(),
    "enters_past_bar_32": lambda s: (s["b_in_bar"] >= 32).mean(),
    "leaves_in_last_8_bars": lambda s: (s["a_bars_left"] < 8).mean(),
    "change_min": lambda s: s.drop_duplicates("mix_id")["change_min"].median(),
    "energy": lambda s: s.drop_duplicates("mix_id")["energy"].median(),
    "bpm": lambda s: s.drop_duplicates("mix_id")["bpm"].median(),
}
WORDS = dict(  # the word a group is named by when it stands out on that STYLE column
    zip(STYLE, "blend cut tension endswap lateswap long early deep full slow loud fast".split())
)


def set_kind(energy: pd.Series, genre: pd.Series) -> pd.Series:
    """The third of its genre's mixes a mix's energy falls in.

    ponytail: thirds are an arbitrary cut until Anas's ear confirms them; move the cut points or add
    energy_rise when the ear test says a third is wrong.
    """
    third = energy.groupby(genre).transform(lambda e: pd.qcut(e, 3, labels=False))
    return third.map(dict(enumerate(KINDS)))


def mix_table() -> pd.DataFrame:
    tracks = load_tracks(columns=["track_id", "genre", "audio", "bpm", *LEVELS.values()])
    for name, column in LEVELS.items():
        tracks[name] = tracks.groupby(["genre", "audio"])[column].rank(pct=True)
    plays = load_plays().merge(tracks.drop(columns=["genre", "audio"]), on="track_id", how="left")
    by_mix = plays.sort_values("position").groupby("mix_id")
    mixes = by_mix.agg(
        source=("source", "first"),
        dj=("dj", "first"),
        dj_key=("dj_key", "first"),
        genre=("genre", "first"),
        mix_title=("mix_title", "first"),
        mix_url=("mix_url", "first"),
        n_tracks=("track_id", "size"),
        n_featured=("energy", "count"),
        bpm=("bpm", "median"),
        **{name: (name, "mean") for name in LEVELS},
    )

    def rise(e: pd.Series) -> float:
        e = e.dropna()
        k = len(e) // 3
        return e.iloc[-k:].mean() - e.iloc[:k].mean() if k else float("nan")

    mixes["energy_rise"] = by_mix["energy"].apply(rise)
    heard = plays.dropna(subset=["first_heard_s"]).sort_values("first_heard_s")
    gap = heard.groupby("mix_id")["first_heard_s"].diff() / 60
    mixes["change_min"] = gap[gap.between(0.3, 15)].groupby(heard["mix_id"]).median()
    mixes["shape"] = by_mix["energy_mean"].apply(
        lambda e: _energy_curve_shape(e.dropna().tolist())
    )
    title = mixes["mix_title"].fillna("")
    mixes["setting"] = ""
    for name, words in reversed(SETTINGS.items()):
        mixes.loc[title.str.contains(words, case=False), "setting"] = name
    enough = mixes["n_featured"] >= MIN_TRACKS
    mixes["set_kind"] = set_kind(mixes.loc[enough, "energy"], mixes.loc[enough, "genre"])
    seams = load_seams(columns=["mix_id", "measured"])
    mixes["seams"] = seams[seams["measured"] == 1].groupby("mix_id").size()
    mixes["seams"] = mixes["seams"].fillna(0).astype(int)
    return mixes.reset_index()


def cluster(style: pd.DataFrame, keep: list[bool]) -> list[int]:
    """Group number per row. A row that fits another group better than its own gets 0, or a group of its
    own when `keep` is set for it. One group when there are too few rows to split."""
    if len(style) < 5:
        return [1] * len(style)
    z = ((style - style.mean()) / style.std()).fillna(0)
    tree = linkage(z, "ward")
    cuts = {k: fcluster(tree, k, "maxclust") for k in range(2, min(4, len(style) - 2) + 1)}
    labels = max(cuts.values(), key=lambda labels: silhouette_score(z, labels))
    fits = silhouette_samples(z, labels)
    alone = iter(range(len(style) + 1, 2 * len(style) + 1))
    return [n if fit >= 0 else next(alone) if k else 0 for n, fit, k in zip(labels, fits, keep)]


def measured_seams(mixes: pd.DataFrame) -> pd.DataFrame:
    """The measured seams with their mix's columns beside them."""
    seams = load_seams()
    seams = seams[seams["measured"] == 1].drop(columns=["dj", "genre"])
    beside = ["mix_id", "dj_key", "genre", "change_min", "energy", "bpm", "group"]
    return seams.merge(mixes[[c for c in beside if c in mixes]], on="mix_id")


def group_table(mixes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(groups, units): one row per group, and one row per DJ and genre with its group."""
    seams = measured_seams(mixes)
    picked = set(seams.loc[seams["source"] == "djs", "dj_key"])
    seams = seams.groupby(["genre", "dj_key"]).filter(
        lambda s: len(s) >= FLOOR.get(s["genre"].iloc[0], FLOOR_DEFAULT)
    )
    by_unit = seams.groupby(["genre", "dj_key"])
    units = pd.DataFrame({name: by_unit.apply(measure) for name, measure in STYLE.items()})
    units["seams"] = by_unit.size()
    units = units.reset_index()
    units["group"] = ""
    keep = (units["seams"] >= DEEP) | units["dj_key"].isin(picked)
    for genre, rows in units.groupby("genre"):
        numbers = cluster(rows[list(STYLE)], keep[rows.index].tolist())
        units.loc[rows.index, "group"] = [f"{genre} {n}" for n in numbers]
    units = units[~units["group"].str.endswith(" 0")].copy()
    groups = units.groupby("group").agg(
        genre=("genre", "first"),
        djs=("dj_key", list),
        seams=("seams", "sum"),
        **{name: (name, "mean") for name in STYLE},
    )
    groups = groups.reset_index()
    names = group_names(groups)
    units["group"], groups["group"] = units["group"].map(names), groups["group"].map(names)
    played = mixes.merge(units[["genre", "dj_key", "group"]], on=["genre", "dj_key"]).groupby(
        "group"
    )
    card = played.agg(
        mixes=("mix_id", "size"),
        bpm_low=("bpm", lambda b: b.quantile(0.1)),
        bpm_high=("bpm", lambda b: b.quantile(0.9)),
        energy_low=("energy", lambda e: e.quantile(0.1)),
        energy_high=("energy", lambda e: e.quantile(0.9)),
    )
    return groups.join(card, on="group"), units


def type_table(mixes: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """One row per `by` and transition type: its share, how it is done, the track that goes there."""
    seams = measured_seams(mixes)
    seams = seams[(seams["group"] != "") & seams["signature"].isin(TYPES)].copy()
    seams["enters_bar"] = seams["b_in_bar"].where(seams["b_in_bar"] >= -2)
    exits = seams.drop_duplicates(["mix_id", "track_a"]).set_index(["mix_id", "track_a"])[
        "a_out_s"
    ]
    own_exit = exits.reindex(pd.MultiIndex.from_frame(seams[["mix_id", "track_b"]])).to_numpy()
    play = (own_exit - seams["b_in_s"]) / 60
    seams["play_min"] = play.where(play.between(0.5, 15))
    energy = load_tracks(columns=["track_id", "energy_mean"]).set_index("track_id")["energy_mean"]
    seams["energy_step"] = seams["track_b"].map(energy) - seams["track_a"].map(energy)
    seams["sweep"] = (seams["sweep_out_bars"] > 0) | (seams["sweep_in_bars"] > 0)
    spread = {"overlap_bars": "overlap_bars", "enters_bar": "enters_bar", "play_min": "play_min"}
    median = {
        "bass_free_bars": "bass_cut_bars",
        "new_bass_at": "b_low_pos",
        "old_bass_out": "a_low_pos",
        "entry_level_db": "b_in_rms_rel_db",
        "bars_left": "a_bars_left",
        "energy_step": "energy_step",
    }
    # A DJ counts by the square root of their seams, so the deepest DJ leads the group without owning it
    # (Anas, 2026-11-18: a weighted average, not biased by a single DJ). Each seam carries 1 / sqrt(n).
    seams["weight"] = seams.groupby(["group", "dj_key"])["seam_id"].transform("size") ** -0.5

    def describe(s: pd.DataFrame) -> pd.Series:
        row = {"seams": len(s), "weight": s["weight"].sum()}
        row["sweep"] = (s["sweep"] * s["weight"]).sum() / row["weight"]
        sections = s.groupby("b_in_section")["weight"].sum()
        row["entry_section"] = sections.idxmax() if len(sections) else ""
        for name, column in (spread | median).items():
            row[name] = weighted_quantile(s[column], s["weight"], 0.5)
        for name, column in spread.items():
            row[f"{name}_low"] = weighted_quantile(s[column], s["weight"], 0.25)
            row[f"{name}_high"] = weighted_quantile(s[column], s["weight"], 0.75)
        return pd.Series(row)

    table = seams.groupby([*by, "signature"]).apply(describe).reset_index()
    share = table["weight"] / table.groupby(by)["weight"].transform("sum")
    table.insert(len(by) + 1, "share", share)
    table["seams"] = table["seams"].astype(int)
    return table.drop(columns="weight").rename(columns={"signature": "type"})


def weighted_quantile(values: pd.Series, weights: pd.Series, q: float) -> float:
    """The value below which share `q` of the weight lies. NaN values are left out."""
    known = values.notna()
    values, weights = values[known].to_numpy(dtype=float), weights[known].to_numpy()
    if not len(values):
        return float("nan")
    order = np.argsort(values)
    reached = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][np.searchsorted(reached, q)])


def group_names(groups: pd.DataFrame) -> dict[str, str]:
    """Numbered group -> "<genre> <word>". The group purest in its main type picks its word first."""
    names = {}
    for genre, rows in groups.groupby("genre"):
        taken = set()
        above = (rows[list(STYLE)] - rows[list(STYLE)].mean()) / rows[list(STYLE)].std()
        for i in rows[TYPES].max(axis=1).sort_values(ascending=False).index:
            row, high = rows.loc[i], above.loc[i].fillna(0).sort_values(ascending=False)
            fits = [c for c in high.index if (row[c] >= NAMED if c in TYPES else high[c] > 0)]
            main = row[TYPES].astype(float).idxmax()
            word = next((WORDS[c] for c in [main, *fits] if WORDS[c] not in taken), row["djs"][0])
            taken.add(word)
            names[rows.loc[i, "group"]] = f"{genre} {word}"
    return names


def demo():
    two_styles = pd.DataFrame(
        {"overlap_bars": [2, 3, 2, 40, 42, 41], "cut": [0.5, 0.6, 0.5, 0, 0.1, 0]}
    )
    labels = cluster(two_styles, [False] * 6)
    assert (
        labels[:3] == [labels[0]] * 3 and labels[3:] == [labels[3]] * 3 and labels[0] != labels[3]
    )
    assert 0 not in labels  # two clean styles, nobody is left out
    rows = pd.DataFrame(
        {"group": ["house 1", "house 2"], "genre": "house"} | dict.fromkeys(STYLE, 0.0)
    )
    rows["djs"] = [["a"], ["b"]]
    rows["centre swap"], rows["bpm"] = [0.6, 0.5], [120.0, 130.0]
    assert group_names(rows) == {"house 1": "house blend", "house 2": "house fast"}
    ones = pd.Series([1.0, 1.0, 1.0])
    assert weighted_quantile(pd.Series([3.0, 1.0, 2.0]), ones, 0.5) == 2.0
    assert weighted_quantile(pd.Series([1.0, 2.0, 9.0]), pd.Series([1.0, 1.0, 10.0]), 0.5) == 9.0
    energy = pd.Series([0.1, 0.5, 0.9, 0.2, 0.6, 0.8])
    genre = pd.Series(["techno"] * 3 + ["house"] * 3)
    assert set_kind(energy, genre).tolist() == KINDS * 2


if __name__ == "__main__":
    demo()
    out = latest() / "mixes.parquet"
    table = mix_table()
    groups, units = group_table(table)
    table = table.merge(units[["genre", "dj_key", "group"]], on=["genre", "dj_key"], how="left")
    table["group"] = table["group"].fillna("")
    groups.to_parquet(latest() / "groups.parquet", index=False)
    type_table(table, ["group"]).to_parquet(latest() / "group_types.parquet", index=False)
    type_table(table, ["group", "dj_key"]).to_parquet(latest() / "dj_types.parquet", index=False)
    print(groups.drop(columns="genre").round(2).to_string())
    table.to_parquet(out, index=False)
    print(out, len(table), "mixes")
    print(
        table.groupby(["genre", "set_kind"]).agg(mixes=("mix_id", "size"), seams=("seams", "sum"))
    )
