"""The dataset the models read: four parquet files built from every source by one command.

    djdata export-dataset --date 2026-10-14        -> data/dataset/2026-10-14/

    tracks.parquet   one row per record, every source, the full-audio row wins over the preview row
    plays.parquet    one row per play in a mix, every source, with the split (train | validation)
    seams.parquet    one row per consecutive pair in a mix, every source; the measured sources carry
                     every column of their seams_index (bands, bass, bars, signature), the old data
                     carries the pair alone. Overlays ("w/" records) are rows with kind = overlay
    layers.parquet   one row per record and band in a span where records stack (DJ data only)

Ids. Every source uses the 1001 scheme: track_id = md5("artist|title")[:12], mix_id = md5(url)[:10]
(`src/data/tracklists1001_client.py`). The old data and the DJ data already carry them; Raveform's
YouTube ids and mix numbers are re-hashed here from the title and the page url and kept in
`source_ref`, so the same record played by two DJs in two sources is one id. `record_key` (letters
only) sits beside the id for joins where punctuation differs ("vs" against "vs."; 49 of 732 old
matches on 2026-10-13).

Validation. `split` on every play and seam, train or validation, by whole DJ (a general planner is
judged on DJs it has never seen, and a mix-level split would leak each DJ's habits). Anas, 2026-10-13:
15 % of mixes, diverse, not so large that training starves. `validation_split` picks DJs with 3 to 40
mixes (a DJ with 200 mixes stays in training), at least MIN_DJS per genre, then fills in a fixed order,
the one-and-two-mix DJs last, until every genre and every source holds VALIDATION_SHARE of its mixes. A mix two sources share (140 DJ-data mixes
are old-data mixes) sits on one side by construction. The README prints the per-genre and per-source
table and the track overlap between the sides.

Genre. `genre_of` is the one mapping from a source's label (the old data's six names, Raveform's
mixesdb tag strings) onto the seven genres. The notebook builder carries a copy; keep them equal.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

log = logging.getLogger("djdata.dataset")

DATASET_DIR = Path("data/dataset")
SOURCES = {
    "old": {
        "plays": Path("data/processed/tracklist_clean.csv"),
        "raw": Path("data/interim/tracklist.csv"),  # the overlay rows live only here
        "features": Path("data/processed/features.parquet"),
        # old records fetched in full by the mixer's fetcher and registered (src/data/register_track.py)
        "full": Path("data/djdata/old/out/track_features.parquet"),
    },
    "raveform": {
        "root": Path("data/djdata/raveform"),
        "raveform_dir": Path("data/raw/raveform/raveform"),
    },
    "djs": {"root": Path("data/djdata/djs")},
}
ENRICH = {
    "spotify": Path("data/interim/enrich/spotify_records.csv"),
    "lastfm": Path("data/interim/enrich/lastfm_artists.csv"),
}
VALIDATION_SHARE = 0.15
VALIDATION_DJ_MIXES = (
    3,
    40,
)  # a validation DJ has this many mixes: diverse, and training keeps the big names
MIN_DJS = 3  # per genre in validation, where the genre has them
MFCC = [f"mfcc_{i}" for i in range(13)]
FEATURE_COLS = [
    "bpm",
    "bpm_source",
    "key",
    "loudness_lufs",
    "energy_mean",
    "energy_std",
    "spectral_centroid",
    "onset_strength",
    *MFCC,
]
TRACK_COLS = [
    "track_id",
    "record_key",
    "source",
    "audio",
    "source_ref",
    "artist",
    "title",
    "genre",
    "minutes",
    "n_vectors",
    "embedding",
    *FEATURE_COLS,
]


# ── ids ────────────────────────────────────────────────────────────────────────


def track_id(artist: str, title: str) -> str:
    """The 1001 scrape's id, unchanged: `tracklists1001_client._track_id`."""
    return hashlib.md5(f"{artist}|{title}".lower().encode()).hexdigest()[:12]


def mix_id(url: str) -> str:
    return hashlib.md5(str(url).encode()).hexdigest()[:10]


def record_key(artist: str, title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", f"{artist}{title}".lower())


def split_title(text: str) -> tuple[str, str]:
    """Raveform's "Artist - Title [Label]" (a "[05] " order prefix on some) -> (artist, title)."""
    t = re.sub(r"^\[\d+\]\s*", "", str(text))
    t = re.sub(r"\s*\[[^\]]*\]\s*$", "", t)
    artist, sep, title = t.partition(" - ")
    return (artist.strip(), title.strip()) if sep else ("", t.strip())


# the loader lives in the repo above the package, as in seam/tempo.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import dj_key  # noqa: E402  one definition, the loader's


def validation_split(plays: pd.DataFrame, share: float = VALIDATION_SHARE) -> pd.Series:
    """mix_id -> "train" | "validation", by whole DJ, deterministic.

    Eligible DJs have VALIDATION_DJ_MIXES mixes. First MIN_DJS per genre (the DJ's main genre), then
    DJs in dj_key order while every genre and source they touch is still under `share` of its mixes.
    """
    mixes = plays.drop_duplicates("mix_id")[["mix_id", "dj_key", "genre", "source"]]
    genre_total = mixes["genre"].value_counts()
    source_total = mixes["source"].value_counts()
    per_dj = mixes.groupby("dj_key")
    n_mixes = per_dj.size()
    eligible = sorted(n_mixes[n_mixes.between(*VALIDATION_DJ_MIXES)].index)
    # a genre owned by a few big names (trance: 3 DJs over 80 mixes, 46 with one or two) fills up
    # from the small DJs, which adds voices and keeps the big names in training
    small = sorted(n_mixes[n_mixes < VALIDATION_DJ_MIXES[0]].index)
    main_genre = per_dj["genre"].agg(lambda s: s.value_counts().idxmax())

    chosen: list[str] = []
    genre_have = {g: 0 for g in genre_total.index}
    source_have = {s: 0 for s in source_total.index}

    def room(dj: str) -> bool:
        g = per_dj.get_group(dj)
        for genre, n in g["genre"].value_counts().items():
            if genre_have[genre] + n > share * genre_total[genre] + VALIDATION_DJ_MIXES[1] / 2:
                return False
        for src, n in (
            g["source"].value_counts().items()
        ):  # a source may run over its share, a genre may not:
            if (
                source_have[src] + n > (share + 0.10) * source_total[src]
            ):  # trance's small DJs are all Raveform
                return False
        return True

    def take(dj: str) -> None:
        chosen.append(dj)
        g = per_dj.get_group(dj)
        for genre, n in g["genre"].value_counts().items():
            genre_have[genre] += n
        for src, n in g["source"].value_counts().items():
            source_have[src] += n

    for genre in sorted(genre_total.index):  # every genre gets voices first
        for dj in [d for d in eligible if main_genre[d] == genre][:MIN_DJS]:
            if dj not in chosen and room(dj):
                take(dj)
    for dj in eligible + small:  # then fill every genre and source to the share
        if dj in chosen:
            continue
        g = per_dj.get_group(dj)
        wants = any(genre_have[ge] < share * genre_total[ge] for ge in g["genre"].unique()) or any(
            source_have[s] < share * source_total[s] for s in g["source"].unique()
        )
        if wants and room(dj):
            take(dj)
    side = mixes["dj_key"].isin(chosen).map({True: "validation", False: "train"})
    return pd.Series(side.values, index=mixes["mix_id"].values)


def genre_of(text: str) -> str:
    """A source's label onto the seven genres (the old data spells drum and bass "drum and base")."""
    t = str(text).lower()
    if "drum" in t or t == "dnb":
        return "drum and bass"
    if "trance" in t:
        return "trance"
    if "tech house" in t:
        return "tech house"
    if "techno" in t:
        return "techno"
    if "melodic" in t or "progressive house" in t:
        return "melodic house"
    if "afro" in t:
        return "afro house"
    if "house" in t:
        return "house"
    return "other"


# ── the old data ───────────────────────────────────────────────────────────────


def old_tables(paths: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """tracks, plays, seams from the tracklists and the preview features. Nothing is recomputed."""
    p = paths or SOURCES["old"]
    tl = pd.read_csv(p["plays"], low_memory=False)
    tl["genre"] = tl["genre"].map(genre_of)
    feats = pd.read_parquet(p["features"]).drop_duplicates("track_id")

    plays = []
    for mid, g in tl.groupby("mix_id", sort=False):
        if (
            g["starting_time"].notna().all()
        ):  # the rule train_model used: only sort when every time is there
            g = g.sort_values("starting_time", kind="stable")
        g = g.assign(position=np.arange(len(g)))
        plays.append(g)
    plays = pd.concat(plays, ignore_index=True)
    plays = pd.DataFrame(
        {
            "mix_id": plays["mix_id"],
            "source": "old",
            "source_ref": plays["mix_id"],
            "dj": plays["dj_name"],
            "genre": plays["genre"],
            "mix_title": plays["mix_title"],
            "mix_url": plays["url"],
            "position": plays["position"],
            "track_id": plays["track_id"],
            "listed_min": plays["starting_time"],
        }
    )

    meta = tl.drop_duplicates("track_id").set_index("track_id")
    tracks = feats[feats["track_id"].isin(meta.index)].copy()
    tracks["artist"] = tracks["track_id"].map(meta["artist_name"])
    tracks["title"] = tracks["track_id"].map(meta["track_name"])
    tracks["genre"] = tracks["track_id"].map(
        meta["genre"]
    )  # the first set it was listed in, as the planner does today
    tracks["source"] = "old"
    tracks["audio"] = "preview"
    tracks["source_ref"] = tracks["track_id"]
    tracks["bpm_source"] = "deeprhythm"
    tracks["minutes"] = np.nan
    tracks["n_vectors"] = np.nan
    full = p.get("full")
    if (
        full is not None and Path(full).exists()
    ):  # registered full-audio rows replace the preview rows
        fr = pd.read_parquet(full).drop_duplicates("track_id", keep="last")
        fr = fr[fr["track_id"].isin(tracks["track_id"])]
        keep = tracks[~tracks["track_id"].isin(fr["track_id"])]
        fr = fr.copy()
        fr["artist"] = fr["track_id"].map(meta["artist_name"])
        fr["title"] = fr["track_id"].map(meta["track_name"])
        fr["genre"] = fr["track_id"].map(meta["genre"])
        fr["source"], fr["audio"], fr["source_ref"] = "old", "full", fr["track_id"]
        if "bpm" not in fr.columns or fr["bpm"].isna().all():
            fr["bpm"] = fr["track_id"].map(tracks.set_index("track_id")["bpm"])
            fr["bpm_source"] = "deeprhythm"
        tracks = pd.concat([keep, fr], ignore_index=True)
        log.info("old source: %d records carry registered full-track features", len(fr))

    seams = consecutive_pairs(plays, "old")
    raw = pd.read_csv(
        p["raw"], usecols=["mix_id", "track_id", "play_type", "overlay_parent"], low_memory=False
    )
    ov = raw[
        (raw["play_type"] == "simultaneous")
        & raw["overlay_parent"].notna()
        & raw["mix_id"].isin(set(plays["mix_id"]))
    ]
    if len(ov):
        mixmeta = plays.drop_duplicates("mix_id").set_index("mix_id")
        overlays = pd.DataFrame(
            {
                "seam_id": ov["mix_id"] + "_" + ov["overlay_parent"] + "_" + ov["track_id"],
                "mix_id": ov["mix_id"],
                "source": "old",
                "dj": ov["mix_id"].map(mixmeta["dj"]),
                "genre": ov["mix_id"].map(mixmeta["genre"]),
                "track_a": ov["overlay_parent"],
                "track_b": ov["track_id"],
                "kind": "overlay",
            }
        )
        seams = pd.concat([seams, overlays], ignore_index=True)
    return _track_frame(tracks), plays, seams


def consecutive_pairs(plays: pd.DataFrame, source: str) -> pd.DataFrame:
    """One row per record and the record listed after it, in play order. Measurement columns absent."""
    rows = []
    for mid, g in plays.groupby("mix_id", sort=False):
        g = g.sort_values("position")
        ids = g["track_id"].tolist()
        for i, (a, b) in enumerate(zip(ids, ids[1:])):
            if a == b:
                continue
            rows.append(
                (
                    f"{mid}_{a}_{b}",
                    mid,
                    source,
                    g["dj"].iloc[0],
                    g["genre"].iloc[0],
                    a,
                    b,
                    i,
                    i + 1,
                    "seam",
                )
            )
    return pd.DataFrame(
        rows,
        columns=[
            "seam_id",
            "mix_id",
            "source",
            "dj",
            "genre",
            "track_a",
            "track_b",
            "order_a",
            "order_b",
            "kind",
        ],
    )


# ── the measured sources ───────────────────────────────────────────────────────


def _raveform_ids(raveform_dir: Path) -> tuple[dict, dict, dict]:
    """YouTube id -> (artist, title), raveform mix number -> page url, mix number -> title."""
    names, urls, titles = {}, {}, {}
    with open(raveform_dir / "mixes.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            urls[d["id"]], titles[d["id"]] = d["url"], d["title"]
            for t in d["tracklist"]:
                if t.get("id") and t["id"] not in names:
                    names[t["id"]] = split_title(t["title"])
    return names, urls, titles


def measured_tables(
    source: str, root: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """tracks, plays, seams, layers from a source's out/ folder, ids on the 1001 scheme."""
    root = root or SOURCES[source]["root"]
    out = root / "out"
    plays = pd.read_csv(out / "plays.csv", low_memory=False)
    plays = plays[plays["is_control"] == 0].copy()
    seams = pd.read_csv(out / "seams_index.csv", low_memory=False)
    feats = pd.read_parquet(out / "track_features.parquet")
    if "n_patches" in feats.columns:  # the Raveform table named it so
        feats = feats.rename(columns={"n_patches": "n_vectors"}).drop(
            columns=[c for c in ("n_vectors_x",) if c in feats.columns]
        )
    feats = feats.loc[:, ~feats.columns.duplicated()]

    if source == "raveform":
        names, urls, titles = _raveform_ids(SOURCES["raveform"]["raveform_dir"])
        tmap = {yt: track_id(a, t) for yt, (a, t) in names.items()}
        mmap = {m: mix_id(u) for m, u in urls.items()}
        name_of = names
        mix_title = plays["mix_id"].map(titles)
    else:
        tmap, mmap = {}, {}
        name_of = {
            tid: split_title(t)
            for tid, t in zip(plays["track_id"], plays["title"])
            if isinstance(t, str)
        }
        raw = pd.read_csv(
            SOURCES["old"]["raw"],
            usecols=["track_id", "artist_name", "track_name", "genre"],
            low_memory=False,
        ).drop_duplicates("track_id")
        for tid, a, t in zip(raw["track_id"], raw["artist_name"], raw["track_name"]):
            if tid not in name_of and isinstance(a, str) and isinstance(t, str):
                name_of[tid] = (a, t)  # a record on disk that the final run never played
        raw_genre = dict(zip(raw["track_id"], raw["genre"].map(genre_of)))
        # Fred's usb_<hash> ids carry a title and join the scheme; every other DJ-data id already is the scheme
        tmap = {
            tid: track_id(*name_of[tid])
            for tid in plays["track_id"]
            if str(tid).startswith("usb_") and tid in name_of and name_of[tid][0]
        }
        mix_title = plays["mix_title"]

    def tid(s):
        return s.map(lambda x: tmap.get(x, x))

    def mid(s):
        return s.map(lambda x: mmap.get(x, x))

    plays_out = pd.DataFrame(
        {
            "mix_id": mid(plays["mix_id"]),
            "source": source,
            "source_ref": plays["mix_id"],
            "dj": plays["dj"],
            "genre": plays["genre"].map(genre_of),
            "mix_title": mix_title,
            "mix_url": plays["mix_id"].map(urls) if source == "raveform" else np.nan,
            "position": plays["order_listed"],
            "track_id": tid(plays["track_id"]),
            "listed_min": plays["listed_min"],
            "found": plays["found"],
            "confidence": plays["confidence"],
            "time_zero_s": plays["time_zero_s"],
            "rate": plays["rate"],
            "mode": plays["mode"] if "mode" in plays.columns else np.nan,
            "played_bpm": plays["played_bpm"] if "played_bpm" in plays.columns else np.nan,
            "first_heard_s": plays["first_heard_s"],
            "last_heard_s": plays["last_heard_s"],
            "played_from_s": plays["played_from_s"],
            "played_to_s": plays["played_to_s"],
            "votes": plays["votes"],
            "floor": plays["floor"],
        }
    )

    seams = seams.copy()
    mode = plays.set_index(["mix_id", "track_id"])["mode"] if "mode" in plays.columns else None
    if mode is not None:
        mode = mode[~mode.index.duplicated()]
        seams["a_mode"] = [mode.get((m, a)) for m, a in zip(seams["mix_id"], seams["track_a"])]
        seams["b_mode"] = [mode.get((m, b)) for m, b in zip(seams["mix_id"], seams["track_b"])]
    seams["source"] = source
    seams["kind"] = "seam"
    seams["genre"] = seams["genre"].map(genre_of)
    seams["track_a"], seams["track_b"], seams["mix_id"] = (
        tid(seams["track_a"]),
        tid(seams["track_b"]),
        mid(seams["mix_id"]),
    )
    seams["seam_id"] = seams["mix_id"] + "_" + seams["track_a"] + "_" + seams["track_b"]

    feats = feats.copy()
    feats["source_ref"] = feats["track_id"]
    feats["track_id"] = tid(feats["track_id"])
    feats["artist"] = feats["source_ref"].map(lambda x: name_of.get(x, ("", ""))[0])
    feats["title"] = feats["source_ref"].map(lambda x: name_of.get(x, ("", ""))[1])
    genre = plays_out.drop_duplicates("track_id").set_index("track_id")["genre"]
    feats["genre"] = feats["track_id"].map(genre)
    if source == "djs":
        feats["genre"] = feats["genre"].fillna(feats["source_ref"].map(raw_genre))
    feats["source"] = source
    feats["audio"] = "full"

    layers = pd.DataFrame()
    if (out / "layer_bands.csv").exists():
        layers = pd.read_csv(out / "layer_bands.csv", low_memory=False)
        layers.insert(0, "source", source)
    return _track_frame(feats), plays_out, seams, layers


def _track_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["embedding"] = [np.asarray(e, dtype=np.float32) for e in df["embedding"]]
    df["record_key"] = [
        record_key(a, t) for a, t in zip(df["artist"].fillna(""), df["title"].fillna(""))
    ]
    for c in TRACK_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[TRACK_COLS]


# ── the export ─────────────────────────────────────────────────────────────────


def enrich(tracks: pd.DataFrame) -> pd.DataFrame:
    """Release year (Spotify) and artist listeners (Last.fm) from the enrichment passes, where present."""
    sp, lf = ENRICH["spotify"], ENRICH["lastfm"]
    tracks["release_year"] = np.nan
    tracks["listeners"] = np.nan
    if sp.exists():
        s = pd.read_csv(sp)
        s = s[s["hit"] == 1]
        key = s["artist"].astype(str).map(lambda a: re.sub(r"[^a-z0-9]", "", a.lower())) + s[
            "title"
        ].astype(str).map(lambda t: re.sub(r"[^a-z0-9]", "", t.lower()))
        year = pd.Series(
            pd.to_datetime(s["release_date"], errors="coerce").dt.year.values, index=key.values
        )
        year = year[~year.index.duplicated()]
        tracks["release_year"] = tracks["record_key"].map(year)
    if lf.exists():
        a = pd.read_csv(lf)
        a = a[a["hit"] == 1].drop_duplicates("ak").set_index("ak")["listeners"]
        first = (
            tracks["artist"]
            .fillna("")
            .map(
                lambda x: re.split(
                    r"\s*(?:&|,| x | vs\.? | feat\.? | ft\.? | pres\.? )\s*", x, flags=re.I
                )[0]
            )
        )
        tracks["listeners"] = first.map(lambda x: re.sub(r"[^a-z0-9]", "", x.lower())).map(a)
    return tracks


def export(date: str, out_dir: Path = DATASET_DIR) -> dict:
    """Join every source into data/dataset/<date>/ and write a README with the counts and input md5s."""
    out = Path(out_dir) / date
    out.mkdir(parents=True, exist_ok=True)

    t_old, p_old, s_old = old_tables()
    t_rv, p_rv, s_rv, l_rv = measured_tables("raveform")
    t_dj, p_dj, s_dj, l_dj = measured_tables("djs")

    # a record in two sources is one row: full audio wins over the preview
    tracks = pd.concat([t_rv, t_dj, t_old], ignore_index=True)
    tracks = tracks.drop_duplicates("track_id", keep="first").reset_index(drop=True)
    # a mix in two sources (140 DJ-data mixes are old-data mixes, same id) keeps its measured rows only
    measured_mixes = set(p_rv["mix_id"]) | set(p_dj["mix_id"])
    shared = int(p_old.loc[p_old["mix_id"].isin(measured_mixes), "mix_id"].nunique())
    p_old = p_old[~p_old["mix_id"].isin(measured_mixes)]
    s_old = s_old[~s_old["mix_id"].isin(measured_mixes)]
    plays = pd.concat([p_old, p_rv, p_dj], ignore_index=True)
    seams = pd.concat([s_old, s_rv, s_dj], ignore_index=True)
    seams = seams.drop_duplicates("seam_id").reset_index(
        drop=True
    )  # a record played twice in one old set
    layers = pd.concat([l_rv, l_dj], ignore_index=True)

    plays["dj_key"] = plays["dj"].map(dj_key)
    split = validation_split(plays)
    plays["split"] = plays["mix_id"].map(split)
    seams["split"] = seams["mix_id"].map(split)

    counts = plays.groupby("track_id").size()
    tracks["play_count"] = tracks["track_id"].map(counts).fillna(0).astype(int)
    tracks["dj_count"] = (
        tracks["track_id"].map(plays.groupby("track_id")["dj_key"].nunique()).fillna(0).astype(int)
    )
    tracks = enrich(tracks)

    tracks.to_parquet(out / "tracks.parquet", index=False)
    plays.to_parquet(out / "plays.parquet", index=False)
    seams.to_parquet(out / "seams.parquet", index=False)
    layers.to_parquet(out / "layers.parquet", index=False)

    # how much the two sides share: a track seen on both is the leak Step 4 wants reported
    by_side = plays.groupby("track_id")["split"].nunique()
    mixes = plays.drop_duplicates("mix_id")
    val = mixes[mixes["split"] == "validation"]
    val_seams = seams[(seams["split"] == "validation") & (seams.get("measured", 0) == 1)]
    report = {
        "date": date,
        "tracks": {s: int(n) for s, n in tracks["source"].value_counts().items()},
        "tracks_full_audio": int((tracks["audio"] == "full").sum()),
        "plays": {s: int(n) for s, n in plays["source"].value_counts().items()},
        "mixes": {
            s: int(n) for s, n in plays.drop_duplicates("mix_id")["source"].value_counts().items()
        },
        "mixes_shared_old_and_measured_kept_measured": shared,
        "seams": {s: int(n) for s, n in seams["source"].value_counts().items()},
        "seams_measured": int((seams.get("measured", pd.Series(dtype=float)) == 1).sum()),
        "seams_signature": {s: int(n) for s, n in seams["signature"].value_counts().items()}
        if "signature" in seams
        else {},
        "layers": int(len(layers)),
        "validation_mixes": int(len(val)),
        "validation_share_of_mixes": round(len(val) / len(mixes), 3),
        "validation_djs": int(val["dj_key"].nunique()),
        "validation_by_genre": {
            g: f"{int(n)} of {int(genre_total)}"
            for g, n, genre_total in zip(
                mixes["genre"].value_counts().index,
                val["genre"].value_counts().reindex(mixes["genre"].value_counts().index).fillna(0),
                mixes["genre"].value_counts(),
            )
        },
        "validation_by_source": {
            s: f"{int(val[val['source'] == s]['mix_id'].nunique())} of {int(n)}"
            for s, n in mixes["source"].value_counts().items()
        },
        "validation_measured_seams": f"{int(len(val_seams))} of {int((seams.get('measured', 0) == 1).sum())}",
        "tracks_on_both_sides": int((by_side > 1).sum()),
        "tracks_on_both_sides_share": round(float((by_side > 1).mean()), 4),
        "inputs": _md5s(),
    }
    (out / "README.md").write_text(_readme(report))
    log.info("dataset %s written to %s", date, out)
    return report


def _md5s() -> dict:
    files = [SOURCES["old"]["plays"], SOURCES["old"]["features"]]
    for s in ("raveform", "djs"):
        files += [
            SOURCES[s]["root"] / "out" / f
            for f in ("plays.csv", "seams_index.csv", "track_features.parquet")
        ]
    out = {}
    for f in files:
        if f.exists():
            h = hashlib.md5()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[str(f)] = h.hexdigest()[:10]
    return out


def _readme(r: dict) -> str:
    lines = [
        f"# Dataset {r['date']}",
        "",
        "Built by `djdata export-dataset`. Four files, every row carries `source`.",
        "",
    ]
    for k, v in r.items():
        if k in ("date", "inputs"):
            continue
        lines.append(f"- {k}: {json.dumps(v)}")
    lines += (
        ["", "Inputs (md5, first 10):", ""]
        + [f"- `{f}` {h}" for f, h in r["inputs"].items()]
        + [""]
    )
    return "\n".join(lines)
