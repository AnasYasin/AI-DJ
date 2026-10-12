"""The record list every enrichment pass works from: one row per distinct (artist, title) over the old tracklists, the
DJ-corpus tracklists and Raveform's. Keys are the letters of artist and title, so one record under two spellings is one row.
Timestamps "[12:34]" in front of Raveform titles and "[Label]" behind them are stripped."""

import json
import re

import pandas as pd


def clean_title(s):
    s = re.sub(r"^\[\d+(:\d+)?\]\s*", "", str(s))
    s = re.sub(r"\s*\[[^\]]*\]\s*$", "", s)
    return s.strip()


def first_artist(s):
    return re.split(
        r"\s*(&|,|\bx\b|\bvs\.?|\bfeat\.?|\bft\.?|\bpres\.?)\s*", str(s), maxsplit=1, flags=re.I
    )[0].strip()


def key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def records():
    rows = []
    d = pd.read_csv(
        "data/processed/tracklist_clean.csv",
        usecols=["artist_name", "track_name"],
        low_memory=False,
    ).dropna()
    rows += list(zip(d.artist_name, d.track_name))
    # the DJ corpus: only the sets that were run (data/interim/tracklist.csv holds every scraped set of the seven DJs)
    dj_mixes = set(pd.read_csv("data/djdata/djs/out/mixes.csv").mix_id)
    d = pd.read_csv(
        "data/interim/tracklist.csv",
        usecols=["mix_id", "artist_name", "track_name"],
        low_memory=False,
    ).dropna()
    d = d[d.mix_id.isin(dj_mixes)]
    rows += list(zip(d.artist_name, d.track_name))
    for m in map(json.loads, open("data/raw/raveform/raveform/mixes.jsonl")):
        for t in m["tracklist"]:
            title = clean_title(t["title"])
            if " - " in title:
                a, b = title.split(" - ", 1)
                rows.append((a.strip(), b.strip()))
    df = pd.DataFrame(rows, columns=["artist", "title"])
    df["title"] = df.title.map(clean_title)
    df = df[
        ~df.title.str.contains(r"^ID$|Unknown", case=False, regex=True)
        & ~df.artist.str.contains(r"^ID$|Unknown", case=False, regex=True)
    ]
    df["k"] = df.artist.map(key) + "|" + df.title.map(key)
    df = df[df.k.str.len() > 4].drop_duplicates("k")
    df["artist1"] = df.artist.map(first_artist)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    r = records()
    print(len(r), "records,", r.artist1.map(key).nunique(), "distinct first artists")
    print(r.sample(5, random_state=0).to_string(index=False))
