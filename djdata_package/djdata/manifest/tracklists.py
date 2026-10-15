"""Seam manifest from our own 1001tracklists scrape (data/interim/tracklist.csv), for the DJs Raveform
lacks or barely covers.

The scrape gives the play order and, for most mixes, a start time to the minute. It gives no track audio
ids and no position inside each track, so this path finds both from the audio after the mix and its
tracks are on disk: seam/locate.py fingerprints every track against the mix and returns the mix time its
time zero falls on and the speed it was played at. Windows are cut from those positions, not from the
listed minutes, which are only a search hint (and can be absent entirely).

Two DJ lists in the config, because the rule that pays for itself on a DJ with a hundred mixes is a waste
on one with six:
  djs_timed  mixes must have start times on `min_timed` of their tracks (cheap, narrow search)
  djs_all    every mix with a tracklist, timed or not (the whole mix is searched where times are missing)
"""

import logging

import pandas as pd

from ..state import State

log = logging.getLogger("djdata.manifest.tracklists")


def build(cfg, state: State, djs_timed: list[str], djs_all: list[str] = (),
          min_tracks: int = 8, min_timed: float = 0.95) -> dict:
    df = pd.read_csv(cfg.tracklists_csv)
    timed = {d.lower() for d in djs_timed}
    everything = {d.lower() for d in djs_all}
    df = df[df["dj_name"].str.lower().isin(timed | everything)]

    excluded = _excluded_mixes(cfg.root)
    n_seams = n_mixes = n_untimed = 0
    for mix_id, group in df.groupby("mix_id"):
        dj = str(group["dj_name"].iloc[0]).lower()
        if mix_id in excluded or len(group) < min_tracks:
            continue
        timed_share = group["starting_time"].notna().mean()
        usable_times = _times_usable(group["starting_time"])
        # 236 of 881 scraped mixes carry 0.0 for every track: the page listed no times and the scraper
        # wrote a zero. That reads as "fully timed" and would point every search at the first seconds of
        # the mix, so the times are dropped and the whole mix is searched instead.
        if dj in timed and dj not in everything and not (usable_times and timed_share >= min_timed):
            continue
        group = group.sort_values("starting_time", na_position="last") if usable_times else group
        rows = group.to_dict("records")
        if not usable_times:
            n_untimed += 1

        added = 0
        for a, b in zip(rows, rows[1:]):
            if a["track_id"] == b["track_id"]:
                continue
            coarse = {
                "needs_locate": True,
                "A": {"track_id": a["track_id"], "listed_mix_t": _minutes(a["starting_time"]) if usable_times else None},
                "B": {"track_id": b["track_id"], "listed_mix_t": _minutes(b["starting_time"]) if usable_times else None},
            }
            for row in (a, b):
                state.add_track(row["track_id"], f"{row['artist_name']} - {row['track_name']}", None, None)
            state.add_seam(f"{mix_id}_{a['track_id']}_{b['track_id']}", mix_id, a["track_id"], b["track_id"],
                           "tracklists", "tracklists", coarse)
            added += 1
        if added:
            state.add_mix(mix_id, rows[0]["mix_title"], None, "1001tracklists", None, [rows[0]["genre"]])
            n_mixes += 1
            n_seams += added
    log.info("tracklists manifest: %d seams in %d mixes (%d without start times)", n_seams, n_mixes, n_untimed)
    return {"seams": n_seams, "mixes": n_mixes, "untimed_mixes": n_untimed}


def _times_usable(times) -> bool:
    """Start times are worth searching around only when at least five are listed, they are not all zero,
    and they run forwards. 13 mixes in 397 step backwards (the hourly rollover in the scrape) and those
    are searched whole rather than guessed at."""
    values = times.dropna().tolist()
    if len(values) < 5 or max(values) <= 0:
        return False
    return all(b >= a for a, b in zip(values, values[1:]))


def _minutes(value) -> float | None:
    """`starting_time` is minutes into the mix, and it rolls over every hour, so it is a hint only."""
    return None if pd.isna(value) else float(value) * 60.0


def _excluded_mixes(root) -> set:
    """Mix ids Anas has dropped from the DJ-profiling set for good, in `<root>/lists/dj_mixes_excluded.csv`
    (mix_id, dj, mix_title, page_url_1001, reason, excluded_on). 208 radio shows on 2026-09-23. The
    manifest skips them, and `scripts/diag/apply_exclusions.py` removes them from a database that already
    holds them, so a rebuild cannot bring them back."""
    path = root / "lists" / "dj_mixes_excluded.csv"
    if not path.exists():
        return set()
    excluded = set(pd.read_csv(path)["mix_id"].astype(str))
    log.info("tracklists manifest: %d mixes on the exclusion list", len(excluded))
    return excluded
