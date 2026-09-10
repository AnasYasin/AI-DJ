"""Seam manifest from our own 1001tracklists scrape (data/interim/tracklist.csv), for DJs Raveform
lacks (Black Coffee, Fred again..).

The scrape gives track order and start times to the minute but no track audio ids and no position
inside each track. So this path needs two extra stages the Raveform path does not:
  1. media_link: read the mix audio URL from the 1001 mix page (browser, see fetch/media_link.py)
  2. coarse anchors from the audio: once the window and both tracks are on disk, the landmark
     fingerprint (legacy track_fetcher.verify_match) gives each track's position in the window to
     within about a bar. seam/coarse.py does that before the analyser runs.
Until stage 2 runs, the seam's coarse anchors carry only the tracklist minutes and `needs_fingerprint`.
Tracks are fetched by name with the legacy fetch_track (YouTube search + preview fingerprint check).
"""

import logging

import pandas as pd

from ..state import State

log = logging.getLogger("djdata.manifest.tracklists")


def build(cfg, state: State, djs: list[str], min_tracks: int = 8, min_timed: float = 0.95) -> dict:
    df = pd.read_csv(cfg.tracklists_csv)
    df = df[df["dj_name"].str.lower().isin([d.lower() for d in djs])]
    n_seams, n_mixes = 0, 0
    for mid, g in df.groupby("mix_id"):
        g = g.sort_values("starting_time")
        if len(g) < min_tracks or g["starting_time"].notna().mean() < min_timed:
            continue
        rows = g.to_dict("records")
        added = 0
        for a, b in zip(rows, rows[1:]):
            if pd.isna(a["starting_time"]) or pd.isna(b["starting_time"]):
                continue
            a_start, b_start = float(a["starting_time"]) * 60, float(b["starting_time"]) * 60
            nxt = [r for r in rows if r["starting_time"] > b["starting_time"]]
            b_end = float(nxt[0]["starting_time"]) * 60 if nxt else b_start + 360
            coarse = {
                "needs_fingerprint": True,
                "A": {"orig_t": None, "mix_t": None, "rate": 1.0, "start_mix_t": a_start},
                "B": {"orig_t": None, "mix_t": None, "rate": 1.0, "start_mix_t": b_start},
                "overlap_start_mix_t": b_start, "overlap_end_mix_t": b_start + 60,
                "window": [max(a_start, b_start - 150), min(b_end, b_start + 150)],
            }
            for r in (a, b):
                state.add_track(r["track_id"], f"{r['artist_name']} - {r['track_name']}", None, None)
            state.add_seam(f"{mid}_{a['track_id']}_{b['track_id']}", mid, a["track_id"], b["track_id"],
                           "tracklists", "tracklists", coarse)
            added += 1
        if added:
            state.add_mix(mid, rows[0]["mix_title"], None, "1001tracklists", None, [rows[0]["genre"]])
            n_mixes += 1
            n_seams += added
    log.info("tracklists manifest: %d seams in %d mixes for %s", n_seams, n_mixes, djs)
    return {"seams": n_seams, "mixes": n_mixes}
