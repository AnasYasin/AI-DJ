"""Seam manifest from the Raveform files (mixes.jsonl, tracks.jsonl, alignments/*.jsonl).

A seam is two consecutive identified tracks of one mix. The Raveform alignment gives each track's
mix-in and mix-out in both mix time and track time plus the playback rate. Those become the coarse
anchors the analyser refines. Raveform's edges are approximate (DTW marks where a record becomes
dominant), so the analysis window is defined from the tracks themselves: it starts where the
incoming track's first beat maps into the mix and ends where the outgoing track's last beat maps out,
bounded by where each record is known to play alone.
"""

import json
import logging
from pathlib import Path
import re

from ..state import State

log = logging.getLogger("djdata.manifest.raveform")


def _year(title: str):
    m = re.match(r"(\d{4})", title)
    return int(m.group(1)) if m else None


def _tier_of(title: str, genres: list, year, tiers: dict, order: list):
    t = title.lower()
    for name in order:
        spec = tiers[name]
        if year is None or year < spec.get("min_year", 0):
            continue
        if "djs" in spec and any(d.lower() in t for d in spec["djs"]):
            return name
        if "genres" in spec and set(spec["genres"]) & set(genres or []):
            return name
    return None


def _coarse(a: dict, b: dict, a_dur, b_dur, pad_s: float, alone_s: float) -> dict:
    """Anchors and window for one seam. All mix times absolute (seconds into the mix)."""
    ra = a["matched_time_track"] / a["matched_time_mix"]
    rb = b["matched_time_track"] / b["matched_time_mix"]
    ov_start, ov_end = b["mixin_time_mix"], a["mixout_time_mix"]
    # where B's first beat and A's last beat fall in mix time, along each record's own mapping
    b_track0_mix = b["mixin_time_mix"] - b["mixin_time_track"] / rb
    a_trackend_mix = a["mixout_time_mix"] + ((a_dur or a["mixout_time_track"]) - a["mixout_time_track"]) / ra
    t0 = max(a["mixin_time_mix"], min(b_track0_mix - pad_s, ov_start - alone_s))
    t1 = min(b["mixout_time_mix"], max(a_trackend_mix + pad_s, ov_end + alone_s))
    return {
        "A": {"orig_t": a["mixin_time_track"], "mix_t": a["mixin_time_mix"], "rate": ra, "duration": a_dur},
        "B": {"orig_t": b["mixin_time_track"], "mix_t": b["mixin_time_mix"], "rate": rb, "duration": b_dur},
        "overlap_start_mix_t": ov_start, "overlap_end_mix_t": ov_end,
        "window": [t0, t1], "match_rate": [a["match_rate"], b["match_rate"]],
        "b_track_start_mix_t": b_track0_mix, "a_track_end_mix_t": a_trackend_mix,
    }


def build(cfg, state: State) -> dict:
    R = Path(cfg.raveform_dir)
    mixes = {m["id"]: m for m in map(json.loads, open(R / "mixes.jsonl"))}
    tracks = {t["id"]: t for t in map(json.loads, open(R / "tracks.jsonl"))}
    u = cfg.usable
    order = list(cfg.tiers.keys())
    n_seams, n_mixes, tier_counts = 0, 0, {}
    for f in sorted((R / "alignments").glob("*.jsonl")):
        als = [json.loads(l) for l in open(f)]
        if not als:
            continue
        mid = als[0]["mix_id"]
        m = mixes[mid]
        year = _year(m["title"])
        tier = _tier_of(m["title"], m.get("genres"), year, cfg.tiers, order)
        if tier is None or (year or 0) < u["min_year"]:
            continue
        ids = {t["id"] for t in m["tracklist"] if t["id"]}
        als = sorted((a for a in als if a["track_id"] in ids), key=lambda a: a["mixin_time_mix"])
        added = 0
        for a, b in zip(als, als[1:]):
            if min(a["match_rate"], b["match_rate"]) < u["min_match_rate"]:
                continue
            if a["mixout_time_mix"] - b["mixin_time_mix"] < -u["max_gap_s"]:
                continue
            if a["track_id"] not in tracks or b["track_id"] not in tracks:
                continue
            a_dur, b_dur = tracks[a["track_id"]].get("duration"), tracks[b["track_id"]].get("duration")
            coarse = _coarse(a, b, a_dur, b_dur, cfg.window["pad_s"], cfg.window["alone_s"])
            seam_id = f"{mid}_{a['track_id']}_{b['track_id']}"
            for tid in (a["track_id"], b["track_id"]):
                state.add_track(tid, tracks[tid]["title"], f"https://www.youtube.com/watch?v={tid}", tracks[tid].get("duration"))
            state.add_seam(seam_id, mid, a["track_id"], b["track_id"], tier, "raveform", coarse)
            added += 1
        if added:
            state.add_mix(mid, m["title"], m["audio_url"], m["audio_source"], year, m.get("genres") or [])
            n_mixes += 1
            n_seams += added
            tier_counts[tier] = tier_counts.get(tier, 0) + added
    log.info("manifest: %d seams in %d mixes; per tier %s", n_seams, n_mixes, tier_counts)
    return {"seams": n_seams, "mixes": n_mixes, "tiers": tier_counts}
