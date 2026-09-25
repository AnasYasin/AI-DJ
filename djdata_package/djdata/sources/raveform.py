"""The Raveform corpus: the alignment it ships stands in for locate, its windows stand in for the cut.

Raveform gives, per record in a mix, where it is mixed in and out in both mix time and record time,
and the playback rate. The track id is the YouTube video id, so there is no wrong-recording risk, and
the alignment lands within 0.06 s of the fingerprint (60 seams, 2026-09-18). So the plays table is
written straight from the alignment: time zero is the mix-in mix time less the mix-in record time
over the rate, first heard is the mix-in, last heard the mix-out.

The full mixes are gone; one window per seam was cut by the manifest's rule and is on disk under
windows/, with its start time in the seam's coarse JSON in state.sqlite. The cut stage for this corpus
adopts those files and audits them the same way.
"""

import json
from pathlib import Path
import sqlite3

from .tracklists import audio_file


def alignments(raveform_dir: Path) -> dict[str, list[dict]]:
    """mix_id -> its alignment rows sorted by mix-in time."""
    out = {}
    for f in sorted((Path(raveform_dir) / "alignments").glob("*.jsonl")):
        rows = [json.loads(line) for line in open(f) if line.strip()]
        if rows:
            out[rows[0]["mix_id"]] = sorted(rows, key=lambda r: r["mixin_time_mix"])
    return out


def mixes_meta(raveform_dir: Path) -> tuple[dict, dict]:
    R = Path(raveform_dir)
    mixes = {m["id"]: m for m in map(json.loads, open(R / "mixes.jsonl"))}
    tracks = {t["id"]: t for t in map(json.loads, open(R / "tracks.jsonl"))}
    return mixes, tracks


def seam_windows(db_path: Path) -> dict[str, dict]:
    """seam_id -> {mix_id, a, b, window_t0, window_t1} from the manifest's coarse JSON."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = {}
    for seam_id, mix_id, a, b, coarse in con.execute(
        "select seam_id, mix_id, a, b, coarse from seams"
    ):
        c = json.loads(coarse)
        if "window" not in c:
            continue
        out[seam_id] = {
            "mix_id": mix_id,
            "a": a,
            "b": b,
            "window_t0": float(c["window"][0]),
            "window_t1": float(c["window"][1]),
        }
    return out


def plays_rows(cfg, only: list[str] | None = None) -> list[dict]:
    """Plays rows for every Raveform mix that has at least one seam window on disk, from the alignment.
    Records with no audio in tracks/ are still rows, marked not found, so nothing is silent."""
    mixes, tracks = mixes_meta(cfg.raveform_dir)
    windows = seam_windows(cfg.db_path)
    on_disk = {}
    for seam_id, w in windows.items():
        if audio_file(cfg.dirs["windows"], seam_id) is not None:
            on_disk.setdefault(w["mix_id"], set()).update((w["a"], w["b"]))
    rows = []
    for mix_id, als in alignments(cfg.raveform_dir).items():
        if mix_id not in on_disk or (only and mix_id not in only):
            continue
        m = mixes[mix_id]
        listed = {t["id"]: i for i, t in enumerate(m["tracklist"], 1) if t.get("id")}
        mix_len = max(a["mixout_time_mix"] for a in als)
        for a in als:
            tid = a["track_id"]
            if tid not in listed:
                continue
            rate = a["matched_time_track"] / a["matched_time_mix"]
            has_audio = audio_file(cfg.dirs["tracks"], tid) is not None
            in_seams = tid in on_disk[mix_id]
            rows.append(
                {
                    "mix_id": mix_id,
                    "dj": m["title"].split(" - ")[0][:60],
                    "genre": " ".join(m.get("genres") or []),
                    "mix_title": m["title"],
                    "mix_minutes": round(mix_len / 60, 1),
                    "track_id": tid,
                    "title": tracks.get(tid, {}).get("title", ""),
                    "is_control": 0,
                    "order_listed": listed[tid],
                    "listed_min": None,
                    "play_type": "sequential",
                    "overlay_parent": "",
                    "time_zero_s": round(a["mixin_time_mix"] - a["mixin_time_track"] / rate, 2),
                    "rate": round(rate, 4),
                    "votes": None,
                    "floor": None,
                    "control_max": None,
                    "control_n": 0,
                    "sections": None,
                    "sections_agree": None,
                    "confidence": "alignment"
                    if has_audio and in_seams
                    else "no audio"
                    if not has_audio
                    else "no window",
                    "found": int(has_audio and in_seams),
                    "drift_min": None,
                    "first_heard_s": round(a["mixin_time_mix"], 1),
                    "last_heard_s": round(a["mixout_time_mix"], 1),
                    "played_from_s": round(a["mixin_time_track"], 1),
                    "played_to_s": round(a["mixout_time_track"], 1),
                    "sweep_votes": None,
                    "sweep_floor": None,
                    "track_len_s": tracks.get(tid, {}).get("duration"),
                    "seconds": 0.0,
                }
            )
    return rows


def mixes_on_disk(cfg, only: list[str] | None = None) -> list[dict]:
    """The same shape as tracklists.mixes_on_disk, for the stages that need each mix's records on
    disk. There is no full mix audio, so `path` is None."""
    mixes, tracks = mixes_meta(cfg.raveform_dir)
    windows = seam_windows(cfg.db_path)
    by_mix = {}
    for seam_id, w in windows.items():
        if audio_file(cfg.dirs["windows"], seam_id) is not None:
            by_mix.setdefault(w["mix_id"], set()).update((w["a"], w["b"]))
    out = []
    for mix_id, ids in by_mix.items():
        if only and mix_id not in only:
            continue
        m = mixes[mix_id]
        have, missing = [], []
        for tid in sorted(ids):
            path = audio_file(cfg.dirs["tracks"], tid)
            if path is None:
                missing.append(tid)
            else:
                have.append(
                    {"track_id": tid, "path": path, "title": tracks.get(tid, {}).get("title", "")}
                )
        out.append(
            {
                "mix_id": mix_id,
                "mix_title": m["title"],
                "dj": m["title"].split(" - ")[0][:60],
                "genre": " ".join(m.get("genres") or []),
                "path": None,
                "tracks": have,
                "missing": missing,
            }
        )
    return out
