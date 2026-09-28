"""The 1001tracklists corpus: which mixes there are, which records each lists, and where the audio is.

A corpus is a folder (`root`) with `mixes_tmp/<mix_id>.<ext>` and `tracks/<track_id>.<ext>`, plus the
scrape in `tracklists_csv` (one row per listed record, in play order, with the listed minute). Audio is
found by glob on the id, never by a path stored anywhere, because stored paths went stale once.

Which mixes count is decided by what is on disk. A mix with no audio is not in the corpus today and will
be when its audio arrives, with no other change.
"""

from collections import defaultdict
import csv
from pathlib import Path
import random


def audio_file(folder: Path, item_id: str) -> Path | None:
    """The one audio file named by this id, whatever its extension."""
    hits = sorted(p for p in Path(folder).glob(f"{item_id}.*") if p.suffix != ".part")
    return hits[0] if hits else None


def listed(tracklists_csv: Path, extra: list | None = None) -> dict[str, list[dict]]:
    """mix_id -> its listed records in file order, each with the columns the plays table wants.

    `extra` names more CSVs in the same columns. A row whose `exact_time` column is 1 carries in
    `starting_time` the exact minute the record is heard inside the mix file (the USB002 app lists,
    2026-09-28), and the locator then searches it there (`near_s`). The 1001 scrape's `starting_time`
    is minutes past the hour and rolls over, so it is never exact."""
    out = defaultdict(list)
    for path in [tracklists_csv, *(extra or [])]:
        with open(path, newline="") as handle:
            for r in csv.DictReader(handle):
                try:
                    listed_min = float(r["starting_time"])
                except (TypeError, ValueError):
                    listed_min = None
                exact = r.get("exact_time", "") == "1" and listed_min is not None
                out[r["mix_id"]].append(
                    {
                        "mix_id": r["mix_id"],
                        "mix_title": r["mix_title"],
                        "dj": r["dj_name"],
                        "genre": r["genre"],
                        "track_id": r["track_id"],
                        "title": f"{r['artist_name']} - {r['track_name']}",
                        "listed_min": listed_min,
                        "play_type": r.get("play_type", ""),
                        "overlay_parent": r.get("overlay_parent", ""),
                        "near_s": round(60 * listed_min, 1) if exact else None,
                    }
                )
    for rows in out.values():
        for order, r in enumerate(rows, 1):
            r["order_listed"] = order
    return out


def mixes_on_disk(
    root: Path, tracklists_csv: Path, only: list[str] | None = None, extra: list | None = None
) -> list[dict]:
    """Every mix with audio in mixes_tmp and a tracklist, with its records that have audio in tracks/.
    [{mix_id, mix_title, dj, genre, path, tracks: [{track_id, path, title, ...}], missing: [track_id]}]"""
    root = Path(root)
    by_mix = listed(tracklists_csv, extra)
    out = []
    for mix_id, rows in by_mix.items():
        if only and mix_id not in only:
            continue
        mix_path = audio_file(root / "mixes_tmp", mix_id)
        if mix_path is None:
            continue
        tracks, missing = [], []
        for r in rows:
            path = audio_file(root / "tracks", r["track_id"])
            if path is None:
                missing.append(r["track_id"])
                continue
            tracks.append({**r, "path": path})
        out.append(
            {
                "mix_id": mix_id,
                "mix_title": rows[0]["mix_title"],
                "dj": rows[0]["dj"],
                "genre": rows[0]["genre"],
                "path": mix_path,
                "tracks": tracks,
                "missing": missing,
            }
        )
    return out


def controls_for(mix: dict, pool: list[dict], n: int, exclude: set | None = None) -> list[dict]:
    """`n` records from other mixes, chosen by a seed from the mix id so the choice does not depend on
    which mixes ran before it. `pool` is [{track_id, path}] over the whole corpus. `exclude` is more
    track ids never to use, for example every record the mix's own DJ lists anywhere."""
    own = {t["track_id"] for t in mix["tracks"]} | (exclude or set())
    candidates = sorted((c for c in pool if c["track_id"] not in own), key=lambda c: c["track_id"])
    rng = random.Random(mix["mix_id"])
    rng.shuffle(candidates)
    return candidates[:n]


def other_dj_pool(
    root: Path, tracklists_csv: Path, control_djs: list[str], own_djs: set
) -> list[dict]:
    """Control records from other DJs: every record listed under `control_djs` with audio in tracks/,
    minus any record a mix by one of `own_djs` lists.

    Why (2026-09-28). A corpus of one DJ's tour draws its controls from his other shows, and he plays
    the same records every night, so a control can be a real hit and lift the floor. With one mix on
    disk there are no other mixes at all, no control runs, and the floor falls to its minimum."""
    root = Path(root)
    wanted, own = set(control_djs), set()
    candidates = {}
    for mix_rows in listed(tracklists_csv).values():
        dj = mix_rows[0]["dj"]
        for r in mix_rows:
            if dj in own_djs:
                own.add(r["track_id"])
            elif dj in wanted:
                candidates.setdefault(r["track_id"], None)
    out = []
    for track_id in sorted(set(candidates) - own):
        path = audio_file(root / "tracks", track_id)
        if path is not None:
            out.append({"track_id": track_id, "path": path})
    return out


def dj_records(tracklists_csv: Path, djs: list, extra: list | None = None) -> set:
    """Every track id any mix by these DJs lists."""
    wanted = set(djs)
    return {
        r["track_id"]
        for rows in listed(tracklists_csv, extra).values()
        if rows[0]["dj"] in wanted
        for r in rows
    }


def track_pool(mixes: list[dict]) -> list[dict]:
    seen = {}
    for m in mixes:
        for t in m["tracks"]:
            seen.setdefault(t["track_id"], {"track_id": t["track_id"], "path": t["path"]})
    return list(seen.values())
