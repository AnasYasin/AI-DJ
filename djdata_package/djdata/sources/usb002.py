"""The USB002 marathon tracklists from usb002-tracklist.app, as prove lists.

The app's `data.json` (saved 2026-09-28 as `fred/lists/usb002_app_2026-09-28.json`) holds 20 shows and
103 sets, each set with its start and end in marathon seconds and every track with the marathon second
it is heard at. The set times equal our `segments.csv` cut times, so a track's time inside a segment
file is its marathon second minus the segment's start, and that is written as `listed_min`.

`usb002_solo_tracks.csv` maps each unique (artist, title) of the solo sets to a track id: a 1001 id
already on disk when title and an artist agree, else a `usb_` id queued for fetching.
"""

import csv
import json
import re


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def solo_lists(app_json, tracks_csv, segments_csv) -> tuple[list[dict], list[dict]]:
    """(PROOF_LISTS rows, candidates rows) for every solo set that has a segment file and tracks.

    One list per solo set, id `usb_<segment file stem>`, records in the app's time order. The
    candidate for each list is its own segment (`file_id` = segment stem)."""
    app = json.load(open(app_json))
    ids = {
        (norm(r["artist"]), norm(r["title"])): r["track_id"]
        for r in csv.DictReader(open(tracks_csv, newline=""))
    }
    segments = {
        int(r["marathon_start_s"]): r
        for r in csv.DictReader(open(segments_csv, newline=""))
        if r["is_b2b"] == "0"
    }
    lists, candidates = [], []
    for show in app["shows"]:
        for st in show["sets"]:
            if st["name"].strip().lower() != "fred":
                continue
            seg = segments.get(int(st["global_start"]))
            tracks = sorted(st["tracks"], key=lambda x: x["t"])
            if seg is None or not tracks:
                continue
            stem = seg["file"].rsplit(".", 1)[0]
            list_id = f"usb_{stem}"
            start = int(seg["marathon_start_s"])
            for order, x in enumerate(tracks, 1):
                lists.append(
                    {
                        "list_id": list_id,
                        "dj": "Fredagain..",
                        "source": "usb002",
                        "order_listed": order,
                        "track_id": ids[(norm(x["artist"]), norm(x["title"]))],
                        "title": f"{x['artist']} - {x['title']}",
                        "listed_min": round((x["t"] - start) / 60, 3),
                        "track_path": "",
                    }
                )
            candidates.append({"file_id": stem, "path": seg["file"], "list_id": list_id})
    return lists, candidates


TRACKLIST_COLUMNS = [
    "mix_id",
    "mix_title",
    "dj_name",
    "genre",
    "url",
    "track_id",
    "starting_time",
    "track_name",
    "artist_name",
    "play_type",
    "overlay_parent",
    "exact_time",
]
MARATHON_URL = "https://www.youtube.com/watch?v=GiXKukOtmeE"


def tracklist_rows(app_json, tracks_csv, segments_csv, genre: str = "house") -> list[dict]:
    """The solo lists in the tracklist.csv columns plus `exact_time` = 1, so the DJ corpus reads them
    beside the 1001 scrape (config `extra_tracklists`). `starting_time` is the exact minute inside the
    segment file. Mix id `usb_<segment stem>`, the name its audio is linked under in mixes_tmp."""
    lists, _ = solo_lists(app_json, tracks_csv, segments_csv)
    shows = {
        f"usb_{r['file'].rsplit('.', 1)[0]}": r.get("show", "")
        for r in csv.DictReader(open(segments_csv, newline=""))
    }
    out = []
    for r in lists:
        artist, _, title = r["title"].partition(" - ")
        out.append(
            {
                "mix_id": r["list_id"],
                "mix_title": f"Fred again.. @ USB002 {shows.get(r['list_id'], '')} (solo set, marathon cut)",
                "dj_name": "Fredagain..",
                "genre": genre,
                "url": MARATHON_URL,
                "track_id": r["track_id"],
                "starting_time": r["listed_min"],
                "track_name": title,
                "artist_name": artist,
                "play_type": "",
                "overlay_parent": "",
                "exact_time": 1,
            }
        )
    return out
