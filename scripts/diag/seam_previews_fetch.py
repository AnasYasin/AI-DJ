"""Pass 1 of D4 for the seam corpus: fetch an iTunes 30 s preview for every seam track, paced so iTunes
does not block us. The fingerprint check against the full file and the feature extraction are pass 2.

Pacing: one lookup every GAP_S seconds, sequential. On a 429 or 403 the gap doubles for the rest of the run
and the lookup is retried after a backoff. Among the iTunes hits, the one whose duration is closest to the
YouTube track's duration wins, which drops most radio edits and other versions before the fingerprint pass.

  python -m scripts.diag.seam_previews_fetch --out data/interim/seam_previews
Resumable: tracks already in the log are skipped.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import time

import httpx
import pandas as pd

SCRATCH = Path(
    "/tmp/claude-1000/-home-anas-yasin-projects-AI-DJ/1d82dd81-ee4c-48ed-bc0a-a609818b1249/scratchpad"
)
GAP_S = 1.5
BACKOFF_S = [30, 60, 120]
DURATION_TOL = 0.08  # 8 % either side counts as the same length


def clean_title(raveform_title: str) -> tuple[str, str]:
    title = re.sub(r"^\[\d+(:\d+)*\]\s*", "", raveform_title)  # leading [mm:ss]
    title = re.sub(r"\[.*?\]", "", title).strip()  # [label]
    artist, _, track = title.partition(" - ")
    return artist.strip(), track.strip()


def seam_tracks() -> pd.DataFrame:
    usable = pd.read_csv(SCRATCH / "usable_seams.csv")
    wanted = set(usable.a) | set(usable.b)
    rows = []
    for line in open("data/raw/raveform/raveform/tracks.jsonl"):
        record = json.loads(line)
        if record["id"] in wanted:
            artist, track = clean_title(record["title"])
            rows.append(
                {
                    "yt": record["id"],
                    "artist": artist,
                    "track": track,
                    "duration_s": record.get("duration"),
                }
            )
    return pd.DataFrame(rows)


class Pacer:
    def __init__(self, gap_s: float):
        self.gap_s, self.last = gap_s, 0.0

    def wait(self):
        elapsed = time.monotonic() - self.last
        if elapsed < self.gap_s:
            time.sleep(self.gap_s - elapsed)
        self.last = time.monotonic()

    def slow_down(self):
        self.gap_s = min(self.gap_s * 2, 30.0)


def lookup(
    client: httpx.Client, pacer: Pacer, artist: str, track: str, duration_s: float | None
) -> dict:
    """iTunes search; returns the best hit's preview url and metadata, or the reason there is none."""
    term = f"{artist} {track}".strip()
    for attempt, backoff in enumerate([0] + BACKOFF_S):
        if backoff:
            time.sleep(backoff)
        pacer.wait()
        try:
            response = client.get(
                "https://itunes.apple.com/search",
                params={"term": term, "media": "music", "entity": "song", "limit": 5},
                timeout=20,
            )
        except Exception as error:  # noqa: BLE001
            return {"status": f"error {type(error).__name__}"}
        if response.status_code in (429, 403):
            pacer.slow_down()
            continue
        if response.status_code != 200:
            return {"status": f"http {response.status_code}"}
        hits = [h for h in response.json().get("results", []) if h.get("previewUrl")]
        if not hits:
            return {"status": "no hit"}
        if duration_s:

            def gap(hit):
                return abs(hit.get("trackTimeMillis", 0) / 1000 - duration_s) / duration_s

            hits.sort(key=gap)
            best = hits[0]
            duration_ok = gap(best) <= DURATION_TOL
        else:
            best, duration_ok = hits[0], None
        return {
            "status": "found",
            "preview_url": best["previewUrl"],
            "itunes_artist": best.get("artistName"),
            "itunes_track": best.get("trackName"),
            "itunes_duration_s": round(best.get("trackTimeMillis", 0) / 1000, 1),
            "duration_ok": duration_ok,
        }
    return {"status": "rate limited"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/interim/seam_previews")
    args = parser.parse_args()
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    log_path = out / "fetch_log.csv"
    done = set(pd.read_csv(log_path).yt) if log_path.exists() else set()
    tracks = seam_tracks()
    todo = tracks[~tracks.yt.isin(done)]
    print(f"seam tracks {len(tracks)}, already logged {len(done)}, to do {len(todo)}", flush=True)

    fields = [
        "yt",
        "artist",
        "track",
        "duration_s",
        "status",
        "itunes_artist",
        "itunes_track",
        "itunes_duration_s",
        "duration_ok",
        "path",
    ]
    new_log = not log_path.exists()
    pacer = Pacer(GAP_S)
    with (
        open(log_path, "a", newline="") as log_file,
        httpx.Client(headers={"User-Agent": "Mozilla/5.0"}) as client,
    ):
        writer = csv.DictWriter(log_file, fieldnames=fields, extrasaction="ignore")
        if new_log:
            writer.writeheader()
        for i, row in enumerate(todo.to_dict("records"), 1):
            result = lookup(client, pacer, row["artist"], row["track"], row["duration_s"])
            row.update(result)
            if result["status"] == "found":
                dest = out / "audio" / f"{row['yt']}.m4a"
                try:
                    dest.write_bytes(client.get(result["preview_url"], timeout=30).content)
                    row["path"] = str(dest)
                except Exception as error:  # noqa: BLE001
                    row["status"] = f"download error {type(error).__name__}"
            writer.writerow(row)
            log_file.flush()
            if i % 50 == 0 or i == len(todo):
                print(
                    f"{i}/{len(todo)}  gap {pacer.gap_s:.1f}s  last: {row['status']}  {row['artist']} - {row['track'][:40]}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
