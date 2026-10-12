"""Fetch an iTunes 30 s preview for every DJ-run track that has none, so its download can be checked.

Why these tracks have no preview. 4,396 of the run's tracks overlap the old dataset and already have one
in `data/raw/previews/`. The rest do not, and they are the more obscure records, which is where a YouTube
search by name is most likely to return something else. 274 of the 1,716 that could be checked, 16 %, are
the wrong recording, and the 1,442 that could not are unmeasured.

Same method as `seam_previews_fetch.py`, which fetched 5,989 previews for the seam corpus without being
blocked. Among the iTunes hits the one closest in length to the downloaded file wins, which drops most
radio edits before the fingerprint ever runs.

Pacing and workers. iTunes rate-limits the search per address, so the request rate is what has to be
held, not the worker count. One Pacer is shared by every worker and hands out one search slot every
gap seconds, so throughput is 1/gap whatever --workers says. The workers exist so the preview download of
one track overlaps the search of the next, since the download is a CDN fetch and is not rate-limited.

The gap adapts. A 429 or 403 doubles it, and a run of clean replies eases it back down, so one refusal
does not cripple the rest of the run. A refused track is NOT slept on inline: it returns at once and goes
into a second pass at the end, at whatever gap the first pass settled on. Measured 2026-09-21: a fixed
0.35 s gap with inline backoffs refused 16 of the first 96 tracks and dropped the rate to 3 a minute,
because eight workers each sat in a 30, 60, 120 second sleep. 1.5 s fetched 5,989 seam previews with no
refusal at all.

  python -m scripts.diag.dj_previews_fetch --out data/interim/dj_previews --workers 8 --gap 0.8
Resumable: tracks already in the log are skipped.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import sqlite3
import threading
import time

import httpx

ROOT = Path("data/djdata/djs")
GAP_S = 0.8
GAP_MIN, GAP_MAX = (
    1.0,
    8.0,
)  # never ease below 1 s: 1.5 s fetched 5,989 seam previews cleanly, and a
# 0.35 s burst had the address refused outright for about 35 minutes
EASE_AFTER = 40  # clean replies in a row before the gap is eased back down
RETRY_WAIT_S = 5.0  # one short wait on a refusal; anything still refused goes to the second pass
DURATION_TOL = 0.08  # 8 % either side counts as the same length


def tracks_without_a_preview(extra_preview_dir: Path | None) -> list[dict]:
    """Every downloaded track with no 30 s preview anywhere to check it against."""
    import sys

    sys.path.insert(0, "djdata_package")
    from djdata.legacy.track_fetcher import preview_paths

    have = preview_paths()
    con = sqlite3.connect(ROOT / "state.sqlite")
    con.row_factory = sqlite3.Row
    rows = []
    for record in con.execute(
        "SELECT track_id, title, duration, path FROM tracks WHERE status='ready'"
    ):
        track_id = record["track_id"]
        if track_id in have and Path(have[track_id]).exists():
            continue
        if extra_preview_dir and (extra_preview_dir / f"{track_id}.m4a").exists():
            continue
        if not Path(record["path"]).exists():
            continue
        artist, _, name = (record["title"] or "").partition(" - ")
        rows.append(
            {
                "track_id": track_id,
                "artist": artist.strip(),
                "track": name.strip(),
                "duration_s": record["duration"],
            }
        )
    return rows


class Pacer:
    """One search slot every gap_s seconds, shared by every worker. The lock is what makes the rate
    a property of the run rather than of the worker count. The gap widens on a refusal and eases back
    down after a run of clean replies, so the run finds the rate iTunes will take."""

    def __init__(self, gap_s: float):
        self.gap_s, self.next_at, self.clean, self.refusals = gap_s, 0.0, 0, 0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            slot = max(time.monotonic(), self.next_at)
            self.next_at = slot + self.gap_s
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def refused(self):
        with self._lock:
            self.gap_s = min(self.gap_s * 2, GAP_MAX)
            self.clean = 0
            self.refusals += 1

    def ok(self):
        with self._lock:
            self.clean += 1
            if self.clean >= EASE_AFTER and self.gap_s > GAP_MIN:
                self.gap_s = max(self.gap_s * 0.8, GAP_MIN)
                self.clean = 0


def lookup(client: httpx.Client, pacer: Pacer, artist: str, track: str, duration_s) -> dict:
    """iTunes search; the best hit's preview url and metadata, or the reason there is none."""
    term = f"{artist} {track}".strip()
    for backoff in (0.0, RETRY_WAIT_S):
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
            pacer.refused()
            continue
        if response.status_code != 200:
            return {"status": f"http {response.status_code}"}
        pacer.ok()
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
    parser.add_argument("--out", default="data/interim/dj_previews")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--tracks",
        default=None,
        help="CSV of track_id,artist,track,duration_s, for running away from the run database",
    )
    parser.add_argument(
        "--gap", type=float, default=GAP_S, help="seconds between searches, shared by all workers"
    )
    args = parser.parse_args()

    out = Path(args.out)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    log_path = out / "fetch_log.csv"

    # a refusal is not an answer, so those rows are retried rather than treated as done
    def answered(status: str) -> bool:
        if not status:
            return False
        return not status.startswith(("rate limited", "download error", "error ", "http "))

    done = set()
    if log_path.exists():
        with open(log_path) as handle:
            done = {r["track_id"] for r in csv.DictReader(handle) if answered(r["status"] or "")}

    if args.tracks:
        with open(args.tracks) as handle:
            wanted = [
                {
                    "track_id": r["track_id"],
                    "artist": r["artist"],
                    "track": r["track"],
                    "duration_s": float(r["duration_s"]) if r.get("duration_s") else None,
                }
                for r in csv.DictReader(handle)
            ]
        wanted = [t for t in wanted if not (audio_dir / f"{t['track_id']}.m4a").exists()]
    else:
        wanted = tracks_without_a_preview(audio_dir)
    todo = [t for t in wanted if t["track_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"tracks with no preview: {len(todo) + len(done)} counting {len(done)} already logged; to do {len(todo)}",
        flush=True,
    )

    fields = [
        "track_id",
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
    pacer = Pacer(args.gap)
    counters = {"done": 0, "found": 0}
    write_lock = threading.Lock()

    with (
        open(log_path, "a", newline="") as log_file,
        httpx.Client(
            headers={"User-Agent": "Mozilla/5.0"},
            limits=httpx.Limits(max_connections=args.workers * 2),
        ) as client,
    ):
        writer = csv.DictWriter(log_file, fieldnames=fields, extrasaction="ignore")
        if new_log:
            writer.writeheader()

        def one(row: dict, refused_out: list | None):
            result = lookup(client, pacer, row["artist"], row["track"], row["duration_s"])
            if result["status"] == "rate limited" and refused_out is not None:
                refused_out.append(
                    dict(row)
                )  # answer it in the second pass, do not sleep on it here
                return
            row.update(result)
            if result["status"] == "found":
                dest = audio_dir / f"{row['track_id']}.m4a"
                try:
                    dest.write_bytes(client.get(result["preview_url"], timeout=30).content)
                    row["path"] = str(dest)
                except Exception as error:  # noqa: BLE001
                    row["status"] = f"download error {type(error).__name__}"
            with write_lock:
                writer.writerow(row)
                log_file.flush()
                counters["done"] += 1
                counters["found"] += result["status"] == "found" and "path" in row
                if counters["done"] % 50 == 0:
                    print(
                        f"{counters['done']}/{len(todo)}  found {counters['found']}  gap {pacer.gap_s:.2f}s  "
                        f"last: {row['status']}  {row['artist']} - {row['track'][:40]}",
                        flush=True,
                    )

        refused: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda row: one(row, refused), todo))

        if refused:
            print(
                f"\nsecond pass over {len(refused)} refused tracks at gap {pacer.gap_s:.2f}s",
                flush=True,
            )
            counters["done"] = 0
            with ThreadPoolExecutor(max_workers=max(2, args.workers // 2)) as pool:
                list(pool.map(lambda row: one(row, None), refused))

    print(
        f"\ndone. {counters['found']} previews in {audio_dir}. "
        f"gap ended at {pacer.gap_s:.2f}s after {pacer.refusals} refusals."
    )


if __name__ == "__main__":
    main()
