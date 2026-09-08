"""Download full DJ mixes (2 per DJ, 45–150 min) from YouTube for analysis.

    python -m scripts.data.fetch_dj_mixes            # all DJs below
Writes data/raw/dj_mixes/<genre>/<dj>/<video id>.m4a and data/raw/dj_mixes/manifest.csv.
"""

import csv
import logging
from pathlib import Path
import subprocess
import sys

DJS = {
    "afro house": ["Black Coffee", "Shimza"],
    "drum and base": ["Noisia", "Sub Focus"],
    "melodic house": ["Tale Of Us", "Ben Böhmer", "Fred again.."],
    "tech house": ["Carl Cox", "Solomun", "Roman Flügel"],
    "techno": ["Adam Beyer", "Amelie Lens"],
    "trance": ["Armin van Buuren", "Paul van Dyk"],
}
OUT = Path("data/raw/dj_mixes")
PER_DJ = 2
MIN_S, MAX_S = 45 * 60, 150 * 60
log = logging.getLogger("dj_mixes")


def search(dj: str, n: int = 12) -> list[dict]:
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print",
        "%(id)s\t%(duration)s\t%(title)s",
        f"ytsearch{n}:{dj} full DJ set",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=180).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or not parts[1].replace(".", "").isdigit():
            continue
        vid, dur, title = parts[0], float(parts[1]), parts[2]
        if MIN_S <= dur <= MAX_S and "b2b" not in title.lower():
            rows.append({"id": vid, "duration_s": int(dur), "title": title})
    return rows


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    man_path = OUT / "manifest.csv"
    rows = list(csv.DictReader(man_path.open())) if man_path.exists() else []
    have = {r["id"] for r in rows}
    for genre, djs in DJS.items():
        for dj in djs:
            slug = (
                dj.lower().replace(" ", "_").replace(".", "").replace("ö", "o").replace("ü", "u")
            )
            d = OUT / genre.replace(" ", "_") / slug
            d.mkdir(parents=True, exist_ok=True)
            got = sum(1 for r in rows if r["dj"] == dj)
            for cand in search(dj):
                if got >= PER_DJ:
                    break
                if cand["id"] in have:
                    continue
                target = d / f"{cand['id']}.m4a"
                log.info(
                    "%s | %s | %d min | %s", genre, dj, cand["duration_s"] // 60, cand["title"]
                )
                r = subprocess.run(
                    [
                        "yt-dlp",
                        "-f",
                        "bestaudio[ext=m4a]/bestaudio",
                        "-x",
                        "--audio-format",
                        "m4a",
                        "-o",
                        str(d / "%(id)s.%(ext)s"),
                        "--no-playlist",
                        "--quiet",
                        "--no-warnings",
                        f"https://www.youtube.com/watch?v={cand['id']}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
                if r.returncode != 0 or not target.exists():
                    log.warning("  failed: %s", r.stderr.strip()[-200:])
                    continue
                rows.append(
                    {
                        "dj": dj,
                        "genre": genre,
                        "id": cand["id"],
                        "title": cand["title"],
                        "duration_s": cand["duration_s"],
                        "url": f"https://www.youtube.com/watch?v={cand['id']}",
                        "path": str(target),
                    }
                )
                have.add(cand["id"])
                got += 1
                with man_path.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
            if got < PER_DJ:
                log.warning("%s: only %d mix(es) found", dj, got)
    log.info("DONE: %d mixes in %s", len(rows), man_path)


if __name__ == "__main__":
    sys.exit(main())
