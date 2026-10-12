"""Listeners and play count per artist from Last.fm (key in .env). One call per distinct first artist. Resumable: appends to
data/interim/enrich/lastfm_artists.csv and skips artists already there.

    nohup conda run -n aidj python scripts/enrich/lastfm_artist_listeners.py > logs/enrich_lastfm.log 2>&1 &
"""

import csv
import os
import sys
import time

import requests

sys.path.insert(0, "scripts/enrich")
from records import key, records

OUT = "data/interim/enrich/lastfm_artists.csv"
env = dict(
    line.strip().split("=", 1) for line in open(".env") if "=" in line and not line.startswith("#")
)
KEY = env["LASTFM_API_KEY"].strip()
recs = records()
recs["ak"] = recs.artist1.map(key)
artists = recs.drop_duplicates("ak")[["artist1", "ak"]]
done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        done = {r["ak"] for r in csv.DictReader(f)}
todo = artists[~artists.ak.isin(done)]
print(f"{len(artists)} artists, {len(done)} done, {len(todo)} to do", flush=True)
new = not os.path.exists(OUT)
with open(OUT, "a", newline="") as f:
    w = csv.DictWriter(
        f, fieldnames=["ak", "artist", "hit", "lf_artist", "listeners", "playcount"]
    )
    if new:
        w.writeheader()
    for i, r in enumerate(todo.itertuples(), 1):
        row = dict(ak=r.ak, artist=r.artist1, hit=0)
        for attempt in range(3):
            try:
                resp = requests.get(
                    "https://ws.audioscrobbler.com/2.0/",
                    params=dict(
                        method="artist.getInfo",
                        artist=r.artist1,
                        api_key=KEY,
                        format="json",
                        autocorrect=1,
                    ),
                    timeout=20,
                )
                if resp.status_code == 429:
                    time.sleep(10)
                    continue
                a = resp.json().get("artist") if resp.ok else None
                if a:
                    row.update(
                        hit=1,
                        lf_artist=a["name"],
                        listeners=int(a["stats"]["listeners"]),
                        playcount=int(a["stats"]["playcount"]),
                    )
                break
            except Exception:
                time.sleep(5)
        w.writerow(row)
        f.flush()
        if i % 500 == 0:
            print(f"{i}/{len(todo)} {time.strftime('%H:%M:%S')}", flush=True)
        time.sleep(
            1.0
        )  # one call a second; Spotify locked us out for 23 h at 3 a second (2026-10-13)
print("done", flush=True)
