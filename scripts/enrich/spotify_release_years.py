"""Release year per record from Spotify (client credentials, .env). Resumable: appends to data/interim/enrich/spotify_records.csv
and skips keys already there. A hit is kept only when the returned title and artist resemble the query (difflib ratio), because
Spotify's search returns another record by the same artist when the asked one is missing. Polite: ~3 calls a second, 429 honoured.

    nohup conda run -n aidj python scripts/enrich/spotify_release_years.py > logs/enrich_spotify.log 2>&1 &
"""

import csv
import difflib
import os
import re
import sys
import time

import requests

sys.path.insert(0, "scripts/enrich")
from records import key, records

OUT = "data/interim/enrich/spotify_records.csv"
env = dict(
    line.strip().split("=", 1) for line in open(".env") if "=" in line and not line.startswith("#")
)


def token():
    return requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(env["SPOTIFY_CLIENT_ID"].strip(), env["SPOTIFY_CLIENT_SECRET"].strip()),
        timeout=20,
    ).json()["access_token"]


def sim(a, b):
    return difflib.SequenceMatcher(None, key(a), key(b)).ratio()


def strip_mix(t):
    return re.sub(
        r"\s*[\(\[-]\s*(Original|Extended|Club|Radio|Edit|Mix)[^\)\]]*[\)\]]?", "", t, flags=re.I
    ).strip()


recs = records()
done = set()
if os.path.exists(OUT):
    with open(OUT) as f:
        done = {r["k"] for r in csv.DictReader(f)}
todo = recs[~recs.k.isin(done)]
print(f"{len(recs)} records, {len(done)} done, {len(todo)} to do", flush=True)
tok, t_tok = token(), time.time()
H = {"Authorization": f"Bearer {tok}"}
new = not os.path.exists(OUT)
with open(OUT, "a", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "k",
            "artist",
            "title",
            "hit",
            "sp_artist",
            "sp_title",
            "sp_track_id",
            "release_date",
            "title_sim",
            "artist_sim",
        ],
    )
    if new:
        w.writeheader()
    for i, r in enumerate(todo.itertuples(), 1):
        if time.time() - t_tok > 3000:
            H = {"Authorization": f"Bearer {token()}"}
            t_tok = time.time()
        q = f"track:{strip_mix(r.title)} artist:{r.artist1}"
        for attempt in range(3):
            try:
                resp = requests.get(
                    "https://api.spotify.com/v1/search",
                    headers=H,
                    params={"q": q, "type": "track", "limit": 3},
                    timeout=20,
                )
            except Exception:
                time.sleep(5)
                continue
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5)) + 1
                print(f"429 at {i}, waiting {wait} s {time.strftime('%H:%M:%S')}", flush=True)
                time.sleep(min(wait, 1800))
                continue
            if resp.status_code == 401:
                H = {"Authorization": f"Bearer {token()}"}
                continue
            break
        row = dict(k=r.k, artist=r.artist, title=r.title, hit=0)
        items = resp.json().get("tracks", {}).get("items", []) if resp.ok else []
        best = None
        for it in items:
            ts, as_ = (
                sim(strip_mix(r.title), strip_mix(it["name"])),
                max(sim(r.artist1, a["name"]) for a in it["artists"]),
            )
            if ts >= 0.75 and as_ >= 0.6 and (best is None or ts + as_ > best[0]):
                best = (ts + as_, it, ts, as_)
        if best:
            _, it, ts, as_ = best
            row.update(
                hit=1,
                sp_artist=it["artists"][0]["name"],
                sp_title=it["name"],
                sp_track_id=it["id"],
                release_date=it.get("album", {}).get("release_date", ""),
                title_sim=round(ts, 2),
                artist_sim=round(as_, 2),
            )
        w.writerow(row)
        f.flush()
        if i % 500 == 0:
            print(f"{i}/{len(todo)} {time.strftime('%H:%M:%S')}", flush=True)
        time.sleep(1.0)  # 0.3 s drew a long 429 lockout after ~500 calls (2026-10-13)
print("done", flush=True)
