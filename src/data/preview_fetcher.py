"""
Audio preview fetcher.

For each unique track in a tracklist CSV, fetches a 30-second preview MP3
from three sources in priority order:
  1. Spotify Search API  — best coverage for electronic music
  2. iTunes Search API   — free, no auth, different catalog coverage
  3. Jamendo API         — Creative Commons / indie music

Downloads each preview to data/raw/previews/{track_id}.mp3
Writes a manifest to data/raw/preview_manifest.csv with columns:
  track_id, artist, track_name, source, local_path

Idempotent: already-downloaded tracks are skipped on re-runs.
"""
import asyncio
import hashlib
import logging
import os
from pathlib import Path

import httpx
import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv()
log = logging.getLogger(__name__)

PREVIEWS_DIR = Path("data/raw/previews")
MANIFEST_PATH = Path("data/raw/preview_manifest.csv")


# ── Track ID ───────────────────────────────────────────────────────────────────

def track_id(artist: str, track: str) -> str:
    """
    Deterministic 12-char ID from artist + track name.
    Stable across runs — same input always produces the same ID.
    Used as the filename for the downloaded MP3.
    """
    return hashlib.md5(f"{artist}|{track}".lower().encode()).hexdigest()[:12]


# ── Source fetchers (each returns a preview URL or None) ──────────────────────

def _spotify_client() -> spotipy.Spotify:
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    ))


async def _spotify_url(sp: spotipy.Spotify, artist: str, track: str) -> str | None:
    """
    Search Spotify by artist + track, return the 30s preview URL.
    NOTE: Spotify deprecated preview_url for most tracks in late 2023.
    Returns None both when the track isn't found AND when preview_url is null.
    """
    try:
        results = sp.search(q=f"artist:{artist} track:{track}", type="track", limit=1)
        items = results["tracks"]["items"]
        if not items:
            return None
        url = items[0]["preview_url"]
        if url is None:
            log.debug("Spotify found track but preview_url is null (deprecated): %s - %s", artist, track)
        return url
    except Exception as e:
        log.debug("Spotify error for %s - %s: %s", artist, track, e)
        return None


async def _itunes_url(client: httpx.AsyncClient, artist: str, track: str) -> str | None:
    """
    iTunes Search API — completely free, no auth required.
    Returns a 30s AAC preview URL or None.
    """
    try:
        resp = await client.get(
            "https://itunes.apple.com/search",
            params={"term": f"{artist} {track}", "media": "music", "limit": 1},
            timeout=10,
        )
        results = resp.json().get("results", [])
        return results[0].get("previewUrl") if results else None
    except Exception as e:
        log.debug("iTunes miss for %s - %s: %s", artist, track, e)
        return None


async def _jamendo_url(client: httpx.AsyncClient, artist: str, track: str) -> str | None:
    """
    Jamendo API — Creative Commons music catalog.
    Returns a direct MP3 stream URL or None.
    Requires JAMENDO_CLIENT_ID in .env.
    """
    client_id = os.getenv("JAMENDO_CLIENT_ID")
    if not client_id:
        return None
    try:
        resp = await client.get(
            "https://api.jamendo.com/v3.0/tracks/",
            params={
                "client_id": client_id,
                "search": f"{artist} {track}",
                "limit": 1,
                "audioformat": "mp32",  # 320kbps MP3 stream
            },
            timeout=10,
        )
        results = resp.json().get("results", [])
        return results[0].get("audio") if results else None
    except Exception as e:
        log.debug("Jamendo miss for %s - %s: %s", artist, track, e)
        return None


# ── Download ───────────────────────────────────────────────────────────────────

async def _download(url: str, dest: Path, client: httpx.AsyncClient) -> bool:
    """Download audio from url and save to dest. Returns True on success."""
    try:
        resp = await client.get(url, follow_redirects=True, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 10_000:  # guard against empty responses
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            return True
    except Exception as e:
        log.debug("Download failed for %s: %s", url, e)
    return False


# ── Main entry point ───────────────────────────────────────────────────────────

async def fetch_previews(tracklist_path: str = "data/interim/romanFlugel.csv") -> pd.DataFrame:
    """
    Read tracklist CSV, fetch a 30s preview for every unique track.

    Tries sources in order: Spotify → iTunes → Jamendo.
    Skips tracks already present in PREVIEWS_DIR (safe to re-run).

    Returns the manifest DataFrame and writes it to MANIFEST_PATH.
    """
    df = pd.read_csv(tracklist_path)
    unique = df[["artist_name", "track_name"]].dropna().drop_duplicates()
    log.info("Found %d unique tracks in %s", len(unique), tracklist_path)

    sp = _spotify_client()
    rows = []

    async with httpx.AsyncClient() as http:
        for _, row in unique.iterrows():
            artist, track_name = row["artist_name"], row["track_name"]
            tid = track_id(artist, track_name)
            dest = PREVIEWS_DIR / f"{tid}.mp3"

            # Skip if already downloaded
            if dest.exists():
                log.info("[cached]  %s - %s", artist, track_name)
                rows.append(_row(tid, artist, track_name, "cached", dest))
                continue

            # Try each source in order
            url, source = None, "not_found"
            for src_name, fetcher in [
                ("spotify", lambda: _spotify_url(sp, artist, track_name)),
                ("itunes",  lambda: _itunes_url(http, artist, track_name)),
                ("jamendo", lambda: _jamendo_url(http, artist, track_name)),
            ]:
                url = await fetcher()
                if url:
                    source = src_name
                    break

            if url and await _download(url, dest, http):
                log.info("[%s]  %s - %s", source, artist, track_name)
                rows.append(_row(tid, artist, track_name, source, dest))
            else:
                log.warning("[not found]  %s - %s", artist, track_name)
                rows.append(_row(tid, artist, track_name, "not_found", None))

    manifest = pd.DataFrame(rows)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    found = (manifest["source"] != "not_found").sum()
    log.info("Done: %d/%d previews fetched → %s", found, len(manifest), MANIFEST_PATH)
    _log_source_breakdown(manifest)
    return manifest


def _row(tid, artist, track_name, source, dest) -> dict:
    return {
        "track_id": tid,
        "artist": artist,
        "track_name": track_name,
        "source": source,
        "local_path": str(dest) if dest else None,
    }


def _log_source_breakdown(manifest: pd.DataFrame) -> None:
    counts = manifest["source"].value_counts()
    for source, n in counts.items():
        log.info("  %-12s %d", source, n)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(fetch_previews("data/interim/romanFlugel.csv"))
