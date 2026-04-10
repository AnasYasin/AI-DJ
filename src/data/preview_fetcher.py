"""
Audio preview fetcher — training pipeline only.

Fetches 30-second previews for commercial tracks from MixesDB using:
  1. Spotify Search API  — deprecated for most tracks since late 2023
  2. iTunes Search API   — primary source, free, no auth, ~65-70% coverage

NOTE: Jamendo is intentionally excluded from this pipeline.
Jamendo contains only Creative Commons music. Searching it for commercial
MixesDB tracks returns completely unrelated audio — wrong training data.
Jamendo is used in Project 2 (API) where users select tracks directly by ID.

Downloads each preview to data/raw/previews/{track_id}.mp3
Writes a manifest to data/raw/preview_manifest.csv with columns:
  track_id, artist, track_name, source, local_path

Idempotent: already-downloaded tracks are skipped on re-runs.
"""
import asyncio
import hashlib
import logging
import os
import re
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


# ── Track name cleaning ────────────────────────────────────────────────────────

# Words inside brackets that indicate a genuine remix/version — keep these
_VERSION_KEYWORDS = {"remix", "mix", "edit", "version", "dub", "instrumental", "reprise", "remaster"}

def clean_track_name(track: str) -> str:
    """
    Strip label annotations from MixesDB track names before searching.

    MixesDB format puts the record label in square brackets:
      "Function [Innate Editions]"      → "Function"
      "We Feel For You [MFF]"           → "We Feel For You"

    But genuine remix/version info also uses brackets — keep those:
      "Track Name [Producer Remix]"     → "Track Name [Producer Remix]"
      "Track Name [Extended Mix]"       → "Track Name [Extended Mix]"

    Rule: strip bracket content UNLESS it contains a version keyword.
    """
    def should_keep(bracket_content: str) -> bool:
        words = bracket_content.lower().split()
        return any(w in _VERSION_KEYWORDS for w in words)

    def replace_bracket(match):
        content = match.group(1)
        return f"[{content}]" if should_keep(content) else ""

    cleaned = re.sub(r"\[([^\]]*)\]", replace_bracket, track)
    return cleaned.strip()


# ── Track ID ───────────────────────────────────────────────────────────────────

def track_id(artist: str, track: str) -> str:
    """
    Deterministic 12-char ID from artist + track name.
    Stable across runs — same input always produces the same ID.
    Used as the filename for the downloaded preview file.
    """
    return hashlib.md5(f"{artist}|{track}".lower().encode()).hexdigest()[:12]


# ── Source fetchers ────────────────────────────────────────────────────────────

def _spotify_client() -> spotipy.Spotify:
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    ))


async def _spotify_url(sp: spotipy.Spotify, artist: str, track: str) -> str | None:
    """
    Search Spotify and return the 30s preview URL.
    NOTE: Spotify removed preview_url from most tracks in late 2023 — None is common.
    """
    try:
        results = sp.search(q=f"artist:{artist} track:{clean_track_name(track)}", type="track", limit=1)
        items = results["tracks"]["items"]
        if not items:
            return None
        url = items[0]["preview_url"]
        if url is None:
            log.debug("Spotify preview_url is null (deprecated) for: %s - %s", artist, track)
        return url
    except Exception as e:
        log.debug("Spotify error for %s - %s: %s", artist, track, e)
        return None


async def _itunes_url(client: httpx.AsyncClient, artist: str, track: str) -> str | None:
    """
    iTunes Search API — free, no auth, returns a 30s AAC preview URL.
    Primary source for this pipeline (~65-70% coverage of electronic music).
    """
    try:
        resp = await client.get(
            "https://itunes.apple.com/search",
            params={"term": f"{artist} {clean_track_name(track)}", "media": "music", "limit": 1},
            timeout=10,
        )
        results = resp.json().get("results", [])
        return results[0].get("previewUrl") if results else None
    except Exception as e:
        log.debug("iTunes miss for %s - %s: %s", artist, track, e)
        return None


# ── Download ───────────────────────────────────────────────────────────────────

async def _download(url: str, dest: Path, client: httpx.AsyncClient) -> bool:
    """Download audio from url and save to dest. Returns True on success."""
    try:
        resp = await client.get(url, follow_redirects=True, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 10_000:
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
    Tries Spotify first, falls back to iTunes.
    Skips tracks already present in PREVIEWS_DIR (safe to re-run).
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

            if dest.exists():
                log.info("[cached]  %s - %s", artist, track_name)
                rows.append(_row(tid, artist, track_name, "cached", dest))
                continue

            url, source = None, "not_found"
            for src_name, fetcher in [
                ("spotify", lambda: _spotify_url(sp, artist, track_name)),
                ("itunes",  lambda: _itunes_url(http, artist, track_name)),
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
    for source, n in manifest["source"].value_counts().items():
        log.info("  %-12s %d", source, n)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(fetch_previews("data/interim/romanFlugel.csv"))
