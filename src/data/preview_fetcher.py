"""
Audio preview fetcher — training pipeline only.

Fetches 30-second previews for commercial tracks from MixesDB using:
  1. iTunes Search API   — primary source, free, no auth, ~65-70% coverage
  2. SoundCloud (yt-dlp) — fallback for underground/Beatport-exclusive tracks

NOTE: Jamendo is intentionally excluded from this pipeline.
Jamendo contains only Creative Commons music. Searching it for commercial
MixesDB tracks returns completely unrelated audio — wrong training data.
Jamendo is used in Project 2 (API) where users select tracks directly by ID.

Downloads each preview to data/raw/previews/{track_id}.{ext}
  .m4a — iTunes (AAC)
  .mp3 — SoundCloud (via yt-dlp, trimmed to 30s)
Writes a manifest to data/raw/preview_manifest.csv with columns:
  track_id, artist, track_name, source, local_path

Idempotent: already-downloaded tracks are skipped on re-runs.
Requires: yt-dlp and ffmpeg on PATH for SoundCloud fallback.
"""

import asyncio
import hashlib
import logging
from pathlib import Path
import re

import httpx
import pandas as pd

log = logging.getLogger(__name__)

PREVIEWS_DIR = Path("data/raw/previews")
NOT_FOUND_PATH = Path("data/raw/not_found.csv")
MANIFEST_PATH = Path("data/raw/preview_manifest.csv")  # rebuilt on demand, not written during run
_ITUNES_GAP = 1.0  # seconds between iTunes requests — sequential, no burst
_RETRY_DELAYS = [5, 15, 30]  # backoff seconds on iTunes 429


# ── Track name cleaning ────────────────────────────────────────────────────────

_VERSION_KEYWORDS = {
    "remix",
    "mix",
    "edit",
    "version",
    "dub",
    "instrumental",
    "reprise",
    "remaster",
}


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
        return any(w in _VERSION_KEYWORDS for w in bracket_content.lower().split())

    def replace_bracket(match):
        content = match.group(1)
        return f"[{content}]" if should_keep(content) else ""

    return re.sub(r"\[([^\]]*)\]", replace_bracket, track).strip()


# ── Track ID ───────────────────────────────────────────────────────────────────


def track_id(artist: str, track: str) -> str:
    """Deterministic 12-char ID from artist + track name."""
    return hashlib.md5(f"{artist}|{track}".lower().encode()).hexdigest()[:12]


# ── Source fetchers ────────────────────────────────────────────────────────────


async def _itunes_url(client: httpx.AsyncClient, artist: str, track: str) -> str | None:
    """
    iTunes Search API — free, no auth, returns a 30s AAC preview URL.
    Called sequentially from the main loop (one at a time). Retries on 429.
    """
    query = f"{artist} {clean_track_name(track)}"
    for delay in [0] + _RETRY_DELAYS:
        if delay:
            await asyncio.sleep(delay)
        try:
            resp = await client.get(
                "https://itunes.apple.com/search",
                params={"term": query, "media": "music", "limit": 1},
                timeout=10,
            )
            if resp.status_code == 429:
                log.debug("iTunes 429, backing off %ds: %s", delay or _RETRY_DELAYS[0], query)
                continue
            if resp.status_code != 200:
                return None
            results = resp.json().get("results", [])
            return results[0].get("previewUrl") if results else None
        except Exception as e:
            log.debug("iTunes error for %s - %s: %s", artist, track, e)
            return None
    return None


def _sc_match(returned_title: str, artist: str, track: str) -> bool:
    """
    Verify the SoundCloud result is the right track.
    Rejects DJ mixes/podcasts. Requires track name words in title;
    artist match is a bonus (some labels omit artist from the title field).
    """
    title_lower = returned_title.lower()
    clean = clean_track_name(track).lower()

    # Reject obvious non-tracks (podcast/radio/episode — NOT "mix", which appears in "Original Mix")
    for bad in ("podcast", "radio show", "episode", "compilation"):
        if bad in title_lower and bad not in clean:
            return False

    # At least one significant word (>3 chars) from track name must appear in title
    track_words = [w for w in clean.split() if len(w) > 3]
    track_match = not track_words or any(w in title_lower for w in track_words)

    # Artist words in title are a bonus — if present, both must confirm; if absent, track match alone passes
    artist_words = [w for w in artist.lower().split() if len(w) > 2]
    artist_in_title = any(w in title_lower for w in artist_words)

    return track_match and (artist_in_title or track_match)


async def _soundcloud_download(tid: str, artist: str, track: str) -> Path | None:
    """
    Search SoundCloud via yt-dlp and download the first 30s as MP3.
    Verifies the result title matches before downloading.
    Requires yt-dlp and ffmpeg on PATH.
    """
    dest = PREVIEWS_DIR / f"{tid}.mp3"
    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    clean = clean_track_name(track)
    for query in [f"{clean} {artist}", f"{artist} {clean}"]:
        try:
            # Step 1: get title without downloading
            probe = await asyncio.create_subprocess_exec(
                "yt-dlp",
                f"scsearch1:{query}",
                "--print",
                "%(title)s",
                "--no-playlist",
                "--quiet",
                "--no-warnings",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await probe.communicate()
            returned_title = stdout.decode().strip()

            if not returned_title or not _sc_match(returned_title, artist, track):
                log.debug("SC mismatch [%s]: got '%s'", query, returned_title)
                continue

            # Step 2: download
            proc = await asyncio.create_subprocess_exec(
                "yt-dlp",
                f"scsearch1:{query}",
                "--download-sections",
                "*0-30",
                "--force-keyframes-at-cuts",
                "-x",
                "--audio-format",
                "mp3",
                "--audio-quality",
                "5",
                "-o",
                str(dest),
                "--no-playlist",
                "--quiet",
                "--no-warnings",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            if dest.exists() and dest.stat().st_size > 10_000:
                return dest
        except Exception as e:
            log.debug("SoundCloud error for %s - %s: %s", artist, track, e)
            break
    return None


# ── Download (iTunes) ──────────────────────────────────────────────────────────


def _audio_ext(content: bytes) -> str:
    return ".m4a" if len(content) >= 8 and content[4:8] == b"ftyp" else ".mp3"


def _find_cached(tid: str) -> Path | None:
    for ext in (".m4a", ".mp3"):
        p = PREVIEWS_DIR / f"{tid}{ext}"
        if p.exists():
            return p
    return None


async def _download(url: str, tid: str, client: httpx.AsyncClient) -> Path | None:
    try:
        resp = await client.get(url, follow_redirects=True, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 10_000:
            dest = PREVIEWS_DIR / f"{tid}{_audio_ext(resp.content)}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            return dest
    except Exception as e:
        log.debug("Download failed for %s: %s", url, e)
    return None


# ── State helpers ──────────────────────────────────────────────────────────────


def _downloaded_ids() -> set[str]:
    """Return track_ids already on disk in PREVIEWS_DIR."""
    return {f.stem for f in PREVIEWS_DIR.glob("*") if f.suffix in (".m4a", ".mp3")}


def _not_found_ids() -> set[str]:
    """Return track_ids previously tried and not found on any source."""
    if not NOT_FOUND_PATH.exists():
        return set()
    return set(pd.read_csv(NOT_FOUND_PATH)["track_id"].dropna())


def _append_not_found(tid: str, artist: str, track_name: str) -> None:
    write_header = not NOT_FOUND_PATH.exists()
    row = pd.DataFrame([{"track_id": tid, "artist": artist, "track_name": track_name}])
    row.to_csv(NOT_FOUND_PATH, mode="a", header=write_header, index=False)


def build_manifest(tracklist_path: str = "data/processed/tracklist_clean.csv") -> pd.DataFrame:
    """
    Reconstruct preview_manifest.csv from disk state + tracklist.
    Call this after a fetch run completes (or any time).
    """
    df = pd.read_csv(tracklist_path)[["artist_name", "track_name"]].dropna().drop_duplicates()
    rows = []
    for _, row in df.iterrows():
        tid = track_id(row["artist_name"], row["track_name"])
        path = _find_cached(tid)
        if path:
            src = "itunes" if path.suffix == ".m4a" else "soundcloud"
            rows.append(
                {
                    "track_id": tid,
                    "artist": row["artist_name"],
                    "track_name": row["track_name"],
                    "source": src,
                    "local_path": str(path),
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST_PATH, index=False)
    log.info("Manifest rebuilt: %d tracks → %s", len(manifest), MANIFEST_PATH)
    return manifest


# ── Main entry point ───────────────────────────────────────────────────────────

_SC_CONCURRENCY = 5  # max concurrent SoundCloud yt-dlp subprocesses


async def fetch_previews(tracklist_path: str = "data/processed/tracklist_clean.csv") -> None:
    """
    Read tracklist CSV, fetch a 30s preview for every unique track.
    Resume logic:
      - Already on disk (previews/) → skip
      - Already in not_found.csv   → skip
    Failures written to not_found.csv immediately (crash-safe).
    Successes are the audio files themselves — no manifest written during run.
    Call build_manifest() after the run to generate preview_manifest.csv.
    """
    df = pd.read_csv(tracklist_path)
    unique = df[["artist_name", "track_name"]].dropna().drop_duplicates()
    total = len(unique)
    log.info("Found %d unique tracks in %s", total, tracklist_path)

    PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    skip_ids = _downloaded_ids() | _not_found_ids()
    log.info("Skipping %d already-processed tracks", len(skip_ids))

    sc_sem = asyncio.Semaphore(_SC_CONCURRENCY)
    sc_tasks: list[asyncio.Task] = []

    async def _process_sc(i: int, tid: str, artist: str, track_name: str) -> None:
        async with sc_sem:
            saved = await _soundcloud_download(tid, artist, track_name)
        if saved:
            log.info("[%d/%d] [soundcld ] %s - %s", i + 1, total, artist, track_name)
        else:
            log.warning("[%d/%d] [not found] %s - %s", i + 1, total, artist, track_name)
            _append_not_found(tid, artist, track_name)

    async with httpx.AsyncClient() as http:
        for i, (_, row) in enumerate(unique.iterrows()):
            artist, track_name = row["artist_name"], row["track_name"]
            tid = track_id(artist, track_name)

            if tid in skip_ids:
                continue

            # iTunes — sequential with fixed gap
            url = await _itunes_url(http, artist, track_name)
            await asyncio.sleep(_ITUNES_GAP)

            if url:
                saved = await _download(url, tid, http)
                if saved:
                    log.info("[%d/%d] [itunes   ] %s - %s", i + 1, total, artist, track_name)
                    continue

            # SoundCloud — background task bounded by sc_sem
            sc_tasks.append(asyncio.create_task(_process_sc(i, tid, artist, track_name)))

        if sc_tasks:
            await asyncio.gather(*sc_tasks)

    downloaded = len(_downloaded_ids())
    not_found = len(_not_found_ids())
    log.info("Done — downloaded: %d  not_found: %d  total: %d", downloaded, not_found, total)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _fmt = "%(asctime)s %(levelname)s %(message)s"
    _log_path = Path("logs/preview_fetcher.log")
    _log_path.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=_fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(_log_path, encoding="utf-8"),
        ],
    )
    log.info("Logging to %s", _log_path)
    asyncio.run(fetch_previews())
