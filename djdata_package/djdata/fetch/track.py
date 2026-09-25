"""Fetch one track.

Raveform seams carry a YouTube id, so the download is direct. Tracklist seams carry only artist and
title, so the candidates come from a search first.

Both go through fetch/yt.py, which is the only place that holds the cookie lock, passes the player
client, forces IPv4, times out a dead socket and raises Blocked. The legacy fetcher has its own
yt-dlp call with none of that; from a datacenter address it fails on the first track, so it is used
here only for finding and ranking candidates and for the fingerprint check.
"""

import logging
from pathlib import Path
import shutil
import subprocess
import tempfile

from . import yt

log = logging.getLogger("djdata.fetch.track")


def _duration_s(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def download_by_url(cfg, url: str, dest_stem: Path) -> tuple[Path, float]:
    with tempfile.TemporaryDirectory(dir=dest_stem.parent) as td:
        got = yt.download(cfg, url, cfg.download["track_format"], Path(td))
        dest = dest_stem.with_suffix(got.suffix)
        shutil.move(str(got), dest)
    dur = _duration_s(dest)
    lo, hi = cfg.download["min_track_s"], cfg.download["max_track_s"]
    if not lo <= dur <= hi:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"duration {dur:.0f}s outside [{lo}, {hi}]")
    return dest, dur


def _candidates(artist: str, name: str, min_s: int | None = None, max_s: int | None = None) -> list[dict]:
    """YouTube first, then SoundCloud if YouTube offers nothing. Ranked by artist, title and remix
    wording, so the right recording is picked before anything is downloaded.

    Duration bounds default to the legacy fetcher's 240 to 900 s. The run passes its own
    `download.min_track_s` / `max_track_s`: on 2026-09-24 the 240 s floor threw away every club track
    under four minutes (Casey Club - Patna Palace at 169 s, Cesco & NINA - Rigid Lord at 226 s) and the
    run reported "no candidate" for records that were the first search result."""
    from ..legacy.track_fetcher import MAX_SECONDS, MIN_SECONDS, _rank, _search

    bounds = {"min_seconds": min_s or MIN_SECONDS, "max_seconds": max_s or MAX_SECONDS}
    query = f"{artist} {name}"
    ranked = _rank(query, _search(query), artist=artist, title=name, **bounds)
    if ranked:
        return ranked
    log.info("  no YouTube candidate for %s - %s, trying SoundCloud", artist, name[:40])
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            res = ydl.extract_info(f"scsearch5:{query}", download=False)
        except Exception as error:  # noqa: BLE001
            log.warning("  soundcloud search failed for %r: %s", query, error)
            return []
    entries = [e for e in (res or {}).get("entries", []) if e]
    return _rank(query, entries, artist=artist, title=name, **bounds)


def _candidates_strict(cfg, artist: str, name: str) -> list[dict]:
    """Both sources every time, ranked by how close the title is (Anas, 2026-09-24).

    YouTube and SoundCloud are both searched, because the version a DJ plays is often only on SoundCloud
    while YouTube still returns the original ("Suavemente (Oppidan Dub)"). A candidate is kept when its
    length is inside the config's bounds, its title carries no rejection word the wanted title lacks, and
    one of the people on the record (a listed artist or the remixer) is in its title or channel. The
    closest title wins; a verified channel adds a little."""
    import yt_dlp
    from ..legacy.track_fetcher import _BAD_TITLE, _name_score, _search
    from . import names

    entries, seen = [], set()
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    for query in names.search_queries(artist, name):
        found = list(_search(query))
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                res = ydl.extract_info(f"scsearch5:{query}", download=False)
                found += [e for e in (res or {}).get("entries", []) if e]
            except Exception as error:  # noqa: BLE001
                log.warning("  soundcloud search failed for %r: %s", query, error)
        for e in found:
            key = e.get("url") or e.get("id")
            if key not in seen:
                seen.add(key); entries.append(e)
    lo, hi = cfg.download.get("min_track_s", 60), cfg.download.get("max_track_s", 1200)
    in_bounds = [e for e in entries if e.get("duration") is not None and lo <= e["duration"] <= hi]

    def scored(e, tier):
        cand_title = e.get("title") or ""
        close = names.closeness(name, cand_title) + (0.2 if e.get("channel_is_verified") else 0.0)
        return {**e, "closeness": round(close, 3), "name_score": _name_score(artist, name, e), "tier": tier}

    strict = [scored(e, "strict") for e in in_bounds
              if not names.bad_title(name, e.get("title") or "", _BAD_TITLE)
              and names.artist_present(artist, name, e.get("title") or "", e.get("channel") or e.get("uploader") or "")]
    if strict:
        return sorted(strict, key=lambda c: -c["closeness"])
    # Nothing passed the strict rules. Anas, 2026-09-24: a 50-50 candidate beats an empty slot, and a wrong
    # pick is caught when the track is looked for in its mix. So the relaxed pass keeps every in-length
    # result and lets the closest title win.
    relaxed = [scored(e, "relaxed") for e in in_bounds
               if not names.bad_title(name, e.get("title") or "", _BAD_TITLE)
               and names.title_words_overlap(name, e.get("title") or "") >= 0.6]
    if relaxed:
        log.info("  %s - %s: no strict candidate, %d relaxed", artist[:25], name[:35], len(relaxed))
    return sorted(relaxed, key=lambda c: -c["closeness"])


def download_by_name(cfg, title: str, track_id: str, dest_dir: Path) -> tuple[Path, float, str]:
    """Tracklist path: 'Artist - Title' → search, rank on names, download through the gate, verify.

    Returns the file, its length and the url it came from. With `download.strict_names` on (Anas,
    2026-09-24) ID tracks are not searched, the closest title among the top three results is taken first
    whatever its score (the score is logged), and the iTunes preview is not consulted at all. A wrong pick is
    caught later, when the track is looked for in its mix."""
    from ..legacy.track_fetcher import CONFIDENT_SCORE, VERIFY_MIN_HASHES, preview_paths, verify_match
    from . import names

    strict = bool(cfg.download.get("strict_names"))
    artist, _, name = title.partition(" - ")
    if strict and names.is_unidentified(name):
        raise RuntimeError(f"{title}: unidentified track, nothing to search")
    if strict:
        ranked = _candidates_strict(cfg, artist, name)[:3]   # closest titles first; the mix judges later
    else:
        ranked = _candidates(artist, name, cfg.download.get("min_track_s"), cfg.download.get("max_track_s"))
    if not ranked:
        raise RuntimeError(f"no candidate for {title}")
    preview = preview_paths().get(track_id)
    preview = preview if preview and Path(preview).exists() else None

    last_error = "no candidate survived the checks"
    for cand in ranked[:3]:
        url = cand.get("url") or f"https://www.youtube.com/watch?v={cand['id']}"
        try:
            path, dur = download_by_url(cfg, url, dest_dir / track_id)
        except yt.Blocked:
            raise                                   # the runner closes the gate and retries this track
        except Exception as error:                  # noqa: BLE001
            last_error = str(error)
            log.info("  %s: candidate rejected (%s)", name[:40], last_error[:80])
            continue

        score = cand.get("name_score", 0.0)
        if score >= CONFIDENT_SCORE:
            log.info("  %s: name score %.2f, accepted  %s", name[:40], score, url)
            return path, dur, url
        if preview and not strict:                # strict path: the mix decides later, not the preview (Anas)
            votes, _ = verify_match(preview, path)
            if votes < VERIFY_MIN_HASHES:
                log.info("  %s: fingerprint %d votes, wrong recording", name[:40], votes)
                path.unlink(missing_ok=True)
                last_error = f"fingerprint {votes} votes"
                continue
            log.info("  %s: name score %.2f, fingerprint %d votes, accepted  %s", name[:40], score, votes, url)
            return path, dur, url
        if strict:
            log.info("  %s: closest title (%s pass), name score %.2f, accepted  %s", name[:40], cand.get("tier", "strict"), score, url)
            return path, dur, url
        log.info("  %s: name score %.2f, no preview to check against, accepted  %s", name[:40], score, url)
        return path, dur, url

    raise RuntimeError(f"{title}: {last_error}")


def fetch(cfg, track: dict) -> tuple[Path, float, str | None]:
    """The file, its length, and the url it was fetched from (None when it was already on disk)."""
    dest_dir = cfg.dirs["tracks"]
    existing = list(dest_dir.glob(f"{track['track_id']}.*"))
    if existing:
        return existing[0], _duration_s(existing[0]), track.get("url")
    if track.get("url"):
        path, dur = download_by_url(cfg, track["url"], dest_dir / track["track_id"])
        return path, dur, track["url"]
    return download_by_name(cfg, track["title"], track["track_id"], dest_dir)
