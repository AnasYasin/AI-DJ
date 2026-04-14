#!/usr/bin/env python3
"""
1001tracklists.com scraper: collect set URLs from DJ pages, then scrape tracklists.

Drop-in replacement for mixesdb_client.py — identical public API and output schema:
  data/interim/tracklist.csv
  columns: mix_id, mix_title, dj_name, genre, url, starting_time, track_name, artist_name

Key differences from MixesDB:
  - starting_time is real cue minutes from mix start (e.g. 6:13 → 6.217),
    scraped from the timestamp shown below the play icon on each track.
    First track defaults to 0.0. Pages with no timestamps at all fall back
    to position index so sort order is still correct in training.
  - Tracks marked as ID / Unknown / Mash are skipped (no artist or title)
  - DJ pages may span multiple pages — _collect_mix_urls follows pagination

CSS selectors to adjust if the site changes:
  _ITEM_SEL    — each track row container
  _ARTIST_SEL  — artist name within a row
  _TITLE_SEL   — track title within a row
  _SKIP_CLASSES — row classes that indicate an unidentified track (skip)
"""
import asyncio
import hashlib
import logging
import random
import re
from pathlib import Path

import nodriver as nd
import pandas as pd
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL    = "https://www.1001tracklists.com"
OUTPUT_PATH = Path("data/interim/tracklist.csv")

# ── CSS selectors (update here if the site restructures) ──────────────────────
_ITEM_SEL = "div.tlpItem"   # one container per track in a set
# Within each tlpItem:
#   meta[itemprop="name"]     → content = "Artist - Track Name"
#   meta[itemprop="byArtist"] → content = "Artist"
#   span.fontXL               → track number ("01") or "w/" for overlay tracks
#   div.cue                   → cue timestamp text ("7:20") or empty
_SKIP_CLASSES = {"tlpUnknown", "tlpMash", "tlpHide", "id_unknown"}


# ── Public scraping functions ──────────────────────────────────────────────────

async def extract_mix_urls_from_dj_page(page) -> list[str]:
    """
    Return all tracklist page URLs found on a DJ artist page.
    Follows "load more" / pagination if a next-page link exists.
    """
    soup  = BeautifulSoup(await page.evaluate("document.documentElement.outerHTML"), "html.parser")
    seen, urls = set(), []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip().split("?")[0]
        # Tracklist pages always contain /tracklist/ in the path
        if "/tracklist/" not in href:
            continue
        # Skip nav/utility links that point to the root tracklist index
        if href.rstrip("/") in ("/tracklist", f"{BASE_URL}/tracklist"):
            continue
        full = (BASE_URL + href) if href.startswith("/") else href
        if full not in seen:
            seen.add(full)
            urls.append(full)

    return urls


async def extract_tracklist(page) -> dict:
    """
    Parse a 1001tracklists set page.
    Returns dict with keys: title, tracklist (list of dicts), track_count, success.

    Each tracklist entry: {artist_name, track_name, starting_time}
    starting_time is float minutes from mix start (e.g. 6:13 → 6.217).
    First track defaults to 0.0 if no timestamp shown.
    Falls back to position index only when NO track on the page has a timestamp.
    """
    soup = BeautifulSoup(await page.evaluate("document.documentElement.outerHTML"), "html.parser")

    # ── Title ──────────────────────────────────────────────────────────────────
    title = "Unknown"
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        # "Set name @ Venue - 1001tracklists" → keep everything before last " - "
        if " - " in title:
            title = title.rsplit(" - ", 1)[0].strip()

    # ── Skip incomplete / unordered mixes ─────────────────────────────────────
    # 1001tracklists shows this notice when no recording exists and the track
    # order may be wrong. Corrupted order = corrupted consecutive pairs.
    page_text = soup.get_text(separator=" ")
    if "track order might not be correct" in page_text.lower():
        log.info("  Skipping — tracklist marked as incomplete/unordered")
        return {"title": title, "tracklist": [], "track_count": 0, "success": False}

    # ── Tracklist ──────────────────────────────────────────────────────────────
    tracklist = _parse_tracklist_html(soup)

    # Fallback: text-based parser if HTML selectors yielded nothing
    if not tracklist:
        log.debug("HTML selectors found nothing, trying text fallback")
        tracklist = _parse_tracklist_text(soup.get_text(separator="\n", strip=True))


    return {
        "title":       title,
        "tracklist":   tracklist,
        "track_count": len(tracklist),
        "success":     bool(tracklist),
    }


async def scrape_multiple_mixes(url_metas: list[dict], browser=None) -> list[dict]:
    """
    Scrape tracklists from a list of set URLs. Reuses browser if provided.
    url_metas: list of dicts with keys: url, dj_name, genre
    """
    own_browser = browser is None
    if own_browser:
        browser = await nd.start(headless=True, no_sandbox=True)

    results = []
    try:
        for i, meta in enumerate(url_metas, 1):
            url = meta["url"]
            log.info("[%d/%d] %s", i, len(url_metas), url)
            page = await browser.get(url, new_tab=True)
            await asyncio.sleep(random.uniform(15, 20))   # initial render
            await _scroll_to_bottom(page, label="mix page")  # load all tracks
            data = await extract_tracklist(page)
            await _close(page)
            data["url"]     = url
            data["dj_name"] = meta["dj_name"]
            data["genre"]   = meta["genre"]
            results.append(data)
            log.info("  → %d tracks", data["track_count"])
            if i % 5 == 0:
                _save(results[-5:])
                log.info("  checkpoint saved (%d mixes done)", i)
            if i < len(url_metas):
                await asyncio.sleep(random.uniform(12, 30))
    except Exception:
        log.exception("Error during mix scraping")
    finally:
        if own_browser:
            browser.stop()
            await asyncio.sleep(0.5)

    _log_summary(results)
    _save(results)
    return results


async def scrape_mixes_from_djs(dj_config: list[dict]) -> list[dict]:
    """
    Main entry point. Two-step process:
      1. Visit each DJ artist page → collect set URLs
      2. Visit each set URL → scrape tracklist
    Saves output to data/interim/tracklist.csv.

    dj_config: list of dicts with keys: url (DJ artist page), genre
      e.g. [{"url": "https://www.1001tracklists.com/dj/charlotte-de-witte/index.html",
              "genre": "techno"}]
    """
    browser = await nd.start(headless=True, no_sandbox=True)
    try:
        url_metas = await _collect_mix_urls(browser, dj_config)
        if not url_metas:
            log.warning("No set URLs found.")
            return []
        return await scrape_multiple_mixes(url_metas, browser=browser)
    finally:
        browser.stop()
        await asyncio.sleep(0.5)


# ── Private helpers ────────────────────────────────────────────────────────────

async def _scroll_to_bottom(page, label: str = "page") -> None:
    """Scroll down until no new content loads (handles infinite-scroll pages)."""
    prev_height = 0
    step = 0
    while True:
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(3)
        height = await page.evaluate("document.body.scrollHeight")
        if height == prev_height:
            log.info("  %s fully scrolled (%d steps, %dpx)", label, step, height)
            break
        prev_height = height
        step += 1
        log.info("  scrolled to %dpx", height)


async def _collect_mix_urls(browser, dj_config: list[dict]) -> list[dict]:
    """
    Step 1: visit each DJ artist page (and its paginated sub-pages) to collect
    all set URLs. Returns list of dicts: {url, dj_name, genre}.
    """
    all_metas, seen = [], set()

    for i, cfg in enumerate(dj_config, 1):
        dj_url = cfg["url"]
        genre  = cfg["genre"]
        name   = _dj_name(dj_url)
        log.info("[%d/%d] Scraping DJ page: %s (%s)", i, len(dj_config), name, genre)

        try:
            # Collect across all paginated pages for this DJ
            page_url = dj_url
            page_num = 1
            while page_url:
                page = await browser.get(page_url, new_tab=True)
                await asyncio.sleep(random.uniform(20, 25))   # wait for initial page render

                # Scroll to bottom until no new content loads (infinite scroll)
                await _scroll_to_bottom(page, label="DJ page")

                soup = BeautifulSoup(await page.get_content(), "html.parser")

                # Log page title to quickly diagnose wrong-slug redirects
                page_title = soup.title.string.strip() if soup.title else "NO TITLE"
                log.info("  page title: %s", page_title)
                await _close(page)

                new_on_page = 0
                for url in await _extract_set_urls_from_soup(soup):
                    if url not in seen:
                        seen.add(url)
                        all_metas.append({"url": url, "dj_name": name, "genre": genre})
                        new_on_page += 1

                log.info("  page %d: +%d sets", page_num, new_on_page)

                # Follow next-page link if present (stops when no more pages)
                page_url = _next_page_url(soup, page_url)
                page_num += 1
                if page_url:
                    log.info("  → next page: %s", page_url)
                    await asyncio.sleep(random.uniform(15, 25))
                else:
                    log.info("  → no next page found")

        except Exception:
            log.exception("Failed to scrape DJ page %s", dj_url)

        if i < len(dj_config):
            await asyncio.sleep(random.uniform(5, 12))

    log.info("Collected %d set URLs total", len(all_metas))
    return all_metas


async def _extract_set_urls_from_soup(soup: BeautifulSoup) -> list[str]:
    """Extract tracklist page URLs from an already-fetched DJ page soup."""
    seen, urls = set(), []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip().split("?")[0]
        if "/tracklist/" not in href:
            continue
        if href.rstrip("/") in ("/tracklist", f"{BASE_URL}/tracklist"):
            continue
        full = (BASE_URL + href) if href.startswith("/") else href
        if full not in seen:
            seen.add(full)
            urls.append(full)
    return urls


def _next_page_url(soup: BeautifulSoup, current_url: str) -> str | None:
    """
    Return the URL of the next paginated page on a DJ artist listing, or None.
    1001tracklists uses rel="next" on pagination links, or an arrow/next button.
    """
    # rel="next" is the most reliable signal
    next_link = soup.find("a", rel=lambda r: r and "next" in r)
    if next_link and next_link.get("href"):
        href = next_link["href"].strip()
        return (BASE_URL + href) if href.startswith("/") else href

    # Fallback: look for a link whose text is ">" or "Next"
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        if text in (">", "»", "Next", "next"):
            href = a["href"].strip()
            return (BASE_URL + href) if href.startswith("/") else href

    return None


def _track_id(artist: str, track: str) -> str:
    """Deterministic 12-char ID — identical to preview_fetcher.py and train_model.py."""
    return hashlib.md5(f"{artist}|{track}".lower().encode()).hexdigest()[:12]


def _parse_tracklist_html(soup: BeautifulSoup) -> list[dict]:
    """
    Primary parser: extracts structured track data from div.tlpItem rows.

    Actual HTML structure (confirmed from live page inspection):
      meta[itemprop="name"]     content="Artist - Track Name"
      meta[itemprop="byArtist"] content="Artist"
      span.fontXL               track number ("01", "02", …) or "w/" for overlay
      div.cue                   cue timestamp text ("7:20") or empty

    starting_time: real cue minutes when timestamp present (7:20 → 7.333).
    First track defaults to 0.0. Mid-set tracks with no timestamp → NaN.
    Items with no itemprop="name" meta (ID / unknown tracks) are skipped.
    """
    _ID_TOKENS = {"id", "unknown", ""}

    tracks = []
    last_seq_track_id = None
    last_seq_time     = None

    for position, item in enumerate(soup.select(_ITEM_SEL), start=1):
        # ── Extract artist + title from structured metadata ───────────────────
        name_meta   = item.select_one('meta[itemprop="name"]')
        artist_meta = item.select_one('meta[itemprop="byArtist"]')

        if not name_meta:
            continue   # ID / unknown track — no structured data

        full_name = (name_meta.get("content") or "").strip()
        artist    = (artist_meta.get("content") or "").strip() if artist_meta else ""

        # Derive track name: strip "Artist - " prefix from full name
        if artist and full_name.startswith(artist):
            title = full_name[len(artist):].lstrip(" -").strip()
        elif " - " in full_name:
            artist, title = full_name.split(" - ", 1)
            artist = artist.strip()
            title  = title.strip()
        else:
            title = full_name

        # Skip "ID - ID" entries that somehow have a meta tag
        if artist.upper() in _ID_TOKENS and title.upper() in _ID_TOKENS:
            continue

        # ── Track number — detects w/ (overlay) tracks ───────────────────────
        tno_span   = item.select_one("span.fontXL")
        tno        = tno_span.get_text(strip=True).lower() if tno_span else ""
        is_overlay = tno.startswith("w") and "/" in tno or tno == "w"

        # ── Cue timestamp ─────────────────────────────────────────────────────
        cue_div       = item.select_one("div.cue")
        time_str      = cue_div.get_text(strip=True) if cue_div else ""
        starting_time = _parse_time_to_minutes(time_str)
        if starting_time is None and position == 1:
            starting_time = 0.0

        tid = _track_id(artist, title)

        if is_overlay:
            tracks.append({
                "track_id":       tid,
                "artist_name":    artist or None,
                "track_name":     title  or None,
                "starting_time":  last_seq_time if last_seq_time is not None else float("nan"),
                "play_type":      "simultaneous",
                "overlay_parent": last_seq_track_id,
            })
        else:
            st = starting_time if starting_time is not None else float("nan")
            last_seq_track_id = tid
            last_seq_time     = st
            tracks.append({
                "track_id":       tid,
                "artist_name":    artist or None,
                "track_name":     title  or None,
                "starting_time":  st,
                "play_type":      "sequential",
                "overlay_parent": None,
            })

    return tracks


def _parse_time_to_minutes(time_str: str | None) -> float | None:
    """
    Convert a cue timestamp string to float minutes.
      "6:13"    → 6.217  (6 min 13 sec)
      "1:05:23" → 65.383 (1 hr 5 min 23 sec)
      None / "" → None
    """
    if not time_str:
        return None
    parts = [int(p) for p in time_str.strip().split(":") if p.isdigit()]
    if len(parts) == 2:       # MM:SS
        return parts[0] + parts[1] / 60
    if len(parts) == 3:       # H:MM:SS
        return parts[0] * 60 + parts[1] + parts[2] / 60
    return None


def _parse_tracklist_text(full_text: str) -> list[dict]:
    """
    Fallback text parser used when HTML selectors yield nothing.
    Looks for numbered lines matching 'Artist - Title' patterns.
    starting_time is NaN — training code uses row order for sequencing.
    """
    tracks = []
    for line in full_text.splitlines():
        s = line.strip()
        if not s or " - " not in s:
            continue
        if len(s) > 200 or s.startswith("#"):
            continue

        # Strip any leading track number like "1." or "1)"
        clean = re.sub(r"^\d+[\.\)]\s*", "", s)

        parts = clean.split(" - ", 1)
        if len(parts) != 2:
            continue
        artist, title = parts[0].strip(), parts[1].strip()
        if not artist or not title:
            continue
        if artist.upper() in ("ID", "UNKNOWN") or title.upper() in ("ID", "UNKNOWN"):
            continue

        tracks.append({
            "artist_name":    artist,
            "track_name":     title,
            "starting_time":  float("nan"),
            "play_type":      "sequential",
            "overlay_parent": None,
        })

    return tracks


def _results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "mix_id":    hashlib.md5(d["url"].encode()).hexdigest()[:10],
            "mix_title": d.get("title", ""),
            "dj_name":   d.get("dj_name", ""),
            "genre":     d.get("genre", ""),
            **track,
        }
        for d in results
        for track in d.get("tracklist", [])
    ]
    return pd.DataFrame(
        rows,
        columns=["mix_id", "mix_title", "dj_name", "genre",
                 "track_id", "starting_time", "track_name", "artist_name",
                 "play_type", "overlay_parent"],
    )


def _dj_name(url: str) -> str:
    """
    Extract a human-readable DJ name from a 1001tracklists artist URL.
    https://www.1001tracklists.com/dj/charlotte-de-witte/index.html
      → "Charlotte De Witte"
    """
    try:
        slug = url.rstrip("/").split("/dj/")[-1].split("/")[0]
        return slug.replace("-", " ").title()
    except Exception:
        return url


def _text(tag) -> str:
    """Return stripped text of a BeautifulSoup tag, or empty string if None."""
    return tag.get_text(strip=True) if tag else ""


async def _close(page) -> None:
    try:
        await page.close()
    except Exception:
        pass


def _log_summary(results: list[dict]) -> None:
    ok = sum(1 for r in results if r["success"])
    log.info("Scraped %d/%d sets successfully", ok, len(results))
    for r in results:
        log.info("  %s → %d tracks", r.get("title", "?"), r["track_count"])


def _save(results: list[dict]) -> None:
    df = _results_to_dataframe(results)
    if df.empty:
        log.warning("No tracks parsed — skipping CSV save")
        return
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        existing = existing[~existing["mix_id"].isin(df["mix_id"])]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved %d rows → %s", len(df), OUTPUT_PATH)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DJ_CONFIG = [
        # Verify slug: open https://www.1001tracklists.com, search for the DJ,
        # and copy the URL from their profile page — slugs are case-sensitive.
        {"url": "https://www.1001tracklists.com/dj/charlottedewitte/index.html", "genre": "techno"},
        {"url": "https://www.1001tracklists.com/dj/adambeyer/index.html",       "genre": "techno"},
        # {"url": "https://www.1001tracklists.com/dj/dixon/index.html",           "genre": "deep house"},
        # {"url": "https://www.1001tracklists.com/dj/richiehawtin/index.html",    "genre": "minimal techno"},
        # {"url": "https://www.1001tracklists.com/dj/paulvandyk/index.html",      "genre": "trance"},
        # {"url": "https://www.1001tracklists.com/dj/goldie/index.html",          "genre": "drum and bass"},
    ]
    asyncio.run(scrape_mixes_from_djs(DJ_CONFIG))
