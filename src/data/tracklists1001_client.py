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
from pathlib import Path
import random
import re

from bs4 import BeautifulSoup
import nodriver as nd
import pandas as pd

LOG_PATH = Path(__file__).parent.parent.parent / "logs" / "tracklists1001_client.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    _fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    _sh  = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    _fh  = logging.FileHandler(LOG_PATH, encoding="utf-8")
    _fh.setFormatter(_fmt)
    log.addHandler(_sh)
    log.addHandler(_fh)
    log.propagate = False   # don't double-log via nodriver's root logger

BASE_URL    = "https://www.1001tracklists.com"
OUTPUT_PATH = Path("data/interim/tracklist.csv")
URL_CACHE   = Path("data/interim/mix_urls.csv")   # persisted URL collection cache

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


CHECKPOINT_EVERY    = 20    # save + flush results every N mixes
BROWSER_RESTART_AT  = 200   # restart Chrome after this many mixes to prevent memory/connection exhaustion

async def scrape_multiple_mixes(url_metas: list[dict], browser=None) -> list[dict]:
    """
    Scrape tracklists from a list of set URLs. Reuses browser if provided.
    url_metas: list of dicts with keys: url, dj_name, genre

    Skips URLs whose mix_id already exists in OUTPUT_PATH (resume support).
    Saves a checkpoint every CHECKPOINT_EVERY mixes and clears the in-memory
    buffer to bound RAM usage.
    """
    # ── Build set of already-scraped URLs for resume ─────────────────────────
    # Prefer direct URL matching (new schema has url column).
    # Legacy fallback: if url column absent, derive done URLs via mix_id hash.
    done_urls: set[str] = set()
    if OUTPUT_PATH.exists():
        try:
            existing = pd.read_csv(OUTPUT_PATH)
            if "url" in existing.columns:
                done_urls = set(existing["url"].dropna().unique())
            else:
                done_ids = set(existing["mix_id"].dropna().unique())
                done_urls = {
                    m["url"] for m in url_metas
                    if hashlib.md5(m["url"].encode()).hexdigest()[:10] in done_ids
                }
            log.info("Resume: %d URLs already scraped — will skip", len(done_urls))
        except Exception:
            log.warning("Could not read existing CSV for resume check — will re-scrape all")

    # Filter out already-done URLs
    pending = [m for m in url_metas if m["url"] not in done_urls]
    skipped = len(url_metas) - len(pending)
    if skipped:
        log.info("Skipping %d already-scraped mixes, %d remaining", skipped, len(pending))

    PAGE_LOAD_TIMEOUT  = 60    # seconds to wait for tab.get()
    SCROLL_TIMEOUT     = 210   # MAX_SCROLL_STEPS * 3s + buffer
    PARSE_TIMEOUT      = 30    # BeautifulSoup parse should be instant
    OFFLINE_RETRY_WAIT = 120   # seconds to wait before retrying when offline
    OFFLINE_MAX_RETRY  = 5     # give up and save+exit after this many consecutive offline hits
    TIMEOUT_MAX        = 3     # exit after this many consecutive timeouts (browser offline)

    async def _start_browser_and_tab():
        b   = await nd.start(headless=False, no_sandbox=True)
        t   = await b.get("about:blank", new_tab=True)
        return b, t

    browser, tab = await _start_browser_and_tab()

    # buffer holds results since the last checkpoint; flushed after each save
    buffer: list[dict] = []
    total_done   = skipped
    offline_streak = 0
    timeout_streak = 0
    scrapes_since_restart = 0

    try:
        for i, meta in enumerate(pending, 1):
            url = meta["url"]
            log.info("[%d/%d] %s", i, len(pending), url)

            # ── Periodic browser restart to prevent memory/connection exhaustion ─
            if scrapes_since_restart >= BROWSER_RESTART_AT:
                log.info("  restarting browser after %d mixes", scrapes_since_restart)
                if buffer:
                    _save(buffer)
                    log.info("  checkpoint saved before restart (%d mixes this run)", i - 1)
                    buffer.clear()
                await _close(tab)
                browser.stop()
                await asyncio.sleep(2)
                browser, tab = await _start_browser_and_tab()
                scrapes_since_restart = 0
                log.info("  browser restarted — continuing from mix %d", i)

            try:
                await asyncio.wait_for(tab.get(url), timeout=PAGE_LOAD_TIMEOUT)
                await asyncio.sleep(random.uniform(3, 5))   # initial render

                # ── Connectivity check ────────────────────────────────────────
                # Detect Chrome error pages (offline / DNS failure / blocked).
                page_title = await tab.evaluate("document.title")
                if any(tok in page_title for tok in
                       ("ERR_", "No internet", "can't be reached", "not available")):
                    offline_streak += 1
                    log.warning("  Offline or blocked (streak %d) — waiting %ds before retry",
                                offline_streak, OFFLINE_RETRY_WAIT)
                    if offline_streak >= OFFLINE_MAX_RETRY:
                        log.error("  %d consecutive offline hits — saving and exiting. "
                                  "Resume when internet is back.", OFFLINE_MAX_RETRY)
                        break
                    await asyncio.sleep(OFFLINE_RETRY_WAIT)
                    pending.insert(i, meta)   # re-queue this URL at current position
                    continue

                offline_streak = 0   # reset on successful load

                await asyncio.wait_for(
                    _scroll_to_bottom(tab, label="mix page"), timeout=SCROLL_TIMEOUT
                )
                data = await asyncio.wait_for(
                    extract_tracklist(tab), timeout=PARSE_TIMEOUT
                )
            except asyncio.TimeoutError:
                timeout_streak += 1
                log.warning("  Timed out (streak %d) — skipping %s", timeout_streak, url)
                if timeout_streak >= TIMEOUT_MAX:
                    log.error("  %d consecutive timeouts — browser likely offline. "
                              "Saving and exiting. Resume when internet is back.", TIMEOUT_MAX)
                    break
                continue

            timeout_streak = 0   # reset on any successful load

            data["url"]     = url
            data["dj_name"] = meta["dj_name"]
            data["genre"]   = meta["genre"]
            buffer.append(data)
            total_done += 1
            scrapes_since_restart += 1
            log.info("  → %d tracks (total mixes done: %d)", data["track_count"], total_done)
            if i % CHECKPOINT_EVERY == 0:
                _save(buffer)
                log.info("  checkpoint saved (%d mixes this run, %d total)", i, total_done)
                buffer.clear()   # free memory
            if i < len(pending):
                await asyncio.sleep(random.uniform(2, 5))
    except Exception:
        log.exception("Error during mix scraping")
    finally:
        await _close(tab)
        if buffer:
            _save(buffer)
            log.info("  final checkpoint saved (%d mixes this run)", len(buffer))
            buffer.clear()
        browser.stop()
        await asyncio.sleep(0.5)

    log.info("scrape_multiple_mixes complete: %d mixes processed this run", total_done - skipped)
    return []


async def scrape_mixes_from_djs(dj_config: list[dict]) -> list[dict]:
    """
    Main entry point. Two-step process:
      1. Visit each DJ artist page → collect set URLs  (skipped per DJ if already cached)
      2. Visit each set URL → scrape tracklist          (skipped if already in tracklist.csv)
    Saves output to data/interim/tracklist.csv.

    dj_config: list of dicts with keys: url (DJ artist page), genre
      e.g. [{"url": "https://www.1001tracklists.com/dj/charlotte-de-witte/index.html",
              "genre": "techno"}]
    """
    browser = await nd.start(headless=False, no_sandbox=True)
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

MAX_SCROLL_STEPS = 60   # safety cap (~3 min at 3s/step)

async def _scroll_to_bottom(page, label: str = "page") -> None:
    """Scroll down until no new content loads (handles infinite-scroll pages)."""
    prev_height = 0
    step = 0
    while step < MAX_SCROLL_STEPS:
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(2)
        height = await page.evaluate("document.body.scrollHeight")
        if height == prev_height:
            log.info("  %s fully scrolled (%d steps, %dpx)", label, step, height)
            break
        prev_height = height
        step += 1
        log.info("  scrolled to %dpx", height)
    else:
        log.warning("  %s hit max scroll steps (%d) at %dpx — continuing anyway", label, MAX_SCROLL_STEPS, prev_height)


MAX_SETS_PER_DJ = 250

async def _collect_mix_urls(browser, dj_config: list[dict]) -> list[dict]:
    """
    Step 1: visit each DJ artist page (and its paginated sub-pages) to collect
    all set URLs. Returns list of dicts: {url, dj_name, genre}.
    Stops collecting for a DJ once MAX_SETS_PER_DJ URLs are found.

    DJs already marked complete=True in mix_urls.csv are skipped — their cached
    URLs are returned directly. After finishing each new DJ, saves to cache with
    complete=True so future runs skip it.
    """
    cache_df = _load_url_cache()
    complete_djs = set(cache_df[cache_df["complete"]]["dj_name"].unique())

    # Seed all_metas and seen ONLY from complete DJs in this config.
    # Incomplete DJs are intentionally excluded so re-collection starts fresh
    # and all their URLs end up in dj_metas → saved correctly to cache.
    # NOTE: load ALL rows for complete DJs (including complete=False ones the user
    # reset for re-scraping) — the actual skip decision lives in scrape_multiple_mixes
    # via done_urls from tracklist.csv. complete=False rows that were never saved to
    # tracklist.csv will appear in pending and get re-scraped correctly.
    config_names = {_dj_name(c["url"]) for c in dj_config}
    complete_config_djs = config_names & complete_djs
    cached_for_batch = cache_df[cache_df["dj_name"].isin(complete_config_djs)]
    all_metas = cached_for_batch[["url", "dj_name", "genre"]].to_dict("records")
    seen = {m["url"] for m in all_metas}

    for i, cfg in enumerate(dj_config, 1):
        dj_url = cfg["url"]
        genre  = cfg["genre"]
        name   = _dj_name(dj_url)

        if name in complete_djs:
            count = sum(1 for m in all_metas if m["dj_name"] == name)
            log.info("[%d/%d] %s — cached (%d URLs), skipping collection", i, len(dj_config), name, count)
            continue

        log.info("[%d/%d] Scraping DJ page: %s (%s)", i, len(dj_config), name, genre)
        dj_metas = []
        dj_count = 0
        try:
            page_url = dj_url
            page_num = 1
            while page_url:
                page = await browser.get(page_url, new_tab=True)
                await asyncio.sleep(random.uniform(20, 25))
                await _scroll_to_bottom(page, label="DJ page")
                soup = BeautifulSoup(await page.get_content(), "html.parser")
                page_title = soup.title.string.strip() if soup.title else "NO TITLE"
                log.info("  page title: %s", page_title)
                await _close(page)

                new_on_page = 0
                for url in await _extract_set_urls_from_soup(soup):
                    if url not in seen:
                        seen.add(url)
                        dj_metas.append({"url": url, "dj_name": name, "genre": genre})
                        new_on_page += 1
                        dj_count += 1
                        if dj_count >= MAX_SETS_PER_DJ:
                            log.info("  reached %d set limit for %s", MAX_SETS_PER_DJ, name)
                            break

                log.info("  page %d: +%d sets (%d total for %s)", page_num, new_on_page, dj_count, name)
                if dj_count >= MAX_SETS_PER_DJ:
                    break

                page_url = _next_page_url(soup, page_url)
                page_num += 1
                if page_url:
                    log.info("  → next page: %s", page_url)
                    await asyncio.sleep(random.uniform(15, 25))
                else:
                    log.info("  → no next page found")

            # Collection finished for this DJ
            if dj_metas:
                all_metas.extend(dj_metas)
                cache_df = cache_df[cache_df["dj_name"] != name]   # drop stale rows
                new_rows = pd.DataFrame([{**m, "complete": True} for m in dj_metas])
                cache_df = pd.concat([cache_df, new_rows], ignore_index=True)
                _save_url_cache(cache_df)
                complete_djs.add(name)
                log.info("  %s collection complete: %d URLs cached", name, dj_count)
            else:
                log.warning("  %s: 0 URLs found — will retry next run", name)

        except Exception:
            log.exception("Failed to scrape DJ page %s", dj_url)
            # Drop any stale rows for this DJ and save partial as incomplete
            # so next run knows to re-collect from scratch
            cache_df = cache_df[cache_df["dj_name"] != name]
            if dj_metas:
                new_rows = pd.DataFrame([{**m, "complete": False} for m in dj_metas])
                cache_df = pd.concat([cache_df, new_rows], ignore_index=True)
                try:
                    _save_url_cache(cache_df)
                except Exception:
                    log.warning("Could not save URL cache after error for %s — continuing", name)
            all_metas.extend(dj_metas)

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
        url = (BASE_URL + href) if href.startswith("/") else href
        return url if url != current_url else None

    # Fallback: look for a link whose text is ">" or "Next"
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        if text in (">", "»", "Next", "next"):
            href = a["href"].strip()
            url = (BASE_URL + href) if href.startswith("/") else href
            return url if url != current_url else None

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
        # Default first sequential track to 0.0 when no timestamp present.
        # Use `not tracks` rather than `position == 1` because position counts
        # all tlpItems including skipped ones — the first real track may have
        # position > 1 if earlier items were skipped.
        if starting_time is None and not tracks:
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
            "track_id":       _track_id(artist, title),
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
            "url":       d["url"],
            **track,
        }
        for d in results
        for track in d.get("tracklist", [])
    ]
    return pd.DataFrame(
        rows,
        columns=["mix_id", "mix_title", "dj_name", "genre", "url",
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
        try:
            existing = pd.read_csv(OUTPUT_PATH)
            existing = existing[~existing["mix_id"].isin(df["mix_id"])]
            df = pd.concat([existing, df], ignore_index=True)
        except Exception:
            log.warning("Could not read existing %s (corrupt?) — overwriting with current batch", OUTPUT_PATH)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved %d rows → %s", len(df), OUTPUT_PATH)


def _load_url_cache() -> pd.DataFrame:
    """
    Load mix_urls.csv. Returns empty DataFrame with correct columns if not found.
    Columns: url, dj_name, genre, complete (bool)
    """
    if URL_CACHE.exists():
        try:
            df = pd.read_csv(URL_CACHE)
            # CSV round-trip stores True/False as strings — normalise to Python bool
            if "complete" not in df.columns:
                df["complete"] = False
            df["complete"] = df["complete"].astype(str).str.lower() == "true"
            return df
        except Exception:
            log.warning("Could not read %s (corrupt?) — starting with empty cache", URL_CACHE)
    return pd.DataFrame(columns=["url", "dj_name", "genre", "complete"])


def _save_url_cache(cache_df: pd.DataFrame) -> None:
    URL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(URL_CACHE, index=False)
    log.info("URL cache saved: %d rows → %s", len(cache_df), URL_CACHE)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DJ_CONFIG = [
        # Verify slug: open https://www.1001tracklists.com, search for the DJ,
        # and copy the URL from their profile page — slugs are case-sensitive.
        # Techno 8
        {"url": "https://www.1001tracklists.com/dj/charlottedewitte/index.html", "genre": "techno"},
        {"url": "https://www.1001tracklists.com/dj/klangkuenstler/index.html",       "genre": "techno"},
        {"url": "https://www.1001tracklists.com/dj/amelielens/index.html",       "genre": "techno"},
        {"url": "https://www.1001tracklists.com/dj/saralandry/index.html",       "genre": "techno"},
        {"url": "https://www.1001tracklists.com/dj/holypriest/index.html",       "genre": "techno"},
        {"url": "https://www.1001tracklists.com/dj/999999999/index.html",       "genre": "techno"},
        {"url": "https://www.1001tracklists.com/dj/ihatemodels/index.html",       "genre": "techno"},
        {"url": "https://www.1001tracklists.com/dj/adambeyer/index.html",       "genre": "techno"},

        #tech house 8
        {"url": "https://www.1001tracklists.com/dj/solomun/index.html",           "genre": "tech house"},
        {"url": "https://www.1001tracklists.com/dj/romanflugel/index.html",           "genre": "tech house"},
        {"url": "https://www.1001tracklists.com/dj/djtennis/index.html",           "genre": "tech house"},
        {"url": "https://www.1001tracklists.com/dj/fisher/index.html",           "genre": "tech house"},
        {"url": "https://www.1001tracklists.com/dj/hotsince82/index.html",           "genre": "tech house"},
        {"url": "https://www.1001tracklists.com/dj/johnsummit/index.html",           "genre": "tech house"},
        {"url": "https://www.1001tracklists.com/dj/chrislake/index.html",           "genre": "tech house"},
        {"url": "https://www.1001tracklists.com/dj/carlcox/index.html",           "genre": "tech house"},

        # melodic house 9
        {"url": "https://www.1001tracklists.com/dj/taleofus/index.html", "genre": "melodic house"},
        {"url": "https://www.1001tracklists.com/dj/anyma/index.html", "genre": "melodic house"},
        {"url": "https://www.1001tracklists.com/dj/massano/index.html", "genre": "melodic house"},
        {"url": "https://www.1001tracklists.com/dj/monolink/index.html", "genre": "melodic house"},
        {"url": "https://www.1001tracklists.com/dj/benbohmer/index.html", "genre": "melodic house"},
        {"url": "https://www.1001tracklists.com/dj/rufusdusol/index.html", "genre": "melodic house"},
        {"url": "https://www.1001tracklists.com/dj/sultanplusshepard/index.html", "genre": "melodic house"},
        {"url": "https://www.1001tracklists.com/dj/argy/index.html", "genre": "melodic house"},
        {"url": "https://www.1001tracklists.com/dj/sultanplusshepard/index.html", "genre": "deep house"},

        # afro house 9
        {"url": "https://www.1001tracklists.com/dj/black-coffee/index.html", "genre": "afro house"},
        {"url": "https://www.1001tracklists.com/dj/themba/index.html", "genre": "afro house"},
        {"url": "https://www.1001tracklists.com/dj/enoo-napa/index.html", "genre": "afro house"},
        {"url": "https://www.1001tracklists.com/dj/da-capo/index.html", "genre": "afro house"},
        {"url": "https://www.1001tracklists.com/dj/shimza/index.html", "genre": "afro house"},
        {"url": "https://www.1001tracklists.com/dj/samm-be/index.html", "genre": "afro house"},
        {"url": "https://www.1001tracklists.com/dj/moblack/index.html", "genre": "afro house"},
        {"url": "https://www.1001tracklists.com/dj/nitefreak/index.html", "genre": "afro house"},
        {"url": "https://www.1001tracklists.com/dj/aaronsevilla/index.html", "genre": "afro house"},

        # trance 7
        {"url": "https://www.1001tracklists.com/dj/paulvandyk/index.html", "genre": "trance"},
        {"url": "https://www.1001tracklists.com/dj/arminvanbuuren/index.html", "genre": "trance"},
        {"url": "https://www.1001tracklists.com/dj/markusschulz/index.html", "genre": "trance"},
        {"url": "https://www.1001tracklists.com/dj/johnocallaghan/index.html", "genre": "trance"},
        {"url": "https://www.1001tracklists.com/dj/ferrycorsten/index.html", "genre": "trance"},
        {"url": "https://www.1001tracklists.com/dj/rank1/index.html", "genre": "trance"},
        {"url": "https://www.1001tracklists.com/dj/alyandfila/index.html", "genre": "trance"},

        # drum and base 9
        {"url": "https://www.1001tracklists.com/dj/hedex/index.html", "genre": "drum and base"},
        {"url": "https://www.1001tracklists.com/dj/andyc/index.html", "genre": "drum and base"},
        {"url": "https://www.1001tracklists.com/dj/calibre/index.html", "genre": "drum and base"},
        {"url": "https://www.1001tracklists.com/dj/ltjbukem/index.html", "genre": "drum and base"},
        {"url": "https://www.1001tracklists.com/dj/noisia/index.html", "genre": "drum and base"},
        {"url": "https://www.1001tracklists.com/dj/subfocus/index.html", "genre": "drum and base"},
        {"url": "https://www.1001tracklists.com/dj/dimension/index.html", "genre": "drum and base"},
        {"url": "https://www.1001tracklists.com/dj/chasestatus/index.html", "genre": "drum and base"},
        {"url": "https://www.1001tracklists.com/dj/pendulum/index.html", "genre": "drum and base"},
    ]
    log.info("Running all %d DJs", len(DJ_CONFIG))
    asyncio.run(scrape_mixes_from_djs(DJ_CONFIG))
