#!/usr/bin/env python3
"""MixesDB scraper: collect mix URLs from DJ category pages, then scrape tracklists."""
import asyncio
import hashlib
import logging
import random
import re
from pathlib import Path
from urllib.parse import unquote

import nodriver as nd
import pandas as pd
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://www.mixesdb.com"
OUTPUT_PATH = Path("data/interim/tracklist.csv")
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".bmp", ".tiff", ".tif")


# ── Public scraping functions ──────────────────────────────────────────────────

async def extract_mix_urls_from_dj_page(page, base_url: str = BASE_URL) -> list[str]:
    """Return all mix page URLs found on a DJ category page."""
    soup = BeautifulSoup(await page.get_content(), "html.parser")
    seen, urls = set(), []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip().split("?")[0]
        if not href.startswith("/w/"):
            continue
        title = href.replace("/w/", "")
        if not title or any(title.startswith(p) for p in ("Category:", "Help:", "MixesDB:")):
            continue
        if href.lower().endswith(_IMAGE_EXT):
            continue
        full = (base_url + href) if href.startswith("/") else href
        if full not in seen:
            seen.add(full)
            urls.append(full)

    return urls


async def extract_tracklist(page) -> dict:
    """Return title and raw tracklist lines from the current mix page."""
    soup = BeautifulSoup(await page.get_content(), "html.parser")

    title = soup.title.string.strip() if soup.title else "Unknown"
    if "|" in title:
        title = title.split("|")[0].strip()

    full_text = soup.get_text(separator="\n", strip=True)
    tracklist = _parse_tracklist_section(full_text)

    return {"title": title, "tracklist": tracklist, "track_count": len(tracklist), "success": bool(tracklist)}


async def scrape_multiple_mixes(url_metas: list[dict], browser=None) -> list[dict]:
    """
    Scrape tracklists from a list of mix URLs. Reuses browser if provided.

    url_metas: list of dicts with keys: url, dj_name, genre
    """
    own_browser = browser is None
    if own_browser:
        browser = await nd.start(headless=False)

    results = []
    try:
        for i, meta in enumerate(url_metas, 1):
            url = meta["url"]
            log.info("[%d/%d] %s", i, len(url_metas), url)
            page = await browser.get(url, new_tab=True)
            await asyncio.sleep(random.uniform(7, 13))
            data = await extract_tracklist(page)
            await _close(page)
            data["url"]     = url
            data["dj_name"] = meta["dj_name"]
            data["genre"]   = meta["genre"]
            results.append(data)
            log.info("  → %d tracks", data["track_count"])
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
      1. Visit each DJ category page → collect mix URLs
      2. Visit each mix URL → scrape tracklist
    Saves output to data/interim/tracklist.csv.

    dj_config: list of dicts with keys: url (DJ category page), genre
      e.g. [{"url": "https://www.mixesdb.com/w/Category:Charlotte_de_Witte", "genre": "techno"}]
    """
    browser = await nd.start(headless=False)
    try:
        url_metas = await _collect_mix_urls(browser, dj_config)
        if not url_metas:
            log.warning("No mix URLs found.")
            return []
        return await scrape_multiple_mixes(url_metas, browser=browser)
    finally:
        browser.stop()
        await asyncio.sleep(0.5)


# ── Private helpers ────────────────────────────────────────────────────────────

async def _collect_mix_urls(browser, dj_config: list[dict]) -> list[dict]:
    """
    Step 1: collect all mix page URLs across the given DJ category pages.
    Returns list of dicts: {url, dj_name, genre} — one entry per mix page found.
    """
    all_metas, seen = [], set()
    for i, cfg in enumerate(dj_config, 1):
        dj_url  = cfg["url"]
        genre   = cfg["genre"]
        name    = _dj_name(dj_url)
        log.info("[%d/%d] Scraping DJ page: %s (%s)", i, len(dj_config), name, genre)
        try:
            page = await browser.get(dj_url, new_tab=True)
            await asyncio.sleep(random.uniform(4, 8))
            for url in await extract_mix_urls_from_dj_page(page):
                if url not in seen:
                    seen.add(url)
                    all_metas.append({"url": url, "dj_name": name, "genre": genre})
            await _close(page)
        except Exception:
            log.exception("Failed to scrape DJ page %s", dj_url)
        if i < len(dj_config):
            await asyncio.sleep(random.uniform(5, 12))
    log.info("Collected %d mix URLs total", len(all_metas))
    return all_metas


def _parse_tracklist_section(full_text: str) -> list[str]:
    """Extract track lines from the full page text."""
    # Primary: regex between "Tracklist" and "Related mixes"
    match = re.search(
        r"(?ms)^Tracklist\s*$\n(.*?)(?=^Related mixes\s*$|^$)",
        full_text,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if match:
        lines = [l.strip() for l in match.group(1).splitlines() if l.strip()]
        tracks = [l for l in lines if re.match(r"^\d+\)", l) or (" - " in l and "[" in l)]
        if tracks:
            return tracks

    # Fallback: line-by-line scan
    tracks, capturing = [], False
    for line in full_text.splitlines():
        s = line.strip()
        if s == "Tracklist":
            capturing = True
            continue
        if capturing:
            if s.startswith("Related mixes"):
                break
            if s and (" - " in s or "[" in s):
                tracks.append(s)
    return tracks


def _parse_track_line(line: str) -> dict | None:
    """Parse 'Artist - Track Name [label] [time]' into structured fields."""
    time_match = re.search(r"\[(\d+)\]", line)
    starting_time = int(time_match.group(1)) if time_match else None
    rest = line[time_match.end():].strip() if time_match else re.sub(r"^\d+[\.\)]\s*", "", line.strip())
    if " - " not in rest:
        return None
    artist, track = rest.split(" - ", 1)
    a, t = artist.strip(), track.strip()
    return {
        "track_id":    hashlib.md5(f"{a}|{t}".lower().encode()).hexdigest()[:12],
        "artist_name": a or None,
        "track_name":  t or None,
        "starting_time": starting_time,
    }


def _results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    rows = [
        {
            # Stable URL hash — unique across DJs, reproducible on re-runs
            "mix_id":    hashlib.md5(d["url"].encode()).hexdigest()[:10],
            "mix_title": d.get("title", ""),
            "dj_name":   d.get("dj_name", ""),
            "genre":     d.get("genre", ""),
            **parsed,
        }
        for d in results
        for line in d.get("tracklist", [])
        if (parsed := _parse_track_line(line))
    ]
    df = pd.DataFrame(
        rows,
        columns=["mix_id", "mix_title", "dj_name", "genre",
                 "track_id", "starting_time", "track_name", "artist_name"],
    )
    df["play_type"]      = "sequential"  # MixesDB has no simultaneous play concept
    df["overlay_parent"] = None
    return df


def _dj_name(url: str) -> str:
    try:
        return unquote(url.split("/w/")[-1].split("?")[0]).replace("Category:", "").replace("_", " ").strip()
    except Exception:
        return url


async def _close(page) -> None:
    try:
        await page.close()
    except Exception:
        pass


def _log_summary(results: list[dict]) -> None:
    ok = sum(1 for r in results if r["success"])
    log.info("Scraped %d/%d mixes successfully", ok, len(results))
    for r in results:
        log.info("  %s → %d tracks", r["title"], r["track_count"])


def _save(results: list[dict]) -> None:
    df = _results_to_dataframe(results)
    if df.empty:
        log.warning("No tracks parsed — skipping CSV save")
        return
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        # Drop any rows for mix_ids we just re-scraped, then append fresh data.
        # This makes the scraper idempotent: re-running a DJ updates their rows
        # without duplicating or losing other DJs already in the file.
        existing = existing[~existing["mix_id"].isin(df["mix_id"])]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved %d rows → %s", len(df), OUTPUT_PATH)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DJ_CONFIG = [
        # {"url": "https://www.mixesdb.com/w/Category:Roman_Fl%C3%BCgel",      "genre": "minimal techno"},
        # Add more DJs here — genre is manual (MixesDB page tags are inconsistent):
        {"url": "https://www.mixesdb.com/w/Category:Charlotte_de_Witte",   "genre": "techno"},
        # {"url": "https://www.mixesdb.com/w/Category:Dixon",                "genre": "deep house"},
        # {"url": "https://www.mixesdb.com/w/Category:Ricardo_Villalobos",   "genre": "minimal techno"},
        # {"url": "https://www.mixesdb.com/w/Category:Paul_van_Dyk",         "genre": "trance"},
        # {"url": "https://www.mixesdb.com/w/Category:Goldie",               "genre": "drum and bass"},
    ]
    asyncio.run(scrape_mixes_from_djs(DJ_CONFIG))
