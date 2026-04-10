#!/usr/bin/env python3
"""MixesDB scraper: collect mix URLs from DJ category pages, then scrape tracklists."""
import asyncio
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


async def scrape_multiple_mixes(urls: list[str], browser=None) -> list[dict]:
    """Scrape tracklists from a list of mix URLs. Reuses browser if provided."""
    own_browser = browser is None
    if own_browser:
        browser = await nd.start(headless=False)

    results = []
    try:
        for i, url in enumerate(urls, 1):
            log.info("[%d/%d] %s", i, len(urls), url)
            page = await browser.get(url, new_tab=True)
            await asyncio.sleep(random.uniform(7, 13))
            data = await extract_tracklist(page)
            await _close(page)
            data["url"] = url
            results.append(data)
            log.info("  → %d tracks", data["track_count"])
            if i < len(urls):
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


async def scrape_mixes_from_djs(dj_urls: list[str]) -> list[dict]:
    """
    Main entry point. Two-step process:
      1. Visit each DJ category page → collect mix URLs
      2. Visit each mix URL → scrape tracklist
    Saves output to data/interim/tracklist.csv.
    """
    browser = await nd.start(headless=False)
    try:
        mix_urls = await _collect_mix_urls(browser, dj_urls)
        if not mix_urls:
            log.warning("No mix URLs found.")
            return []
        return await scrape_multiple_mixes(mix_urls, browser=browser)
    finally:
        browser.stop()
        await asyncio.sleep(0.5)


# ── Private helpers ────────────────────────────────────────────────────────────

async def _collect_mix_urls(browser, dj_urls: list[str]) -> list[str]:
    """Step 1: collect all mix page URLs across the given DJ category pages."""
    all_urls, seen = [], set()
    for i, dj_url in enumerate(dj_urls, 1):
        log.info("[%d/%d] Scraping DJ page: %s", i, len(dj_urls), _dj_name(dj_url))
        try:
            page = await browser.get(dj_url, new_tab=True)
            await asyncio.sleep(random.uniform(4, 8))
            for url in await extract_mix_urls_from_dj_page(page):
                if url not in seen:
                    seen.add(url)
                    all_urls.append(url)
            await _close(page)
        except Exception:
            log.exception("Failed to scrape DJ page %s", dj_url)
        if i < len(dj_urls):
            await asyncio.sleep(random.uniform(5, 12))
    log.info("Collected %d mix URLs total", len(all_urls))
    return all_urls


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
    return {"artist_name": artist.strip() or None, "track_name": track.strip() or None, "starting_time": starting_time}


def _results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    rows = [
        {"mix_id": mix_id, "url": d["url"], **parsed}
        for mix_id, d in enumerate(results)
        for line in d.get("tracklist", [])
        if (parsed := _parse_track_line(line))
    ]
    return pd.DataFrame(rows, columns=["mix_id", "url", "starting_time", "track_name", "artist_name"])


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
    df.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved %d rows → %s", len(df), OUTPUT_PATH)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DJ_URLS = [
        "https://www.mixesdb.com/w/Category:Roman_Fl%C3%BCgel",
    ]
    asyncio.run(scrape_mixes_from_djs(DJ_URLS))
