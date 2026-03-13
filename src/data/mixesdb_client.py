import asyncio
import nodriver as nd
from bs4 import BeautifulSoup
import re
import random


async def extract_tracklist(page) -> dict:
    """Extract title and tracklist from current page"""
    html = await page.get_content()
    soup = BeautifulSoup(html, "html.parser")

    # Title cleanup
    title = soup.title.string.strip() if soup.title else "Unknown"
    if "|" in title:
        title = title.split("|")[0].strip()

    # Full visible text with line breaks preserved
    full_text = soup.get_text(separator="\n", strip=True)
    print(full_text)
    # Your pattern: from line "Tracklist" until line "Related mixes"
    pattern = r"(?ms)^Tracklist\s*$\n(.*?)(?=^Related mixes\s*$|^$)"

    match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)

    tracklist = []
    if match:
        section = match.group(1).strip()
        lines = [line.strip() for line in section.splitlines() if line.strip()]

        # Filter to keep only actual track lines
        for line in lines:
            # Typical track patterns: number) artist - title [label]  or just artist - title [label]
            if re.match(r'^\d+\)', line) or (" - " in line and "[" in line):
                tracklist.append(line)
            elif tracklist and len(line) > 20:  # continue adding if already started
                tracklist.append(line)

    # Fallback if regex missed (common with slight formatting differences)
    if not tracklist:
        lines = full_text.splitlines()
        capturing = False
        for line in lines:
            stripped = line.strip()
            if stripped == "Tracklist":
                capturing = True
                continue
            if capturing:
                if stripped.startswith("Related mixes"):
                    break
                if stripped and (" - " in stripped or "[" in stripped):
                    tracklist.append(stripped)

    return {
        "title": title,
        "tracklist": tracklist,
        "track_count": len(tracklist),
        "success": len(tracklist) > 0
    }


async def scrape_multiple_mixes(urls: list[str]):
    browser = await nd.start(headless=False)  # Keep False as it works for you

    results = []

    try:
        for idx, url in enumerate(urls, 1):
            print(f"[{idx}/{len(urls)}] Opening: {url}")

            # New tab for each mix
            page = await browser.get(url, new_tab=True)

            # Wait for load + random human-like pause
            await asyncio.sleep(random.uniform(7, 13))

            data = await extract_tracklist(page)
            data["url"] = url
            results.append(data)

            # Delay between pages (anti-detection)
            if idx < len(urls):
                delay = random.uniform(12, 30)  # adjust range as needed
                print(f"   Waiting {delay:.1f} seconds before next...")
                await asyncio.sleep(delay)

    except Exception as e:
        print(f"\nError during session: {e}")

    finally:
        print("\nClosing browser session.")
        browser.stop()

    # Final summary
    print("\n" + "═" * 60)
    print("SCRAPING SUMMARY")
    print("═" * 60)
    success = sum(1 for r in results if r["success"])
    print(f"Success: {success}/{len(urls)} mixes")
    for r in results:
        print(f"• {r['title']} → {r['track_count']} tracks")

    return results


# ────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────
if __name__ == "__main__":
    dj_name = [
        "https://www.mixesdb.com/w/Category:Roman_Fl%C3%BCgel",
    ]
    urls = [
        "https://www.mixesdb.com/w/Category:Roman_Fl%C3%BCgel"
        # "https://www.mixesdb.com/w/2020-07-21_-_Ricardo_Da_Rhythm_-_Da_Rhythm_Sessions,_D3EP_Radio_Network",
        # "https://www.mixesdb.com/w/2020-08-25_-_Ricardo_Da_Rhythm_-_Da_Rhythm_Sessions,_D3EP_Radio_Network",
        # "https://www.mixesdb.com/w/2020-08-18_-_Ricardo_Da_Rhythm_-_Da_Rhythm_Sessions,_D3EP_Radio_Network",
    ]

    results = asyncio.run(scrape_multiple_mixes(urls))

    print(results)