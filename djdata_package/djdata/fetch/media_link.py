"""Mix audio link from a 1001tracklists mix page (SoundCloud, Mixcloud or YouTube embed).

A plain HTTP fetch works once and then gets a challenge page, so this uses the same nodriver browser
as the legacy scraper, headless off. On a server run it under a virtual display:
    xvfb-run -a djdata media-links --config config.yaml
"""

import asyncio
import logging
import re

import nodriver as nd

from ..state import State

log = logging.getLogger("djdata.fetch.media_link")

PATTERNS = [
    re.compile(r"https?://soundcloud\.com/(?!1001tracklists)[\w\-]+/[\w\-]+"),
    re.compile(r"https?://(?:www\.)?mixcloud\.com/[\w\-]+/[\w\-]+"),
    re.compile(r"youtube\.com/embed/([\w\-]{11})"),
    re.compile(r"youtube\.com/watch\?v=([\w\-]{11})"),
]


def links_in(html: str) -> list[str]:
    out = []
    for pat in PATTERNS:
        for m in pat.finditer(html):
            out.append(f"https://www.youtube.com/watch?v={m.group(1)}" if m.groups() else m.group(0))
    return list(dict.fromkeys(out))


async def _fetch_all(state: State, pending: list[dict]) -> int:
    browser = await nd.start(headless=False, no_sandbox=True)
    n = 0
    try:
        for mix in pending:
            page = await browser.get(mix["page_url"])
            await asyncio.sleep(3)
            html = await page.get_content()
            found = links_in(html)
            if found:
                state._conn().execute("UPDATE mixes SET url=?, updated=strftime('%s','now') WHERE mix_id=?",
                                      (found[0], mix["mix_id"]))
                n += 1
                log.info("media link %s → %s", mix["mix_id"], found[0])
            else:
                state.set_mix(mix["mix_id"], "failed", error="no media link on page")
                log.warning("no media link for %s", mix["page_url"])
            await page.close()
    finally:
        browser.stop()
    return n


def fill_media_links(cfg, state: State) -> int:
    """For tracklist-sourced mixes with no audio url yet: read the mix page, store the first link."""
    import pandas as pd

    df = pd.read_csv(cfg.tracklists_csv)[["mix_id", "url"]].drop_duplicates("mix_id")
    page_url = dict(zip(df["mix_id"], df["url"]))
    rows = state._conn().execute("SELECT mix_id FROM mixes WHERE source='1001tracklists' AND url IS NULL").fetchall()
    pending = [{"mix_id": r["mix_id"], "page_url": page_url[r["mix_id"]]} for r in rows if r["mix_id"] in page_url]
    if not pending:
        return 0
    return asyncio.run(_fetch_all(state, pending))
