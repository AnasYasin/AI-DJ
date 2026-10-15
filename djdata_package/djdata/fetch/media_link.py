"""Mix audio link from a 1001tracklists mix page (SoundCloud, Mixcloud or YouTube embed).

A plain HTTP fetch works once and then gets a challenge page, so this uses the same nodriver browser
as the legacy scraper, headless off. On a server run it under a virtual display:
    xvfb-run -a djdata media-links --config config.yaml
"""

import asyncio
import logging
import re
import urllib.parse

import nodriver as nd

from ..state import State

log = logging.getLogger("djdata.fetch.media_link")

FIRST_WAIT_S = 16.0    # Cloudflare has to finish before the page carries its media link
RETRY_WAIT_S = 8.0     # a page with nothing on it gets one more look before it is written off

# In order of preference. The mix audio on a 1001 page is nearly always a SoundCloud player embed
# carrying the track's API id; the plain profile links (soundcloud.com/amelielens) and the player
# script (soundcloud.com/player/api.js) are not audio and must not be mistaken for it.
PATTERNS = [
    re.compile(r"api(?:-v2)?\.soundcloud\.com/tracks/(\d+)"),
    re.compile(r"youtube\.com/embed/([\w\-]{11})"),
    re.compile(r"youtube\.com/watch\?v=([\w\-]{11})"),
    re.compile(r"youtu\.be/([\w\-]{11})"),
    # Only the www host serves shows. Widening this to any subdomain pulled in thumbnailer.mixcloud.com,
    # which is artwork; the player host is handled by MIXCLOUD_WIDGET below instead.
    re.compile(r"https?://(?:www\.)?mixcloud\.com/(?!1001tracklists|widget/)[\w\-]+/[\w\-]+"),
    re.compile(r"https?://soundcloud\.com/(?!1001tracklists/|player/)[\w\-]+/(?!sets/|tracks/)[\w\-]{3,}"),
]
TRACK_ID = {0: "https://api.soundcloud.com/tracks/{}"}
VIDEO_ID = {1: True, 2: True, 3: True}

# A Mixcloud show is embedded from player-widget.mixcloud.com, and the show itself is a percent-encoded
# url in the widget's `feed` parameter. The host defeated the plain mixcloud pattern above, which wanted
# `mixcloud.com` or `www.mixcloud.com`, so a page carrying only a Mixcloud player read as having no audio
# at all (Black Coffee @ Circoloco Radio 001, 2026-09-20). The feed is read first because it names the
# exact show, the way a SoundCloud track id does.
MIXCLOUD_WIDGET = re.compile(r"mixcloud\.com/widget/iframe/\?[^\"'<>\s]+")


def mixcloud_feeds(html: str) -> list[str]:
    """Show urls pulled out of the `feed` parameter of any Mixcloud widget embed."""
    out = []
    for match in MIXCLOUD_WIDGET.finditer(html):
        query = urllib.parse.urlparse("https://" + match.group(0)).query
        for feed in urllib.parse.parse_qs(query).get("feed", []):
            feed = feed.strip()
            if not feed:
                continue
            url = feed if feed.startswith("http") else "https://www.mixcloud.com" + feed
            out.append(url.rstrip("/"))
    return out


def links_in(html: str) -> list[str]:
    """Audio links found on a mix page, best first. A SoundCloud track id beats a YouTube video of the
    set, which beats a bare profile link."""
    out = mixcloud_feeds(html)
    for i, pat in enumerate(PATTERNS):
        for m in pat.finditer(html):
            if i in TRACK_ID:
                out.append(TRACK_ID[i].format(m.group(1)))
            elif i in VIDEO_ID:
                out.append(f"https://www.youtube.com/watch?v={m.group(1)}")
            else:
                out.append(m.group(0))
    return list(dict.fromkeys(out))


async def _fetch_all(state: State, pending: list[dict]) -> int:
    browser = await nd.start(headless=False, no_sandbox=True)
    n = 0
    try:
        for mix in pending:
            # one tab reused for every page: closing the only tab leaves the browser with no target
            # and the next navigation fails ("No target with given id found")
            try:
                page = await browser.get(mix["page_url"])
                # 2026-09-15: 3-4 s returns the challenge page. 2026-09-19: 12 s was still short on many
                # pages (6 links in 19 tries), so the wait is longer and a page that looks empty gets a
                # second look, because Cloudflare sometimes releases it a few seconds late.
                await asyncio.sleep(FIRST_WAIT_S)
                html = await page.get_content()
                if not links_in(html):
                    await asyncio.sleep(RETRY_WAIT_S)
                    html = await page.get_content()
            except Exception as e:  # one bad page must not end the stage
                log.warning("page failed %s: %s", mix["page_url"], e)
                continue
            found = links_in(html)
            if found:
                state._conn().execute("UPDATE mixes SET url=?, updated=strftime('%s','now') WHERE mix_id=?",
                                      (found[0], mix["mix_id"]))
                n += 1
                log.info("media link %s → %s", mix["mix_id"], found[0])
            else:
                state.set_mix(mix["mix_id"], "failed", error="no media link on page")
                log.warning("no media link for %s", mix["page_url"])
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
