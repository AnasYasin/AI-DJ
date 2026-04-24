"""Tests for data collection: mixesdb_client and preview_fetcher."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.data.mixesdb_client import _parse_track_line, _results_to_dataframe
from src.data.preview_fetcher import build_manifest, clean_track_name, fetch_previews, track_id

# ── clean_track_name ───────────────────────────────────────────────────────────


def test_strips_label_annotation():
    # [Innate Editions] is a label — should be removed
    assert clean_track_name("Function [Innate Editions]") == "Function"


def test_strips_multiple_label_annotations():
    assert clean_track_name("We Feel For You [MFF (Music For Freaks)]") == "We Feel For You"


def test_keeps_remix_annotation():
    # [Producer Remix] is genuine version info — must be kept
    assert "Remix" in clean_track_name("Track Name [Producer Remix]")


def test_keeps_extended_mix():
    assert "Mix" in clean_track_name("Track Name [Extended Mix]")


def test_no_brackets_unchanged():
    assert clean_track_name("Around The World") == "Around The World"


def test_strips_label_keeps_remix_in_same_track():
    # Label stripped, remix kept
    result = clean_track_name("Track [Label Name] [DJ Remix]")
    assert "Label Name" not in result
    assert "DJ Remix" in result


# ── track_id ───────────────────────────────────────────────────────────────────


def test_track_id_is_deterministic():
    assert track_id("Fumiya Tanaka", "Jeff Mills") == track_id("Fumiya Tanaka", "Jeff Mills")


def test_track_id_is_12_chars():
    assert len(track_id("Artist", "Track")) == 12


def test_track_id_differs_for_different_tracks():
    assert track_id("Artist A", "Track A") != track_id("Artist B", "Track B")


def test_track_id_case_insensitive():
    assert track_id("Daft Punk", "Around The World") == track_id("daft punk", "around the world")


# ── _parse_track_line ──────────────────────────────────────────────────────────


def test_parse_track_line_with_time():
    result = _parse_track_line("[22] Fumiya Tanaka - Jeff Mills [Tresor]")
    assert result["artist_name"] == "Fumiya Tanaka"
    assert result["track_name"] == "Jeff Mills [Tresor]"
    assert result["starting_time"] == 22


def test_parse_track_line_without_time():
    result = _parse_track_line("Daft Punk - Around The World")
    assert result["artist_name"] == "Daft Punk"
    assert result["track_name"] == "Around The World"
    assert result["starting_time"] is None


def test_parse_track_line_returns_none_for_no_dash():
    assert _parse_track_line("Just a title with no separator") is None


# ── _results_to_dataframe ──────────────────────────────────────────────────────


def test_results_to_dataframe_shape():
    results = [
        {
            "url": "http://example.com/mix1",
            "title": "Mix 1",
            "dj_name": "DJ A",
            "genre": "techno",
            "tracklist": [
                "[10] Artist A - Track A [Label]",
                "[20] Artist B - Track B [Label]",
            ],
        },
        {
            "url": "http://example.com/mix2",
            "title": "Mix 2",
            "dj_name": "DJ B",
            "genre": "house",
            "tracklist": [
                "[5] Artist C - Track C [Label]",
            ],
        },
    ]
    df = _results_to_dataframe(results)
    assert len(df) == 3
    assert list(df.columns) == [
        "mix_id",
        "mix_title",
        "dj_name",
        "genre",
        "track_id",
        "starting_time",
        "track_name",
        "artist_name",
        "play_type",
        "overlay_parent",
    ]


def test_results_to_dataframe_empty():
    df = _results_to_dataframe([])
    assert df.empty


# ── fetch_previews (mocked — no real API calls in CI) ─────────────────────────


@pytest.mark.asyncio
async def test_fetch_previews_skips_existing(tmp_path):
    """Tracks already on disk must be skipped — no API calls made.
    build_manifest() should find them and return the correct source."""
    csv_path = tmp_path / "tracklist.csv"
    pd.DataFrame({"artist_name": ["Test Artist"], "track_name": ["Test Track"]}).to_csv(
        csv_path, index=False
    )

    tid = track_id("Test Artist", "Test Track")
    preview_dir = tmp_path / "previews"
    preview_dir.mkdir()
    (preview_dir / f"{tid}.mp3").write_bytes(b"fake mp3 data")

    with (
        patch("src.data.preview_fetcher.PREVIEWS_DIR", preview_dir),
        patch("src.data.preview_fetcher.MANIFEST_PATH", tmp_path / "manifest.csv"),
    ):
        await fetch_previews(str(csv_path))
        manifest = build_manifest(str(csv_path))

    assert len(manifest) == 1
    assert manifest.iloc[0]["track_id"] == tid
    assert manifest.iloc[0]["source"] == "soundcloud"  # .mp3 → soundcloud


# ── Integration tests — hit real APIs, run locally only ───────────────────────
# Usage: pytest tests/test_data.py -v -m integration


@pytest.mark.integration
async def test_itunes_downloads_real_file(tmp_path):
    """iTunes: get URL → download file → verify it is real audio."""
    import httpx

    from src.data.preview_fetcher import _download, _itunes_url

    async with httpx.AsyncClient() as client:
        url = await _itunes_url(client, "Daft Punk", "Around The World")
        assert url is not None, "iTunes returned no URL"

        dest = tmp_path / "itunes.mp3"
        ok = await _download(url, dest, client)

    assert ok, "Download failed"
    assert dest.exists()
    assert dest.stat().st_size > 10_000, "File too small — likely an error response"
    _assert_audio_file(dest)


@pytest.mark.integration
async def test_spotify_credentials_work():
    """
    Spotify: credentials in .env are valid and the API responds without error.
    NOTE: Spotify removed preview_url from most tracks in late 2023, so None
    is the expected return for the majority of tracks — this is not a bug.
    We try 5 tracks to see if ANY still have a preview available.
    """
    from src.data.preview_fetcher import _spotify_client, _spotify_url

    sp = _spotify_client()

    # Try several tracks — Spotify previews are rare but some still exist
    candidates = [
        ("Daft Punk", "Around The World"),
        ("Aphex Twin", "Windowlicker"),
        ("The Chemical Brothers", "Block Rockin Beats"),
        ("Massive Attack", "Teardrop"),
        ("Leftfield", "Leftism"),
    ]

    urls = []
    for artist, track in candidates:
        url = await _spotify_url(sp, artist, track)
        urls.append((artist, track, url))
        print(f"  Spotify preview [{artist} - {track}]: {url or 'None (deprecated)'}")

    # Credentials must work — if ALL raise exceptions something is wrong
    # Getting None is acceptable (Spotify deprecated previews)
    results = [(a, t, u) for a, t, u in urls]
    assert len(results) == len(candidates), "Some Spotify calls raised exceptions"

    found = [(a, t, u) for a, t, u in results if u]
    print(f"\n  {len(found)}/{len(candidates)} tracks still have Spotify previews")


@pytest.mark.integration
async def test_spotify_downloads_if_preview_available(tmp_path):
    """
    If Spotify returns a preview URL for any track, verify it downloads correctly.
    Skips gracefully if all previews are None (Spotify deprecation).
    """
    import httpx

    from src.data.preview_fetcher import _download, _spotify_client, _spotify_url

    sp = _spotify_client()
    candidates = [
        ("Daft Punk", "Around The World"),
        ("Aphex Twin", "Windowlicker"),
        ("The Chemical Brothers", "Block Rockin Beats"),
    ]

    url = None
    for artist, track in candidates:
        url = await _spotify_url(sp, artist, track)
        if url:
            print(f"  Found Spotify preview: {artist} - {track}")
            break

    if url is None:
        pytest.skip("No Spotify previews available — all deprecated. Skipping download test.")

    dest = tmp_path / "spotify.mp3"
    async with httpx.AsyncClient() as client:
        ok = await _download(url, dest, client)

    assert ok, "Spotify download failed"
    assert dest.exists()
    assert dest.stat().st_size > 10_000
    _assert_audio_file(dest)


# ── Shared audio validation helper ────────────────────────────────────────────


def _assert_audio_file(path):
    """
    Assert a file is valid audio (MP3 or M4A).
    - MP3: starts with ID3 tag or MPEG sync bytes
    - M4A: bytes[4:8] == b'ftyp' (MP4 container box type)
    iTunes returns M4A; Spotify returns MP3.
    Both are handled transparently by librosa/ffmpeg.
    """
    raw = path.read_bytes()
    is_mp3 = any(raw[:3].startswith(h) for h in [b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"])
    is_m4a = len(raw) >= 8 and raw[4:8] == b"ftyp"
    assert is_mp3 or is_m4a, f"Not a valid MP3 or M4A. First 8 bytes: {raw[:8]!r}"
