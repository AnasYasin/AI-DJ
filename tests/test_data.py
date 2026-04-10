"""Tests for data collection: mixesdb_client and preview_fetcher."""
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data.preview_fetcher import track_id, fetch_previews
from src.data.mixesdb_client import _parse_track_line, _results_to_dataframe


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
        {"url": "http://example.com/mix1", "tracklist": [
            "[10] Artist A - Track A [Label]",
            "[20] Artist B - Track B [Label]",
        ]},
        {"url": "http://example.com/mix2", "tracklist": [
            "[5] Artist C - Track C [Label]",
        ]},
    ]
    df = _results_to_dataframe(results)
    assert len(df) == 3
    assert list(df.columns) == ["mix_id", "url", "starting_time", "track_name", "artist_name"]


def test_results_to_dataframe_empty():
    df = _results_to_dataframe([])
    assert df.empty


# ── fetch_previews (mocked — no real API calls in CI) ─────────────────────────

@pytest.mark.asyncio
async def test_fetch_previews_skips_existing(tmp_path):
    """If a preview MP3 already exists it should be marked 'cached' without any API call."""
    csv_path = tmp_path / "tracklist.csv"
    pd.DataFrame({"artist_name": ["Test Artist"], "track_name": ["Test Track"]}).to_csv(csv_path, index=False)

    tid = track_id("Test Artist", "Test Track")
    preview_dir = tmp_path / "previews"
    preview_dir.mkdir()
    (preview_dir / f"{tid}.mp3").write_bytes(b"fake mp3 data")

    with (
        patch("src.data.preview_fetcher.PREVIEWS_DIR", preview_dir),
        patch("src.data.preview_fetcher.MANIFEST_PATH", tmp_path / "manifest.csv"),
        patch("src.data.preview_fetcher._spotify_client", return_value=MagicMock()),
    ):
        manifest = await fetch_previews(str(csv_path))

    assert manifest.iloc[0]["source"] == "cached"
    assert manifest.iloc[0]["track_id"] == tid


# ── Integration tests — hit real APIs, run locally only ───────────────────────
# Usage: pytest tests/test_data.py -v -m integration

@pytest.mark.integration
async def test_itunes_returns_url_for_known_track():
    """iTunes Search API returns a preview URL for a well-known track."""
    import httpx
    from src.data.preview_fetcher import _itunes_url

    async with httpx.AsyncClient() as client:
        url = await _itunes_url(client, "Daft Punk", "Around The World")

    # iTunes has this track — should always return a URL, not None
    assert url is not None, "iTunes returned nothing for Daft Punk - Around The World"
    assert url.startswith("http"), f"Expected http URL, got: {url}"


@pytest.mark.integration
async def test_spotify_returns_url_or_none_for_known_track():
    """
    Spotify credentials work and the function returns without crashing.
    NOTE: Spotify removed preview_url from most tracks in late 2023 — None is valid.
    We test that credentials are correct and the function handles null previews gracefully.
    """
    from src.data.preview_fetcher import _spotify_client, _spotify_url

    sp = _spotify_client()
    url = await _spotify_url(sp, "Daft Punk", "Around The World")

    # None is acceptable — Spotify deprecated previews for most tracks
    # What matters: no exception was raised and the return type is correct
    assert url is None or url.startswith("http"), f"Unexpected value: {url!r}"


@pytest.mark.integration
async def test_jamendo_returns_url_for_known_track():
    """Jamendo API returns a stream URL for a track in its catalog."""
    import httpx
    from src.data.preview_fetcher import _jamendo_url

    async with httpx.AsyncClient() as client:
        # Use a simple genre search — Jamendo has lots of electronic/ambient music
        url = await _jamendo_url(client, "lofi", "chill")

    # Jamendo may or may not match exactly, but should not error
    assert url is None or url.startswith("http"), f"Unexpected value: {url}"


@pytest.mark.integration
async def test_mp3_file_is_downloaded_and_valid(tmp_path):
    """
    End-to-end: fetch a real preview URL from iTunes and download it.
    Checks:
      - File exists on disk
      - File is larger than 10KB (guards against empty/error responses)
      - File starts with MP3 magic bytes (ID3 header or sync bytes)
    """
    import httpx
    from src.data.preview_fetcher import _itunes_url, _download

    async with httpx.AsyncClient() as client:
        url = await _itunes_url(client, "Daft Punk", "Around The World")

    assert url is not None, "Could not get iTunes URL — check internet connection"

    dest = tmp_path / "test_preview.mp3"
    async with httpx.AsyncClient() as client:
        success = await _download(url, dest, client)

    # File was downloaded
    assert success, "Download returned False"
    assert dest.exists(), "MP3 file was not created on disk"

    # File is a real audio file, not an empty or error response
    size = dest.stat().st_size
    assert size > 10_000, f"File too small ({size} bytes) — likely an error response"

    # Check audio format magic bytes.
    # iTunes returns AAC wrapped in an M4A (MP4) container — NOT a raw MP3.
    # Spotify (when available) returns MP3.
    # Both are valid — librosa/ffmpeg handle both formats transparently.
    #
    # MP3 signatures:
    #   b"ID3"       — ID3 metadata header (most common)
    #   b"\xff\xfb"  — MPEG sync bytes
    #   b"\xff\xf3"  — MPEG sync bytes variant
    #   b"\xff\xf2"  — MPEG sync bytes variant
    #
    # M4A/AAC (MP4 container) signature:
    #   bytes[4:8] == b"ftyp"  — MP4 box type at offset 4
    raw = dest.read_bytes()
    is_mp3 = any(raw[:3].startswith(h) for h in [b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"])
    is_m4a = raw[4:8] == b"ftyp"
    assert is_mp3 or is_m4a, (
        f"File is neither MP3 nor M4A. First 8 bytes: {raw[:8]!r}"
    )
