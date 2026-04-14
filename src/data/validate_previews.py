"""
Audio preview validator — run before build_features.py.

For every track in preview_manifest.csv that has a local file:
  1. Check the file with soundfile (fast header check, never hangs)
  2. If corrupt → try re-downloading from iTunes
  3. If re-download fails or is still corrupt → delete file, mark as not_found

Updates preview_manifest.csv in place.

Run:
  conda activate djtest
  python src/data/validate_previews.py
"""
import asyncio
import logging
from pathlib import Path

import httpx
import pandas as pd
from mutagen import File as MutaFile

from preview_fetcher import _itunes_url, _download, clean_track_name

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/raw/preview_manifest.csv")


def is_valid(path: Path) -> bool:
    """Return True if mutagen can read the file header and it has audio duration."""
    try:
        f = MutaFile(str(path))
        return f is not None and f.info.length > 0
    except Exception:
        return False


async def validate_and_fix() -> None:
    if not MANIFEST_PATH.exists():
        log.error("Manifest not found: %s", MANIFEST_PATH)
        return

    manifest = pd.read_csv(MANIFEST_PATH)
    total     = len(manifest)
    corrupt   = 0
    fixed     = 0
    removed   = 0

    async with httpx.AsyncClient() as http:
        for idx, row in manifest.iterrows():
            path = Path(str(row["local_path"])) if pd.notna(row.get("local_path")) else None

            if not path or not path.exists():
                continue   # already not_found — skip

            if is_valid(path):
                continue   # file is fine

            corrupt += 1
            artist     = str(row["artist_name"]) if "artist_name" in row else str(row.get("artist", ""))
            track_name = str(row["track_name"])
            log.warning("Corrupt: %s - %s (%s)", artist, track_name, path.name)

            # Delete corrupt file before attempting re-download
            log.warning("  Deleting corrupt file: %s", path)
            path.unlink(missing_ok=True)

            # Try iTunes re-download
            clean = clean_track_name(track_name)
            url   = await _itunes_url(http, artist, clean)

            if url and await _download(url, path, http):
                if is_valid(path):
                    log.info("  ✓ Re-downloaded successfully")
                    manifest.at[idx, "source"] = "itunes"
                    fixed += 1
                    continue
                else:
                    log.warning("  Re-downloaded file is also corrupt — deleting: %s", path)
                    path.unlink(missing_ok=True)

            # Could not fix — mark as not_found
            manifest.at[idx, "source"]     = "not_found"
            manifest.at[idx, "local_path"] = None
            removed += 1
            log.warning("  Marked as not_found")

    manifest.to_csv(MANIFEST_PATH, index=False)
    log.info(
        "Done. %d/%d files checked — %d corrupt, %d re-downloaded, %d removed.",
        total, total, corrupt, fixed, removed,
    )


if __name__ == "__main__":
    asyncio.run(validate_and_fix())
