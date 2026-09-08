"""Numbers-only regression check of the seam set (no audio written).

For every pair in tests/regression/manifest.json that carries an `approved` block, render the pair
with the current mixer to a temp file and compare the seam shift and overlap gains with the values
Anas approved by ear. Tolerances are audibility limits, not measurement noise.

Slow (minutes): opt in with `pytest -m regression tests/test_regression_set.py`.
"""

import json
from pathlib import Path

import pytest

MANIFEST = Path("tests/regression/manifest.json")
SHIFT_TOL_MS = 20.0  # a seam this far from the approved one starts to flam
GAIN_TOL_DB = 1.5  # a level step this size at the seam edge is heard as a step


def _pairs():
    if not MANIFEST.exists():
        return []
    man = json.loads(MANIFEST.read_text())
    return [p for p in man["pairs"] if p.get("approved", {}).get("shift_ms") is not None]


@pytest.mark.regression
@pytest.mark.parametrize("pair", _pairs(), ids=lambda p: p["id"])
def test_pair_matches_approved_numbers(pair, tmp_path):
    from scripts.regression_set import render_pair

    for side in ("A", "B"):
        if not Path(pair[side]["path"]).exists():
            pytest.skip(f"{pair[side]['path']} not cached locally")
    tr, _, _, _ = render_pair(pair, tmp_path)
    app = pair["approved"]
    shift = float(tr["seam_offset_before_ms"])
    assert abs(shift - float(app["shift_ms"])) <= SHIFT_TOL_MS, (
        f"{pair['id']}: seam shift {shift:+.0f} ms, approved {app['shift_ms']:+.0f} ms ({tr['seam_band']})"
    )
    if app.get("gain_db") is not None:
        g_in, g_out = tr["overlap_gain_db"]
        assert abs(g_in - app["gain_db"][0]) <= GAIN_TOL_DB, f"{pair['id']}: gain in {g_in} vs {app['gain_db'][0]}"
        assert abs(g_out - app["gain_db"][1]) <= GAIN_TOL_DB, f"{pair['id']}: gain out {g_out} vs {app['gain_db'][1]}"
    if app.get("bars") is not None:
        assert tr["bars"] == app["bars"], f"{pair['id']}: overlap {tr['bars']} bars, approved {app['bars']}"


def test_manifest_is_well_formed():
    if not MANIFEST.exists():
        pytest.skip("no manifest yet")
    man = json.loads(MANIFEST.read_text())
    seams = [p["seam_at"] for p in man["pairs"]]
    assert all(s % 60 == 0 for s in seams), "every seam must start on a whole minute"
    assert seams == sorted(seams) and len(set(seams)) == len(seams), "slots must be ordered and distinct"
    ids = [p["id"] for p in man["pairs"]]
    assert len(set(ids)) == len(ids)
    for p in man["pairs"]:
        for k in ("edge_case", "reason", "genre", "A", "B"):
            assert k in p, f"{p.get('id')}: missing {k}"
