"""Tests for planner rules that are easy to evade by accident."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.models.predict_model import CURVES, split_artists


@pytest.mark.parametrize(
    "credit,expected",
    [
        ("Reinier Zonneveld & Miro", {"reinier zonneveld", "miro"}),
        ("Odd Mob & OMNOM pres. HYPERBEAM", {"odd mob", "omnom", "hyperbeam"}),
        ("Eric Prydz vs. Pink Floyd", {"eric prydz", "pink floyd"}),
        ("Simon Doty feat. Forrest", {"simon doty", "forrest"}),
        ("Solardo x JOSHWA", {"solardo", "joshwa"}),
        ("Charlotte de Witte", {"charlotte de witte"}),
        ("Mind Against", {"mind against"}),
        ("", set()),
    ],
)
def test_split_artists(credit, expected):
    assert split_artists(credit) == expected


def test_a_collaboration_blocks_the_solo_artist():
    """The bug this guards: comparing whole credit strings let Reinier Zonneveld
    open and close the same three-track set."""
    assert split_artists("Reinier Zonneveld & Miro") & split_artists("Reinier Zonneveld")


def test_unrelated_artists_do_not_collide():
    assert not (split_artists("Mind Against") & split_artists("Charlotte de Witte"))


def test_trailing_period_is_not_left_behind():
    """`pres.` and `ft.` must take their period with them."""
    for credit in ("A pres. B", "A ft. B", "A feat. B", "A vs. B"):
        assert all(not n.startswith(".") for n in split_artists(credit)), credit


# ── Energy curves ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(CURVES))
def test_curves_stay_in_range(name):
    import numpy as np

    t = np.linspace(0, 1, 32)
    v = CURVES[name](t)
    assert v.min() >= -1e-9 and v.max() <= 1 + 1e-9, name


def test_chill_curve_stays_low():
    import numpy as np

    assert CURVES["chill"](np.linspace(0, 1, 32)).max() < 0.3


def test_build_curve_rises_monotonically():
    import numpy as np

    v = CURVES["build"](np.linspace(0, 1, 32))
    assert (np.diff(v) >= 0).all() and v[0] < v[-1]


# ── Slot repair ────────────────────────────────────────────────────────────────


def test_make_mix_repairs_the_slot_and_keeps_the_verified_tracks(monkeypatch, tmp_path):
    """One track cannot be fetched: the other three stay, the empty slot is refilled
    from the model's candidates, and the whole set is never replanned."""
    import scripts.make_mix as mm

    plan = {
        "genre": "techno",
        "curve": "arc",
        "tracks": [
            {
                "n": i + 1,
                "track_id": f"t{i}",
                "artist": f"A{i}",
                "title": f"T{i}",
                "target_energy": 0.3,
                "energy_target01": 0.5,
            }
            for i in range(4)
        ],
    }
    calls = {"plan": 0}

    def fake_plan_mix(*a, **k):
        calls["plan"] += 1
        return json.loads(json.dumps(plan))

    def fake_fetch_plan(plan_path, out_dir):
        p = json.loads(Path(plan_path).read_text())
        return {
            "results": [
                {
                    "track_id": t["track_id"],
                    "status": "no_verified_candidate" if t["track_id"] == "t2" else "ok",
                }
                for t in p["tracks"]
            ]
        }

    def fake_repair(plan_, slot, bpm, excluded, n, cw, mep):
        assert slot == 2 and "t2" in excluded
        return [
            {
                "track_id": "bad",
                "artist": "X",
                "title": "x",
                "n": 3,
                "target_energy": 0.3,
                "energy_target01": 0.5,
            },
            {
                "track_id": "good",
                "artist": "Y",
                "title": "y",
                "n": 3,
                "target_energy": 0.3,
                "energy_target01": 0.5,
                "next_link": {"d_bpm": 1.0, "cam_dist": 1.0},
            },
        ]

    def fake_fetch_track(artist, title, tid, out_dir, preview=None):
        return {"track_id": tid, "status": "ok" if tid == "good" else "no_verified_candidate"}

    rendered = {}
    monkeypatch.setattr(mm, "plan_mix", fake_plan_mix)
    monkeypatch.setattr(mm, "fetch_plan", fake_fetch_plan)
    monkeypatch.setattr(mm, "repair_candidates", fake_repair)
    monkeypatch.setattr(mm, "fetch_track", fake_fetch_track)
    monkeypatch.setattr(mm, "preview_paths", lambda: {})
    monkeypatch.setattr(
        mm,
        "render_plan",
        lambda pp, td, out, **k: rendered.update(plan=json.loads(Path(pp).read_text())) or {},
    )
    mm.make_mix("techno", (120, 130), tmp_path / "m.flac", n_tracks=4, work_dir=tmp_path)
    ids = [t["track_id"] for t in rendered["plan"]["tracks"]]
    assert ids == ["t0", "t1", "good", "t3"]
    assert calls["plan"] == 1, "the set was repaired, not replanned"
    assert rendered["plan"]["tracks"][3]["d_bpm"] == 1.0, (
        "the link into the next track is refreshed"
    )


@pytest.mark.skipif(
    not Path("data/processed/features.parquet").exists(), reason="needs the catalog"
)
def test_repair_candidates_fit_both_neighbours():
    """Real techno pool: the candidates for a middle slot pass the hard rules
    against the previous AND the next track and repeat no track or artist."""
    from src.models.key_rules import sequence_allowed
    from src.models.predict_model import (
        MAX_BPM_LOG_RATIO,
        plan_mix,
        repair_candidates,
        split_artists,
    )

    plan = plan_mix("techno", (126, 136), n_tracks=4, curve="arc", compat_weight=5.0)
    slot = 1
    excluded = {plan["tracks"][slot]["track_id"]}
    cands = repair_candidates(plan, slot, (126, 136), excluded, n=5, compat_weight=5.0)
    assert len(cands) == 5
    used = {t["track_id"] for i, t in enumerate(plan["tracks"]) if i != slot}
    artists = set().union(
        *(split_artists(t["artist"]) for i, t in enumerate(plan["tracks"]) if i != slot)
    )
    prev, nxt = plan["tracks"][slot - 1], plan["tracks"][slot + 1]
    for c in cands:
        assert c["track_id"] not in used and c["track_id"] not in excluded
        assert not (split_artists(c["artist"]) & artists)
        for nb in (prev, nxt):
            assert abs(np.log(c["bpm"] / nb["bpm"])) <= MAX_BPM_LOG_RATIO + 0.01
        # no key veto any more: the repaired sequence must respect the clash cap
        joins = [c["cam_dist"], c["next_link"]["cam_dist"]] + [
            t["cam_dist"] for t in plan["tracks"][slot + 2 :]
        ]
        assert sequence_allowed(joins, len(plan["tracks"]) - 1)
