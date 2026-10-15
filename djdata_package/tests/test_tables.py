"""Tables are append-only CSVs with a fixed header and a key for resume."""

import pytest

from djdata.store import tables


def test_append_writes_header_once_and_keeps_column_order(tmp_path):
    t = tables.Table(tmp_path / "t.csv", ["k", "a", "b"], key="k")
    assert not t.exists()
    t.append([{"k": "x", "a": 1.5, "b": None}])
    t.append([{"k": "y", "b": True, "a": 2}])
    lines = (tmp_path / "t.csv").read_text().splitlines()
    assert lines[0] == "k,a,b"
    assert lines[1] == "x,1.5,"
    assert lines[2] == "y,2,1"
    assert t.keys() == {"x", "y"}


def test_extra_column_is_an_error(tmp_path):
    t = tables.Table(tmp_path / "t.csv", ["k"], key="k")
    with pytest.raises(ValueError, match="not in the table"):
        t.append([{"k": "x", "surprise": 1}])


def test_compound_key_and_drop(tmp_path):
    t = tables.Table(tmp_path / "t.csv", ["m", "t", "v"], key=("m", "t"))
    t.append(
        [
            {"m": "m1", "t": "t1", "v": 1},
            {"m": "m1", "t": "t2", "v": 2},
            {"m": "m2", "t": "t1", "v": 3},
        ]
    )
    assert ("m1", "t2") in t.keys()
    assert t.drop({("m1", "t1"), ("m1", "t2")}) == 2
    assert [r["m"] for r in t.rows()] == ["m2"]


def test_floats_are_written_short_and_read_back_as_numbers(tmp_path):
    t = tables.Table(tmp_path / "t.csv", ["k", "v"], key="k")
    t.append([{"k": "a", "v": 0.1 + 0.2}, {"k": "b", "v": 90.0}, {"k": "c", "v": float("nan")}])
    rows = {r["k"]: r["v"] for r in t.rows()}
    assert rows["a"] == "0.3"
    assert rows["b"] == "90"
    assert rows["c"] == ""
    assert tables.num(rows["a"]) == 0.3
    assert tables.num(rows["b"]) == 90 and isinstance(tables.num(rows["b"]), int)
    assert tables.num(rows["c"]) is None
    assert tables.num("", default=0) == 0


def test_plays_table_has_the_agreed_columns(tmp_path):
    t = tables.plays(tmp_path)
    for c in (
        "mix_id",
        "track_id",
        "is_control",
        "time_zero_s",
        "rate",
        "votes",
        "floor",
        "confidence",
        "first_heard_s",
        "last_heard_s",
        "played_from_s",
        "played_to_s",
    ):
        assert c in t.columns
    assert t.key == ("mix_id", "track_id")
