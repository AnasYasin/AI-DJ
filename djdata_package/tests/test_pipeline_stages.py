"""Every stage after locate, end to end on the synthetic corpus, through the same functions the CLI
calls. Resume on each stage: a second call does nothing."""

import pytest

from djdata import pipeline
from djdata.store import tables
from tests.test_pipeline_locate import corpus  # noqa: F401  the fixture: two mixes, five records


@pytest.fixture(scope="module")
def located(corpus):  # noqa: F811
    pipeline.locate(corpus, workers=2)
    return corpus


def test_pairs_stage(located):
    counts = pipeline.pairs(located)
    assert counts["mixes"] == 2 and counts["seams"] == 2 and counts["usable"] == 2
    rows = {r["seam_id"]: r for r in tables.seams(located.dirs["out"]).rows()}
    assert set(rows) == {"m1_t1_t2", "m2_t4_t5"}
    s = rows["m1_t1_t2"]
    assert s["order_ok"] == "1" and s["third_records"] == ""
    assert tables.num(s["window_t0"]) < 60.0 < tables.num(s["window_t1"])
    assert pipeline.pairs(located)["mixes"] == 0


def test_cut_stage_audits_both_ends(located):
    pipeline.pairs(located)
    counts = pipeline.cut(located, workers=2)
    assert counts["cut"] == 2 and counts["failed"] == 0
    rows = tables.cuts(located.dirs["out"]).rows()
    for r in rows:
        assert r["status"] == "ok"
        assert (
            abs(tables.num(r["cut_error_s"])) < 0.1
        )  # wav packets; mp3 frames land within 0.026 s
        assert (located.dirs["windows"] / r["window_file"]).exists()
        assert r["audit_start_ok"] == "1" and r["audit_end_ok"] == "1"
    assert pipeline.cut(located, workers=1)["cut"] == 0


def test_measure_tempo_label_export(located):
    pipeline.pairs(located)
    pipeline.cut(located, workers=2)
    counts = pipeline.measure(located, workers=2)
    assert counts["measured"] == 2 and counts["failed"] == 0 and counts["both_clear"] == 2
    m = {r["seam_id"]: r for r in tables.measures(located.dirs["out"]).rows()}
    row = m["m1_t1_t2"]
    assert tables.num(row["in_s"]) is not None and tables.num(row["out_s"]) is not None
    assert tables.num(row["a_best_separation"]) > 3 and tables.num(row["b_best_separation"]) > 3
    tempos = pipeline.tempo(located, workers=2)
    assert tempos["measured"] == 4 and tempos["failed"] == 0  # the four records in seams
    labelled = pipeline.label(located)
    assert labelled["labelled"] == 2
    got = {r["seam_id"]: r for r in tables.labels(located.dirs["out"]).rows()}
    assert (
        got["m1_t1_t2"]["measured"] == "1" and tables.num(got["m1_t1_t2"]["swap_pos"]) is not None
    )
    typed = pipeline.types(located)
    assert typed["typed"] == 2
    kinds = {r["seam_id"]: r["signature"] for r in tables.types(located.dirs["out"]).rows()}
    assert kinds["m1_t1_t2"] in ("centre swap", "late swap", "end swap", "cut", "tension", "loop")
    exported = pipeline.export(located)
    assert exported["seams"] == 2
    index = tables.Table(located.dirs["out"] / "seams_index.csv", [], key="seam_id").rows()
    assert {r["seam_id"] for r in index} == {"m1_t1_t2", "m2_t4_t5"}
    assert "signature" in index[0] and "bass_swap_s" in index[0] and "window_file" in index[0]
    assert "low_b_carried_db" in index[0]
    assert pipeline.measure(located, workers=1)["measured"] == 0
    assert pipeline.label(located)["labelled"] == 0
