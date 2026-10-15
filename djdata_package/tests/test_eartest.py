"""The ear-test clip puts the mix in the left ear and both records, placed as located, in the right."""

import numpy as np
import soundfile as sf

from djdata.seam import eartest
from djdata.seam.fingerprint import SR
from tests import synth


def test_record_on_window_places_time_zero_and_speed():
    rec = synth.record(20.0, seed=3)
    n = int(30.0 * SR)
    # window starts at mix 100; the record's 0:00 falls at mix 110, played at 1.02
    out = eartest.record_on_window(rec, 110.0, 1.02, 100.0, n)
    assert np.abs(out[: int(9.9 * SR)]).max() == 0.0
    assert np.abs(out[int(10.1 * SR) : int(11 * SR)]).max() > 0.0
    played_len = len(rec) / 1.02 / SR
    assert np.abs(out[int((10.0 + played_len + 0.1) * SR) :]).max() == 0.0
    # a record whose time zero is before the window start is trimmed, not shifted
    out = eartest.record_on_window(rec, 95.0, 1.0, 100.0, n)
    assert np.abs(out[:SR]).max() > 0.0
    assert np.abs(out[int(15.1 * SR) :]).max() == 0.0
    # a record entirely outside the window is silence
    assert np.abs(eartest.record_on_window(rec, 500.0, 1.0, 100.0, n)).max() == 0.0


def test_clip_is_stereo_with_the_window_left_and_records_right(tmp_path):
    a = synth.record(40.0, seed=4)
    b = synth.record(40.0, seed=5)
    window = synth.mix([(a, 0.0, 1.0), (b, 20.0, 1.0)], 50.0)
    for name, audio in (("w", window), ("a", a), ("b", b)):
        sf.write(tmp_path / f"{name}.wav", audio, SR)
    out = tmp_path / "clip.wav"
    length = eartest.clip(
        tmp_path / "w.wav",
        0.0,
        {"path": tmp_path / "a.wav", "time_zero_s": 0.0, "rate": 1.0},
        {"path": tmp_path / "b.wav", "time_zero_s": 20.0, "rate": 1.0},
        out,
    )
    assert length == 50.0
    stereo, sr = sf.read(out)
    assert sr == SR and stereo.shape[1] == 2
    left, right = stereo[:, 0], stereo[:, 1]
    # the right ear is the two records summed, in place: it correlates with the left ear
    corr = np.corrcoef(left[: 40 * SR], right[: 40 * SR])[0, 1]
    assert corr > 0.9
    assert np.abs(stereo).max() <= 0.9 + 1e-6
    marks = eartest.write_marks(tmp_path, [{"n": 1, "file": "clip.wav", "seam_id": "x"}])
    assert marks.exists() and (tmp_path / "README.txt").exists()
    assert marks.read_text().splitlines()[0].startswith("n,file,seam_id")
