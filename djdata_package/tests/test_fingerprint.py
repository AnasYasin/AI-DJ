"""The fingerprint votes for one offset when the record is there and scatters when it is not."""

import numpy as np
import pytest

from djdata.seam import fingerprint as fp
from tests import synth


@pytest.fixture(scope="module")
def two_records():
    return synth.record(60.0, seed=1), synth.record(60.0, seed=2)


def test_constants_are_the_fetchers():
    from djdata.legacy import track_fetcher as legacy

    assert fp.SR == legacy.VERIFY_SR == 22_050
    assert fp.HOP == legacy.VERIFY_HOP == 256
    assert abs(fp.FRAME_S - 256 / 22_050) < 1e-12


def test_record_matches_itself_at_a_known_frame_shift(two_records):
    a, _ = two_records
    shift_frames = 3000
    padded = np.concatenate([np.zeros(shift_frames * fp.HOP, dtype=np.float32), a])
    votes, offset = fp.match(fp.fingerprint(a), fp.as_arrays(fp.fingerprint(padded)))
    assert offset == shift_frames
    assert votes > 500


def test_wrong_record_scatters(two_records):
    a, b = two_records
    right, _ = fp.match(fp.fingerprint(a), fp.as_arrays(fp.fingerprint(a)))
    wrong, _ = fp.match(fp.fingerprint(b), fp.as_arrays(fp.fingerprint(a)))
    assert right > 20 * wrong


def test_votes_near_counts_only_the_agreeing_offset(two_records):
    a, _ = two_records
    table = fp.as_arrays(fp.fingerprint(a))
    at_zero = fp.votes_near(fp.fingerprint(a), table, offset=0)
    far = fp.votes_near(fp.fingerprint(a), table, offset=5000)
    assert at_zero > 500
    assert far < at_zero / 20


def test_at_mix_speed_changes_length_by_the_rate(two_records):
    a, _ = two_records
    faster = fp.at_mix_speed(a, 1.03)
    assert abs(len(faster) / len(a) - 1 / 1.03) < 0.002
    assert fp.at_mix_speed(a, 1.0) is a


def test_speeded_record_only_matches_at_its_speed(two_records):
    a, _ = two_records
    mix_table = fp.as_arrays(fp.fingerprint(fp.at_mix_speed(a, 1.03)))
    at_speed, _ = fp.match(fp.fingerprint(fp.at_mix_speed(a, 1.03)), mix_table)
    at_one, _ = fp.match(fp.fingerprint(a), mix_table)
    assert at_speed > 5 * at_one


def test_mix_fingerprint_chunks_join_on_one_frame_axis(tmp_path, two_records):
    import soundfile as sf

    a, _ = two_records
    # a mix longer than one chunk, the record starting inside the second chunk
    start_s = fp.CHUNK_S + 12.0
    audio = np.zeros(int((fp.CHUNK_S + 90.0) * fp.SR), dtype=np.float32)
    s = int(start_s * fp.SR)
    audio[s : s + len(a)] = a
    path = tmp_path / "mix.wav"
    sf.write(path, audio, fp.SR)
    table, length = fp.mix_fingerprint(path)
    assert abs(length - (fp.CHUNK_S + 90.0)) < 1.0
    votes, offset = fp.match(fp.fingerprint(a), table)
    assert votes > 500
    assert abs(offset * fp.FRAME_S - start_s) < 0.03
