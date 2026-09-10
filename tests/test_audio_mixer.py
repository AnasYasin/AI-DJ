"""Tests for the mixer fixes: intent gating, EQ phase, loudness, seam alignment."""

import json

import numpy as np
import pytest

from src.audio.audio_mixer import (
    BPM_TIGHT,
    DEFAULT_BARS,
    DEFAULT_PLAY_MINUTES,
    ENERGY_SLAM,
    GENRE_PLAY_MINUTES,
    LEAD,
    LENGTH_FLEX_BARS,
    LONG_STRETCH,
    LOUD_MELT,
    ONSET_HIGH,
    RECIPES,
    SR,
    STRETCHABLE,
    _analyse_body,
    _apply_recipe,
    _bands,
    _choose_window,
    _env,
    _lead_envelopes,
    _level_db,
    _precise_bpm,
    _sweep_cutoffs,
    _swept_filter,
    _to_mono,
    gate_transition,
    load_audio,
    max_overlap_seconds,
    measure_seam_offset,
    measured_transition,
    normalise_key,
    normalise_onset,
    pair_compatibility,
    render_mix,
)
from src.audio.key_shift import _CAMELOT  # noqa: F401

BPM = 128.0
# A real overlap is bars long, not milliseconds. _env smooths with a 50 ms
# kernel, so an unrealistically short envelope is all smoothing and nothing else.
ENV_N = int(16 * 4 * 60 / BPM * SR)  # 16 bars at 128 bpm


def _kick_pattern(shift_s: float = 0.0, seconds: float = 8.0, bpm: float = BPM) -> np.ndarray:
    """A bare 4/4 kick at `bpm`, offset by `shift_s`."""
    beat = 60.0 / bpm
    y = np.zeros(int(SR * seconds), dtype=np.float32)
    tail = np.arange(1500)
    hit = (np.sin(2 * np.pi * 55 * tail / SR) * np.exp(-tail / SR * 30)).astype(np.float32)
    hit[-300:] *= np.linspace(1.0, 0.0, 300, dtype=np.float32)  # no click at the end of the hit
    for k in range(int(seconds / beat)):
        i = int((k * beat + shift_s) * SR)
        if 0 <= i < len(y) - len(hit):
            y[i : i + len(hit)] += hit
    return y


# ── Issue 1: the set's intent gates the pair rules ─────────────────────────────


def test_chill_set_never_slams():
    """A big measured energy jump must not produce a slam in a chill set."""
    a = {"bpm": 128, "energy": 0.10, "key": "Am", "lufs": -9.0, "onset": 0.2}
    b = {"bpm": 128, "energy": 0.40, "key": "Am", "lufs": -9.0, "onset": 0.2}
    assert measured_transition(a, b) == "slam"
    assert gate_transition("slam", "chill") == "melt"
    assert gate_transition("slam", "chill", 0.9) == "melt"


def test_peak_set_keeps_its_slam():
    assert gate_transition("slam", "peak", 0.8) == "slam"


def test_slam_is_too_early_in_a_build():
    """A build is still climbing at the start, so an early slam is demoted."""
    assert gate_transition("slam", "build", 0.2) == "rise"
    assert gate_transition("slam", "build", 0.9) == "slam"


def test_no_curve_leaves_the_measured_type_alone():
    for t in RECIPES:
        assert gate_transition(t, None, 0.5) == t


# ── Issue 2: audio quality ─────────────────────────────────────────────────────


def test_band_split_reconstructs_the_signal():
    y = _kick_pattern()
    low, mid, high = _bands(y)
    assert np.abs(low + mid + high - y).max() < 1e-9


def test_band_split_is_zero_phase():
    """The low band must not lag the input, or the kick smears against the hats."""
    y = _kick_pattern(seconds=1.0)
    low, _, _ = _bands(y)
    lag_ms = (int(np.argmax(np.abs(low))) - int(np.argmax(np.abs(y)))) / SR * 1000
    assert abs(lag_ms) < 0.5, f"low band lags {lag_ms:.2f} ms"


def test_apply_recipe_returns_float32_of_the_same_length():
    y = _kick_pattern()
    for name, r in RECIPES.items():
        out = _apply_recipe(y, r["A_vol"], r["A_bands"], "out")
        assert out.dtype == np.float32, name
        assert len(out) == len(y), name
        assert np.isfinite(out).all(), name


def test_level_matches_a_known_gain():
    """A 6 dB gain has to read as 6 dB, whichever meter is in use."""
    y = _kick_pattern(seconds=3.0)
    assert _level_db(y * 2.0) - _level_db(y) == pytest.approx(6.02, abs=0.15)


def test_every_recipe_envelope_stays_in_range():
    for name, r in RECIPES.items():
        for spec in (r["A_vol"], r["B_vol"]):
            e = _env(spec, ENV_N, "out")
            assert e.min() >= -0.01 and e.max() <= 1.01, name
        for bands in (r["A_bands"], r["B_bands"]):
            for band, spec in (bands or {}).items():
                if spec is None:
                    continue
                e = _env(spec, ENV_N, "in")
                assert e.min() >= -0.01 and e.max() <= 1.01, f"{name}/{band}"


# ── Issue 5: seam alignment ────────────────────────────────────────────────────


@pytest.mark.parametrize("delay_ms", [-40, -25, -13, -6, 0, 6, 13, 25, 40])
def test_seam_offset_measures_a_known_delay(delay_ms):
    """Positive result means the incoming beat lands late."""
    got = measure_seam_offset(_kick_pattern(0.0), _kick_pattern(delay_ms / 1000), BPM) * 1000
    assert got == pytest.approx(delay_ms, abs=1.5)


def test_seam_offset_is_zero_for_locked_beats():
    y = _kick_pattern()
    assert abs(measure_seam_offset(y, y, BPM)) < 0.001


def test_correcting_the_offset_locks_the_seam():
    """Shifting the head by the measured offset is what render_mix does."""
    tail, head = _kick_pattern(0.0), _kick_pattern(0.019)
    offset = measure_seam_offset(tail, head, BPM)
    shifted = head[int(round(offset * SR)) :]
    n = min(len(tail), len(shifted))
    assert abs(measure_seam_offset(tail[:n], shifted[:n], BPM)) < 0.004


# ── The drop-aligned recipe ────────────────────────────────────────────────────


def test_drop_recipe_cuts_the_incoming_bass_for_the_whole_buildup():
    low = _env(RECIPES["drop"]["B_bands"]["low"], ENV_N, "in")
    assert low.max() < 1e-6, "B must have no low end before the drop"


def test_drop_recipe_opens_the_incoming_filter_across_the_buildup():
    """A real high-pass whose corner falls, not a gain on a fixed band."""
    sweep = RECIPES["drop"]["B_sweep"]
    assert sweep["kind"] == "highpass"
    hz = _sweep_cutoffs(sweep["hz"], ENV_N)
    assert hz[0] > 1000, "starts closed, only the top of the track is through"
    assert hz[-1] < 50, "wide open by the drop"
    assert (np.diff(hz) <= 1e-6).all(), "the corner must fall monotonically"


@pytest.mark.parametrize("ttype", ["rise", "wave", "drop"])
def test_sweeps_are_real_filters_with_a_moving_corner(ttype):
    for side in ("A_sweep", "B_sweep"):
        sweep = RECIPES[ttype].get(side)
        if sweep is None:
            continue
        assert sweep["kind"] in ("highpass", "lowpass")
        hz = _sweep_cutoffs(sweep["hz"], 4096)
        assert hz.min() >= 19.9 and hz.max() <= SR / 2  # log-interp lands on 20 ± eps
        assert hz.max() / hz.min() > 4, f"{ttype}/{side}: corner barely moves"


def test_swept_filter_corner_tracks_the_automation():
    """Measured against white noise: the corner lands where it was asked to."""
    n = SR * 8
    y = (np.random.default_rng(0).standard_normal(n) * 0.1).astype(np.float32)
    cut = _sweep_cutoffs([(0.0, 4000.0), (1.0, 30.0)], n)
    out = _swept_filter(y, cut, "highpass")
    assert out.shape == y.shape
    for frac in (0.2, 0.5, 0.8):
        a = int(frac * n) - SR // 4
        seg_o = out[a : a + SR // 2].astype(np.float32)
        seg_i = y[a : a + SR // 2]
        S = np.abs(np.fft.rfft(seg_o)) + 1e-12
        R = np.abs(np.fft.rfft(seg_i)) + 1e-12
        freqs = np.fft.rfftfreq(len(seg_o), 1 / SR)
        resp = 20 * np.log10(S / R)
        corner = freqs[np.argmax(resp > -6)]
        want = cut[int(frac * n)]
        assert 0.4 * want < corner < 2.5 * want, f"corner {corner:.0f} vs asked {want:.0f}"


def test_a_short_overlap_still_filters():
    """Below the block size the sweep degrades to a fixed cutoff, not a crash."""
    y = np.zeros((1000, 2), dtype=np.float32)
    y[:, 0] = np.random.default_rng(0).standard_normal(1000) * 0.1
    out = _swept_filter(y, _sweep_cutoffs([(0.0, 2000.0), (1.0, 50.0)], 1000), "highpass")
    assert out.shape == y.shape and np.isfinite(out).all()


def test_drop_recipe_hands_the_low_end_over_before_the_seam():
    a_low = _env(RECIPES["drop"]["A_bands"]["low"], ENV_N, "out")
    assert a_low[0] > 0.9, "A holds the bottom while B builds"
    assert a_low[-1] < 0.05, "A has cleared out by the drop"


# ── Segmentation cache ─────────────────────────────────────────────────────────


def test_segment_cache_returns_the_same_result_and_reanalyses_a_changed_file(
    tmp_path, monkeypatch
):
    """Beat tracking and HPSS cost ~20 s per track, and the mixer re-analyses
    the same files on every render."""
    import soundfile as sf

    from src.data import audio_segmenter

    monkeypatch.setattr(audio_segmenter, "CACHE_DIR", tmp_path / "cache")
    calls = []
    real = audio_segmenter._segment_uncached
    monkeypatch.setattr(
        audio_segmenter,
        "_segment_uncached",
        lambda p: (calls.append(p), real(p))[1],
    )

    sr = 22_050
    audio = tmp_path / "t.wav"
    beat = 60.0 / BPM
    y = np.zeros(int(sr * 30), dtype=np.float32)
    hit = np.sin(2 * np.pi * 55 * np.arange(1200) / sr) * np.exp(-np.arange(1200) / sr * 30)
    for k in range(int(30 / beat)):
        i = int(k * beat * sr)
        y[i : i + 1200] += hit.astype(np.float32)
    sf.write(audio, y, sr)

    first = audio_segmenter.segment(audio)
    second = audio_segmenter.segment(audio)
    assert first == second
    assert len(calls) == 1, "second call must come from the cache"

    sf.write(audio, y * 0.5, sr)  # same size, new mtime → new key
    audio_segmenter.segment(audio)
    assert len(calls) == 2, "an edited file must be re-analysed"


# ── Play length: per track, from the genre's measured median ───────────────────


def _fake_info(n_bars=200, bpm=BPM, sections=None, energy=None):
    """Minimal segment() output for window selection."""
    bar = 4 * 60.0 / bpm
    return {
        "bpm": bpm,
        "bars": [i * bar for i in range(n_bars)],
        "beats": [i * bar / 4 for i in range(n_bars * 4)],
        "sections": sections
        if sections is not None
        else [
            {"label": "intro", "bars": [0, 16]},
            {"label": "drop", "bars": [16, 120]},
            {"label": "outro", "bars": [120, n_bars]},
        ],
        "bar_energy": list(energy if energy is not None else np.zeros(n_bars)),
    }


def test_genre_sets_the_play_length():
    """Measured from 24,831 real track changes; dnb is held half as long as afro house."""
    assert GENRE_PLAY_MINUTES["drum and base"] < GENRE_PLAY_MINUTES["techno"]
    assert GENRE_PLAY_MINUTES["techno"] < GENRE_PLAY_MINUTES["afro house"]
    assert 4.0 <= DEFAULT_PLAY_MINUTES <= 4.3


def _boundary_info():
    return _fake_info(
        n_bars=200,
        sections=[
            {"label": "intro", "bars": [0, 8]},
            {"label": "drop", "bars": [8, 72]},
            {"label": "outro", "bars": [72, 200]},
        ],
        energy=np.zeros(200),
    )


def test_window_lands_on_section_boundaries():
    """Ending a record mid-phrase is what makes a set sound arbitrary."""
    start, end = _choose_window(_boundary_info(), target_bars=64, tail_bars=16, e_target01=None)
    assert (start, end) == (8, 72), "should sit exactly on the section it was given"


def test_window_stretches_a_little_to_catch_both_boundaries():
    """Target 56, section runs 8-72. Stretching 8 bars buys the cue-in too."""
    start, end = _choose_window(_boundary_info(), target_bars=56, tail_bars=16, e_target01=None)
    assert (start, end) == (8, 72)
    assert end - start == 64, "length flexed by +8 bars to sit on the whole section"


def test_window_keeps_the_target_length_when_the_stretch_would_be_large():
    """Target 48 against a 64-bar section. A 33% stretch is not worth one boundary,
    so it slides to hit the cue-out at the right length instead."""
    start, end = _choose_window(_boundary_info(), target_bars=48, tail_bars=16, e_target01=None)
    assert end == 72, "cue-out still lands on the section change"
    assert end - start == 48, "length stays on target"


def test_window_will_not_stretch_beyond_the_flex_limit():
    """A boundary further away than LENGTH_FLEX_BARS must not drag the window."""
    start, end = _choose_window(_boundary_info(), target_bars=24, tail_bars=16, e_target01=None)
    assert end - start <= 24 + LENGTH_FLEX_BARS


def test_window_length_stays_near_the_genre_target():
    info = _fake_info(n_bars=300, energy=np.zeros(300))
    for target in (48, 64, 96):
        start, end = _choose_window(info, target_bars=target, tail_bars=16, e_target01=None)
        assert abs((end - start) - target) <= LENGTH_FLEX_BARS


def test_short_track_plays_what_it_has():
    info = _fake_info(n_bars=40, energy=np.zeros(40))
    start, end = _choose_window(info, target_bars=128, tail_bars=16, e_target01=None)
    assert 0 <= start < end <= 40


def test_energy_target_steers_which_window_is_played():
    """Low target avoids the loud half, high target seeks it."""
    energy = np.concatenate([np.zeros(100), np.ones(100) * 3.0])
    info = _fake_info(
        n_bars=200,
        sections=[{"label": "drop", "bars": [0, 200]}],
        energy=energy,
    )
    lo_start, lo_end = _choose_window(info, 48, 16, e_target01=0.0)
    hi_start, hi_end = _choose_window(info, 48, 16, e_target01=1.0)
    assert np.mean(energy[lo_start:lo_end]) < np.mean(energy[hi_start:hi_end])


# ── Energy targets must be raw curve values ────────────────────────────────────


def test_absolute_track_energies_are_rejected(tmp_path):
    """The exact bug this guards: the plan's `target_energy` is not 0-1."""
    with pytest.raises(ValueError, match="energy_target01"):
        render_mix(["a.mp3", "b.mp3"], tmp_path / "out.flac", energy_targets=[0.18, 1.4])


def test_energy_target_count_must_match_the_tracks(tmp_path):
    with pytest.raises(ValueError, match="energy_targets has"):
        render_mix(["a.mp3", "b.mp3"], tmp_path / "out.flac", energy_targets=[0.5])


def test_render_plan_passes_the_curve_values_through(tmp_path, monkeypatch):
    from src.audio import audio_mixer

    plan = {
        "genre": "techno",
        "curve": "arc",
        "tracks": [
            {"n": 1, "track_id": "aaa", "artist": "A", "title": "x", "energy_target01": 0.0},
            {"n": 2, "track_id": "bbb", "artist": "B", "title": "y", "energy_target01": 1.0},
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "aaa.m4a").touch()
    (tracks / "bbb.m4a").touch()

    seen = {}
    monkeypatch.setattr(audio_mixer, "render_mix", lambda *a, **k: seen.update(k) or {"ok": 1})
    audio_mixer.render_plan(plan_path, tracks, tmp_path / "out.flac")

    assert seen["energy_targets"] == [0.0, 1.0]
    assert seen["curve"] == "arc"
    assert seen["genre"] == "techno"


def test_render_plan_drops_tracks_with_no_audio(tmp_path, monkeypatch):
    from src.audio import audio_mixer

    plan = {
        "genre": "techno",
        "curve": "build",
        "tracks": [
            {"n": i, "track_id": t, "artist": "A", "title": "x", "energy_target01": 0.5}
            for i, t in enumerate(["aaa", "bbb", "ccc"])
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    (tracks / "aaa.m4a").touch()
    (tracks / "ccc.m4a").touch()  # bbb never fetched

    seen = {}
    monkeypatch.setattr(
        audio_mixer, "render_mix", lambda paths, *a, **k: seen.update(paths=paths, **k) or {}
    )
    audio_mixer.render_plan(plan_path, tracks, tmp_path / "out.flac")
    assert [p.stem for p in seen["paths"]] == ["aaa", "ccc"]
    assert len(seen["energy_targets"]) == 2


def test_render_plan_needs_two_playable_tracks(tmp_path):
    from src.audio import audio_mixer

    plan = {
        "genre": "techno",
        "curve": "build",
        "tracks": [
            {"n": 1, "track_id": "aaa", "artist": "A", "title": "x", "energy_target01": 0.5}
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    tracks = tmp_path / "tracks"
    tracks.mkdir()
    with pytest.raises(ValueError, match="have audio"):
        audio_mixer.render_plan(plan_path, tracks, tmp_path / "out.flac")


def test_tail_reserve_does_not_eat_the_play_window():
    """Reserving the longest transition on every track capped a 137-bar track at
    half the genre's play length."""
    reserve = int(np.median(list(DEFAULT_BARS.values())))
    assert reserve < max(DEFAULT_BARS.values())
    n_bars = 137
    assert n_bars - 1 - reserve >= 112, "a 137-bar track must still fit a real window"


# ── Stereo path ────────────────────────────────────────────────────────────────
# Rendering in mono discarded the side channel of every record, which measured
# −10 to −17 dB relative to mid on the real sources. These pin the audio path
# open at two channels while the analysis stays mono.


def _stereo_kicks(seconds=4.0):
    """A kick centred, plus a hat only in the right channel."""
    left = _kick_pattern(seconds=seconds)
    right = left.copy()
    hat = (np.random.default_rng(0).standard_normal(600) * 0.2).astype(np.float32)
    for i in range(0, len(right) - len(hat), int(SR * 0.25)):
        right[i : i + len(hat)] += hat
    return np.stack([left, right], axis=1)


def test_load_audio_returns_samples_by_channels(tmp_path):
    import soundfile as sf

    path = tmp_path / "s.wav"
    sf.write(path, _stereo_kicks(), SR)
    y = load_audio(path)
    assert y.ndim == 2 and y.shape[1] == 2
    assert y.shape[0] > y.shape[1], "time must be axis 0, so slicing indexes samples"


def test_a_mono_file_is_widened_to_two_channels(tmp_path):
    """Tracks are concatenated, so every track needs the same channel count."""
    import soundfile as sf

    path = tmp_path / "m.wav"
    sf.write(path, _kick_pattern(), SR)
    y = load_audio(path)
    assert y.shape[1] == 2
    assert np.allclose(y[:, 0], y[:, 1])


def test_band_split_reconstructs_each_channel():
    y = _stereo_kicks().astype(np.float64)
    low, mid, high = _bands(y)
    assert low.shape == y.shape
    assert np.abs(low + mid + high - y).max() < 1e-9


def test_band_split_does_not_mix_the_channels():
    """A signal only in the right channel must stay only in the right channel."""
    y = np.zeros((SR, 2))
    y[:, 1] = _kick_pattern(seconds=1.0)
    low, mid, high = _bands(y)
    for band in (low, mid, high):
        assert np.abs(band[:, 0]).max() < 1e-9


def test_apply_recipe_keeps_both_channels():
    y = _stereo_kicks()
    for name, r in RECIPES.items():
        out = _apply_recipe(y, r["B_vol"], r["B_bands"], "in")
        assert out.shape == y.shape, name
        assert out.dtype == np.float32, name
        assert np.isfinite(out).all(), name


def test_apply_recipe_preserves_stereo_difference():
    """The envelope must scale both channels, not collapse them."""
    y = _stereo_kicks()
    out = _apply_recipe(y, RECIPES["blend"]["B_vol"], RECIPES["blend"]["B_bands"], "in")
    assert np.abs(out[:, 0] - out[:, 1]).max() > 1e-4, "channels became identical"


def test_analysis_still_runs_on_the_mono_sum():
    """Timing questions have one answer, not one per channel."""
    stereo = _stereo_kicks(seconds=8.0)
    mono = _to_mono(stereo)
    assert mono.ndim == 1
    offset_stereo = measure_seam_offset(stereo, stereo, BPM)
    assert abs(offset_stereo) < 0.001


def test_seam_offset_accepts_stereo_and_agrees_with_mono():
    tail = _stereo_kicks(seconds=8.0)
    head = np.stack([_kick_pattern(0.013, seconds=8.0)] * 2, axis=1)
    stereo_offset = measure_seam_offset(tail, head, BPM) * 1000
    mono_offset = measure_seam_offset(_to_mono(tail), _to_mono(head), BPM) * 1000
    assert stereo_offset == pytest.approx(mono_offset, abs=0.5)
    assert stereo_offset == pytest.approx(13, abs=1.5)


def test_loudness_reads_stereo():
    y = _stereo_kicks(seconds=3.0)
    assert _level_db(y * 2.0) - _level_db(y) == pytest.approx(6.02, abs=0.15)


# ── One record leads at a time ─────────────────────────────────────────────────
# Sitting both records at the same level for a minute gave the ear nothing to
# hold onto. These pin the lead/support structure open.


def _levels(spec_a, spec_b, n_bars, bpm=134.0):
    secs = n_bars * 4 * 60 / bpm
    n = int(secs * SR)
    a, b = _env(spec_a, n, "out"), _env(spec_b, n, "in")
    a_db = 20 * np.log10(np.maximum(a, 1e-6))
    b_db = 20 * np.log10(np.maximum(b, 1e-6))
    return a, b, a_db, b_db, secs


def _ambiguous_seconds(spec_a, spec_b, n_bars):
    """Seconds where both records are audible and within 3 dB of each other."""
    a, b, a_db, b_db, secs = _levels(spec_a, spec_b, n_bars)
    return ((a > 0.05) & (b > 0.05) & (np.abs(a_db - b_db) < 3.0)).mean() * secs


@pytest.mark.parametrize("ttype", sorted(LEAD))
def test_the_ambiguous_zone_is_short(ttype):
    """Whatever the overlap length, only a few seconds have no clear lead."""
    n_bars = DEFAULT_BARS[ttype]
    a, b, _, _ = _lead_envelopes(n_bars, LEAD[ttype])
    assert _ambiguous_seconds(a, b, n_bars) < 12


def test_a_long_overlap_is_not_a_long_crossfade():
    """A 64-bar melt is 115 s. The handover must stay about as short as a
    16-bar blend's, not scale up with the overlap."""
    melt = _ambiguous_seconds(*_lead_envelopes(64, LEAD["melt"])[:2], 64)
    blend = _ambiguous_seconds(*_lead_envelopes(16, LEAD["blend"])[:2], 16)
    assert abs(melt - blend) < 6, f"melt {melt:.0f}s vs blend {blend:.0f}s"


def test_the_incoming_record_sits_under_the_outgoing_one_first():
    a_vol, b_vol, _, _ = _lead_envelopes(64, LEAD["melt"])
    a, b, a_db, b_db, secs = _levels(a_vol, b_vol, 64)
    i = int(0.4 * len(a))  # well inside the bed, before the handover
    assert a_db[i] > b_db[i] + 5, "A must clearly lead during the bed"
    assert b[i] > 0.2, "B must still be audible, not silent"


def test_the_lead_changes_hands_by_the_end():
    for ttype, cfg in LEAD.items():
        a_vol, b_vol, _, _ = _lead_envelopes(DEFAULT_BARS[ttype], cfg)
        a, b, a_db, b_db, _ = _levels(a_vol, b_vol, DEFAULT_BARS[ttype])
        assert a_db[0] > b_db[0], f"{ttype}: A leads at the start"
        assert b_db[-1] > a_db[-1], f"{ttype}: B leads at the end"


def test_the_supporting_record_gives_up_midrange():
    """Two records fighting over the same band is what makes an overlap muddy."""
    _, _, a_mid, b_mid = _lead_envelopes(64, LEAD["melt"])
    n = 200_000
    a, b = _env(a_mid, n, "out"), _env(b_mid, n, "in")
    assert b[0] < 0.8, "B's mids are cut while it is supporting"
    assert a[-1] < 0.8, "A's mids are cut once it is supporting"
    assert a[0] > 0.95 and b[-1] > 0.95, "the lead keeps its mids"


def test_a_very_short_overlap_still_produces_valid_envelopes():
    """The overlap clamp can cut a transition to a couple of bars."""
    for n_bars in (2, 3, 4, 6):
        specs = _lead_envelopes(n_bars, LEAD["melt"])
        for spec in specs:
            ts = [t for t, _ in spec]
            assert ts == sorted(ts), f"{n_bars} bars: breakpoints out of order"
            assert 0.0 <= min(ts) and max(ts) <= 1.0
            e = _env(spec, 4096, "in")
            assert np.isfinite(e).all() and e.min() >= -0.01 and e.max() <= 1.01


# ── Long overlaps are earned, not assumed ──────────────────────────────────────
# A 64-bar melt over an ill-matched pair is two songs at once. Length is a
# property of the pair, not of the transition type.


def _track(key="Am", energy=0.25, lufs=-9.0, bpm=128.0, onset=0.3):
    return {"key": key, "energy": energy, "lufs": lufs, "bpm": bpm, "onset": onset}


def test_an_identical_pair_scores_at_the_top():
    t = _track()
    assert pair_compatibility(t, t) == pytest.approx(1.0, abs=1e-6)


def test_a_mismatched_pair_scores_low():
    a = _track(key="Am", energy=0.15, lufs=-14.0, bpm=124.0)
    b = _track(key="C#", energy=0.45, lufs=-6.0, bpm=136.0)
    assert pair_compatibility(a, b) < 0.2


def test_each_axis_moves_the_score_on_its_own():
    base = _track()
    for changed in (
        _track(key="F#"),  # harmonic distance
        _track(energy=0.45),  # density
        _track(lufs=-15.0),  # loudness
        _track(bpm=136.0),  # how far it must be stretched
    ):
        assert pair_compatibility(base, changed) < pair_compatibility(base, base)


def test_only_a_well_matched_pair_earns_a_long_overlap():
    tight = pair_compatibility(_track(), _track(key="Em", energy=0.27))
    loose = pair_compatibility(_track(), _track(key="C#", energy=0.45, bpm=135.0))
    assert max_overlap_seconds(tight) >= 60
    assert max_overlap_seconds(loose) <= 32


def test_the_ceiling_never_increases_as_the_pair_gets_worse():
    seconds = [max_overlap_seconds(s) for s in np.linspace(1.0, 0.0, 40)]
    assert all(b <= a for a, b in zip(seconds, seconds[1:]))


def test_thresholds_match_the_calibrated_tiers():
    """Set from the 40th, 75th and 90th percentiles of 43,073 real pairs."""
    assert max_overlap_seconds(0.75) == 90.0
    assert max_overlap_seconds(0.60) == 60.0
    assert max_overlap_seconds(0.40) == 32.0
    assert max_overlap_seconds(0.10) == 16.0


# ── Onset lives on one scale ───────────────────────────────────────────────────
# build_features stores librosa's RAW mean onset_strength (about 1.1 to 2.5
# across the catalog). Every threshold is written against raw/ONSET_SCALE. The
# labeler used to compare the raw column against 0.35, which 100% of tracks
# exceed, so the `wave` rule collapsed and 28% of pairs were labelled wave.


def test_normalise_onset_maps_the_catalog_range_onto_the_threshold():
    """Catalog median is 1.71 raw; the threshold sits at 0.35."""
    assert normalise_onset(1.71) == pytest.approx(0.342, abs=0.002)
    assert normalise_onset(1.12) < ONSET_HIGH < normalise_onset(2.54)


def test_normalise_onset_clips_at_one():
    assert normalise_onset(5.0) == pytest.approx(1.0)
    assert normalise_onset(12.0) == pytest.approx(1.0)
    assert normalise_onset(0.0) == pytest.approx(0.0)


def test_raw_onset_would_trivially_pass_the_threshold():
    """The bug: every raw catalog value clears 0.35, so the test meant nothing."""
    for raw in (1.12, 1.71, 2.54):
        assert raw > ONSET_HIGH
        assert normalise_onset(raw) < 1.0


def test_wave_needs_both_tracks_to_be_punchy():
    """Same pair, only the onsets differ."""

    def pair(onset):
        a = {"bpm": 128.0, "energy": 0.30, "key": "Am", "lufs": -9.0, "onset": onset}
        b = {"bpm": 128.0, "energy": 0.31, "key": "F", "lufs": -9.0, "onset": onset}
        return measured_transition(a, b)

    assert pair(normalise_onset(2.6)) == "wave", "two busy tracks ride together"
    assert pair(normalise_onset(1.0)) == "blend", "two mellow tracks just blend"


def test_the_mixer_and_the_labeler_share_one_threshold():
    """They were duplicated, which is how the scales drifted apart."""
    from src.features import transition_labeler as lab

    assert ONSET_HIGH is lab.ONSET_HIGH_MIN
    assert BPM_TIGHT is lab.BPM_RATIO_TIGHT
    assert ENERGY_SLAM is lab.ENERGY_SLAM_MIN
    assert LOUD_MELT is lab.LOUD_MELT_MAX


def test_analyse_body_reports_onset_on_the_normalised_scale():
    y = _kick_pattern(seconds=3.0)
    m = _analyse_body(np.stack([y, y], axis=1))
    assert 0.0 <= m["onset"] <= 1.0


# ── Key names: essentia says Eb, the catalog and the Camelot table say D# ───────


def test_flat_key_names_are_normalised_to_the_catalog_spelling():
    assert normalise_key("Eb", "minor") == "D#m"
    assert normalise_key("Bb", "major") == "A#"
    assert normalise_key("Ab", "minor") == "G#m"
    assert normalise_key("C", "minor") == "Cm"
    assert normalise_key("F#", "major") == "F#"


def test_every_normalised_key_is_on_the_camelot_wheel():
    """An unknown key gets distance 2.5, which blocks melt and rise for the pair."""
    for note in ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]:
        for scale in ("major", "minor"):
            assert normalise_key(note, scale) in _CAMELOT


# ── Tempo: measured from the audio, not the frame-quantised beat grid ──────────


def _quantised_grid(bpm: float, seconds: float, hop_s: float = 512 / 22_050) -> dict:
    """What the segmenter hands the mixer: beat times rounded to its 23 ms frame."""
    beats = np.arange(0, seconds, 60.0 / bpm)
    beats = np.round(beats / hop_s) * hop_s
    return {"bpm": float(round(bpm)), "beats": [round(float(b), 3) for b in beats]}


def test_precise_bpm_recovers_a_non_integer_tempo_from_the_audio():
    """127.4 BPM lies between the grid's quantised values; the audio still gives it."""
    y = _kick_pattern(seconds=60.0, bpm=127.4)
    info = _quantised_grid(127.4, 60.0)
    assert abs(_precise_bpm(info, np.stack([y, y], axis=1)) - 127.4) < 0.05


def test_the_beat_grid_alone_snaps_to_a_quantised_tempo():
    """Documents the failure the audio measurement replaces: 134 BPM reads as 136."""
    info = _quantised_grid(134.0, 240.0)
    grid_only = _precise_bpm(info)
    assert abs(grid_only - 134.0) > 1.0, "the grid median cannot resolve 134 BPM"
    y = _kick_pattern(seconds=60.0, bpm=134.0)
    assert abs(_precise_bpm(info, np.stack([y, y], axis=1)) - 134.0) < 0.05


def test_precise_bpm_falls_back_to_the_grid_when_the_audio_has_no_beat():
    info = _quantised_grid(128.0, 60.0)
    silence = np.zeros((SR * 60, 2), dtype=np.float32)
    assert _precise_bpm(info, silence) == pytest.approx(_precise_bpm(info))


def test_two_tracks_stretched_to_one_tempo_do_not_drift():
    """
    The bug as heard: two tracks at 130.00 and 127.99 both measured 129.31 from
    their grids, were stretched by the same rate, and drifted half a beat apart
    inside one overlap. Measured from the audio, both land on the target.
    """
    target = 128.5
    for true_bpm in (130.0, 127.99):
        y = _kick_pattern(seconds=60.0, bpm=true_bpm)
        est = _precise_bpm(_quantised_grid(true_bpm, 60.0), np.stack([y, y], axis=1))
        actual_after_stretch = true_bpm * (target / est)
        drift_per_64_bars_ms = (
            abs(actual_after_stretch - target) / target * 64 * 4 * 60 / target * 1000
        )
        assert drift_per_64_bars_ms < 10, f"{true_bpm}: {drift_per_64_bars_ms:.0f} ms drift"


# ── A pair that fits may run past its type's default ───────────────────────────


def test_stretchable_types_exclude_the_cut_and_the_drop():
    """A slam is a cut on a bar line; a drop is pinned to the incoming downbeat."""
    assert "slam" not in STRETCHABLE
    assert "drop" not in STRETCHABLE
    assert {"blend", "melt", "rise", "fade", "wave"} <= STRETCHABLE


def test_a_great_pair_earns_more_than_the_type_default():
    """16-bar blend, 90 s ceiling at 136 bpm ≈ 51 bars, so it stretches to 32."""
    bar_dur = 4 * 60 / 136
    ceiling = max(int(max_overlap_seconds(0.80) / bar_dur), 2)
    assert min(ceiling, int(16 * LONG_STRETCH)) == 32


def test_a_poor_pair_is_cut_below_the_type_default():
    bar_dur = 4 * 60 / 136
    ceiling = max(int(max_overlap_seconds(0.20) / bar_dur), 2)
    assert min(ceiling, int(64 * LONG_STRETCH)) < 64


def test_stretching_never_exceeds_twice_the_default():
    bar_dur = 4 * 60 / 136
    ceiling = max(int(max_overlap_seconds(1.0) / bar_dur), 2)
    assert min(ceiling, int(16 * LONG_STRETCH)) == 32, "a blend must not become a melt"


# ── Downbeats and phrases: every move lands on the "1" of a phrase ─────────────


def test_tracker_grid_is_built_from_the_clean_run_of_downbeats(monkeypatch):
    """
    What the raw tracker really returns: a burst of "downbeats" on every beat
    in the intro, one missed downbeat, one beat slip in the beat list. The grid
    must still be regular 4-beat bars on the phase of the clean run.
    """
    from src.data import audio_segmenter

    beats = np.arange(0, 60, 0.5)  # 120 bpm, bar = 2 s
    true_bars = beats[2::4]  # downbeats on beat index 2, 6, 10, ...
    downbeats = np.concatenate([beats[:10], true_bars[3:12], true_bars[13:]])  # burst, gap
    beats_with_slip = np.delete(beats, 60)  # tracker dropped one beat at 30 s
    monkeypatch.setattr(audio_segmenter, "BEAT_THIS_ENABLED", True)
    monkeypatch.setattr(
        audio_segmenter._tracked_grid,
        "_model",
        lambda y, sr: (beats_with_slip, downbeats),
        raising=False,
    )
    got_beats, bars, conf = audio_segmenter._tracked_grid(np.zeros(10), 22_050)
    assert np.allclose(bars, true_bars[: len(bars)], atol=0.05), "phase from the clean run"
    assert np.allclose(np.diff(bars), 2.0, atol=0.05), "every bar is one bar long"
    inner = got_beats[(got_beats >= bars[0]) & (got_beats < bars[-1])]
    assert len(inner) == 4 * (len(bars) - 1), "beats are four subdivisions of each bar"
    assert np.allclose(np.diff(inner), 0.5, atol=0.01)
    assert 0.6 < conf < 0.9, "the intro burst is counted against confidence"


def test_tracker_grid_needs_a_clean_run():
    from src.data.audio_segmenter import regular_bars

    beats = np.arange(0, 20, 0.5)
    assert regular_bars(beats, beats[::3]) is None, "3-beat gaps are not bars"


def test_phrase_offset_is_where_the_section_changes_land():
    """Changes at bars 3, 11, 19, 35 all sit on residue 3 of an 8-bar grid."""
    from src.data.audio_segmenter import phrase_offset

    novelty = np.zeros(48)
    peaks = [3, 11, 19, 35]
    for p in peaks:
        novelty[p] = 1.0
    novelty[6] = 0.4  # a weaker change off the grid must not win
    peaks.append(6)
    assert phrase_offset(peaks, novelty, 8) == 3
    assert phrase_offset([], novelty, 8) == 0


def test_window_lands_on_the_tracks_own_phrase_grid():
    """A track whose phrases start at bar 3 must cue in and out on 3, 11, 19, ..."""
    from src.audio.audio_mixer import _choose_window

    info = _fake_info(
        n_bars=200,
        sections=[
            {"label": "intro", "bars": [0, 11]},
            {"label": "drop", "bars": [11, 75]},
            {"label": "outro", "bars": [75, 200]},
        ],
        energy=np.zeros(200),
    )
    info["phrase_bars"], info["phrase_offset"] = 8, 3
    start, end = _choose_window(info, target_bars=64, tail_bars=16, e_target01=None)
    assert (start - 3) % 8 == 0 and (end - 3) % 8 == 0, (start, end)
    assert (start, end) == (11, 75)


def test_overlaps_are_whole_phrases():
    """45 and 50 bars, what the compatibility ceiling produced, end mid-phrase."""
    from src.audio.audio_mixer import _whole_phrases

    assert _whole_phrases(45, 8) == 40
    assert _whole_phrases(50, 8) == 48
    assert _whole_phrases(16, 8) == 16
    assert _whole_phrases(6, 8) == 6, "below one phrase, whole bars are all there is"
    assert _whole_phrases(9, 4) == 8, "dnb phrases are 4 half-time bars"


def test_bass_swap_starts_on_a_bar_line_and_is_one_decisive_move():
    """(0.46, 1) → (0.54, 0) on 45 bars is a 3.6-bar swap starting at bar 20.7."""
    from src.audio.audio_mixer import SWAP_BARS, _snap_low_swap

    spec = _snap_low_swap([(0.0, 1), (0.46, 1), (0.54, 0)], 45, 4)
    (t0, g0), (t1, g1) = spec[1], spec[2]
    start_bar, end_bar = t0 * 45, t1 * 45
    assert start_bar == pytest.approx(24) and start_bar % 4 == 0
    assert end_bar - start_bar == pytest.approx(SWAP_BARS)
    assert (g0, g1) == (1, 0)


def test_bass_swap_snapping_mirrors_for_the_incoming_record():
    from src.audio.audio_mixer import _snap_low_swap

    a = _snap_low_swap([(0.0, 1), (0.46, 1), (0.54, 0)], 32, 4)
    b = _snap_low_swap([(0.0, 0), (0.46, 0), (0.54, 1)], 32, 4)
    assert [t for t, _ in a] == [t for t, _ in b], "A leaves and B arrives on the same bar"


def test_specs_without_a_single_swap_are_left_alone():
    from src.audio.audio_mixer import _snap_low_swap

    assert _snap_low_swap(None, 32, 4) is None
    flat = [(0.0, 0), (1.0, 0)]
    assert _snap_low_swap(flat, 32, 4) == flat


def test_every_recipe_still_swaps_bass_after_snapping():
    """Snapping must keep the one-bass rule: A's low leaves when B's arrives."""
    from src.audio.audio_mixer import _snap_bands

    for name, r in RECIPES.items():
        for n_bars in (8, 16, 32, 45, 64):
            a = _snap_bands(r["A_bands"], n_bars, 4)
            b = _snap_bands(r["B_bands"], n_bars, 4)
            if not a or not b or a.get("low") is None or b.get("low") is None:
                continue
            ea = _env(a["low"], ENV_N, "out")
            eb = _env(b["low"], ENV_N, "in")
            assert (ea + eb).max() < 1.6, f"{name}/{n_bars}: both basses in together"
            assert ea[0] == pytest.approx(1, abs=0.02), f"{name}: A must start with its bass"
            if len(b["low"]) == 4:  # a snapped swap; the drop keeps B's bass cut throughout
                assert eb[-1] == pytest.approx(1, abs=0.02), f"{name}: B must end with its bass"


def test_lead_handover_starts_on_the_half_phrase_grid():
    for ttype, cfg in LEAD.items():
        for n_bars in (16, 32, 45, 64):
            a_vol, _, _, _ = _lead_envelopes(n_bars, cfg, 4)
            h0_bars = a_vol[1][0] * n_bars
            assert h0_bars == pytest.approx(round(h0_bars / 4) * 4, abs=1e-6), (ttype, n_bars)


# ── Sound quality: stretch engine and stretch cap ──────────────────────────────


def test_stretch_uses_the_r3_engine(monkeypatch):
    """R3 keeps kick transients; measured sharper than R2 on real audio."""
    import pyrubberband

    from src.audio import audio_mixer

    seen = {}

    def fake(y, sr, rate, rbargs=None):
        seen["rbargs"] = rbargs
        return y

    monkeypatch.setattr(pyrubberband, "time_stretch", fake)
    monkeypatch.setattr(audio_mixer.shutil, "which", lambda name: "/usr/bin/rubberband")
    audio_mixer._stretch(np.zeros((SR, 2), dtype=np.float32), 0.97)
    assert "-3" in seen["rbargs"], "the R3 flag must reach the rubberband CLI"


def test_stretch_cap_warns_then_refuses(caplog):
    from src.audio.audio_mixer import STRETCH_MAX, STRETCH_WARN, _check_stretch_rate

    _check_stretch_rate(1.03, "ok")  # a normal DJ pitch, silent
    with caplog.at_level("WARNING"):
        _check_stretch_rate(np.exp(STRETCH_WARN + 0.01), "warned")
    assert "warned" in caplog.text
    with pytest.raises(ValueError, match="too far from the set tempo"):
        _check_stretch_rate(np.exp(STRETCH_MAX + 0.01), "refused")
    with pytest.raises(ValueError):
        _check_stretch_rate(np.exp(-(STRETCH_MAX + 0.01)), "slowed down too far")


# ── Sound quality: the sweep has no block edges ────────────────────────────────


def test_swept_filter_with_a_constant_cutoff_equals_the_fixed_filter():
    """The block design left 20%-of-peak transients every 8192 samples."""
    from src.audio.audio_mixer import _fixed_filter

    y = (np.random.default_rng(1).standard_normal((SR * 4, 2)) * 0.1).astype(np.float32)
    for hz in (300.0, 1000.0):
        ref = _fixed_filter(y, hz, "highpass")
        out = _swept_filter(y, np.full(len(y), hz), "highpass")
        assert np.abs(out - ref).max() < 1e-9 * max(np.abs(ref).max(), 1e-12) + 1e-9, hz


def test_swept_filter_adds_no_broadband_artefacts():
    """Steady tones through a rise sweep: anything above 6 kHz is artefact."""
    from scipy.signal import butter, sosfiltfilt

    n = SR * 4
    t = np.arange(n) / SR
    tones = (0.2 * np.sin(2 * np.pi * 110 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)).astype(
        np.float32
    )
    out = _swept_filter(
        tones, _sweep_cutoffs([(0.0, 2500.0), (0.75, 25.0), (1.0, 25.0)], n), "highpass"
    )
    art = sosfiltfilt(butter(6, 6000, "highpass", fs=SR, output="sos"), out)
    rel_db = 20 * np.log10(np.sqrt((art**2).mean()) / np.sqrt((out**2).mean()) + 1e-15)
    assert rel_db < -90, f"artefact energy {rel_db:.1f} dB"


# ── Sound quality: the overlap sits level with both bodies ─────────────────────


def test_overlap_gain_anchors_to_both_bodies():
    """A ends at -14 dB: the overlap must open at A's level whatever the summed
    records measure, and end at unity (B alone at its matched level)."""
    from src.audio.audio_mixer import _overlap_gain_ride

    bar_n = SR // 2
    rng = np.random.default_rng(3)
    a_body = rng.standard_normal((8 * bar_n, 2)).astype(np.float32) * 10 ** (-14 / 20)
    b_body = rng.standard_normal((8 * bar_n, 2)).astype(np.float32) * 10 ** (-18 / 20)
    overlap = rng.standard_normal((16 * bar_n, 2)).astype(np.float32) * 10 ** (-20 / 20)
    out, g0, g1 = _overlap_gain_ride(overlap, a_body, b_body, bar_n)

    def db(x):
        return 20 * np.log10(np.sqrt((x**2).mean()))

    assert db(out[:bar_n]) == pytest.approx(-14, abs=0.7)
    assert db(out[-bar_n:]) == pytest.approx(-20, abs=0.7)
    assert g0 == pytest.approx(6, abs=0.7) and g1 == 0.0


def test_overlap_gain_is_clamped_and_leaves_a_level_overlap_alone():
    from src.audio.audio_mixer import OVERLAP_GAIN_MAX_DB, _overlap_gain_ride

    bar_n = SR // 2
    rng = np.random.default_rng(4)
    level = rng.standard_normal((8 * bar_n, 2)).astype(np.float32) * 0.1
    same, g0, g1 = _overlap_gain_ride(level.copy(), level, level, bar_n)
    assert abs(g0) < 0.5 and abs(g1) < 0.5
    assert np.abs(same - level).max() < 0.1 * 0.1
    quiet = level * 10 ** (-30 / 20)
    _, g0, g1 = _overlap_gain_ride(quiet, level, level, bar_n)
    assert g0 == OVERLAP_GAIN_MAX_DB and g1 == 0.0


def test_kick_periodicity_separates_a_running_kick_from_noise():
    from src.audio.audio_mixer import KICK_PERIODIC_MIN, _kick_periodicity

    kick = _kick_pattern(seconds=8.0)
    noise = (np.random.default_rng(5).standard_normal(SR * 8) * 0.1).astype(np.float32)
    assert _kick_periodicity(kick, BPM) > 0.5
    assert _kick_periodicity(noise, BPM) < KICK_PERIODIC_MIN


# ── Sound quality: true-peak limiter ───────────────────────────────────────────


def _true_peak_db(x):
    from src.audio.audio_mixer import _true_peak_per_sample

    return 20 * np.log10(_true_peak_per_sample(x).max())


def test_limiter_holds_true_peak_under_the_ceiling():
    from src.audio.audio_mixer import LIMIT_CEILING_DB, _true_peak_limiter

    rng = np.random.default_rng(6)
    x = (rng.standard_normal((SR * 3, 2)) * 0.3).astype(np.float32)  # peaks well above 0 dBFS
    x[SR : SR + 20] *= 4  # and an isolated burst
    y, stats = _true_peak_limiter(x)
    assert _true_peak_db(y) <= LIMIT_CEILING_DB + 0.1, _true_peak_db(y)
    assert stats["max_reduction_db"] > 3


def test_limiter_leaves_quiet_material_alone():
    from src.audio.audio_mixer import _true_peak_limiter

    x = (np.random.default_rng(7).standard_normal((SR, 2)) * 0.05).astype(np.float32)
    y, stats = _true_peak_limiter(x)
    assert np.array_equal(x, y) and stats["max_reduction_db"] == 0.0


def test_limiter_only_touches_the_loud_moment():
    """One burst in quiet material: 300 ms later the gain is back to unity."""
    from src.audio.audio_mixer import _true_peak_limiter

    x = (np.random.default_rng(8).standard_normal((SR * 2, 2)) * 0.05).astype(np.float32)
    x[SR : SR + 50] = 0.99
    y, _ = _true_peak_limiter(x)
    before = slice(0, SR - int(0.02 * SR))
    later = slice(SR + int(0.3 * SR), SR * 2)
    assert np.allclose(y[before], x[before], atol=1e-6), "no change before the lookahead"
    assert np.allclose(y[later], x[later], atol=2e-3), "released within 300 ms"


# ── Structure by intent: the curve decides how often the drop is the moment ────


def test_structure_regime_thresholds():
    from src.audio.audio_mixer import STRUCTURE_HIGH, STRUCTURE_LOW, structure_regime

    assert structure_regime(None) == "mid"
    assert structure_regime(STRUCTURE_LOW - 0.01) == "low"
    assert structure_regime(STRUCTURE_LOW) == "mid"
    assert structure_regime(STRUCTURE_HIGH - 0.01) == "mid"
    assert structure_regime(STRUCTURE_HIGH) == "high"


def test_landing_on_the_drop_ends_the_overlap_there_in_whole_phrases():
    from src.audio.audio_mixer import _land_on_drop

    assert _land_on_drop(16, 48, 8, 64) == (16, 32)
    assert _land_on_drop(16, 44, 8, 64) == (20, 24), "28-bar span rounds down to 3 phrases"
    assert _land_on_drop(16, 48, 8, 16) == (32, 16), "capped: enter later, still end on the drop"
    assert _land_on_drop(16, 20, 8, 64) is None, "less than a phrase to the drop"


def test_end_anchored_swap_hands_the_bass_over_in_the_last_bar():
    from src.audio.audio_mixer import _end_anchored_swap

    a, b = _end_anchored_swap(32)
    assert a[1][0] == pytest.approx(31 / 32) and a[-1] == (1.0, 0)
    assert b[1][0] == pytest.approx(31 / 32) and b[-1] == (1.0, 1)
    ea, eb = _env(a, ENV_N, "out"), _env(b, ENV_N, "in")
    assert (ea + eb).max() < 1.6, "one bass at a time"


def test_a_low_next_slot_prefers_a_quiet_tail():
    """Two cue-outs on boundaries: 64 (tail = groove) or 72 (tail = breakdown).
    Only when the next slot is low does the breakdown tail win the drift penalty."""
    from src.audio.audio_mixer import _choose_window

    info = _fake_info(
        n_bars=200,
        sections=[
            {"label": "intro", "bars": [0, 8]},
            {"label": "drop", "bars": [8, 64]},
            {"label": "groove", "bars": [64, 72]},
            {"label": "breakdown", "bars": [72, 200]},
        ],
        energy=np.zeros(200),
    )
    assert _choose_window(info, 56, 16, None) == (8, 64)
    assert _choose_window(info, 56, 16, None, quiet_tail=True) == (8, 72)


def test_landing_targets_the_first_drop_a_phrase_past_the_cue_in():
    """B's drop AT the cue-in is the entry, not a landing; the next one is."""
    from src.audio.audio_mixer import _next_drop_after

    sections = [
        {"label": "drop", "bars": [16, 24]},
        {"label": "buildup", "bars": [24, 32]},
        {"label": "drop", "bars": [32, 64]},
    ]
    assert _next_drop_after(sections, 16, 8) == 32
    assert _next_drop_after(sections, 40, 8) is None


def test_gain_ride_leaves_the_end_at_unity_when_landing_on_a_drop():
    from src.audio.audio_mixer import _overlap_gain_ride

    bar_n = SR // 2
    rng = np.random.default_rng(9)
    a_body = rng.standard_normal((8 * bar_n, 2)).astype(np.float32) * 10 ** (-14 / 20)
    drop = rng.standard_normal((8 * bar_n, 2)).astype(np.float32) * 10 ** (-10 / 20)
    buildup = rng.standard_normal((16 * bar_n, 2)).astype(np.float32) * 10 ** (-18 / 20)
    _, g0, g1 = _overlap_gain_ride(buildup, a_body, drop, bar_n, anchor_end=False)
    assert g0 == pytest.approx(4, abs=0.7) and g1 == 0.0


# ── Seam measurement when the outgoing record has no kick ──────────────────────


def _stab_pattern(shift_s: float = 0.0, seconds: float = 8.0, bpm: float = BPM) -> np.ndarray:
    """On-beat 3 kHz stabs with a soft 10 ms attack: rhythm, but no kick and no
    broadband click that would leak into the kick band."""
    beat = 60.0 / bpm
    y = np.zeros(int(SR * seconds), dtype=np.float32)
    length = int(0.12 * SR)
    tt = np.arange(length) / SR
    hit = (np.sin(2 * np.pi * 3000 * tt) * np.minimum(tt / 0.01, 1.0) * np.exp(-tt * 25)).astype(
        np.float32
    )
    for k in range(int(seconds / beat)):
        i = int((k * beat + shift_s) * SR)
        if 0 <= i < len(y) - length:
            y[i : i + length] += hit
    return y


def test_seam_falls_back_to_the_full_band_when_a_record_has_no_kick():
    """Outgoing breakdown (stabs, no kick) against an incoming kick 19 ms late."""
    from src.audio.audio_mixer import _seam_envelopes

    tail = _stab_pattern(0.0)
    head = _kick_pattern(0.019) + _stab_pattern(0.019)
    chosen = _seam_envelopes(tail, head, BPM)
    assert chosen is not None and chosen[2] == "full"
    # The full-band envelope mixes the kick onset with the stab's 10 ms ramp, so
    # the estimate sits a few ms early. Kick vs hats on real tracks showed the
    # same ~6 ms bias; a flam starts being audible around 10 ms.
    assert measure_seam_offset(tail, head, BPM) * 1000 == pytest.approx(19, abs=5)


def test_seam_uses_the_kick_when_both_have_one():
    from src.audio.audio_mixer import _seam_envelopes

    assert _seam_envelopes(_kick_pattern(), _kick_pattern(0.01), BPM)[2] == "kick"


def test_seam_shifts_nothing_when_neither_record_has_a_pulse():
    from src.audio.audio_mixer import _seam_envelopes

    rng = np.random.default_rng(11)
    a = (rng.standard_normal(SR * 8) * 0.1).astype(np.float32)
    b = (rng.standard_normal(SR * 8) * 0.1).astype(np.float32)
    assert _seam_envelopes(a, b, BPM) is None
    assert measure_seam_offset(a, b, BPM) == 0.0


def test_a_silent_kick_band_does_not_count_as_a_kick():
    """Leakage from a 3 kHz stab is periodic but carries ~0.1% of the onset
    energy; real sections carry 5-45%. The energy floor tells them apart."""
    from src.audio.audio_mixer import KICK_ENERGY_MIN_SHARE, _kick_envelope, _onset_envelope

    stab = _stab_pattern()
    share = _kick_envelope(stab).sum() / _onset_envelope(stab).sum()
    assert share < KICK_ENERGY_MIN_SHARE / 5
    kick = _kick_pattern()
    assert _kick_envelope(kick).sum() / _onset_envelope(kick).sum() > KICK_ENERGY_MIN_SHARE * 5


# ── Alignment against the record's own grid ────────────────────────────────────


def _grid(bpm: float = BPM, seconds: float = 8.0, shift_s: float = 0.0) -> np.ndarray:
    return np.arange(0, seconds, 60.0 / bpm) + shift_s


def test_grid_phase_reads_how_late_the_sounds_sit_on_the_grid():
    from src.audio.audio_mixer import grid_phase

    y = _kick_pattern(0.019)  # kicks 19 ms after the grid beats
    off, corr, band = grid_phase(y, _grid(), BPM)
    assert off * 1000 == pytest.approx(19, abs=2) and band == "kick" and corr > 0.1


def test_grid_phase_ignores_off_beat_layers():
    """Open hats on the off-beat sit half a beat from the grid, far outside the
    40 ms window, so they cannot win; the on-grid layer does."""
    from src.audio.audio_mixer import grid_phase

    half_beat = 0.5 * 60 / BPM
    y = _kick_pattern(0.010) + 2.0 * _stab_pattern(half_beat)  # loud off-beat stabs
    off, _, band = grid_phase(y, _grid(), BPM)
    assert off * 1000 == pytest.approx(10, abs=2), band


def test_grid_phase_is_none_when_nothing_follows_the_grid():
    from src.audio.audio_mixer import grid_phase

    noise = (np.random.default_rng(12).standard_normal(SR * 8) * 0.1).astype(np.float32)
    assert grid_phase(noise, _grid(), BPM) is None


def test_seam_from_grids_is_the_difference_of_the_two_residuals():
    from src.audio.audio_mixer import seam_offset_from_grids

    tail = _kick_pattern(0.005)  # A's kicks 5 ms late on A's grid
    head = _kick_pattern(0.024)  # B's kicks 24 ms late on B's grid
    off, detail = seam_offset_from_grids(tail, head, _grid(), _grid(), BPM)
    assert off * 1000 == pytest.approx(19, abs=2)
    assert detail["tail"][2] == "kick" and detail["head"][2] == "kick"


def test_seam_from_grids_can_never_jump_half_a_beat():
    """A's grid is right but A's loudest layer is off-beat; B is plain. The old
    envelope correlation read a half beat here. The grid method stays small."""
    from src.audio.audio_mixer import seam_offset_from_grids

    half_beat = 0.5 * 60 / BPM
    tail = 0.3 * _kick_pattern(0.0) + 2.0 * _stab_pattern(half_beat)
    head = _kick_pattern(0.012)
    off, _ = seam_offset_from_grids(tail, head, _grid(), _grid(), BPM)
    assert abs(off) < 0.04
    assert off * 1000 == pytest.approx(12, abs=3)


def test_own_grid_calibration_recovers_a_late_grid():
    from src.audio.audio_mixer import calibrate_grid_phase_own

    y = _kick_pattern(0.0, seconds=30.0)
    late_grid = _grid(seconds=30.0, shift_s=-0.025)  # grid 25 ms before the kicks
    assert calibrate_grid_phase_own(y, late_grid, BPM) * 1000 == pytest.approx(25, abs=2)


def test_grid_seam_shift_is_applied_exactly_once(monkeypatch, tmp_path):
    """The own-grid residual does not move with the cut, so re-measuring after the
    shift returns the same number; looping on it multiplied the shift by four."""
    import soundfile as sf

    from src.audio import audio_mixer

    monkeypatch.setattr(audio_mixer, "SEAM_METHOD", "grid")
    calls = []
    real = audio_mixer.seam_offset_from_grids

    def spy(tail, head, bt, bh, bpm_):
        out = real(tail, head, bt, bh, bpm_)
        calls.append(len(tail))
        return out

    monkeypatch.setattr(audio_mixer, "seam_offset_from_grids", spy)
    bpm = 128.0
    a = _kick_pattern(0.0, seconds=200.0, bpm=bpm)
    b = _kick_pattern(0.020, seconds=200.0, bpm=bpm)
    sf.write(tmp_path / "a.wav", np.stack([a, a], 1), SR)
    sf.write(tmp_path / "b.wav", np.stack([b, b], 1), SR)
    rep = render_mix(
        [tmp_path / "a.wav", tmp_path / "b.wav"],
        tmp_path / "m.wav",
        target_bpm=bpm,
        play_minutes=1.5,
    )
    tr = rep["transitions"][0]
    assert abs(tr["seam_offset_ms"]) < 4.5, "one shift leaves only rounding"
    full_length_calls = [n for n in calls if n == max(calls)]
    assert len(full_length_calls) == 1, "the whole-overlap residual is measured once, not looped"


def test_grid_phase_catches_a_grid_that_is_a_quarter_beat_off():
    """Ben Böhmer – Vale: kick, bass and mids all 122 ms before the tracker grid.
    A 40 ms window could not see it; the rhythm bands over half a beat can."""
    from src.audio.audio_mixer import grid_phase

    quarter = 0.25 * 60 / BPM
    y = _kick_pattern(0.0, seconds=16.0)
    off, _, band = grid_phase(y, _grid(seconds=16.0, shift_s=quarter), BPM)  # grid a quarter late
    assert off == pytest.approx(-quarter, abs=0.003) and band == "kick"


def _bass_pattern(seconds: float = 16.0, bpm: float = BPM) -> np.ndarray:
    """A 300 Hz bass note on every beat, nothing in the kick band."""
    beat = 60 / bpm
    n = int(SR * seconds)
    y = np.zeros(n, dtype=np.float32)
    length = int(0.15 * SR)
    tt = np.arange(length) / SR
    hit = (np.sin(2 * np.pi * 300 * tt) * np.minimum(tt / 0.005, 1.0) * np.exp(-tt * 20)).astype(
        np.float32
    )
    for k in range(int(seconds / beat)):
        i = int(k * beat * SR)
        if i + length < n:
            y[i : i + length] += hit
    return y


def test_grid_phase_prefers_the_bass_over_a_weak_off_beat_kick():
    """Sin Sin – Break Down: kick weak and off the beat, bass on the beat."""
    from src.audio.audio_mixer import grid_phase

    beat = 60 / BPM
    kick = _kick_pattern(beat / 3, seconds=16.0)
    # a syncopated kick: only beats 2 and 4 of each bar carry one
    for k in range(int(16 / beat)):
        if k % 4 in (0, 2):
            i = int((k * beat + beat / 3) * SR)
            kick[i : i + 1500] = 0
    y = _bass_pattern() + 0.15 * kick
    off, _, band = grid_phase(y, _grid(seconds=16.0), BPM)
    assert band == "lowmid" and abs(off) < 0.006, (band, off)


def test_grid_phase_takes_the_kick_when_kick_and_bass_agree():
    from src.audio.audio_mixer import grid_phase

    y = _bass_pattern() + _kick_pattern(0.008, seconds=16.0)  # kick 8 ms late, bass on the grid
    off, _, band = grid_phase(y, _grid(seconds=16.0), BPM)
    assert band == "kick" and off * 1000 == pytest.approx(8, abs=2)


def test_anchor_level_ignores_drop_out_bars():
    """Materium's last bars: -22 -42 -25 -17. The ear holds -17; the plain median said -23."""
    from src.audio.audio_mixer import _anchor_level_db

    assert _anchor_level_db([-22, -42, -25, -17]) == pytest.approx(-19.5)  # median of -22, -17
    assert _anchor_level_db([-47, -23, -18, -18, -22, -42, -25, -17]) == pytest.approx(-18.0)
    assert _anchor_level_db([-16, -16.5, -16, -17]) == pytest.approx(-16.25)


def test_gain_ride_is_not_dragged_down_by_a_silent_bar():
    from src.audio.audio_mixer import _overlap_gain_ride

    bar_n = SR // 2
    rng = np.random.default_rng(13)
    a_body = rng.standard_normal((8 * bar_n, 2)).astype(np.float32) * 10 ** (-16 / 20)
    a_body[5 * bar_n : 6 * bar_n] *= 10 ** (-30 / 20)  # a one-bar drop-out near the end
    b_body = rng.standard_normal((8 * bar_n, 2)).astype(np.float32) * 10 ** (-16 / 20)
    overlap = rng.standard_normal((16 * bar_n, 2)).astype(np.float32) * 10 ** (-20 / 20)
    out, g0, _ = _overlap_gain_ride(overlap, a_body, b_body, bar_n)
    assert g0 == pytest.approx(4, abs=0.7), (
        "anchored to -16, not to a median pulled down by silence"
    )


def test_edge_level_is_the_last_two_real_bars():
    """A's last 8 bars on the house clip; the ear holds what it just heard."""
    from src.audio.audio_mixer import _edge_level_db

    last8 = [-37.1, -23.9, -17.9, -18.2, -21.4, -43.9, -25.1, -16.9]
    assert _edge_level_db(last8, "end") == pytest.approx(-16.9)
    assert _edge_level_db([-16.9, -25.1, -43.9, -21.4], "start") == pytest.approx(-16.9)
    assert _edge_level_db([-16, -16, -16, -16], "end") == pytest.approx(-16)


def test_gain_ride_ignores_a_breakdown_after_the_seam():
    """techno_build, Amelie Lens → Mosaic: Mosaic's first 8 body bars are an 8-bar
    breakdown (-21..-25) under a -11 body. The end is unity, so a level overlap is
    left alone and the breakdown stays a breakdown."""
    from src.audio.audio_mixer import _overlap_gain_ride

    bar_n = SR // 2
    rng = np.random.default_rng(15)
    a_body = rng.standard_normal((32 * bar_n, 2)).astype(np.float32) * 10 ** (-11 / 20)
    b_body = rng.standard_normal((48 * bar_n, 2)).astype(np.float32) * 10 ** (-11 / 20)
    b_body[: 8 * bar_n] *= 10 ** (-12 / 20)  # the breakdown right after the seam
    overlap = rng.standard_normal((32 * bar_n, 2)).astype(np.float32) * 10 ** (-11 / 20)
    _, g0, g1 = _overlap_gain_ride(overlap, a_body, b_body, bar_n)
    assert abs(g0) < 0.7, g0
    assert abs(g1) < 0.7, g1


def test_gain_ride_opens_at_the_last_bars_not_the_body():
    """Regression p04, Farrago → Heil: A's body is -12 but its last 2 bars are a
    break at -19; anchoring on the body lifted the first overlap bar +9 dB, a step
    exactly at the seam. The start must follow what the ear just heard (-19)."""
    from src.audio.audio_mixer import _overlap_gain_ride

    bar_n = SR // 2
    rng = np.random.default_rng(16)
    a_body = rng.standard_normal((32 * bar_n, 2)).astype(np.float32) * 10 ** (-12 / 20)
    a_body[-2 * bar_n :] *= 10 ** (-7 / 20)  # the break begins two bars before the seam
    b_body = rng.standard_normal((16 * bar_n, 2)).astype(np.float32) * 10 ** (-12 / 20)
    overlap = rng.standard_normal((16 * bar_n, 2)).astype(np.float32) * 10 ** (-24 / 20)
    _, g0, g1 = _overlap_gain_ride(overlap, a_body, b_body, bar_n)
    # the edge rule skips the -19 bars as drop-outs and would lift +9 (cap); the
    # continuity cap holds the first bar to last-heard -19 + 1.5 → +6.5
    assert g0 == pytest.approx(6.5, abs=0.7), g0
    assert g1 == 0.0


def test_gain_ride_start_anchor_reads_the_overlap_opening_not_its_loud_end():
    """Farrago – Sinner: A ends at -17, the overlap opens at -24 and ramps to -13.
    The start gain must be about +7 dB, not read off the loud end of the ramp."""
    from src.audio.audio_mixer import _overlap_gain_ride

    bar_n = SR // 2
    rng = np.random.default_rng(14)
    a_body = rng.standard_normal((8 * bar_n, 2)).astype(np.float32) * 10 ** (-17 / 20)
    b_body = rng.standard_normal((8 * bar_n, 2)).astype(np.float32) * 10 ** (-13 / 20)
    ramp_db = np.linspace(-24, -13, 16 * bar_n, dtype=np.float32)
    overlap = (
        rng.standard_normal((16 * bar_n, 2)).astype(np.float32) * (10 ** (ramp_db / 20))[:, None]
    )
    _, g0, g1 = _overlap_gain_ride(overlap, a_body, b_body, bar_n)
    assert g0 == pytest.approx(7, abs=1.0), g0
    assert g1 == pytest.approx(0, abs=1.0), g1


def test_last_track_plays_on_to_the_requested_length(tmp_path):
    """Two 4-minute tracks, 3 minutes asked in total: the last body is trimmed to
    land within a phrase of 3:00. Asked for 6 minutes, it plays on toward its end."""
    import soundfile as sf

    bpm = 128.0
    y = _kick_pattern(0.0, seconds=240.0, bpm=bpm)
    for name in ("a", "b"):
        sf.write(tmp_path / f"{name}.wav", np.stack([y, y], 1), SR)
    paths = [tmp_path / "a.wav", tmp_path / "b.wav"]
    short = render_mix(
        paths, tmp_path / "s.wav", target_bpm=bpm, play_minutes=1.5, total_minutes=3.0
    )
    long_ = render_mix(
        paths, tmp_path / "l.wav", target_bpm=bpm, play_minutes=1.5, total_minutes=6.0
    )
    plain = render_mix(paths, tmp_path / "p.wav", target_bpm=bpm, play_minutes=1.5)
    phrase_s = 8 * 4 * 60 / bpm
    assert abs(short["duration_s"] - 180) <= phrase_s, short["duration_s"]
    assert long_["duration_s"] > plain["duration_s"] + 45, (
        long_["duration_s"],
        plain["duration_s"],
    )
    assert long_["duration_s"] <= 240 + 240, "cannot play past the end of the track"


# ── Seam decision: when the kick is weak, the other layers decide ──────────────


def _fake_envs(monkeypatch, kick_off_s, others_off_s, bpm=BPM):
    """Two records whose kick band and other bands sit at different offsets."""
    from src.audio import audio_mixer

    tail = {
        "kick": _kick_pattern(0.0),
        "lowmid": _bass_pattern(8.0),
        "mid": _stab_pattern(0.0),
        "high": _stab_pattern(0.0),
        "full": _kick_pattern(0.0) + _bass_pattern(8.0),
    }
    head = {
        "kick": _kick_pattern(kick_off_s),
        "lowmid": _bass_pattern(8.0),
        "mid": _stab_pattern(others_off_s),
        "high": _stab_pattern(others_off_s),
        "full": _kick_pattern(others_off_s) + _bass_pattern(8.0),
    }
    # roll the bass by the requested offset so every "other" layer agrees
    shift = int(others_off_s * SR)
    head["lowmid"] = np.roll(head["lowmid"], shift)
    real = audio_mixer._band_envelopes

    def fake(y):
        for env in (tail, head):
            if y is env["kick"]:
                return {k: real(v)[k] for k, v in env.items()}
        return real(y)

    monkeypatch.setattr(audio_mixer, "_band_envelopes", fake)
    return tail["kick"], head["kick"]


def test_seam_decision_takes_the_kick_when_layers_agree(monkeypatch):
    from src.audio.audio_mixer import seam_decision

    tail, head = _fake_envs(monkeypatch, 0.019, 0.019)
    off, label, _ = seam_decision(tail, head, BPM)
    assert label == "kick" and off * 1000 == pytest.approx(19, abs=3)


def test_seam_decision_keeps_the_kick_on_a_half_beat_disagreement(monkeypatch):
    """Sin Sin → Nova: off-beat hats put every other layer half a beat away."""
    from src.audio.audio_mixer import seam_decision

    half = 0.5 * 60 / BPM
    tail, head = _fake_envs(monkeypatch, 0.010, 0.010 + half)
    off, label, _ = seam_decision(tail, head, BPM)
    assert label.startswith("kick") and off * 1000 == pytest.approx(10, abs=3)


def test_seam_decision_uses_the_consensus_on_a_quarter_beat_disagreement(monkeypatch):
    """Wigbert → Joyhauser: weak kick a quarter beat from bass, mids and hats."""
    from src.audio.audio_mixer import seam_decision

    quarter = 0.25 * 60 / BPM
    tail, head = _fake_envs(monkeypatch, 0.0, quarter)
    off, label, _ = seam_decision(tail, head, BPM)
    assert label == "consensus" and off == pytest.approx(quarter, abs=0.004)


def test_seam_decision_rule_on_the_two_real_seams():
    """The numbers measured on the two real seams, fed to the rule directly."""
    from src.audio.audio_mixer import SEAM_AGREE_S, SEAM_HALF_BEAT_TOL

    def rule(kick, consensus, bpm):
        half = 0.5 * 60 / bpm
        d = abs(kick - consensus)
        if d <= SEAM_AGREE_S:
            return "kick"
        if abs(d - half) <= SEAM_HALF_BEAT_TOL * half:
            return "kick(offbeat layers)"
        return "consensus"

    assert rule(0.103, -0.116, 127.5).startswith("kick")  # Sin Sin → Nova, Anas preferred the kick
    assert (
        rule(0.075, 0.194, 137.0) == "consensus"
    )  # Wigbert → Joyhauser, Anas preferred the consensus


# ── Kick-hit verifier ──────────────────────────────────────────────────────────


def _kick_train(bpm, n_bars, offset_s=0.0, every=1, seed=0):
    """Stereo click train: a 60 Hz burst on every `every`-th beat, plus a little noise."""
    beat = 60.0 / bpm
    n = int(n_bars * 4 * beat * SR)
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n).astype(np.float32) * 0.002
    t = np.arange(int(0.05 * SR)) / SR
    burst = (np.sin(2 * np.pi * 60 * t) * np.exp(-t / 0.02)).astype(np.float32)
    for k in range(0, n_bars * 4, every):
        i = int((k * beat + offset_s) * SR)
        if 0 <= i < n - len(burst):
            y[i : i + len(burst)] += burst
    return np.stack([y, y], axis=1)


def test_kick_hit_residual_reads_a_late_head():
    from src.audio.audio_mixer import kick_hit_residual

    tail = _kick_train(128, 16)
    head = _kick_train(128, 16, offset_s=0.100, seed=1)  # B's kicks 100 ms late
    out = kick_hit_residual(tail, head, 128)
    assert out is not None
    residual, detail = out
    assert residual == pytest.approx(0.100, abs=0.006), residual
    assert detail["a_per_beat"] >= 0.9 and detail["b_per_beat"] >= 0.9


def test_kick_hit_residual_stays_silent_without_a_kick_per_beat():
    from src.audio.audio_mixer import kick_hit_residual

    tail = _kick_train(128, 16)
    sparse = _kick_train(128, 16, offset_s=0.100, every=2, seed=1)  # half-time: gate must fail
    assert kick_hit_residual(tail, sparse, 128) is None


def test_kick_hit_residual_refuses_more_than_a_third_of_a_beat():
    from src.audio.audio_mixer import kick_hit_residual

    tail = _kick_train(128, 16)
    far = _kick_train(128, 16, offset_s=0.200, seed=1)  # beat = 469 ms; 200 ms > a third
    assert kick_hit_residual(tail, far, 128) is None


def test_kick_hit_residual_confirms_an_aligned_pair():
    from src.audio.audio_mixer import kick_hit_residual

    out = kick_hit_residual(_kick_train(128, 16), _kick_train(128, 16, seed=1), 128)
    assert out is not None and abs(out[0]) < 0.005
