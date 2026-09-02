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
    _sweep_cutoffs,
    _swept_filter,
    _to_mono,
    gate_transition,
    load_audio,
    max_overlap_seconds,
    measure_seam_offset,
    measured_transition,
    normalise_onset,
    pair_compatibility,
    render_mix,
)

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
        out = _apply_recipe(y, r["A_vol"], r["A_bands"], "out", r["duck"])
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
        out = _apply_recipe(y, r["B_vol"], r["B_bands"], "in", r["duck"])
        assert out.shape == y.shape, name
        assert out.dtype == np.float32, name
        assert np.isfinite(out).all(), name


def test_apply_recipe_preserves_stereo_difference():
    """The envelope must scale both channels, not collapse them."""
    y = _stereo_kicks()
    out = _apply_recipe(y, RECIPES["blend"]["B_vol"], RECIPES["blend"]["B_bands"], "in", 1.0)
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
