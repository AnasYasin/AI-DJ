"""
Phase 8b — audio mixer: ordered tracks → one continuous mixed .mp3.

Per track:
  1. Structure map from audio_segmenter (bpm, bars, labeled sections).
  2. Time-stretch to the mix's target BPM (pyrubberband if the rubberband
     CLI is installed, else librosa phase vocoder — fine for small ratios).
  3. Cue-in after the intro, cue-out at the outro (phrase-aligned bars).
Per transition:
  4. Overlap of `transition_bars` bars, sample-aligned on the bar grid.
  5. Equal-power crossfade + bass swap: incoming is high-passed until the
     swap point (60% through the overlap), then the outgoing loses its lows —
     two basslines never play together.
  6. RMS gain matching to the first track.

Run:
  python -m src.audio.audio_mixer out.mp3 track1.mp3 track2.mp3 ...
"""

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from src.data.audio_segmenter import segment

log = logging.getLogger(__name__)

SR = 44_100
TRANSITION_BARS = 16
MIN_BODY_BARS = 24          # minimum bars a track plays solo
BASS_SWAP_AT = 0.6          # fraction of the overlap where basslines swap
BASS_HZ = 180
SWAP_RAMP_S = 0.4           # smoothstep ramp width around the swap point


# ── Helpers ────────────────────────────────────────────────────────────────────


def _stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Time-stretch by `rate` (>1 = faster). rubberband if available."""
    if abs(rate - 1.0) < 1e-4:
        return y
    if shutil.which("rubberband"):
        import pyrubberband

        return pyrubberband.time_stretch(y, SR, rate)
    return librosa.effects.time_stretch(y, rate=rate)


def _highpass(y: np.ndarray, hz: float = BASS_HZ) -> np.ndarray:
    sos = butter(4, hz, btype="highpass", fs=SR, output="sos")
    return sosfilt(sos, y)


def _smoothstep_mask(n: int, swap_i: int, ramp: int) -> np.ndarray:
    """0 before swap point, 1 after, smooth ramp of `ramp` samples between."""
    m = np.zeros(n, dtype=np.float32)
    a, b = max(swap_i - ramp // 2, 0), min(swap_i + ramp // 2, n)
    m[b:] = 1.0
    if b > a:
        t = np.linspace(0, 1, b - a, dtype=np.float32)
        m[a:b] = t * t * (3 - 2 * t)
    return m


def _prepare_track(path: str | Path, target_bpm: float, transition_bars: int) -> dict:
    """Load, analyse, stretch to target BPM, choose cue bars."""
    info = segment(path)
    rate = target_bpm / info["bpm"]
    if not 0.8 <= rate <= 1.25:
        log.warning("%s: stretch rate %.2f is extreme", Path(path).name, rate)
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    y = _stretch(y, rate).astype(np.float32)
    bars = np.array(info["bars"], dtype=np.float64) / rate  # stretched bar times

    labels = {s["label"]: s for s in info["sections"]}
    cue_in_bar = labels["intro"]["bars"][1] if "intro" in labels else min(8, len(bars) - 1)
    cue_out_bar = labels["outro"]["bars"][0] if "outro" in labels else len(bars) - 1
    # always leave a full transition tail after the cue-out
    cue_out_bar = min(cue_out_bar, len(bars) - 1 - transition_bars)
    if cue_out_bar - cue_in_bar < MIN_BODY_BARS:  # too aggressive — use more of the track
        cue_in_bar = 0
        cue_out_bar = max(len(bars) - 1 - transition_bars, cue_in_bar + 1)

    return {
        "name": Path(path).stem,
        "audio": y,
        "bars": bars,
        "bpm": info["bpm"],
        "rate": rate,
        "cue_in_bar": int(cue_in_bar),
        "cue_out_bar": int(cue_out_bar),
    }


# ── Mix rendering ──────────────────────────────────────────────────────────────


def render_mix(
    track_paths: list[str | Path],
    out_path: str | Path,
    target_bpm: float | None = None,
    transition_bars: int = TRANSITION_BARS,
) -> dict:
    """Render an ordered list of full tracks into one mixed file."""
    tracks = []
    if target_bpm is None:  # first pass: BPMs only, target = median
        bpms = [segment(p)["bpm"] for p in track_paths]
        target_bpm = float(np.median(bpms))
        log.info("target BPM (median): %.1f  (tracks: %s)", target_bpm, bpms)
    for p in track_paths:
        t = _prepare_track(p, target_bpm, transition_bars)
        log.info(
            "  %s: %.0f→%.0f bpm (rate %.3f), cue bars %d–%d",
            t["name"][:40], t["bpm"], target_bpm, t["rate"], t["cue_in_bar"], t["cue_out_bar"],
        )
        tracks.append(t)

    bar_dur = 4 * 60.0 / target_bpm
    ref_rms = np.sqrt(np.mean(tracks[0]["audio"] ** 2)) + 1e-9
    ramp = int(SWAP_RAMP_S * SR)

    def cut(t, bar_a, bar_b_time):
        a = int(t["bars"][bar_a] * SR)
        b = int(bar_b_time * SR)
        return t["audio"][a:b]

    segments = []
    trans_report = []
    for i, t in enumerate(tracks):
        gain = ref_rms / (np.sqrt(np.mean(t["audio"] ** 2)) + 1e-9)
        t["audio"] = t["audio"] * min(gain, 2.0)

        cue_in_t = t["bars"][t["cue_in_bar"]]
        cue_out_t = t["bars"][t["cue_out_bar"]]

        n_over = 0
        if i < len(tracks) - 1:
            avail_after = len(t["audio"]) / SR - cue_out_t
            n_over = min(transition_bars, int(avail_after / bar_dur))
            nxt = tracks[i + 1]
            avail_next = (
                len(nxt["audio"]) / SR - nxt["bars"][nxt["cue_in_bar"]]
            )
            n_over = max(min(n_over, int(avail_next / bar_dur) - MIN_BODY_BARS), 2)
        t["overlap_bars"] = n_over
        t["cue_in_t"], t["cue_out_t"] = cue_in_t, cue_out_t

    mix = tracks[0]["audio"][
        int(tracks[0]["cue_in_t"] * SR) : int(tracks[0]["cue_out_t"] * SR)
    ].copy()

    for i in range(1, len(tracks)):
        prev, cur = tracks[i - 1], tracks[i]
        n_over = prev["overlap_bars"]
        over_dur = n_over * bar_dur
        O = int(over_dur * SR)

        tail = prev["audio"][
            int(prev["cue_out_t"] * SR) : int(prev["cue_out_t"] * SR) + O
        ]
        head_start = int(cur["cue_in_t"] * SR)
        head = cur["audio"][head_start : head_start + O]
        O = min(len(tail), len(head))
        tail, head = tail[:O], head[:O]

        u = np.linspace(0, 1, O, dtype=np.float32)
        g_out, g_in = np.cos(u * np.pi / 2), np.sin(u * np.pi / 2)

        swap_i = int(O * BASS_SWAP_AT)
        m = _smoothstep_mask(O, swap_i, ramp)  # 0 → before swap, 1 → after
        head_mixed = _highpass(head) * (1 - m) + head * m  # incoming: no bass, then bass
        tail_mixed = tail * (1 - m) + _highpass(tail) * m  # outgoing: bass, then no bass

        overlap = tail_mixed * g_out + head_mixed * g_in
        body = cur["audio"][head_start + O : int(cur["cue_out_t"] * SR)]
        mix = np.concatenate([mix, overlap, body])
        trans_report.append(
            {
                "from": prev["name"][:32],
                "to": cur["name"][:32],
                "bars": n_over,
                "seconds": round(O / SR, 1),
            }
        )

    peak = np.abs(mix).max()
    if peak > 0.98:
        mix = mix * (0.98 / peak)

    out_path = Path(out_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, mix, SR)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name,
             "-codec:a", "libmp3lame", "-b:a", "192k", str(out_path)],
            check=True,
        )
        Path(tmp.name).unlink()

    report = {
        "out": str(out_path),
        "duration_s": round(len(mix) / SR, 1),
        "target_bpm": target_bpm,
        "n_tracks": len(tracks),
        "transitions": trans_report,
    }
    log.info("mix rendered: %.1f min → %s", report["duration_s"] / 60, out_path)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) < 4:
        sys.exit("usage: python -m src.audio.audio_mixer OUT.mp3 TRACK1 TRACK2 [...]")
    rep = render_mix(sys.argv[2:], sys.argv[1])
    print(f"\n{rep['out']}  {rep['duration_s']/60:.1f} min @ {rep['target_bpm']:.0f} bpm")
    for tr in rep["transitions"]:
        print(f"  {tr['from']}  →  {tr['to']}   {tr['bars']} bars ({tr['seconds']}s)")
