import json
import logging
import sys

import librosa
import numpy as np

logging.disable(logging.WARNING)
sys.path.insert(0, ".")
from scipy.signal import correlate

from src.audio import audio_mixer as am
from src.audio.audio_mixer import ONSET_HOP, SR, _envelope_periodicity, _to_mono

plan = json.load(open("data/external/ear_test_2026-09-07/techno/plan.json"))["tracks"]


def env_band(y, lo, hi):
    S = np.abs(librosa.stft(_to_mono(y), n_fft=1024, hop_length=ONSET_HOP))
    f = librosa.fft_frequencies(sr=SR, n_fft=1024)
    return np.maximum(np.diff(S[(f >= lo) & (f < hi)].sum(0), prepend=0), 0)


def off(ea, eb, bpm):
    n = min(len(ea), len(eb))
    ea, eb = ea[:n], eb[:n]
    ea = (ea - ea.mean()) / (ea.std() + 1e-9)
    eb = (eb - eb.mean()) / (eb.std() + 1e-9)
    ml = max(int(0.5 * (60 / bpm) * SR / ONSET_HOP), 1)
    c = correlate(ea, eb, "full") / n
    ctr = n - 1
    w = c[ctr - ml : ctr + ml + 1]
    k = int(np.argmax(w))
    return -(k - ml) * ONSET_HOP / SR * 1000, w.max()


calls = []
orig = am.measure_seam_offset


def spy(tail, head, bpm):
    o = orig(tail, head, bpm)
    ch = am._seam_envelopes(tail, head, bpm)
    band = ch[2] if ch else "none"
    rec = {
        "len_bars": round(len(tail) / SR / (4 * 60 / bpm), 1),
        "applied_ms": round(o * 1000, 1),
        "band": band,
    }
    for lo, hi, lab in (
        (30, 130, "kick"),
        (130, 1000, "lowmid"),
        (1000, 5000, "mid"),
        (5000, 12000, "hats"),
        (20, 16000, "full"),
    ):
        ea, eb = env_band(tail, lo, hi), env_band(head, lo, hi)
        o2, c = off(ea, eb, bpm)
        rec[lab] = (
            f"{o2:+6.0f}({c:.2f}|{_envelope_periodicity(ea, bpm):+.2f}/{_envelope_periodicity(eb, bpm):+.2f})"
        )
    calls.append(rec)
    return o


am.measure_seam_offset = spy
for pair in (int(sys.argv[1]),):
    i, j = 2 * (pair - 1), 2 * (pair - 1) + 1
    a, b = plan[i], plan[j]
    paths = [
        f"data/external/test_tracks/techno/{a['track_id']}.m4a",
        f"data/external/test_tracks/techno/{b['track_id']}.m4a",
    ]
    calls.clear()
    rep = am.render_mix(
        paths,
        f"/tmp/claude-1000/-home-anas-yasin-projects-AI-DJ/7b53d0e9-c755-4a45-ae4b-a83b3ac18d8e/scratchpad/diag_techno{pair}.flac",
        genre="techno",
        curve="arc",
        energy_targets=[a["energy_target01"], b["energy_target01"]],
    )
    tr = rep["transitions"][0]
    print(
        f"\nTECHNO PAIR {pair}: {tr['type']} {tr['bars']} bars, seam {tr['seam_offset_ms']} ({tr['seam_band']}), drift {tr['seam_drift_ms']} {tr['seam_windows_off']}, A tail {tr['tail_section']}, B head {tr['head_section']}"
    )
    print(
        "call  len   applied  band   kick(corr|perA/perB)  lowmid  mid  hats  full   [offset ms]"
    )
    for c in calls:
        print(
            f"{c['len_bars']:5.1f}b {c['applied_ms']:+7.1f} {c['band']:5s} {c['kick']} {c['lowmid']} {c['mid']} {c['hats']} {c['full']}"
        )
