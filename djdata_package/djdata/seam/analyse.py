"""Analyse one seam: window audio + two original tracks + coarse anchors → one result row and a
per-beat curves table. Pure function of its inputs; the runner handles state and files."""

import logging
from pathlib import Path
import time

import numpy as np

from .align import SR, Aligner, band_indices, load_mono, mix_beat_period, onset_env, power_spec
from .gains import Fitter
from .params import parameters, relative_curves

log = logging.getLogger("djdata.seam.analyse")


def analyse(window_path: str | Path, a_path: str | Path, b_path: str | Path, coarse: dict, cfg_an: dict) -> dict:
    t_all = time.time()
    timing = {}
    t0w, t1w = coarse["window"]
    t = time.time()
    mix = load_mono(window_path)
    SM = power_spec(mix)
    SS = {"A": power_spec(load_mono(a_path)), "B": power_spec(load_mono(b_path))}
    oe = {"M": onset_env(SM), "A": onset_env(SS["A"]), "B": onset_env(SS["B"])}
    timing["load_stft"] = round(time.time() - t, 1)

    bands = cfg_an["bands"]
    idx = band_indices(bands)
    # Raveform can put A's mix-out before B's mix-in (a gap); the seam region is then [earlier, later]
    ov_start = min(coarse["overlap_start_mix_t"], coarse["overlap_end_mix_t"]) - t0w
    ov_end = max(coarse["overlap_start_mix_t"], coarse["overlap_end_mix_t"]) - t0w
    # tempo: the mix's own onset autocorrelation, seeded by the outgoing track's beat period over its rate
    seed_bpm = coarse.get("bpm_guess") or 128.0
    beat = mix_beat_period(oe["M"], seed_bpm) if coarse.get("bpm_guess") else _beat_from_scan(oe["M"])
    al = Aligner(SM, SS, oe, beat, idx)

    t = time.time()
    maps = {}
    for tag, side in (("A", "before"), ("B", "after")):
        c = coarse[tag]
        # anchor in window time; B's anchor is moved to where it plays alone (after the overlap)
        m_anchor = c["mix_t"] - t0w
        o_anchor = c["orig_t"]
        if tag == "B":
            shift = ov_end - m_anchor
            m_anchor, o_anchor = m_anchor + shift, o_anchor + shift * c["rate"]
        else:
            shift = ov_start - m_anchor
            m_anchor, o_anchor = m_anchor + shift, o_anchor + shift * c["rate"]
        res = al.align(tag, side, m_anchor, o_anchor, c["rate"])
        maps[tag] = (res["o"], res["m"], res["rate"])
        maps[tag + "_info"] = res
    timing["align"] = round(time.time() - t, 1)

    # beat grid anchored on A's alignment
    a_m = maps["A"][1]
    g0 = a_m - np.floor(a_m / beat) * beat
    beats = np.arange(g0, len(mix) / SR - beat, beat)
    k0 = int(np.argmin(np.abs(beats - ov_start)))
    k1 = int(np.argmin(np.abs(beats - ov_end)))
    nb = len(beats) - 1

    def omap(tag, mix_t):
        o, m, r = maps[tag]
        return o + (mix_t - m) * r

    # alone regions (for reference levels and calibration) and absent regions (for presence floors)
    alone = {"A": list(range(max(k0 - 40, 0), max(k0 - 8, 0))), "B": list(range(min(k1 + 8, nb), min(k1 + 40, nb)))}
    b_absent = [k for k in range(0, k0) if omap("B", beats[k + 1]) < 0.0]
    a_dur = coarse["A"].get("duration")
    a_absent = [k for k in range(k1, nb) if a_dur and omap("A", beats[k]) > a_dur]
    absent = {"A": a_absent, "B": b_absent}

    t = time.time()
    fitter = Fitter(SM, SS, {k: v for k, v in maps.items() if not k.endswith("_info")}, idx, beats,
                    cfg_an["lam"], cfg_an["local_ms"])
    fitter.calibrate(alone)
    fit = fitter.fit()
    timing["fit"] = round(time.time() - t, 1)

    curves = relative_curves(fit, bands, alone, absent, cfg_an["presence_db"])
    beats_abs = beats + t0w
    par = parameters(curves["rel"], curves["thr"], fit, beats_abs, k0, k1, bands, beat)
    pres = par.pop("presence")

    row = {
        "tempo_bpm": round(60.0 / beat, 2),
        "rate_A": round(maps["A"][2], 5), "rate_B": round(maps["B"][2], 5),
        "bar_fix_A": maps["A_info"]["bar_fix"], "bar_fix_B": maps["B_info"]["bar_fix"],
        "onset_shift_A_s": round(maps["A_info"]["onset_shift_s"], 3), "onset_shift_B_s": round(maps["B_info"]["onset_shift_s"], 3),
        "absent_region_B_beats": len(b_absent), "absent_region_A_beats": len(a_absent),
        "raveform_overlap_start_t": coarse["overlap_start_mix_t"], "raveform_overlap_end_t": coarse["overlap_end_mix_t"],
        **par,
        "timing_s": {**timing, "total": round(time.time() - t_all, 1)},
    }
    curve_rows = []
    for k in range(nb):
        r = {"beat": k - k0, "mix_t": round(float(beats_abs[k]), 3), "A_present": int(pres["A"][k]), "B_present": int(pres["B"][k]),
             "unexplained": round(float(fit["unexplained"][k]), 3)}
        for tag in ("A", "B"):
            for b in range(len(bands)):
                v = curves["rel"][(tag, b)][k]
                r[f"{tag}_b{b}_db"] = None if np.isnan(v) else round(float(v), 1)
                r[f"{tag}_b{b}_ambiguous"] = int(fit["ambiguous"][(tag, b)][k])
        curve_rows.append(r)
    return {"row": row, "curves": curve_rows}


def _beat_from_scan(oe_mix) -> float:
    """No tempo seed: pick the strongest onset autocorrelation lag between 80 and 180 bpm."""
    import librosa

    from .align import HOP

    fps = SR / HOP
    a = oe_mix[: int(min(len(oe_mix), 60 * fps))]
    ac = librosa.autocorrelate(a - a.mean())
    lag = np.arange(len(ac)) / fps
    idx = np.where((lag > 60 / 180) & (lag < 60 / 80))[0]
    return mix_beat_period(oe_mix, 60.0 / lag[idx[int(np.argmax(ac[idx]))]])
