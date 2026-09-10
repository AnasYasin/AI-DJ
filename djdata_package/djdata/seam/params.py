"""Parameters, gimmicks and diagnostics from the per-beat gain curves of one seam.

Levels are dB relative to each record playing alone. Presence uses the low and mid bands only: the
high band of the quieter record is unreliable below about -10 dB (hi-hats of two records look alike).
Entry of a slow bed is uncertain by a bar or more; the -6 dB crossing is the reliable entry marker.
"""

import numpy as np
from scipy.ndimage import median_filter

from .gains import db


def crossing(x, level, direction):
    x = np.asarray(x)
    for k in range(1, len(x)):
        if direction == "down" and x[k - 1] >= level > x[k]:
            return k
        if direction == "up" and x[k - 1] < level <= x[k]:
            return k
    return None


def relative_curves(fit: dict, bands: list, alone: dict, absent: dict, presence_db: float) -> dict:
    """Gains → dB relative to the alone level, NaN where undefined, plus per-band presence thresholds."""
    rel, thr = {}, {}
    for tag in ("A", "B"):
        for b in range(len(bands)):
            x = db(fit["gains"][tag][b])
            q = fit["quiet"][(tag, b)]
            ks = np.array(alone[tag], dtype=int)
            ok = ks[~q[ks]] if len(ks) else ks
            if len(ok) >= 4:
                ref = float(np.median(x[ok]))
            else:  # the record's band is silent through its alone region: reference it where the band exists
                allk = np.where(~q)[0]
                ref = float(np.median(x[allk])) if len(allk) else 0.0
            x = median_filter(x - ref, 3)
            x[q] = np.nan
            rel[(tag, b)] = x
            ab = np.array(absent[tag], dtype=int)
            vals = x[ab] if len(ab) else np.array([])
            vals = vals[np.isfinite(vals)]
            thr[(tag, b)] = max(float(np.percentile(vals, 90)) + 6.0, presence_db) if len(vals) >= 4 else presence_db
    return {"rel": rel, "thr": thr}


def presence(rel, thr, ambiguous, n_bands_lowmid=2) -> dict:
    out = {}
    for tag in ("A", "B"):
        votes = []
        for b in range(n_bands_lowmid):
            v = np.nan_to_num(rel[(tag, b)], nan=-99.0) - thr[(tag, b)] > 0
            v[ambiguous[(tag, b)]] = False
            votes.append(v)
        out[tag] = np.logical_or.reduce(votes)
    return out


def parameters(rel, thr, fit, beats_abs, k0, k1, bands: list, beat_s: float) -> dict:
    """k0/k1: beat indices of the coarse overlap start and end. Times returned in mix seconds."""
    nb = len(beats_abs) - 1
    pres = presence(rel, thr, fit["ambiguous"])
    lo, hi = max(k0 - 8, 0), min(k1 + 8, nb)
    b_on = np.where(pres["B"][lo:])[0]
    a_on = np.where(pres["A"][:hi])[0]
    start = int(b_on[0]) + lo if len(b_on) else None
    end = int(a_on[-1]) + 1 if len(a_on) else None
    a_low = np.nan_to_num(rel[("A", 0)], nan=-60.0)
    b_low = np.nan_to_num(rel[("B", 0)], nan=-60.0)
    diff = a_low - b_low
    # bass swap: A's bass falls under B's and stays there for at least a bar (a one-beat dip is not a swap).
    # Searched from 16 beats before Raveform's edge, which can sit after the real move.
    swap = None
    for k in range(max(k0 - 16, 1), hi):
        if diff[k - 1] >= 0 > diff[k] and np.all(diff[k: min(k + 4, nb)] < 0):
            swap = k
            break
    t = lambda k: None if k is None else float(beats_abs[k])

    def crossings(tag):
        d = "down" if tag == "A" else "up"
        out = {}
        for b in range(len(bands)):
            x = np.nan_to_num(rel[(tag, b)], nan=-60.0)[lo:hi]
            out[b] = {"m6": t(lo + crossing(x, -6, d)) if crossing(x, -6, d) is not None else None,
                      "m12": t(lo + crossing(x, -12, d)) if crossing(x, -12, d) is not None else None}
        return out

    # fader proxy: loudest band per beat; volume cuts: dips ≥ 6 dB that recover within two beats
    cuts = []
    for tag in ("A", "B"):
        f = np.nanmax(np.stack([rel[(tag, b)] for b in range(len(bands))]), 0)
        f = np.nan_to_num(f, nan=-60.0)
        for k in range(max(lo, 1), min(hi, nb - 2)):
            if f[k] < f[k - 1] - 6 and (f[k + 1] > f[k] + 6 or f[k + 2] > f[k] + 6):
                cuts.append({"record": tag, "t": t(k), "depth_db": round(float(f[k - 1] - f[k]), 1)})
    un = fit["unexplained"]
    base = np.percentile(un[max(k0 - 40, 0): max(k0 - 8, 1)], 95) if k0 > 12 else np.percentile(un, 50)
    spikes = [{"t": t(k), "unexplained": round(float(un[k]), 3)} for k in range(lo, hi) if un[k] > max(2 * base, 0.5)]
    overlap_beats = (end - start) if (start is not None and end is not None) else None
    return {
        "overlap_start_t": t(start), "overlap_end_t": t(end), "overlap_beats": overlap_beats,
        "overlap_bars": None if overlap_beats is None else round(overlap_beats / 4, 2),
        "cut_vs_blend": None if overlap_beats is None else ("cut" if overlap_beats <= 4 else "blend"),
        "bass_swap_t": t(swap),
        "A_crossings": crossings("A"), "B_crossings": crossings("B"),
        "volume_cuts": cuts, "unexplained_spikes": spikes,
        "unexplained_mean_overlap": round(float(un[k0:k1].mean()), 3) if k1 > k0 else None,
        "ambiguous_frac_overlap": {f"{tg}_{b}": round(float(v[k0:k1].mean()), 3) for (tg, b), v in fit["ambiguous"].items()} if k1 > k0 else {},
        "quiet_frac_overlap": {f"{tg}_{b}": round(float(v[k0:k1].mean()), 3) for (tg, b), v in fit["quiet"].items()} if k1 > k0 else {},
        "thresholds_db": {f"{tg}_{b}": round(v, 1) for (tg, b), v in thr.items()},
        "presence": pres,
    }
