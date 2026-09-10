"""Per-beat, per-band gains of both records from the mix window.

Model per beat k, band b, over every (bin, sub-slot) inside the beat:
    |M|² ≈ ga² · |A|² + gb² · |B|²          (non-negative least squares)
Before the fit each record gets a per-bin transfer H(f) learned where it plays alone (codec and
mastering differences), and a local ±local_ms re-alignment whose per-beat value is the median over
±8 beats of beats where the record is clearly present (an absent record must not slide its kick onto
the other's). The fit is pulled toward the previous beat's solution (lam) in a forward and a backward
pass; where the two passes disagree by more than 3 dB the band is collinear (same kick on the same
slot) and is flagged ambiguous. Where a record's own band is silent (breakdown) the gain is
undefined and flagged quiet.
"""

import numpy as np
from scipy.optimize import nnls

from .align import SR, slots


def db(x) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.asarray(x, float), 1e-3))


class Fitter:
    def __init__(self, SM, SS: dict, maps: dict, idx: dict, beats: np.ndarray, lam: float, local_ms: float):
        self.SM, self.SS, self.maps, self.idx, self.beats = SM, SS, maps, idx, beats
        self.lam, self.local = lam, local_ms
        self.H = {"A": 1.0, "B": 1.0}

    def omap(self, tag, mix_t):
        o, m, r = self.maps[tag]
        return o + (mix_t - m) * r

    def beat_data(self, k, shifts=(0.0, 0.0)):
        mt0, mt1 = self.beats[k], self.beats[k + 1]
        M = slots(self.SM, mt0, mt1)
        X = []
        for tag, s in zip(("A", "B"), shifts):
            o0, o1 = self.omap(tag, mt0) + s, self.omap(tag, mt1) + s
            Xt = slots(self.SS[tag], o0, o1)
            H = self.H[tag]
            X.append(Xt * (H[:, None] if np.ndim(H) else 1.0))
        return M, X[0], X[1]

    def calibrate(self, alone: dict):
        """alone: {'A': range of beats where A plays alone, 'B': ...}. Learns H per record."""
        for tag, ks in alone.items():
            if len(ks) < 4:
                continue
            num = den = 0.0
            for k in ks:
                M, XA, XB = self.beat_data(k)
                num = num + M.sum(1)
                den = den + (XA if tag == "A" else XB).sum(1)
            logH = np.log(num / (den + 1e-12) + 1e-12)
            out = np.empty_like(logH)
            for i in range(len(logH)):
                w = max(int(i * 0.23 / 2), 2)
                out[i] = logH[max(i - w, 0): i + w + 1].mean()
            self.H[tag] = np.exp(out)

    def fit(self) -> dict:
        nb = len(self.beats) - 1
        cand = np.arange(-self.local, self.local + 1e-9, 1.0) / 1000.0
        best_s = {"A": np.zeros(nb), "B": np.zeros(nb)}
        g1 = {"A": np.zeros(nb), "B": np.zeros(nb)}
        Ms = [slots(self.SM, self.beats[k], self.beats[k + 1]) for k in range(nb)]
        for k in range(nb):
            m = Ms[k].ravel()
            for i, tag in enumerate(("A", "B")):
                best = None
                for sh in cand:
                    x = self.beat_data(k, (sh, 0.0) if tag == "A" else (0.0, sh))[1 + i].ravel()
                    gg = max(np.dot(x, m) / (np.dot(x, x) + 1e-12), 0)
                    res = np.dot(m - gg * x, m - gg * x)
                    if best is None or res < best[0]:
                        best = (res, sh, gg)
                best_s[tag][k], g1[tag][k] = best[1], best[2]
        shift = {}
        for tag in ("A", "B"):
            present = db(np.sqrt(g1[tag])) > db(np.sqrt(g1[tag].max() + 1e-12)) - 12.0
            sm = np.zeros(nb)
            for k in range(nb):
                lo, hi = max(k - 8, 0), min(k + 9, nb)
                sel = best_s[tag][lo:hi][present[lo:hi]]
                sm[k] = np.median(sel) if len(sel) >= 4 else 0.0
            shift[tag] = sm
        data = [self.beat_data(k, (shift["A"][k], shift["B"][k])) for k in range(nb)]
        gains = {"A": {}, "B": {}}
        ambiguous, quiet, resid = {}, {}, np.zeros(nb)
        for b, idx in self.idx.items():
            Xs, ms = [], []
            for M, XA, XB in data:
                m = M[idx].ravel()
                s = np.linalg.norm(m) + 1e-12
                Xs.append(np.stack([XA[idx].ravel(), XB[idx].ravel()], 1) / s)
                ms.append(m / s)
            sols = []
            for order in (range(nb), range(nb - 1, -1, -1)):
                prev, sol_dir = None, np.zeros((nb, 2))
                for k in order:
                    X, m = Xs[k], ms[k]
                    if prev is not None and self.lam > 0:
                        X = np.vstack([X, np.sqrt(self.lam) * np.eye(2)])
                        m = np.concatenate([m, np.sqrt(self.lam) * prev])
                    sol, rn = nnls(X, m)
                    sol_dir[k] = sol
                    prev = sol
                sols.append(sol_dir)
            sol = 0.5 * (sols[0] + sols[1])
            gains["A"][b], gains["B"][b] = np.sqrt(np.maximum(sol[:, 0], 0)), np.sqrt(np.maximum(sol[:, 1], 0))
            fb = np.abs(db(np.sqrt(np.maximum(sols[0], 0))) - db(np.sqrt(np.maximum(sols[1], 0))))
            ambiguous[("A", b)], ambiguous[("B", b)] = fb[:, 0] > 3.0, fb[:, 1] > 3.0
            for i, tag in enumerate(("A", "B")):
                src_pow = np.array([data[k][1 + i][idx].sum() for k in range(nb)])
                quiet[(tag, b)] = 10 * np.log10(src_pow + 1e-12) < 10 * np.log10(np.median(src_pow) + 1e-12) - 15.0
        for k in range(nb):
            M, XA, XB = data[k]
            m = M.ravel()
            X = np.stack([XA.ravel(), XB.ravel()], 1)
            s = np.linalg.norm(m) + 1e-12
            _, rn = nnls(X / s, m / s)
            resid[k] = rn ** 2
        return {"gains": gains, "ambiguous": ambiguous, "quiet": quiet, "unexplained": resid, "shift": shift}
