"""Can the transition type (signature) and the overlap length be predicted from the two records, before any model is built?"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, r2_score
from sklearn.preprocessing import StandardScaler

from src.data.dataset import load_seams, load_tracks
from src.features.transition_labeler import BPM_RATIO_LOOSE as BPM_LOOSE
from src.features.transition_labeler import BPM_RATIO_TIGHT as BPM_TIGHT
from src.features.transition_labeler import ENERGY_FALL_MIN as ENERGY_FALL
from src.features.transition_labeler import ENERGY_MELT_MAX as ENERGY_MELT
from src.features.transition_labeler import ENERGY_RISE_MIN as ENERGY_RISE
from src.features.transition_labeler import ENERGY_SLAM_MIN as ENERGY_SLAM
from src.features.transition_labeler import (
    HARM_CLASH,
    HARM_COMPATIBLE,
    HARM_PERFECT,
    ONSET_HIGH_MIN,
    normalise_onset,
)
from src.features.transition_labeler import LOUD_MELT_MAX as LOUD_MELT
from src.models.track_arrays import camelot_arrays, camelot_distance

t = load_tracks().set_index("track_id")
s = load_seams()
s = s[
    (s.measured == 1)
    & s.signature.isin(["cut", "centre swap", "late swap", "end swap", "tension"])
]
s = s[s.track_a.isin(t.index) & s.track_b.isin(t.index)].copy()
cols = [
    "bpm",
    "energy_mean",
    "loudness_lufs",
    "onset_strength",
    "spectral_centroid",
    "energy_std",
    "release_year",
    "listeners",
]
A = t.loc[s.track_a, cols].to_numpy(float)
B = t.loc[s.track_b, cols].to_numpy(float)
pa, ma = camelot_arrays(t.loc[s.track_a, "key"].astype(str))
pb, mb = camelot_arrays(t.loc[s.track_b, "key"].astype(str))
cam = camelot_distance(pa, ma, pb, mb)
ea = np.stack(t.loc[s.track_a, "embedding"].values)
eb = np.stack(t.loc[s.track_b, "embedding"].values)
cos = (ea * eb).sum(1) / (np.linalg.norm(ea, axis=1) * np.linalg.norm(eb, axis=1) + 1e-9)
genre = pd.get_dummies(s.genre).to_numpy(float)
X_pair = np.column_stack([A, B, B - A, np.abs(B - A), cam, cos, genre])
X_pair = np.nan_to_num(X_pair, nan=0.0)
y = s.signature.values
tr = (s.split == "train").to_numpy()
va = ~tr
print(
    "seams",
    len(s),
    "train",
    tr.sum(),
    "validation",
    va.sum(),
    "| classes:",
    pd.Series(y).value_counts().to_dict(),
)
maj = pd.Series(y[tr]).mode()[0]
print(f"majority ({maj}) accuracy on validation: {accuracy_score(y[va], [maj] * va.sum()):.3f}")
# the mixer's rules as a reference row, mapped to signatures
de = B[:, 1] - A[:, 1]
dl = B[:, 2] - A[:, 2]
r = np.abs(np.log(B[:, 0] / A[:, 0]))
oa, ob = normalise_onset(A[:, 3]), normalise_onset(B[:, 3])
rule = np.full(len(s), "late swap", dtype=object)
tight, loose = r <= BPM_TIGHT, r <= BPM_LOOSE
rule[tight & (oa > ONSET_HIGH_MIN) & (ob > ONSET_HIGH_MIN)] = "centre swap"  # wave
rule[tight & (cam <= HARM_PERFECT) & (np.abs(de) < ENERGY_MELT) & (np.abs(dl) < LOUD_MELT)] = (
    "centre swap"  # melt
)
rule[de < ENERGY_FALL] = "centre swap"  # fade
rule[(de > ENERGY_RISE) & loose & (cam <= HARM_COMPATIBLE)] = "end swap"  # rise
rule[(tight & (de > ENERGY_SLAM)) | ((cam >= HARM_CLASH) & (de > ENERGY_RISE))] = "cut"  # slam
print(
    f"mixer rules mapped to signatures, accuracy on validation: {accuracy_score(y[va], rule[va]):.3f}  (rule class shares: {pd.Series(rule).value_counts(normalize=True).round(2).to_dict()})"
)


def run(name, X):
    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X[tr]), y[tr])
    gb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4, random_state=0
    ).fit(X[tr], y[tr])
    for mname, m, Xv in [("logistic", lr, sc.transform(X[va])), ("GBM", gb, X[va])]:
        p = m.predict(Xv)
        pr = m.predict_proba(Xv)
        print(
            f"  {name:28s} {mname:9s} acc {accuracy_score(y[va], p):.3f}  macro-F1 {f1_score(y[va], p, average='macro'):.3f}  log-loss {log_loss(y[va], pr, labels=m.classes_):.3f}"
        )


print("\nsignature from the two records (validation = unseen DJs):")
run("pair features + genre", X_pair)
run("pair features + effnet A,B", np.column_stack([X_pair, ea, eb]))
# DJ as input: only meaningful within DJs, so a second split: random 20 % of each DJ's seams
rng = np.random.default_rng(0)
within = rng.random(len(s)) < 0.8
dj = pd.get_dummies(s.dj).to_numpy(float)


def run_within(name, X):
    sc = StandardScaler().fit(X[within])
    lr = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X[within]), y[within])
    gb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4, random_state=0
    ).fit(X[within], y[within])
    for mname, m, Xv in [("logistic", lr, sc.transform(X[~within])), ("GBM", gb, X[~within])]:
        p = m.predict(Xv)
        print(
            f"  {name:28s} {mname:9s} acc {accuracy_score(y[~within], p):.3f}  macro-F1 {f1_score(y[~within], p, average='macro'):.3f}"
        )


majw = pd.Series(y[within]).mode()[0]
djmaj = pd.Series(y[within]).groupby(s.dj.values[within]).agg(lambda v: v.mode()[0])
print(
    f"\nwithin-DJ split (seams of the same DJs on both sides): majority {accuracy_score(y[~within], [majw] * (~within).sum()):.3f}, per-DJ majority {accuracy_score(y[~within], pd.Series(s.dj.values[~within]).map(djmaj).fillna(majw).values):.3f}"
)
run_within("pair features + genre", X_pair)
run_within("pair features + DJ one-hot", np.column_stack([X_pair, dj]))
# overlap length
ob_ = s.overlap_bars.to_numpy(float)
ok = np.isfinite(ob_) & (ob_ > 0) & (s.signature != "cut").to_numpy()
print(f"\noverlap bars (non-cut seams, n={ok.sum()}): median {np.nanmedian(ob_[ok]):.0f}")
for name, X in [
    ("pair features + genre", X_pair),
    ("pair features + DJ one-hot", np.column_stack([X_pair, dj])),
]:
    m = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=4, random_state=0
    ).fit(X[ok & within], np.log1p(ob_[ok & within]))
    print(
        f"  {name:28s} GBM R² on within-DJ held-out: {r2_score(np.log1p(ob_[ok & ~within]), m.predict(X[ok & ~within])):.3f}"
    )
m = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.05, max_depth=4, random_state=0
).fit(X_pair[ok & tr], np.log1p(ob_[ok & tr]))
print(
    f"  pair features + genre        GBM R² on unseen DJs: {r2_score(np.log1p(ob_[ok & va]), m.predict(X_pair[ok & va])):.3f}"
)
