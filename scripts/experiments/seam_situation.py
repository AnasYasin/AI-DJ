"""
Does the situation at the seam (how record A ends where the DJ left it, how record B starts where the
DJ brought it in) decide the transition type? Whole-record features do not (signature_learnability.py).

    PYTHONPATH=. python scripts/experiments/seam_situation.py curves  [--procs 6]   # per-track level curves, resumable
    PYTHONPATH=. python scripts/experiments/seam_situation.py features              # per-seam situation features
    PYTHONPATH=. python scripts/experiments/seam_situation.py test                  # the pattern and the learnability test

Curves: for every record of a measured seam, RMS, bass RMS (< 150 Hz) and onset strength on a 0.25 s
hop, cached in data/interim/situation/<source_ref>.npz. Features: those curves read around the
seam's two cue points in record time ((mix_t - time_zero_s) * rate, as the pipeline does), relative to
the record's own median. Test: shares of the transition type per situation bin, then the same
logistic / GBM test as signature_learnability.py on the dataset's DJ split.
"""

import argparse
import logging
from multiprocessing import Pool
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from src.data.dataset import load_seams, load_tracks

log = logging.getLogger("situation")
SR = 22050
HOP_S = 0.25
CURVES = Path("data/interim/situation")
TRACKS = {"raveform": Path("data/djdata/raveform/tracks"), "djs": Path("data/djdata/djs/tracks")}
SIGNATURES = ["cut", "centre swap", "late swap", "end swap", "tension"]


def seams_and_tracks():
    t = load_tracks().set_index("track_id")
    s = load_seams()
    s = s[(s.measured == 1) & s.signature.isin(SIGNATURES + ["loop"])]
    s = s[s.track_a.isin(t.index) & s.track_b.isin(t.index)].copy()
    return s, t


def track_file(t, track_id) -> Path | None:
    row = t.loc[track_id]
    hits = list(TRACKS[row.source].glob(f"{row.source_ref}.*"))
    return hits[0] if hits else None


def decode(path: Path) -> np.ndarray:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(out, dtype=np.float32)


def curves(path: Path) -> dict:
    import librosa
    from scipy.signal import butter, sosfilt

    y = decode(path)
    hop = int(HOP_S * SR)
    n = len(y) // hop

    def level(sig):
        return 20 * np.log10(np.sqrt((sig[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12))

    low = sosfilt(butter(2, 150, "low", fs=SR, output="sos"), y).astype(np.float32)
    mid = sosfilt(butter(2, [150, 2000], "band", fs=SR, output="sos"), y).astype(np.float32)
    high = sosfilt(butter(2, 2000, "high", fs=SR, output="sos"), y).astype(np.float32)
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=512)
    # onset is on a 512-sample hop; average it onto the 0.25 s frames
    per = hop // 512
    onset = (
        onset[: n * per].reshape(n, per).mean(1) if len(onset) >= n * per else np.resize(onset, n)
    )
    return {
        "rms_db": level(y),
        "low_db": level(low),
        "mid_db": level(mid),
        "high_db": level(high),
        "onset": onset.astype(np.float32),
    }


def _one(args):
    ref, path = args
    dest = CURVES / f"{ref}.v2.npz"
    if dest.exists():
        return ref, "cached"
    try:
        np.savez(dest, **curves(path))
        return ref, "ok"
    except Exception as e:  # a bad file must not stop the run
        return ref, f"failed: {e}"


def cmd_curves(procs: int) -> None:
    s, t = seams_and_tracks()
    CURVES.mkdir(parents=True, exist_ok=True)
    ids = pd.unique(np.concatenate([s.track_a.values, s.track_b.values]))
    jobs, missing = [], 0
    for tid in ids:
        p = track_file(t, tid)
        if p is None:
            missing += 1
            continue
        jobs.append((t.loc[tid].source_ref, p))
    log.info("%d records, %d files found, %d missing", len(ids), len(jobs), missing)
    done = 0
    with Pool(procs) as pool:
        for ref, status in pool.imap_unordered(_one, jobs, chunksize=4):
            done += 1
            if status.startswith("failed"):
                log.warning("%s %s", ref, status)
            if done % 200 == 0:
                log.info("%d / %d", done, len(jobs))
    log.info("curves done: %d", done)


def window(c: np.ndarray, t0: float, t1: float) -> np.ndarray:
    i0, i1 = max(int(t0 / HOP_S), 0), min(int(t1 / HOP_S), len(c))
    return c[i0:i1] if i1 > i0 else np.array([np.nan])


def situation(c: dict, t_cue: float, bar: float, side: str) -> dict:
    """The record around a cue point, relative to its own median. side 'out' looks back from the
    cue (record A leaving), 'in' looks forward (record B entering). Per band and overall: the level
    over the 8 bars at the cue, the slope from the 8 bars before those (a build-up or a fall), the
    build over 32 bars, how repetitive the 8 bars are (correlation of the loudness curve with the
    previous 8 bars), the mids against lows and highs, position in the record, bars to or since the
    record's full level."""
    dur = len(c["rms_db"]) * HOP_S
    med = {k: np.nanmedian(v) for k, v in c.items()}
    sign = -1 if side == "out" else 1

    def span(b0, b1):  # bars from the cue, positive away from it on this side
        a, b = sorted((t_cue + sign * b0 * bar, t_cue + sign * b1 * bar))
        return a, b

    f = {}
    for k in ("rms_db", "low_db", "mid_db", "high_db", "onset"):
        n8 = np.nanmean(window(c[k], *span(0, 8))) - med[k]
        f8 = np.nanmean(window(c[k], *span(8, 16))) - med[k]
        f32 = np.nanmean(window(c[k], *span(24, 32))) - med[k]
        f[f"{side}_{k}_8"] = n8
        f[f"{side}_{k}_slope"] = (n8 - f8) * sign  # out: rising into the cue; in: falling after it
        f[f"{side}_{k}_build"] = (n8 - f32) * sign
    a, b = window(c["rms_db"], *span(0, 8)), window(c["rms_db"], *span(8, 16))
    m = min(len(a), len(b))
    f[f"{side}_repeat"] = (
        float(np.corrcoef(a[:m], b[:m])[0, 1])
        if m > 4 and np.std(a[:m]) > 0 and np.std(b[:m]) > 0
        else np.nan
    )
    f[f"{side}_mids"] = (
        f[f"{side}_mid_db_8"] - (f[f"{side}_low_db_8"] + f[f"{side}_high_db_8"]) / 2
    )
    f[f"{side}_pos"] = t_cue / dur if dur else np.nan
    f[f"{side}_left_s"] = (dur - t_cue) if side == "out" else t_cue
    # bars to the next passage at or above the record's median level (a drop or body proxy)
    body = window(c["rms_db"], *span(0, 64)) - med["rms_db"]
    if side == "out":
        body = body[::-1]
    hi = np.flatnonzero(body >= 0)
    f[f"{side}_bars_to_body"] = hi[0] * HOP_S / bar if len(hi) else 64.0
    return f


def cmd_features() -> None:
    s, t = seams_and_tracks()
    rows, skipped = [], 0
    for _, r in s.iterrows():
        ca, cb = (
            CURVES / f"{t.loc[r.track_a].source_ref}.v2.npz",
            CURVES / f"{t.loc[r.track_b].source_ref}.v2.npz",
        )
        if not (ca.exists() and cb.exists()) or not np.isfinite(r.bar_s):
            skipped += 1
            continue
        A, B = dict(np.load(ca)), dict(np.load(cb))
        t_out = (r.a_last_heard_s - r.a_time_zero_s) * r.a_rate
        t_in = (r.b_first_heard_s - r.b_time_zero_s) * r.b_rate
        f = {
            "seam_id": r.seam_id,
            "dj": r.dj,
            "genre": r.genre,
            "signature": r.signature,
            "split": r.split,
        }
        f.update(situation(A, t_out, r.bar_s * r.a_rate, "out"))
        f.update(situation(B, t_in, r.bar_s * r.b_rate, "in"))
        f["out_in_rms_gap"] = f["in_rms_db_8"] - f["out_rms_db_8"]
        f["out_in_bass_gap"] = f["in_low_db_8"] - f["out_low_db_8"]
        rows.append(f)
    df = pd.DataFrame(rows)
    df.to_parquet("reports/seam_situation_features.parquet")
    log.info(
        "features for %d seams (%d skipped) → reports/seam_situation_features.parquet",
        len(df),
        skipped,
    )


def signal_check(df: pd.DataFrame) -> None:
    """Is there a signal, or do all records look alike at the cue and the cells are chance?"""
    from scipy.stats import chi2_contingency

    print("\nSIGNAL CHECK")
    print(
        "1. spread of the situation at the cue (dB relative to the record's own median; p10 / p50 / p90):"
    )
    for c in (
        "out_rms_db_8",
        "out_low_db_8",
        "out_mid_db_8",
        "out_high_db_8",
        "out_rms_db_build",
        "out_repeat",
        "in_rms_db_8",
        "in_low_db_8",
        "in_bars_to_body",
    ):
        q = df[c].quantile([0.1, 0.5, 0.9]).round(1).tolist()
        print(
            f"   {c:18s} {q[0]:6.1f} / {q[1]:6.1f} / {q[2]:6.1f}   share quiet (< -3 dB): {(df[c] < -3).mean():.2f}"
        )
    df = df.copy()
    df["cell"] = df.A_exit + ", " + df.B_entry + ", " + df.A_bass + ", " + df.B_bass
    rng = np.random.default_rng(0)

    def chi(frame, label):
        ct = pd.crosstab(frame.cell, frame.signature)
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            print(f"   {label:28s} too few cells")
            return
        chi2, p, dof, _ = chi2_contingency(ct)
        v = np.sqrt(chi2 / (len(frame) * (min(ct.shape) - 1)))
        sh = frame.assign(signature=rng.permutation(frame.signature.values))
        chi2s, ps, _, _ = chi2_contingency(pd.crosstab(sh.cell, sh.signature))
        print(
            f"   {label:28s} n {len(frame):5d}  cells {ct.shape[0]:2d}  Cramér's V {v:.2f}  p {p:.4f}   (types shuffled: V {np.sqrt(chi2s / (len(frame) * (min(ct.shape) - 1))):.2f}  p {ps:.2f})"
        )

    print(
        "2. does the type depend on the situation cell (chi-square; a shuffled copy shows chance):"
    )
    chi(df, "all seams")
    for dj, g in df.groupby("dj"):
        if len(g) >= 80:
            chi(g, f"within {dj}")
    print("3. per-cell confidence (n, top type, its share, lead over the second):")
    rows = []
    for cell, g in df.groupby("cell"):
        sh = g.signature.value_counts(normalize=True)
        if len(sh) == 1:
            rows.append((cell, len(g), sh.index[0], round(sh.iloc[0], 2), round(sh.iloc[0], 2)))
        else:
            rows.append(
                (
                    cell,
                    len(g),
                    sh.index[0],
                    round(sh.iloc[0], 2),
                    round(sh.iloc[0] - sh.iloc[1], 2),
                )
            )
    print(
        pd.DataFrame(rows, columns=["cell", "n", "top", "share", "lead"])
        .sort_values("n", ascending=False)
        .to_string(index=False)
    )
    print("   overall shares:", df.signature.value_counts(normalize=True).round(2).to_dict())


def cmd_test(genre: str | None = None) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, log_loss
    from sklearn.preprocessing import StandardScaler

    df = pd.read_parquet("reports/seam_situation_features.parquet")
    df = df[df.signature.isin(SIGNATURES)].copy()
    if genre:
        df = df[df.genre == genre].copy()
    feats = [c for c in df.columns if c.startswith(("out_", "in_"))]
    print(
        f"{len(df)} seams, {len(feats)} situation features; classes {df.signature.value_counts().to_dict()}"
    )
    # the pattern, readable: A leaving from a quiet or loud passage, B entering quiet or loud
    df["A_exit"] = np.where(df.out_rms_db_8 < -3, "A quiet", "A loud")
    df["B_entry"] = np.where(df.in_rms_db_8 < -3, "B quiet", "B loud")
    df["A_bass"] = np.where(df.out_low_db_8 < -6, "A no bass", "A bass")
    df["B_bass"] = np.where(df.in_low_db_8 < -6, "B no bass", "B bass")
    df["A_build"] = np.where(
        df.out_rms_db_build > 2,
        "A building",
        np.where(df.out_rms_db_build < -2, "A falling", "A flat"),
    )
    df["A_mids"] = np.where(df.out_mids < -3, "A mids down", "A mids in")
    df["A_repeat"] = np.where(df.out_repeat > 0.7, "A repetitive", "A varying")
    print("\ntransition type share by level at the cue points (rows = situation):")
    print(
        pd.crosstab([df.A_exit, df.B_entry], df.signature, normalize="index")
        .round(2)
        .assign(n=df.groupby([df.A_exit, df.B_entry]).size())
        .to_string()
    )
    for cols, title in [
        (
            ["A_build", "B_entry"],
            "by A building, falling or flat into the cue, and B's entry level",
        ),
        (["A_mids", "A_repeat"], "by A's mids and repetitiveness at the cue"),
    ]:
        print(f"\n{title}:")
        print(
            pd.crosstab([df[c] for c in cols], df.signature, normalize="index")
            .round(2)
            .assign(n=df.groupby(cols).size())
            .to_string()
        )
    print("\nby bass at the cue points:")
    print(
        pd.crosstab([df.A_bass, df.B_bass], df.signature, normalize="index")
        .round(2)
        .assign(n=df.groupby([df.A_bass, df.B_bass]).size())
        .to_string()
    )
    signal_check(df)
    print("\nmedian situation per transition type:")
    print(
        df.groupby("signature")[
            [
                "out_rms_db_8",
                "out_low_db_8",
                "out_mid_db_8",
                "out_high_db_8",
                "out_rms_db_build",
                "out_repeat",
                "out_mids",
                "out_pos",
                "in_rms_db_8",
                "in_low_db_8",
                "in_mid_db_8",
                "in_rms_db_build",
                "in_repeat",
                "in_pos",
                "in_bars_to_body",
            ]
        ]
        .median()
        .round(2)
        .to_string()
    )
    X = np.nan_to_num(df[feats].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    y = df.signature.values
    tr = (df.split == "train").to_numpy()
    va = ~tr
    maj = pd.Series(y[tr]).mode()[0]
    print(
        f"\nunseen DJs ({va.sum()} seams): majority ({maj}) {accuracy_score(y[va], [maj] * va.sum()):.3f}"
    )
    genre = pd.get_dummies(df.genre).to_numpy(float)
    for name, Xn in [("situation", X), ("situation + genre", np.column_stack([X, genre]))]:
        sc = StandardScaler().fit(Xn[tr])
        lr = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(Xn[tr]), y[tr])
        gb = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=4, random_state=0
        ).fit(Xn[tr], y[tr])
        for mname, m, Xv in [("logistic", lr, sc.transform(Xn[va])), ("GBM", gb, Xn[va])]:
            p = m.predict(Xv)
            print(
                f"  {name:20s} {mname:9s} acc {accuracy_score(y[va], p):.3f}  macro-F1 {f1_score(y[va], p, average='macro'):.3f}  log-loss {log_loss(y[va], m.predict_proba(Xv), labels=m.classes_):.3f}  top-3 hit {np.mean([yy in m.classes_[np.argsort(pr)[-3:]] for yy, pr in zip(y[va], m.predict_proba(Xv))]):.3f}"
            )
    rng = np.random.default_rng(0)
    within = rng.random(len(df)) < 0.8
    dj = pd.get_dummies(df.dj).to_numpy(float)
    djmaj = pd.Series(y[within]).groupby(df.dj.values[within]).agg(lambda v: v.mode()[0])
    pm = pd.Series(df.dj.values[~within]).map(djmaj).fillna(maj).values
    print(
        f"\nwithin DJs ({(~within).sum()} seams): per-DJ majority {accuracy_score(y[~within], pm):.3f}"
    )
    for name, Xn in [
        ("situation + DJ", np.column_stack([X, dj])),
        ("genre + DJ (no situation)", np.column_stack([genre, dj])),
    ]:
        gb = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=4, random_state=0
        ).fit(Xn[within], y[within])
        p = gb.predict(Xn[~within])
        print(
            f"  {name:26s} GBM       acc {accuracy_score(y[~within], p):.3f}  macro-F1 {f1_score(y[~within], p, average='macro'):.3f}"
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["curves", "features", "test"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--genre", default=None, help="test on one genre only")
    a = ap.parse_args()
    {
        "curves": lambda: cmd_curves(a.procs),
        "features": cmd_features,
        "test": lambda: cmd_test(a.genre),
    }[a.cmd]()
    sys.exit(0)
