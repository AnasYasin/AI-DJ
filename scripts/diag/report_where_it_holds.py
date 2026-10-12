import importlib.util
import sys

sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("b", "notebooks/build_dj_data_report.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
g = {}
exec(b.SETUP, g)
exec(b.LOAD, g)
import pandas as pd  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
rv = g["rv"]
GENRES = g["GENRES"]
LABELS = ["cut", "centre swap", "late swap", "end swap", "tension", "loop"]
ok = rv[rv.signature.isin(LABELS) & rv.bar_s.notna()].copy()
ok["year"] = ok.year
print("ok", len(ok), "settings", ok.setting.value_counts().to_dict())


def shares(df, by):
    s = (
        df.groupby([by, "signature"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LABELS, fill_value=0)
    )
    n = s.sum(axis=1)
    return (s.div(n, axis=0) * 100).round(0).assign(n=n)


def eta2(df, by):
    """share of variance of each signature indicator explained by groups (between / total)."""
    out = {}
    for lab in LABELS:
        y = (df.signature == lab).astype(float)
        tot = ((y - y.mean()) ** 2).sum()
        grp = df.groupby(by)
        between = (
            grp.size() * (grp.apply(lambda d: (d.signature == lab).mean()) - y.mean()) ** 2
        ).sum()
        out[lab] = round(between / tot * 100, 1) if tot else 0
    return out


sub = ok[ok.genre7.isin(GENRES)]
print("\n== by genre\n", shares(sub, "genre7"))
print("\n== by setting (all genres)\n", shares(ok[ok.setting != "unknown"], "setting"))
for gname in ["techno", "tech house", "house", "trance", "drum and bass"]:
    d = sub[(sub.genre7 == gname) & (sub.setting != "unknown")]
    print(f"\n== {gname} by setting\n", shares(d, "setting"))
panel = ok.groupby("dj").size().pipe(lambda c: c[c >= 50]).index
pd_ = ok[ok.dj.isin(panel)]
print("\n== by DJ (50+)\n", shares(pd_, "dj").sort_values("centre swap"))
print("\n== eta2 %: genre", eta2(sub, "genre7"))
print("== eta2 %: setting", eta2(ok[ok.setting != "unknown"], "setting"))
print("== eta2 %: dj (panel)", eta2(pd_, "dj"))
print("== eta2 %: mix (panel)", eta2(pd_, "mix_id"))
print("== eta2 %: year", eta2(ok, "year"))
# continuous moves: do they separate better?
cols = [
    c
    for c in [
        "overlap_bars",
        "swap_pos",
        "bass_both_bars",
        "bass_cut_bars",
        "b_low_pos",
        "b_high_pos",
        "a_low_pos",
        "a_high_pos",
    ]
    if c in ok.columns
]


def eta2c(df, by, c):
    y = df[c].astype(float)
    m = y.notna()
    y = y[m]
    d = df[m]
    tot = ((y - y.mean()) ** 2).sum()
    grp = y.groupby(d[by])
    return round((grp.size() * (grp.mean() - y.mean()) ** 2).sum() / tot * 100, 1)


print("\n== eta2 % of continuous moves")
for c in cols:
    print(
        f"{c:16s} genre {eta2c(sub, 'genre7', c):5} setting {eta2c(ok[ok.setting != 'unknown'], 'setting', c):5} dj {eta2c(pd_, 'dj', c):5} mix {eta2c(pd_, 'mix_id', c):5}"
    )
print("\n== medians by genre\n", sub.groupby("genre7")[cols].median().round(2))
print(
    "\n== medians by setting\n",
    ok[ok.setting != "unknown"].groupby("setting")[cols].median().round(2),
)
print(
    "\n== medians by DJ\n", pd_.groupby("dj")[cols].median().round(2).sort_values("overlap_bars")
)
