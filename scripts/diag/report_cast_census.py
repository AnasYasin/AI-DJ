import importlib.util
import re
import warnings

warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("b", "notebooks/build_dj_data_report.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
g = {}
exec(b.SETUP, g)
exec(b.LOAD, g)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 100)
old, rv, dj, mixes, rv_mixes, canonical = (
    g["old"],
    g["rv"],
    g["dj"],
    g["mixes"],
    g["rv_mixes"],
    g["canonical"],
)
LABELS = ["cut", "centre swap", "late swap", "end swap", "tension", "loop"]
ok = rv[rv.signature.isin(LABELS) & rv.bar_s.notna()]
allshare = ok.signature.value_counts(normalize=True) * 100


def key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# plays: one row per listed record per mix, with a record key that works across sources
src = mixes.set_index("mix_id").source
o = old[old.play_type.ne("overlay") if "play_type" in old else slice(None)]
plays_old = pd.DataFrame(
    {
        "mix_id": old.mix_id,
        "dj": old.dj_name,
        "rec": (old.artist_name.fillna("") + " - " + old.track_name.fillna("")),
        "artist": old.artist_name.fillna(""),
    }
)
rows = []
for m in rv.mix_id.unique():
    mm = rv_mixes[m]
    d = canonical(rv[rv.mix_id == m].dj.iloc[0])
    for t in mm["tracklist"]:
        title = re.sub(r"^\[\d+\]\s*", "", t["title"])
        rows.append((m, d, title, title.split(" - ")[0]))
plays_rv = pd.DataFrame(rows, columns=["mix_id", "dj", "rec", "artist"])
plays = pd.concat([plays_old, plays_rv])
plays = plays[plays.mix_id.map(src).notna()]
plays = plays[
    plays.mix_id.map(src) == plays.mix_id.map(src)
]  # keep all; mixes dedupe is by mix_id so a mix appears once per source anyway
plays["k"] = plays.rec.map(key)
plays = plays[plays.k.str.len() > 3]
plays = plays[
    ~plays.rec.str.contains(r"^ID - ID|Unknown|^ID$|\bID\b - \bID\b", case=False, regex=True)
]
plays = plays.drop_duplicates(["mix_id", "k"])
mix_src = plays.mix_id.map(src)
# ---- per DJ census
cast = sorted(
    set(mixes.dj.value_counts().head(25).index)
    | {
        "Scuba",
        "Amoss",
        "Boris",
        "Manuel Le Saux",
        "Fred again..",
        "Claptone",
        "Doc Scott",
        "Silent Dust",
        "Filterheadz",
    }
)
out = []
for d in cast:
    pm = plays[plays.dj == d]
    n_mix = mixes[mixes.dj == d].mix_id.nunique()
    m_with_tl = pm.mix_id.nunique()
    cnt = pm.k.value_counts()
    top = cnt.iloc[0] if len(cnt) else 0
    topname = pm[pm.k == cnt.index[0]].rec.iloc[0][:38] if len(cnt) else ""
    repeat_share = (cnt[cnt >= 2].sum() / len(pm) * 100) if len(pm) else np.nan
    own = (
        (pm.artist.map(key).str.contains(key(d)[:6]) & (key(d)[:6] != "")).mean() * 100
        if len(pm)
        else np.nan
    )
    t = ok[ok.dj == d]
    tdj = dj[dj.dj == d]
    if len(t):
        s = t.signature.value_counts(normalize=True) * 100
        fav = s.idxmax()
        lift = s.max() - allshare[fav]
    else:
        fav, lift = "", np.nan
    genres = mixes[mixes.dj == d].genre7.value_counts(normalize=True)
    set_ = mixes[mixes.dj == d].setting.value_counts(normalize=True)
    out.append(
        dict(
            dj=d,
            mixes=n_mix,
            with_tracklist=m_with_tl,
            records=len(pm),
            distinct=pm.k.nunique(),
            top_record=top,
            top_name=topname,
            repeat_pct=round(repeat_share),
            own_pct=round(own) if own == own else None,
            transitions_rv=len(t),
            transitions_djcorpus=len(tdj),
            fav_move=fav,
            fav_lift_pp=round(lift) if lift == lift else None,
            genre=f"{genres.index[0]} {genres.iloc[0] * 100:.0f}%",
            club_pct=round(set_.get("club", 0) * 100),
            radio_pct=round(set_.get("radio", 0) * 100),
        )
    )
df = pd.DataFrame(out).sort_values("mixes", ascending=False)
print(df.to_string(index=False))
