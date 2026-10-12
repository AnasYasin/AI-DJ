"""Writes notebooks/dj_data_report.ipynb cell by cell, then executes it so the charts render on GitHub.

    python notebooks/build_dj_data_report.py          # write + execute
    python notebooks/build_dj_data_report.py --no-run # write only

The notebook treats the three sources as one dataset (split only by genre, setting and DJ), reads the
live tables under data/djdata/<corpus>/out/ and data/processed/, and stamps every table it read (rows,
md5) into data/dataset/findings_provenance.json. Chapters are added here one at a time.
"""

import argparse
import subprocess
from pathlib import Path

import nbformat as nbf

NB = Path("notebooks/dj_data_report.ipynb")
HTML = Path("reports/dj_data_report.html")   # static copy to keep open in a browser and refresh

SETUP = r'''
import hashlib, json, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt, seaborn as sns
from IPython.display import display, HTML

ROOT = Path("..") if Path.cwd().name == "notebooks" else Path(".")
OUT = ROOT / "data/dataset"; OUT.mkdir(parents=True, exist_ok=True)

# one style, like the architecture diagrams: a dark card, monospace, two hues, thin marks, air
CARD, SURF, INK, INK2, INK3, GRID = "#0f1217", "#151a21", "#c9cfd8", "#8b95a5", "#5c6675", "#242a34"
WHITE, CYAN, WARM = "#7f8899", "#4cc9f0", "#f4a261"
SERIES = [CYAN, WARM, "#7bd389", "#f2cc60", "#e07ba0", "#9b8cf0", "#f06a6a", "#5fb3b3"]
A_COLOR, B_COLOR = WHITE, CYAN            # outgoing track, incoming track
sns.set_theme(style="white", font="DejaVu Sans Mono")
mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "figure.figsize": (9, 3.8),
    "figure.facecolor": CARD, "axes.facecolor": "none", "savefig.facecolor": CARD,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.spines.bottom": False,
    "axes.grid": False, "axes.titlesize": 9.5, "axes.titleweight": "normal", "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 8, "legend.fontsize": 8,
    "text.color": INK2, "axes.labelcolor": INK3, "xtick.color": INK3, "ytick.color": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False, "axes.titlepad": 14,
    "patch.force_edgecolor": False, "patch.edgecolor": "none", "patch.linewidth": 0,   # no outline on any bar
})
SETTING_COLOR = {"club": CYAN, "radio": WHITE, "unknown": INK3}
GENRES = ["techno", "tech house", "house", "trance", "drum and bass", "melodic house", "afro house"]
DJ_NAMES = {"Amelielens": "Amelie Lens", "Sultanplusshepard": "Sultan + Shepard", "Djtennis": "DJ Tennis",
            "Fredagain..": "Fred again..", "Romanflugel": "Roman Flügel", "Chasestatus": "Chase & Status",
            "Rufusdusol": "RÜFÜS DU SOL", "Benbohmer": "Ben Böhmer"}   # the DJ corpus keys are 1001 slugs

def card(fig, pad=0.0):
    """the rounded card behind every chart, like the diagrams"""
    fig.patches.append(mpl.patches.FancyBboxPatch((0.004, 0.004), 0.992, 0.992, boxstyle="round,pad=0,rounding_size=0.02",
                       transform=fig.transFigure, facecolor=SURF, edgecolor="#1b2129", linewidth=1, zorder=-1))

def guide(ax, x, text, y_text, color=INK3, text_color=INK2):
    """a dotted vertical guide with its name, the way the diagrams mark events"""
    ax.axvline(x, color=color, lw=0.8, ls=(0, (2, 3)), alpha=0.8, zorder=0)
    ax.text(x, y_text, cap(text), ha="center", va="top", fontsize=7, color=text_color, bbox=dict(facecolor=SURF, edgecolor="none", pad=1.5))

DIVERGING = mpl.colors.LinearSegmentedColormap.from_list("card_div", [CYAN, SURF, WARM])
SINGLE = mpl.colors.LinearSegmentedColormap.from_list("card_one", [SURF, CYAN])

def cap(text): return text[:1].upper() + text[1:] if text else text

def header(fig, title, subtitle=None, key=None, x=0.05):
    """the same header on every chart: title, one line under it, then the colour key as coloured words"""
    h = fig.get_figheight()   # the lines sit a fixed distance in inches from the top, whatever the chart's height
    lines = [fig.text(x, 1 - 0.2 / h, cap(title), fontsize=10.5, color=INK, va="top")]
    if subtitle: lines.append(fig.text(x, 1 - 0.4 / h, cap(subtitle), fontsize=8.5, color=INK2, va="top"))
    fig.canvas.draw()
    for t in lines:
        right = t.get_window_extent().transformed(fig.transFigure.inverted()).x1
        assert right < 0.97, f"header line runs off the card: {t.get_text()!r}"
    if key:
        xx = x
        for word, color in key:
            t = fig.text(xx, 1 - 0.6 / h, cap(word), fontsize=8.5, color=color, va="top")
            fig.canvas.draw(); bb = t.get_window_extent().transformed(fig.transFigure.inverted())
            xx = bb.x1 + 0.025

def note(ax, x, y, text, color=INK2, ha="left", size=7.5):
    ax.text(x, y, text, color=color, ha=ha, va="center", fontsize=size)

def genre_of(text):
    """Raveform's tag combinations and the other sources' labels onto the seven genres."""
    t = str(text).lower()
    if "drum" in t or t == "dnb": return "drum and bass"   # the old dataset spells it "drum and base"
    if "trance" in t: return "trance"
    if "tech house" in t: return "tech house"
    if "techno" in t: return "techno"
    if "melodic" in t or "progressive house" in t: return "melodic house"
    if "afro" in t: return "afro house"
    if "house" in t: return "house"
    return "other"

RADIO = re.compile(r"radio|podcast|episode|\bshow\b|sessions|transmission|broadcast|fm\b|di\.fm|mix series|weekly|#\s?\d|\b\d{3}\b|\bep\.?\s?\d", re.I)
CLUB = re.compile(r"@|\blive\b|festival|\bclub\b|boiler room|closing|opening|warehouse|arena|stage|b2b|\bhall\b|beach|rooftop|circus|parade|open air", re.I)

def setting_of(*texts):
    """club or festival, radio or podcast, or unknown, from a mix's title, tags and page url. Agrees with
    the hand-made club flag on 99.5 % of the 1,176 Raveform mixes that have one (2026-10-08)."""
    t = " ".join(str(x) for x in texts if isinstance(x, str))
    c, r = bool(CLUB.search(t)), bool(RADIO.search(t))
    if c and not r: return "club"
    if r and not c: return "radio"
    if c and r: return "club" if "@" in t else "radio"
    return "unknown"

PROVENANCE = {}
def read(path, **kw):
    """Read a table and remember its rows and md5, so every number can be traced to the file it came from."""
    p = ROOT / path
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, **kw)
    PROVENANCE[str(path)] = {"rows": len(df), "md5": hashlib.md5(p.read_bytes()).hexdigest()[:10]}
    return df

FINDINGS = []
def finding(chapter, item, text, number, unit, filt, n):
    FINDINGS.append(dict(chapter=chapter, item=item, finding=text, number=number, unit=unit,
                         filter=filt, n=n, date=pd.Timestamp.today().date().isoformat()))

def save_findings():
    pd.DataFrame(FINDINGS).to_csv(OUT / "findings.csv", index=False)
    (OUT / "findings_provenance.json").write_text(json.dumps(PROVENANCE, indent=1))

def bars(ax, labels, values, color, fmt="{:,.0f}", height=0.36):
    """Horizontal bars, thin, baseline at zero, the value written at the end; no axis, no grid."""
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, height=height)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xticks([])
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.015, yi, fmt.format(v), va="center", fontsize=7.5, color=INK2)
    ax.tick_params(axis="y", length=0)

def show(df):
    """A plain-text table, so it reads the same in every viewer, GitHub and PyCharm included."""
    print(df.to_string())

def tiles(items, width=10):
    """A row of headline numbers: big figure, small label, no axes. items = [(label, value, note), ...]"""
    fig, axes = plt.subplots(1, len(items), figsize=(width, 1.7))
    for ax, (label, value, note) in zip(np.atleast_1d(axes), items):
        ax.axis("off")
        ax.text(0.0, 0.62, value, fontsize=20, fontweight="semibold", color=INK, transform=ax.transAxes)
        ax.text(0.0, 0.30, label, fontsize=9, color=INK2, transform=ax.transAxes, va="top", linespacing=1.3)
        if note: ax.text(0.0, -0.08, note, fontsize=7.5, color=INK2, transform=ax.transAxes, va="top")
    plt.subplots_adjust(wspace=0.15)
    plt.show()
'''

LOAD = r'''
# ---- the three sources, then one dataset --------------------------------------------------------
old = read("data/processed/tracklist_clean.csv")
old["genre7"] = old.genre.map(genre_of)
old["setting"] = [setting_of(t, u.replace("-", " ") if isinstance(u, str) else "") for t, u in zip(old.mix_title, old.url)]

rv_mixes = {m["id"]: m for m in map(json.loads, open(ROOT / "data/raw/raveform/raveform/mixes.jsonl"))}
rv = read("data/djdata/raveform/out/seams_index.csv")
rv = rv[(rv.usable == 1) & (rv.measured == 1)].copy()
rv["genre7"] = rv.genre.map(genre_of)
rv["duration_h"] = rv.mix_id.map(lambda m: rv_mixes[m]["duration"] / 3600)
rv["year"] = rv.mix_id.map(lambda m: int(rv_mixes[m]["title"][:4]))
rv["setting"] = rv.mix_id.map(lambda m: setting_of(rv_mixes[m]["title"], " ".join(t["key"] for t in rv_mixes[m].get("tags", []))))

dj = read("data/djdata/djs/out/seams_index.csv")
dj = dj[(dj.usable == 1) & (dj.measured == 1)].copy()
dj["genre7"] = dj.genre.map(genre_of)
dj["dj"] = dj.dj.map(lambda d: DJ_NAMES.get(d, d))
dj["setting"] = "club"   # radio shows were excluded from this corpus when it was built
dj_mix_len = read("data/djdata/djs/out/mixes.csv").set_index("mix_id")
rv_tracks = read("data/djdata/raveform/out/track_features.parquet")
dj_tracks = read("data/djdata/djs/out/track_features.parquet")

# one DJ, one name: the old dataset and the DJ corpus carry 1001 slugs ("Charlottedewitte"), Raveform
# carries names. The key is the letters alone; the display name is the properly cased form seen in a
# title ("Charlotte de Witte @ ..."), else the slug.
def dj_key(name):
    """letters only, with & and + spelled out the way the 1001 slugs spell them (Aly & Fila -> alyandfila)"""
    return re.sub(r"[^a-z0-9]", "", str(name).lower().replace("&", "and").replace("+", "plus"))
display_name = {}
for name in list(rv.dj.unique()) + list(DJ_NAMES.values()):
    display_name.setdefault(dj_key(name), name)
for slug, name in DJ_NAMES.items():   # a 1001 slug that no title spells out ("Chasestatus")
    display_name[dj_key(slug)] = name
for title in old.mix_title.dropna().unique():
    cand = re.split(r" @ | - ", str(title))[0].strip()
    display_name.setdefault(dj_key(cand), cand)
def canonical(name): return display_name.get(dj_key(name), str(name))
old["dj_name"] = old.dj_name.map(canonical); rv["dj"] = rv.dj.map(canonical); dj["dj"] = dj.dj.map(canonical)

# one dataset: every mix and every measured transition, whatever the source; a mix that sits in two
# sources (the DJ corpus is cut from 1001 sets the old dataset also lists) is counted once
mixes = pd.concat([
    old.drop_duplicates("mix_id")[["mix_id", "dj_name", "genre7", "setting"]].rename(columns={"dj_name": "dj"}).assign(source="old"),
    rv.drop_duplicates("mix_id")[["mix_id", "dj", "genre7", "setting", "duration_h"]].assign(source="raveform"),
    dj.drop_duplicates("mix_id")[["mix_id", "dj", "genre7", "setting"]].assign(
        source="djs", duration_h=lambda d: d.mix_id.map(dj_mix_len.duration_s) / 3600),
], ignore_index=True).drop_duplicates("mix_id", keep="last")
seams = pd.concat([rv.assign(source="raveform"), dj.assign(source="djs")], ignore_index=True)
listed_transitions = int((old.groupby("mix_id").size() - 1).clip(lower=0).sum())
'''

CH1_MD = r'''
# What real DJs do, from the data

A study of real DJ sets: which tracks DJs play, how they move from one to the next, and how much they
repeat themselves. Every number comes from published tracklists or from measurements on the audio of
the mixes. How the measuring was done, and how far to trust each number, is in the appendix.

## 1. The data

Real DJ sets, from published tracklists and from the audio of the mixes themselves. For every set we
know the DJ, the genre, and whether it was a club or festival set or a radio show or podcast. For a
part of them the audio of every track and its exact place in the mix was known, and every transition
in those sets was cut out and measured: when each sound comes in and goes out, when the bass swaps,
how long two tracks play together. Where the sets came from is in the appendix.
'''

CH1_CODE = r'''
# ---- the data in numbers ------------------------------------------------------------------------
hours = mixes.duration_h.sum()
numbers = pd.DataFrame(
    [["DJs", f"{mixes.dj.nunique():,}"],
     ["Mixes", f"{mixes.mix_id.nunique():,}"],
     ["Transitions listed in tracklists", f"{listed_transitions + len(seams):,}"],
     ["Transitions measured on the audio", f"{len(seams):,}"],
     ["Distinct tracks", f"{old.track_id.nunique() + len(rv_tracks) + len(dj_tracks):,}"],
     ["Hours of mix audio measured", f"{hours:,.0f}"],
     ["Years", f"{rv.year.min()} to 2026"]],
    columns=["", "Count"])
display(numbers.style.hide(axis="index").set_properties(**{"text-align": "left"}).set_table_styles(
    [{"selector": "th", "props": "text-align: left; font-weight: 600;"}]))
by_setting = mixes.setting.value_counts().reindex(["club", "radio", "unknown"]).fillna(0).astype(int)

finding(1, 1, f"{mixes.dj.nunique():,} DJs, {mixes.mix_id.nunique():,} mixes, {len(seams):,} transitions measured on the audio",
        len(seams), "transitions", "usable, measured", int(mixes.mix_id.nunique()))
finding(1, 2, f"{by_setting['radio']:,} radio or podcast sets and {by_setting['club']:,} club or festival sets; {by_setting['unknown']:,} could not be placed from the title",
        int(by_setting["unknown"]), "mixes of unknown setting", "all mixes", int(len(mixes)))
'''

CH1_CHARTS = r'''
# ---- mixes per genre, club against radio; mixes per DJ, all DJs ----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.9)); card(fig)
fig.subplots_adjust(left=0.14, right=0.97, top=0.76, bottom=0.05)
header(fig, "mixes per genre, and the twenty DJs with the most sets", "every set in the data, split by where it was played",
       [("club or festival", CYAN), ("radio or podcast", WHITE), ("unknown", INK3)])

per_genre = (mixes.groupby(["genre7", "setting"]).size().unstack(fill_value=0)
                .reindex(GENRES).reindex(columns=["club", "radio", "unknown"], fill_value=0))
ax = axes[0]; y = np.arange(len(GENRES)); h = 0.22
for k, (col, color) in enumerate([("club", CYAN), ("radio", WHITE), ("unknown", INK3)]):
    ax.barh(y + (k - 1) * (h + 0.04), per_genre[col], height=h, color=color)
    for yi, v in zip(y, per_genre[col]):
        if v: ax.text(v + 12, yi + (k - 1) * (h + 0.04), f"{int(v):,}", va="center", fontsize=6.5, color=INK2)
ax.set_yticks(y, [cap(g) for g in GENRES]); ax.invert_yaxis(); ax.set_xticks([]); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, per_genre.values.max() * 1.25)

per_dj = (mixes.groupby("dj").agg(mixes=("mix_id", "nunique"))
            .join(seams.groupby("dj").size().rename("measured")).fillna(0)
            .sort_values("mixes", ascending=False).head(20))
ax = axes[1]
bars(ax, per_dj.index, per_dj.mixes.values, WHITE, height=0.3)
ax.set_xlim(0, per_dj.mixes.max() * 1.18)
plt.show()

median_mixes = int(mixes.groupby("dj").mix_id.nunique().median())
finding(1, 3, f"{per_dj.index[0]} has the most sets in the data, {int(per_dj.mixes.iloc[0]):,}; half of the {mixes.dj.nunique()} DJs have only one",
        int(per_dj.mixes.iloc[0]), "mixes", "all mixes", int(mixes.dj.nunique()))
other = 100 * (mixes.genre7 == "other").mean()
finding(1, 4, f"{100 - other:.0f}% of mixes fall in the seven genres; the rest is 'other' and is not charted",
        round(other, 1), "% of mixes in 'other'", "all mixes", int(len(mixes)))
save_findings()
'''

CH2_MD = r'''
## 2. Transitions

Every measured transition carries what was measured: how long the two tracks played together, where the
bass swapped, when each band of each track arrived or left, whether the bass was held back or a track
looped. From those readings it is given the name of the nearest mixer transition above. How long a transition
is runs through none of this; it is a number of its own.

| transition | what the DJ did |
|---|---|
| slam | the tracks overlap 8 bars or less, a hard switch |
| fade | the bass swaps in the middle of the overlap and the outgoing track eases down |
| melt | the bass swaps in the middle of the overlap with both tracks at one level |
| wave | the bass swaps in the middle of the overlap and both rhythms stay in |
| blend | the bass swaps late, the highs of the incoming track hand over early |
| rise | the bass swaps at the very end of the overlap |
| tension | the mix has no bass at all for 8 bars or more while a track with bass is playing |
| loop | the outgoing track is looped before the change |

The reader hears where the bass swapped and when each band came and went. It cannot hear how loud a band
was, so fade, melt and wave, and rise and drop, are told apart by the nearest mixer recipe, not by a
reading; a drop is counted as a rise. Changes with a gap of more than 2 bars between the tracks (talk,
an edit, a track missing from the tracklist) are not transitions and are left out. Only transitions
whose track placement passed the tempo check are shown (the appendix says how). The DJ charts show a cast
of famous DJs that runs through the whole study; the other DJs sit inside the "All DJs" row.
'''

CH2_TYPES = r'''
# ---- the seven mixer types, each as a rule list and a small picture, read from the mixer's own recipes ----
import sys
sys.path.insert(0, str(ROOT))
from src.audio.audio_mixer import RECIPES, LEAD, DEFAULT_BARS, LONG_STRETCH, SWAP_BARS, _lead_envelopes

import textwrap
def env(points, x):
    """a breakpoint envelope [(fraction, gain), ...] sampled at x; a named crossfade is the mixer's"""
    if points == "xfade_out": return np.cos(x * np.pi / 2)
    if points == "xfade_in": return np.sin(x * np.pi / 2)
    px, py = zip(*points); return np.interp(x, px, py)
def openness(sweep, x):
    """how much of the spectrum a filter lets through, 1 = fully open; the join between two pieces of a
    sweep is eased over a bar so the recipe's breakpoints never show as a bump"""
    if sweep is None: return np.ones_like(x)
    hz = env(sweep["hz"], x); frac = np.log(hz / 20) / np.log(20000 / 20)
    y = 1 - frac if sweep["kind"] == "highpass" else frac
    n = max(len(x) // BARS, 3)
    return np.convolve(np.concatenate([np.full(n, y[0]), y, np.full(n, y[-1])]), np.ones(n) / n, mode="same")[n:-n]
def passes(sweep, x, band):
    """how much of a band a moving filter lets through, 1 = all. Edges in Hz: where the band is untouched / gone."""
    if sweep is None: return np.ones_like(x)
    hz = env(sweep["hz"], x)
    keep, kill = {("highpass", "low"): (60, 250), ("highpass", "mid"): (300, 2000), ("highpass", "high"): (2000, 8000),
                  ("lowpass", "low"): (250, 60), ("lowpass", "mid"): (3000, 600), ("lowpass", "high"): (8000, 2000)}[(sweep["kind"], band)]
    return np.clip((np.log(hz) - np.log(kill)) / (np.log(keep) - np.log(kill)), 0, 1)
def band_curve(r, side, band, x):
    """EQ gain of the band times what the filter lets through, for track A or B"""
    bands = r.get(f"{side}_bands") or {}
    eq = env(bands[band], x) if bands.get(band) else np.ones_like(x)
    return eq * passes(r.get(f"{side}_sweep"), x, band)
def hz(v): return f"{v / 1000:g} kHz" if v >= 1000 else f"{v:g} Hz"
def pct(f): return f"{round(f * 100):d} %"
def db(v): return f"−{abs(v):.0f} dB" if v < 0 else f"{v:.0f} dB"
def sweep_words(sweep, track):
    """one filter sweep as an engineer would log it: kind, track, cutoff from, to, over which part"""
    kind = {"highpass": "High-pass", "lowpass": "Low-pass"}[sweep["kind"]]
    pts = list(sweep["hz"])
    def moves(i): return abs(np.log(pts[i][1] / pts[i - 1][1])) > np.log(1.25)           # under a third of an octave is no move
    first = next(i for i in range(1, len(pts)) if moves(i))                               # the move starts here
    last = max(i for i in range(1, len(pts)) if moves(i))                                 # and ends here
    (f0, h0), (f1, h1) = pts[first - 1], pts[last]
    if f0 == 0 and f1 == 1: span = "across the whole overlap"
    elif f0 == 0: span = f"by {pct(f1)} of the overlap"
    else: span = f"over the last {pct(1 - f0)}"
    return f"{kind} on the {track}, cutoff {hz(h0)} {'down' if h1 < h0 else 'up'} to {hz(h1)} {span}."
def filter_words(r):
    parts = [sweep_words(r["B_sweep"], "incoming") if r.get("B_sweep") else None,
             sweep_words(r["A_sweep"], "outgoing") if r.get("A_sweep") else None]
    return " ".join(w for w in parts if w) or "None."
SWAP_AT = {"fade": 0.5, "melt": 0.5, "wave": 0.5, "rise": 0.85, "blend": 0.75, "drop": 0.925}   # the recipe's low-band swap, snapped by the mixer to one bar on a bar line
def eq_words(t, r):
    if t == "slam": return "None."
    if t == "blend": return "Three-band hand-over, each band to −16 dB: highs over 20 to 40 % of the overlap, mids over 45 to 60 %, lows in a one-bar swap at 75 %."
    if t == "drop": return "The incoming low band is held off for the whole overlap and arrives with the drop. The outgoing low band leaves in a one-bar swap at 92 %; its mids shelve to −14 dB and its highs to −20 dB from 70 %."
    where = {0.5: "at the centre of the overlap", 0.85: "at 85 % of the overlap"}[SWAP_AT[t]]
    return f"One-bar low-band swap {where}, on a bar line."
def level_words(t):
    if t == "slam": return "A hard cut on a bar line."
    c = LEAD[t]
    return (f"The incoming enters over {c['entry_bars']} bars to {db(c['support_db'])} under the outgoing, its mids {db(c['mid_dip_db'])}. "
            f"The levels swap at {pct(c['handover'])} of the overlap across {c['swap_bars']} bars; the outgoing falls away by the end.")
def length_words(t):
    if t == "slam": return "4 bars, a cut, never stretched."
    if t == "drop": return f"{DEFAULT_BARS[t]} bars, pinned to the incoming track's drop, never stretched."
    return f"{DEFAULT_BARS[t]} bars by default; a well-matched pair may run to {int(DEFAULT_BARS[t] * LONG_STRETCH)}."

def level_curves(t, x):
    """(A volume, B volume, A mids, B mids) over the overlap, the mixer's lead handover for a BARS-bar overlap"""
    if t == "slam":
        return env(RECIPES[t]["A_vol"], x), env(RECIPES[t]["B_vol"], x), np.ones_like(x), np.ones_like(x)
    a_vol, b_vol, a_mid, b_mid = _lead_envelopes(BARS, LEAD[t])
    return env(a_vol, x), env(b_vol, x), env(a_mid, x), env(b_mid, x)
def low_curves(t, x):
    """(A lows, B lows): the one-bar swap where the recipe puts it; the drop keeps the incoming lows off"""
    if t == "slam": return np.ones_like(x), np.ones_like(x)
    c, w = SWAP_AT[t], SWAP_BARS / BARS
    a = np.clip((c + w / 2 - x) / w, 0, 1); b = 1 - a
    if t == "drop": b = np.zeros_like(x)
    return a, b
def band_level(t, side, band, x):
    """how much of a band of a track is heard: level, EQ and filter together"""
    r = RECIPES[t]; a_vol, b_vol, a_mid, b_mid = level_curves(t, x); a_low, b_low = low_curves(t, x)
    vol = a_vol if side == "A" else b_vol
    bands_ = r.get(f"{side}_bands") or {}
    if band == "low": eq = a_low if side == "A" else b_low
    elif band == "mid": eq = (a_mid if side == "A" else b_mid) * (env(bands_["mid"], x) if bands_.get("mid") else 1)
    else: eq = env(bands_["high"], x) if bands_.get("high") else np.ones_like(x)
    return vol * eq * passes(r.get(f"{side}_sweep"), x, band)

VOL_COLOR, EQ_COLOR, FILT_COLOR = CYAN, "#f2cc60", "#e07ba0"   # the three controls, as the Spotify picture colours them
BARS, PRE, POST = 16, 4, 4
x = np.linspace(0, 1, 400); bars_x = x * BARS

# the music under each card: real tracks from the laptop, chosen to suit the type (notebooks/card_music.py
# caches every track's 20 ms loudness curve and tempo; this picks two tracks and the sections that fit)
MUSIC = read("data/interim/card_music.parquet")
MUSIC = MUSIC[MUSIC.seconds >= 240].reset_index(drop=True)
STEP_S = 0.02

SPAN = PRE + BARS + POST   # every track is drawn across the whole picture; the parts the DJ does not play are dim

def window(curve, bpm, score, hop_bars=1):
    """the SPAN bars of this track whose overlap part scores highest, as a loudness curve; the track's
    first and last 16 bars are never used, a DJ cues inside a track and leaves it before its end"""
    per_bar = 4 * 60 / bpm / STEP_S
    n = int(SPAN * per_bar); hop = max(int(hop_bars * per_bar), 1); margin = int(16 * per_bar)
    i0, i1 = int(PRE * per_bar), int((PRE + BARS) * per_bar)
    best, best_s = None, -1e9
    for a in range(margin, len(curve) - n - margin, hop):
        w = curve[a:a + n].astype(np.float32); sc = score(w, i0, i1)
        if sc > best_s: best, best_s = w, sc
    return np.convolve(best, np.ones(5) / 5, mode="same")   # 100 ms, the eye's smoothing

steady = lambda w, i0, i1: w[i0:i1].mean() - 2 * w[i0:i1].std()                 # loud and even through the overlap
gentle = lambda w, i0, i1: -w[i0:i1].std() - abs(w[i0:i1].mean() - 0.6)          # soft, slow movement, mid level
build = lambda w, i0, i1: np.polyfit(np.linspace(0, 1, i1 - i0), w[i0:i1], 1)[0] - w[i0:i1].std() * 0.5   # rising through it
def into_drop(w, i0, i1):
    """a breakdown through the overlap that breaks into the drop at its end: the part after the overlap
    loud and still pumping, the overlap itself quieter but not silent"""
    after, during = w[i1:], w[i0 + (i1 - i0) // 4: i1]
    return after.mean() - during.mean() if after.std() > 0.1 and during.mean() > 0.25 else -1.0

CAST = {  # per type: which kind of track for the outgoing and the incoming, and the section that suits
    "slam":  (("dense", steady), ("dense", steady)),
    "rise":  (("groove", steady), ("groove", build)),
    "fade":  (("smooth", gentle), ("smooth", gentle)),
    "melt":  (("smooth", gentle), ("smooth", gentle)),
    "wave":  (("groove", steady), ("groove", steady)),
    "blend": (("groove", steady), ("groove", steady)),
    "drop":  (("groove", steady), ("groove", into_drop)),
}
ranked = MUSIC.sort_values("crest").reset_index(drop=True)
third = len(ranked) // 3
POOL = {"smooth": list(ranked.index[:third]), "groove": list(ranked.index[third:2 * third]), "dense": list(ranked.index[2 * third:])}
rng = np.random.default_rng(7)
for k in POOL: rng.shuffle(POOL[k])
def cast(kind):
    """the next unused track of this kind, so no two cards share music"""
    return ranked.loc[POOL[kind].pop()]

def heard(t, side, x):
    """how loud the track is heard: level, EQ and filter together, lows weighing most"""
    return 0.5 * band_level(t, side, "low", x) + 0.3 * band_level(t, side, "mid", x) + 0.2 * band_level(t, side, "high", x)

def wave(ax, t, e, y0, color, alpha=0.55):
    ax.fill_between(t, y0 + 0.5 - e * 0.45, y0 + 0.5 + e * 0.45, color=color, alpha=alpha, lw=0)

for t in ["slam", "rise", "fade", "melt", "wave", "blend", "drop"]:
    r, lead = RECIPES[t], LEAD.get(t)
    lines = [("Volume", level_words(t)), ("EQ", eq_words(t, r)), ("Filter", filter_words(r)), ("Length", length_words(t))]
    n_lines = sum(len(textwrap.wrap(v, 40)) for _, v in lines) + len(lines)
    height = max(4.0, 1.6 + 0.19 * n_lines)
    fig = plt.figure(figsize=(10, height)); card(fig)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.6], left=0.04, right=0.97, top=1 - 1.0 / height, bottom=0.44 / height, wspace=0.05)
    header(fig, t, None, [("outgoing track", A_COLOR), ("incoming track", B_COLOR), ("volume", VOL_COLOR), ("bass EQ", EQ_COLOR), ("filter", FILT_COLOR)])
    txt = fig.add_subplot(gs[0]); txt.axis("off")
    y = 0.98
    for k, v in lines:
        wrapped = textwrap.wrap(v, 40)
        txt.text(0.0, y, k, fontsize=7.5, color=INK3, va="top", transform=txt.transAxes)
        txt.text(0.2, y, "\n".join(wrapped), fontsize=7.5, color=INK, va="top", transform=txt.transAxes, linespacing=1.4)
        y -= (0.19 * len(wrapped) + 0.13) / (height - 1.44)
    ax = fig.add_subplot(gs[1])
    (kind_a, score_a), (kind_b, score_b) = CAST[t]
    rec_a, rec_b = cast(kind_a), cast(kind_b)
    pulse_a = window(rec_a.curve, rec_a.bpm, score_a); pulse_b = window(rec_b.curve, rec_b.bpm, score_b)
    t_a = np.linspace(-PRE, BARS + POST, len(pulse_a)); t_b = np.linspace(-PRE, BARS + POST, len(pulse_b))
    # the tracks: the outgoing at full before the overlap, shaped by the recipe inside it; the incoming the other way
    shape_a = np.where(t_a < 0, 1.0, np.where(t_a > BARS, 0.0, np.interp(np.clip(t_a, 0, BARS) / BARS, x, heard(t, "A", x))))
    shape_b = np.where(t_b > BARS, 1.0, np.where(t_b < 0, 0.0, np.interp(np.clip(t_b, 0, BARS) / BARS, x, heard(t, "B", x))))
    # the full depth of the recipe, but every change eases over two bars, the way a hand turns a knob
    def mild(shape, t):
        n = max(int(2 * len(t) / (t[-1] - t[0])), 3)   # samples in two bars
        padded = np.concatenate([np.full(n, shape[0]), shape, np.full(n, shape[-1])])
        return np.convolve(padded, np.ones(n) / n, mode="same")[n:-n]
    shape_a, shape_b = mild(shape_a, t_a), mild(shape_b, t_b)
    # what the DJ does not play is still there, dim: the outgoing runs on after the cut, the incoming ran before the cue
    wave(ax, t_a, pulse_a, 1.0, A_COLOR, 0.15); wave(ax, t_b, pulse_b, 0.0, B_COLOR, 0.15)
    wave(ax, t_a, pulse_a * shape_a, 1.0, A_COLOR); wave(ax, t_b, pulse_b * shape_b, 0.0, B_COLOR)
    a_vol, b_vol, _, _ = level_curves(t, x); a_low, b_low = low_curves(t, x)
    for side, y0, vol, low_ in (("A", 1.0, a_vol, a_low), ("B", 0.0, b_vol, b_low)):
        for curve, color in ((vol, VOL_COLOR), (low_, EQ_COLOR), (openness(r.get(f"{side}_sweep"), x), FILT_COLOR)):
            ax.plot(bars_x, y0 + 0.08 + np.clip(curve, 0, 1) * 0.84, color=color, lw=1.5)
    for b in range(-PRE, BARS + POST + 1): ax.axvline(b, color=GRID, lw=0.6, zorder=0)
    ax.axvline(0, color=INK3, lw=0.8, ls=(0, (2, 3))); ax.axvline(BARS, color=INK3, lw=0.8, ls=(0, (2, 3)))
    ax.set_xlim(-PRE, BARS + POST); ax.set_ylim(0, 2); ax.set_yticks([])
    ax.set_xticks([0, BARS // 2, BARS], ["Overlap starts", f"{BARS // 2} bars", "Overlap ends"]); ax.tick_params(axis="x", length=0, pad=4)
    plt.show()
'''

CH2_PICTURE = r'''
# ---- the typed transitions of both corpora, and the cast --------------------------------
# the DJ transition types (djdata types stage); only transitions whose placement passed the tempo check carry one
# the signatures the seam reader can tell apart (djdata types stage); a gap between tracks is not a transition
LABELS = ["cut", "centre swap", "late swap", "end swap", "tension", "loop"]
# the mixer words for each signature: a cut is a slam; a centre swap is a fade, a melt or a wave; a late swap is a
# blend; an end swap is a rise or a drop. The split inside a group is the nearest recipe, not a reading.
LABEL_NAME = {"cut": "slam", "centre swap": "fade, melt or wave", "late swap": "blend", "end swap": "rise or drop",
              "tension": "tension", "loop": "loop"}
ok = seams[seams.signature.isin(LABELS) & seams.bar_s.notna()].copy()
# the cast: the famous DJs that run through every chapter, one owner per move where one exists (draft 2026-10-12,
# Anas picks the names; a DJ needs CAST_MIN typed transitions to appear in a move chart)
DJ_CAST = ["Amelie Lens", "Charlotte de Witte", "Solomun", "Black Coffee", "Claptone", "Adam Beyer", "Armin van Buuren",
           "DJ Tennis", "Fred again.."]
CAST_MIN = 30   # typed transitions a cast DJ needs for a move chart (Charlotte 42, Fred 33; the n is on every row)
'''

CH2_SHARES = r'''
# ---- the move each DJ in the cast makes most, in the mixer's words: the nearest recipe (slam to drop) for every typed
# ---- seam, tension and loop as themselves (no recipe yet); one shaded grid with every DJ as the reference row ----------
sub = ok[ok.genre7.isin(GENRES)]
MOVES = ["slam", "fade", "melt", "wave", "blend", "rise", "tension", "loop"]   # drop reads on 8 seams of 5,184, left out
ok["move"] = np.where(ok.signature.isin(["tension", "loop"]), ok.signature, ok.type)
cast = [d for d in DJ_CAST if (ok.dj == d).sum() >= CAST_MIN]
share = (ok[ok.dj.isin(cast)].groupby("dj").move.value_counts(normalize=True).unstack(fill_value=0)
         .reindex(index=cast, columns=MOVES, fill_value=0) * 100)
share.loc["All DJs"] = ok.move.value_counts(normalize=True).reindex(MOVES, fill_value=0) * 100
rows = ["All DJs"] + cast
grid = share.loc[rows]
n_by = ok.groupby("dj").size(); n_by["All DJs"] = len(ok)

fig, ax = plt.subplots(figsize=(10, 0.36 * len(rows) + 2.8)); card(fig)
fig.subplots_adjust(left=0.27, right=0.97, top=0.76, bottom=0.05)
header(fig, "the transition each DJ uses most",
       "share of each DJ's measured transitions, in percent. Slam to rise is the nearest mixer recipe to what was measured")
ax.imshow(grid.values, cmap=SINGLE, vmin=0, vmax=70, aspect="auto")
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        v = grid.values[i, j]
        if v >= 1: ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color=CARD if v > 40 else INK)
ax.set_xticks(range(len(MOVES)), [cap(m) for m in MOVES], fontsize=7.5)
ax.xaxis.tick_top(); ax.tick_params(axis="x", pad=6)
ax.set_yticks(range(len(rows)), [f"{d}   n={n_by[d]:,}" for d in rows])
ax.set_xticks(np.arange(-0.5, len(MOVES)), minor=True); ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
ax.grid(which="minor", color=SURF, lw=2); ax.tick_params(which="minor", length=0)
ax.axhline(0.5, color=INK3, lw=0.8)
plt.show()

for d in cast:
    m = (share.loc[d] - share.loc["All DJs"]).idxmax(); lift = share.loc[d, m] - share.loc["All DJs", m]
    finding(2, 1, f"{d}: {m} in {share.loc[d, m]:.0f}% of transitions, {lift:+.0f} points over all DJs",
            round(share.loc[d, m], 1), f"% {m}", "typed, placement checked, cast, nearest recipe", int(n_by[d]))
'''

CH2_OVERLAP = r'''
# ---- how long two tracks play together: every transition one dot, per genre (design B, Anas 2026-10-12; no medians) ----
blend = sub[(sub.signature != "cut") & (sub.overlap_bars > 0)]
order = [g for g in GENRES if (blend.genre7 == g).sum() >= 30]
fig, ax = plt.subplots(figsize=(9, 0.45 * len(order) + 2.2)); card(fig)
fig.subplots_adjust(left=0.18, right=0.96, top=0.8, bottom=0.14)
header(fig, "how long two tracks play together", "every measured transition is one dot, in bars. The lines mark 32 and 64 bars")
rng = np.random.default_rng(0)
for i, g in enumerate(order):
    x = blend[blend.genre7 == g].overlap_bars.clip(upper=128).values
    ax.scatter(x, i + rng.uniform(-0.28, 0.28, len(x)), s=4, color=CYAN, alpha=0.18, linewidths=0)
for v in (32, 64): ax.axvline(v, color=INK3, lw=0.7, ls=(0, (2, 3)))
ax.set_yticks(range(len(order)), [cap(g) for g in order]); ax.set_ylim(len(order) - 0.5, -0.6); ax.tick_params(axis="y", length=0)
ax.set_xlim(-1, 132); ax.set_xticks([0, 16, 32, 64, 96, 128], ["0", "16", "32", "64", "96", "128+"]); ax.set_xlabel("Bars")
plt.show()

longest = blend.sort_values("overlap_bars", ascending=False).head(5)[["dj", "genre7", "overlap_bars", "overlap_s"]]
longest.columns = ["DJ", "genre", "bars together", "seconds"]
longest["seconds"] = longest["seconds"].round(0).astype(int); longest["bars together"] = longest["bars together"].round(0).astype(int)
for g in order:
    x = blend[blend.genre7 == g].overlap_bars
    finding(2, 3, f"{g}: {(x >= 32).mean()*100:.0f}% of transitions overlap 32 bars or more, {(x >= 64).mean()*100:.0f}% 64 or more",
            round((x >= 32).mean() * 100, 1), "% over 32 bars", "overlapping transitions, measured", int(len(x)))
finding(2, 4, f"the longest blend in the data is {int(longest.iloc[0]['bars together'])} bars by {longest.iloc[0]['DJ']}", int(longest.iloc[0]["bars together"]), "bars", "blends, measured", int(len(blend)))
save_findings()
'''



CH3_MD = r"""
## 3. The tracks

What DJs play, from the tracklists of every set: who plays a track first, how long a track stays in DJ sets,
the favourite tracks of each genre, and how much of a set is the DJ's own music. A set's date comes from its
page address or its title. A track is the same track under any spelling of its artist and title. The last chart
scores how much a DJ's sets change from one to the next: the tracks shared between consecutive sets, and the
mix of transitions where both sets were measured, each a distance from 0 to 1, averaged.
"""

CH3_LOAD = r"""
# ---- every play with its set, DJ, genre and date; the track key is the letters of artist and title -----------
def _date_of(url, title):
    for text in (str(url), str(title)):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        if m: return pd.to_datetime(m.group(1), errors="coerce") if m.group(1)[:2] in ("19", "20") else pd.NaT
    return pd.NaT
def _rec_key(artist, title): return re.sub(r"[^a-z0-9]", "", f"{artist}|{title}".lower())
def _clean_title(t):
    t = re.sub(r"^\[\d+(:\d+)?\]\s*", "", str(t)); return re.sub(r"\s*\[[^\]]*\]\s*$", "", t).strip()

po = pd.DataFrame({"mix_id": old.mix_id, "dj": old.dj_name, "genre7": old.genre7, "artist": old.artist_name.fillna(""), "title": old.track_name.fillna(""),
                   "date": [_date_of(u, t) for u, t in zip(old.url, old.mix_title)], "order": old.groupby("mix_id").cumcount()})
# the DJ-corpus sets that the old scrape does not hold, from the fuller scrape of the seven DJs
tl = read("data/interim/tracklist.csv", usecols=["mix_id", "mix_title", "dj_name", "genre", "url", "track_name", "artist_name"], low_memory=False)
tl = tl[tl.mix_id.isin(set(dj.mix_id)) & ~tl.mix_id.isin(set(old.mix_id))]
pdj = pd.DataFrame({"mix_id": tl.mix_id, "dj": tl.dj_name.map(lambda d: DJ_NAMES.get(d, d)).map(canonical), "genre7": tl.genre.map(genre_of), "artist": tl.artist_name.fillna(""), "title": tl.track_name.fillna(""),
                    "date": [_date_of(u, t) for u, t in zip(tl.url, tl.mix_title)], "order": tl.groupby("mix_id").cumcount()})
rows = []
for m, mm in rv_mixes.items():
    if m not in set(rv.mix_id): continue
    d = canonical(rv[rv.mix_id == m].dj.iloc[0]); g = genre_of(" ".join(t["key"] for t in mm.get("tags", [])))
    for i, t in enumerate(mm["tracklist"]):
        title = _clean_title(t["title"]); a, b = (title.split(" - ", 1) + [""])[:2] if " - " in title else ("", title)
        rows.append((m, d, g, a.strip(), b.strip(), pd.to_datetime(mm["title"][:10], errors="coerce"), i, t.get("id")))
prv = pd.DataFrame(rows, columns=["mix_id", "dj", "genre7", "artist", "title", "date", "order", "youtube"])
plays = pd.concat([po, pdj, prv], ignore_index=True)
plays = plays[~plays.title.str.contains(r"^ID$|^\s*$|Unknown", case=False, regex=True) & ~plays.artist.str.contains(r"^ID$|Unknown", case=False, regex=True)]
plays["rec"] = [_rec_key(a, t) for a, t in zip(plays.artist, plays.title)]
plays = plays[plays.rec.str.len() > 4].drop_duplicates(["mix_id", "rec"])
plays["name"] = plays.artist.str.strip() + " - " + plays.title.str.strip()
sets = plays.groupby("mix_id").agg(dj=("dj", "first"), genre7=("genre7", "first"), date=("date", "first"), n=("rec", "size")).reset_index()
pass
"""

CH3_FIRST = r"""
# ---- who plays a track first: of the tracks several DJs play, the share each DJ was first to play ----------
dated = plays[plays.date.notna()]
by_rec = dated.groupby("rec").agg(n_djs=("dj", "nunique"), first_date=("date", "min"))
shared = dated[dated.rec.isin(by_rec[by_rec.n_djs >= 3].index)]
first = shared.sort_values("date").groupby("rec").first()[["dj", "date"]]
last = shared.sort_values("date").groupby("rec").last()[["date"]]
first["months_ahead"] = (shared.groupby("rec").date.apply(lambda d: d.sort_values().iloc[1]) - first.date).dt.days / 30.4
per_dj = shared.drop_duplicates(["dj", "rec"]).groupby("dj").rec.nunique().rename("shared")
firsts = first.groupby("dj").size().rename("first")
tab = pd.concat([per_dj, firsts], axis=1).fillna(0); tab = tab[tab.shared >= 20]
tab["share"] = tab["first"] / tab.shared * 100; tab = tab.sort_values("share", ascending=False)
fig, ax = plt.subplots(figsize=(10, 0.3 * min(len(tab), 20) + 2.4)); card(fig)
fig.subplots_adjust(left=0.3, right=0.9, top=1 - 1.3 / fig.get_figheight(), bottom=0.3 / fig.get_figheight())
header(fig, "who plays a track first", "of the tracks three or more DJs play, the share this DJ played first. DJs with 20 or more such tracks")
top = tab.head(20)
bars(ax, [f"{d}   n={int(v):,}" for d, v in zip(top.index, top.shared)], top.share.values, CYAN, fmt="{:.0f}")
plt.show()
for d, r in top.head(5).iterrows():
    finding(3, 1, f"{d} played {r.share:.0f}% of the shared tracks before anyone else", round(r.share, 1), "% first", "tracks with 3+ DJs, dated sets", int(r.shared))
finding(3, 1, f"{len(first):,} tracks are played by three or more DJs; the first DJ is a median {first.months_ahead.median():.0f} months ahead of the second", round(first.months_ahead.median(), 1), "months", "tracks with 3+ DJs", int(len(first)))
"""

CH3_LIFE = r"""
# ---- how long a track stays in DJ sets: first set to last, per genre, every track one dot ------------------
dated = plays[plays.date.notna()]
life = dated.groupby("rec").agg(first=("date", "min"), last=("date", "max"), n=("mix_id", "nunique"), genre7=("genre7", lambda g: g.mode().iloc[0]))
life = life[life.n >= 2]; life["months"] = (life["last"] - life["first"]).dt.days / 30.4
order = [g for g in GENRES if (life.genre7 == g).sum() >= 30]
fig, ax = plt.subplots(figsize=(9, 0.45 * len(order) + 2.2)); card(fig)
fig.subplots_adjust(left=0.18, right=0.96, top=0.8, bottom=0.14)
header(fig, "how long a track stays in DJ sets", "every track played in two or more sets is one dot, months from its first set to its last")
rng = np.random.default_rng(0)
for i, g in enumerate(order):
    x = life[life.genre7 == g].months.clip(upper=60).values
    ax.scatter(x, i + rng.uniform(-0.28, 0.28, len(x)), s=4, color=CYAN, alpha=0.18, linewidths=0)
for v in (12, 24): ax.axvline(v, color=INK3, lw=0.7, ls=(0, (2, 3)))
ax.set_yticks(range(len(order)), [f"{cap(g)}   n={(life.genre7 == g).sum():,}" for g in order]); ax.set_ylim(len(order) - 0.5, -0.6); ax.tick_params(axis="y", length=0)
ax.set_xlim(-1, 62); ax.set_xticks([0, 12, 24, 36, 48, 60], ["0", "12", "24", "36", "48", "60+"]); ax.set_xlabel("Months")
plt.show()
for g in order:
    x = life[life.genre7 == g].months
    finding(3, 2, f"{g}: {(x >= 12).mean()*100:.0f}% of repeated tracks are still played a year after their first set, {(x >= 24).mean()*100:.0f}% after two", round((x >= 12).mean() * 100, 1), "% over 12 months", "tracks in 2+ dated sets", int(len(x)))
"""

CH3_FAV = r"""
# ---- favourite tracks per genre: the 30 tracks played in the most sets ------------------------------------
for g in GENRES:
    sub = plays[plays.genre7 == g]
    if sub.mix_id.nunique() < 30: continue
    top = (sub.groupby("rec").agg(djs=("dj", "nunique"), sets=("mix_id", "nunique"), name=("name", "first"))
           .sort_values(["sets", "djs"], ascending=False).head(30))
    fig, ax = plt.subplots(figsize=(10, 0.26 * len(top) + 2.2)); card(fig)
    fig.subplots_adjust(left=0.52, right=0.93, top=1 - 1.3 / fig.get_figheight(), bottom=0.3 / fig.get_figheight())
    header(fig, f"favourite tracks, {g}", f"the 30 tracks played in the most {g} sets, of {sub.mix_id.nunique():,} sets by {sub.dj.nunique():,} DJs. The number of DJs who play it is beside the name")
    bars(ax, [(n if len(n) <= 52 else n[:50] + "..") + f"   {d} DJ{'s' if d > 1 else ''}" for n, d in zip(top.name, top.djs)], top.sets.values, CYAN, fmt="{:,.0f}", height=0.5)
    plt.show()
    finding(3, 3, f"{g}: the most played track is {top.name.iloc[0]}, in {int(top.sets.iloc[0])} of {sub.mix_id.nunique():,} sets, by {int(top.djs.iloc[0])} DJs", int(top.sets.iloc[0]), "sets", g, int(sub.mix_id.nunique()))
"""

CH3_OWN = r"""
# ---- how much of a set is the DJ's own music: the cast, plus every DJ with 25 or more sets --------------------
def _is_own(artist, djname):
    k = re.sub(r"[^a-z0-9]", "", str(djname).lower())[:8]
    return len(k) >= 4 and k in re.sub(r"[^a-z0-9]", "", str(artist).lower())
plays["own"] = [_is_own(a, d) for a, d in zip(plays.artist, plays.dj)]
n_sets = plays.groupby("dj").mix_id.nunique()
own = plays.groupby("dj").own.mean().mul(100)
show = own[(n_sets >= 25)].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 0.3 * len(show) + 2.4)); card(fig)
fig.subplots_adjust(left=0.3, right=0.9, top=1 - 1.3 / fig.get_figheight(), bottom=0.3 / fig.get_figheight())
header(fig, "how much of a set is the DJ's own music", "share of a DJ's tracks that are the DJ's own, in percent. DJs with 25 or more sets")
bars(ax, [f"{d}   n={n_sets[d]:,} sets" for d in show.index], show.values, CYAN, fmt="{:.0f}")
plt.show()
for d, v in show.head(5).items():
    finding(3, 4, f"{d}: {v:.0f}% of the tracks in their sets are their own", round(v, 1), "% own", "DJs with 25+ sets", int(n_sets[d]))
"""

CH4_VARY = r"""
# ---- how much a DJ's sets change from one to the next: tracks shared between consecutive sets, and the mix of
# ---- transitions where both sets have it; the 20 most monotonous and the 20 most varied DJs ---------------------
recs_by_set = plays.groupby("mix_id").rec.apply(set)
tsig = ok.groupby("mix_id").move.value_counts(normalize=True).unstack(fill_value=0)
rows = []
for d, grp in sets[sets.date.notna()].sort_values("date").groupby("dj"):
    if len(grp) < 10: continue
    ids = grp.mix_id.tolist(); rec_d, tr_d = [], []
    for a, b in zip(ids, ids[1:]):
        A, B = recs_by_set.get(a, set()), recs_by_set.get(b, set())
        if A and B: rec_d.append(1 - len(A & B) / len(A | B))
        if a in tsig.index and b in tsig.index and ok[ok.mix_id == a].shape[0] >= 5 and ok[ok.mix_id == b].shape[0] >= 5:
            tr_d.append(np.abs(tsig.loc[a] - tsig.loc[b]).sum() / 2)
    parts = {"tracks": np.mean(rec_d) if rec_d else np.nan, "transitions": np.mean(tr_d) if len(tr_d) >= 3 else np.nan}
    rows.append(dict(dj=d, sets=len(grp), **parts, index=np.nanmean(list(parts.values()))))
vary = pd.DataFrame(rows).sort_values("index")
fig, axes = plt.subplots(1, 2, figsize=(10, 0.3 * 20 + 2.6)); card(fig)
fig.subplots_adjust(left=0.22, right=0.97, top=1 - 1.5 / fig.get_figheight(), bottom=0.3 / fig.get_figheight(), wspace=0.9)
header(fig, "how much a DJ's sets change from one to the next", "0 means the next set repeats the last, 1 means nothing returns. DJs with 10 or more dated sets")
for ax, part, title in ((axes[0], vary.head(20), "the 20 most monotonous"), (axes[1], vary.tail(20).iloc[::-1], "the 20 most varied")):
    bars(ax, [f"{d}   n={n:,}" for d, n in zip(part.dj, part.sets)], part["index"].values, CYAN, fmt="{:.2f}")
    ax.set_title(cap(title), loc="left", fontsize=8.5, color=INK2); ax.set_xlim(0, 1.15)
plt.show()
for _, r in pd.concat([vary.head(3), vary.tail(3)]).iterrows():
    finding(4, 5, f"{r.dj}: {r['index']:.2f} of a set changes from one to the next", round(r["index"], 2), "variation index", "DJs with 10+ dated sets", int(r.sets))
save_findings()
"""



CH4_MD = r"""
## 4. The set

How old the music is, whether music has got faster, and where in a set the tempo peaks. Release years come from
Spotify for the tracks it matched, listeners from Last.fm; both files are still filling and the charts say how
many tracks they cover.
"""

CH4_LOAD = r"""
# ---- every play with its played tempo (track BPM times the speed the DJ played it at), its set date and genre, its
# ---- position in the set, and the track's release year and its artist's listeners where known --------------------
def _k(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())
tempo_rows = []
for corpus in ("raveform", "djs"):
    pl = read(f"data/djdata/{corpus}/out/plays.csv", low_memory=False); pl = pl[(pl.is_control == 0) & (pl.found == 1)]
    tf = read(f"data/djdata/{corpus}/out/track_features.parquet")[["track_id", "bpm", "energy_mean"]]
    pl = pl.merge(tf, on="track_id", how="left")
    pl["played_bpm"] = pl.played_bpm if "played_bpm" in pl and pl.played_bpm.notna().any() else pl.bpm * pl.rate
    pl["played_bpm"] = pl.played_bpm.fillna(pl.bpm * pl.rate)
    pl["pos"] = pl.first_heard_s / (pl.mix_minutes * 60)   # where in the set the track is first heard, 0 to 1
    tempo_rows.append(pl[["mix_id", "track_id", "played_bpm", "rate", "energy_mean", "pos"]].assign(corpus=corpus))
tempo = pd.concat(tempo_rows).merge(sets[["mix_id", "dj", "genre7", "date"]], on="mix_id", how="inner")
tempo = tempo[tempo.played_bpm.between(60, 200)]
tempo["year"] = tempo.date.dt.year
pass

sp = read("data/interim/enrich/spotify_records.csv"); sp = sp[sp.hit == 1]
sp["rec"] = sp.k.map(_k); sp["release_year"] = pd.to_numeric(sp.release_date.astype(str).str[:4], errors="coerce")
plays = plays.merge(sp[["rec", "release_year"]].drop_duplicates("rec"), on="rec", how="left")
plays["age"] = plays.date.dt.year - plays.release_year
lf = read("data/interim/enrich/lastfm_artists.csv"); lf = lf[lf.hit == 1]
plays["ak"] = plays.artist.map(lambda a: _k(re.split(r"\s*(&|,|\bx\b|\bvs\.?|\bfeat\.?|\bft\.?|\bpres\.?)\s*", str(a), maxsplit=1, flags=re.I)[0]))
plays = plays.merge(lf[["ak", "listeners"]].drop_duplicates("ak"), on="ak", how="left")
pass
"""

CH4_AGE = r"""
# ---- how old the music is, per genre: every play one dot, years between the track's release and the set ---------
aged = plays[plays.age.notna() & plays.age.between(-1, 40)]
order = [g for g in GENRES if (aged.genre7 == g).sum() >= 30]
fig, ax = plt.subplots(figsize=(9, 0.45 * len(order) + 2.2)); card(fig)
fig.subplots_adjust(left=0.2, right=0.96, top=0.8, bottom=0.14)
header(fig, "how old the music is", f"every play is one dot, years from the track's release to the set. {plays.release_year.notna().sum():,} plays with a known release year so far")
rng = np.random.default_rng(0)
for i, g in enumerate(order):
    x = aged[aged.genre7 == g].age.clip(upper=30).values
    ax.scatter(x, i + rng.uniform(-0.28, 0.28, len(x)), s=5, color=CYAN, alpha=0.25, linewidths=0)
for v in (1, 5, 10): ax.axvline(v, color=INK3, lw=0.7, ls=(0, (2, 3)))
ax.set_yticks(range(len(order)), [f"{cap(g)}   n={(aged.genre7 == g).sum():,}" for g in order]); ax.set_ylim(len(order) - 0.5, -0.6); ax.tick_params(axis="y", length=0)
ax.set_xlim(-1.5, 31); ax.set_xticks([0, 1, 5, 10, 20, 30], ["0", "1", "5", "10", "20", "30+"]); ax.set_xlabel("Years")
plt.show()
for g in order:
    x = aged[aged.genre7 == g].age
    finding(4, 6, f"{g}: {(x <= 1).mean()*100:.0f}% of plays are tracks from the last year, {(x >= 5).mean()*100:.0f}% are five years old or more", round((x <= 1).mean() * 100, 1), "% within a year", "plays with a known release year", int(len(x)))
"""

CH4_TEMPO = r"""
# ---- played tempo by year, one line per genre ---------------------------------------------------------------------
# drum and bass is left out: its measured track tempos sit at 127 to 132 BPM in 2014-15 and 2019-20 and at 172 in
# 2016-18, a 3:4 fold in the readings, not the music (DATASET_STATE.md 2026-10-13); back once the tempos are checked
ty = tempo[tempo.year.between(2012, 2026) & (tempo.genre7 != "drum and bass")].groupby(["genre7", "year"]).played_bpm.agg(["median", "size"]).reset_index()
ty = ty[ty["size"] >= 30]
order = [g for g in GENRES if (ty.genre7 == g).sum() >= 4]
fig, ax = plt.subplots(figsize=(9, 4.2)); card(fig)
fig.subplots_adjust(left=0.1, right=0.86, top=0.8, bottom=0.14)
header(fig, "played tempo by year", "the typical tempo DJs played a genre at, in BPM, in years with 30 or more measured plays")
for i, g in enumerate(order):
    d = ty[ty.genre7 == g].sort_values("year"); ax.plot(d.year, d["median"], color=SERIES[i % len(SERIES)], lw=1.3, alpha=0.9)
    ax.text(d.year.iloc[-1] + 0.2, d["median"].iloc[-1], cap(g), fontsize=7.5, color=SERIES[i % len(SERIES)], va="center")
ax.set_xlabel("Year"); ax.set_ylabel("BPM"); ax.grid(axis="y", color=GRID, lw=0.5)
plt.show()
for g in order:
    d = ty[ty.genre7 == g].sort_values("year")
    if len(d) >= 2: finding(4, 7, f"{g}: played tempo {d['median'].iloc[0]:.0f} BPM in {int(d.year.iloc[0])}, {d['median'].iloc[-1]:.0f} in {int(d.year.iloc[-1])}", round(d['median'].iloc[-1] - d['median'].iloc[0], 1), "BPM change", "measured plays, years with 30+", int(d['size'].sum()))
"""

CH4_PEAK = r"""
# ---- where in the set the tempo peaks: each play's tempo against its own set's opening tempo, by fifth of the set ----
tp = tempo[tempo.pos.between(0, 1)].copy(); tp["fifth"] = np.minimum((tp.pos * 5).astype(int), 4)
opening = tp[tp.fifth == 0].groupby("mix_id").played_bpm.mean().rename("open_bpm")
tp = tp.merge(opening, on="mix_id"); tp["rel"] = tp.played_bpm - tp.open_bpm
pk = tp.groupby(["genre7", "fifth"]).rel.agg(["mean", "size"]).reset_index()
order = [g for g in GENRES if g != "drum and bass" and (pk.genre7 == g).sum() == 5 and pk[pk.genre7 == g]["size"].min() >= 30]
fig, ax = plt.subplots(figsize=(9, 4.2)); card(fig)
fig.subplots_adjust(left=0.1, right=0.86, top=0.8, bottom=0.14)
header(fig, "where in the set the tempo peaks", "played tempo against the set's own opening fifth, in BPM, by fifth of the set")
for g in order:
    d = pk[pk.genre7 == g].sort_values("fifth"); ax.plot(d.fifth, d["mean"], color=CYAN, lw=1.2, alpha=0.8, marker="o", ms=3)
    ax.text(4.08, d["mean"].iloc[-1], cap(g), fontsize=7.5, color=INK2, va="center")
ax.axhline(0, color=INK3, lw=0.7); ax.set_xticks(range(5), ["Start", "", "Middle", "", "End"]); ax.set_ylabel("BPM against the opening"); ax.grid(axis="y", color=GRID, lw=0.5)
plt.show()
for g in order:
    d = pk[pk.genre7 == g].sort_values("fifth"); top = d.loc[d["mean"].idxmax()]
    finding(4, 8, f"{g}: the tempo peaks in fifth {int(top.fifth) + 1} of the set, {top['mean']:+.1f} BPM over the opening", round(top["mean"], 1), "BPM over the opening", "measured plays", int(d["size"].sum()))
"""

CH5_MD = r"""
## 5. The cast

The same DJs through every chart: how known the artists they play are, big names against small, how old their music
is, their favourite tracks, and the tracks only they play.
"""

CH5_FAME = r"""
# ---- how known are the artists each DJ plays: every play one dot, the artist's Last.fm listeners on a log scale ----
cast5 = [d for d in DJ_CAST if (plays.dj == d).sum() >= 50]
fm = plays[plays.listeners.notna() & (plays.listeners > 0)]
fig, ax = plt.subplots(figsize=(9, 0.45 * len(cast5) + 2.2)); card(fig)
fig.subplots_adjust(left=0.24, right=0.96, top=0.8, bottom=0.14)
header(fig, "how known are the artists each DJ plays", f"every play is one dot, the artist's listeners on Last.fm. {fm.ak.nunique():,} artists known so far")
rng = np.random.default_rng(0)
for i, d in enumerate(cast5):
    x = fm[fm.dj == d].listeners.values
    ax.scatter(x, i + rng.uniform(-0.28, 0.28, len(x)), s=5, color=CYAN, alpha=0.25, linewidths=0)
ax.set_xscale("log"); ax.set_xlim(100, 5e6); ax.set_xticks([1e3, 1e4, 1e5, 1e6], ["1 thousand", "10 thousand", "100 thousand", "1 million"])
ax.set_yticks(range(len(cast5)), [f"{d}   n={(fm.dj == d).sum():,}" for d in cast5]); ax.set_ylim(len(cast5) - 0.5, -0.6); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Listeners")
plt.show()
bands = [(0, 1e4, "under 10 thousand"), (1e4, 1e5, "10 to 100 thousand"), (1e5, 1e6, "100 thousand to 1 million"), (1e6, 1e12, "over 1 million")]
grid = pd.DataFrame([[((fm[fm.dj == d].listeners > lo) & (fm[fm.dj == d].listeners <= hi)).mean() * 100 for lo, hi, _ in bands] for d in cast5], index=cast5, columns=[c for _, _, c in bands])
grid.loc["All DJs"] = [((fm.listeners > lo) & (fm.listeners <= hi)).mean() * 100 for lo, hi, _ in bands]
rows = ["All DJs"] + cast5; grid = grid.loc[rows]
fig, ax = plt.subplots(figsize=(9, 0.36 * len(rows) + 2.6)); card(fig)
fig.subplots_adjust(left=0.27, right=0.97, top=0.74, bottom=0.05)
header(fig, "big names against small names", "share of each DJ's plays by how many listeners the artist has, in percent")
ax.imshow(grid.values, cmap=SINGLE, vmin=0, vmax=70, aspect="auto")
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        v = grid.values[i, j]; ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color=CARD if v > 40 else INK)
ax.set_xticks(range(len(bands)), [c.replace(" to ", " to\n").replace("under ", "under\n").replace("over ", "over\n") for _, _, c in bands], fontsize=7.5); ax.xaxis.tick_top(); ax.tick_params(axis="x", pad=6)
ax.set_yticks(range(len(rows)), [f"{d}   n={len(fm) if d == 'All DJs' else (fm.dj == d).sum():,}" for d in rows])
ax.set_xticks(np.arange(-0.5, len(bands)), minor=True); ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
ax.grid(which="minor", color=SURF, lw=2); ax.tick_params(which="minor", length=0); ax.axhline(0.5, color=INK3, lw=0.8)
plt.show()
for d in cast5:
    finding(5, 9, f"{d}: {grid.loc[d].iloc[0]:.0f}% of plays are artists under 10 thousand listeners, {grid.loc[d].iloc[3]:.0f}% over a million", round(grid.loc[d].iloc[0], 1), "% small artists", "plays with a known artist", int((fm.dj == d).sum()))
"""

CH5_AGE = r"""
# ---- how old the music is, per cast DJ --------------------------------------------------------------------------
cast_a = [d for d in DJ_CAST if (aged.dj == d).sum() >= 20]
fig, ax = plt.subplots(figsize=(9, 0.45 * max(len(cast_a), 1) + 2.2)); card(fig)
fig.subplots_adjust(left=0.24, right=0.96, top=0.8, bottom=0.14)
header(fig, "how old the music is, per DJ", "every play is one dot, years from the track's release to the set. DJs with 20 or more known plays")
for i, d in enumerate(cast_a):
    x = aged[aged.dj == d].age.clip(upper=30).values
    ax.scatter(x, i + rng.uniform(-0.28, 0.28, len(x)), s=6, color=CYAN, alpha=0.3, linewidths=0)
for v in (1, 5, 10): ax.axvline(v, color=INK3, lw=0.7, ls=(0, (2, 3)))
ax.set_yticks(range(len(cast_a)), [f"{d}   n={(aged.dj == d).sum():,}" for d in cast_a]); ax.set_ylim(len(cast_a) - 0.5, -0.6); ax.tick_params(axis="y", length=0)
ax.set_xlim(-1.5, 31); ax.set_xticks([0, 1, 5, 10, 20, 30], ["0", "1", "5", "10", "20", "30+"]); ax.set_xlabel("Years")
plt.show()
"""

CH5_FAV = r"""
# ---- the cast's tracks: tracks played in two or more sets, as a table with the link where Raveform has one; and the
# ---- tracks only this DJ plays. A DJ with fewer than five repeated tracks gets a sentence instead of a table ----------
djs_per_rec = plays.groupby("rec").dj.nunique()
for d in DJ_CAST:
    mine = plays[plays.dj == d]
    if mine.mix_id.nunique() < 10: continue
    top = mine.groupby("rec").agg(sets=("mix_id", "nunique"), name=("name", "first"), youtube=("youtube", lambda s: s.dropna().iloc[0] if s.notna().any() else "")).sort_values("sets", ascending=False)
    rep = top[top.sets >= 2].head(30)
    only = mine[mine.rec.map(djs_per_rec) == 1].groupby("rec").agg(sets=("mix_id", "nunique"), name=("name", "first")).sort_values("sets", ascending=False).head(10)
    head = f"<h3 style='margin-top:1.5em'>{d}</h3><p style='color:#8b95a5'>{mine.mix_id.nunique():,} sets, {len(top):,} different tracks.</p>"
    if len(rep) < 5:
        body = f"<p>{d} plays almost every track once: {int((top.sets == 1).sum()):,} of {len(top):,} tracks appear in a single set, so there is no list of favourites.</p>"
    else:
        tbl = rep.assign(link=lambda t: t.youtube.map(lambda y: f'<a href="https://www.youtube.com/watch?v={y}">play</a>' if y else ""))[["name", "sets", "link"]]
        tbl.columns = ["track", "sets", "listen"]
        body = f"<p>Played in more than one set:</p>" + tbl.to_html(escape=False, index=False)
    body += "<p style='color:#8b95a5;margin-top:1em'>No other DJ in the collection plays these: " + ", ".join(f"{n}" for n in only.name) + "</p>"
    display(HTML(head + body))
    if len(rep) >= 5: finding(5, 12, f"{d}: the track most played is {rep.name.iloc[0]}, in {int(rep.sets.iloc[0])} of {mine.mix_id.nunique():,} sets", int(rep.sets.iloc[0]), "sets", d, int(mine.mix_id.nunique()))
save_findings()
"""



F1_MD = r"""
## 2. What DJs play

**DJs do not share their tracks.** We counted every track in 3,876 sets. Nine tracks in ten were played by a single DJ,
and eight in ten appeared in one set only. One track in the whole collection was played by ten or more DJs. There is no
playlist that the DJs of a genre all play from.
"""
F1_SHARED = r"""
# ---- tracks by how many DJs play them ------------------------------------------------------------------------
djs_per_rec = plays.groupby("rec").dj.nunique()
bands = [(1, 1, "one DJ"), (2, 2, "two DJs"), (3, 4, "three or four"), (5, 9, "five to nine"), (10, 10**6, "ten or more")]
counts = [int(djs_per_rec.between(lo, hi).sum()) for lo, hi, _ in bands]
fig, ax = plt.subplots(figsize=(9, 3.4)); card(fig)
fig.subplots_adjust(left=0.22, right=0.88, top=0.76, bottom=0.08)
header(fig, "how many DJs play a track", f"the {len(djs_per_rec):,} tracks in the data, by the number of DJs who play them")
bars(ax, [cap(c) for _, _, c in bands], counts, CYAN, fmt="{:,.0f}")
plt.show()
finding(2, 1, f"{(djs_per_rec == 1).mean()*100:.0f}% of tracks are played by one DJ only; {int((djs_per_rec >= 10).sum())} track is played by ten or more", round((djs_per_rec == 1).mean() * 100, 1), "% one DJ", "all tracks", int(len(djs_per_rec)))
"""
F2_MD = r"""
**Where tracks are shared, it is between two DJs, not across a genre.** When a track is played by more than one DJ,
it is usually because two particular DJs play a lot of the same music. Amelie Lens and Charlotte de Witte have 89 tracks in common,
Aly & Fila and Paul van Dyk 89, Chris Lake and FISHER 66. Take two DJs of the same genre at random and they share about
ten.
"""
F2_PAIRS = r"""
# ---- the DJ pairs that share the most tracks ------------------------------------------------------------------
n_sets = plays.groupby("dj").mix_id.nunique(); big = plays[plays.dj.map(n_sets) >= 25]
recs = big.groupby("dj").rec.apply(set); genre_of_dj = big.groupby("dj").genre7.agg(lambda x: x.mode().iloc[0])
pairs = []
names = list(recs.index)
for i, a in enumerate(names):
    for b2 in names[i + 1:]:
        shared = len(recs[a] & recs[b2])
        if shared: pairs.append((f"{a} and {b2}", shared, genre_of_dj[a] == genre_of_dj[b2]))
pp = pd.DataFrame(pairs, columns=["pair", "shared", "same_genre"]).sort_values("shared", ascending=False)
top = pp.head(12)
fig, ax = plt.subplots(figsize=(9, 0.3 * len(top) + 2.4)); card(fig)
fig.subplots_adjust(left=0.38, right=0.9, top=1 - 1.3 / fig.get_figheight(), bottom=0.3 / fig.get_figheight())
header(fig, "the DJ pairs that share the most tracks", f"tracks both DJs play, among the {len(names)} DJs with 25 or more sets")
bars(ax, top.pair.tolist(), top.shared.values, CYAN, fmt="{:,.0f}")
plt.show()
for _, r in top.head(4).iterrows(): finding(2, 2, f"{r.pair} share {int(r.shared)} tracks", int(r.shared), "tracks", "DJs with 25+ sets", int(len(names)))
"""
F3_MD = r"""
## 3. How DJs build a set

**In a club, DJs play tracks they have played before. On radio they play new ones.** In club and festival sets, four plays
in ten are tracks the DJ had already played in an earlier set. In radio shows and podcasts it is one play in ten. Ben
Böhmer, RÜFÜS DU SOL and Anyma repeat seven club plays in ten. Claptone, Scuba and Noisia almost never play a track
twice.
"""
F3_REPEAT = r"""
# ---- share of a DJ's plays that repeat a track they played before, club against radio -------------------------
setting = mixes.set_index("mix_id").setting
rep = big.assign(setting=big.mix_id.map(setting)).sort_values("date")
rep["seen"] = rep.groupby("dj").rec.transform(lambda r: r.duplicated())
by = rep[rep.setting.isin(["club", "radio"])].groupby(["dj", "setting"]).agg(rep=("seen", "mean"), n=("seen", "size")).reset_index()
by = by[by.n >= 100]; wide = by.pivot(index="dj", columns="setting", values="rep").mul(100)
wide = wide.dropna(how="all").sort_values(["club", "radio"], ascending=False)
fig, ax = plt.subplots(figsize=(9, 0.32 * len(wide) + 2.6)); card(fig)
fig.subplots_adjust(left=0.3, right=0.9, top=1 - 1.5 / fig.get_figheight(), bottom=0.3 / fig.get_figheight())
header(fig, "how often a DJ plays a track again", "share of plays that are tracks the DJ had played before, in percent", [("club", CYAN), ("radio", WHITE)])
y = np.arange(len(wide))
ax.barh(y - 0.18, wide.club.fillna(0), height=0.34, color=CYAN); ax.barh(y + 0.18, wide.radio.fillna(0), height=0.34, color=WHITE)
for i, (c, r) in enumerate(zip(wide.club, wide.radio)):
    if pd.notna(c): ax.text(c + 1, i - 0.18, f"{c:.0f}", va="center", fontsize=7, color=INK2)
    if pd.notna(r): ax.text(r + 1, i + 0.18, f"{r:.0f}", va="center", fontsize=7, color=INK2)
ax.set_yticks(y, wide.index); ax.set_ylim(len(wide) - 0.5, -0.7); ax.set_xticks([]); ax.tick_params(axis="y", length=0)
plt.show()
club, radio = rep[rep.setting == "club"].seen.mean() * 100, rep[rep.setting == "radio"].seen.mean() * 100
finding(3, 3, f"in club sets {club:.0f}% of plays repeat a track the DJ played before, on radio {radio:.0f}%", round(club, 1), "% repeats, club", "DJs with 25+ sets", int(len(rep)))
"""
F4_MD = r"""
**Most tracks stay in DJ sets for a few months only.** Of the tracks that were played more than once, four in ten were never played
again after three months. One in six was still being played two years later. Trance and drum and bass keep tracks the
longest.
"""
F5_MD = r"""
## 4. Tempo

**Techno is played faster every year.** In 2016 techno sets ran at 126 beats per minute. In 2026 they run at 145, a
step up almost every year in between. No other genre has moved like that.
"""
F6_MD = r"""
**DJs change the speed of the tracks they play.** Half of all plays run more than one percent faster or slower than
the track's own tempo, and faster is more common than slower. Techno DJs change the speed most, afro house DJs least.
"""
F6_SPEED = r"""
# ---- speed changes: slowed, untouched, sped up, per genre ------------------------------------------------------
spd = tempo[tempo.rate.notna() & (tempo.genre7 != "drum and bass")]
grid = pd.DataFrame({g: [(spd[spd.genre7 == g].rate < 0.99).mean() * 100, spd[spd.genre7 == g].rate.between(0.99, 1.01).mean() * 100, (spd[spd.genre7 == g].rate > 1.01).mean() * 100]
                     for g in GENRES if (spd.genre7 == g).sum() >= 100}, index=["slowed", "untouched", "sped up"]).T
grid.loc["All genres"] = [(spd.rate < 0.99).mean() * 100, spd.rate.between(0.99, 1.01).mean() * 100, (spd.rate > 1.01).mean() * 100]
rows = ["All genres"] + [g for g in grid.index if g != "All genres"]; grid = grid.loc[rows]
fig, ax = plt.subplots(figsize=(8, 0.36 * len(rows) + 2.6)); card(fig)
fig.subplots_adjust(left=0.3, right=0.97, top=0.76, bottom=0.05)
header(fig, "how DJs change a track's speed", "share of plays slower than, within, or faster than one percent of the track's own tempo")
ax.imshow(grid.values, cmap=SINGLE, vmin=0, vmax=70, aspect="auto")
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        v = grid.values[i, j]; ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color=CARD if v > 40 else INK)
ax.set_xticks(range(3), [cap(c) for c in grid.columns], fontsize=7.5); ax.xaxis.tick_top(); ax.tick_params(axis="x", pad=6)
ax.set_yticks(range(len(rows)), [f"{cap(g)}   n={len(spd) if g == 'All genres' else (spd.genre7 == g).sum():,}" for g in rows])
ax.set_xticks(np.arange(-0.5, 3), minor=True); ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
ax.grid(which="minor", color=SURF, lw=2); ax.tick_params(which="minor", length=0); ax.axhline(0.5, color=INK3, lw=0.8)
plt.show()
off = spd[np.abs(spd.rate - 1) > 0.01]
finding(4, 6, f"{(np.abs(spd.rate - 1) > 0.01).mean()*100:.0f}% of plays are more than 1% off the track's speed; of those {(off.rate > 1).mean()*100:.0f}% are sped up", round((np.abs(spd.rate - 1) > 0.01).mean() * 100, 1), "% off speed", "measured plays, drum and bass out", int(len(spd)))
"""
F7_MD = r"""
## 5. The DJs

**Some DJs mostly play their own music.** Chase & Status fill 57 of every 100 plays with their own tracks, Anyma 47,
Pendulum 46, FISHER 30. Carl Cox plays his own tracks 4 times in 100, Solomun 3.
"""
F8_MD = r"""
**The artists a DJ plays show their taste.** For every play we looked up how many people listen to that artist on
Last.fm. Amelie Lens fills four plays in ten with artists under ten thousand listeners. Fred again.. plays almost only
his own tracks, so 98 plays in 100 are an artist with over a million listeners. The listener counts are still being
fetched, and each row says how many plays they cover.
"""
CH_PLAYLISTS_MD = r"""
## 6. The cast's tracks

For each DJ of the cast, the tracks they have played in more than one set, with a link where the data has one, and
the tracks that no other DJ in the collection plays. A DJ who plays almost every track once has no favourites, and
the page says so.
"""
CH_TRANS_MD = r"""
## 7. How they mix

The transitions, measured on the audio of 7,600 changes from one track to the next. The cards explain each kind of
transition the mixer knows, the grid shows which kind each DJ uses most, and the dots show how long two tracks play at
the same time. The genre explains almost none of it; the DJ does.
"""



F2B_MD = r"""
**The most played tracks of each genre.** The twenty tracks that appear in the most sets, with how many DJs play
them. Where one DJ has many sets in the data, the list is that DJ's repertoire, and the DJ count beside each track
says so.
"""
F2B_TOP = r"""
# ---- the most played tracks per genre, as tables -------------------------------------------------------------
parts = []
for g in GENRES:
    sub = plays[plays.genre7 == g]
    if sub.mix_id.nunique() < 30: continue
    top = (sub.groupby("rec").agg(sets=("mix_id", "nunique"), djs=("dj", "nunique"), name=("name", "first"))
           .sort_values(["sets", "djs"], ascending=False).head(20))
    tbl = top[["name", "sets", "djs"]]; tbl.columns = ["Track", "Sets", "DJs"]
    parts.append(f"<h3 style='margin-top:1.5em'>{cap(g)}</h3><p style='color:#8b95a5'>{sub.mix_id.nunique():,} sets by {sub.dj.nunique():,} DJs</p>" + tbl.to_html(index=False))
    finding(2, 3, f"{g}: the most played track is {top.name.iloc[0]}, in {int(top.sets.iloc[0])} of {sub.mix_id.nunique():,} sets, by {int(top.djs.iloc[0])} DJs", int(top.sets.iloc[0]), "sets", g, int(sub.mix_id.nunique()))
display(HTML("".join(parts)))
"""


def build(run: bool):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "aidj", "display_name": "aidj", "language": "python"}
    nb.cells = [
        nbf.v4.new_markdown_cell(CH1_MD.strip()),
        nbf.v4.new_code_cell(SETUP.strip()),
        nbf.v4.new_code_cell(LOAD.strip()),
        nbf.v4.new_code_cell(CH1_CODE.strip()),
        nbf.v4.new_code_cell(CH1_CHARTS.strip()),
        nbf.v4.new_code_cell(CH2_PICTURE.strip()),      # the typed transitions and the cast, no chart
        nbf.v4.new_code_cell(CH3_LOAD.strip()),         # every play with its set and date, no chart
        nbf.v4.new_code_cell(CH4_LOAD.strip()),         # played tempo, release year, listeners, no chart
        nbf.v4.new_markdown_cell(F1_MD.strip()),
        nbf.v4.new_code_cell(F1_SHARED.strip()),
        nbf.v4.new_markdown_cell(F2_MD.strip()),
        nbf.v4.new_code_cell(F2_PAIRS.strip()),
        nbf.v4.new_markdown_cell(F2B_MD.strip()),
        nbf.v4.new_code_cell(F2B_TOP.strip()),
        nbf.v4.new_markdown_cell(F3_MD.strip()),
        nbf.v4.new_code_cell(F3_REPEAT.strip()),
        nbf.v4.new_markdown_cell(F4_MD.strip()),
        nbf.v4.new_code_cell(CH3_LIFE.strip()),
        nbf.v4.new_markdown_cell(F5_MD.strip()),
        nbf.v4.new_code_cell(CH4_TEMPO.strip()),
        nbf.v4.new_markdown_cell(F6_MD.strip()),
        nbf.v4.new_markdown_cell(F7_MD.strip()),
        nbf.v4.new_code_cell(CH3_OWN.strip()),
        nbf.v4.new_markdown_cell(F8_MD.strip()),
        nbf.v4.new_code_cell(CH5_FAME.strip()),
        nbf.v4.new_markdown_cell(CH_PLAYLISTS_MD.strip()),
        nbf.v4.new_code_cell(CH5_FAV.strip()),
        nbf.v4.new_markdown_cell(CH_TRANS_MD.strip()),
        nbf.v4.new_code_cell(CH2_TYPES.strip()),
        nbf.v4.new_code_cell(CH2_SHARES.strip()),
        nbf.v4.new_code_cell(CH2_OVERLAP.strip()),
    ]
    nbf.write(nb, NB)
    print("wrote", NB, len(nb.cells), "cells")
    if run:
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
             "--ExecutePreprocessor.timeout=600", str(NB)],
            check=True,
        )
        print("executed")
        HTML.parent.mkdir(exist_ok=True)
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "html", "--no-input", "--output", str(HTML.resolve()), str(NB)],
            check=True, capture_output=True,
        )
        html = HTML.read_text()
        html = html.replace("</head>", "<style>.jp-RenderedImage img, .output_png img {max-width: 900px; width: 100%;} "
                            "body {max-width: 1100px; margin: 0 auto;} table {margin-left: auto !important; margin-right: auto !important;}</style></head>", 1)
        HTML.write_text(html)
        print("html", HTML)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true")
    build(run=not ap.parse_args().no_run)
