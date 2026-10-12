# The DJ data report (notebook), state and decisions

A study of what the data says about DJs and electronic music, for a reader who loves the music and
wants to know what real DJs do, and for the community. Findings first, one chart and one sentence
each. No chapter on what the system should do with it; that comes after the analysis, as decided by
Anas on 2026-10-07 ("jumping straight to training decisions will not be right").

## State, 2026-10-13 (read this first; the dated sections below hold the detail)

**Decided.**
- The page tells findings, one chart and one sentence each, and only what the data shows (Anas, 2026-10-12/13). The
  mixing measurements are craft, not insight; they close the report in one chapter.
- Eight findings, in this order: DJs do not play the same records; the exceptions are pairs; a club set is a repeat and a
  radio show is new; most records are gone in a season; techno got faster; DJs do not play records at their speed; some
  DJs play themselves; taste shows in whom you play. Numbers in "Chapters" below.
- One cast of famous DJs through every chart, chosen for variety of transition, not for counts. Draft `DJ_CAST` in the
  builder (Amelie Lens, Charlotte de Witte, Solomun, Black Coffee, Claptone, Adam Beyer, Armin van Buuren, DJ Tennis, Fred
  again..), floor 30 typed transitions. Anas has not confirmed the names.
- No medians for a DJ; a per-DJ chart shows the whole distribution. Per-genre summaries may use a median.
- Wording: "track" never "record" on the page (Anas, 2026-10-13); "transition" never "move"; "plays a record first" never "breaks"; "how long a record stays in DJ sets"; the
  mixer's words (slam, fade, melt, wave, blend, rise, tension, loop) as column names; every chart text read as prose, first
  word capital, no colon or semicolon inside a sentence. Tables centred. Line charts one colour per genre.
- Fame comes from Last.fm artist listeners, age from Spotify release dates; both in `data/interim/enrich/`, filled by
  resumable passes at one call a second; the charts read them live. Rate limits are read before any pass and a block
  stops the pass (memory `api-rate-limits-first`).

**Done.**
- Chapters 1 to 7 built on the final tables of both corpora (DJ corpus 2,617 seams, Raveform 5,060): the data, the eight
  findings, the cast's records with YouTube links, how they mix (cards, per-DJ grid in the mixer's words, overlap dot
  strips). `reports/dj_data_report.html`, findings in `data/dataset/findings.csv`.
- Set dates for 98 % of sets (page address or title), one record under any spelling, played tempo per play from both
  corpora, Spotify sample of 417 release years, Last.fm at about 2,500 artists and rising.
- Every finding text rewritten for a reader who has never seen the data (Anas, 2026-10-13: "tell them what we found
  cleanly"); the cast's records are tables of records played in more than one set, full names, link beside, and a
  sentence for a DJ who plays almost everything once. No bars for lists.
- Cut after the analysis: favourites per genre (one DJ's repertoire), who plays first (noise on so few shared records),
  tempo peak (5 BPM), age per DJ (sample too thin), the mix-variation index, the speed grid, the level sketch, the
  five-longest table, the per-genre signature grid.

**To improve.**
- Drum and bass tempos fold (127 / 172 / 132 BPM by year); out of the tempo chart until checked (`DATASET_STATE.md`).
- The fame and age charts carry a fraction until the passes finish; Spotify is locked out until 03:20 on 2026-10-14.
- The techno favourites were Charlotte de Witte's repertoire; if favourites return, a DJ's own records need a rule.
- The DJ-corpus transition reads lean on tension and slam; the 24-clip ear set (`data/external/ear_test/types_2026-10-13/`)
  decides whether that is the DJs or the reader.
- Chart 3 (club against radio) is tall; a shorter version with the cast only may read better.

**Done late on 2026-10-13.** The most played tracks per genre are back as tables (twenty per genre, sets and DJ count
beside each) after the shared-pairs chart; "track" replaces "record" on the page; the headlines are plain claims
("DJs do not share their tracks", "Where tracks are shared, it is between two DJs, not across a genre"); the chapter 1
table labels are capitalised; the chart header text is a point bigger; `SYSTEM_OVERVIEW.md` written (the system as it
runs, per stage: planned at the start, what is there, planned next).

**Remaining.**
- Anas: the ear verdicts on the 24 clips, the cast names, whether the eight findings are the right eight, and a rule for
  a DJ's own tracks in the per-genre tables (techno is Charlotte de Witte's repertoire).
- Spotify release-year pass: locked out until 03:20 German time 2026-10-14; one probe, then resume at 1 call a second
  (`scripts/enrich/spotify_release_years.py`, 498 of about 45,000 done). Last.fm pass running at 1 call a second,
  `scripts/enrich/lastfm_artist_listeners.py`; both charts fill in on the next build, no code change.
- Drum and bass tempo fold to separate before it returns to the tempo chart (`DATASET_STATE.md`, 2026-10-13).
- Club-against-radio chart is tall; a cast-only version may read better.
- The appendix: method and trust, where the sets came from, limitations, the findings file.
- The one-page-per-DJ layout, if wanted.
- The DJ music library on the website (TO DO below).
- Commit: docs, notebook, `scripts/enrich`, `scripts/diag`, `scripts/vm` are all uncommitted.

## Files

```
notebooks/build_dj_data_report.py   writes the notebook cell by cell, executes it, exports the HTML
notebooks/dj_data_report.ipynb      the notebook with its outputs (renders on GitHub and in PyCharm)
reports/dj_data_report.html         static export, images capped at 900 px; keep it open in a browser
                                    and refresh after each build (PyCharm Community cannot run notebooks,
                                    and a JupyterLab tab autosaves its stale copy over the file)
data/dataset/findings.csv           one row per finding: chapter, item, sentence, number, unit, filter, n, date
data/dataset/findings_provenance.json   rows and md5 of every table the notebook read
notebooks/card_music.py             caches every local record's 20 ms loudness curve and tempo (5 min) into
data/interim/card_music.parquet     ... this; the transition cards pick their music from it, so a build never decodes audio
reports/rise_designs.html           the four candidate card designs of 2026-10-09 (A was chosen); scratch, can go
```

Build: `conda activate aidj && python notebooks/build_dj_data_report.py` (about 1 min). Needs
`nbformat nbconvert ipykernel jupyter_client jupyterlab` (installed 2026-10-08) and the `aidj` kernel.

## Decided (2026-10-07 and 08)

- One dataset. The three sources (old tracklists, Raveform, DJ corpus) are never shown as sources;
  charts split only by genre, setting and DJ. Where the sets came from goes in the appendix.
- Genres: techno, tech house, house, trance, drum and bass, melodic house, afro house, plus "other"
  (counted in chapter 1, never charted). Raveform's tag combinations map by rule (`genre_of`).
- Setting: club or festival, radio or podcast, unknown. From the title, tags and 1001 page url
  (`setting_of`); agrees with the hand-made club flag on 99.5 % of 1,176 Raveform mixes. The DJ corpus is club by
  construction. 2026-10-08 counts: 1,710 club, 1,698 radio, 479 unknown of 3,887 mixes.
- One DJ, one name: 1001 slugs and Raveform names are joined on the letters alone with & and + spelled
  out (`dj_key`, `canonical`); 461 DJs.
- Per genre first, then each DJ against their own genre and setting. DJ panels: every DJ with 50 or
  more measured transitions (the seven profiling DJs plus thirteen Raveform names). Anas has not
  confirmed this cut; it is the default in use.
- Club and radio are pooled in chapter 2 onward (Anas, 2026-10-08); chapter 1 keeps the split to show where
  the sets come from.
- Playback speed and played BPM come from Raveform transitions only (Anas, 2026-10-08): the DJ corpus speeds
  fail a beat-match check (`DATASET_STATE.md`, 2026-10-08). The appendix says so; no chart names a source.
- The transition cards (2026-10-09, after six rounds with Anas). One card per mixer type: on the left the rule as
  sentences with the real numbers from `RECIPES` and `LEAD` (Volume, EQ, Filter, Length; no "the data will decide",
  no "Lead" row; bass swaps are one bar because the mixer snaps them); on the right Spotify's picture: outgoing
  record above, incoming below, real audio, the volume, bass EQ and filter lines over each in cyan, yellow, pink.
  The music is different per type and chosen to suit it from `card_music.parquet` (a rise's incoming is a real
  build-up, a drop's a breakdown into its drop, fade and melt softer records, slam and wave denser ones); each
  record is cued from inside the record and drawn across the whole picture, dim where the DJ does not play it.
  The waveform is the loudness curve shaped by the recipe at full depth, every change eased over two bars; the
  bent filter curves are right (plain Hz interpolation), only a join between recipe pieces gets eased. Rejected:
  four-row band pictures, synthetic pulses (read as combs), the same two records on every card, half-depth
  shaping, abrupt steps, the incoming "starting from the record's start", the bar picture of a real seam.
- Lengths in bars first, seconds beside. Fred in, flagged. Position in a set by time, five bins.
- No medians for a DJ (Anas, 2026-10-12). A per-DJ chart shows the whole distribution of a length or a move, because
  a DJ like Fred again.. does two things (slams of a few bars and layers of 64 bars or more) and a median describes
  neither. Per-genre charts may keep median and middle half. The same rule binds the DJ profiles and the transition
  model (`DATASET_STATE.md`, decisions section).
- Static matplotlib with seaborn, one style: dark card, monospace, two hues (muted grey for one thing,
  cyan for the other, orange only for "more"), thin marks, no legends (the colour key is a line of
  coloured words under the title), the same header on every chart (title, one grey line, key),
  sentence case, no outlines on bars, numbers in one grey beside the marks. Anas rejected busy charts,
  rainbow stacks, box plots, pure white, lowercase everything, and text scattered around the plot.
- The notebook reads the live tables and stamps them; no snapshot copy. Findings go to the file
  silently, never echoed in the notebook.

## Chapters (rebuilt around the findings, 2026-10-13, Anas: "do the analysis first, tell only what we found")

The page is eight findings, one chart each, in this order. Everything else was cut (favourites per genre, who plays
first, tempo peak, age per DJ, mix-variation index, the sketch, the five-longest table).

1. The data. Numbers table, mixes per genre by setting, the twenty DJs with the most sets.
2. What DJs play. (1) DJs do not play the same records: 91 % of 45,422 records are played by one DJ only, 80 % appear in
   one set, one record is played by ten or more DJs (bars by number of DJs). (2) The exceptions are pairs: Adam Beyer and
   Carl Cox 98 shared records, Amelie Lens and Charlotte de Witte 89, Aly & Fila and Paul van Dyk 89 (bars of pairs).
3. How DJs build a set. (3) A club set is a repeat, a radio show is new: 38 % of club plays repeat a record the DJ played
   before, 9 % on radio (paired bars per DJ, club cyan, radio grey). (4) Most records are gone in a season: of records
   played twice or more, 42 % are last seen within three months, 17 % after two years (dot strips per genre).
4. Tempo. (5) Techno got faster, 126 BPM in 2016 to 145 in 2026 (one coloured line per genre; drum and bass out until
   its tempo readings are checked). (6) Half of all plays are more than 1 % off the record's speed, 31 % sped up against
   21 % slowed (grid per genre).
5. The DJs. (7) Some DJs play themselves: Chase & Status 57 %, Anyma 47 %, Pendulum 46 %, Carl Cox 4 %, Solomun 3 %
   (bars). (8) Taste shows in whom you play: Amelie Lens 41 % of plays under ten thousand listeners, Fred again.. 98 %
   over a million (dot strips and a grid; Last.fm pass still filling, the n on each row says how far).
6. The cast's records. Top 30 per cast DJ with YouTube links where Raveform has one, and the records only this DJ plays.
7. How they mix. The cards, the per-DJ transition grid, the overlap dot strips.
Appendix: not written yet (method and trust, where the sets came from, the findings file).

## Numbers in findings.csv (2026-10-12, final tables of both corpora)

459 DJs, 3,892 mixes, 6,717 measured transitions (chapter 1 counts usable and measured). Typed transitions with a trusted
placement: 5,184 (Raveform 4,076, DJ corpus 1,108). Over all DJs: centre swap 36 %, cut 22 %, tension 20 %, end swap 13 %,
late swap 8 %, loop 1 %. The draft cast against that: Amelie Lens cut 48 %, Solomun tension 63 %, Black Coffee tension 58 %,
Claptone centre swap 45 %, Adam Beyer tension 37 % (centre swap 43 %), Armin van Buuren cut 35 %, DJ Tennis tension 35 %
(loop 7 %, the only looper). Overlaps of 32 bars or more, cuts excluded: drum and bass 60 %, techno 58 %, tech house 44 %, house 37 %, melodic house
15 %, trance 12 %, afro house 8 % (findings.csv, 2026-10-12); the longest blend is 216 bars by Black Coffee. The DJ corpus leans on tension and
cut (38 % and 33 %) far more than Raveform (15 % and 19 %); whether that is the DJs or the reader is the ear check.

## PENDING, waiting on the VM run (written 2026-10-09, late evening; pick up here)

**Status 2026-10-12.** Steps 1 to 3 below are done: the tables landed (2026-10-11), the checks passed (beat-match median
0.05 %, `DATASET_STATE.md` 2026-10-11 section), the DJ corpus is back in the pool and chapter 2 is rebuilt. The Raveform
`dj` column held dates again after the re-export and was fixed at the root (plays and seams) on 2026-10-12. The 14 long windows under 50 minutes were measured on 2026-10-12 (DJ corpus 2,617 rows), the VM is OFF. Open: Anas's
ear check of the 12 signature clips, the cast (draft `DJ_CAST` in the builder, nine names), then chapters 3 to 5 (the overlap chart is
now design B, dot strips, from `reports/overlap_designs.html`; the scratch file can go).

**Why the notebook is stopped.** Both seam corpora are being re-run on the VM (`scripts/vm/rerun_placement_2026-10-09.sh`,
tmux `rerun`, VM `i-023e6cd5518feff75`; what runs and where it lands is in `DATASET_STATE.md` 2026-10-08/09 and the memory
note `vm-rerun-2026-10-09`). Until it ends, the laptop's Raveform `labels.csv`, `types.csv` and `seams_index.csv` are
provisional (new label and types code on the old measures) and the DJ corpus has no signatures at all, because its record
placement was wrong on key-locked records and every stage after locate runs again. The seven DJ-corpus DJs (Amelie Lens,
Solomun, Sultan + Shepard, Black Coffee, Fred again.., DJ Tennis, Roman Flügel) are most of the famous names with measured
transitions, so the cast below cannot be chosen until their moves are known. At 01:00 German time on 2026-10-09 the locate
stage had placed 28 of 286 mixes at about 72 minutes per mix on 16 workers, so the whole run ends late on 2026-10-09 at the
earliest. After that, in order:

1. Pull the tables. `aws s3 sync s3://aidj-1/djdata_djs/out/ data/djdata/djs/out/` and
   `aws s3 sync s3://aidj-1/djdata/out/ data/djdata/raveform/out/`. Then stop the VM.
2. Verify, before any chart. DJ-corpus beat-match mismatch on blends (record BPM times speed must agree on both sides of a
   blend; target a median under 0.1 %, the old run read 0.82 %), the `mode` share in `plays.csv` (how many records matched
   key-locked), the Raveform signature counts against the provisional ones in the section above (centre swap 2,001, tension
   625, cut 600, late swap 457, end swap 436, loop 37), and the render test from `DATASET_STATE.md` 2026-10-09 if the code
   changed. Then a 12-clip ear check of placement and signatures. If a check fails, stop and say so; nothing below is built
   on a failed table.
3. In the builder, put the DJ corpus back into the pool (`ok` is `rv` only today; make it `seams`, both corpora, with the
   same `signature` and `bar_s` filter) and rebuild chapter 2.
4. Pick the cast (below), then the chapter 2 charts, then chapters 3, 4, 5 and the appendix.

**What the provisional table already showed (2026-10-09, scripts `scripts/diag/report_where_it_holds.py` and
`report_cast_census.py`, run with `conda run -n aidj python <script>` from the repo root; they import the builder's own
load cell so they see the same frame).** Share of the variance each split explains (between-group over total, per signature
as a 0/1 indicator; 4,156 typed Raveform transitions, DJs with 50 or more):

```
split                 cut   tension   centre swap   overlap length
genre                 2 %      3 %        1 %           12 %
club vs radio         2 %      4 %        0 %            8 %
DJ (50+)             22 %     17 %        8 %           49 %
```

So the per-genre signature grid says nothing (centre swap is 44 to 52 % in every genre) and the DJ is where the move
holds. Club against radio is one DJ, because Adam Beyer is 411 of the 849 club transitions; without him club and radio
differ by a few points, and no other DJ has 20 transitions in both settings. Overlap length is the one genre fact (techno
median 39 bars, trance 15) and its chart stays. Decided with Anas on 2026-10-09: the per-genre grid goes, the "who leans on
which transition" lift chart goes (a lift against a genre that holds nothing is a lift against everyone), and one per-DJ
grid replaces both, with an "All DJs" row as the reference. The six signature names are ours, so no heading may call them
"transition types"; the grey line under the title says they are defined above, from where the bass swaps.

**The cast, what we need and why.** Anas wants the same famous DJs in every chapter, so a reader follows the same names
from the move they make, to the records they play most, to how much they vary set to set. Not many names; four to nine. The
cast is chosen for VARIETY of transition between the DJs, never for the number of transitions. What to check on each
candidate, all from `report_cast_census.py`:

- mixes with a tracklist (the record and variation chapters need 25 or more)
- measured transitions, both corpora (the move charts need 50 or more; a cast member under 50 is dropped from those charts
  only, and the grey line says so)
- the move the DJ owns, as the largest lift in points over all DJs (not the largest share; Adam Beyer's largest share is
  centre swap and his move is tension, +29 over techno)
- top record plays and the repeat share (how much of their plays are records they played before), for the favourites chapter
- own-music share (artist name contains the DJ name), for chapter 4
- club and radio shares, genre

How each is used. The per-DJ move grid (chapter 2) shows the cast with 50+ transitions, one DJ owning each move where one
exists. Chapter 3 (which sound first, bass swap timing, speed change) shows the same names. Chapter 4 (top records, repeats,
own music, openers and closers) and the playlists appendix show the whole cast. Chapter 5 (variation) needs both ends in
the cast, a repeater (FISHER plays Losing It in 49 of 65 sets, repeat share 71 %; Charlotte de Witte 41 of 167, 64 %) and
a varier (Fred again.., top record 4 plays in 27 sets, 62 % own music; Noisia 17 % repeat share, 208 sets). Candidates with
their numbers on the provisional table (transitions are the old DJ-corpus run plus provisional Raveform; the cast does not
depend on them, the eligibility does):

```
DJ                   genre           mixes  transitions  top record plays  repeat share  owns (provisional)
Adam Beyer           techno            226     499             15               50 %     tension +29 vs techno
Charlotte de Witte   techno            167      42             41               64 %     centre swap +21
Amelie Lens          techno            150     593             19               41 %     unknown until the VM
Solomun              tech house        137     730              9               23 %     unknown until the VM
Carl Cox             tech house         95      34             14               31 %     centre swap +17
FISHER               tech house         65       0             49               71 %     none measured
Sultan + Shepard     melodic house     167     261              6               19 %     unknown until the VM
Black Coffee         afro house         78     152              9               42 %     unknown until the VM
Fred again..         house              27     113              4               42 %     unknown until the VM
Armin van Buuren     trance             67      74             13               31 %     cut 24 %, the only famous looper
Noisia               drum and bass     208       0             12               17 %     none measured
```

Clear owners that are not famous, kept as a fallback for a move no famous DJ owns: Scuba cut +39 (102 transitions, 25
sets), Amoss tension +41 (75, 12), Boris centre swap +38 (66, 11), Manuel Le Saux late swap +17 (71, 15). How to pick,
once the VM tables are in: rank every DJ with 50+ transitions by the distance of their six move shares from everyone's,
keep the famous ones, and take one DJ per move so that each move has an owner; add the repeater and the varier from the
list above if they are not already in; cap at nine. Anas confirms the names before any chart is rebuilt.

**What remains to build, chapter by chapter.** Every chart keeps the rules in "Decided" above and in the memory note
`chart-feedback-2026-10-09`; for any new chart family, several candidate designs of one chart first, Anas picks, then it is
applied to all.

- Chapter 2, rebuild: cards and the level sketch unchanged; the per-DJ move grid (new, replaces two charts); overlap length
  per genre with the five longest (unchanged, DJ corpus added).
- Chapter 3, in the mix: which band of the incoming record arrives first (`b_high_pos`, `b_mid_pos`, `b_low_pos` in
  `labels.csv`, as a fraction of the overlap); where the bass swaps and how long both basses play (`swap_pos`,
  `bass_both_bars`); the tension cut (`bass_cut_bars`); how much the records are changed (share untouched, sped up, slowed
  down, by how much, from Raveform rates only, `a_rate`, `b_rate` in `seams_index.csv`, with loops beside it from `loop`,
  `loop_steps`); whether a set drifts faster through the night (played BPM against position in the set, five time bins).
  Per genre, then the cast.
- Chapter 4, the records: top records per genre and per cast DJ with play counts; how often records come back and the long
  tail (plays per distinct record); share of a set that is the DJ's own music; opening and closing records. From the
  tracklists of all three sources (`plays` as built in `report_cast_census.py`; the record key is the letters of
  artist and title, so the same record under two ids counts once).
- Chapter 5, repeat or vary: set-to-set repetition in records and in moves per cast DJ; a variation index with five
  equal-weight parts shown (records, moves, overlap length, speed, set shape), ranked per setting; the same DJ on radio
  against in a club where both exist (only Adam Beyer today); genres compared; how much of the variation genre and DJ explain
  (the same between-over-total share as above).
- Appendix: playlists (top 20 per cast DJ by repeat count; a DJ who never repeats gets favourites by genre), method and
  trust, limitations (no key-lock detection so no key claims, no stacks away from seams, no transitions for the old dataset,
  DJ-corpus tempo above 150 BPM, speed from Raveform only), where the sets came from, the findings file.

**Taste and fame, the chapters 3 to 5 that replace the knob work (Anas, 2026-10-12, late).** The mixing measurements
(which band first, bass swap, speed changes, tension, layers, tempo drift) matter for the system and not for a reader; they
go into one short appendix section. The reader chapters become: 3 the records (who plays a record first, how long a
record stays in DJ sets, the canon every DJ plays and each DJ's exclusives, how much of a set is the DJ's own music); 4 the set (the
same set every night against a new one, how old the music is, has music got faster by year, when the peak comes); 5 the
cast, each of those per DJ, with the taste axis Anas asked for: DJs who play small artists and old or unfashionable
records against DJs who play what is in trend. Checked 2026-10-12: fame cannot come from inside the data (92 % of the
43,932 records are played by one DJ only, so "no other DJ plays it" is true of everyone and only mirrors how many DJs of a
genre we hold); Spotify's API in development mode withholds followers and popularity (403 on artists, None on a single
artist, no popularity on search items) but matches 72 % of old-tracklist records and 51 % of Raveform's with the right
record (12 of 12 checked) and gives the release date, so it is the source for the age of the music; fame comes from Last.fm
(key in `.env` as `LASTFM_API_KEY`, added 2026-10-13). Checked on the same 200 records: the artist was found for 100 % of
the old-tracklist records and 71 % of Raveform's (the misses are titles that still carry a `[mm:ss]` timestamp, to strip),
artist listeners run from about 1,000 (10th percentile) to 480,000 (90th), Fred again.. 1.4 million; track listeners are
too thin (median 144) so the fame number is ARTIST listeners. Full passes not yet run: Spotify for release year (about
42,000 searches, 0.75 s each) and Last.fm for artist listeners (one call per distinct artist, far fewer); both resumable,
both at a polite rate on the laptop. Scratch scripts for the checks were in the session scratchpad; each is a few lines
of requests.

**TO DO after the notebook, the DJ music library on the website (Anas, 2026-10-09).** A small page where a reader searches
for a DJ or picks one of the top names and sees that DJ's library, the records sorted from most played to least with the
number of sets each was played in, and every record a YouTube link, so a click on Solomun shows what he plays most and one
more click plays it. A filter for the records only this DJ plays, that no other DJ in the data plays, so people can find new
music from what the DJs know; people Shazam in clubs for exactly this and the information is sitting in our tables. The data
is the `plays` table of chapter 4 (all three sources, record key as above); the YouTube ids come from Raveform's tracklists
(`data/raw/raveform/raveform/mixes.jsonl`, `tracklist[].id`) and from the fetcher's verified downloads for the DJ corpus,
and the old dataset's records need a search. Not designed yet; it follows the playlists appendix, which is the same table
in print.
