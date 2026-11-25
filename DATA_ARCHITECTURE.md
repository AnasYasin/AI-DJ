# Data architecture, as it runs (2026-10-13)

Three sources feed one dataset, and the models read only the dataset. The sources keep one upgrading copy
of their tables; the dataset is a dated, frozen folder built from them by one command. Git tracks code,
configs and READMEs, never the parquet. The previous layout and what was wrong with it are in
`docs/notes-archive/data_architecture_2026-09-24.md`.

```
data/raw/                      inputs as received, never edited: the 1001 scrape, the Raveform zip, the previews
data/djdata/<source>/          old | raveform | djs            one folder per source, the same inside
  tracks/ windows/ mixes_tmp/    audio named by id (the laptop holds little; S3 holds all of it; no VM exists between runs)
  curves/ segments/              one file per held track, named by the same id: level curves and bar grid with sections
  out/                           the tables that source's pipeline writes, one live copy
data/dataset/<date>/           tracks, plays, seams, layers (parquet) + README     what the models read
                               mixes, groups, group_types, dj_types (parquet)      the profiles built from those four
data/archive/<date>/           superseded files moved whole, with MOVED.md         never read by code
models/                        the saved models; a model's README names the dataset date it trained on
S3 aidj-telfy/                     one prefix per source, the archive, the previews; the full layout is the next section
```

## S3, what is there

The bucket is the one complete copy of the audio and the tables. A VM is started when a run needs one, pulls
from here, and is terminated after; the laptop holds the tables and a sample of the audio.

```
s3://aidj-telfy/
  djdata/                        Raveform source            = data/djdata/raveform/
    tracks/                      6,917 full records, 41 GB
    windows/                     6,218 cut seam windows from the Raveform manifest, 15 GB
    embeddings/                  float16 patch embeddings, 6.5 GB
    curves/                      6,917 files, 211 MB, <id>.npz: rms_db, low_db (under 150 Hz), mid_db, high_db
                                 (over 2 kHz) and onset on a 0.25 s grid over the whole track (2026-11-13)
    segments/                    6,917 files, 69 MB, <id>.json: beats, bars, bpm, phrase_offset, sections,
                                 bar_energy, downbeat_source, downbeat_confidence, what the mixer's segmenter
                                 returns (2026-11-13)
    out/                         the live tables: cuts, measures, labels, types, seams, seams_index, plays,
                                 tempos, track_features.parquet; old_gainfit_2026-09-14/ is an earlier run's tables;
                                 seam_cues.csv, one row per measured seam, where in each track it happens;
                                 track_structure_log.csv, one row per track of the curves and segments run
    logs/                        djdata.log per run under a dated name, run scripts, memory and stderr logs
  djdata_djs/                    DJ corpus                  = data/djdata/djs/
    tracks/                      5,314 full records, 31 GB
    mixes_tmp/                   the full mix recordings, one file per mix id, 31 GB; the usb_<segment>.m4a
                                 names are links to fred/segments/ on disk and are not objects here
    fred/                        Fred again: segments/ (43 recordings), lists/, checks/, 5 YouTube mixes
    windows/                     2,617 cut seam windows of the current placement, 12 GB, one per row of
                                 out/cuts.csv, same names and sizes as the folder the measures were read from
    embeddings/                  4.9 GB
    curves/  segments/           5,294 files each, 159 MB and 50 MB, same content as under djdata/
    out/                         live tables, plus seam_cues.csv, track_structure_log.csv, layers, layer_bands, presence, mixes,
                                 measure_big_windows_2026-10-11.csv (35 windows over 25 min and the decision on each)
    logs/
  archive/2026-10-09_old_placement/   the run before the key-lock placement fix, the only backup of it
    djs/out/  raveform/out/      its tables
    djs/windows/                 1,694 of its 2,459 windows; the other 771 are gone (overwritten) and are
                                 re-cut from mixes_tmp/ with djs/out/cuts.csv if ever needed
    README.txt                   names the 1,694
  archive/scripts_diag_untracked_2026-09-30.tar.gz
  dataset/<date>/                the dataset folder with its README and the profile tables  = data/dataset/<date>/
                                 2026-11-14 and 2026-11-18 are there
  raw/previews/                  30,146 iTunes 30 s previews, 31 GB      = data/raw/previews/
  seam_previews/                 audio/ 5,989 and played_part/ 594 clips for the seam ear tests
  processed/                     the old source's tables (features, embeddings, tracklist_clean)
  models/  interim/  tmp/
```

Rules that come from the bucket itself:

- No versioning. An overwrite is final. A new run writes to a dated prefix first; the switch to the live
  prefix is a separate step after the counts match.
- The VM role `aidj-eip` can list, read, copy and write on `aidj-telfy` and cannot delete. Deletes run from
  the laptop with the `talhanonstatic` profile.
- `aws s3 sync` follows symlinks, so a dry run from a corpus folder lists the Fred links as missing
  files; they are not.
- The run scripts sync `out/` only. `windows/` is synced by hand after a cut, or S3 holds the previous
  cut under the new tables.

The old source has no audio stages, so its `out/` is `data/processed/` as it always was:
`tracklist_clean.csv` (the plays) and `features.parquet` (the preview features). `data/interim/tracklist.csv`
is the raw scrape it came from, and the overlay rows live only there. `SOURCES` in
`djdata_package/djdata/dataset.py` names these paths.

## The three sources

| source | what it is | tracks | plays | mixes | seams measured |
|---|---|---|---|---|---|
| old | 1001tracklists sets, 30 s iTunes previews, features on the preview | 28,460 | 47,390 | 2,852 | none |
| raveform | KAIST Raveform: mix audio windows, YouTube track ids, its own alignment | 6,917 full | 14,570 | 939 | 4,723 |
| djs | seven DJs' 1001 sets with the mix audio from the page link, records located by fingerprint, Fred's solo sets | 5,294 full | 5,684 | 286 | 2,008 |

Counts from the 2026-10-14 export. 140 DJ-data mixes are old-data mixes (same id), and 2,628 old records
have full audio through the DJ data; the dataset keeps the measured or full row and drops the other.

## Ids

One scheme everywhere, the 1001 scrape's (`src/data/tracklists1001_client.py`):

```
track_id  = md5("artist|title".lower())[:12]
mix_id    = md5(page url)[:10]
seam_id   = mix_id + "_" + track_a + "_" + track_b
```

The old data and the DJ data carry these already. Raveform's YouTube ids and `mixNNNN` numbers are
re-hashed by the export from the title (after the `[Label]` suffix) and the mixesdb url, and the
original is kept in `source_ref`, which is also how the audio file is found. So one record played by
two DJs in two sources is one id. `record_key` (letters and digits only) sits beside the id for joins
where punctuation differs ("vs" against "vs.": 49 of 732 old matches on 2026-10-13). `dj_key` is the
same idea for DJ names, with & and + spelled out and accents dropped (`dj_key()` in the loader). Until
2026-11-18 an accented letter was removed whole, so Roman Flügel was `romanflgel` in Raveform and
`romanflugel` in the DJ data and counted as two DJs. Nine names changed key and only his two merged.

## The dataset

`djdata export-dataset --date <date>` (`djdata_package/djdata/dataset.py`) reads every source and writes
`data/dataset/<date>/`. About 9 s. Every row carries `source`. `python -m src.data.profiles` then adds the
six profile tables to the newest folder, about 4 minutes. The newest folder is 2026-11-18.

**tracks.parquet**, one row per record, 37,612. `track_id, record_key, source, audio (preview | full),
source_ref, artist, title, genre, minutes, n_vectors, embedding (1,280 float32), bpm, bpm_source
(deeprhythm | measured), key, loudness_lufs, energy_mean, energy_std, spectral_centroid, onset_strength
(raw), mfcc_0..12, play_count, dj_count, release_year (Spotify), listeners (Last.fm artist)`. A record in
two sources keeps the full-audio row. 146 records have no BPM (138 uncertain Raveform, 8 with no
periodic kick); training drops them.

**plays.parquet**, one row per play in a mix, 65,513. `mix_id, source, source_ref, dj, dj_key, genre,
mix_title, mix_url, position, track_id, listed_min, split` for every source, and for the measured
sources `found, confidence, time_zero_s, rate, mode (keylock | resample), played_bpm, first_heard_s,
last_heard_s, played_from_s, played_to_s, votes, floor`. A mix in two sources keeps its measured rows.
11 % of plays name a track that has no features row (unfetched or unidentified); they stay, as order
data.

**seams.parquet**, one row per consecutive pair in a mix, 53,721. `seam_id, mix_id, source, dj, genre,
track_a, track_b, order_a, order_b, kind (seam | overlay), split`, and for the measured sources every
column of `seams_index.csv` (placement, window and audit, per-band entry and exit with votes, floors
and separations, carried dB, bass swap and bass words, presence and loop, bars and positions,
signature, nearest recipe, distance) plus `a_mode, b_mode`. The old data's pairs carry the identity
columns only. `measured == 1` on 6,731 rows; filter on the per-band separations before trusting a
number. The 867 overlay rows are the old data's "w/" records (parent as `track_a`).
Since the 2026-11-14 export the measured seams also carry the 37 columns of `out/seam_cues.csv`
(`scripts/experiments/seam_cues.py`), joined by `seam_id`. They say where in each track the seam happens.
`b_in_*` is where the incoming track is first heard, `a_out_*` where the outgoing track is last heard,
`a_swap_*` and `b_swap_*` where each track is at the bass swap. Each comes as `_s` (track seconds),
`_bar` (bar index on the track's own downbeats, negative before the first bar), `_phrase_pos` (bar inside
the 8-bar phrase), `_section` (the segmenter's label there), `_edge_bars` (bars to the nearest section
boundary), `_rms_rel_db` and `_low_rel_db` (level over 8 s after the entry, before the exit or around the
swap, minus the track's own 90th percentile). `a_bars_left` is the bars of the outgoing track after its
exit. Per track there are `_n_bars`, `_duration_s`, `_downbeat_source` and `_downbeat_confidence`.
Mix time is the seam window reading (`window_t0 + in_s`, `out_s`, `bass_swap_s`) and track time is
(mix time - `time_zero_s`) * `rate`. Limits, measured 2026-11-14, are in `DATASET_STATE.md`.

**layers.parquet**, one row per record and band in a span where records stack, 3,705, DJ data only.
The per-window `presence.csv` and the per-beat curves stay in the source's `out/`; they are
intermediates of the measurement, joinable by mix id and time.

**mixes.parquet**, one row per mix, 3,937, every source (`mix_table()` in `src/data/profiles.py`). `mix_id,
source, dj, dj_key, genre, mix_title, mix_url, n_tracks, n_featured, seams`, the mix's `bpm` (median of its
tracks), `energy, loudness, brightness, density` (mean over its tracks of each track's percentile inside its
genre and audio kind), `energy_rise` (last third minus first third), `shape` (the old mix profiler's curve
shape), `change_min` (median minutes from one track's entry to the next, measured mixes), `setting`
(festival, club or radio from words in the title, empty on 2,710), `set_kind` and `group`. `set_kind`
(warm-up, middle, peak-time) is the third of the genre's mixes the mix's energy falls in, on the 3,413 mixes
with 8 tracks with features. It is unproven: energy is close to loudness here (correlation 0.94), no title
says warm-up, and inside one DJ it changes nothing consistent about the transitions.

**groups.parquet**, one row per style group, 15 (`group_table()`). A unit is one DJ in one genre with the
genre's floor of measured seams (`FLOOR` in the module, with the numbers behind each floor: house 10, drum
and bass 10, tech house 12, trance 12, techno 15). The units of a genre are clustered on twelve style
numbers. They are the share of each of the five transition types, median overlap, share of entries before
bar 8 and past bar 32, share of exits in the last 8 bars, `change_min`, energy and BPM. A DJ nearer another
group than their own is left out, unless the DJ has 50 seams or is one of the seven of the DJ data, who
then forms a group alone. The row holds the group's `djs`, `seams`, the mean of each style number, and
what a request is matched against: `mixes`, `bpm_low`, `bpm_high`, `energy_low`, `energy_high` (10th and
90th percentile over the group's mixes). A group is named by its genre and one word, for example
`techno blend`, `techno cut`, `techno fast`. 95 units, 5,043 of the 6,731 measured seams. A DJ's held-out
half of mixes lands in their own group on 0.87 to 0.97 of tries by genre, against 0.26 to 0.50 by chance.

**group_types.parquet**, one row per group and transition type, 73 (`type_table()`). `share` and `seams`,
then how the type is done (`overlap_bars`, `bass_free_bars`, `new_bass_at`, `old_bass_out`, `sweep`) and
what the incoming track is like in that place (`enters_bar`, `entry_level_db`, `entry_section`,
`bars_left`, `play_min`, `energy_step`). `overlap_bars`, `enters_bar` and `play_min` also come as `_low`
and `_high`, the 25th and 75th percentile. A DJ counts by the square root of their seams in the group, in
the share and in every percentile (Anas, 2026-11-18), so the deepest DJ leads a group without owning it.
20 rows rest on under 15 seams.

**dj_types.parquet**, the same columns per group, DJ and type, 380 rows for 87 DJs. A request that names
a DJ reads this.

**djs.parquet**, one row per group and DJ, 95 (`group_table()`). The card a group has, per DJ: `mixes`,
`bpm_low`, `bpm`, `bpm_high`, `energy_low`, `energy_high`, `no_bass`, `no_bass_high`. A named DJ is
matched against their own. Fred again's sets sit at a median 134 BPM where his group's sit at 125.

**structure.parquet**, one row per held track with a segment file, 12,153 (`structure_table()`).
`n_bars`, `bar_s`, `intro_bars`, `first_drop_bar`, `outro_bars`, `downbeat_confidence` (under 0.5 on
19 % of tracks) and `no_bass`, the share of the track with its lows more than 10 dB under its own loud
lows (median 0.27). The planner reads it for slot fit, for cut seams and to leave out tracks emptier than
a style really plays. Groups carry `no_bass` and `no_bass_high` beside their tempo and energy columns, the
median and 75th percentile over their real plays of the bass-free share of what was played of a track.

**README.md** in the folder: counts per source, the validation mixes per genre and per source, the md5 of
every input table.

**Validation split.** `split` is `train` or `validation` on every play, and on every seam through its
mix, by whole DJ, so no validation DJ is ever trained on (`validation_split()` in
`djdata_package/djdata/dataset.py`, `VALIDATION` in the loader). DJs with 3 to 40 mixes are
eligible; at least three per genre are taken first, then more while every genre stays under 15 % of its
mixes. On 2026-10-14: 561 mixes (14.2 %) from 58 DJs, every genre and every source represented (old 291,
Raveform 234, DJ data 36), 1,370 of the 6,731 measured seams. The 2026-11-18 export has 561
validation mixes again, from 56 DJs. The merged key for Roman Flügel moved two of his Raveform mixes into
validation and two Above & Beyond radio shows out of it. The pair model was trained on the earlier split. It replaced a `md5(dj_key) % 5` fold whose
validation fold held no DJ-data mix. The old fold text is in `docs/notes-archive/`.

**Genre.** `genre_of` in `dataset.py` is the one mapping from a source's label to the seven genres
(techno, tech house, house, trance, drum and bass, melodic house, afro house, else other). Raveform's
mixesdb tag strings go through it; the notebook builder carries a copy. An old record's genre is the
first set it was listed in.

## Who reads what

```
src/data/dataset.py     load_tracks(date=, sources=, audio=)  load_plays  load_seams  load_layers
                        load_mixes  load_groups  load_group_types  load_dj_types  load_djs  load_structure
                        the newest date when none is given
src/data/profiles.py           plays, tracks, seams, curves and segments; writes the six profile tables
src/models/set_plan.py         groups, group_types, dj_types: the styles and the seam budget of a planned set
src/models/predict_model.py    with a style: groups, djs, structure, plays (the style's pool, `join_fit`)
src/models/pair_model.py       the pair model: tracks + plays, the `split` column (validation DJs never trained on)
src/models/track_arrays.py     TrackArrays(df): a pool's column arrays for the planner and the pair model
src/models/predict_model.py    the planner's pool: tracks (genre, BPM, energy floor, artist, title)
src/features/mix_profiler.py, transition_labeler.py   tracks (legacy phase 4 scripts)
notebooks/build_dj_data_report.py   reads the sources' out/ tables directly, not the dataset
src/audio/audio_mixer.py       nothing from here; it measures every track itself at render time
```

Leaving a source out of training is an argument (`sources=["raveform", "djs"]`), never a different
table. No vector index: 40k by 1,280 float32 is 200 MB in memory and a cosine over all of it takes
milliseconds; the hard-negative search in Model A is a matrix product per genre.

## Rules

1. **Source tables upgrade in place.** A re-run overwrites `out/`. The previous copy goes to
   `archive/<date>/` on S3 and leaves the laptop.
2. **Dataset folders are frozen.** Written once by the export. A model trained on it names the date. A
   new export is a new date. Old dates leave the laptop once no model in use points at them.
3. **Git tracks code, configs and READMEs**, never the parquet.
4. **Archive what cannot be rebuilt, delete what can.** Ear marks, previews, raw scrapes and the tables
   a model trained on go to `data/archive/<date>/` with a line in `MOVED.md`; re-creatable
   intermediates are deleted.
5. **Formats.** Parquet for every table (typed, compressed, columnar: tracks with the embeddings is 202
   MB). CSV only for what a person edits (ear marks, exclusion lists, the cast). Patch embeddings are one
   float16 `.npy` per track on S3 (`djdata/embeddings/`, `djdata_djs/embeddings/`, 11.5 GB), pulled per
   seam when the transition model needs them.
6. **Adding data.** A new DJ is rows in the tracklist and audio on disk, then the source's stages. A new
   source is a config and a folder. Either ends with one export.

## Where things are

Laptop: the sources' `out/` tables, the dataset, the archive (1.4 GB, not on S3), `data/raw/previews`
(28 GB), a sample of the DJ-data and Raveform tracks and windows. S3: all audio and tables of both measured
sources, the previews, the old source's tables, the archive of the previous placement run (section above).
No VM exists between runs; `djdata_package/VM_SETUP.md` sets one up and pulls the corpora from S3.
`data/interim` holds the raw scrape, the caches (`segments/`, `stretch/`), the preview stores, the
enrichment tables and `old_sample_full/` (867 MB of fetched old records, not on S3).

## Open

- Fold 0 holds no DJ-data mix.
- The old record's genre is the first set's; 882 records sit in sets of two genres.
- 242 Raveform mixes are tagged both techno and tech house and the rule calls them all tech house. Checked
  2026-10-13: their played tempo sits between the two (median 127 BPM against 125 and 128), the records the
  old data knows from them are techno twice as often (64 against 32), the DJs are Adam Beyer, Eddie
  Halliwell, Scuba, Boris. Left as tech house (Anas) until a genre from the audio decides all 242 by one
  measure; 10 clips for his ear is the fallback.
- The archive is not on S3. The dataset is, since 2026-11-14 (`s3://aidj-telfy/dataset/<date>/`).
- The profile tables are read only on the planned path (`--dj`, `--style`, `--group`).
- Back-to-back sets and pairings are their own `dj_key` ("Adam Beyer b2b Ida Engberg"), so they count as
  separate DJs in the groups.
- `raveform` plays carry no match rate; it is in the alignment files if a filter needs it.
