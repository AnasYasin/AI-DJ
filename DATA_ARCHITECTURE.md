# Data architecture, as it runs (2026-10-13)

Three sources feed one dataset, and the models read only the dataset. The sources keep one upgrading copy
of their tables; the dataset is a dated, frozen folder built from them by one command. Git tracks code,
configs and READMEs, never the parquet. The previous layout and what was wrong with it are in
`docs/notes-archive/data_architecture_2026-09-24.md`.

```
data/raw/                      inputs as received, never edited: the 1001 scrape, the Raveform zip, the previews
data/djdata/<source>/          old | raveform | djs            one folder per source, the same inside
  tracks/ windows/ mixes_tmp/    audio named by id (the laptop holds little; the VM and S3 hold all)
  out/                           the tables that source's pipeline writes, one live copy
data/dataset/<date>/           tracks, plays, seams, layers (parquet) + README     what the models read
data/archive/<date>/           superseded files moved whole, with MOVED.md         never read by code
models/                        the saved models; a model's README names the dataset date it trained on
S3 aidj-1/                     djdata/ djdata_djs/ dataset/<date>/ archive/<date>/ models/ processed/ raw/previews/
```

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
same idea for DJ names, with & and + spelled out.

## The dataset

`djdata export-dataset --date <date>` (`djdata_package/djdata/dataset.py`) reads every source and writes
`data/dataset/<date>/`. About 9 s. Every row carries `source`.

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

**layers.parquet**, one row per record and band in a span where records stack, 3,705, DJ data only.
The per-window `presence.csv` and the per-beat curves stay in the source's `out/`; they are
intermediates of the measurement, joinable by mix id and time.

**README.md** in the folder: counts per source, the validation mixes per genre and per source, the md5 of
every input table.

**Validation split.** `split` is `train` or `validation` on every play, and on every seam through its
mix, by whole DJ, so no validation DJ is ever trained on (`validation_split()` in
`djdata_package/djdata/dataset.py`, `VALIDATION` in the loader). DJs with 3 to 40 mixes are
eligible; at least three per genre are taken first, then more while every genre stays under 15 % of its
mixes. On 2026-10-14: 561 mixes (14.2 %) from 58 DJs, every genre and every source represented (old 291,
Raveform 234, DJ data 36), 1,370 of the 6,731 measured seams. It replaced a `md5(dj_key) % 5` fold whose
validation fold held no DJ-data mix. The old fold text is in `docs/notes-archive/`.

**Genre.** `genre_of` in `dataset.py` is the one mapping from a source's label to the seven genres
(techno, tech house, house, trance, drum and bass, melodic house, afro house, else other). Raveform's
mixesdb tag strings go through it; the notebook builder carries a copy. An old record's genre is the
first set it was listed in.

## Who reads what

```
src/data/dataset.py     load_tracks(date=, sources=, audio=)  load_plays  load_seams  load_layers
                        the newest date when none is given
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

## Where things are, 2026-10-13

Laptop: the sources' `out/` tables, the dataset, the archive (1.4 GB, not yet on S3), `data/raw/previews`
(28 GB), the DJ-data tracks and windows, the Raveform windows and tracks. VM: all audio of both measured
sources, off when not in use. S3: everything above except the archive and the dataset. `data/interim`
holds the raw scrape, the caches (`segments/`, `stretch/`), the preview stores, the enrichment tables
and `old_sample_full/` (867 MB of fetched old records, not yet on S3).

## Open

- Fold 0 holds no DJ-data mix.
- The old record's genre is the first set's; 882 records sit in sets of two genres.
- 242 Raveform mixes are tagged both techno and tech house and the rule calls them all tech house. Checked
  2026-10-13: their played tempo sits between the two (median 127 BPM against 125 and 128), the records the
  old data knows from them are techno twice as often (64 against 32), the DJs are Adam Beyer, Eddie
  Halliwell, Scuba, Boris. Left as tech house (Anas) until a genre from the audio decides all 242 by one
  measure; 10 clips for his ear is the fallback.
- The archive and the dataset are not on S3.
- `raveform` plays carry no match rate; it is in the alignment files if a filter needs it.
