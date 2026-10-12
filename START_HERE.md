# AI-DJ, start here (index, 2026-09-27)

A prompt turns into a real, beat-matched DJ mix. The mixer is deterministic and ear-verified. The learned
parts choose the track, the length, the move and its parameters, trained on how real DJs actually mix.
Nothing learned replaces a rule until it beats it in a blind listen.

This file is only an index. Read it, then go to the file for the work at hand and read that in full.
Detail never lives here.

## Where to look for what

| you want | read |
|---|---|
| the whole plan, its ten steps, and Anas's decisions in date order | **`PLAN.md`** (Anas writes it; a dated status paragraph is the most anyone else adds) |
| the mixer, the planner, the fetcher, the models, their gotchas, and how sessions work in this repo | **`CLAUDE.md`** (read at the start of every session) |
| the whole system as it runs today, stage by stage: planned at the start, what is there, planned next | **`SYSTEM_OVERVIEW.md`** (written 2026-10-13, before the next changes) |
| how to run the seam pipeline, its stages, tables, configs, how to add data, the watch list for a run | **`DATA_PIPELINE.md`** |
| the state of the data: what is measured, what was found, what is closed, the numbers and their dates | **`DATASET_STATE.md`** |
| the data layout as it runs: the three sources, the ids, the dataset the models read, the loader, the archive, S3 | **`DATA_ARCHITECTURE.md`** |
| the djdata package: install, stages, adding a corpus, reading the tables | **`djdata_package/README.md`** |
| Fred again, a separate job: his sources, segments, what is proved | **`FRED_AGAIN_SETS.md`** |
| track features and embeddings for the seam corpora: the plan, the checks, the numbers as they land | **`EMBEDDINGS.md`** |
| the DJ data report notebook: what was decided, the chapters, the style, how to build it | **`DJ_DATA_REPORT.md`** |
| how to set the VM up, cookies, yt-dlp, the token server, the changing address | **`djdata_package/VM_SETUP.md`** |
| ssh to the VM whatever its address is today | `scripts/vm/vmssh` |
| how to run the mixer end to end, the product side | **`README.md`** |
| old session records and superseded notes, kept whole with their dates | `docs/notes-archive/` |

New kinds of work, for example model fine-tuning or retraining, get their own md and a row here.

## State in one paragraph (2026-09-30)

Both seam runs are done. The DJ profiling run: 285 mixes, 3,365 seams, 2,459 cut, measured and
labelled, layer tables for the stacking DJs (`DATASET_STATE.md` 2026-09-29). The Raveform run: 5,060
seams cut, measured, tempo'd and labelled, 4,639 with both records clear (`DATASET_STATE.md`
2026-09-30, which also has the webm start fix it needed). `DATA_PIPELINE.md` has how each runs. The
tables are on the laptop in `data/djdata/djs/out/` and `data/djdata/raveform/out/`, on the VM, and on
S3 at `s3://aidj-1/djdata_djs/out/` and `s3://aidj-1/djdata/out/`. The old untracked `scripts/diag`
code is deleted and archived on S3. Raveform's tempos were checked against its beat annotations on
2026-10-07 and 1,056 seams relabelled (`DATASET_STATE.md`). Track features and embeddings for both seam
corpora are extracted from the full tracks (2026-10-07, 12,211 tracks, `data/djdata/<corpus>/out/
track_features.parquet` and `embeddings/`, laptop, VM and S3; plan, checks and numbers in
`EMBEDDINGS.md`); every track has a measured BPM except 146 that stay null. How the old and the new
data are used from here is the top section of `DATASET_STATE.md`, with the data layout as an open
proposal. The Raveform `dj` column held dates until 2026-10-08 and is fixed. 2026-10-09: the DJ corpus placement
was wrong on key-locked records and is fixed (`DATASET_STATE.md` 2026-10-08/09); the seam reader was tested on
mixer renders; the nine type names are gone, replaced by measured moves plus a signature and the nearest mixer
recipe (`djdata types`). Both corpora were re-run on the VM with the new placement and types (2026-10-09 to 11; the run was OOM-killed twice
and finished at 20:38 German time on 2026-10-11, how in the memory note `vm-run-restart-2026-10-10`): DJ corpus 2,617 seams
measured and typed, 21 windows over 50 minutes left out as not transitions (`data/djdata/djs/out/measure_big_windows_2026-10-11.csv`),
Raveform 5,060. The tables are on the laptop, the VM and S3; the old ones in `out/before_2026-10-09/`. The VM is OFF. Current work: the
DJ data report notebook (`DJ_DATA_REPORT.md`, read its "State, 2026-10-13" section first): rebuilt on 2026-10-13 around
eight findings, chapters 1 to 7 drafted, the appendix not written; two enrichment passes (Spotify release years, Last.fm
listeners) fill two charts in the background; training decisions wait for the analysis. `SYSTEM_OVERVIEW.md` holds the
system as it runs today. The pick-up (verify the tables, choose the cast of famous DJs that runs through every
chapter, which charts go and which come, every remaining plot, the DJ music library for the website) is the last section of
`DJ_DATA_REPORT.md`, "PENDING, waiting on the VM run". Read it before touching the notebook. 2026-10-13: the dataset layer is
built (`DATA_ARCHITECTURE.md`): `djdata export-dataset` writes `data/dataset/<date>/`, every model and the planner read it
through `src/data/dataset.py`, ChromaDB and `split_mixes.csv` are gone, 1.4 GB of superseded files sit in `data/archive/2026-10-14/`. After a VM start, `sudo swapon /swapfile`. Never call the
old dataset "the catalog". Raveform's tier 1 and tier 2 are one dataset; the split was run order only.

## How Anas works, follow this

- **Say what you are about to do and wait for a go.** Even in auto mode, even inside a job he approved.
  Running things to "see if they work" has cost real time and got a site to block us.
- **Measure first, act second.** Never the reverse. A threshold is a measurement, not a constant.
- Report results in chat. He writes PLAN.md himself unless he asks for a section.
- Plain words in everything he reads: column names, files, variables. No `rf_b_in_s`. Times as 1:45.
- His ear is ground truth. No automatic check counts as proof.
- Bulk audio work runs on the VM where the files are, not over his home connection.
- Do not lose data: the sets were already trimmed once, so anything skipped gets written down and a list
  saved.
