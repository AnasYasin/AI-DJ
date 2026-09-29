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
| how to run the seam pipeline, its stages, tables, configs, how to add data, the watch list for a run | **`DATA_PIPELINE.md`** |
| the state of the data: what is measured, what was found, what is closed, the numbers and their dates | **`DATASET_STATE.md`** |
| every store, every CSV, how the ids join, what lives on the laptop, the VM and S3, the planned layout | **`DATA_ARCHITECTURE.md`** |
| the djdata package: install, stages, adding a corpus, reading the tables | **`djdata_package/README.md`** |
| Fred again, a separate job: his sources, segments, what is proved | **`FRED_AGAIN_SETS.md`** |
| how to set the VM up, cookies, yt-dlp, the token server, the changing address | **`djdata_package/VM_SETUP.md`** |
| ssh to the VM whatever its address is today | `scripts/vm/vmssh` |
| how to run the mixer end to end, the product side | **`README.md`** |
| old session records and superseded notes, kept whole with their dates | `docs/notes-archive/` |

New kinds of work, for example model fine-tuning or retraining, get their own md and a row here.

## State in one paragraph (2026-09-29)

The DJ profiling run over every DJ with Fred again.. is done (285 mixes, 3,365 seams, 2,459 cut,
measured and labelled, layer tables for the stacking DJs; `DATASET_STATE.md` 2026-09-29 for the numbers,
`DATA_PIPELINE.md` for how it runs). The tables are on the VM in `data/djdata/djs/out/` and on S3 at
`s3://aidj-1/djdata_djs/out_2026-09-28/`. Next: Raveform, starting with feature extraction. Then the old
`scripts/diag` code goes. Never call the old dataset "the catalog". Raveform's tier 1 and tier 2 are one
dataset; the split was run order only.

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
