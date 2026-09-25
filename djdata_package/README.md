# djdata

Data mining for the AI-DJ project: build a manifest of real DJ transitions (seams), fetch the mix
and the two original tracks for each, and extract what the DJ did on the mixer, beat by beat.
Resumable, parallel, one config file, three CSV outputs. Every stage is a function, so Airflow can
call it later.

## Install

```
pip install -e ./djdata_package
sudo apt install ffmpeg            # cutting windows, durations
sudo apt install xvfb              # only for the two browser stages (1001tracklists pages)
```

## The profiling pipeline (2026-09-25)

One stage per subcommand, in order, each resumable, each writing one table under `root/out/`. The
same commands run every corpus: the DJ corpus (`config_djs.yaml`), Fred (`config_fred.yaml`) and
Raveform (`config.yaml`, where the shipped alignment stands in for locate and the windows the
manifest cut stand in for the cut).

```
djdata locate       --config config_djs.yaml --workers 16   every listed record found in its mix   -> out/plays.csv
djdata pairs        --config config_djs.yaml                which record follows which, windows     -> out/seams.csv
djdata cut          --config config_djs.yaml --workers 16   windows stream copied and audited       -> out/cuts.csv
djdata measure      --config config_djs.yaml --workers 16   bands, bass, presence, loop per seam    -> out/measures.csv
djdata tempo        --config config_djs.yaml --workers 16   BPM per record from its own audio       -> out/tempos.csv
djdata label        --config config_djs.yaml                the nine transition types, in bars      -> out/labels.csv
djdata export-seams --config config_djs.yaml                one flat table joining all of the above -> out/seams_index.csv
```

The tables are the dataset. `state.sqlite` is a resume log nobody reads. Every number sits next to
the control floor that judged it. Times are seconds into the mix in `plays.csv` and seconds into
the window everywhere else. The rules, thresholds and what was measured on them are in the module
docstrings under `djdata/seam/` and in `DATASET_STATE.md`.

Tests: `cd djdata_package && pytest tests -q`. They build synthetic records and mixes with a known
answer (`tests/synth.py`), no audio files and no network. `make ci` at the repo root runs them.

Heard 2026-09-25: the ear-test clips of the third development pass were right by Anas's ear, so the
locate, pairing, cut and labels hold on real DJ audio. `DATASET_STATE.md` has the numbers.

### Adding a corpus or a DJ

A corpus is a folder with `mixes_tmp/<mix_id>.<ext>`, `tracks/<track_id>.<ext>` and the tracklist
CSV named in its config (`tracklists_csv`, one row per listed record in play order). Audio is found by
id on disk, never by a stored path. A new DJ is more rows in the tracklist and more audio; the stages
pick up what is on disk and skip what their table already holds. A Raveform-like source that ships an
alignment sets `source: raveform` and the alignment stands in for locate. Adding a source is one file
in `djdata/sources/` that answers two questions: which mixes are there, and where is each one's audio.

### Reading the tables

`plays.csv` one row per record play (and per control record, `is_control` 1). `seams.csv` one row per
consecutive pair, usable or not, with the reason. `cuts.csv` the window file and its audit.
`measures.csv` the seam numbers, each beside its floor. `tempos.csv`, `labels.csv`. `seams_index.csv`
joins seams, cuts, measures and labels on `seam_id`. Times: seconds into the mix in `plays.csv`,
seconds into the window elsewhere. Filter on `measured == 1` and the separations before using a number.

## Run (Raveform source, the older fetch path)

```
djdata manifest   --config config.yaml   # Raveform files → state db (seams, mixes, tracks, tiers)
djdata probe      --config config.yaml   # 20 YouTube downloads from THIS machine. Do this first on a VM.
djdata run        --config config.yaml   # fetch + analyse, resumable: rerun the same command after a kill
djdata status     --config config.yaml
djdata export     --config config.yaml   # out/seams.csv, out/curves.csv, out/qa.csv
```

Files under `root` (config.yaml):

```
state.sqlite      the resume log: every mix, track and seam with its status
mixes_tmp/        full mixes, deleted as soon as their seam windows are cut
windows/          one audio file per seam (codec copy of the mix, no re-encode)
tracks/           one file per track id
out/seams.jsonl   one json row per analysed seam (append only)
out/curves/       one csv per seam, one row per beat
out/*.csv         export
logs/djdata.log   one line per event with worker name, stage, id, elapsed seconds
```

## Workers

`workers.mix` should stay 1. It waits while `max_pending_mixes` downloaded mixes still have
unanalysed seams, so the disk never fills with windows nobody is ready to use. `workers.tracks`
YouTube streams run in parallel (4 is safe from one IP). `workers.seams` analysis processes take any
seam whose window and both tracks exist, so analysis starts with the first pair of tracks.
Measured on a laptop: a seam is about 30 s of analysis, a mix 6.6 seams and 8 new tracks.

## Moving finished audio to S3

```
djdata archivable --config config.yaml           # json: tracks and windows no unfinished seam still needs
# move them, then record it so nothing looks for them again:
djdata archived   --config config.yaml path/to/file1 path/to/file2 ...
```

## DJs Raveform does not have (Black Coffee, Fred again..)

Same pipeline, different manifest. Needs the 1001tracklists scrape (legacy scraper, kept in
`djdata/legacy/`) and the mix pages' media links. Both browser stages need a display:

```
xvfb-run -a djdata scrape-tracklists --dj https://www.1001tracklists.com/dj/fredagain../index.html --genre house
# set source: tracklists in config.yaml, then
djdata manifest    --config config.yaml --djs "Black Coffee" "Fred again.."
xvfb-run -a djdata media-links --config config.yaml
djdata run         --config config.yaml
```

For these seams the coarse anchors come from the audio (landmark fingerprint of a 60 s piece of the
window inside each track) before the fine alignment runs. This path is written but has not been run
end to end yet.

## Airflow

One task per subcommand, in this order: manifest → run → export. `run` is idempotent and resumable,
so a retry is safe. For the tracklists source add scrape-tracklists and media-links before manifest,
wrapped in `xvfb-run -a`.

## Method

See `djdata/seam/align.py`, `gains.py`, `params.py` docstrings. Measured accuracy on rendered seams
with known truth: alignment within 9 ms from a coarse guess wrong by a bar; outgoing record's band
gains within 1 dB; bass swap exact in 13 of 14 runs; -6 dB crossings within 2 beats (outgoing) and
5 beats (incoming). Entry of a slow bed is uncertain by a bar or more, so use the -6 dB crossing.
The high band of the quieter record is not trusted below about -10 dB.
