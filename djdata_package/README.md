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

## Run (Raveform source)

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
