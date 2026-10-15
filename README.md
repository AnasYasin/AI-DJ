# AI DJ

Generates a real mixed audio file (.mp3) from a description of what you want to hear.

> "high energy techno set, slow build" → ordered tracklist → mixed .mp3

Trained on 3,937 real DJ sets and 65,513 plays from three sources (1001tracklists
tracklists, the Raveform dataset, and the DJ data pipeline below) across seven
electronic genres: techno, tech house, house, trance, drum and bass, melodic house,
afro house. 12,153 of the 37,612 records are held as full audio.

## Data pipeline (djdata)

`djdata_package/` turns real DJ mixes and the records in them into a labelled seam table. Eight
stages, one command each, resumable: `locate` finds every listed record in its mix at its speed,
`pairs` says which record follows which from what was heard, `cut` cuts and audits one window per
seam, `measure` reads per band entry and exit, the bass swap and the bass words, `tempo` and `label`
turn that into transition types in bars, `export-seams` writes one flat table, `ear-test` writes
clips for a listen. Every threshold is a floor measured on control records in the same audio, and
every number sits beside its floor. `START_HERE.md` says how to run it and read the tables,
`DATASET_STATE.md` what was measured and heard.

## How it works

```
scrape tracklists → fetch 30s previews → discogs-effnet embeddings + BPM/key/energy
→ pair model (which record follows which, given the set so far)
→ hard mixing rules → beam-search set planner
→ structural segmenter (intro/buildup/drop/breakdown/outro) → mixer → .mp3
```

The planner picks and orders tracks. Candidates are filtered by genre and BPM,
pruned by hard rules (BPM within 5 %, no repeated record, one record per artist),
scored by the pair model and the fit to an energy curve (build / peak / wave /
chill / arc), and searched with a beam. There is no key rule: real sets clash on
58 % of joins, and the pair model sees the Camelot distance itself. The opener is
chosen from the slower half of the pool, where 71 % of real sets open.

The fetcher turns the plan into files. A record held as full audio is copied from
disk or S3. A Raveform record is downloaded by its YouTube id. Anything else is
searched on YouTube, and a download is accepted only if the decoded audio is at
least four minutes long and a constellation fingerprint locates the track's own
30-second preview inside it, so radio edits and wrong recordings never reach the
mixer. A fetched record is registered as held, so it is never searched again.

The mixer turns those files into audio. Each track plays a ~3-minute
phrase-aligned window chosen against the energy curve (low energy → windows that
avoid drops), not the whole track. How long it plays comes from the genre's
measured median across 1,665 real sets, and flexes per track to start and end on
section boundaries. Every transition is classified — slam / rise /
fade / melt / wave / blend / drop — from the two tracks' energy, onset, key and
BPM, then gated by the set's energy curve, so a chill set never fires a slam.
Each type has its own recipe: bar-aligned overlaps with per-band EQ automation
(bass swaps, 3-band fades, staggered filter-style sweeps). A `drop` transition
brings the incoming track in over its own buildup with the bass cut and the
filter closed, and lands the bass swap on the drop itself.

Beat alignment is measured rather than assumed. The two grids are phase-corrected
against real kick transients, then the residual at each seam is measured by kick
envelope cross-correlation and removed, which took the test seams from 180-215 ms
out to 0.1 ms.

What each part contributes (the model is scored on DJs it never saw in training):

| Component | What it does | Result |
|---|---|---|
| Pair model | scores the next record given the last one and the set so far | true next record in the top 10 of 100 for 0.649 of pairs, AUC 0.876, on 58 DJs never trained on |
| Segmenter | phrase-aligned section maps + per-bar energy | drives cue points & windows |
| Mixer | typed transitions, EQ automation, time-stretch | renders the actual audio |
| Fetcher | plan → verified full audio | duration + fingerprint gated |

## Quick start

```bash
conda activate aidj

# plan a set from intent
python -m src.models.predict_model --genre techno --bpm 126 136 --n 10 --curve build

# fetch and verify full audio for that plan
python -m src.models.predict_model --genre techno --bpm 126 136 --n 10 --json plan.json
python -m src.data.track_fetcher --plan plan.json --out data/external/run1

# render the plan, which carries the genre, the curve and the per-track
# energy targets through to the mixer (.flac for a lossless render)
python -m src.audio.audio_mixer out.flac --plan plan.json --tracks-dir data/external/run1

# or render loose files directly
python -m src.audio.audio_mixer out.flac track1.mp3 track2.mp3 --genre techno --curve build

# plan, fetch and render in one command, from held audio only
python -m scripts.make_mix --genre techno --bpm 126 132 --curve chill --n 6 \
    --minutes 20 --audio full --out mix.flac

# retrain the pair model (CPU, about 6 h for 5 seeds)
python -m src.models.pair_model --seeds 5
```

## What's not done yet

- Jamendo as an alternative full-audio source. YouTube works but its audio is
  already lossy, so the render is a second generation of loss
- Natural-language intent parsing (one Claude call) and the FastAPI service
- Transition types and recipes from the measured seams, per DJ and per genre
  (`TRANSITION_EXPERIMENTS.md`); today the mixer's rules choose them
- Play length and cue points from the measured plays
- DJ style profiles, and a club or radio setting (`TODO.md`)

## Stack

- **Embeddings**: discogs-effnet (frozen) + librosa/DeepRhythm features
- **Model**: PyTorch (the pair model, a record encoder and a pair head)
- **Data**: one dated dataset folder of four parquet files, built from the three sources by `djdata export-dataset` (`DATA_ARCHITECTURE.md`)
- **Tracking**: MLflow + W&B
- **Audio**: librosa, essentia, scipy (EQ), ffmpeg
- **Orchestration**: Airflow
