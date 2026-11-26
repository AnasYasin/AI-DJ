# AI DJ

Builds a beat-matched stereo DJ mix from a genre, a BPM range and an energy curve, or in the style
of real DJs.

```
real DJ mixes → measured tracks and transitions → pair model → set plan
→ full audio fetched and verified → mixed .flac or .mp3
```

## Data

| Genre | Mixes | DJs | Tracks | Measured transitions |
|---|---:|---:|---:|---:|
| Tech house | 903 | 168 | 8,889 | 2,585 |
| Trance | 652 | 61 | 6,816 | 751 |
| Techno | 643 | 27 | 5,838 | 1,187 |
| Drum and bass | 564 | 99 | 7,194 | 777 |
| Melodic house | 526 | 10 | 3,739 | 271 |
| Afro house | 414 | 9 | 2,981 | 156 |
| House | 235 | 116 | 2,155 | 1,004 |
| **All** | **3,937** | **473** | **37,612** | **6,731** |

![Mixes per genre, and the twenty DJs with the most sets](docs/figures/mixes_per_genre.png)

## Pipeline

- **Data.** Three sources: 1001tracklists, the Raveform dataset and the DJ data pipeline
  (`djdata_package/`), which finds each track inside real DJ mixes and measures every transition.
- **Planner.** The pair model scores which track should come next, given the set so far. A beam
  search orders the set along the energy curve. On 58 DJs it never trained on, the real next track
  is in its top 10 of 100 for 65 % of pairs.
- **Styles.** The measured transitions sort the DJs of each genre into 15 style groups, for example
  techno DJs who blend for long and techno DJs who cut. A set asked for in a style takes its tracks
  from what those DJs play, and its share of each transition type, its overlap lengths and its play
  times from their measured sets.
- **Fetcher.** Copies held tracks from disk or S3 and downloads the rest from YouTube. A download is
  kept only when its fingerprint matches the track's own preview.
- **Mixer.** Plays a phrase-aligned section of each track, picks a transition type, aligns the beats
  at every transition and renders in stereo at −14 LUFS.

## Quick start

```bash
conda activate aidj

# plan, fetch and render in one command
python -m scripts.make_mix --genre techno --bpm 126 132 --curve chill --n 6 \
    --minutes 20 --audio full --out mix.flac
```

A set in the style of real DJs, by style group or by DJ name:

```bash
python -m scripts.make_mix --genre techno --style auto --bpm 126 134 --n 8 --out mix.flac
python -m scripts.make_mix --genre techno --style blend cut --n 8 --out mix.flac
python -m scripts.make_mix --dj "fred again" --n 6 --out mix.flac
```

Each stage also runs alone: `src.models.predict_model`, `src.data.track_fetcher`,
`src.audio.audio_mixer`.

## Not built yet

- A natural-language prompt and an API
- A critic that judges a rendered transition
- A club or radio setting
- A named DJ's stops and restarts between tracks, and cue points learned per track

## Docs

`START_HERE.md` is the index. It points to the file for each stage.

## Stack

PyTorch, discogs-effnet, librosa, essentia, rubberband, ffmpeg, Airflow, S3.
