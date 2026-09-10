# AI DJ

Generates a real mixed audio file (.mp3) from a description of what you want to hear.

> "high energy techno set, slow build" → ordered tracklist → mixed .mp3

Trained on ~2,800 real DJ sets (1001tracklists) across six electronic genres:
techno, trance, tech house, drum and bass, melodic house, afro house.

## How it works

```
scrape tracklists → fetch 30s previews → discogs-effnet embeddings + BPM/key/energy
→ contrastive track encoder (Model A) → sequence transformer (Model B)
→ GBM edge scorer + hard mixing rules → beam-search set planner
→ structural segmenter (intro/buildup/drop/breakdown/outro) → mixer → .mp3
```

The planner picks and orders tracks: candidates filtered by genre and BPM, ranked
by Model B's context vector (what fits *after the tracks already played*), pruned
by hard rules (BPM ±5%, Camelot-compatible keys, one track per artist), scored by
the GBM, searched with a beam. An energy curve (build / peak / wave / chill / arc)
shapes the set.

The fetcher turns the plan into files. It searches YouTube for each track and
accepts a download only if the decoded audio is at least four minutes long and a
constellation fingerprint locates the track's own 30-second preview inside it, so
radio edits and wrong recordings never reach the mixer.

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

What each learned part contributes (validated on held-out mixes, no leakage):

| Component | What it does | Result |
|---|---|---|
| Model A (contrastive encoder) | 128-dim "mixability" space for retrieval | 0.663 AUC (BPM alone: 0.624) |
| Model B (sequence transformer) | picks next track given the set so far | +17% MRR over context-free |
| Edge scorer (GBM) | scores a single transition | 0.685 AUC |
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

# retrain everything (CPU is fine, ~10 min each)
python -m src.models.train_model      # Model A
python -m src.models.train_sequence   # Model B
python -m src.models.edge_scorer      # GBM
```

## What's not done yet

- Jamendo as an alternative full-audio source. YouTube works but its audio is
  already lossy, so the render is a second generation of loss
- Natural-language intent parsing (one Claude call) and the FastAPI service
- DJ style profiles / archetype conditioning
- v2 research: aligning mix audio to recover real transitions, transition critic

## Stack

- **Embeddings**: discogs-effnet (frozen) + librosa/DeepRhythm features
- **Models**: PyTorch (contrastive MLP, causal transformer), scikit-learn (GBM)
- **Vector store**: ChromaDB (HNSW)
- **Tracking**: MLflow + W&B
- **Audio**: librosa, essentia, scipy (EQ), ffmpeg
- **Orchestration**: Airflow
