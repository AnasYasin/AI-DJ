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

The mixer turns the plan into audio. Each track plays a ~3-minute phrase-aligned
window chosen against the energy curve (low energy → windows that avoid drops),
not the whole track. Every transition is classified — slam / rise / fade / melt /
wave / blend — from the two tracks' energy, onset, key and BPM, then rendered by
its own recipe: bar-aligned overlaps with per-band EQ automation (bass swaps,
3-band fades, staggered filter-style sweeps).

What each learned part contributes (validated on held-out mixes, no leakage):

| Component | What it does | Result |
|---|---|---|
| Model A (contrastive encoder) | 128-dim "mixability" space for retrieval | 0.663 AUC (BPM alone: 0.624) |
| Model B (sequence transformer) | picks next track given the set so far | +17% MRR over context-free |
| Edge scorer (GBM) | scores a single transition | 0.685 AUC |
| Segmenter | phrase-aligned section maps + per-bar energy | drives cue points & windows |
| Mixer | typed transitions, EQ automation, time-stretch | renders the actual audio |

## Quick start

```bash
conda activate aidj

# plan a set from intent
python -m src.models.predict_model --genre techno --bpm 126 136 --n 10 --curve build

# render full tracks into one mix
python -m src.audio.audio_mixer out.mp3 track1.mp3 track2.mp3 --genre techno

# retrain everything (CPU is fine, ~10 min each)
python -m src.models.train_model      # Model A
python -m src.models.train_sequence   # Model B
python -m src.models.edge_scorer      # GBM
```

## What's not done yet

- Planner and mixer aren't connected: the training catalog is 30s previews, so
  rendering a planned set needs a full-audio source (Jamendo — API phase)
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
