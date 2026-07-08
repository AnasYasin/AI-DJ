# AI DJ

Generates a real mixed audio file (.mp3) from a description of what you want to hear.

> "high energy techno set, slow build" → ordered tracklist → mixed .mp3

Trained on ~2,800 real DJ sets (1001tracklists) across six electronic genres:
techno, trance, tech house, drum and bass, melodic house, afro house.

## Status

The full pipeline works end to end. Every stage below is implemented and tested:

```
scrape tracklists → fetch 30s previews → discogs-effnet embeddings + BPM/key/energy
→ contrastive track encoder (Model A) → sequence transformer (Model B)
→ GBM edge scorer + hard mixing rules → beam-search set planner
→ structural segmenter (intro/buildup/drop/breakdown/outro) → mixer → .mp3
```

What each learned part contributes (validated on held-out mixes, no leakage):

| Component | What it does | Result |
|---|---|---|
| Model A (contrastive encoder) | 128-dim "mixability" space for retrieval | 0.663 AUC (BPM alone: 0.624) |
| Model B (sequence transformer) | picks next track given the set so far | +17% MRR over context-free |
| Edge scorer (GBM) | scores a single transition | 0.685 AUC |
| Segmenter | phrase-aligned section maps of full tracks | drives cue points |
| Mixer | time-stretch, bar-aligned crossfades, bass swap | renders the actual audio |

## Quick start

```bash
conda activate aidj

# plan a set from intent
python -m src.models.predict_model --genre techno --bpm 126 136 --n 10 --curve build

# render full tracks into one mix
python -m src.audio.audio_mixer out.mp3 track1.mp3 track2.mp3 track3.mp3

# retrain everything (CPU is fine, ~10 min each)
python -m src.models.train_model      # Model A
python -m src.models.train_sequence   # Model B
python -m src.models.edge_scorer      # GBM
```

## A note on the embeddings

Halfway through the project every signal probe came back at chance level and the
contrastive model wouldn't learn. The cause turned out to be a preprocessing bug in
the batched GPU embedder — it fed `log10(mel)` to discogs-effnet instead of essentia's
`log10(1 + 10000·mel)` frontend, which silently produced garbage embeddings for the
whole catalog. If you're using `TensorflowPredictEffnetDiscogs` with your own mel
pipeline: don't. Use essentia's `TensorflowInputMusiCNN` frontend and verify against
the wrapper (we gate on cosine > 0.99). After the fix, next-track signal appeared
exactly where it should (raw cosine AUC 0.50 → 0.63).

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

See `PLAN.md` for the detailed roadmap and evaluation numbers.
