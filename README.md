# AI DJ

Generates a real mixed audio file (.mp3) from a natural language prompt.

> "high energy techno set for 2 hours" → mixed .mp3

## How it works

1. Scrape DJ mixes from 1001tracklists to extract tracklists
2. Fetch 30s audio previews (iTunes) and extract MERT embeddings + librosa features
3. Store embeddings in ChromaDB for nearest-neighbour search
4. Generate heuristic transition labels (slam / rise / fade / melt / wave / blend)
5. Train a contrastive encoder that learns track mixability, and a transition classifier
6. At inference: parse user prompt (Claude API) → select tracks from Jamendo → detect cue points → render .mp3 with BPM sync + bar-aligned crossfades + EQ blending

## Stack

- **Embeddings**: MERT-v1-95M (frozen) + librosa
- **Vector store**: ChromaDB (HNSW)
- **Training**: NT-Xent contrastive loss, MLflow + W&B
- **Orchestration**: Airflow
- **Audio rendering**: pyrubberband (BPM sync), scipy (EQ filters), pydub

## Running the pipeline

```bash
# 1. Scrape tracklists (requires real browser — run on host)
conda activate djtest
python src/data/tracklists1001_client.py

# 2. Start services
docker compose up mlflow chromadb airflow

# 3. Trigger the DAG
# Airflow UI → http://localhost:8080 → ai_dj_data_pipeline → Trigger
```
