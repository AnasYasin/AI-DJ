"""
Phase 6 — Airflow DAG for the AI DJ training pipeline.

Architecture
────────────
Airflow (official lightweight image) orchestrates tasks by spawning sibling
containers from the ai-dj:latest ML image via DockerOperator. The Docker socket
is mounted in the Airflow container so it can reach the host Docker daemon.

This separation keeps Airflow lean (no torch/librosa/etc.) while each task
runs in a fully reproducible, isolated container with all ML dependencies.

  Airflow container (apache/airflow:2.9.0)
      │  /var/run/docker.sock        ← DockerOperator tasks
      │  SSH → host.docker.internal  ← SSHOperator scraper task
      ↓
  Host Docker daemon / Host bash
      ├── SSH exec: tracklists1001_client.py (real Chrome, conda env, host display)
      ├── spawns: fetch_previews    task (ai-dj:latest)
      ├── spawns: build_features    task (ai-dj:latest)
      ├── spawns: populate_chroma   task (ai-dj:latest)
      ├── spawns: label_transitions task (ai-dj:latest)
      └── spawns: train_models      task (ai-dj:latest)

All task containers join the ai-dj-net network so they can reach mlflow and
chromadb by service name (http://mlflow:5000, http://chromadb:8000).

Pipeline (linear chain)
───────────────────────
  scrape_mixesdb >> fetch_previews >> build_features >> populate_chroma
    >> label_transitions >> check_labels_exist >> train_models

Schedule: @weekly  (picks up newly scraped MixesDB data)
catchup:  False    (only the latest scheduled period runs)

Trigger manually:
  airflow dags trigger ai_dj_pipeline
  Or: http://localhost:8080
"""
from datetime import datetime, timedelta
import os
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# ── Image & network ────────────────────────────────────────────────────────────
# ai-dj:latest is built by `docker-compose build trainer`.
# ai-dj-net is declared in docker-compose.yml — all services join it so task
# containers can reach mlflow (port 5000) and chromadb (port 8000) by name.
ML_IMAGE   = "ai-dj:latest"
NETWORK    = "ai-dj-net"
DOCKER_URL = "unix://var/run/docker.sock"

# ── Host paths for volume mounts ───────────────────────────────────────────────
# DockerOperator spawns sibling containers on the HOST Docker daemon, so volume
# source paths must be absolute paths on the HOST — not paths inside the Airflow
# container. HOST_PROJECT_ROOT is set to ${PWD} in docker-compose.yml.
_HOST_ROOT = os.environ.get("HOST_PROJECT_ROOT", "")

_MOUNTS = [
    Mount(source=f"{_HOST_ROOT}/data",   target="/app/data",   type="bind"),
    Mount(source=f"{_HOST_ROOT}/models", target="/app/models", type="bind"),
]

# ── Path for ShortCircuitOperator (runs inside Airflow, uses read-only mount) ──
# ./data is mounted read-only at /opt/airflow/host_data in the Airflow container.
_AIRFLOW_DATA = "/opt/airflow/host_data"

# ── Shared DockerOperator kwargs ───────────────────────────────────────────────
_DOCKER_DEFAULTS = dict(
    image=ML_IMAGE,
    docker_url=DOCKER_URL,
    network_mode=NETWORK,
    working_dir="/app",
    environment={
        "PYTHONPATH": "/app",
        # Task containers reach these services by name via ai-dj-net
        "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        "CHROMA_HOST":         "chromadb",
        "CHROMA_PORT":         "8000",
    },
    mounts=_MOUNTS,
    auto_remove=True,   # clean up the task container after it exits
    mount_tmp_dir=False,
)

# ── Default task arguments ─────────────────────────────────────────────────────
_DEFAULT_ARGS = {
    "owner":            "ai-dj",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}


with DAG(
    dag_id="ai_dj_data_pipeline",
    description=(
        "AI DJ data pipeline: "
        "1001Tracklists → previews → MERT features → ChromaDB → heuristic labels"
    ),
    schedule_interval="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=_DEFAULT_ARGS,
    tags=["ai-dj", "training"],
    doc_md=__doc__,
) as dag:

    # ── 1. Guard: tracklist CSV must exist before pipeline runs ──────────────
    # tracklists1001_client.py requires a real browser (nodriver) — run it
    # manually on the host before triggering this DAG:
    #   conda activate djtest && python src/data/tracklists1001_client.py
    def _check_tracklist(**_):
        csv = Path(f"{_AIRFLOW_DATA}/interim/tracklist.csv")
        if not csv.exists():
            raise FileNotFoundError(
                "data/interim/tracklist.csv not found. "
                "Run on host: conda activate djtest && python src/data/tracklists1001_client.py"
            )

    scrape_mixes = PythonOperator(
        task_id="check_tracklist_exists",
        python_callable=_check_tracklist,
        doc_md="""
**check_tracklist_exists** — manual prerequisite guard

`tracklists1001_client.py` requires a real browser window. Run it manually
on the host before triggering this DAG:

  conda activate djtest && python src/data/tracklists1001_client.py

This task fails fast with a clear message if the CSV is missing.
        """,
    )

    # ── 2. Fetch previews ──────────────────────────────────────────────────────
    fetch_previews = DockerOperator(
        task_id="fetch_previews",
        command="python src/data/preview_fetcher.py",
        execution_timeout=timedelta(hours=5),
        retries=2,   # network requests can flake
        doc_md="""
**fetch_previews** — Phase 1

Downloads 30s audio previews for all tracks in `data/interim/*.csv`.

- Primary source: iTunes Search API (free, no auth required)
- Fallback: Spotify preview_url (deprecated for most tracks since late 2023)
- Output: `data/raw/preview_manifest.csv` + `data/raw/previews/`
- Idempotent: skips already-downloaded tracks
        """,
        **_DOCKER_DEFAULTS,
    )

    # ── 3. Validate previews ──────────────────────────────────────────────────
    validate_previews = DockerOperator(
        task_id="validate_previews",
        command="python src/data/validate_previews.py",
        execution_timeout=timedelta(hours=1),
        doc_md="""
**validate_previews** — Phase 1b

Checks every downloaded audio file for corruption (soundfile header check).
Corrupt files are re-downloaded from iTunes; unfixable files are removed and
marked as not_found in the manifest. Prevents build_features from hanging
on broken MP3s.
        """,
        **_DOCKER_DEFAULTS,
    )

    # ── 4. Build features ──────────────────────────────────────────────────────
    build_features = DockerOperator(
        task_id="build_features",
        command="python src/features/build_features.py",
        execution_timeout=timedelta(hours=6),   # MERT inference is slow on CPU
        doc_md="""
**build_features** — Phase 2

Extracts two feature sets per track from the 30s preview:

1. **MERT embedding** (768-dim): mean-pooled last hidden state of
   `m-a-p/MERT-v1-95M` — captures sub-genre timbral differences.
2. **Librosa features**: BPM, key (Krumhansl-Schmuckler), LUFS, energy
   mean/std, spectral centroid, onset strength, MFCC (13 coefficients).

Output: `data/processed/features.parquet` (~785 columns per track)
Idempotent: skips tracks already in the parquet file.
        """,
        **_DOCKER_DEFAULTS,
    )

    # ── 4. Compute mix metadata ───────────────────────────────────────────────
    compute_mix_metadata = DockerOperator(
        task_id="compute_mix_metadata",
        command="python src/features/mix_profiler.py",
        execution_timeout=timedelta(minutes=10),
        doc_md="""
**compute_mix_metadata** — Phase 9

Reads tracklist.csv + features.parquet, computes an energy curve shape
for each mix, and writes `data/processed/mix_metadata.csv`.

Shapes: escalating / chill-down / peak-drop / wave / plateau
Used by label_transitions to add mix_energy_curve_shape to each pair.
        """,
        **_DOCKER_DEFAULTS,
    )

    # ── 5. Populate ChromaDB ───────────────────────────────────────────────────
    populate_chroma = DockerOperator(
        task_id="populate_vector_store",
        command="python src/features/vector_store.py",
        execution_timeout=timedelta(hours=1),
        doc_md="""
**populate_vector_store** — Phase 3

Loads MERT embeddings from `features.parquet` into ChromaDB (HNSW index,
cosine similarity). Enables O(log N) nearest-neighbor search for semi-hard
negative mining during contrastive encoder training.

Semi-hard window: cosine distance [0.10, 0.60] — see CLAUDE.md.
Output: `data/processed/chromadb/`
Upserts are safe — re-running after new tracks are added will update.
        """,
        **_DOCKER_DEFAULTS,
    )

    # ── 5. Label transitions ───────────────────────────────────────────────────
    label_transitions = DockerOperator(
        task_id="label_transitions",
        command="python src/features/transition_labeler.py",
        execution_timeout=timedelta(minutes=30),
        doc_md="""
        
**label_transitions** — Phase 4

Applies heuristic rules to consecutive track pairs from `data/interim/*.csv`.
Classifies each transition using audio feature deltas (priority order):

  slam   — energy spike or key clash
  rise   — positive energy delta, compatible key, loose BPM
  fade   — negative energy delta
  melt   — tight BPM, same key, minimal energy + loudness shift
  wave   — high onset strength on both tracks, tight BPM
  blend  — default catch-all

Output: `data/processed/transition_labels.csv`
        """,
        **_DOCKER_DEFAULTS,
    )

#     # ── 6. Guard: labels must exist before training ────────────────────────────
#     # Runs inside Airflow (not a container). Reads from the read-only data mount
#     # at /opt/airflow/host_data. Short-circuits downstream tasks if labels are
#     # missing — protects train_models from running with no classifier labels.
#     check_labels = ShortCircuitOperator(
#         task_id="check_labels_exist",
#         python_callable=lambda: Path(
#             f"{_AIRFLOW_DATA}/processed/transition_labels.csv"
#         ).exists(),
#         doc_md="""
# **check_labels_exist** — guard
#
# Short-circuits the pipeline if `transition_labels.csv` is missing.
# Runs inside Airflow (not a Docker container) using the read-only data mount.
#
# If skipped: re-run `label_transitions` and check feature coverage of mix CSVs.
#         """,
#     )
#
#     # ── 7. Train models ────────────────────────────────────────────────────────
#     train_models = DockerOperator(
#         task_id="train_models",
#         command="python src/models/train_model.py",
#         execution_timeout=timedelta(hours=6),
#         retries=0,   # training is expensive — do not retry automatically
#         doc_md="""
# **train_models** — Phase 5
#
# Trains two models sequentially:
#
# **Contrastive encoder** → `models/contrastive_encoder.pt`
# - MLP 768 → 256 → 128 (L2-normalised, unit hypersphere)
# - NT-Xent loss (temperature=0.07)
# - Positives: consecutive tracks in a DJ mix
# - Negatives: semi-hard negatives from ChromaDB (distance window [0.10, 0.60])
#
# **Transition classifier** → `models/transition_classifier.pt`
# - MLP 260 → 128 → 64 → 6
# - Input: concat(emb_A[128], emb_B[128], bpm_ratio, energy_delta, harm_dist, time_gap)
# - Output: slam / rise / fade / melt / wave / blend
#
# Logs all params and metrics to MLflow (http://localhost:5000) and W&B.
#         """,
#         **_DOCKER_DEFAULTS,
#     )

    # ── Pipeline dependency chain ──────────────────────────────────────────────
    (
        scrape_mixes
        >> fetch_previews
        >> validate_previews
        >> build_features
        >> compute_mix_metadata
        >> populate_chroma
        >> label_transitions
        # >> check_labels
        # >> train_models
    )
