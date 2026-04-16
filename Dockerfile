# ── Base image ─────────────────────────────────────────────────────────────────
# Pin Python 3.10 to match pyproject.toml constraint.
# slim = no build tools, smaller image; we add only what we need.
FROM python:3.10-slim

# ── System dependencies ────────────────────────────────────────────────────────
# ffmpeg: lets librosa/soundfile decode MP3 files (not just WAV)
# libsndfile1: C library that soundfile wraps — required at runtime
# git: needed by some HuggingFace transformers downloads
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ────────────────────────────────────────────────
# Copy requirements BEFORE source code.
# Why: Docker caches each layer. If only code changes (not deps),
# this layer is reused and pip install is skipped — much faster rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ───────────────────────────────────────────────────────────
COPY . .

# ── Environment ────────────────────────────────────────────────────────────────
# PYTHONPATH=/app lets Python find `src/` as a package root without install tricks.
ENV PYTHONPATH=/app

# ── Default command ────────────────────────────────────────────────────────────
# Override this per-service in docker-compose.yml.
CMD ["python", "--version"]
