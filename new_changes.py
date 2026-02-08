FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ARG USE_PROXY=false
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

# Proxy (set ONCE if needed)
ENV http_proxy=""
ENV https_proxy=""

RUN if [ "$USE_PROXY" = "true" ]; then \
        echo "🔐 Enabling proxy for builder stage"; \
        export http_proxy=${HTTP_PROXY}; \
        export https_proxy=${HTTPS_PROXY}; \
    else \
        echo "🌐 Proxy disabled for builder stage"; \
    fi

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# Python tooling
RUN python3.10 -m pip install --no-cache-dir -U pip setuptools wheel && \
    python3.10 -m pip install --no-cache-dir "Cython<3.0"

# Torch + Torchaudio (LOCKED FIRST)
RUN python3.10 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 \
    torchaudio==2.5.1

# App dependencies
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# NeMo (PINNED, ABI SAFE)
RUN python3.10 -m pip install --no-cache-dir --no-build-isolation \
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@v1.23.0"

#  MODEL PRELOAD + CACHE
ENV HF_HOME=/srv/hf_cache \
    TRANSFORMERS_CACHE=/srv/hf_cache \
    TORCH_HOME=/srv/hf_cache \
    NEMO_CACHE_DIR=/srv/hf_cache

RUN python3.10 - << 'PY'
import os
print("📦 HF_HOME:", os.environ["HF_HOME"])

# Whisper
print("⬇️  Preloading Whisper Turbo...")
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo")

# Nemotron
print("⬇️  Preloading Nemotron...")
import nemo.collections.asr as nemo_asr
nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-speech-streaming-en-0.6b"
)

print("✅ Model preload complete")
PY

# --- STAGE 2: RUNTIME ---
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/srv \
    HF_HOME=/srv/hf_cache \
    TRANSFORMERS_CACHE=/srv/hf_cache \
    TORCH_HOME=/srv/hf_cache \
    NEMO_CACHE_DIR=/srv/hf_cache \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /srv

# Runtime-only deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Python env + models
COPY --from=builder /usr/local /usr/local
COPY --from=builder /srv/hf_cache /srv/hf_cache

# Security: non-root
RUN useradd -m appuser && chown -R appuser:appuser /srv
USER appuser

# App code (last → keeps cache!)
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts

EXPOSE 8002
CMD ["python3.10", "scripts/run_server.py", "--host", "0.0.0.0", "--port", "8002"]

#docker build --build-arg USE_PROXY=true --build-arg HTTP_PROXY="http://163.116.128.80:8080" --build-arg HTTPS_PROXY="http://163.116.128.80:8080" -t cx_asr_realtime .
