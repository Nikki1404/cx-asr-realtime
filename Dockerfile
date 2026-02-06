#==========================
# --- STAGE 1: BUILDER ---
#==========================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /build

# 1. Standard build tools + git for NeMo
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and lock Cython for youtokentome
RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3.10 -m pip install --no-cache-dir "Cython<3.0"

# 2. Install Torch first (cached layer) (Keep this separate for massive caching benefits)
RUN python3.10 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchaudio==2.5.1

# 3. Install requirements
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# 4. Install NeMo (Pinned version via git tag) (with the isolation fix)
RUN python3.10 -m pip install --no-cache-dir --no-build-isolation \
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@v1.23.0"

#==========================
# --- STAGE 2: RUNTIME ---
#==========================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/srv \
    HF_HOME=/srv/hf_cache \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /srv

# 5. Runtime dependencies only (ffmpeg/libsndfile are critical for ASR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 6. THE ROBUST COPY (Fixes the crash risk)
# Copying /usr/local ensures that shared libraries (.so) and CLI tools (uvicorn)
# are preserved. It's only slightly larger but 100% reliable.
COPY --from=builder /usr/local /usr/local

# 7. Production Security: Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /srv
USER appuser

# 8. App Code
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser scripts ./scripts

EXPOSE 8002
CMD ["python3.10", "scripts/run_server.py", "--host", "0.0.0.0", "--port", "8002"]
