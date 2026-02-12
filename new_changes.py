FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# =====================================================
# OPTIONAL BUILD PROXY SUPPORT (EC2 corporate builds)
# =====================================================
ARG USE_PROXY=false
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV http_proxy=""
ENV https_proxy=""

RUN if [ "$USE_PROXY" = "true" ]; then \
        echo "🔐 Enabling proxy"; \
        export http_proxy=${HTTP_PROXY}; \
        export https_proxy=${HTTPS_PROXY}; \
    else \
        echo "🌐 Proxy disabled"; \
    fi

# =====================================================
# Runtime Environment
# =====================================================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/srv \
    HF_HOME=/srv/hf_cache \
    TRANSFORMERS_CACHE=/srv/hf_cache \
    TORCH_HOME=/srv/hf_cache \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /srv

# =====================================================
# System Dependencies
# =====================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    libsox-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# =====================================================
# Python Base Tools
# =====================================================
RUN python3 -m pip install -U pip setuptools wheel

# =====================================================
# Python Dependencies
# =====================================================
COPY requirements.txt /srv/requirements.txt

RUN python3.10 -m pip install --no-cache-dir -r /srv/requirements.txt

# =====================================================
# NeMo
# =====================================================
RUN python3 -m pip install --no-cache-dir \
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"

# =====================================================
# Torch (lock CUDA ABI)
# =====================================================
RUN python3 -m pip install --no-cache-dir --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 \
    torchaudio==2.5.1

# =====================================================
# Google Speech Dependencies
# =====================================================
RUN python3 -m pip install --no-cache-dir \
    google-cloud-speech \
    grpcio \
    grpcio-status

# =====================================================
# Copy App
# =====================================================
COPY app /srv/app
COPY scripts /srv/scripts

# =====================================================
# Copy Google Credentials INSIDE image
# =====================================================
COPY app/google_credentials.json /srv/google_credentials.json

# =====================================================
# Google ENV (baked inside image)
# =====================================================
ENV GOOGLE_APPLICATION_CREDENTIALS=/srv/google_credentials.json
ENV GOOGLE_RECOGNIZER=projects/eci-ugi-digital-ccaipoc/locations/us-central1/recognizers/google-stt-default
ENV GOOGLE_REGION=us-central1
ENV GOOGLE_LANGUAGE=en-US
ENV GOOGLE_MODEL=latest_short
ENV GOOGLE_INTERIM=true
ENV GOOGLE_EXPLICIT_DECODING=true

# =====================================================
# Expose Port
# =====================================================
EXPOSE 8002

# =====================================================
# Start Server
# =====================================================
CMD ["python3", "scripts/run_server.py", "--host", "0.0.0.0", "--port", "8002"]
