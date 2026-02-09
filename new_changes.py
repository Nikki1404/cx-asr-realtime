FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

# ---- Proxy control (build-time) ----
ARG USE_PROXY=false
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

RUN if [ "$USE_PROXY" = "true" ]; then \
        echo "Enabling proxy for build stage"; \
        export http_proxy=${HTTP_PROXY}; \
        export https_proxy=${HTTPS_PROXY}; \
    else \
        echo "🌐 Proxy disabled for build stage"; \
    fi

# ---------------------------
# System dependencies
# ---------------------------
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

# ---------------------------
# Python tooling
# ---------------------------
RUN python3.10 -m pip install -U \
    pip \
    setuptools \
    wheel

# ---------------------------
# App dependencies FIRST
# ---------------------------
COPY requirements.txt /srv/requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r /srv/requirements.txt

# ============================
# NeMo (same behavior as working image)
# ============================
RUN python3 -m pip install --no-cache-dir \
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"

# ============================
# Torch LAST (lock CUDA ABI)
# ============================
RUN python3 -m pip install --no-cache-dir --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 \
    torchaudio==2.5.1


# ===========================
# --- STAGE 2: RUNTIME ---
# ===========================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ---- Proxy control (runtime) ----
ARG USE_PROXY=false
ARG HTTP_PROXY
ARG HTTPS_PROXY

# Empty by default (K8s-safe)
ENV http_proxy=""
ENV https_proxy=""

RUN if [ "$USE_PROXY" = "true" ]; then \
        echo " Enabling proxy for runtime"; \
        export http_proxy=${HTTP_PROXY}; \
        export https_proxy=${HTTPS_PROXY}; \
    else \
        echo "🌐 Proxy disabled for runtime"; \
    fi

# ---------------------------
# Runtime environment
# ---------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/srv \
    HF_HOME=/srv/hf_cache \
    TRANSFORMERS_CACHE=/srv/hf_cache \
    TORCH_HOME=/srv/hf_cache \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /srv

# ---------------------------
# Runtime system deps
# ---------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Copy Python environment from builder
# ---------------------------
COPY --from=builder /usr/local /usr/local

# ---------------------------
# Application code
# ---------------------------
COPY app /srv/app
COPY scripts /srv/scripts

EXPOSE 8002

CMD ["python3", "scripts/run_server.py", "--host", "0.0.0.0", "--port", "8002"]
